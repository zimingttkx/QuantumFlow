"""优先级队列 — 支持优先级调度和 anti-starvation 机制"""

from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantumflow.inference.engine import QueuedRequest


@dataclass(order=True)
class PriorityQueueItem:
    """
    优先级队列中的元素

    使用 order=True 让 heapq 能正确比较
    注意：order=True 会按所有字段排序，我们主要用 priority 和 submit_time
    """
    priority: int
    submit_time: float
    request_id: str = ""
    model_name: str = ""
    prompt: str = ""
    tenant_id: str = "default"
    is_high_priority_processed: int = 0  # 用于 anti-starvation 计数


class PriorityQueue:
    """
    优先级队列，支持 anti-starvation 机制

    特性：
    1. 按优先级排序（0 最高，10 最低）
    2. 相同优先级按 FIFO 顺序处理
    3. Anti-starvation：每 N 个高优先级请求后，检查低优先级请求

    使用示例：
        queue = PriorityQueue()
        await queue.put(request, priority=5)
        request = await queue.get()
    """

    def __init__(
        self,
        anti_starvation_threshold: int = 5,
        low_priority_starvation_timeout_seconds: float = 30.0,
    ):
        """
        Args:
            anti_starvation_threshold: 高优先级请求处理多少次后允许低优先级请求插队
            low_priority_starvation_timeout_seconds: 低优先级请求最长等待时间
        """
        # Heap stores tuples of (priority, submit_time, request_id, queued_request)
        self._heap: list[tuple[int, float, str, "QueuedRequest"]] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

        # Anti-starvation 配置
        self._anti_starvation_threshold = anti_starvation_threshold
        self._low_priority_starvation_timeout = low_priority_starvation_timeout_seconds

        # 统计
        self._stats = {
            "total_put": 0,
            "total_get": 0,
            "high_priority_processed": 0,
            "low_priority_starved": 0,
        }

    async def put(self, request: "QueuedRequest", priority: int | None = None) -> None:
        """
        将请求放入队列

        Args:
            request: QueuedRequest 对象
            priority: 可选的优先级覆盖，如果不提供则使用 request.priority
        """
        if priority is None:
            priority = request.priority

        async with self._not_empty:
            heapq.heappush(
                self._heap,
                (priority, request.submit_time, request.request_id, request),
            )
            self._stats["total_put"] += 1
            self._not_empty.notify()

    async def get(self) -> "QueuedRequest | None":
        """
        从队列取出最高优先级的请求

        包含 anti-starvation 逻辑：
        - 如果只有低优先级请求但等待超时，也允许处理

        Returns:
            QueuedRequest 或 None（如果队列为空）
        """
        async with self._not_empty:
            while not self._heap:
                await self._not_empty.wait()

            # 检查是否需要 anti-starvation 处理
            selected_index = self._get_next_item_index_unlocked()
            if selected_index is None:
                return None

            # 从堆中移除选中的项
            if selected_index == 0:
                # 选中的是堆顶，直接 heappop
                _, _, _, request = heapq.heappop(self._heap)
            else:
                # 选中的不是堆顶，删除指定索引并重建堆
                _, _, _, request = self._heap.pop(selected_index)
                heapq.heapify(self._heap)

            self._stats["total_get"] += 1

            # 更新统计
            if request.priority < 5:  # 高优先级
                self._stats["high_priority_processed"] += 1
            elif request.priority >= 7:  # 低优先级
                # 重置 anti-starvation 计数器
                self._stats["high_priority_processed"] = 0
                self._stats["low_priority_starved"] += 1

            return request

    def _get_next_item_index_unlocked(self) -> int | None:
        """
        在持有锁的情况下获取下一个要处理的项的索引

        包含 anti-starvation 逻辑：
        1. 如果队首是高优先级，直接返回索引 0
        2. 如果队首是低优先级，检查是否超时或高优先级请求足够多
        3. 否则扫描队列找可处理的低优先级请求

        Returns:
            要处理的项的索引，或 None（如果堆为空）
        """
        if not self._heap:
            return None

        # 获取当前时间
        now = time.time()

        # 检查队首
        first_priority, _, _, first_request = self._heap[0]

        # 如果是高优先级 (priority < 5)，直接返回
        if first_priority < 5:
            return 0

        # 如果是低优先级 (priority >= 7)
        if first_priority >= 7:
            # 检查超时
            wait_time = now - first_request.submit_time
            if wait_time > self._low_priority_starvation_timeout:
                return 0

            # 检查高优先级请求是否处理了足够多
            if self._stats["high_priority_processed"] >= self._anti_starvation_threshold:
                return 0

        # 扫描队列找非高优先级的超时或达到阈值的项
        for i, (priority, _, _, request) in enumerate(self._heap):
            if priority < 5:
                continue  # 跳过高优先级

            wait_time = now - request.submit_time

            # 超时或高优先级处理足够多
            if wait_time > self._low_priority_starvation_timeout:
                return i

            if self._stats["high_priority_processed"] >= self._anti_starvation_threshold:
                return i

        # 没有特殊情况，返回队首
        return 0

    async def peek(self) -> "QueuedRequest | None":
        """查看但不移除最高优先级的项"""
        async with self._not_empty:
            if not self._heap:
                return None
            return self._heap[0][3]

    async def size(self) -> int:
        """返回队列大小"""
        async with self._not_empty:
            return len(self._heap)

    async def qsize(self) -> int:
        """返回队列大小（asyncio.Queue 兼容别名）"""
        return await self.size()

    async def is_empty(self) -> bool:
        """检查队列是否为空"""
        async with self._not_empty:
            return len(self._heap) == 0

    async def empty(self) -> bool:
        """检查队列是否为空（asyncio.Queue 兼容别名）"""
        return await self.is_empty()

    async def remove_by_request_id(self, request_id: str) -> bool:
        """
        根据 request_id 移除请求

        Returns:
            是否成功移除
        """
        async with self._not_empty:
            for i, (_, _, rid, _) in enumerate(self._heap):
                if rid == request_id:
                    del self._heap[i]
                    heapq.heapify(self._heap)
                    return True
            return False

    def get_stats(self) -> dict:
        """获取队列统计信息"""
        return {
            **self._stats,
            "queue_size": len(self._heap),
        }

    async def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "total_put": 0,
            "total_get": 0,
            "high_priority_processed": 0,
            "low_priority_starved": 0,
        }
