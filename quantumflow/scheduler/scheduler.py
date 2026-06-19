"""调度器核心"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from quantumflow.scheduler.strategy.adaptive import AdaptiveSchedulingStrategy
from quantumflow.scheduler.strategy.base import (
    NodeResource,
    SchedulingRequest,
    SchedulingResult,
    SchedulingStrategy,
)
from quantumflow.scheduler.strategy.gang import GangSchedulingStrategy
from quantumflow.scheduler.strategy.pack import PackSchedulingStrategy
from quantumflow.scheduler.worker_client import WorkerClient, WorkerEndpoint, get_worker_registry
from quantumflow.utils.config import get_config

logger = structlog.get_logger().bind(component="scheduler")


@dataclass
class QueueItem:
    """队列项"""

    request: SchedulingRequest
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: datetime | None = None
    result: SchedulingResult | None = None
    retry_count: int = 0


class Scheduler:
    """
    核心调度器

    负责：
    - 请求队列管理（带背压）
    - 节点资源管理 + GPU 预留/释放
    - 调度策略选择
    - 请求分发到 Worker
    - 失败节点隔离（连续失败 ≥ 3 隔离 N 秒）
    """

    def __init__(
        self,
        default_strategy: str = "adaptive",
        loop_interval_ms: int = 100,
        max_retries: int = 3,
        failure_quarantine_seconds: float = 60.0,
    ):
        self.config = get_config()

        # 调度配置
        self.default_strategy = default_strategy
        self.loop_interval_ms = loop_interval_ms
        self.max_retries = max_retries
        self.failure_quarantine_seconds = failure_quarantine_seconds

        # 策略注册
        self.strategies: dict[str, SchedulingStrategy] = {}
        self._init_strategies()

        # 自适应策略
        self.adaptive_strategy = AdaptiveSchedulingStrategy(strategies=self.strategies)

        # 请求队列
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._request_counter: int = 0
        self.running_requests: dict[str, QueueItem] = {}

        # 节点状态
        self.available_nodes: dict[str, NodeResource] = {}
        self.node_update_callbacks: list[Callable] = []

        # 节点失败记录（用于隔离）
        # node_id -> list[float]  (最近 N 次失败的时间戳,滑动窗口)
        self._node_failure_timestamps: dict[str, list[float]] = {}
        # 已隔离的节点：node_id -> quarantine_until_ts
        self._quarantined_nodes: dict[str, float] = {}

        # 调度循环
        self._scheduling_task: asyncio.Task | None = None
        self._running = False

        # 背压控制
        self._consecutive_empty_loops = 0

        # 请求完成回调
        self.request_complete_callbacks: list[Callable] = []

        # Bug fix (H-R1): 持有 background tasks 引用,避免 GC 清理导致 task 取消
        # 也保证异常能被记录(通过 add_done_callback)
        self._background_tasks: set[asyncio.Task] = set()

        # 统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "pending_requests": 0,
            "quarantined_nodes": 0,
        }

    # ------------------------------------------------------------------ 初始化

    def _init_strategies(self):
        """初始化调度策略"""
        self.strategies["gang"] = GangSchedulingStrategy(config=self.config.scheduler.model_dump())
        self.strategies["pack"] = PackSchedulingStrategy(config=self.config.scheduler.model_dump())

    # ------------------------------------------------------------------ 生命周期

    async def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("scheduler_already_running")
            return

        self._running = True
        self._scheduling_task = asyncio.create_task(self._scheduling_loop())

        logger.info("scheduler_started", strategy=self.default_strategy)

    async def stop(self):
        """停止调度器"""
        self._running = False

        if self._scheduling_task:
            self._scheduling_task.cancel()
            try:
                await self._scheduling_task
            except asyncio.CancelledError:
                pass

        # 释放所有预留
        for node in self.available_nodes.values():
            node.reserved_gpus.clear()
            node.reserved_memory_by_request.clear()
            node.reserved_memory_bytes = 0

        logger.info("scheduler_stopped")

    # ------------------------------------------------------------------ 队列

    async def submit(self, request: SchedulingRequest) -> str:
        """提交推理请求"""
        self.stats["total_requests"] += 1
        self.stats["pending_requests"] += 1

        logger.info(
            "request_submitted",
            request_id=request.request_id,
            model=request.model,
            priority=request.priority,
        )

        priority = request.priority
        self._request_counter += 1
        await self.pending_queue.put(
            (priority, request.created_at, self._request_counter, request)
        )
        return request.request_id

    # ------------------------------------------------------------------ 主循环

    async def _scheduling_loop(self):
        """调度循环（带背压）"""
        logger.info("scheduling_loop_started")

        while self._running:
            try:
                # 背压：队列空时拉长间隔
                interval = self.loop_interval_ms / 1000.0
                if self.pending_queue.empty():
                    self._consecutive_empty_loops += 1
                    interval = min(interval * (1 + self._consecutive_empty_loops * 0.1), 5.0)
                else:
                    self._consecutive_empty_loops = 0

                await asyncio.sleep(interval)

                # 释放过期隔离
                self._release_expired_quarantines()

                if not self.available_nodes:
                    continue

                await self._process_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("scheduling_loop_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("scheduling_loop_stopped")

    async def _process_batch(self):
        """批量处理请求"""
        batch_size = min(self.pending_queue.qsize(), 10)
        requests_batch: list[SchedulingRequest] = []

        for _ in range(batch_size):
            try:
                _, _, _, request = self.pending_queue.get_nowait()
                requests_batch.append(request)
            except asyncio.QueueEmpty:
                break

        if not requests_batch:
            return

        for request in requests_batch:
            try:
                result = await self._schedule_request(request)

                if result.success:
                    await self._dispatch(request, result)
                else:
                    await self._handle_scheduling_failure(request, result)
            except Exception as e:
                # Bug fix (C-R3): 防止请求在 get_nowait 和 _schedule_request 之间
                # 因异常丢失。重新入队并记录错误。
                # Bug fix (C-R3 follow-up): 递增 retry_count 并检查 max_retries，
                # 防止持久性错误导致无限重入队。
                request.retry_count += 1
                logger.error(
                    "request_lost_on_exception",
                    request_id=request.request_id,
                    error=str(e),
                    exc_type=type(e).__name__,
                    retry_count=request.retry_count,
                )
                if request.retry_count < self.max_retries:
                    self._request_counter += 1
                    await self.pending_queue.put(
                        (request.priority, request.created_at, self._request_counter, request)
                    )
                else:
                    self.stats["failed_requests"] += 1
                    self.stats["pending_requests"] -= 1

    # ------------------------------------------------------------------ 调度核心

    async def _schedule_request(self, request: SchedulingRequest) -> SchedulingResult:
        """调度单个请求"""
        # 过滤掉已隔离的节点
        nodes = self._filter_quarantined_nodes(list(self.available_nodes.values()))

        if not nodes:
            return SchedulingResult(success=False, reason="No available nodes")

        strategy = self._get_strategy(request, nodes)
        if not strategy:
            return SchedulingResult(
                success=False,
                reason=f"Strategy not found: {self.default_strategy}",
            )

        return strategy.select_nodes(request, nodes)

    def _get_strategy(
        self, request: SchedulingRequest, nodes: list[NodeResource]
    ) -> SchedulingStrategy | None:
        if self.default_strategy == "adaptive":
            return self.adaptive_strategy
        return self.strategies.get(self.default_strategy)

    # ------------------------------------------------------------------ 预留

    def _reserve_for_request(
        self, request: SchedulingRequest, result: SchedulingResult
    ) -> bool:
        """在所有分配的节点上预留 GPU 显存

        成功：返回 True
        失败：返回 False（不修改任何预留）

        Bug fix (M-R3): 原本跨节点非原子 — 节点 A 校验通过并 reserve, 节点 B
        校验失败 → return False 但 A 的 reservation 泄漏。修复为两阶段:
        阶段1: 全部节点只 check 不 reserve
        阶段2: 全部 check 通过后才统一 reserve
        """
        per_gpu_mem = (
            request.estimated_memory_per_gpu_bytes
            if request.parameter_count > 0
            else int(request.model_config.get("estimated_memory", 0))
        )
        if per_gpu_mem == 0:
            # Bug fix (H-R5): 优先使用 SchedulingRequest.estimated_memory_per_gpu_bytes
            # 作为更好的估算（考虑量化、TP、KV cache），再回退到 16GB 兜底。
            per_gpu_mem = request.estimated_memory_per_gpu_bytes
        if per_gpu_mem == 0:
            # 兜底：每张卡按 16GB 预留（避免 0 字节预留导致重复调度）
            per_gpu_mem = 16 * 1024**3
            logger.warning(
                "per_gpu_memory_fallback_16gb",
                request_id=request.request_id,
                model=request.model,
                parameter_count=request.parameter_count,
                reason="No memory estimate available, using 16GB fallback",
            )

        # 阶段1: 全部节点只 check,不 reserve
        # 记录待 reserve 的 (node, gpu_id) 列表,阶段2 统一处理
        pending_reservations: list[tuple[NodeResource, int]] = []
        for node_id, gpu_ids in result.assigned_gpus.items():
            node = self.available_nodes.get(node_id)
            if node is None:
                return False
            for gpu_id in gpu_ids:
                if node.effective_available_memory_per_gpu(gpu_id) < per_gpu_mem:
                    return False
                pending_reservations.append((node, gpu_id))

        # 阶段2: 全部 check 通过,统一 reserve
        for node, gpu_id in pending_reservations:
            node.reserve_gpu(request.request_id, gpu_id, per_gpu_mem)
        return True

    def _release_for_request(self, request_id: str) -> None:
        """释放某请求的所有预留"""
        for node in self.available_nodes.values():
            node.release_reservation(request_id)

    # ------------------------------------------------------------------ 隔离

    def _release_expired_quarantines(self) -> None:
        """释放过期的节点隔离"""
        now = asyncio.get_event_loop().time()
        expired = [
            nid
            for nid, until in self._quarantined_nodes.items()
            if now >= until
        ]
        for nid in expired:
            del self._quarantined_nodes[nid]
            self._node_failure_timestamps.pop(nid, None)
            logger.info("node_released_from_quarantine", node_id=nid)

    def _filter_quarantined_nodes(
        self, nodes: list[NodeResource]
    ) -> list[NodeResource]:
        return [n for n in nodes if n.node_id not in self._quarantined_nodes]

    def _record_node_failure(self, node_id: str, reason: str) -> bool:
        """记录节点失败。返回 True 表示需要隔离。

        Bug fix (M-R10): 使用滑动窗口替代硬窗口重置。
        保留最近 N 次失败时间戳，统计 60s 窗口内的失败次数。
        避免"每 61s 失败一次永不隔离"的问题。
        """
        now = asyncio.get_event_loop().time()
        timestamps = self._node_failure_timestamps.get(node_id, [])
        # 追加当前失败时间戳
        timestamps.append(now)
        # 只保留最近 60s 窗口内的失败记录
        window_start = now - 60.0
        timestamps = [t for t in timestamps if t >= window_start]
        self._node_failure_timestamps[node_id] = timestamps
        count = len(timestamps)
        if count >= 3:
            self._quarantined_nodes[node_id] = now + self.failure_quarantine_seconds
            self.stats["quarantined_nodes"] = len(self._quarantined_nodes)
            logger.warning(
                "node_quarantined",
                node_id=node_id,
                quarantine_seconds=self.failure_quarantine_seconds,
                reason=reason,
                failure_count=count,
            )
            return True
        return False

    # ------------------------------------------------------------------ 分发

    async def _dispatch(self, request: SchedulingRequest, result: SchedulingResult):
        """分发请求到 Worker

        1. 预留 GPU 显存
        2. 推送到 running_requests
        3. 异步发送给 Worker
        """
        request_id = request.request_id

        # 1. 预留（原子）
        reserved = self._reserve_for_request(request, result)
        if not reserved:
            # 显存被并发抢占，调度失败回退
            logger.warning(
                "reservation_failed_after_scheduling",
                request_id=request_id,
            )
            await self._handle_scheduling_failure(
                request,
                SchedulingResult(success=False, reason="GPU overcommitted"),
            )
            return

        # 2. 跟踪
        self.running_requests[request_id] = QueueItem(
            request=request,
            scheduled_at=datetime.now(),
            result=result,
        )
        self.stats["pending_requests"] -= 1

        logger.info(
            "request_dispatched",
            request_id=request_id,
            nodes=result.assigned_nodes,
            strategy=result.strategy_used,
        )

        # 3. 异步发送给所有分配的 worker（多 worker 协同推理时不会漏发）
        # Bug fix (H-R1): 保存 task 引用,避免 GC 取消;加 done_callback 记录异常
        task = asyncio.create_task(self._send_to_workers(request, result))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        def _log_task_exception(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "background_dispatch_task_failed",
                    request_id=request_id,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
        task.add_done_callback(_log_task_exception)

    async def _handle_scheduling_failure(
        self, request: SchedulingRequest, result: SchedulingResult
    ):
        """处理调度失败（含重试）"""
        request_id = request.request_id
        request.retry_count += 1

        if request.retry_count < self.max_retries:
            logger.warning(
                "scheduling_retry",
                request_id=request_id,
                reason=result.reason,
                retry_count=request.retry_count,
            )
            self._request_counter += 1
            await self.pending_queue.put(
                (request.priority, request.created_at, self._request_counter, request)
            )
            # pending_requests 计数没变（之前没增过）
        else:
            logger.error(
                "scheduling_failed",
                request_id=request_id,
                reason=result.reason,
                max_retries=self.max_retries,
            )
            self.stats["failed_requests"] += 1
            self.stats["pending_requests"] -= 1
            await self._notify_completion(request, success=False, error=result.reason)

    async def _send_to_workers(
        self, request: SchedulingRequest, result: SchedulingResult
    ):
        """发送请求到所有分配的 Worker"""
        request_id = request.request_id

        try:
            sampling_params = request.model_config.get(
                "sampling_params",
                {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": request.max_tokens,
                },
            )

            registry = get_worker_registry()

            # 解析每个节点的 worker endpoint
            endpoints: list[WorkerEndpoint] = []
            for node_id in result.assigned_nodes:
                endpoint = await registry.get_worker(node_id)
                if endpoint is None:
                    # 节点未注册，尝试从 available_nodes 构建
                    node = self.available_nodes.get(node_id)
                    if node:
                        endpoint = WorkerEndpoint(
                            node_id=node_id,
                            host=node.ip,
                            port=getattr(node, "worker_port", 8000),
                        )
                if endpoint is None:
                    logger.error(
                        "worker_endpoint_not_found",
                        node_id=node_id,
                        request_id=request_id,
                    )
                    continue
                endpoints.append((node_id, endpoint))

            if not endpoints:
                await self._mark_failure(request, "no_worker_endpoints")
                return

            # 并发发送给所有 endpoint
            client = WorkerClient(timeout=30.0)
            try:
                tasks = [
                    client.inference(
                        endpoint=ep,
                        request_id=request_id,
                        model_name=request.model,
                        prompt=request.prompt,
                        sampling_params=sampling_params,
                    )
                    for _node_id, ep in endpoints
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                await self._handle_worker_results(request, endpoints, results)
            finally:
                await client.close()

        except Exception as e:
            logger.error(
                "worker_dispatch_error",
                request_id=request_id,
                error=str(e),
            )
            await self._mark_failure(request, str(e))

    async def _handle_worker_results(
        self,
        request: SchedulingRequest,
        endpoints: list[tuple[str, WorkerEndpoint]],
        results: list[Any],
    ):
        """处理 worker 返回的结果"""
        request_id = request.request_id
        any_success = False
        first_error: str | None = None

        for (node_id, endpoint), result in zip(endpoints, results):
            if isinstance(result, Exception):
                self._record_node_failure(node_id, str(result))
                first_error = first_error or str(result)
                continue
            if isinstance(result, dict) and result.get("status") == "success":
                any_success = True
            else:
                err = result.get("error", "unknown") if isinstance(result, dict) else "unknown"
                self._record_node_failure(node_id, err)
                first_error = first_error or err

        if any_success:
            self.stats["successful_requests"] += 1
            await self._notify_completion(request, success=True, result={"request_id": request_id})
        else:
            await self._mark_failure(request, first_error or "all_workers_failed")

    async def _mark_failure(self, request: SchedulingRequest, error: str) -> None:
        self.stats["failed_requests"] += 1
        self._release_for_request(request.request_id)
        if request.request_id in self.running_requests:
            del self.running_requests[request.request_id]
        await self._notify_completion(request, success=False, error=error)

    async def _notify_completion(
        self,
        request: SchedulingRequest,
        success: bool,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """通知请求完成，释放预留"""
        self._release_for_request(request.request_id)
        if request.request_id in self.running_requests:
            del self.running_requests[request.request_id]
        for cb in self.request_complete_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(request, success, result, error)
                else:
                    cb(request, success, result, error)
            except Exception as e:
                logger.error("completion_callback_error", error=str(e))

    def on_request_complete(self, callback: Callable):
        """注册请求完成回调"""
        self.request_complete_callbacks.append(callback)

    # ------------------------------------------------------------------ 节点管理

    async def register_node(self, node: NodeResource):
        """注册节点"""
        self.available_nodes[node.node_id] = node

        logger.info(
            "node_registered",
            node_id=node.node_id,
            gpu_count=node.gpu_count,
            status=node.status,
        )

        for callback in self.node_update_callbacks:
            try:
                await callback("register", node)
            except Exception as e:
                logger.error("node_callback_error", error=str(e))

    async def unregister_node(self, node_id: str):
        """注销节点"""
        if node_id in self.available_nodes:
            node = self.available_nodes.pop(node_id)
            self._quarantined_nodes.pop(node_id, None)
            self._node_failure_timestamps.pop(node_id, None)

            logger.info("node_unregistered", node_id=node_id)

            for callback in self.node_update_callbacks:
                try:
                    await callback("unregister", node)
                except Exception as e:
                    logger.error("node_callback_error", error=str(e))

    async def update_node(self, node: NodeResource):
        """更新节点状态"""
        self.available_nodes[node.node_id] = node

        logger.debug(
            "node_updated",
            node_id=node.node_id,
            gpu_count=node.gpu_count,
            load=node.load,
        )

    def on_node_update(self, callback: Callable):
        """注册节点更新回调"""
        self.node_update_callbacks.append(callback)

    # ------------------------------------------------------------------ 查询

    def get_pending_requests(self) -> list[SchedulingRequest]:
        """获取待调度请求"""
        return [
            req
            for _, _, _, req in self.pending_queue._queue  # type: ignore[attr-defined]
            if isinstance(req, SchedulingRequest)
        ]

    def get_running_requests(self) -> dict[str, QueueItem]:
        """获取运行中的请求"""
        return self.running_requests.copy()

    def get_stats(self) -> dict[str, Any]:
        """获取调度统计"""
        return {
            **self.stats,
            "queue_size": self.pending_queue.qsize(),
            "running_size": len(self.running_requests),
            "available_nodes": len(self.available_nodes),
        }
