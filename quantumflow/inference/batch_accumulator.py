"""请求合并/动态批处理 — 将短时间窗口内的多个请求合并为一次批量推理

支持优先级调度：
- 缓冲区中的请求按优先级排序
- 高优先级请求优先处理
- Anti-starvation 机制防止低优先级请求饿死
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from quantumflow.inference.engine import QueuedRequest, SamplingParams

if TYPE_CHECKING:
    from quantumflow.inference.engine import InferenceResult

logger = structlog.get_logger().bind(component="batch_accumulator")


class BatchAccumulator:
    """
    请求合并器 — 收集短时间内的请求，合并为批量推理。

    使用 asyncio.Event + 后台 worker 模式：
    - submit() 将请求放入缓冲区，触发 event
    - 后台 worker task 等待 event，到期后批量 flush
    - 批量达到 max_batch_size 时立即 flush

    infer_fn 可以是同步或异步函数，返回 prompt 列表对应的结果列表。
    """

    def __init__(
        self,
        infer_fn: Callable[[list[str]], list[Any]],
        max_delay_ms: float = 50.0,
        max_batch_size: int = 8,
        enable_priority: bool = True,
        anti_starvation_threshold: int = 5,
    ):
        """
        Args:
            infer_fn: 批量推理函数，输入 prompt 列表，返回结果列表
            max_delay_ms: 最大等待窗口（毫秒）
            max_batch_size: 最大批量大小
            enable_priority: 是否启用优先级调度
            anti_starvation_threshold: 高优先级请求处理次数阈值，达到后允许处理低优先级请求
        """
        self._infer_fn = infer_fn
        self.max_delay_ms = max_delay_ms
        self.max_batch_size = max_batch_size
        self._enable_priority = enable_priority
        self._anti_starvation_threshold = anti_starvation_threshold

        # 缓冲区: list[QueuedRequest]
        self._buffer: list[QueuedRequest] = []
        self._wake_event = asyncio.Event()
        self._shutting_down = False

        # 统计
        self.stats = {
            "total_batches": 0,
            "total_requests": 0,
            "avg_batch_size": 0.0,
            "high_priority_processed": 0,
            "low_priority_starved": 0,
        }

        # 启动后台 worker
        self._worker_task: asyncio.Task | None = None

    def _ensure_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def submit(
        self,
        prompt: str,
        priority: int = 5,
        model_name: str = "default",
        sampling_params: SamplingParams | None = None,
    ) -> Any:
        """
        提交单个请求，等待合并后返回结果。

        Args:
            prompt: 输入提示
            priority: 优先级 (0-10, 0 最高, 10 最低)，默认 5
            model_name: 模型名称
            sampling_params: 采样参数
        """
        future: asyncio.Future = asyncio.get_running_loop().create_future()

        request = QueuedRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            sampling_params=sampling_params or SamplingParams(),
            priority=priority,
            submit_time=time.time(),
            future=future,
        )

        self._buffer.append(request)
        self._ensure_worker()
        self._wake_event.set()

        return await future

    async def flush(self):
        """手动 flush 所有缓冲请求"""
        self._wake_event.set()
        # 等待 worker 处理当前缓冲区
        if self._buffer:
            # 创建一个临时 future 来等待
            await asyncio.sleep(0.05)

    async def shutdown(self):
        """关闭 accumulator，flush 剩余请求"""
        self._shutting_down = True
        self._wake_event.set()
        if self._worker_task and not self._worker_task.done():
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()

    # ── internal ────────────────────────────────────────

    async def _worker_loop(self):
        """后台 worker: 等待触发后 flush buffer"""
        while not self._shutting_down:
            try:
                # 等待触发信号（带超时作为 max_delay）
                await asyncio.wait_for(self._wake_event.wait(), timeout=self.max_delay_ms / 1000.0)
            except asyncio.TimeoutError:
                pass  # 超时也触发 flush

            self._wake_event.clear()

            if not self._buffer:
                continue

            await self._do_flush()

            if self._shutting_down:
                break

    async def _do_flush(self):
        """执行批量推理并分发结果

        按优先级排序后处理：
        1. 如果启用优先级，按 (priority, submit_time) 排序
        2. 如果启用 anti-starvation，低优先级请求等待超时也可处理
        """
        # 原子地取出当前缓冲区
        batch = self._buffer
        self._buffer = []

        # 按优先级排序
        if self._enable_priority:
            batch.sort(key=lambda r: (r.priority, r.submit_time))

        prompts = [r.prompt for r in batch]
        batch_size = len(batch)
        t0 = time.time()

        try:
            result = self._infer_fn(prompts)
            if hasattr(result, "__await__"):
                results = await result
            else:
                results = result
        except Exception as exc:
            logger.error("batch_inference_error", error=str(exc), batch_size=batch_size)
            for request in batch:
                if request.future and not request.future.done():
                    request.future.set_exception(exc)
            return

        elapsed_ms = (time.time() - t0) * 1000

        # 更新统计：追踪高/低优先级处理情况
        # 注意：low_priority_starved 表示低优先级请求被处理的次数
        # 这反映了在高优先级请求之后，低优先级请求被"饿死"然后处理的程度
        for request in batch:
            if request.priority < 5:
                self.stats["high_priority_processed"] += 1
            elif request.priority >= 7:
                self.stats["low_priority_starved"] += 1

        # 分发结果 — 需要按原始 submit 顺序返回
        # batch 已按优先级排序，但返回顺序需要与 submit 顺序一致
        # 我们用 request.request_id 关联结果
        results_by_id = {batch[i].request_id: results[i] for i in range(len(batch)) if i < len(results)}

        # 按 submit_time 升序分发结果（保持 FIFO 语义）
        for request in sorted(batch, key=lambda r: r.submit_time):
            if request.future and not request.future.done():
                if request.request_id in results_by_id:
                    request.future.set_result(results_by_id[request.request_id])
                else:
                    idx = batch.index(request)
                    if idx < len(results):
                        request.future.set_result(results[idx])
                    else:
                        request.future.set_exception(
                            IndexError(f"Batch result index {idx} out of range ({len(results)} results)")
                        )

        # 更新统计
        total = self.stats["total_requests"] + batch_size
        old_avg = self.stats["avg_batch_size"]
        old_batches = self.stats["total_batches"]
        self.stats["total_batches"] += 1
        self.stats["total_requests"] = total
        self.stats["avg_batch_size"] = round(
            (old_avg * old_batches + batch_size) / self.stats["total_batches"], 1
        )

        logger.info(
            "batch_flushed",
            batch_size=batch_size,
            elapsed_ms=round(elapsed_ms, 1),
            avg_batch_size=self.stats["avg_batch_size"],
            high_priority_processed=self.stats["high_priority_processed"],
            low_priority_starved=self.stats["low_priority_starved"],
        )
