"""请求合并/动态批处理 — 将短时间窗口内的多个请求合并为一次批量推理"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import structlog

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
    ):
        self._infer_fn = infer_fn
        self.max_delay_ms = max_delay_ms
        self.max_batch_size = max_batch_size

        # 缓冲区: (prompt, future)
        self._buffer: list[tuple[str, asyncio.Future]] = []
        self._wake_event = asyncio.Event()
        self._shutting_down = False

        # 统计
        self.stats = {
            "total_batches": 0,
            "total_requests": 0,
            "avg_batch_size": 0.0,
        }

        # 启动后台 worker
        self._worker_task: asyncio.Task | None = None

    def _ensure_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def submit(self, prompt: str) -> Any:
        """
        提交单个请求，等待合并后返回结果。
        """
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._buffer.append((prompt, future))

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
        """执行批量推理并分发结果"""
        # 原子地取出当前缓冲区
        batch = self._buffer
        self._buffer = []

        prompts = [p for p, _ in batch]
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
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return

        elapsed_ms = (time.time() - t0) * 1000

        # 分发结果 — 保持顺序
        for i, (_, future) in enumerate(batch):
            if not future.done():
                if i < len(results):
                    future.set_result(results[i])
                else:
                    future.set_exception(
                        IndexError(f"Batch result index {i} out of range ({len(results)} results)")
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
        )
