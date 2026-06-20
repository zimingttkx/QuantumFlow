"""SharedBatchCoordinator — 多模型共享批处理协调器

允许多个不同模型的请求共享 GPU 批处理资源，提高整体 GPU 利用率。

架构：
┌─────────────────────────────────────────────────────────────────┐
│                    SharedBatchCoordinator                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Global Priority Queue                        │   │
│  │  (所有模型的请求按优先级排序)                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │              BatchScheduler (per-GPU)                     │   │
│  │  GPU 0 ──▶ Scheduler ──▶ Batch                         │   │
│  │  GPU 1 ──▶ Scheduler ──▶ Batch                         │   │
│  │  ...                                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

特性：
- 全局优先级队列，所有模型共享
- 模型亲和性：相同模型优先路由到同一 GPU
- VRAM 感知：每个 GPU 独立调度
- 动态 batch size：根据 pending 请求数调整
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import structlog

from quantumflow.inference.batch_config import DynamicBatchConfig
from quantumflow.inference.batch_scheduler import BatchScheduler
from quantumflow.inference.engine import InferenceResult, QueuedRequest, SamplingParams
from quantumflow.inference.priority_queue import PriorityQueue

if TYPE_CHECKING:
    from quantumflow.inference.engine_manager import EngineManager
    from quantumflow.inference.vram_manager import VRAMManager

logger = structlog.get_logger().bind(component="batch_coordinator")


class SharedBatchCoordinator:
    """
    多模型共享批处理协调器

    负责：
    - 接收所有模型的推理请求
    - 按优先级排序后分发到各 GPU
    - 维护模型亲和性（相同模型优先同一 GPU）
    - 追踪统计信息
    """

    def __init__(
        self,
        vram_manager: "VRAMManager",
        engine_manager: "EngineManager",
        config: DynamicBatchConfig | None = None,
    ):
        """
        Args:
            vram_manager: VRAM 管理器
            engine_manager: 推理引擎管理器（用于实际调度推理）
            config: 动态批处理配置
        """
        self._vram_manager = vram_manager
        self._engine_manager = engine_manager
        self._config = config or DynamicBatchConfig()

        # 调度唤醒事件：有新请求到达时通知调度器
        self._wake_event = asyncio.Event()

        # 全局优先级队列
        self._global_queue: PriorityQueue[QueuedRequest] = PriorityQueue()

        # Per-GPU 调度器
        self._schedulers: dict[int, BatchScheduler] = {}

        # 模型亲和性缓存：model_name -> gpu_id
        # 相同模型的请求优先路由到同一 GPU
        self._model_affinity: dict[str, int] = {}

        # 统计
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_models": 0,
            "avg_batch_size": 0.0,
            "high_priority_processed": 0,
            "low_priority_starved": 0,
        }

        # 后台调度任务
        self._schedule_task: asyncio.Task | None = None
        self._shutting_down = False

    async def submit(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
        priority: int = 5,
        tenant_id: str = "default",
    ) -> InferenceResult:
        """
        提交推理请求

        Args:
            model_name: 模型名称
            prompt: 输入提示
            sampling_params: 采样参数
            priority: 优先级 (0-10, 0 最高)
            tenant_id: 租户 ID

        Returns:
            推理结果
        """
        # 创建请求
        request = QueuedRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            sampling_params=sampling_params,
            priority=priority,
            submit_time=time.time(),
            future=asyncio.get_running_loop().create_future(),
            tenant_id=tenant_id,
        )

        # 加入全局队列
        await self._global_queue.put(request, priority=priority)

        # 唤醒调度器
        self._wake_event.set()

        # 确保调度任务在运行
        self._ensure_schedule_task()

        # 更新亲和性
        self._update_model_affinity(model_name)

        # 更新统计
        self.stats["total_requests"] += 1
        if model_name not in self._model_affinity:
            self.stats["total_models"] += 1

        # 等待结果
        return await request.future

    def _ensure_schedule_task(self):
        """确保调度任务在运行"""
        if self._schedule_task is None or self._schedule_task.done():
            self._schedule_task = asyncio.create_task(self._schedule_loop())

    async def _schedule_loop(self):
        """调度循环（事件驱动，不再 busy-wait）"""
        while not self._shutting_down:
            try:
                await self._schedule()
                # 等待新请求到达事件，超时 1 秒防止永久阻塞
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                self._wake_event.clear()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("schedule_loop_error", error=str(e))

    async def _schedule(self):
        """执行调度逻辑"""
        # 获取可用的 GPU
        available_gpus = self._get_available_gpus()

        for gpu_id in available_gpus:
            scheduler = self._get_or_create_scheduler(gpu_id)

            # 计算该 GPU 的最优 batch size
            pending_count = await self._global_queue.size()
            batch_size = scheduler.compute_batch_size(pending_count)

            if batch_size <= 0 or await self._global_queue.is_empty():
                continue

            # 从队列取出请求
            wrapper: list[QueuedRequest] = []
            while len(wrapper) < batch_size and not await self._global_queue.is_empty():
                request = await self._global_queue.get()
                if request:
                    wrapper.append(request)

            if wrapper:
                await self._execute_batch(gpu_id, wrapper)

    async def _execute_batch(self, gpu_id: int, batch: list[QueuedRequest]):
        """
        在指定 GPU 上执行一批请求

        Args:
            gpu_id: GPU ID
            batch: 请求列表
        """
        if not batch:
            return

        # 按模型分组
        model_groups: dict[str, list[QueuedRequest]] = {}
        for request in batch:
            if request.model_name not in model_groups:
                model_groups[request.model_name] = []
            model_groups[request.model_name].append(request)

        # 更新统计
        self.stats["total_batches"] += 1
        for request in batch:
            if request.priority < 5:
                self.stats["high_priority_processed"] += 1
            elif request.priority >= 7:
                self.stats["low_priority_starved"] += 1

        # 按模型分组调用 engine_manager.generate() 实际执行推理
        for model_name, group_requests in model_groups.items():
            for request in group_requests:
                try:
                    results: list[InferenceResult] = await self._engine_manager.generate(
                        model_name=model_name,
                        prompts=[request.prompt],
                        sampling_params=request.sampling_params,
                    )
                    # 将结果分发回对应的 future
                    if request.future and not request.future.done():
                        request.future.set_result(results[0])
                except Exception as e:
                    logger.error(
                        "batch_generate_error",
                        model_name=model_name,
                        gpu_id=gpu_id,
                        error=str(e),
                    )
                    # 失败时设置异常到该请求的 future
                    if request.future and not request.future.done():
                        request.future.set_exception(e)

        logger.info(
            "batch_executed",
            gpu_id=gpu_id,
            batch_size=len(batch),
            models=list(model_groups.keys()),
        )

    def _select_gpu_for_model(self, model_name: str) -> int:
        """
        为模型选择 GPU

        策略：
        1. 优先使用模型亲和性（相同模型路由到相同 GPU）
        2. 否则选择负载最低的 GPU

        Args:
            model_name: 模型名称

        Returns:
            GPU ID
        """
        # 1. 检查模型亲和性
        if model_name in self._model_affinity:
            cached_gpu = self._model_affinity[model_name]
            if cached_gpu in self._schedulers:
                return cached_gpu

        # 2. 选择负载最低的 GPU
        if self._schedulers:
            return min(
                self._schedulers.keys(),
                key=lambda gpu_id: self._schedulers[gpu_id].get_current_batch_size(),
            )

        # 3. 默认返回 GPU 0
        return 0

    def _update_model_affinity(self, model_name: str):
        """更新模型亲和性"""
        gpu_id = self._select_gpu_for_model(model_name)
        self._model_affinity[model_name] = gpu_id

    def _get_available_gpus(self) -> list[int]:
        """获取可用 GPU 列表"""
        # 目前返回 [0]，后续从 GPU monitor 获取
        return [0]

    def _get_or_create_scheduler(self, gpu_id: int) -> BatchScheduler:
        """获取或创建 GPU 调度器"""
        if gpu_id not in self._schedulers:
            self._schedulers[gpu_id] = BatchScheduler(
                vram_manager=self._vram_manager,
                config=self._config,
            )
        return self._schedulers[gpu_id]

    async def shutdown(self):
        """关闭协调器"""
        self._shutting_down = True
        if self._schedule_task and not self._schedule_task.done():
            self._schedule_task.cancel()
            try:
                await self._schedule_task
            except asyncio.CancelledError:
                pass

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            "schedulers": {
                gpu_id: {
                    "current_batch_size": scheduler.get_current_batch_size(),
                    "last_vram_utilization": scheduler.get_last_vram_utilization(),
                }
                for gpu_id, scheduler in self._schedulers.items()
            },
        }

    def get_model_affinity(self) -> dict[str, int]:
        """获取模型亲和性映射"""
        return dict(self._model_affinity)
