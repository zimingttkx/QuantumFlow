"""推理引擎管理器 - 单例模式管理所有推理后端"""

import asyncio
from collections.abc import AsyncIterator
from typing import Optional

import structlog

from quantumflow.core.constants import InferenceBackendType
from quantumflow.core.exceptions import InferenceError, ModelNotFoundError
from quantumflow.inference.backends.huggingface import HuggingFaceEngine
from quantumflow.inference.backends.vllm import VLLMEngine
from quantumflow.inference.batch_accumulator import BatchAccumulator
from quantumflow.inference.engine import (
    InferenceEngine,
    InferenceResult,
    ModelConfig,
    SamplingParams,
)
from quantumflow.inference.gpu_monitor import GPUMonitor
from quantumflow.inference.vram_manager import VRAMManager

logger = structlog.get_logger().bind(component="engine_manager")


class EngineManager:
    """
    推理引擎管理器

    负责：
    - 管理多个推理引擎实例
    - 模型加载/卸载
    - 请求路由
    """

    _instance: Optional["EngineManager"] = None

    def __new__(cls) -> "EngineManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._engines: dict[InferenceBackendType, InferenceEngine] = {}
        self._default_engine: InferenceEngine | None = None
        self._loaded_models: dict[str, InferenceEngine] = {}  # model_name -> engine
        self._vram_manager = VRAMManager(
            safety_factor=0.7,
            idle_ttl_seconds=0.0,  # 默认禁用空闲卸载
        )
        self._batch_accumulators: dict[str, BatchAccumulator] = {}
        self._batch_sampling_params: dict[str, SamplingParams] = {}
        self._gpu_monitor = GPUMonitor(interval_seconds=5.0)
        self._eviction_task: asyncio.Task | None = None

    async def initialize(
        self, backend: InferenceBackendType = InferenceBackendType.HUGGINGFACE
    ) -> bool:
        """
        初始化指定后端的推理引擎

        Args:
            backend: 推理后端类型

        Returns:
            是否初始化成功
        """
        try:
            if backend == InferenceBackendType.VLLM:
                engine = VLLMEngine()
                success = await engine.initialize()
                if success:
                    self._engines[backend] = engine
                    self._default_engine = engine
                    logger.info("engine_manager_initialized", backend=backend.value)
                    return True
            elif backend == InferenceBackendType.HUGGINGFACE:
                engine = HuggingFaceEngine()
                success = await engine.initialize()
                if success:
                    self._engines[backend] = engine
                    self._default_engine = engine
                    logger.info("engine_manager_initialized", backend=backend.value)
                    return True
            else:
                logger.warning("unsupported_backend", backend=backend.value)
                return False

        except Exception as e:
            logger.error("engine_init_failed", backend=backend.value, error=str(e))

        return False

    async def load_model(
        self,
        model_name: str,
        model_path: str,
        backend: InferenceBackendType = InferenceBackendType.HUGGINGFACE,
        **kwargs,
    ) -> tuple[bool, str]:
        """
        加载模型到指定后端（VRAM感知）。

        Returns:
            (success, message) — message 描述加载结果/原因
        """
        max_model_len = kwargs.get("max_model_len", 2048)

        # 1. VRAM预估与检查
        required_vram = self._vram_manager.estimate_model_vram_gb(model_path, max_model_len)
        can_load, reason, to_evict = self._vram_manager.can_load(model_name, required_vram)

        if not can_load:
            logger.warning(
                "vram_reject", model=model_name, required_vram_gb=required_vram, reason=reason
            )
            return False, reason

        # 2. 如需淘汰，先淘汰LRU模型
        if to_evict:
            logger.info("vram_evicting", models=to_evict, reason=reason)
            for evict_name in to_evict:
                await self.unload_model(evict_name)

        # 3. 获取或创建引擎
        engine = self._engines.get(backend)
        if not engine:
            success = await self.initialize(backend)
            if not success:
                raise InferenceError(f"Failed to initialize {backend.value} engine")
            engine = self._engines[backend]

        # 4. 创建模型配置
        config = ModelConfig(
            model_name=model_name,
            model_path=model_path,
            tensor_parallel=kwargs.get("tensor_parallel", 1),
            pipeline_parallel=kwargs.get("pipeline_parallel", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.8),
            max_model_len=max_model_len,
            dtype=kwargs.get("dtype", "auto"),
            quantization=kwargs.get("quantization"),
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )

        # 5. 加载模型
        success = await engine.load_model(config)
        if success:
            self._loaded_models[model_name] = engine
            self._vram_manager.record_loaded(model_name, required_vram)
            # 记录实际显存占用（加载后立即读取）
            self._vram_manager.update_actual_vram(model_name)
            logger.info(
                "model_loaded",
                model=model_name,
                backend=backend.value,
                estimated_vram_gb=required_vram,
            )
            return True, reason

        return False, "引擎加载失败"

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name not in self._loaded_models:
            logger.warning("model_not_loaded", model=model_name)
            return False

        engine = self._loaded_models[model_name]
        success = await engine.unload_model(model_name)

        if success:
            del self._loaded_models[model_name]
            self._vram_manager.record_unloaded(model_name)
            logger.info("model_unloaded", model=model_name)

        return success

    async def generate(
        self,
        model_name: str,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[InferenceResult]:
        if model_name not in self._loaded_models:
            raise ModelNotFoundError(model_name)

        self._vram_manager.mark_in_use(model_name)

        # BlockPool: 分配 KV Cache blocks（基于预估 token 数）
        # prompt_tokens 估算 + max_tokens = 本次推理预估总 token 数
        import uuid

        request_id = str(uuid.uuid4())
        # 简单估算：每 token ≈ 4 字符
        est_prompt_tokens = sum(len(p) // 4 for p in prompts)
        est_total_tokens = est_prompt_tokens + sampling_params.max_tokens
        block_ids = self._vram_manager.allocate_blocks(model_name, est_total_tokens, request_id)

        try:
            engine = self._loaded_models[model_name]
            results = await engine.generate(model_name, prompts, sampling_params)

            # 更新实际显存
            self._vram_manager.update_actual_vram(model_name)

            logger.info(
                "generate_completed",
                model=model_name,
                num_prompts=len(prompts),
                num_results=len(results),
            )
            return results
        finally:
            self._vram_manager.mark_idle(model_name)
            # BlockPool: 释放 blocks
            if block_ids is not None:
                self._vram_manager.release_blocks(model_name, block_ids, request_id)

    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        if model_name not in self._loaded_models:
            raise ModelNotFoundError(model_name)

        self._vram_manager.mark_in_use(model_name)

        # BlockPool: 分配
        import uuid

        request_id = str(uuid.uuid4())
        est_tokens = len(prompt) // 4 + sampling_params.max_tokens
        block_ids = self._vram_manager.allocate_blocks(model_name, est_tokens, request_id)

        try:
            engine = self._loaded_models[model_name]
            async for text_chunk in engine.generate_stream(model_name, prompt, sampling_params):
                yield text_chunk
        finally:
            self._vram_manager.mark_idle(model_name)
            if block_ids is not None:
                self._vram_manager.release_blocks(model_name, block_ids, request_id)

    def get_loaded_models(self) -> list[str]:
        """获取已加载模型列表"""
        return list(self._loaded_models.keys())

    def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self._loaded_models

    def get_stats(self) -> dict:
        """获取引擎统计信息"""
        stats = {}
        for backend, engine in self._engines.items():
            stats[backend.value] = {
                "is_ready": engine.is_ready,
                "loaded_models": engine.loaded_model_names,
            }
        return stats

    def get_vram_status(self) -> dict:
        """获取VRAM状态"""
        return {
            "available_vram_gb": round(self._vram_manager.get_available_vram_gb(), 1),
            "safety_factor": self._vram_manager.safety_factor,
            "loaded_models": self._vram_manager.get_loaded_models(),
        }

    def get_batch_accumulator(
        self,
        model_name: str,
        sampling_params: SamplingParams,
        max_delay_ms: float = 50.0,
        max_batch_size: int = 8,
    ) -> BatchAccumulator:
        """
        获取或创建请求合并器（per-model + per-sampling-config）。

        合并器将短时间内的多个请求合并为一次批量推理调用，
        减少 GPU 空闲等待。

        Args:
            model_name: 模型名称
            sampling_params: 采样参数（相同参数的请求才能合并）
            max_delay_ms: 最大等待窗口
            max_batch_size: 最大批量大小
        """
        key = f"{model_name}_{sampling_params.temperature}_{sampling_params.max_tokens}"
        if key not in self._batch_accumulators:

            async def _batched_infer(prompts: list[str]):
                return await self.generate(model_name, prompts, sampling_params)

            self._batch_accumulators[key] = BatchAccumulator(
                infer_fn=_batched_infer,
                max_delay_ms=max_delay_ms,
                max_batch_size=max_batch_size,
            )
        return self._batch_accumulators[key]

    def get_batch_stats(self) -> dict:
        """获取批处理统计"""
        return {key: acc.stats for key, acc in self._batch_accumulators.items()}

    # ── 空闲淘汰 ───────────────────────────────────────

    def configure_idle_eviction(self, idle_ttl_seconds: float):
        """配置空闲超时淘汰（0=禁用）"""
        self._vram_manager.idle_ttl_seconds = idle_ttl_seconds
        if idle_ttl_seconds > 0:
            logger.info("idle_eviction_enabled", ttl_seconds=idle_ttl_seconds)
        else:
            logger.info("idle_eviction_disabled")

    async def start_idle_eviction_checker(self, check_interval_seconds: float = 30.0):
        """启动后台空闲淘汰检查"""
        if self._vram_manager.idle_ttl_seconds <= 0:
            logger.info("idle_eviction_skipped", reason="ttl_disabled")
            return

        logger.info("idle_eviction_checker_started", interval_seconds=check_interval_seconds)

        async def _loop():
            while True:
                await asyncio.sleep(check_interval_seconds)
                idle_models = self._vram_manager.get_idle_models_to_evict()
                for name in idle_models:
                    logger.info("idle_evicting", model=name)
                    await self.unload_model(name)

        self._eviction_task = asyncio.create_task(_loop())

    # ── GPU 监控 ──────────────────────────────────────────

    async def start_gpu_monitoring(self):
        """启动GPU后台监控"""
        await self._gpu_monitor.start()

    async def stop_gpu_monitoring(self):
        """停止GPU监控"""
        await self._gpu_monitor.stop()

    def get_gpu_status(self) -> list[dict]:
        """获取最新GPU状态快照"""
        return [s.to_dict() for s in self._gpu_monitor.latest]

    def get_gpu_snapshot(self) -> list[dict]:
        """按需采集一次GPU状态（不依赖后台监控）"""
        return [s.to_dict() for s in self._gpu_monitor.collect_snapshot()]


# 全局单例
_engine_manager: EngineManager | None = None


def get_engine_manager() -> EngineManager:
    """获取引擎管理器单例"""
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = EngineManager()
    return _engine_manager
