"""推理引擎管理器 - 单例模式管理所有推理后端"""

from typing import Dict, Optional, AsyncIterator
import structlog

from quantumflow.inference.engine import (
    InferenceEngine,
    ModelConfig,
    SamplingParams,
    InferenceResult,
)
from quantumflow.inference.backends.vllm import VLLMEngine
from quantumflow.core.constants import InferenceBackendType
from quantumflow.core.exceptions import InferenceError, ModelNotFoundError

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
        self._engines: Dict[InferenceBackendType, InferenceEngine] = {}
        self._default_engine: Optional[InferenceEngine] = None
        self._loaded_models: Dict[str, InferenceEngine] = {}  # model_name -> engine

    async def initialize(self, backend: InferenceBackendType = InferenceBackendType.VLLM) -> bool:
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
        backend: InferenceBackendType = InferenceBackendType.VLLM,
        **kwargs,
    ) -> bool:
        """
        加载模型到指定后端

        Args:
            model_name: 模型名称标识
            model_path: 模型路径或HuggingFace ID
            backend: 推理后端
            **kwargs: 其他模型配置参数

        Returns:
            是否加载成功
        """
        # 获取或创建引擎
        engine = self._engines.get(backend)
        if not engine:
            success = await self.initialize(backend)
            if not success:
                raise InferenceError(f"Failed to initialize {backend.value} engine")
            engine = self._engines[backend]

        # 创建模型配置
        config = ModelConfig(
            model_name=model_name,
            model_path=model_path,
            tensor_parallel=kwargs.get("tensor_parallel", 1),
            pipeline_parallel=kwargs.get("pipeline_parallel", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            max_model_len=kwargs.get("max_model_len", 8192),
            dtype=kwargs.get("dtype", "auto"),
            quantization=kwargs.get("quantization"),
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )

        # 加载模型
        success = await engine.load_model(config)
        if success:
            self._loaded_models[model_name] = engine
            logger.info("model_loaded", model=model_name, backend=backend.value)

        return success

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name not in self._loaded_models:
            logger.warning("model_not_loaded", model=model_name)
            return False

        engine = self._loaded_models[model_name]
        success = await engine.unload_model(model_name)

        if success:
            del self._loaded_models[model_name]
            logger.info("model_unloaded", model=model_name)

        return success

    async def generate(
        self,
        model_name: str,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[InferenceResult]:
        """
        生成文本

        Args:
            model_name: 模型名称
            prompts: 提示词列表
            sampling_params: 采样参数

        Returns:
            推理结果列表
        """
        if model_name not in self._loaded_models:
            raise ModelNotFoundError(model_name)

        engine = self._loaded_models[model_name]
        results = await engine.generate(model_name, prompts, sampling_params)

        logger.info(
            "generate_completed",
            model=model_name,
            num_prompts=len(prompts),
            num_results=len(results),
        )

        return results

    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """
        流式生成文本

        Args:
            model_name: 模型名称
            prompt: 提示词
            sampling_params: 采样参数

        Yields:
            生成的文本片段
        """
        if model_name not in self._loaded_models:
            raise ModelNotFoundError(model_name)

        engine = self._loaded_models[model_name]

        async for text_chunk in engine.generate_stream(model_name, prompt, sampling_params):
            yield text_chunk

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


# 全局单例
_engine_manager: Optional[EngineManager] = None


def get_engine_manager() -> EngineManager:
    """获取引擎管理器单例"""
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = EngineManager()
    return _engine_manager
