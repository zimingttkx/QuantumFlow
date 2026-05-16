"""模型注册表"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

import structlog

logger = structlog.get_logger().bind(component="model_registry")


class ModelStatus(Enum):
    """模型状态"""

    AVAILABLE = "available"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelInfo:
    """模型信息"""

    name: str
    path: str
    backend: str
    parameter_count: int
    recommended_tensor_parallel: int
    min_memory_gb: int
    max_memory_gb: int
    supported_backends: List[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.AVAILABLE
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """
    模型注册表

    管理所有可用模型的元信息，
    包括模型路径、后端支持、显存需求等。
    """

    # 内置模型列表
    BUILTIN_MODELS: Dict[str, ModelInfo] = {
        "Qwen2.5-7B-Instruct": ModelInfo(
            name="Qwen2.5-7B-Instruct",
            path="Qwen/Qwen2.5-7B-Instruct",
            backend="vllm",
            parameter_count=7_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=14,
            max_memory_gb=18,
            supported_backends=["vllm", "tgi", "sglang"],
        ),
        "Qwen2.5-14B-Instruct": ModelInfo(
            name="Qwen2.5-14B-Instruct",
            path="Qwen/Qwen2.5-14B-Instruct",
            backend="vllm",
            parameter_count=14_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=28,
            max_memory_gb=32,
            supported_backends=["vllm", "tgi", "sglang"],
        ),
        "Qwen2.5-72B-Instruct": ModelInfo(
            name="Qwen2.5-72B-Instruct",
            path="Qwen/Qwen2.5-72B-Instruct",
            backend="vllm",
            parameter_count=72_000_000_000,
            recommended_tensor_parallel=4,
            min_memory_gb=140,
            max_memory_gb=160,
            supported_backends=["vllm", "tgi", "sglang"],
        ),
        "LLaMA-3-8B-Instruct": ModelInfo(
            name="LLaMA-3-8B-Instruct",
            path="meta-llama/Meta-Llama-3-8B-Instruct",
            backend="vllm",
            parameter_count=8_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=16,
            max_memory_gb=20,
            supported_backends=["vllm", "tgi", "sglang"],
        ),
        "LLaMA-3-70B-Instruct": ModelInfo(
            name="LLaMA-3-70B-Instruct",
            path="meta-llama/Meta-Llama-3-70B-Instruct",
            backend="vllm",
            parameter_count=70_000_000_000,
            recommended_tensor_parallel=4,
            min_memory_gb=135,
            max_memory_gb=155,
            supported_backends=["vllm", "tgi", "sglang"],
        ),
        "GLM-4-9B": ModelInfo(
            name="GLM-4-9B",
            path="THUDM/GLM-4-9B",
            backend="vllm",
            parameter_count=9_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=18,
            max_memory_gb=22,
            supported_backends=["vllm", "tgi", "sglang"],
        ),
        "DeepSeek-V2": ModelInfo(
            name="DeepSeek-V2",
            path="deepseek-ai/DeepSeek-V2",
            backend="vllm",
            parameter_count=236_000_000_000,
            recommended_tensor_parallel=8,
            min_memory_gb=450,
            max_memory_gb=520,
            supported_backends=["vllm"],
        ),
        "Yi-1.5-34B": ModelInfo(
            name="Yi-1.5-34B",
            path="01-ai/Yi-1.5-34B",
            backend="vllm",
            parameter_count=34_000_000_000,
            recommended_tensor_parallel=2,
            min_memory_gb=65,
            max_memory_gb=75,
            supported_backends=["vllm", "tgi", "sglang"],
        ),
    }

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._load_builtin_models()

    def _load_builtin_models(self):
        """加载内置模型"""
        for name, info in self.BUILTIN_MODELS.items():
            self._models[name] = info

    def register_model(self, model_info: ModelInfo) -> bool:
        """
        注册新模型

        Args:
            model_info: 模型信息

        Returns:
            是否成功
        """
        if model_info.name in self._models:
            logger.warning(
                "model_already_registered",
                model=model_info.name,
            )
            return False

        self._models[model_info.name] = model_info
        logger.info(
            "model_registered",
            model=model_info.name,
            backend=model_info.backend,
        )
        return True

    def unregister_model(self, name: str) -> bool:
        """
        注销模型

        Args:
            name: 模型名称

        Returns:
            是否成功
        """
        if name in self._models:
            del self._models[name]
            logger.info("model_unregistered", model=name)
            return True
        return False

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """
        获取模型信息

        Args:
            name: 模型名称

        Returns:
            模型信息或None
        """
        return self._models.get(name)

    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        backend: Optional[str] = None,
    ) -> List[ModelInfo]:
        """
        列出模型

        Args:
            status: 按状态过滤
            backend: 按后端过滤

        Returns:
            模型列表
        """
        models = list(self._models.values())

        if status:
            models = [m for m in models if m.status == status]

        if backend:
            models = [m for m in models if backend in m.supported_backends]

        return models

    def update_model_status(self, name: str, status: ModelStatus) -> bool:
        """
        更新模型状态

        Args:
            name: 模型名称
            status: 新状态

        Returns:
            是否成功
        """
        if name not in self._models:
            return False

        old_status = self._models[name].status
        self._models[name].status = status

        logger.info(
            "model_status_updated",
            model=name,
            old_status=old_status.value,
            new_status=status.value,
        )
        return True

    def get_models_by_backend(self, backend: str) -> List[ModelInfo]:
        """获取支持特定后端的所有模型"""
        return [
            m for m in self._models.values()
            if backend in m.supported_backends
        ]

    def suggest_tensor_parallel(self, name: str) -> int:
        """获取推荐的tensor parallel大小"""
        model = self.get_model(name)
        if model:
            return model.recommended_tensor_parallel
        return 1

    def estimate_memory(self, name: str, tensor_parallel: int = 1) -> int:
        """
        估算所需显存

        Args:
            name: 模型名称
            tensor_parallel: tensor parallel大小

        Returns:
            估算显存（GB）
        """
        model = self.get_model(name)
        if model:
            # 考虑tensor parallel的显存需求
            return model.min_memory_gb // tensor_parallel
        return 16  # 默认值


# 全局注册表实例
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """获取全局注册表"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
