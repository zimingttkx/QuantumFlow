"""模型注册表 — 支持内置 + 外部 JSON 配置 + 动态注册 + 量化感知"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

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
    supported_backends: list[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.AVAILABLE
    quantization: str | None = None
    model_family: str = ""  # qwen/llama/glm/deepseek/yi
    max_model_len: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """
    模型注册表

    - 内置常用大模型（Qwen / LLaMA / GLM / DeepSeek / Yi）
    - 支持从 JSON 文件加载外部模型
    - 支持运行时动态注册/注销
    - 估算显存时考虑量化与 TP 切分
    """

    # 内置模型列表
    BUILTIN_MODELS: dict[str, ModelInfo] = {
        "Qwen2.5-7B-Instruct": ModelInfo(
            name="Qwen2.5-7B-Instruct",
            path="Qwen/Qwen2.5-7B-Instruct",
            backend="vllm",
            parameter_count=7_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=14,
            max_memory_gb=18,
            supported_backends=["vllm", "tgi", "sglang"],
            model_family="qwen",
            max_model_len=32768,
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
            model_family="qwen",
            max_model_len=32768,
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
            model_family="qwen",
            max_model_len=32768,
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
            model_family="llama",
            max_model_len=8192,
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
            model_family="llama",
            max_model_len=8192,
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
            model_family="glm",
            max_model_len=8192,
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
            model_family="deepseek",
            max_model_len=32768,
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
            model_family="yi",
            max_model_len=4096,
        ),
    }

    def __init__(self, external_config_path: str | Path | None = None):
        self._models: dict[str, ModelInfo] = {}
        self._load_builtin_models()
        if external_config_path and Path(external_config_path).exists():
            self.load_from_file(external_config_path)

    def _load_builtin_models(self):
        """加载内置模型"""
        for name, info in self.BUILTIN_MODELS.items():
            self._models[name] = info

    def load_from_file(self, path: str | Path) -> int:
        """从 JSON 文件加载外部模型定义

        Args:
            path: JSON 路径，格式 {"models": [{...ModelInfo fields...}]}

        Returns:
            成功加载的模型数
        """
        path = Path(path)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning("model_config_file_not_found", path=str(path))
            return 0
        except json.JSONDecodeError as e:
            logger.error("model_config_file_invalid_json", path=str(path), error=str(e))
            return 0

        count = 0
        for item in data.get("models", []):
            try:
                info = ModelInfo(**item)
                self._models[info.name] = info
                count += 1
            except Exception as e:
                logger.warning(
                    "model_load_from_file_failed",
                    name=item.get("name"),
                    error=str(e),
                )
        logger.info("models_loaded_from_file", path=str(path), count=count)
        return count

    def register_model(self, model_info: ModelInfo, overwrite: bool = False) -> bool:
        """注册新模型（动态）

        Args:
            model_info: 模型信息
            overwrite: 如果已存在是否覆盖（默认 False，重名注册视为失败）

        Returns:
            是否注册成功（重名且未 overwrite 时返回 False）
        """
        if model_info.name in self._models and not overwrite:
            logger.warning("model_already_registered", model=model_info.name)
            return False

        overwritten = model_info.name in self._models
        self._models[model_info.name] = model_info
        logger.info(
            "model_registered",
            model=model_info.name,
            backend=model_info.backend,
            overwritten=overwritten,
        )
        return True

    def unregister_model(self, name: str) -> bool:
        """注销模型"""
        if name in self._models:
            del self._models[name]
            logger.info("model_unregistered", model=name)
            return True
        return False

    def get_model(self, name: str) -> ModelInfo | None:
        """获取模型信息"""
        return self._models.get(name)

    def list_models(
        self,
        status: ModelStatus | None = None,
        backend: str | None = None,
    ) -> list[ModelInfo]:
        """列出模型"""
        models = list(self._models.values())

        if status:
            models = [m for m in models if m.status == status]

        if backend:
            models = [m for m in models if backend in m.supported_backends]

        return models

    def update_model_status(self, name: str, status: ModelStatus) -> bool:
        """更新模型状态"""
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

    def get_models_by_backend(self, backend: str) -> list[ModelInfo]:
        """获取支持特定后端的所有模型"""
        return [m for m in self._models.values() if backend in m.supported_backends]

    def get_supported_backends(self, name: str) -> list[str]:
        """获取模型支持的所有后端"""
        m = self.get_model(name)
        return list(m.supported_backends) if m else []

    def suggest_tensor_parallel(self, name: str) -> int:
        """获取推荐的 tensor parallel 大小"""
        model = self.get_model(name)
        if model:
            return model.recommended_tensor_parallel
        return 1

    def estimate_memory(self, name: str, tensor_parallel: int = 1) -> int:
        """估算每张 GPU 所需显存（GB），向后兼容方法

        新代码请使用 :meth:`estimate_memory_per_gpu`
        """
        m = self.get_model(name)
        if not m:
            return 16
        # 老逻辑：min_memory_gb 是总需求除以 TP
        per_gpu = m.min_memory_gb // max(tensor_parallel, 1)
        return max(1, per_gpu)

    def estimate_memory_per_gpu(
        self,
        name: str,
        tensor_parallel: int = 1,
        quantization: str | None = None,
    ) -> int:
        """估算每张 GPU 所需显存（GB），考虑量化

        Args:
            name: 模型名
            tensor_parallel: 张量并行度
            quantization: 量化方法（None / fp16 / int8 / awq / gptq / gguf / bnb）

        Returns:
            估算的 per-GPU 显存（GB）
        """
        m = self.get_model(name)
        if not m:
            return 16

        # 优先用调用方传入的 quantization
        quant = quantization or m.quantization
        bytes_per_param = {
            None: 2.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "fp32": 4.0,
            "int8": 1.0,
            "awq": 0.55,
            "gptq": 0.55,
            "gguf": 0.6,
            "bnb": 0.6,
        }.get(quant, 2.0)
        # 权重 + 20% 框架开销
        raw_gb = (m.parameter_count * bytes_per_param * 1.2) / 1024**3
        # KV cache（按 max_model_len 估算）：每 1K context 约 0.1GB per GPU
        kv_gb = (m.max_model_len / 1024) * 0.1
        per_gpu = (raw_gb / max(tensor_parallel, 1)) + kv_gb
        return max(1, int(per_gpu))


# 全局注册表实例
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """获取全局注册表"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry() -> None:
    """重置全局注册表（仅测试使用）"""
    global _registry
    _registry = None
