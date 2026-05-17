"""推理引擎基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, AsyncIterator
from enum import Enum

from quantumflow.core.constants import InferenceBackendType


@dataclass
class ModelConfig:
    """模型配置"""

    model_name: str
    model_path: str
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    gpu_memory_utilization: float = 0.8  # 80% VRAM for vLLM (RTX 4080 12GB)
    max_model_len: int = 2048  # 传给vLLM的max_model_len，KV cache由gpu_memory_utilization控制
    dtype: str = "auto"  # float16, bfloat16, float32, auto
    quantization: Optional[str] = None  # awq, gptq, gguf
    trust_remote_code: bool = True

    # vLLM特有配置
    block_size: int = 16
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True

    # HuggingFace 特有配置
    prefill_chunk_size: int = 512  # 分块预填充的块大小（tokens），超过此长度自动分块
    torch_compile: bool = True     # 是否启用 torch.compile 加速

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel,
            "pipeline_parallel_size": self.pipeline_parallel,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass
class SamplingParams:
    """采样参数"""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "stop": self.stop,
        }


@dataclass
class InferenceResult:
    """推理结果"""

    request_id: str
    outputs: List[str]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


class InferenceEngine(ABC):
    """推理引擎抽象基类"""

    def __init__(self, backend_type: InferenceBackendType):
        self.backend_type = backend_type
        self._is_initialized = False
        self._loaded_models: Dict[str, ModelConfig] = {}

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化引擎"""
        pass

    @abstractmethod
    async def load_model(self, config: ModelConfig) -> bool:
        """加载模型"""
        pass

    @abstractmethod
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        pass

    @abstractmethod
    async def generate(
        self,
        model_name: str,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[InferenceResult]:
        """同步生成"""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """流式生成"""
        pass

    @abstractmethod
    async def get_stats(self, model_name: str) -> Dict[str, float]:
        """获取引擎统计"""
        pass

    @property
    def is_ready(self) -> bool:
        """引擎是否就绪"""
        return self._is_initialized

    @property
    def loaded_model_names(self) -> List[str]:
        """已加载的模型列表"""
        return list(self._loaded_models.keys())

    async def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self._loaded_models.get(model_name)

    async def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self._loaded_models
