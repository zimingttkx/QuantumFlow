"""推理引擎基类"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

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
    quantization: str | None = None  # awq, gptq, gguf
    trust_remote_code: bool = True

    # vLLM特有配置
    block_size: int = 16
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # 默认禁用（有 bug，需配合模型的 chunked prefill 支持）

    # HuggingFace 特有配置
    prefill_chunk_size: int = 512  # 分块预填充的块大小（tokens），超过此长度自动分块
    torch_compile: bool = True  # 是否启用 torch.compile 加速

    # 新增字段：与 ModelRegistry 关联
    model_id: str = ""                              # 规范化 ID（与 ModelInfo.name 对应）
    model_family: str = ""                          # qwen / llama / glm / deepseek
    backend: str = ""                               # 显式指定后端（vllm / tgi / sglang / huggingface / tensorrt_llm）
    preferred_gpu_families: list[str] = field(default_factory=list)  # ["hopper","ampere"]

    def to_dict(self) -> dict[str, Any]:
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
            "model_id": self.model_id,
            "model_family": self.model_family,
            "backend": self.backend,
            "preferred_gpu_families": list(self.preferred_gpu_families),
        }


@dataclass
class SamplingParams:
    """采样参数"""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    repetition_penalty: float = 1.0
    stop: list[str] | None = None
    # SGLang/TGI 特有参数
    presence_penalty: float = 0.0  # SGLang 支持
    frequency_penalty: float = 0.0  # SGLang 支持
    details: bool = False  # TGI 支持，返回 token 级详情

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "details": self.details,
        }


@dataclass
class InferenceResult:
    """推理结果"""

    request_id: str
    outputs: list[str]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class QueuedRequest:
    """
    队列中的请求项 — 支持优先级调度

    字段说明：
    - request_id: 请求唯一标识
    - model_name: 模型名称
    - prompt: 输入提示
    - sampling_params: 采样参数
    - priority: 优先级 (0-10, 0 最高, 10 最低)
    - submit_time: 提交时间戳（用于 FIFO 排序）
    - future: 异步 Future，推理完成后填充结果
    - tenant_id: 租户 ID（用于多租户场景）
    """

    request_id: str
    model_name: str
    prompt: str
    sampling_params: SamplingParams
    priority: int = 5  # 默认优先级
    submit_time: float = 0.0  # 将在 submit 时设置
    future: "asyncio.Future[InferenceResult] | None" = None
    tenant_id: str = "default"

    def __post_init__(self):
        if self.submit_time == 0.0:
            import time
            self.submit_time = time.time()

    def __lt__(self, other: "QueuedRequest") -> bool:
        """比较优先级：先按 priority，再按 submit_time (FIFO)"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.submit_time < other.submit_time


class InferenceEngine(ABC):
    """推理引擎抽象基类"""

    def __init__(self, backend_type: InferenceBackendType):
        self.backend_type = backend_type
        self._is_initialized = False
        self._loaded_models: dict[str, ModelConfig] = {}

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
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[InferenceResult]:
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
    async def get_stats(self, model_name: str) -> dict[str, float]:
        """获取引擎统计"""
        pass

    @property
    def is_ready(self) -> bool:
        """引擎是否就绪"""
        return self._is_initialized

    @property
    def loaded_model_names(self) -> list[str]:
        """已加载的模型列表"""
        return list(self._loaded_models.keys())

    async def get_model_config(self, model_name: str) -> ModelConfig | None:
        """获取模型配置"""
        return self._loaded_models.get(model_name)

    async def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self._loaded_models
