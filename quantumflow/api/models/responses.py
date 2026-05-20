"""API响应模型定义"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class InferenceResponse(BaseModel):
    """推理响应"""

    request_id: str = Field(..., description="请求ID")
    model: str = Field(..., description="模型名称")
    prompt: str = Field(..., description="输入提示词")
    generated_text: str = Field(..., description="生成的文本")
    finish_reason: str = Field(..., description="结束原因: stop, length, timeout")
    latency_ms: float = Field(..., description="延迟（毫秒）")
    usage: dict[str, int] = Field(
        ...,
        description="Token使用统计",
        json_schema_extra={
            "example": {"prompt_tokens": 100, "completion_tokens": 500, "total_tokens": 600}
        },
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_00000001",
                "model": "Qwen2.5-7B-Instruct",
                "prompt": "解释量子计算的基本原理",
                "generated_text": "量子计算是一种利用量子力学原理进行信息处理的计算方式...",
                "finish_reason": "stop",
                "latency_ms": 1523.4,
                "usage": {"prompt_tokens": 50, "completion_tokens": 500, "total_tokens": 550},
            }
        }
    }


class StreamResponse(BaseModel):
    """流式响应"""

    request_id: str = Field(..., description="请求ID")
    delta: str = Field(..., description="增量文本")
    is_final: bool = Field(..., description="是否是最后一个片段")
    usage: dict[str, int] | None = Field(default=None, description="Token使用统计")
    finish_reason: str | None = Field(default=None, description="结束原因")


class BatchInferenceResponse(BaseModel):
    """批量推理响应"""

    batch_id: str = Field(..., description="批次ID")
    model: str = Field(..., description="模型名称")
    total: int = Field(..., description="总请求数")
    completed: int = Field(..., description="完成数")
    failed: int = Field(..., description="失败数")
    results: list[InferenceResponse] = Field(..., description="推理结果列表")
    total_latency_ms: float = Field(..., description="总延迟（毫秒）")
    avg_latency_ms: float = Field(..., description="平均延迟（毫秒）")


class TokenUsage(BaseModel):
    """Token使用统计"""

    prompt_tokens: int = Field(..., description="输入Token数")
    completion_tokens: int = Field(..., description="生成Token数")
    total_tokens: int = Field(..., description="总Token数")


class GPUInfo(BaseModel):
    """GPU信息"""

    gpu_id: int = Field(..., description="GPU ID")
    name: str = Field(..., description="GPU名称")
    memory_total: int = Field(..., description="总显存（字节）")
    memory_used: int = Field(..., description="已用显存（字节）")
    memory_free: int = Field(..., description="可用显存（字节）")
    utilization: float = Field(..., description="利用率（0-1）")
    temperature: float = Field(..., description="温度（摄氏度）")

    @property
    def memory_used_percent(self) -> float:
        """显存使用百分比"""
        if self.memory_total == 0:
            return 0.0
        return self.memory_used / self.memory_total


class ModelInfo(BaseModel):
    """模型信息"""

    model_id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    architecture: str = Field(..., description="模型架构")
    parameter_count: int = Field(..., description="参数量")
    quantization: str | None = Field(default=None, description="量化方式")
    dtype: str = Field(..., description="数据类型")
    status: str = Field(..., description="状态: loading, ready, error")
    replicas: int = Field(..., description="副本数")
    tensor_parallel: int = Field(..., description="张量并行度")
    max_model_length: int = Field(..., description="最大模型长度")
    backend: str = Field(..., description="推理后端")
    loaded_on_nodes: list[str] = Field(default_factory=list, description="加载该模型的节点列表")
    gpu_usage: dict[str, float] = Field(default_factory=dict, description="GPU显存使用")
    request_count: int = Field(default=0, description="请求计数")
    avg_latency_ms: float = Field(default=0.0, description="平均延迟")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "qwen2.5-7b",
                "name": "Qwen2.5-7B-Instruct",
                "architecture": "Qwen2ForCausalLM",
                "parameter_count": 7_000_000_000,
                "dtype": "bfloat16",
                "status": "ready",
                "replicas": 2,
                "tensor_parallel": 1,
                "max_model_length": 8192,
                "backend": "vllm",
                "loaded_on_nodes": ["node-1", "node-2"],
            }
        }
    }


class NodeInfo(BaseModel):
    """节点信息"""

    node_id: str = Field(..., description="节点ID")
    hostname: str = Field(..., description="主机名")
    ip: str = Field(..., description="IP地址")
    port: int = Field(..., description="端口")
    status: str = Field(..., description="状态: healthy, unhealthy, draining, offline")
    gpu_count: int = Field(..., description="GPU数量")
    gpu_info: list[GPUInfo] = Field(default_factory=list, description="GPU详细信息")
    cpu_count: int = Field(..., description="CPU核心数")
    memory_total: int = Field(..., description="总内存（字节）")
    memory_available: int = Field(..., description="可用内存（字节）")
    disk_total: int = Field(..., description="总磁盘空间（字节）")
    disk_available: int = Field(..., description="可用磁盘空间（字节）")
    current_load: float = Field(..., description="当前负载（0-1）")
    labels: dict[str, str] = Field(default_factory=dict, description="节点标签")
    version: str = Field(..., description="QuantumFlow版本")
    uptime_seconds: int = Field(..., description="运行时间（秒）")
    last_heartbeat: datetime = Field(..., description="最后心跳时间")
    loaded_models: list[str] = Field(default_factory=list, description="已加载模型")

    @property
    def memory_available_percent(self) -> float:
        """内存可用百分比"""
        if self.memory_total == 0:
            return 0.0
        return self.memory_available / self.memory_total


class JobInfo(BaseModel):
    """作业信息"""

    job_id: str = Field(..., description="作业ID")
    model: str = Field(..., description="模型名称")
    status: str = Field(
        ..., description="状态: queued, scheduling, running, completed, failed, cancelled"
    )
    priority: int = Field(..., description="优先级")
    created_at: datetime = Field(..., description="创建时间")
    started_at: datetime | None = Field(default=None, description="开始时间")
    completed_at: datetime | None = Field(default=None, description="完成时间")
    progress: float = Field(..., description="进度（0-1）")
    prompt: str = Field(..., description="输入提示")
    result: str | None = Field(default=None, description="生成结果")
    error: str | None = Field(default=None, description="错误信息")
    allocated_nodes: list[str] = Field(default_factory=list, description="分配的节点")
    retry_count: int = Field(default=0, description="重试次数")


class ClusterStatus(BaseModel):
    """集群状态"""

    total_nodes: int = Field(..., description="总节点数")
    healthy_nodes: int = Field(..., description="健康节点数")
    unhealthy_nodes: int = Field(..., description="不健康节点数")
    draining_nodes: int = Field(..., description="排空中的节点数")
    total_gpus: int = Field(..., description="总GPU数")
    available_gpus: int = Field(..., description="可用GPU数")
    active_models: int = Field(..., description="活跃模型数")
    pending_jobs: int = Field(..., description="等待中的作业数")
    running_jobs: int = Field(..., description="运行中的作业数")
    system_metrics: dict[str, float] = Field(default_factory=dict, description="系统指标")
    uptime_seconds: int = Field(..., description="运行时间")


class DeployResponse(BaseModel):
    """部署响应"""

    model_id: str = Field(..., description="模型ID")
    status: str = Field(..., description="状态")
    replicas: int = Field(..., description="副本数")
    message: str = Field(..., description="消息")


class UndeployResponse(BaseModel):
    """卸载响应"""

    model_id: str = Field(..., description="模型ID")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="状态: healthy, degraded, unhealthy")
    version: str = Field(..., description="版本")
    uptime_seconds: int = Field(..., description="运行时间")
    checks: dict[str, Any] = Field(default_factory=dict, description="各项检查结果")


class ErrorResponse(BaseModel):
    """错误响应"""

    error: ErrorDetail = Field(..., description="错误详情")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "MODEL_NOT_FOUND",
                    "message": "Model not found: Qwen2.5-7B",
                    "details": {},
                }
            }
        }
    }


class ErrorDetail(BaseModel):
    """错误详情"""

    code: str = Field(..., description="错误码")
    message: str = Field(..., description="错误消息")
    details: dict[str, Any] = Field(default_factory=dict, description="详细信息")


class BenchmarkResponse(BaseModel):
    """基准测试响应"""

    benchmark_id: str = Field(..., description="测试ID")
    model: str = Field(..., description="模型名称")
    test_set: str = Field(..., description="测试集")
    status: str = Field(..., description="状态")
    total_samples: int = Field(..., description="总样本数")
    completed_samples: int = Field(..., description="已完成样本数")
    results: dict[str, Any] | None = Field(default=None, description="测试结果")
    metrics: dict[str, float] | None = Field(default=None, description="性能指标")


class MetricsResponse(BaseModel):
    """指标响应"""

    timestamp: datetime = Field(..., description="时间戳")
    system: dict[str, float] = Field(default_factory=dict, description="系统指标")
    models: dict[str, dict[str, float]] = Field(default_factory=dict, description="模型指标")
    nodes: dict[str, dict[str, float]] = Field(default_factory=dict, description="节点指标")


class LoadModelRequest(BaseModel):
    """加载模型请求"""

    model: str = Field(..., description="模型名称（简称，如 Qwen2.5-7B）")
    model_path: str | None = Field(None, description="模型路径（HuggingFace ID 或本地路径）")
    backend: str | None = Field(
        "huggingface", description="推理后端: huggingface, vllm, tgi, sglang"
    )
    tensor_parallel: int | None = Field(1, description="张量并行度")
    gpu_memory_utilization: float | None = Field(
        0.6, description="GPU显存利用率"
    )  # 降低显存使用率适应RTX 4080 Laptop
    max_model_len: int | None = Field(2048, description="最大模型长度")  # 降低max_model_len适应显存
    dtype: str | None = Field("auto", description="数据类型: auto, float16, bfloat16, float32")
    quantization: str | None = Field(None, description="量化方式: awq, gptq, gguf")
    # TGI 特有配置
    tgi_base_url: str | None = Field(
        None, description="TGI 服务器地址（仅 TGI 后端生效）"
    )
    # SGLang 特有配置
    sglang_base_url: str | None = Field(
        None, description="SGLang 服务器地址（仅 SGLang 后端生效）"
    )
    sglang_timeout: int | None = Field(
        None, description="SGLang 请求超时（秒，仅 SGLang 后端生效）"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "Qwen2.5-1.5B",
                "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "backend": "huggingface",
                "tensor_parallel": 1,
                "gpu_memory_utilization": 0.8,
            }
        }
    }


class LoadModelResponse(BaseModel):
    """加载模型响应"""

    model: str = Field(..., description="模型名称")
    status: str = Field(..., description="状态: loaded, loading, failed")
    message: str = Field(..., description="消息")


class UnloadModelResponse(BaseModel):
    """卸载模型响应"""

    model: str = Field(..., description="模型名称")
    status: str = Field(..., description="状态: unloaded")
    message: str = Field(..., description="消息")


class ModelStatusResponse(BaseModel):
    """模型状态响应"""

    loaded_models: list[str] = Field(..., description="已加载模型列表")
    total: int = Field(..., description="总数")
