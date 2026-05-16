"""API请求模型定义"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class SamplingParams(BaseModel):
    """采样参数"""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p采样")
    top_k: int = Field(default=50, ge=0, description="Top-k采样")
    max_tokens: int = Field(default=2048, ge=1, le=32768, description="最大生成token数")
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0, description="重复惩罚")
    stop: Optional[List[str]] = Field(default=None, description="停止词列表")

    @field_validator("max_tokens", mode="before")
    @classmethod
    def validate_max_tokens(cls, v):
        if v is None:
            return 2048
        return v


class InferenceRequest(BaseModel):
    """推理请求"""

    model: str = Field(..., description="模型名称或路径")
    prompt: str = Field(..., description="输入提示词")
    sampling_params: Optional[SamplingParams] = Field(
        default_factory=SamplingParams, description="采样参数"
    )
    stream: bool = Field(default=False, description="是否使用流式输出")
    session_id: Optional[str] = Field(default=None, description="会话ID，用于上下文")
    priority: int = Field(default=5, ge=0, le=10, description="请求优先级")
    tags: Optional[Dict[str, str]] = Field(default=None, description="标签，用于资源隔离")
    request_id: Optional[str] = Field(default=None, description="请求ID，不提供则自动生成")

    model_config = {"json_schema_extra": {"example": {
        "model": "Qwen2.5-7B-Instruct",
        "prompt": "解释量子计算的基本原理",
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 1024
        },
        "stream": False
    }}}


class BatchInferenceRequest(BaseModel):
    """批量推理请求"""

    model: str = Field(..., description="模型名称或路径")
    prompts: List[str] = Field(..., min_length=1, max_length=100, description="提示词列表")
    sampling_params: Optional[SamplingParams] = Field(
        default_factory=SamplingParams, description="采样参数"
    )
    priority: int = Field(default=5, ge=0, le=10, description="请求优先级")

    model_config = {"json_schema_extra": {"example": {
        "model": "Qwen2.5-7B-Instruct",
        "prompts": [
            "解释量子计算",
            "什么是机器学习"
        ],
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 512
        }
    }}}


class ChatMessage(BaseModel):
    """对话消息"""

    role: str = Field(..., description="角色: system, user, assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """对话请求"""

    model: str = Field(..., description="模型名称或路径")
    messages: List[ChatMessage] = Field(..., min_length=1, description="对话历史")
    sampling_params: Optional[SamplingParams] = Field(
        default_factory=SamplingParams, description="采样参数"
    )
    stream: bool = Field(default=False, description="是否使用流式输出")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    priority: int = Field(default=5, ge=0, le=10, description="请求优先级")

    model_config = {"json_schema_extra": {"example": {
        "model": "Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手"},
            {"role": "user", "content": "解释什么是量子计算"}
        ],
        "stream": False
    }}}


class DeployRequest(BaseModel):
    """模型部署请求"""

    model: str = Field(..., description="模型名称或HuggingFace路径")
    tensor_parallel: int = Field(default=1, ge=1, le=8, description="张量并行度")
    pipeline_parallel: int = Field(default=1, ge=1, le=4, description="流水线并行度")
    gpu_memory_utilization: float = Field(
        default=0.9, ge=0.1, le=1.0, description="GPU显存使用比例"
    )
    max_model_length: Optional[int] = Field(
        default=None, ge=1, le=32768, description="最大模型长度"
    )
    backend: str = Field(default="vllm", description="推理后端")
    auto_scaling: bool = Field(default=False, description="是否启用自动扩缩容")
    replicas: int = Field(default=1, ge=1, le=10, description="副本数")
    min_replicas: int = Field(default=1, ge=0, description="最小副本数")
    max_replicas: int = Field(default=3, ge=1, description="最大副本数")
    target_gpu_utilization: float = Field(
        default=0.7, ge=0.1, le=1.0, description="目标GPU利用率"
    )
    quantization: Optional[str] = Field(
        default=None, description="量化方式: awq, gptq, gguf"
    )
    dtype: str = Field(default="auto", description="数据类型: auto, float16, bfloat16, float32")

    model_config = {"json_schema_extra": {"example": {
        "model": "Qwen2.5-72B-Instruct",
        "tensor_parallel": 4,
        "gpu_memory_utilization": 0.9,
        "backend": "vllm",
        "replicas": 2
    }}}


class UndeployRequest(BaseModel):
    """模型卸载请求"""

    model: str = Field(..., description="模型名称")
    force: bool = Field(default=False, description="是否强制卸载")


class NodeActionRequest(BaseModel):
    """节点操作请求"""

    action: str = Field(..., description="操作: drain, uncordon, restart")
    reason: Optional[str] = Field(default=None, description="操作原因")


class ModelFilterRequest(BaseModel):
    """模型过滤请求"""

    status: Optional[str] = Field(default=None, description="状态过滤")
    backend: Optional[str] = Field(default=None, description="后端过滤")
    labels: Optional[Dict[str, str]] = Field(default=None, description="标签过滤")


class BenchmarkRequest(BaseModel):
    """基准测试请求"""

    model: str = Field(..., description="模型名称")
    test_set: str = Field(
        default="mmlu", description="测试集: mmlu, humaneval, math, custom"
    )
    num_samples: int = Field(default=100, ge=1, le=1000, description="样本数量")
    backend: str = Field(default="vllm", description="推理后端")
    tensor_parallel: int = Field(default=1, ge=1, description="张量并行度")
    sampling_params: Optional[SamplingParams] = Field(
        default_factory=SamplingParams, description="采样参数"
    )
