"""配置管理模块"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self


class APIConfig(BaseModel):
    """API配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 300
    cors_enabled: bool = True
    cors_origins: list[str] = ["*"]
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 20


class SchedulerConfig(BaseModel):
    """调度器配置"""

    enabled: bool = True
    loop_interval_ms: int = 100
    max_concurrent_requests: int = 1000
    queue_max_size: int = 10000
    queue_high_priority_threshold: int = 8
    queue_default_priority: int = 5
    strategy_default: str = "adaptive"
    strategy_gang_enabled: bool = True
    strategy_gang_timeout_seconds: int = 300
    strategy_pack_enabled: bool = True
    strategy_pack_max_batch_size: int = 32


class ClusterConfig(BaseModel):
    """集群配置"""

    heartbeat_interval_seconds: int = 5
    heartbeat_timeout_seconds: int = 30
    node_labels: list[str] = ["type", "zone", "gpu_type"]


class WorkerConfig(BaseModel):
    """Worker配置"""

    heartbeat_interval_seconds: int = 3
    gpu_monitoring_interval_seconds: int = 5
    model_cache_size_gb: int = 100
    max_concurrent_inferences: int = 10


class InferenceBackendConfig(BaseModel):
    """推理后端基础配置"""

    backend_type: str = "vllm"
    default_tensor_parallel: int = 1
    default_pipeline_parallel: int = 1
    default_gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    enforce_eager: bool = False
    trust_remote_code: bool = True


class VLLMBackendConfig(InferenceBackendConfig):
    """vLLM后端配置"""

    backend_type: str = "vllm"
    block_size: int = 16
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True
    use_queue_for_batch_size: bool = True


class InferenceConfig(BaseModel):
    """推理配置"""

    default_backend: str = "vllm"
    backends: dict[str, InferenceBackendConfig] = {}
    defaults: dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 2048,
    }

    @field_validator("backends", mode="before")
    @classmethod
    def setup_backends(cls, v):
        if not v:
            return {"vllm": InferenceBackendConfig()}
        return v


class RedisConfig(BaseModel):
    """Redis配置"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    pool_size: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


class StorageConfig(BaseModel):
    """存储配置"""

    redis: RedisConfig = Field(default_factory=RedisConfig)
    model_registry_type: str = "filesystem"
    model_registry_base_path: str = "/models"


class MonitoringConfig(BaseModel):
    """监控配置"""

    enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    health_check_path: str = "/health"
    health_check_detailed: bool = True


class GrpcRateLimitConfig(BaseModel):
    """gRPC 限流配置"""

    enabled: bool = True
    qps: int = 100
    burst: int = 200


class RestApiRateLimitConfig(BaseModel):
    """REST API 限流配置"""

    enabled: bool = True
    qps: int = 100
    burst: int = 200
    per_endpoint: bool = False  # 是否按端点限流


class RateLimitConfig(BaseModel):
    """限流总配置"""

    enabled: bool = True
    rest_api: RestApiRateLimitConfig = Field(default_factory=RestApiRateLimitConfig)
    grpc: GrpcRateLimitConfig = Field(default_factory=GrpcRateLimitConfig)


class ServerConfig(BaseModel):
    """服务器配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class GrpcAuthConfig(BaseModel):
    """gRPC 认证配置"""

    enabled: bool = False
    api_keys: dict[str, str] = {}


class GrpcConfig(BaseModel):
    """gRPC 配置"""

    enabled: bool = False
    port: int = 50051
    max_workers: int = 10
    reflection_enabled: bool = True
    rate_limit: GrpcRateLimitConfig = Field(default_factory=GrpcRateLimitConfig)
    auth: GrpcAuthConfig = Field(default_factory=GrpcAuthConfig)


class ModelConfig(BaseModel):
    """模型配置"""

    allowed_download_sources: list[str] = ["hf://", "modelscope://"]
    cache_dir: str = "/root/.cache/huggingface"
    download_timeout_seconds: int = 3600


class AppConfig(BaseModel):
    """应用配置"""

    name: str = "QuantumFlow"
    version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"


class QuantumFlowConfig(BaseModel):
    """QuantumFlow完整配置"""

    app: AppConfig = Field(default_factory=AppConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    grpc: GrpcConfig = Field(default_factory=GrpcConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """从YAML文件加载配置"""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data) if data else cls()

    @classmethod
    def from_env(cls) -> Self:
        """从环境变量加载配置"""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    def to_yaml(self, path: str | Path) -> None:
        """保存为YAML文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# 全局配置实例
_config: QuantumFlowConfig | None = None


def get_config() -> QuantumFlowConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = QuantumFlowConfig()
    return _config


def load_config(
    config_file: str | Path | None = None,
    environment: str | None = None,
) -> QuantumFlowConfig:
    """加载配置

    加载顺序（后面的覆盖前面的）：
    1. 默认配置
    2. 配置文件
    3. 环境变量
    """
    global _config

    # 确定配置文件路径
    if config_file is None:
        env = environment or os.getenv("QF_ENVIRONMENT", "development")
        config_file = os.getenv("QF_CONFIG_FILE", f"configs/{env}.yaml")

    # 加载配置文件
    if config_file and Path(config_file).exists():
        _config = QuantumFlowConfig.from_file(config_file)
    else:
        _config = QuantumFlowConfig()

    return _config


def reload_config() -> QuantumFlowConfig:
    """重新加载配置"""
    return load_config()


@lru_cache(maxsize=1)
def get_default_config() -> QuantumFlowConfig:
    """获取默认配置（带缓存）"""
    return QuantumFlowConfig()
