"""常量定义"""

from enum import Enum

# 版本信息
VERSION = "1.0.0"
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


# 节点状态
class NodeStatus(str, Enum):
    INITIALIZING = "initializing"
    JOINING = "joining"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


# 模型状态
class ModelStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    ERROR = "error"


# 作业状态
class JobStatus(str, Enum):
    QUEUED = "queued"
    SCHEDULING = "scheduling"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 推理后端类型
class InferenceBackendType(str, Enum):
    VLLM = "vllm"
    TGI = "text-generation-inference"
    SGLANG = "sglang"
    TRT_LLM = "tensorrt-llm"
    LIGER = "liger"
    HUGGINGFACE = "huggingface"


# 调度策略类型
class SchedulingStrategyType(str, Enum):
    GANG = "gang"
    PACK = "pack"
    ADAPTIVE = "adaptive"


# 并行策略类型
class ParallelStrategyType(str, Enum):
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    DATA_PARALLEL = "data_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"


# 资源类型
class ResourceType(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"


# 默认配置值
DEFAULT_CONFIG = {
    # API配置
    "api.host": "0.0.0.0",
    "api.port": 8000,
    "api.workers": 4,
    "api.timeout": 300,
    # 调度器配置
    "scheduler.loop_interval_ms": 100,
    "scheduler.max_concurrent_requests": 1000,
    "scheduler.queue.max_size": 10000,
    "scheduler.strategy.default": "adaptive",
    # 集群配置
    "cluster.heartbeat_interval_seconds": 5,
    "cluster.heartbeat_timeout_seconds": 30,
    # Worker配置
    "worker.heartbeat_interval_seconds": 3,
    "worker.gpu_monitoring_interval_seconds": 5,
    # 推理配置
    "inference.default_backend": "vllm",
    "inference.vllm.default_tensor_parallel": 1,
    "inference.vllm.default_gpu_memory_utilization": 0.9,
    "inference.vllm.max_model_len": 8192,
    # 存储配置
    "storage.redis.host": "localhost",
    "storage.redis.port": 6379,
    "storage.redis.db": 0,
    # 监控配置
    "monitoring.enabled": True,
    "monitoring.metrics_port": 9090,
}

# GPU相关常量
GPU_MEMORY_FRACTION = 0.92  # GPU显存使用比例
GPU_UTILIZATION_THRESHOLD = 0.95  # GPU利用率阈值


# 超时配置
TIMEOUT_CONFIG = {
    "request.default": 300,  # 默认请求超时
    "request.max": 3600,  # 最大请求超时
    "model.load": 600,  # 模型加载超时
    "node.heartbeat": 30,  # 节点心跳超时
    "schedule.evaluate": 5,  # 调度评估超时
}

# 性能调优参数
PERFORMANCE_CONFIG = {
    "batch.max_size": 32,  # 最大批处理大小
    "batch.timeout_ms": 100,  # 批处理超时
    "cache.model.size_gb": 100,  # 模型缓存大小
    "prefill.max_batch_size": 16,  # Prefill阶段最大批处理
    "decode.max_batch_size": 64,  # Decode阶段最大批处理
}

# 日志配置
LOG_CONFIG = {
    "format": "json",
    "level": "INFO",
    "request_id_header": "X-Request-ID",
    "include_timestamp": True,
    "include_extra": True,
}

# gRPC配置
GRPC_CONFIG = {
    "max_receive_message_length": 100 * 1024 * 1024,  # 100MB
    "max_send_message_length": 100 * 1024 * 1024,  # 100MB
    "compression": "gzip",
}

# 指标收集配置
METRICS_CONFIG = {
    "enabled": True,
    "port": 9090,
    "path": "/metrics",
    "collect_interval_seconds": 10,
}

# 健康检查配置
HEALTH_CHECK_CONFIG = {
    "enabled": True,
    "path": "/health",
    "detailed": True,
    "checks": ["redis", "cluster", "models"],
}

# 租户相关常量
TENANT_PREFIX = "qf:tenant:"
TENANT_API_KEY_PREFIX = "qf:apikey:"
TENANT_QUOTA_PREFIX = "qf:quota:"
TENANT_USAGE_PREFIX = "qf:usage:"
DEFAULT_TENANT_QUOTA = {
    "requests_per_minute": 60,
    "requests_per_day": 10000,
    "max_tokens_per_request": 8192,
    "gpu_memory_mb": 4096,
    "concurrent_requests": 10,
}
