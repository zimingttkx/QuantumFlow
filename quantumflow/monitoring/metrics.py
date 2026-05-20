"""Prometheus监控指标"""

from prometheus_client import Counter, Gauge, Histogram, Info

# ==================== 请求指标 ====================

REQUEST_COUNT = Counter(
    "quantumflow_requests_total",
    "Total number of inference requests",
    ["node_id", "model", "status"],
)

REQUEST_LATENCY = Histogram(
    "quantumflow_request_latency_seconds",
    "Request latency in seconds",
    ["node_id", "model"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
)

# ==================== 队列指标 ====================

QUEUE_SIZE = Gauge(
    "quantumflow_queue_size",
    "Current queue size",
)

PENDING_REQUESTS = Gauge(
    "quantumflow_pending_requests",
    "Number of pending requests",
    ["node_id"],
)

# ==================== 集群指标 ====================

NODE_COUNT = Gauge(
    "quantumflow_nodes_total",
    "Total number of nodes",
    ["node_id", "status"],
)

# ==================== GPU指标 ====================

GPU_UTILIZATION = Gauge(
    "quantumflow_gpu_utilization",
    "GPU utilization percentage",
    ["node_id", "gpu_id"],
)

GPU_MEMORY = Gauge(
    "quantumflow_gpu_memory_bytes",
    "GPU memory usage in bytes",
    ["node_id", "gpu_id"],
)

# ==================== 模型指标 ====================

MODEL_LOADED = Gauge(
    "quantumflow_model_loaded",
    "Whether a model is loaded (1=yes, 0=no)",
    ["node_id", "model"],
)

# ==================== 推理指标 ====================

ACTIVE_INFERENCES = Gauge(
    "quantumflow_active_inferences",
    "Number of active inferences",
    ["node_id", "model"],
)

# ==================== 系统指标 ====================

SYSTEM_INFO = Info(
    "quantumflow_system",
    "System information",
)

# 初始化系统信息
from quantumflow.version import __version__

SYSTEM_INFO.info(
    {
        "version": __version__,
        "name": "QuantumFlow",
    }
)
