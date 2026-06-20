"""QuantumFlow 核心常量类专业测试

测试策略:
1. 所有枚举类成员值、字符串/整数基类行为
2. 配置字典结构完整性：所有必需键存在且值类型正确
3. 枚举成员不可变性（不存在隐藏的无效成员）
4. 枚举值唯一性
"""

from quantumflow.core.constants import (
    API_PREFIX,
    API_VERSION,
    DEFAULT_CONFIG,
    GRPC_CONFIG,
    GPU_MEMORY_FRACTION,
    GPU_UTILIZATION_THRESHOLD,
    HEALTH_CHECK_CONFIG,
    InferenceBackendType,
    JobStatus,
    LOG_CONFIG,
    METRICS_CONFIG,
    ModelStatus,
    NodeStatus,
    PERFORMANCE_CONFIG,
    SchedulingStrategyType,
    ParallelStrategyType,
    ResourceType,
    TIMEOUT_CONFIG,
    VERSION,
)
from quantumflow.storage.redis_queue import QueuePriority


# ═══════════════════════════════════════════════════════════════════════════════
# 版本和前缀常量
# ═══════════════════════════════════════════════════════════════════════════════


class TestVersionAndPrefix:
    """版本号和 API 前缀"""

    def test_version_is_non_empty_string(self):
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0

    def test_api_prefix_contains_api_version(self):
        assert API_VERSION in API_PREFIX
        assert API_PREFIX.startswith("/api/")
        assert API_PREFIX == f"/api/{API_VERSION}"


# ═══════════════════════════════════════════════════════════════════════════════
# 枚举类通用行为
# ═══════════════════════════════════════════════════════════════════════════════


class TestNodeStatus:
    """节点状态枚举"""

    def test_all_members_exist(self):
        members = {m.name for m in NodeStatus}
        expected = {"INITIALIZING", "JOINING", "HEALTHY", "UNHEALTHY", "DRAINING", "OFFLINE"}
        assert members == expected

    def test_string_values(self):
        assert NodeStatus.INITIALIZING.value == "initializing"
        assert NodeStatus.JOINING.value == "joining"
        assert NodeStatus.HEALTHY.value == "healthy"
        assert NodeStatus.UNHEALTHY.value == "unhealthy"
        assert NodeStatus.DRAINING.value == "draining"
        assert NodeStatus.OFFLINE.value == "offline"

    def test_is_str_enum(self):
        from enum import Enum
        assert issubclass(NodeStatus, str)
        assert issubclass(NodeStatus, Enum)

    def test_direct_string_comparison(self):
        assert NodeStatus.HEALTHY == "healthy"
        assert "healthy" == NodeStatus.HEALTHY

    def test_no_duplicate_values(self):
        values = [m.value for m in NodeStatus]
        assert len(values) == len(set(values))


class TestModelStatus:
    """模型状态枚举"""

    def test_all_members_exist(self):
        members = {m.name for m in ModelStatus}
        expected = {"LOADING", "READY", "UNLOADING", "ERROR"}
        assert members == expected

    def test_string_values(self):
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.UNLOADING.value == "unloading"
        assert ModelStatus.ERROR.value == "error"


class TestJobStatus:
    """作业状态枚举"""

    def test_all_members_exist(self):
        members = {m.name for m in JobStatus}
        expected = {
            "QUEUED", "SCHEDULING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED",
        }
        assert members == expected

    def test_string_values(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.SCHEDULING.value == "scheduling"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_terminal_states_are_distinct(self):
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        assert len(terminal) == 3


class TestInferenceBackendType:
    """推理后端类型枚举"""

    def test_all_members_exist(self):
        members = {m.name for m in InferenceBackendType}
        expected = {"VLLM", "TGI", "SGLANG", "TRT_LLM", "LIGER", "HUGGINGFACE"}
        assert members == expected

    def test_string_values(self):
        assert InferenceBackendType.VLLM.value == "vllm"
        assert InferenceBackendType.TGI.value == "text-generation-inference"
        assert InferenceBackendType.SGLANG.value == "sglang"
        assert InferenceBackendType.TRT_LLM.value == "tensorrt-llm"
        assert InferenceBackendType.LIGER.value == "liger"
        assert InferenceBackendType.HUGGINGFACE.value == "huggingface"


class TestSchedulingStrategyType:
    """调度策略类型枚举"""

    def test_all_members_exist(self):
        members = {m.name for m in SchedulingStrategyType}
        expected = {"GANG", "PACK", "ADAPTIVE"}
        assert members == expected

    def test_string_values(self):
        assert SchedulingStrategyType.GANG.value == "gang"
        assert SchedulingStrategyType.PACK.value == "pack"
        assert SchedulingStrategyType.ADAPTIVE.value == "adaptive"


class TestParallelStrategyType:
    """并行策略类型枚举"""

    def test_all_members_exist(self):
        members = {m.name for m in ParallelStrategyType}
        expected = {
            "TENSOR_PARALLEL", "PIPELINE_PARALLEL",
            "DATA_PARALLEL", "HYBRID_PARALLEL",
        }
        assert members == expected

    def test_string_values(self):
        assert ParallelStrategyType.TENSOR_PARALLEL.value == "tensor_parallel"
        assert ParallelStrategyType.PIPELINE_PARALLEL.value == "pipeline_parallel"
        assert ParallelStrategyType.DATA_PARALLEL.value == "data_parallel"
        assert ParallelStrategyType.HYBRID_PARALLEL.value == "hybrid_parallel"


class TestResourceType:
    """资源类型枚举"""

    def test_all_members_exist(self):
        members = {m.name for m in ResourceType}
        expected = {"GPU", "CPU", "MEMORY", "DISK"}
        assert members == expected

    def test_string_values(self):
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.DISK.value == "disk"


class TestQueuePriority:
    """队列优先级枚举"""

    def test_all_members_exist(self):
        members = {m.name for m in QueuePriority}
        expected = {"LOW", "NORMAL", "HIGH", "CRITICAL"}
        assert members == expected

    def test_int_values_and_ordering(self):
        assert QueuePriority.LOW.value == 0
        assert QueuePriority.NORMAL.value == 5
        assert QueuePriority.HIGH.value == 7
        assert QueuePriority.CRITICAL.value == 9
        assert QueuePriority.LOW < QueuePriority.NORMAL < QueuePriority.HIGH < QueuePriority.CRITICAL

    def test_is_int_enum(self):
        from enum import Enum
        assert issubclass(QueuePriority, int)
        assert issubclass(QueuePriority, Enum)

    def test_direct_int_comparison(self):
        assert QueuePriority.LOW == 0
        assert QueuePriority.NORMAL == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 数字常量
# ═══════════════════════════════════════════════════════════════════════════════


class TestGPUNumericConstants:
    """GPU 相关数值常量"""

    def test_gpu_memory_fraction_range(self):
        assert 0.0 < GPU_MEMORY_FRACTION <= 1.0

    def test_gpu_utilization_threshold_range(self):
        assert 0.0 < GPU_UTILIZATION_THRESHOLD <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 配置字典结构完整性
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefaultConfig:
    """DEFAULT_CONFIG 结构验证"""

    def test_api_section_keys(self):
        for key in ("api.host", "api.port", "api.workers", "api.timeout"):
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_scheduler_section_keys(self):
        for key in (
            "scheduler.loop_interval_ms",
            "scheduler.max_concurrent_requests",
            "scheduler.queue.max_size",
            "scheduler.strategy.default",
        ):
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_cluster_section_keys(self):
        for key in (
            "cluster.heartbeat_interval_seconds",
            "cluster.heartbeat_timeout_seconds",
        ):
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_worker_section_keys(self):
        for key in (
            "worker.heartbeat_interval_seconds",
            "worker.gpu_monitoring_interval_seconds",
        ):
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_inference_section_keys(self):
        for key in (
            "inference.default_backend",
            "inference.vllm.default_tensor_parallel",
            "inference.vllm.default_gpu_memory_utilization",
            "inference.vllm.max_model_len",
        ):
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_storage_section_keys(self):
        for key in (
            "storage.redis.host",
            "storage.redis.port",
            "storage.redis.db",
        ):
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_monitoring_section_keys(self):
        for key in ("monitoring.enabled", "monitoring.metrics_port"):
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_value_types(self):
        assert isinstance(DEFAULT_CONFIG["api.port"], int)
        assert isinstance(DEFAULT_CONFIG["api.workers"], int)
        assert isinstance(DEFAULT_CONFIG["scheduler.loop_interval_ms"], int)
        assert isinstance(DEFAULT_CONFIG["inference.vllm.default_gpu_memory_utilization"], float)
        assert isinstance(DEFAULT_CONFIG["storage.redis.port"], int)
        assert isinstance(DEFAULT_CONFIG["storage.redis.db"], int)
        assert isinstance(DEFAULT_CONFIG["monitoring.enabled"], bool)
        assert isinstance(DEFAULT_CONFIG["monitoring.metrics_port"], int)

    def test_reasonable_value_ranges(self):
        assert DEFAULT_CONFIG["api.port"] > 0 and DEFAULT_CONFIG["api.port"] < 65536
        assert DEFAULT_CONFIG["api.workers"] >= 1
        assert DEFAULT_CONFIG["api.timeout"] > 0
        assert DEFAULT_CONFIG["scheduler.loop_interval_ms"] > 0
        assert DEFAULT_CONFIG["scheduler.max_concurrent_requests"] > 0
        assert DEFAULT_CONFIG["scheduler.queue.max_size"] > 0
        assert 0.0 < DEFAULT_CONFIG["inference.vllm.default_gpu_memory_utilization"] <= 1.0
        assert DEFAULT_CONFIG["inference.vllm.max_model_len"] > 0
        assert DEFAULT_CONFIG["storage.redis.port"] > 0


class TestTimeoutConfig:
    """TIMEOUT_CONFIG 结构验证"""

    REQUIRED_KEYS = [
        "request.default",
        "request.max",
        "model.load",
        "node.heartbeat",
        "schedule.evaluate",
    ]

    def test_all_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in TIMEOUT_CONFIG, f"Missing key: {key}"

    def test_all_values_are_positive_integers(self):
        for key, val in TIMEOUT_CONFIG.items():
            assert isinstance(val, int), f"{key} is not int: {type(val)}"
            assert val > 0, f"{key} is not positive: {val}"

    def test_request_max_gte_request_default(self):
        assert TIMEOUT_CONFIG["request.max"] >= TIMEOUT_CONFIG["request.default"]


class TestPerformanceConfig:
    """PERFORMANCE_CONFIG 结构验证"""

    REQUIRED_KEYS = [
        "batch.max_size",
        "batch.timeout_ms",
        "cache.model.size_gb",
        "prefill.max_batch_size",
        "decode.max_batch_size",
    ]

    def test_all_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in PERFORMANCE_CONFIG, f"Missing key: {key}"

    def test_all_values_are_positive_integers(self):
        for key, val in PERFORMANCE_CONFIG.items():
            assert isinstance(val, int), f"{key} is not int: {type(val)}"
            assert val > 0, f"{key} is not positive: {val}"


class TestLogConfig:
    """LOG_CONFIG 结构验证"""

    REQUIRED_KEYS = [
        "format",
        "level",
        "request_id_header",
        "include_timestamp",
        "include_extra",
    ]

    def test_all_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in LOG_CONFIG, f"Missing key: {key}"

    def test_format_is_json(self):
        assert LOG_CONFIG["format"] == "json"

    def test_boolean_flags_are_bools(self):
        assert isinstance(LOG_CONFIG["include_timestamp"], bool)
        assert isinstance(LOG_CONFIG["include_extra"], bool)


class TestGRPCConfig:
    """GRPC_CONFIG 结构验证"""

    REQUIRED_KEYS = [
        "max_receive_message_length",
        "max_send_message_length",
        "compression",
    ]

    def test_all_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in GRPC_CONFIG, f"Missing key: {key}"

    def test_message_lengths_exceed_minimum(self):
        # 至少 4MB（gRPC 默认最小值）
        assert GRPC_CONFIG["max_receive_message_length"] >= 4 * 1024 * 1024
        assert GRPC_CONFIG["max_send_message_length"] >= 4 * 1024 * 1024

    def test_compression_is_gzip(self):
        assert GRPC_CONFIG["compression"] == "gzip"


class TestMetricsConfig:
    """METRICS_CONFIG 结构验证"""

    REQUIRED_KEYS = [
        "enabled",
        "port",
        "path",
        "collect_interval_seconds",
    ]

    def test_all_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in METRICS_CONFIG, f"Missing key: {key}"

    def test_path_starts_with_slash(self):
        assert METRICS_CONFIG["path"].startswith("/")

    def test_enabled_is_bool(self):
        assert isinstance(METRICS_CONFIG["enabled"], bool)

    def test_interval_is_positive(self):
        assert METRICS_CONFIG["collect_interval_seconds"] > 0


class TestHealthCheckConfig:
    """HEALTH_CHECK_CONFIG 结构验证"""

    REQUIRED_KEYS = ["enabled", "path", "detailed", "checks"]

    def test_all_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in HEALTH_CHECK_CONFIG, f"Missing key: {key}"

    def test_checks_is_non_empty_list(self):
        assert isinstance(HEALTH_CHECK_CONFIG["checks"], list)
        assert len(HEALTH_CHECK_CONFIG["checks"]) > 0

    def test_checks_contain_expected_items(self):
        for check in ["redis", "cluster", "models"]:
            assert check in HEALTH_CHECK_CONFIG["checks"], f"Missing check: {check}"

    def test_enabled_and_detailed_are_bools(self):
        assert isinstance(HEALTH_CHECK_CONFIG["enabled"], bool)
        assert isinstance(HEALTH_CHECK_CONFIG["detailed"], bool)

    def test_path_starts_with_slash(self):
        assert HEALTH_CHECK_CONFIG["path"].startswith("/")


# ═══════════════════════════════════════════════════════════════════════════════
# 一致性交叉验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestConsistencyAcrossConfigs:
    """各配置字典间的交叉一致性"""

    def test_metrics_port_consistent(self):
        """DEFAULT_CONFIG 和 METRICS_CONFIG 的端口一致"""
        assert (
            DEFAULT_CONFIG["monitoring.metrics_port"]
            == METRICS_CONFIG["port"]
        )

    def test_timeout_references_match(self):
        """超时值在合理范围内交叉一致"""
        assert (
            DEFAULT_CONFIG["cluster.heartbeat_timeout_seconds"]
            > DEFAULT_CONFIG["cluster.heartbeat_interval_seconds"]
        )
