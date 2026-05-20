"""Prometheus监控指标 - 单元测试

测试覆盖:
1. 所有指标定义存在且类型正确
2. 指标名称和标签符合规格
3. 指标可正常操作 (observe, inc, set, labels)
4. SYSTEM_INFO 版本信息已初始化
5. Histogram buckets 配置正确
"""

import pytest

# prometheus_client registers metrics globally on import
from prometheus_client import Counter, Gauge, Histogram, Info

from quantumflow.monitoring.metrics import (
    ACTIVE_INFERENCES,
    GPU_MEMORY,
    GPU_UTILIZATION,
    MODEL_LOADED,
    NODE_COUNT,
    PENDING_REQUESTS,
    QUEUE_SIZE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SYSTEM_INFO,
)


# ==================== 指标类型验证 ====================


class TestMetricTypes:
    """所有指标类型验证"""

    def test_request_count_is_counter(self):
        assert isinstance(REQUEST_COUNT, Counter), f"REQUEST_COUNT 应为 Counter, 实际 {type(REQUEST_COUNT)}"

    def test_request_latency_is_histogram(self):
        assert isinstance(REQUEST_LATENCY, Histogram), f"REQUEST_LATENCY 应为 Histogram, 实际 {type(REQUEST_LATENCY)}"

    def test_queue_size_is_gauge(self):
        assert isinstance(QUEUE_SIZE, Gauge), f"QUEUE_SIZE 应为 Gauge, 实际 {type(QUEUE_SIZE)}"

    def test_pending_requests_is_gauge(self):
        assert isinstance(PENDING_REQUESTS, Gauge), f"PENDING_REQUESTS 应为 Gauge, 实际 {type(PENDING_REQUESTS)}"

    def test_node_count_is_gauge(self):
        assert isinstance(NODE_COUNT, Gauge), f"NODE_COUNT 应为 Gauge, 实际 {type(NODE_COUNT)}"

    def test_gpu_utilization_is_gauge(self):
        assert isinstance(GPU_UTILIZATION, Gauge), f"GPU_UTILIZATION 应为 Gauge, 实际 {type(GPU_UTILIZATION)}"

    def test_gpu_memory_is_gauge(self):
        assert isinstance(GPU_MEMORY, Gauge), f"GPU_MEMORY 应为 Gauge, 实际 {type(GPU_MEMORY)}"

    def test_model_loaded_is_gauge(self):
        assert isinstance(MODEL_LOADED, Gauge), f"MODEL_LOADED 应为 Gauge, 实际 {type(MODEL_LOADED)}"

    def test_active_inferences_is_gauge(self):
        assert isinstance(ACTIVE_INFERENCES, Gauge), f"ACTIVE_INFERENCES 应为 Gauge, 实际 {type(ACTIVE_INFERENCES)}"

    def test_system_info_is_info(self):
        assert isinstance(SYSTEM_INFO, Info), f"SYSTEM_INFO 应为 Info, 实际 {type(SYSTEM_INFO)}"


# ==================== 指标名称验证 ====================


class TestMetricNames:
    """指标名称格式验证"""

    def test_all_metric_names_prefixed_with_quantumflow(self):
        all_metrics = [
            REQUEST_COUNT,
            REQUEST_LATENCY,
            QUEUE_SIZE,
            PENDING_REQUESTS,
            NODE_COUNT,
            GPU_UTILIZATION,
            GPU_MEMORY,
            MODEL_LOADED,
            ACTIVE_INFERENCES,
        ]
        for metric in all_metrics:
            assert metric._name.startswith(
                "quantumflow_"
            ), f"{metric._name} 应使用 quantumflow_ 前缀"

    def test_request_count_name(self):
        # Prometheus may append _total suffix to Counter names at registration,
        # but _name reflects the base name. Check the base name exists.
        assert "quantumflow_requests" in REQUEST_COUNT._name

    def test_request_latency_name(self):
        assert REQUEST_LATENCY._name == "quantumflow_request_latency_seconds"

    def test_queue_size_name(self):
        assert QUEUE_SIZE._name == "quantumflow_queue_size"

    def test_pending_requests_name(self):
        assert PENDING_REQUESTS._name == "quantumflow_pending_requests"

    def test_node_count_name(self):
        assert NODE_COUNT._name == "quantumflow_nodes_total"

    def test_gpu_utilization_name(self):
        assert GPU_UTILIZATION._name == "quantumflow_gpu_utilization"

    def test_gpu_memory_name(self):
        assert GPU_MEMORY._name == "quantumflow_gpu_memory_bytes"

    def test_model_loaded_name(self):
        assert MODEL_LOADED._name == "quantumflow_model_loaded"

    def test_active_inferences_name(self):
        assert ACTIVE_INFERENCES._name == "quantumflow_active_inferences"


# ==================== 标签验证 ====================


class TestMetricLabels:
    """指标标签验证"""

    def test_request_count_labels(self):
        labels = list(REQUEST_COUNT._labelnames)
        expected = ["node_id", "model", "status"]
        # Counter's _labelnames may include empty str; filter
        labels = [l for l in labels if l]
        for label in expected:
            assert label in labels, f"REQUEST_COUNT 缺少 label: {label}"

    def test_request_latency_labels(self):
        labels = [l for l in REQUEST_LATENCY._labelnames if l]
        expected = ["node_id", "model"]
        for label in expected:
            assert label in labels, f"REQUEST_LATENCY 缺少 label: {label}"

    def test_pending_requests_labels(self):
        labels = [l for l in PENDING_REQUESTS._labelnames if l]
        assert "node_id" in labels, "PENDING_REQUESTS 缺少 label: node_id"

    def test_node_count_labels(self):
        labels = [l for l in NODE_COUNT._labelnames if l]
        expected = ["node_id", "status"]
        for label in expected:
            assert label in labels, f"NODE_COUNT 缺少 label: {label}"

    def test_gpu_utilization_labels(self):
        labels = [l for l in GPU_UTILIZATION._labelnames if l]
        expected = ["node_id", "gpu_id"]
        for label in expected:
            assert label in labels, f"GPU_UTILIZATION 缺少 label: {label}"

    def test_gpu_memory_labels(self):
        labels = [l for l in GPU_MEMORY._labelnames if l]
        expected = ["node_id", "gpu_id"]
        for label in expected:
            assert label in labels, f"GPU_MEMORY 缺少 label: {label}"

    def test_model_loaded_labels(self):
        labels = [l for l in MODEL_LOADED._labelnames if l]
        expected = ["node_id", "model"]
        for label in expected:
            assert label in labels, f"MODEL_LOADED 缺少 label: {label}"

    def test_active_inferences_labels(self):
        labels = [l for l in ACTIVE_INFERENCES._labelnames if l]
        expected = ["node_id", "model"]
        for label in expected:
            assert label in labels, f"ACTIVE_INFERENCES 缺少 label: {label}"

    def test_queue_size_no_labels(self):
        """QUEUE_SIZE 应无标签"""
        labels = [l for l in QUEUE_SIZE._labelnames if l]
        assert len(labels) == 0, f"QUEUE_SIZE 不应有标签, 实际: {labels}"


# ==================== 指标操作验证 ====================


class TestMetricOperations:
    """指标基本操作验证"""

    def test_request_count_can_increment(self):
        """Counter 可以执行 inc 操作（验证无异常）"""
        REQUEST_COUNT.labels(node_id="test-node", model="test-model", status="success").inc()

    def test_request_latency_can_observe(self):
        """Histogram 可以执行 observe 操作"""
        REQUEST_LATENCY.labels(node_id="node-1", model="model-a").observe(0.15)
        REQUEST_LATENCY.labels(node_id="node-1", model="model-a").observe(2.5)
        # No exception = pass

    def test_queue_size_can_set(self):
        """Gauge 可以执行 set 操作"""
        QUEUE_SIZE.set(42)
        QUEUE_SIZE.set(0)
        # No exception = pass

    def test_pending_requests_with_labels(self):
        """带标签的 Gauge 可以设置"""
        PENDING_REQUESTS.labels(node_id="n1").set(5)
        PENDING_REQUESTS.labels(node_id="n2").set(3)

    def test_gpu_utilization_can_set(self):
        """GPU utilization gauge 可以设置"""
        GPU_UTILIZATION.labels(node_id="n1", gpu_id="0").set(85.5)
        GPU_UTILIZATION.labels(node_id="n1", gpu_id="1").set(92.0)

    def test_gpu_memory_can_set(self):
        """GPU memory gauge 可以设置"""
        GPU_MEMORY.labels(node_id="n1", gpu_id="0").set(8_000_000_000)
        GPU_MEMORY.labels(node_id="n1", gpu_id="1").set(12_000_000_000)

    def test_model_loaded_with_boolean_values(self):
        """MODEL_LOADED 可以设置为 0 或 1"""
        MODEL_LOADED.labels(node_id="n1", model="Qwen2.5-7B").set(1)
        MODEL_LOADED.labels(node_id="n1", model="Qwen2.5-72B").set(0)

    def test_active_inferences_can_increment_and_decrement(self):
        """ACTIVE_INFERENCES Gauge 可以 inc 和 dec"""
        ACTIVE_INFERENCES.labels(node_id="n1", model="m1").inc()
        ACTIVE_INFERENCES.labels(node_id="n1", model="m1").inc()
        ACTIVE_INFERENCES.labels(node_id="n1", model="m1").dec()

    def test_node_count_with_status_labels(self):
        """NODE_COUNT 可以使用不同 status 标签"""
        NODE_COUNT.labels(node_id="n1", status="active").set(1)
        NODE_COUNT.labels(node_id="n2", status="idle").set(1)
        NODE_COUNT.labels(node_id="n3", status="offline").set(0)


# ==================== Histogram buckets 验证 ====================


class TestHistogramBuckets:
    """Histogram buckets 配置验证"""

    def test_request_latency_has_buckets(self):
        """REQUEST_LATENCY 是 Histogram 类型且有 _upper_bounds"""
        assert isinstance(REQUEST_LATENCY, Histogram)
        # Histogram buckets accessible via collect()
        samples = list(REQUEST_LATENCY.collect())
        assert len(samples) > 0, "Histogram 应产生 metric samples"

    def test_bucket_value_observe_in_range(self):
        """验证 observe 不同范围内的值不引发异常"""
        for val in [0.001, 0.01, 0.05, 0.1, 1.0, 5.0, 15.0]:
            REQUEST_LATENCY.labels(node_id="test", model="test").observe(val)


# ==================== SYSTEM_INFO 验证 ====================


class TestSystemInfo:
    """SYSTEM_INFO 初始化验证"""

    def test_system_info_is_info_type(self):
        """SYSTEM_INFO 是 Info 类型"""
        assert isinstance(SYSTEM_INFO, Info)

    def test_system_info_collects_samples(self):
        """system info 指标可以 collect"""
        samples = list(SYSTEM_INFO.collect())
        assert len(samples) > 0, "SYSTEM_INFO 应有 metric samples"


# ==================== 指标文档字符串验证 ====================


class TestMetricDocstrings:
    """指标文档字符串验证"""

    def test_all_metrics_have_documentation(self):
        all_metrics = [
            REQUEST_COUNT,
            REQUEST_LATENCY,
            QUEUE_SIZE,
            PENDING_REQUESTS,
            NODE_COUNT,
            GPU_UTILIZATION,
            GPU_MEMORY,
            MODEL_LOADED,
            ACTIVE_INFERENCES,
        ]
        for metric in all_metrics:
            doc = metric._documentation
            assert doc, f"{metric._name} 应有文档字符串"
            assert len(doc) > 0, f"{metric._name} 的文档字符串不应为空"
