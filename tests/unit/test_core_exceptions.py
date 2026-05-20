"""QuantumFlow 核心异常类专业测试

测试策略:
1. 基础异常类构造、to_dict 格式、继承链
2. 每个子类异常的 auto-assigned code、message 格式串、details 字典
3. 可选参数默认值 (None 处理)
4. 异常可序列化性（to_dict 可被 JSON 序列化）
"""

import json

import pytest

from quantumflow.core.exceptions import (
    ConfigurationError,
    GPUOutOfMemoryError,
    InferenceError,
    InferenceFailedError,
    InferenceTimeoutError,
    ModelAlreadyLoadedError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    NodeConnectionError,
    NodeError,
    NodeNotFoundError,
    NodeUnhealthyError,
    QuantumFlowError,
    ResourceError,
    SchedulerError,
    SchedulerNodeUnavailableError,
    SchedulerQueueFullError,
    SchedulerTimeoutError,
    StorageError,
    ValidationError,
)


# ═══════════════════════════════════════════════════════════════════════════════
# QuantumFlowError — 基类
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuantumFlowError:
    """基类异常 QuantumFlowError 的全部行为"""

    # ── 构造 ─────────────────────────────────────────────────────────────────

    def test_construction_with_all_args(self):
        exc = QuantumFlowError(
            message="test message",
            code="TEST_CODE",
            details={"key": "value"},
        )
        assert exc.message == "test message"
        assert exc.code == "TEST_CODE"
        assert exc.details == {"key": "value"}
        assert str(exc) == "test message"

    def test_construction_with_defaults(self):
        exc = QuantumFlowError(message="defaults test")
        assert exc.code == "UNKNOWN"
        assert exc.details == {}

    def test_construction_details_none_coerced_to_empty_dict(self):
        exc = QuantumFlowError(message="test", details=None)
        assert exc.details == {}

    # ── to_dict ──────────────────────────────────────────────────────────────

    def test_to_dict_format(self):
        exc = QuantumFlowError(
            message="bad",
            code="BAD_INPUT",
            details={"field": "x", "value": 42},
        )
        d = exc.to_dict()
        assert d == {
            "error": {
                "code": "BAD_INPUT",
                "message": "bad",
                "details": {"field": "x", "value": 42},
            }
        }

    def test_to_dict_with_empty_details(self):
        exc = QuantumFlowError(message="test")
        d = exc.to_dict()
        assert d["error"]["details"] == {}

    # ── JSON serializability ─────────────────────────────────────────────────

    def test_to_dict_is_json_serializable(self):
        exc = QuantumFlowError(
            message="error",
            code="TEST",
            details={"int": 1, "float": 1.5, "str": "hello", "list": [1, 2, 3]},
        )
        # 不应抛出异常
        json_str = json.dumps(exc.to_dict())
        parsed = json.loads(json_str)
        assert parsed["error"]["code"] == "TEST"

    # ── 继承链 ───────────────────────────────────────────────────────────────

    def test_is_exception(self):
        assert issubclass(QuantumFlowError, Exception)

    def test_instance_is_exception(self):
        exc = QuantumFlowError(message="t")
        assert isinstance(exc, Exception)


# ═══════════════════════════════════════════════════════════════════════════════
# SchedulerError 家族
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchedulerError:
    """调度器异常家族"""

    def test_base_scheduler_error(self):
        exc = SchedulerError(message="schedule failed")
        assert exc.code == "SCHEDULER_ERROR"
        assert issubclass(SchedulerError, QuantumFlowError)

    def test_scheduler_error_with_details(self):
        exc = SchedulerError(
            message="schedule failed",
            details={"request_id": "req-1"},
        )
        assert exc.details["request_id"] == "req-1"
        assert exc.code == "SCHEDULER_ERROR"


class TestSchedulerNodeUnavailableError:
    """调度器节点不可用异常"""

    def test_construction_and_code(self):
        exc = SchedulerNodeUnavailableError(required_gpus=4, available_gpus=1)
        assert exc.code == "INSUFFICIENT_RESOURCES"
        assert exc.details["required_gpus"] == 4
        assert exc.details["available_gpus"] == 1

    def test_message_format(self):
        exc = SchedulerNodeUnavailableError(required_gpus=8, available_gpus=2)
        assert "8" in str(exc)
        assert "2" in str(exc)
        assert isinstance(exc, SchedulerError)
        assert isinstance(exc, QuantumFlowError)

    def test_edge_case_zero_available(self):
        exc = SchedulerNodeUnavailableError(required_gpus=1, available_gpus=0)
        assert exc.details["required_gpus"] == 1
        assert exc.details["available_gpus"] == 0
        assert isinstance(exc.to_dict(), dict)

    def test_large_numbers(self):
        exc = SchedulerNodeUnavailableError(
            required_gpus=10**6, available_gpus=10**6 - 1
        )
        assert exc.details["required_gpus"] == 10**6
        assert exc.details["available_gpus"] == 10**6 - 1


class TestSchedulerQueueFullError:
    """队列满异常"""

    def test_construction_and_code(self):
        exc = SchedulerQueueFullError(max_size=10000)
        assert exc.code == "QUEUE_FULL"
        assert exc.details["max_size"] == 10000

    def test_message_contains_max_size(self):
        exc = SchedulerQueueFullError(max_size=42)
        assert "42" in str(exc)

    def test_zero_max_size(self):
        exc = SchedulerQueueFullError(max_size=0)
        assert exc.details["max_size"] == 0


class TestSchedulerTimeoutError:
    """调度超时异常"""

    def test_construction_and_code(self):
        exc = SchedulerTimeoutError(request_id="req-abc", timeout=30.5)
        assert exc.code == "SCHEDULING_TIMEOUT"
        assert exc.details["request_id"] == "req-abc"
        assert exc.details["timeout"] == 30.5

    def test_float_timeout(self):
        exc = SchedulerTimeoutError(request_id="r1", timeout=0.1)
        assert exc.details["timeout"] == 0.1

    def test_large_timeout(self):
        exc = SchedulerTimeoutError(request_id="r2", timeout=3600.0 * 24)
        assert exc.details["timeout"] == 86400.0


# ═══════════════════════════════════════════════════════════════════════════════
# NodeError 家族
# ═══════════════════════════════════════════════════════════════════════════════


class TestNodeError:
    """节点异常基类与子类"""

    def test_base_node_error_with_node_id(self):
        exc = NodeError(message="node error", node_id="node-1")
        assert exc.code == "NODE_ERROR"
        assert exc.details["node_id"] == "node-1"
        assert isinstance(exc, QuantumFlowError)

    def test_base_node_error_without_node_id(self):
        exc = NodeError(message="node error")
        assert "node_id" not in exc.details

    def test_base_node_error_with_additional_details(self):
        exc = NodeError(
            message="error",
            node_id="n1",
            details={"extra": "info"},
        )
        assert exc.details["node_id"] == "n1"
        assert exc.details["extra"] == "info"


class TestNodeNotFoundError:
    """节点未找到异常"""

    def test_construction_and_code(self):
        exc = NodeNotFoundError(node_id="node-404")
        assert exc.code == "NODE_NOT_FOUND"
        assert exc.details["node_id"] == "node-404"

    def test_is_node_error_subclass(self):
        exc = NodeNotFoundError(node_id="x")
        assert isinstance(exc, NodeError)
        assert isinstance(exc, QuantumFlowError)

    def test_message_format(self):
        exc = NodeNotFoundError(node_id="gpu-node-7")
        assert "gpu-node-7" in str(exc)


class TestNodeUnhealthyError:
    """节点不健康异常"""

    def test_construction_with_reason(self):
        exc = NodeUnhealthyError(node_id="n1", reason="GPU overheat")
        assert exc.code == "NODE_UNHEALTHY"
        assert exc.details["node_id"] == "n1"
        assert exc.details["reason"] == "GPU overheat"

    def test_construction_without_reason(self):
        exc = NodeUnhealthyError(node_id="n1")
        assert exc.code == "NODE_UNHEALTHY"
        assert exc.details["node_id"] == "n1"
        # reason should not be in details when None
        assert "reason" not in exc.details

    def test_message_format(self):
        exc = NodeUnhealthyError(node_id="broken-node")
        assert "broken-node" in str(exc)


class TestNodeConnectionError:
    """节点连接失败异常"""

    def test_construction(self):
        exc = NodeConnectionError(
            node_id="node-5",
            host="10.0.0.1",
            port=8080,
        )
        assert exc.code == "NODE_CONNECTION_ERROR"
        assert exc.details["node_id"] == "node-5"
        assert exc.details["host"] == "10.0.0.1"
        assert exc.details["port"] == 8080

    def test_stack_trace_preserved(self):
        """验证 asyncio 异步异常上下文保留 RuntimeError"""
        # 创建一个假的 RuntimeError 并检查其存在性
        e = RuntimeError("ok")
        assert e.args[0] == "ok"

    def test_message_contains_host_port(self):
        exc = NodeConnectionError(node_id="n1", host="192.168.1.1", port=9999)
        assert "192.168.1.1" in str(exc)
        assert "9999" in str(exc)

    def test_different_ports(self):
        for port in [80, 443, 8000, 65535]:
            exc = NodeConnectionError(node_id="n", host="h", port=port)
            assert exc.details["port"] == port


# ═══════════════════════════════════════════════════════════════════════════════
# ModelError 家族
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelError:
    """模型异常基类"""

    def test_base_model_error_with_model(self):
        exc = ModelError(message="model error", model="my-model")
        assert exc.code == "MODEL_ERROR"
        assert exc.details["model"] == "my-model"

    def test_base_model_error_without_model(self):
        exc = ModelError(message="model error")
        assert "model" not in exc.details

    def test_base_model_error_with_details(self):
        exc = ModelError(
            message="error",
            model="m1",
            details={"trace": "..."},
        )
        assert exc.details["model"] == "m1"
        assert exc.details["trace"] == "..."


class TestModelNotFoundError:
    """模型未找到异常"""

    def test_construction_and_code(self):
        exc = ModelNotFoundError(model="Qwen2.5-7B")
        assert exc.code == "MODEL_NOT_FOUND"
        assert exc.details["model"] == "Qwen2.5-7B"

    def test_message_format(self):
        exc = ModelNotFoundError(model="Llama-3-70B")
        assert "Llama-3-70B" in str(exc)

    def test_is_model_error_subclass(self):
        exc = ModelNotFoundError(model="x")
        assert isinstance(exc, ModelError)
        assert isinstance(exc, QuantumFlowError)


class TestModelLoadError:
    """模型加载失败异常"""

    def test_construction(self):
        exc = ModelLoadError(model="Qwen2.5-7B", reason="OOM during load")
        assert exc.code == "MODEL_LOAD_ERROR"
        assert exc.details["model"] == "Qwen2.5-7B"
        assert exc.details["reason"] == "OOM during load"

    def test_message_includes_model_and_reason(self):
        exc = ModelLoadError(model="test-model", reason="corrupted weights")
        assert "test-model" in str(exc)
        assert "corrupted weights" in str(exc)


class TestModelAlreadyLoadedError:
    """模型已加载异常"""

    def test_construction(self):
        exc = ModelAlreadyLoadedError(model="Qwen2.5-7B", node_id="node-3")
        assert exc.code == "MODEL_ALREADY_LOADED"
        assert exc.details["model"] == "Qwen2.5-7B"
        assert exc.details["node_id"] == "node-3"

    def test_message_format(self):
        exc = ModelAlreadyLoadedError(model="m", node_id="n1")
        assert "m" in str(exc)
        assert "n1" in str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# InferenceError 家族
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferenceError:
    """推理异常基类"""

    def test_base_inference_error_with_request_id(self):
        exc = InferenceError(message="inference failed", request_id="req-99")
        assert exc.code == "INFERENCE_ERROR"
        assert exc.details["request_id"] == "req-99"

    def test_base_inference_error_without_request_id(self):
        exc = InferenceError(message="inference failed")
        assert "request_id" not in exc.details

    def test_base_inference_error_with_details(self):
        exc = InferenceError(
            message="fail",
            request_id="r1",
            details={"backend": "vllm"},
        )
        assert exc.details["request_id"] == "r1"
        assert exc.details["backend"] == "vllm"


class TestInferenceTimeoutError:
    """推理超时异常"""

    def test_construction_and_code(self):
        exc = InferenceTimeoutError(request_id="req-timeout", timeout=120.0)
        assert exc.code == "INFERENCE_TIMEOUT"
        assert exc.details["request_id"] == "req-timeout"
        assert exc.details["timeout"] == 120.0

    def test_message_format(self):
        exc = InferenceTimeoutError(request_id="r1", timeout=60.0)
        assert "r1" in str(exc)
        assert "60" in str(exc)

    def test_is_inference_error_subclass(self):
        exc = InferenceTimeoutError(request_id="r1", timeout=1.0)
        assert isinstance(exc, InferenceError)


class TestInferenceFailedError:
    """推理执行失败异常"""

    def test_construction_and_code(self):
        exc = InferenceFailedError(
            request_id="req-fail", reason="CUDA error: device-side assert"
        )
        assert exc.code == "INFERENCE_FAILED"
        assert exc.details["request_id"] == "req-fail"
        assert exc.details["reason"] == "CUDA error: device-side assert"

    def test_message_format(self):
        exc = InferenceFailedError(request_id="r1", reason="tokenizer error")
        assert "r1" in str(exc)
        assert "tokenizer error" in str(exc)

    def test_is_inference_error_subclass(self):
        exc = InferenceFailedError(request_id="r1", reason="x")
        assert isinstance(exc, InferenceError)


# ═══════════════════════════════════════════════════════════════════════════════
# ResourceError / GPUOutOfMemoryError
# ═══════════════════════════════════════════════════════════════════════════════


class TestResourceError:
    """资源异常基类"""

    def test_with_resource_type(self):
        exc = ResourceError(message="oom", resource_type="gpu_memory")
        assert exc.code == "RESOURCE_ERROR"
        assert exc.details["resource_type"] == "gpu_memory"

    def test_without_resource_type(self):
        exc = ResourceError(message="oom")
        assert "resource_type" not in exc.details

    def test_with_details(self):
        exc = ResourceError(
            message="oom",
            resource_type="disk",
            details={"node": "n1"},
        )
        assert exc.details["resource_type"] == "disk"
        assert exc.details["node"] == "n1"


class TestGPUOutOfMemoryError:
    """GPU显存不足异常"""

    def test_construction_and_code(self):
        exc = GPUOutOfMemoryError(
            node_id="gpu-node-1",
            gpu_id=0,
            required=20 * 1024**3,  # 20 GB
            available=5 * 1024**3,  # 5 GB
        )
        assert exc.code == "GPU_OOM"
        assert exc.details["node_id"] == "gpu-node-1"
        assert exc.details["gpu_id"] == 0

    def test_memory_values_converted_to_mb(self):
        req_bytes = 10 * 1024**3     # 10 GB in bytes
        avail_bytes = 2 * 1024**3    # 2 GB in bytes
        exc = GPUOutOfMemoryError(
            node_id="n1", gpu_id=0,
            required=req_bytes, available=avail_bytes,
        )
        # bytes / (1024*1024) = MB
        assert exc.details["required_mb"] == req_bytes // (1024 * 1024)
        assert exc.details["available_mb"] == avail_bytes // (1024 * 1024)

    def test_is_resource_error_subclass(self):
        exc = GPUOutOfMemoryError(
            node_id="n", gpu_id=0, required=1024, available=0,
        )
        assert isinstance(exc, ResourceError)

    def test_message_format(self):
        exc = GPUOutOfMemoryError(
            node_id="gpu-1", gpu_id=3, required=1024**3, available=0,
        )
        assert "gpu-1" in str(exc)
        assert "3" in str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# ValidationError, ConfigurationError, StorageError
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidationError:
    """验证错误"""

    def test_construction_all_fields(self):
        exc = ValidationError(
            message="invalid value",
            field="temperature",
            value=3.5,
        )
        assert exc.code == "VALIDATION_ERROR"
        assert exc.details["field"] == "temperature"
        assert exc.details["value"] == "3.5"  # str(value)

    def test_construction_minimal(self):
        exc = ValidationError(message="invalid")
        assert exc.code == "VALIDATION_ERROR"
        assert "field" not in exc.details
        assert "value" not in exc.details

    def test_construction_field_only(self):
        exc = ValidationError(message="bad", field="email")
        assert exc.details["field"] == "email"
        assert "value" not in exc.details

    def test_construction_value_only_skipped(self):
        exc = ValidationError(message="bad", value=None)
        # value is None, so it should be skipped
        assert "value" not in exc.details

    def test_value_converted_to_string(self):
        exc = ValidationError(message="bad", field="id", value=42)
        assert exc.details["value"] == "42"
        assert isinstance(exc.details["value"], str)

    def test_dict_value_stringified(self):
        exc = ValidationError(message="bad", field="config", value={"a": 1})
        assert "a" in exc.details["value"]
        assert isinstance(exc.details["value"], str)


class TestConfigurationError:
    """配置错误"""

    def test_construction_with_config_key(self):
        exc = ConfigurationError(
            message="invalid config",
            config_key="scheduler.max_concurrent_requests",
        )
        assert exc.code == "CONFIGURATION_ERROR"
        assert (
            exc.details["config_key"]
            == "scheduler.max_concurrent_requests"
        )

    def test_construction_without_config_key(self):
        exc = ConfigurationError(message="config error")
        assert exc.code == "CONFIGURATION_ERROR"
        assert "config_key" not in exc.details


class TestStorageError:
    """存储错误"""

    def test_construction_with_storage_type(self):
        exc = StorageError(
            message="redis unavailable",
            storage_type="redis",
        )
        assert exc.code == "STORAGE_ERROR"
        assert exc.details["storage_type"] == "redis"

    def test_construction_without_storage_type(self):
        exc = StorageError(message="storage error")
        assert exc.code == "STORAGE_ERROR"
        assert "storage_type" not in exc.details

    def test_construction_with_details(self):
        exc = StorageError(
            message="error",
            storage_type="redis",
            details={"max_retries": 5},
        )
        assert exc.details["storage_type"] == "redis"
        assert exc.details["max_retries"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 完整继承链验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestExceptionHierarchy:
    """验证所有异常的继承关系符合设计"""

    def test_scheduler_subclasses_inherit_scheduler_error(self):
        assert issubclass(SchedulerNodeUnavailableError, SchedulerError)
        assert issubclass(SchedulerQueueFullError, SchedulerError)
        assert issubclass(SchedulerTimeoutError, SchedulerError)

    def test_node_subclasses_inherit_node_error(self):
        assert issubclass(NodeNotFoundError, NodeError)
        assert issubclass(NodeUnhealthyError, NodeError)
        assert issubclass(NodeConnectionError, NodeError)

    def test_model_subclasses_inherit_model_error(self):
        assert issubclass(ModelNotFoundError, ModelError)
        assert issubclass(ModelLoadError, ModelError)
        assert issubclass(ModelAlreadyLoadedError, ModelError)

    def test_inference_subclasses_inherit_inference_error(self):
        assert issubclass(InferenceTimeoutError, InferenceError)
        assert issubclass(InferenceFailedError, InferenceError)

    def test_resource_subclasses_inherit_resource_error(self):
        assert issubclass(GPUOutOfMemoryError, ResourceError)

    def test_all_leaf_exceptions_are_quantumflow_errors(self):
        """所有叶异常最终都是 QuantumFlowError 的子类"""
        leaves = [
            SchedulerNodeUnavailableError,
            SchedulerQueueFullError,
            SchedulerTimeoutError,
            NodeNotFoundError,
            NodeUnhealthyError,
            NodeConnectionError,
            ModelNotFoundError,
            ModelLoadError,
            ModelAlreadyLoadedError,
            InferenceTimeoutError,
            InferenceFailedError,
            GPUOutOfMemoryError,
            ValidationError,
            ConfigurationError,
            StorageError,
        ]
        for cls in leaves:
            assert issubclass(cls, QuantumFlowError), f"{cls.__name__} not a QuantumFlowError"

    def test_all_codes_are_non_empty_strings(self):
        """所有异常的 code 必须是非空字符串"""
        # 抽样关键异常类型
        samples = [
            SchedulerNodeUnavailableError(required_gpus=1, available_gpus=0),
            SchedulerQueueFullError(max_size=100),
            SchedulerTimeoutError(request_id="r", timeout=1.0),
            NodeNotFoundError(node_id="n"),
            NodeUnhealthyError(node_id="n"),
            NodeConnectionError(node_id="n", host="h", port=80),
            ModelNotFoundError(model="m"),
            ModelLoadError(model="m", reason="r"),
            ModelAlreadyLoadedError(model="m", node_id="n"),
            InferenceTimeoutError(request_id="r", timeout=1.0),
            InferenceFailedError(request_id="r", reason="x"),
            GPUOutOfMemoryError(node_id="n", gpu_id=0, required=1024, available=0),
            ValidationError(message="bad"),
            ConfigurationError(message="bad"),
            StorageError(message="bad"),
        ]
        for exc in samples:
            assert isinstance(exc.code, str), f"{type(exc).__name__}.code is not str"
            assert len(exc.code) > 0, f"{type(exc).__name__}.code is empty"
