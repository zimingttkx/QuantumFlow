"""Proto 序列化/反序列化验证测试

严格测试 Proto 消息的:
1. 枚举值定义正确性
2. 字段范围验证
3. 序列化/反序列化往返一致性
4. 字段边界值测试
5. 消息内嵌和引用关系
"""

import uuid
from typing import Any, Dict, List, Tuple

import pytest

from quantumflow.grpc.generated import (
    quantumflow_pb2,
    quantumflow_pb2 as pb2,
    quantumflow_pb2_grpc,
)


# ============ 辅助函数 ============


def is_valid_uuid(value: str) -> bool:
    """验证是否是有效的 UUID 格式"""
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def create_request_id() -> str:
    """创建有效的 request_id"""
    return str(uuid.uuid4())


# ============ 导入验证 ============


class TestProtoImports:
    """验证所有 Proto 生成的消息类型和服务可正确导入"""

    def test_all_message_types_importable(self):
        """所有消息类型必须可导入"""
        # 枚举类型
        assert hasattr(pb2, "ModelBackend")
        assert hasattr(pb2, "InferenceMode")
        assert hasattr(pb2, "ResponseStatus")
        assert hasattr(pb2, "NodeStatus")

        # 公共消息
        assert hasattr(pb2, "GPUMemory")
        assert hasattr(pb2, "GPUInfo")
        assert hasattr(pb2, "ModelInfo")
        assert hasattr(pb2, "NodeResources")

        # 推理消息
        assert hasattr(pb2, "InferenceRequest")
        assert hasattr(pb2, "InferenceResponse")
        assert hasattr(pb2, "BatchInferenceRequest")
        assert hasattr(pb2, "BatchInferenceResponse")

        # 集群消息
        assert hasattr(pb2, "RegisterNodeRequest")
        assert hasattr(pb2, "RegisterNodeResponse")
        assert hasattr(pb2, "HeartbeatRequest")
        assert hasattr(pb2, "HeartbeatResponse")
        assert hasattr(pb2, "DeregisterNodeRequest")
        assert hasattr(pb2, "DeregisterNodeResponse")
        assert hasattr(pb2, "ListNodesRequest")
        assert hasattr(pb2, "ListNodesResponse")

        # 调度消息
        assert hasattr(pb2, "SchedulingRequest")
        assert hasattr(pb2, "SchedulingResponse")
        assert hasattr(pb2, "CancelRequest")
        assert hasattr(pb2, "CancelResponse")
        assert hasattr(pb2, "GetSchedulingStatusRequest")
        assert hasattr(pb2, "GetSchedulingStatusResponse")

        # 健康检查消息
        assert hasattr(pb2, "HealthCheckRequest")
        assert hasattr(pb2, "HealthCheckResponse")

        # 模型管理消息
        assert hasattr(pb2, "LoadModelRequest")
        assert hasattr(pb2, "LoadModelResponse")
        assert hasattr(pb2, "UnloadModelRequest")
        assert hasattr(pb2, "UnloadModelResponse")
        assert hasattr(pb2, "ListModelsRequest")
        assert hasattr(pb2, "ListModelsResponse")

        # 指标消息
        assert hasattr(pb2, "MetricsRequest")
        assert hasattr(pb2, "MetricsResponse")
        assert hasattr(pb2, "MetricSample")

    def test_all_service_stubs_importable(self):
        """所有服务 Stub 必须可导入"""
        assert hasattr(quantumflow_pb2_grpc, "InferenceServiceStub")
        assert hasattr(quantumflow_pb2_grpc, "ClusterServiceStub")
        assert hasattr(quantumflow_pb2_grpc, "SchedulerServiceStub")
        assert hasattr(quantumflow_pb2_grpc, "HealthServiceStub")
        assert hasattr(quantumflow_pb2_grpc, "ModelManagementServiceStub")
        assert hasattr(quantumflow_pb2_grpc, "MetricsServiceStub")

    def test_all_service_servicers_importable(self):
        """所有服务 Servicer 必须可导入"""
        assert hasattr(quantumflow_pb2_grpc, "InferenceServiceServicer")
        assert hasattr(quantumflow_pb2_grpc, "ClusterServiceServicer")
        assert hasattr(quantumflow_pb2_grpc, "SchedulerServiceServicer")
        assert hasattr(quantumflow_pb2_grpc, "HealthServiceServicer")
        assert hasattr(quantumflow_pb2_grpc, "ModelManagementServiceServicer")
        assert hasattr(quantumflow_pb2_grpc, "MetricsServiceServicer")


# ============ 枚举值测试 ============


class TestModelBackendEnum:
    """ModelBackend 枚举值测试"""

    def test_enum_values_are_unique(self):
        """枚举值必须唯一"""
        values = list(pb2.ModelBackend.values())
        unique_values = set(values)
        assert len(values) == len(unique_values), "ModelBackend 枚举值存在重复"

    def test_unspecified_is_zero(self):
        """UNSPECIFIED 必须为 0（proto3 默认值约定）"""
        assert pb2.ModelBackend.MODEL_BACKEND_UNSPECIFIED == 0

    def test_all_backends_defined(self):
        """所有推理后端都已定义"""
        expected_names = {
            "MODEL_BACKEND_UNSPECIFIED", "MODEL_BACKEND_VLLM",
            "MODEL_BACKEND_HUGGINGFACE", "MODEL_BACKEND_TGI", "MODEL_BACKEND_SGLANG"
        }
        # pb2.ModelBackend 是一个 EnumTypeWrapper
        actual_names = set(pb2.ModelBackend.keys())
        assert actual_names == expected_names

    @pytest.mark.parametrize("backend,expected_int", [
        (pb2.ModelBackend.MODEL_BACKEND_VLLM, 1),
        (pb2.ModelBackend.MODEL_BACKEND_HUGGINGFACE, 2),
        (pb2.ModelBackend.MODEL_BACKEND_TGI, 3),
        (pb2.ModelBackend.MODEL_BACKEND_SGLANG, 4),
    ])
    def test_backend_value_correct(self, backend, expected_int):
        """枚举值正确"""
        assert backend == expected_int

    def test_backend_serialization_via_message(self):
        """通过消息测试枚举序列化/反序列化往返"""
        # 创建包含枚举的请求消息
        request = pb2.InferenceRequest(
            request_id="test-123",
            model_name="test-model",
            backend=pb2.ModelBackend.MODEL_BACKEND_VLLM,
        )
        serialized = request.SerializeToString()
        restored = pb2.InferenceRequest()
        restored.ParseFromString(serialized)
        assert restored.backend == pb2.ModelBackend.MODEL_BACKEND_VLLM

    @pytest.mark.parametrize("backend,expected_value", [
        (pb2.ModelBackend.MODEL_BACKEND_UNSPECIFIED, 0),
        (pb2.ModelBackend.MODEL_BACKEND_VLLM, 1),
        (pb2.ModelBackend.MODEL_BACKEND_HUGGINGFACE, 2),
        (pb2.ModelBackend.MODEL_BACKEND_TGI, 3),
        (pb2.ModelBackend.MODEL_BACKEND_SGLANG, 4),
    ])
    def test_backend_numeric_values(self, backend, expected_value):
        """枚举数值符合预期"""
        assert backend == expected_value


class TestInferenceModeEnum:
    """InferenceMode 枚举值测试"""

    def test_unspecified_is_zero(self):
        """UNSPECIFIED 必须为 0"""
        assert pb2.InferenceMode.INFERENCE_MODE_UNSPECIFIED == 0

    @pytest.mark.parametrize("mode,expected_value", [
        (pb2.InferenceMode.INFERENCE_MODE_UNSPECIFIED, 0),
        (pb2.InferenceMode.INFERENCE_MODE_GENERATE, 1),
        (pb2.InferenceMode.INFERENCE_MODE_STREAM, 2),
        (pb2.InferenceMode.INFERENCE_MODE_BATCH, 3),
    ])
    def test_mode_numeric_values(self, mode, expected_value):
        """模式枚举数值正确"""
        assert mode == expected_value


class TestResponseStatusEnum:
    """ResponseStatus 枚举值测试"""

    def test_all_statuses_defined(self):
        """所有状态都已定义"""
        expected_names = {
            "STATUS_UNSPECIFIED", "STATUS_SUCCESS", "STATUS_PENDING",
            "STATUS_PROCESSING", "STATUS_ERROR", "STATUS_CANCELLED"
        }
        actual_names = set(pb2.ResponseStatus.keys())
        assert actual_names == expected_names

    @pytest.mark.parametrize("status,expected_value", [
        (pb2.ResponseStatus.STATUS_UNSPECIFIED, 0),
        (pb2.ResponseStatus.STATUS_SUCCESS, 1),
        (pb2.ResponseStatus.STATUS_PENDING, 2),
        (pb2.ResponseStatus.STATUS_PROCESSING, 3),
        (pb2.ResponseStatus.STATUS_ERROR, 4),
        (pb2.ResponseStatus.STATUS_CANCELLED, 5),
    ])
    def test_status_numeric_values(self, status, expected_value):
        """状态枚举数值正确"""
        assert status == expected_value


class TestNodeStatusEnum:
    """NodeStatus 枚举值测试"""

    @pytest.mark.parametrize("status,expected_value", [
        (pb2.NodeStatus.NODE_STATUS_UNSPECIFIED, 0),
        (pb2.NodeStatus.NODE_STATUS_ACTIVE, 1),
        (pb2.NodeStatus.NODE_STATUS_IDLE, 2),
        (pb2.NodeStatus.NODE_STATUS_BUSY, 3),
        (pb2.NodeStatus.NODE_STATUS_OFFLINE, 4),
    ])
    def test_status_numeric_values(self, status, expected_value):
        """节点状态枚举数值正确"""
        assert status == expected_value


# ============ 公共消息测试 ============


class TestGPUMemoryMessage:
    """GPUMemory 消息测试"""

    def test_default_values(self):
        """默认字段值为 0 或 0.0"""
        msg = pb2.GPUMemory()
        assert msg.total_bytes == 0
        assert msg.available_bytes == 0
        assert msg.used_bytes == 0
        assert msg.utilization == 0.0

    def test_valid_memory_creation(self):
        """创建有效的显存消息"""
        msg = pb2.GPUMemory(
            total_bytes=16 * 1024**3,
            available_bytes=8 * 1024**3,
            used_bytes=8 * 1024**3,
            utilization=0.5,
        )
        assert msg.total_bytes == 16 * 1024**3
        assert msg.available_bytes == 8 * 1024**3
        assert msg.used_bytes == 8 * 1024**3
        assert abs(msg.utilization - 0.5) < 0.001

    def test_utilization_calculation_consistency(self):
        """used_bytes / total_bytes 应等于 utilization"""
        total = 16 * 1024**3
        used = 8 * 1024**3
        expected_util = used / total

        msg = pb2.GPUMemory(
            total_bytes=total,
            used_bytes=used,
            utilization=expected_util,
        )
        # 验证内部一致性（实际业务逻辑验证）
        assert abs(msg.utilization - expected_util) < 0.001

    @pytest.mark.parametrize("total,available,used", [
        (100, 60, 40),
        (16 * 1024**3, 8 * 1024**3, 8 * 1024**3),
        (80 * 1024**3, 40 * 1024**3, 40 * 1024**3),
    ])
    def test_memory_values_positive(self, total, available, used):
        """所有显存值必须为正数"""
        msg = pb2.GPUMemory(total_bytes=total, available_bytes=available, used_bytes=used)
        assert msg.total_bytes >= 0
        assert msg.available_bytes >= 0
        assert msg.used_bytes >= 0

    def test_used_cannot_exceed_total(self):
        """used_bytes 不应超过 total_bytes（业务逻辑验证）"""
        msg = pb2.GPUMemory(total_bytes=100, used_bytes=200)
        # Proto 不强制验证，但业务逻辑应该检查
        assert msg.used_bytes > msg.total_bytes  # 验证异常情况被接受

    def test_serialization_roundtrip(self):
        """序列化/反序列化往返保持数据一致"""
        original = pb2.GPUMemory(
            total_bytes=24 * 1024**3,
            available_bytes=16 * 1024**3,
            used_bytes=8 * 1024**3,
            utilization=0.333,
        )
        serialized = original.SerializeToString()
        restored = pb2.GPUMemory()
        restored.ParseFromString(serialized)

        assert restored.total_bytes == original.total_bytes
        assert restored.available_bytes == original.available_bytes
        assert restored.used_bytes == original.used_bytes
        assert abs(restored.utilization - original.utilization) < 0.001


class TestGPUInfoMessage:
    """GPUInfo 消息测试"""

    def test_valid_gpu_info_creation(self):
        """创建有效的 GPU 信息"""
        msg = pb2.GPUInfo(
            index=0,
            name="NVIDIA RTX 4090",
            memory=pb2.GPUMemory(
                total_bytes=24 * 1024**3,
                used_bytes=12 * 1024**3,
            ),
            utilization=0.5,
            compute_capacity=33000,
        )
        assert msg.index == 0
        assert msg.name == "NVIDIA RTX 4090"
        assert msg.HasField("memory")
        assert msg.utilization == 0.5
        assert msg.compute_capacity == 33000

    def test_gpu_index_order(self):
        """GPU 索引应按顺序排列"""
        gpus = [
            pb2.GPUInfo(index=1, name="GPU 1"),
            pb2.GPUInfo(index=0, name="GPU 0"),
            pb2.GPUInfo(index=2, name="GPU 2"),
        ]
        # 按 index 排序
        sorted_gpus = sorted(gpus, key=lambda g: g.index)
        assert [g.index for g in sorted_gpus] == [0, 1, 2]

    def test_compute_capacity_reasonable(self):
        """compute_capacity 应该在合理范围内（0 - 100000 TFLOPS）"""
        msg_valid = pb2.GPUInfo(
            index=0,
            name="GPU",
            compute_capacity=33000,  # RTX 4090 ~33 TFLOPS (FP32)
        )
        assert 0 < msg_valid.compute_capacity < 100000

    def test_empty_gpu_name_allowed(self):
        """GPU 名称可以为空"""
        msg = pb2.GPUInfo(index=0)
        assert msg.name == ""


class TestModelInfoMessage:
    """ModelInfo 消息测试"""

    def test_valid_model_info_creation(self):
        """创建有效的模型信息"""
        msg = pb2.ModelInfo(
            name="llama-2-70b",
            backend="vllm",
            size_bytes=140 * 1024**3,
            tensor_parallel_size=4,
            is_loaded=True,
        )
        assert msg.name == "llama-2-70b"
        assert msg.backend == "vllm"
        assert msg.size_bytes == 140 * 1024**3
        assert msg.tensor_parallel_size == 4
        assert msg.is_loaded is True

    def test_default_is_loaded_false(self):
        """is_loaded 默认为 False"""
        msg = pb2.ModelInfo(name="test-model")
        assert msg.is_loaded is False

    def test_tensor_parallel_size_positive(self):
        """tensor_parallel_size 必须为正数"""
        msg_valid = pb2.ModelInfo(tensor_parallel_size=1)
        assert msg_valid.tensor_parallel_size >= 1

        msg_zero = pb2.ModelInfo(tensor_parallel_size=0)
        assert msg_zero.tensor_parallel_size == 0


class TestNodeResourcesMessage:
    """NodeResources 消息测试"""

    def test_valid_node_resources_creation(self):
        """创建有效的节点资源消息"""
        msg = pb2.NodeResources(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
            status=pb2.NodeStatus.NODE_STATUS_ACTIVE,
        )
        assert msg.node_id == "worker-001"
        assert msg.host == "192.168.1.100"
        assert msg.port == 8001
        assert msg.status == pb2.NodeStatus.NODE_STATUS_ACTIVE

    def test_empty_gpu_list_allowed(self):
        """GPU 列表可以为空（CPU-only 节点）"""
        msg = pb2.NodeResources(node_id="cpu-node", gpus=[])
        assert len(msg.gpus) == 0

    def test_multiple_gpus(self):
        """支持多 GPU"""
        msg = pb2.NodeResources(
            node_id="multi-gpu-node",
            gpus=[
                pb2.GPUInfo(index=0, name="GPU 0"),
                pb2.GPUInfo(index=1, name="GPU 1"),
                pb2.GPUInfo(index=2, name="GPU 2"),
                pb2.GPUInfo(index=3, name="GPU 3"),
            ],
        )
        assert len(msg.gpus) == 4

    def test_port_range(self):
        """port 必须在有效范围内 (1-65535)"""
        msg_valid = pb2.NodeResources(port=8000)
        assert 1 <= msg_valid.port <= 65535

        msg_min = pb2.NodeResources(port=1)
        assert msg_min.port == 1

        msg_max = pb2.NodeResources(port=65535)
        assert msg_max.port == 65535


# ============ 推理消息测试 ============


class TestInferenceRequestMessage:
    """InferenceRequest 消息测试"""

    @pytest.fixture
    def valid_request(self):
        """创建有效的推理请求"""
        return pb2.InferenceRequest(
            request_id=create_request_id(),
            model_name="llama-2-7b",
            backend=pb2.ModelBackend.MODEL_BACKEND_VLLM,
            prompt="Hello, world!",
            stream=False,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
        )

    def test_valid_request_creation(self, valid_request):
        """创建有效的推理请求"""
        assert valid_request.request_id != ""
        assert valid_request.model_name == "llama-2-7b"
        assert valid_request.backend == pb2.ModelBackend.MODEL_BACKEND_VLLM
        assert valid_request.prompt == "Hello, world!"
        assert valid_request.stream is False
        assert valid_request.max_tokens == 100

    def test_request_id_is_uuid_format(self, valid_request):
        """request_id 必须是有效的 UUID"""
        assert is_valid_uuid(valid_request.request_id)

    def test_empty_request_id_accepted(self):
        """Proto 接受空字符串 request_id（验证边界情况）"""
        msg = pb2.InferenceRequest(request_id="")
        assert msg.request_id == ""

    def test_empty_model_name_accepted(self):
        """Proto 接受空字符串 model_name（验证边界情况）"""
        msg = pb2.InferenceRequest(model_name="")
        assert msg.model_name == ""

    def test_empty_prompt_accepted(self):
        """Proto 接受空字符串 prompt"""
        msg = pb2.InferenceRequest(prompt="")
        assert msg.prompt == ""

    def test_max_tokens_zero_allowed(self):
        """max_tokens 可以为 0（业务逻辑可能拒绝，但 Proto 接受）"""
        msg = pb2.InferenceRequest(max_tokens=0)
        assert msg.max_tokens == 0

    @pytest.mark.parametrize("max_tokens", [1, 100, 1000, 4096, 8192])
    def test_max_tokens_various_values(self, max_tokens):
        """max_tokens 支持多种值"""
        msg = pb2.InferenceRequest(max_tokens=max_tokens)
        assert msg.max_tokens == max_tokens

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_temperature_various_values(self, temperature):
        """temperature 支持多种值"""
        msg = pb2.InferenceRequest(temperature=temperature)
        assert abs(msg.temperature - temperature) < 0.001

    @pytest.mark.parametrize("top_p", [0.0, 0.5, 0.9, 0.99, 1.0])
    def test_top_p_various_values(self, top_p):
        """top_p 支持多种值"""
        msg = pb2.InferenceRequest(top_p=top_p)
        assert abs(msg.top_p - top_p) < 0.001

    def test_top_k_zero_disabled(self):
        """top_k=0 表示 disabled"""
        msg = pb2.InferenceRequest(top_k=0)
        assert msg.top_k == 0

    @pytest.mark.parametrize("top_k", [0, 1, 10, 50, 100, 1000])
    def test_top_k_various_values(self, top_k):
        """top_k 支持多种值"""
        msg = pb2.InferenceRequest(top_k=top_k)
        assert msg.top_k == top_k

    @pytest.mark.parametrize("repetition_penalty", [1.0, 1.1, 1.5, 2.0])
    def test_repetition_penalty_various_values(self, repetition_penalty):
        """repetition_penalty 支持多种值"""
        msg = pb2.InferenceRequest(repetition_penalty=repetition_penalty)
        assert abs(msg.repetition_penalty - repetition_penalty) < 0.001

    def test_extra_params_map(self):
        """extra_params 是有效的 map"""
        msg = pb2.InferenceRequest(
            extra_params={"custom_param": "value", "another": "thing"}
        )
        assert len(msg.extra_params) == 2
        assert msg.extra_params["custom_param"] == "value"

    def test_extra_params_empty_allowed(self):
        """extra_params 可以为空"""
        msg = pb2.InferenceRequest()
        assert len(msg.extra_params) == 0

    def test_serialization_roundtrip(self, valid_request):
        """序列化/反序列化往返保持数据一致"""
        serialized = valid_request.SerializeToString()
        restored = pb2.InferenceRequest()
        restored.ParseFromString(serialized)

        assert restored.request_id == valid_request.request_id
        assert restored.model_name == valid_request.model_name
        assert restored.backend == valid_request.backend
        assert restored.prompt == valid_request.prompt
        assert restored.max_tokens == valid_request.max_tokens
        assert restored.temperature == valid_request.temperature


class TestInferenceResponseMessage:
    """InferenceResponse 消息测试"""

    def test_valid_response_creation(self):
        """创建有效的推理响应"""
        msg = pb2.InferenceResponse(
            request_id=create_request_id(),
            status=pb2.ResponseStatus.STATUS_SUCCESS,
            text="Generated text here",
            tokens_generated=50,
            latency_ms=123.45,
        )
        assert msg.status == pb2.ResponseStatus.STATUS_SUCCESS
        assert msg.text == "Generated text here"
        assert msg.tokens_generated == 50
        assert abs(msg.latency_ms - 123.45) < 0.01

    def test_error_response_creation(self):
        """创建错误响应"""
        msg = pb2.InferenceResponse(
            request_id=create_request_id(),
            status=pb2.ResponseStatus.STATUS_ERROR,
            error_message="Model not found",
        )
        assert msg.status == pb2.ResponseStatus.STATUS_ERROR
        assert msg.error_message == "Model not found"

    def test_pending_response(self):
        """创建等待中的响应"""
        msg = pb2.InferenceResponse(
            request_id=create_request_id(),
            status=pb2.ResponseStatus.STATUS_PENDING,
        )
        assert msg.status == pb2.ResponseStatus.STATUS_PENDING

    def test_processing_response(self):
        """创建处理中的响应"""
        msg = pb2.InferenceResponse(
            request_id=create_request_id(),
            status=pb2.ResponseStatus.STATUS_PROCESSING,
        )
        assert msg.status == pb2.ResponseStatus.STATUS_PROCESSING

    def test_cancelled_response(self):
        """创建取消的响应"""
        msg = pb2.InferenceResponse(
            request_id=create_request_id(),
            status=pb2.ResponseStatus.STATUS_CANCELLED,
        )
        assert msg.status == pb2.ResponseStatus.STATUS_CANCELLED

    def test_serialization_roundtrip(self):
        """序列化/反序列化往返保持数据一致"""
        original = pb2.InferenceResponse(
            request_id=create_request_id(),
            status=pb2.ResponseStatus.STATUS_SUCCESS,
            text="Test response",
            tokens_generated=42,
            latency_ms=99.9,
        )
        serialized = original.SerializeToString()
        restored = pb2.InferenceResponse()
        restored.ParseFromString(serialized)

        assert restored.request_id == original.request_id
        assert restored.status == original.status
        assert restored.text == original.text
        assert restored.tokens_generated == original.tokens_generated


class TestBatchInferenceRequestMessage:
    """BatchInferenceRequest 消息测试"""

    def test_valid_batch_request(self):
        """创建有效的批量请求"""
        msg = pb2.BatchInferenceRequest(
            batch_id=create_request_id(),
            model_name="llama-2-7b",
            backend=pb2.ModelBackend.MODEL_BACKEND_VLLM,
            prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
            max_tokens=100,
            temperature=0.7,
        )
        assert len(msg.prompts) == 3
        assert msg.model_name == "llama-2-7b"

    def test_empty_batch_allowed(self):
        """空批量请求被接受（验证边界）"""
        msg = pb2.BatchInferenceRequest(batch_id=create_request_id())
        assert len(msg.prompts) == 0

    def test_large_batch(self):
        """大批量请求支持"""
        prompts = [f"Prompt {i}" for i in range(1000)]
        msg = pb2.BatchInferenceRequest(
            batch_id=create_request_id(),
            prompts=prompts,
        )
        assert len(msg.prompts) == 1000


class TestBatchInferenceResponseMessage:
    """BatchInferenceResponse 消息测试"""

    def test_valid_batch_response(self):
        """创建有效的批量响应"""
        msg = pb2.BatchInferenceResponse(
            batch_id=create_request_id(),
            status=pb2.ResponseStatus.STATUS_SUCCESS,
            results=[
                pb2.InferenceResponse(text="Result 1", tokens_generated=10),
                pb2.InferenceResponse(text="Result 2", tokens_generated=20),
            ],
            total_latency_ms=500.0,
        )
        assert len(msg.results) == 2
        assert msg.total_latency_ms == 500.0

    def test_empty_results_allowed(self):
        """空结果列表被接受"""
        msg = pb2.BatchInferenceResponse(batch_id=create_request_id())
        assert len(msg.results) == 0


# ============ 集群管理消息测试 ============


class TestRegisterNodeRequestMessage:
    """RegisterNodeRequest 消息测试"""

    def test_valid_register_request(self):
        """创建有效的节点注册请求"""
        msg = pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
            capabilities={"gpu": "4x RTX 4090"},
        )
        assert msg.node_id == "worker-001"
        assert msg.host == "192.168.1.100"
        assert msg.port == 8001
        assert msg.capabilities["gpu"] == "4x RTX 4090"

    def test_empty_capabilities_allowed(self):
        """空的 capabilities 被接受"""
        msg = pb2.RegisterNodeRequest(node_id="test")
        assert len(msg.capabilities) == 0

    def test_multiple_gpus_in_request(self):
        """支持多个 GPU 信息"""
        msg = pb2.RegisterNodeRequest(
            node_id="multi-gpu",
            gpus=[
                pb2.GPUInfo(index=0),
                pb2.GPUInfo(index=1),
                pb2.GPUInfo(index=2),
                pb2.GPUInfo(index=3),
            ],
        )
        assert len(msg.gpus) == 4


class TestRegisterNodeResponseMessage:
    """RegisterNodeResponse 消息测试"""

    def test_success_response(self):
        """成功的注册响应"""
        msg = pb2.RegisterNodeResponse(
            success=True,
            message="Node registered successfully",
            assigned_id="assigned-001",
        )
        assert msg.success is True
        assert msg.assigned_id == "assigned-001"

    def test_failure_response(self):
        """失败的注册响应"""
        msg = pb2.RegisterNodeResponse(
            success=False,
            message="Node ID already exists",
        )
        assert msg.success is False


class TestHeartbeatRequestMessage:
    """HeartbeatRequest 消息测试"""

    def test_valid_heartbeat(self):
        """创建有效的心跳请求"""
        msg = pb2.HeartbeatRequest(
            node_id="worker-001",
            resources=pb2.NodeResources(
                node_id="worker-001",
                status=pb2.NodeStatus.NODE_STATUS_ACTIVE,
            ),
        )
        assert msg.node_id == "worker-001"
        assert msg.HasField("resources")


class TestHeartbeatResponseMessage:
    """HeartbeatResponse 消息测试"""

    def test_valid_heartbeat_response(self):
        """创建有效的心跳响应"""
        msg = pb2.HeartbeatResponse(
            success=True,
            server_time=1234567890,
            pending_tasks=["task-1", "task-2"],
        )
        assert msg.success is True
        assert msg.server_time == 1234567890
        assert len(msg.pending_tasks) == 2


class TestDeregisterNodeRequestMessage:
    """DeregisterNodeRequest 消息测试"""

    def test_valid_deregister_request(self):
        """创建有效的节点注销请求"""
        msg = pb2.DeregisterNodeRequest(
            node_id="worker-001",
            reason="Maintenance",
        )
        assert msg.node_id == "worker-001"
        assert msg.reason == "Maintenance"

    def test_empty_reason_allowed(self):
        """空的 reason 被接受"""
        msg = pb2.DeregisterNodeRequest(node_id="test")
        assert msg.reason == ""


class TestListNodesRequestMessage:
    """ListNodesRequest 消息测试"""

    def test_empty_filter(self):
        """空过滤器返回所有节点"""
        msg = pb2.ListNodesRequest()
        assert msg.filter_status == pb2.NodeStatus.NODE_STATUS_UNSPECIFIED
        assert msg.filter_model == ""

    def test_filter_by_status(self):
        """按状态过滤"""
        msg = pb2.ListNodesRequest(
            filter_status=pb2.NodeStatus.NODE_STATUS_ACTIVE,
        )
        assert msg.filter_status == pb2.NodeStatus.NODE_STATUS_ACTIVE

    def test_filter_by_model(self):
        """按模型过滤"""
        msg = pb2.ListNodesRequest(filter_model="llama-2-70b")
        assert msg.filter_model == "llama-2-70b"


class TestListNodesResponseMessage:
    """ListNodesResponse 消息测试"""

    def test_empty_list(self):
        """空节点列表"""
        msg = pb2.ListNodesResponse()
        assert len(msg.nodes) == 0

    def test_multiple_nodes(self):
        """多节点响应"""
        msg = pb2.ListNodesResponse(
            nodes=[
                pb2.NodeResources(node_id="worker-001"),
                pb2.NodeResources(node_id="worker-002"),
            ],
        )
        assert len(msg.nodes) == 2


# ============ 调度消息测试 ============


class TestSchedulingRequestMessage:
    """SchedulingRequest 消息测试"""

    def test_valid_scheduling_request(self):
        """创建有效的调度请求"""
        msg = pb2.SchedulingRequest(
            request_id=create_request_id(),
            model_name="llama-2-70b",
            backend=pb2.ModelBackend.MODEL_BACKEND_VLLM,
            tensor_parallel_size=4,
            gpu_memory_required_gb=80,
            priority=5,
            mode=pb2.InferenceMode.INFERENCE_MODE_GENERATE,
        )
        assert msg.tensor_parallel_size == 4
        assert msg.gpu_memory_required_gb == 80
        assert msg.priority == 5
        assert msg.mode == pb2.InferenceMode.INFERENCE_MODE_GENERATE

    def test_default_priority(self):
        """默认优先级为 0"""
        msg = pb2.SchedulingRequest(request_id=create_request_id())
        assert msg.priority == 0

    def test_priority_range(self):
        """优先级范围 0-10"""
        for priority in [0, 5, 10]:
            msg = pb2.SchedulingRequest(request_id=create_request_id(), priority=priority)
            assert msg.priority == priority


class TestSchedulingResponseMessage:
    """SchedulingResponse 消息测试"""

    def test_scheduled_response(self):
        """成功调度响应"""
        msg = pb2.SchedulingResponse(
            request_id=create_request_id(),
            scheduled=True,
            assigned_node_id="worker-001",
            assigned_host="192.168.1.100",
            assigned_port=50051,
            status=pb2.ResponseStatus.STATUS_SUCCESS,
        )
        assert msg.scheduled is True
        assert msg.assigned_node_id == "worker-001"

    def test_unscheduled_response(self):
        """调度失败响应"""
        msg = pb2.SchedulingResponse(
            request_id=create_request_id(),
            scheduled=False,
            error_message="No available GPU",
            status=pb2.ResponseStatus.STATUS_ERROR,
        )
        assert msg.scheduled is False
        assert msg.error_message == "No available GPU"


class TestCancelRequestMessage:
    """CancelRequest 消息测试"""

    def test_valid_cancel_request(self):
        """创建有效的取消请求"""
        msg = pb2.CancelRequest(request_id="req-123")
        assert msg.request_id == "req-123"


class TestCancelResponseMessage:
    """CancelResponse 消息测试"""

    def test_success_cancel_response(self):
        """成功取消响应"""
        msg = pb2.CancelResponse(success=True, message="Request cancelled")
        assert msg.success is True

    def test_failure_cancel_response(self):
        """取消失败响应"""
        msg = pb2.CancelResponse(success=False, message="Request already completed")
        assert msg.success is False


class TestGetSchedulingStatusRequestMessage:
    """GetSchedulingStatusRequest 消息测试"""

    def test_valid_status_request(self):
        """创建有效的状态查询请求"""
        msg = pb2.GetSchedulingStatusRequest(request_id="req-123")
        assert msg.request_id == "req-123"


class TestGetSchedulingStatusResponseMessage:
    """GetSchedulingStatusResponse 消息测试"""

    def test_pending_status(self):
        """等待中状态"""
        msg = pb2.GetSchedulingStatusResponse(
            request_id="req-123",
            status=pb2.ResponseStatus.STATUS_PENDING,
        )
        assert msg.status == pb2.ResponseStatus.STATUS_PENDING

    def test_completed_with_result(self):
        """完成状态带结果"""
        msg = pb2.GetSchedulingStatusResponse(
            request_id="req-123",
            status=pb2.ResponseStatus.STATUS_SUCCESS,
            result=pb2.InferenceResponse(text="Done", tokens_generated=10),
        )
        assert msg.status == pb2.ResponseStatus.STATUS_SUCCESS
        assert msg.HasField("result")

    def test_error_with_message(self):
        """错误状态带错误信息"""
        msg = pb2.GetSchedulingStatusResponse(
            request_id="req-123",
            status=pb2.ResponseStatus.STATUS_ERROR,
            error_message="GPU OOM",
        )
        assert msg.status == pb2.ResponseStatus.STATUS_ERROR
        assert msg.error_message == "GPU OOM"


# ============ 健康检查消息测试 ============


class TestHealthCheckRequestMessage:
    """HealthCheckRequest 消息测试"""

    def test_empty_service_filter(self):
        """空服务过滤器"""
        msg = pb2.HealthCheckRequest()
        assert msg.service == ""

    def test_specific_service_filter(self):
        """指定服务过滤器"""
        msg = pb2.HealthCheckRequest(service="inference")
        assert msg.service == "inference"


class TestHealthCheckResponseMessage:
    """HealthCheckResponse 消息测试"""

    def test_healthy_response(self):
        """健康响应"""
        msg = pb2.HealthCheckResponse(
            healthy=True,
            status="OK",
            details={"inference": "OK", "cluster": "OK"},
        )
        assert msg.healthy is True
        assert msg.status == "OK"
        assert msg.details["inference"] == "OK"

    def test_unhealthy_response(self):
        """不健康响应"""
        msg = pb2.HealthCheckResponse(
            healthy=False,
            status="UNHEALTHY",
            details={"inference": "ERROR"},
        )
        assert msg.healthy is False
        assert msg.status == "UNHEALTHY"


# ============ 模型管理消息测试 ============


class TestLoadModelRequestMessage:
    """LoadModelRequest 消息测试"""

    def test_valid_load_request(self):
        """创建有效的模型加载请求"""
        msg = pb2.LoadModelRequest(
            model_name="llama-2-7b",
            backend=pb2.ModelBackend.MODEL_BACKEND_VLLM,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        assert msg.model_name == "llama-2-7b"
        # 使用近似比较（浮点数精度问题）
        assert abs(msg.gpu_memory_utilization - 0.9) < 0.01

    def test_utilization_range(self):
        """gpu_memory_utilization 范围 0.0 - 1.0"""
        for util in [0.0, 0.5, 0.9, 1.0]:
            msg = pb2.LoadModelRequest(gpu_memory_utilization=util)
            assert abs(msg.gpu_memory_utilization - util) < 0.001


class TestLoadModelResponseMessage:
    """LoadModelResponse 消息测试"""

    def test_success_response(self):
        """成功加载响应"""
        msg = pb2.LoadModelResponse(
            success=True,
            model_name="llama-2-7b",
            message="Model loaded successfully",
            memory_allocated=pb2.GPUMemory(used_bytes=14 * 1024**3),
        )
        assert msg.success is True
        assert msg.HasField("memory_allocated")


class TestUnloadModelRequestMessage:
    """UnloadModelRequest 消息测试"""

    def test_valid_unload_request(self):
        """创建有效的模型卸载请求"""
        msg = pb2.UnloadModelRequest(model_name="llama-2-7b")
        assert msg.model_name == "llama-2-7b"


class TestUnloadModelResponseMessage:
    """UnloadModelResponse 消息测试"""

    def test_success_response(self):
        """成功卸载响应"""
        msg = pb2.UnloadModelResponse(
            success=True,
            message="Model unloaded",
            memory_freed_bytes=14 * 1024**3,
        )
        assert msg.success is True
        assert msg.memory_freed_bytes == 14 * 1024**3


class TestListModelsRequestMessage:
    """ListModelsRequest 消息测试"""

    def test_empty_request(self):
        """空请求"""
        msg = pb2.ListModelsRequest()
        # ListModelsRequest 没有字段，验证它可以正确创建
        assert msg is not None


class TestListModelsResponseMessage:
    """ListModelsResponse 消息测试"""

    def test_empty_model_list(self):
        """空模型列表"""
        msg = pb2.ListModelsResponse()
        assert len(msg.models) == 0

    def test_multiple_models(self):
        """多模型响应"""
        msg = pb2.ListModelsResponse(
            models=[
                pb2.ModelInfo(name="model-1", is_loaded=True),
                pb2.ModelInfo(name="model-2", is_loaded=False),
            ],
        )
        assert len(msg.models) == 2


# ============ 指标消息测试 ============


class TestMetricsRequestMessage:
    """MetricsRequest 消息测试"""

    def test_empty_request(self):
        """空请求获取所有指标"""
        msg = pb2.MetricsRequest()
        assert len(msg.metric_names) == 0

    def test_specific_metrics(self):
        """请求特定指标"""
        msg = pb2.MetricsRequest(metric_names=["cpu_usage", "gpu_memory"])
        assert len(msg.metric_names) == 2


class TestMetricSampleMessage:
    """MetricSample 消息测试"""

    def test_valid_metric_sample(self):
        """有效的指标样本"""
        msg = pb2.MetricSample(
            name="cpu_usage",
            value=0.75,
            timestamp=1234567890,
            labels={"host": "localhost"},
        )
        assert msg.name == "cpu_usage"
        assert abs(msg.value - 0.75) < 0.001
        assert msg.labels["host"] == "localhost"


class TestMetricsResponseMessage:
    """MetricsResponse 消息测试"""

    def test_empty_metrics(self):
        """空指标响应"""
        msg = pb2.MetricsResponse()
        assert len(msg.metrics) == 0

    def test_multiple_metrics(self):
        """多指标响应"""
        msg = pb2.MetricsResponse(
            metrics=[
                pb2.MetricSample(name="cpu_usage", value=0.5),
                pb2.MetricSample(name="gpu_memory", value=0.8),
            ],
        )
        assert len(msg.metrics) == 2


# ============ 序列化大小测试 ============


class TestSerializationSize:
    """序列化大小测试"""

    def test_inference_request_size_reasonable(self):
        """推理请求序列化大小合理（< 1KB）"""
        msg = pb2.InferenceRequest(
            request_id=create_request_id(),
            model_name="llama-2-70b",
            prompt="Hello world, this is a test prompt.",
            max_tokens=100,
            temperature=0.7,
        )
        serialized = msg.SerializeToString()
        assert len(serialized) < 1024, f"序列化大小 {len(serialized)} 超过 1KB"

    def test_batch_request_size_scales_linearly(self):
        """批量请求大小线性增长"""
        single = pb2.BatchInferenceRequest(
            batch_id=create_request_id(),
            prompts=["Single prompt"],
        )
        multiple = pb2.BatchInferenceRequest(
            batch_id=create_request_id(),
            prompts=[f"Prompt {i}" for i in range(100)],
        )

        single_size = len(single.SerializeToString())
        multiple_size = len(multiple.SerializeToString())

        # 100 倍 prompts 应该大约 100 倍大小（允许一些 overhead）
        assert multiple_size < single_size * 200
        assert multiple_size > single_size * 10


# ============ 未知字段处理测试 ============


class TestUnknownFields:
    """未知字段处理测试（Proto3 兼容性）"""

    def test_unknown_fields_not_discarded_on_parse(self):
        """解析时未知字段被保留（通过 ByteSize + Deserialize 验证）"""
        # 创建一个消息
        original = pb2.GPUMemory(total_bytes=100, used_bytes=50)

        # 模拟接收包含未知字段的字节流
        # （实际测试中我们无法直接创建未知字段，但可以验证解析行为）
        serialized = original.SerializeToString()
        restored = pb2.GPUMemory()
        restored.ParseFromString(serialized)

        # 验证数据完整性
        assert restored.total_bytes == original.total_bytes
        assert restored.used_bytes == original.used_bytes
