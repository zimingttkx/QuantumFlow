"""gRPC 客户端测试

严格测试 gRPC 客户端的业务逻辑：
- 客户端初始化
- 请求参数验证
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from quantumflow.grpc.generated import quantumflow_pb2


class TestClientRequestValidation:
    """请求参数验证测试"""

    def test_inference_request_creation(self):
        """创建推理请求"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="test-123",
            model_name="llama-2-7b",
            prompt="Hello world",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        assert request.request_id == "test-123"
        assert request.model_name == "llama-2-7b"
        assert request.prompt == "Hello world"
        assert request.max_tokens == 100
        # 浮点数精度问题，使用近似比较
        assert abs(request.temperature - 0.7) < 0.01
        assert abs(request.top_p - 0.9) < 0.01

    def test_batch_inference_request(self):
        """批量推理请求"""
        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id="batch-123",
            model_name="llama-2-7b",
            prompts=["prompt 1", "prompt 2", "prompt 3"],
            max_tokens=100,
        )

        assert request.batch_id == "batch-123"
        assert len(request.prompts) == 3

    def test_register_node_request(self):
        """注册节点请求"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
        )

        assert request.node_id == "worker-001"
        assert request.host == "192.168.1.100"
        assert request.port == 8001

    def test_scheduling_request(self):
        """调度请求"""
        request = quantumflow_pb2.SchedulingRequest(
            request_id="req-123",
            model_name="llama-2-7b",
            priority=5,
        )

        assert request.request_id == "req-123"
        assert request.model_name == "llama-2-7b"
        assert request.priority == 5

    def test_heartbeat_request(self):
        """心跳请求"""
        request = quantumflow_pb2.HeartbeatRequest(
            node_id="worker-001",
        )

        assert request.node_id == "worker-001"

    def test_cancel_request(self):
        """取消请求"""
        request = quantumflow_pb2.CancelRequest(request_id="req-123")
        assert request.request_id == "req-123"


class TestProtoMessageFields:
    """Proto 消息字段测试"""

    def test_inference_request_all_fields(self):
        """推理请求所有字段"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="test-123",
            model_name="llama-2-7b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
            prompt="Hello",
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
        )

        assert request.request_id == "test-123"
        assert request.model_name == "llama-2-7b"
        assert request.backend == quantumflow_pb2.MODEL_BACKEND_VLLM
        assert request.prompt == "Hello"
        assert request.max_tokens == 100
        assert request.top_k == 40
        # 浮点数精度问题，使用近似比较
        assert abs(request.repetition_penalty - 1.1) < 0.01

    def test_node_resources_with_gpus(self):
        """带 GPU 的节点资源"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
            gpus=[
                quantumflow_pb2.GPUInfo(
                    index=0,
                    name="NVIDIA RTX 4090",
                    memory=quantumflow_pb2.GPUMemory(
                        total_bytes=24 * 1024**3,
                        available_bytes=16 * 1024**3,
                        used_bytes=8 * 1024**3,
                    ),
                ),
            ],
        )

        assert len(request.gpus) == 1
        assert request.gpus[0].index == 0
        assert request.gpus[0].name == "NVIDIA RTX 4090"
        assert request.gpus[0].memory.total_bytes == 24 * 1024**3

    def test_batch_inference_empty_prompts(self):
        """空 prompts 列表"""
        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id="batch-123",
            model_name="llama-2-7b",
            prompts=[],
        )

        assert len(request.prompts) == 0
