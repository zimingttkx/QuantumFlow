"""API单元测试"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from quantumflow.api.models import (
    SamplingParams,
    InferenceRequest,
    BatchInferenceRequest,
    ChatMessage,
    ChatRequest,
    DeployRequest,
    InferenceResponse,
    ModelInfo,
    NodeInfo,
    GPUInfo,
    ClusterStatus,
    HealthResponse,
    ErrorResponse,
    ErrorDetail,
)


class TestSamplingParams:
    """采样参数测试"""

    def test_default_values(self):
        """测试默认值"""
        params = SamplingParams()

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 2048
        assert params.repetition_penalty == 1.0
        assert params.stop is None

    def test_custom_values(self):
        """测试自定义值"""
        params = SamplingParams(
            temperature=0.5,
            top_p=0.95,
            top_k=100,
            max_tokens=1024,
            stop=["END", "STOP"],
        )

        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 100
        assert params.max_tokens == 1024
        assert params.stop == ["END", "STOP"]

    def test_temperature_validation(self):
        """测试温度参数验证"""
        with pytest.raises(ValidationError):
            SamplingParams(temperature=3.0)  # 超出范围

        with pytest.raises(ValidationError):
            SamplingParams(temperature=-0.1)  # 低于范围

    def test_max_tokens_validation(self):
        """测试最大token验证"""
        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=0)  # 必须大于0

        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=50000)  # 超出最大限制


class TestInferenceRequest:
    """推理请求测试"""

    def test_required_fields(self):
        """测试必需字段"""
        request = InferenceRequest(
            model="Qwen2.5-7B",
            prompt="Hello, world!",
        )

        assert request.model == "Qwen2.5-7B"
        assert request.prompt == "Hello, world!"
        assert request.stream is False
        assert request.priority == 5

    def test_all_fields(self):
        """测试所有字段"""
        request = InferenceRequest(
            model="Qwen2.5-7B",
            prompt="Hello",
            sampling_params=SamplingParams(temperature=0.8),
            stream=True,
            session_id="session-123",
            priority=8,
            tags={"user": "test"},
            request_id="req-001",
        )

        assert request.stream is True
        assert request.session_id == "session-123"
        assert request.priority == 8
        assert request.tags == {"user": "test"}
        assert request.request_id == "req-001"

    def test_default_sampling_params(self):
        """测试默认采样参数"""
        request = InferenceRequest(
            model="test",
            prompt="test",
        )

        assert isinstance(request.sampling_params, SamplingParams)


class TestBatchInferenceRequest:
    """批量推理请求测试"""

    def test_single_prompt(self):
        """测试单个提示"""
        request = BatchInferenceRequest(
            model="Qwen2.5-7B",
            prompts=["Hello"],
        )

        assert len(request.prompts) == 1

    def test_multiple_prompts(self):
        """测试多个提示"""
        request = BatchInferenceRequest(
            model="Qwen2.5-7B",
            prompts=[f"Prompt {i}" for i in range(10)],
        )

        assert len(request.prompts) == 10

    def test_max_prompts_validation(self):
        """测试最大提示数量验证"""
        with pytest.raises(ValidationError):
            BatchInferenceRequest(
                model="test",
                prompts=[f"Prompt {i}" for i in range(101)],  # 超出限制
            )

    def test_min_prompts_validation(self):
        """测试最小提示数量验证"""
        with pytest.raises(ValidationError):
            BatchInferenceRequest(
                model="test",
                prompts=[],  # 不能为空
            )


class TestChatMessage:
    """对话消息测试"""

    def test_valid_roles(self):
        """测试有效角色"""
        for role in ["system", "user", "assistant"]:
            message = ChatMessage(role=role, content="Hello")
            assert message.role == role


class TestChatRequest:
    """对话请求测试"""

    def test_valid_request(self):
        """测试有效请求"""
        request = ChatRequest(
            model="Qwen2.5-7B",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
        )

        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"

    def test_empty_messages_validation(self):
        """测试空消息验证"""
        with pytest.raises(ValidationError):
            ChatRequest(
                model="test",
                messages=[],
            )


class TestDeployRequest:
    """部署请求测试"""

    def test_required_fields(self):
        """测试必需字段"""
        request = DeployRequest(model="Qwen2.5-7B")

        assert request.model == "Qwen2.5-7B"
        assert request.tensor_parallel == 1
        assert request.pipeline_parallel == 1
        assert request.gpu_memory_utilization == 0.9

    def test_parallel_validation(self):
        """测试并行度验证"""
        with pytest.raises(ValidationError):
            DeployRequest(model="test", tensor_parallel=0)  # 必须 >= 1

        with pytest.raises(ValidationError):
            DeployRequest(model="test", tensor_parallel=16)  # 最大为8

    def test_gpu_memory_utilization_validation(self):
        """测试GPU显存利用率验证"""
        with pytest.raises(ValidationError):
            DeployRequest(model="test", gpu_memory_utilization=0.05)

        with pytest.raises(ValidationError):
            DeployRequest(model="test", gpu_memory_utilization=1.5)


class TestInferenceResponse:
    """推理响应测试"""

    def test_valid_response(self):
        """测试有效响应"""
        response = InferenceResponse(
            request_id="req-001",
            model="Qwen2.5-7B",
            prompt="Hello",
            generated_text="Hi there!",
            finish_reason="stop",
            latency_ms=123.45,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )

        assert response.request_id == "req-001"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 30


class TestGPUInfo:
    """GPU信息测试"""

    def test_valid_gpu_info(self):
        """测试有效GPU信息"""
        info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024**3,
            memory_used=10 * 1024**3,
            memory_free=14 * 1024**3,
            utilization=0.5,
            temperature=45.0,
        )

        assert info.memory_used_percent == pytest.approx(0.4167, rel=0.01)

    def test_memory_used_percent_zero_total(self):
        """测试零总显存"""
        info = GPUInfo(
            gpu_id=0,
            name="Test",
            memory_total=0,
            memory_used=0,
            memory_free=0,
            utilization=0.0,
            temperature=0.0,
        )

        assert info.memory_used_percent == 0.0


class TestModelInfo:
    """模型信息测试"""

    def test_valid_model_info(self):
        """测试有效模型信息"""
        info = ModelInfo(
            model_id="qwen2.5-7b",
            name="Qwen2.5-7B-Instruct",
            architecture="Qwen2ForCausalLM",
            parameter_count=7_000_000_000,
            dtype="bfloat16",
            status="ready",
            replicas=2,
            tensor_parallel=1,
            max_model_length=8192,
            backend="vllm",
        )

        assert info.model_id == "qwen2.5-7b"
        assert info.parameter_count == 7_000_000_000
        assert info.status == "ready"


class TestNodeInfo:
    """节点信息测试"""

    def test_valid_node_info(self):
        """测试有效节点信息"""
        from datetime import datetime

        info = NodeInfo(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.101",
            port=8001,
            status="healthy",
            gpu_count=4,
            gpu_info=[],
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            current_load=0.3,
            labels={"zone": "zone-a"},
            version="1.0.0",
            uptime_seconds=3600,
            last_heartbeat=datetime.now(),
        )

        assert info.node_id == "node-1"
        assert info.memory_available_percent == pytest.approx(0.5, rel=0.01)


class TestClusterStatus:
    """集群状态测试"""

    def test_valid_status(self):
        """测试有效状态"""
        status = ClusterStatus(
            total_nodes=5,
            healthy_nodes=4,
            unhealthy_nodes=1,
            draining_nodes=0,
            total_gpus=20,
            available_gpus=16,
            active_models=3,
            pending_jobs=5,
            running_jobs=10,
            system_metrics={"cpu_usage": 0.5},
            uptime_seconds=86400,
        )

        assert status.total_nodes == 5
        assert status.healthy_nodes == 4
        assert status.available_gpus == 16


class TestHealthResponse:
    """健康响应测试"""

    def test_healthy_response(self):
        """测试健康响应"""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600,
            checks={"api": "ok", "redis": "ok"},
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"


class TestErrorResponse:
    """错误响应测试"""

    def test_error_detail(self):
        """测试错误详情"""
        error = ErrorDetail(
            code="MODEL_NOT_FOUND",
            message="Model not found",
            details={"model": "test"},
        )

        assert error.code == "MODEL_NOT_FOUND"
        assert error.details["model"] == "test"

    def test_error_response(self):
        """测试错误响应"""
        error_detail = ErrorDetail(
            code="INVALID_REQUEST",
            message="Invalid parameters",
        )

        response = ErrorResponse(error=error_detail)

        assert response.error.code == "INVALID_REQUEST"
