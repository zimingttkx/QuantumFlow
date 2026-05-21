"""SDK 数据模型测试"""
import pytest
from quantumflow.sdk.models import SamplingParams, InferenceRequest, ChatMessage, InferenceResponse


class TestSamplingParams:
    """SamplingParams 测试"""

    def test_default_values(self):
        """验证：默认参数值"""
        params = SamplingParams()
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 2048
        assert params.repetition_penalty == 1.0

    def test_custom_values(self):
        """验证：自定义参数值"""
        params = SamplingParams(temperature=0.5, max_tokens=100, top_k=20)
        assert params.temperature == 0.5
        assert params.max_tokens == 100
        assert params.top_k == 20

    def test_to_dict(self):
        """验证：转换为字典"""
        params = SamplingParams(temperature=0.8, max_tokens=512)
        data = params.to_dict()
        assert data["temperature"] == 0.8
        assert data["max_tokens"] == 512
        assert data["top_p"] == 0.9  # default


class TestInferenceRequest:
    """InferenceRequest 测试"""

    def test_basic_request(self):
        """验证：基本请求创建"""
        request = InferenceRequest(model="test-model", prompt="Hello, world!")
        assert request.model == "test-model"
        assert request.prompt == "Hello, world!"
        assert request.stream is False
        assert request.priority == 5

    def test_request_with_sampling(self):
        """验证：带采样参数的请求"""
        sampling = SamplingParams(temperature=0.3, max_tokens=100)
        request = InferenceRequest(
            model="test-model",
            prompt="Hello",
            sampling_params=sampling
        )
        assert request.sampling_params.temperature == 0.3
        assert request.sampling_params.max_tokens == 100

    def test_request_with_session_id(self):
        """验证：带 session_id 的请求"""
        request = InferenceRequest(
            model="test-model",
            prompt="Hello",
            session_id="session-123"
        )
        assert request.session_id == "session-123"
        data = request.to_dict()
        assert data["session_id"] == "session-123"

    def test_to_dict(self):
        """验证：转换为字典"""
        request = InferenceRequest(model="test-model", prompt="Hello")
        data = request.to_dict()
        assert data["model"] == "test-model"
        assert data["prompt"] == "Hello"
        assert "sampling_params" in data


class TestChatMessage:
    """ChatMessage 测试"""

    def test_user_message(self):
        """验证：用户消息"""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        """验证：助手消息"""
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_to_dict(self):
        """验证：转换为字典"""
        msg = ChatMessage(role="system", content="You are helpful.")
        data = msg.to_dict()
        assert data["role"] == "system"
        assert data["content"] == "You are helpful."


class TestInferenceResponse:
    """InferenceResponse 测试"""

    def test_response_creation(self):
        """验证：响应创建"""
        response = InferenceResponse(
            request_id="req_001",
            model="test-model",
            generated_text="Hello, world!",
            finish_reason="stop",
            latency_ms=150.5,
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8
        )
        assert response.request_id == "req_001"
        assert response.generated_text == "Hello, world!"
        assert response.finish_reason == "stop"
        assert response.latency_ms == 150.5
        assert response.total_tokens == 8

    def test_usage_properties(self):
        """验证：使用量属性"""
        response = InferenceResponse(
            request_id="req_002",
            model="test-model",
            generated_text="Test",
            finish_reason="stop",
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15