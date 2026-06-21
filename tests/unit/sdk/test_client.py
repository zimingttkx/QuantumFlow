"""SDK 同步客户端测试"""
import httpx
import pytest
from unittest.mock import Mock, patch, MagicMock
from quantumflow.sdk.client import SyncQuantumFlowClient
from quantumflow.sdk.exceptions import APIError, RateLimitError, TimeoutError


class TestSyncQuantumFlowClientInit:
    """客户端初始化测试"""

    def test_default_initialization(self):
        """验证：默认初始化"""
        client = SyncQuantumFlowClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0

    def test_custom_initialization(self):
        """验证：自定义初始化"""
        client = SyncQuantumFlowClient(
            base_url="http://example.com:9000",
            api_key="secret-key-123",
            timeout=60.0
        )
        assert client.base_url == "http://example.com:9000"
        assert client.api_key == "secret-key-123"
        assert client.timeout == 60.0

    def test_base_url_trailing_slash(self):
        """验证：base_url 去除尾部斜杠"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"


class TestSyncClientGenerate:
    """generate 方法测试"""

    @patch("httpx.Client")
    def test_generate_basic(self, mock_client_class):
        """验证：基本生成请求"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "req_001",
            "model": "test-model",
            "prompt": "Hello",
            "generated_text": "Hello, world!",
            "finish_reason": "stop",
            "latency_ms": 150.0,
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        }
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        response = client.generate(model="test-model", prompt="Hello")

        assert response.request_id == "req_001"
        assert response.generated_text == "Hello, world!"
        assert response.model == "test-model"
        mock_client_instance.post.assert_called_once()

    @patch("httpx.Client")
    def test_generate_with_custom_params(self, mock_client_class):
        """验证：带自定义参数的生成请求"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "req_002",
            "model": "test-model",
            "prompt": "Hello",
            "generated_text": "Response",
            "finish_reason": "stop",
            "latency_ms": 100.0,
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        response = client.generate(
            model="test-model",
            prompt="Hello",
            temperature=0.5,
            max_tokens=100
        )

        assert response.request_id == "req_002"
        # Verify the request was made with correct params
        call_args = mock_client_instance.post.call_args
        json_data = call_args.kwargs.get("json") or call_args[1].get("json")
        assert json_data["sampling_params"]["temperature"] == 0.5
        assert json_data["sampling_params"]["max_tokens"] == 100


class TestSyncClientListModels:
    """list_models 方法测试"""

    @patch("httpx.Client")
    def test_list_models(self, mock_client_class):
        """验证：获取模型列表"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "model-1", "status": "ready"},
            {"name": "model-2", "status": "loading"}
        ]
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        models = client.list_models()

        assert len(models) == 2
        assert models[0]["name"] == "model-1"
        mock_client_instance.get.assert_called_once()


class TestSyncClientHealthCheck:
    """health_check 方法测试"""

    @patch("httpx.Client")
    def test_health_check(self, mock_client_class):
        """验证：健康检查"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": 3600
        }
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        health = client.health_check()

        assert health["status"] == "healthy"
        assert health["version"] == "1.0.0"


class TestSyncClientErrorHandling:
    """错误处理测试"""

    @patch("httpx.Client")
    def test_rate_limit_error(self, mock_client_class):
        """验证：POST 触发限流错误"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        from quantumflow.sdk.exceptions import RateLimitError
        with pytest.raises(RateLimitError):
            client.generate(model="test", prompt="hello")

    @patch("httpx.Client")
    def test_api_error(self, mock_client_class):
        """验证：POST 触发 API 错误"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        from quantumflow.sdk.exceptions import APIError
        with pytest.raises(APIError) as exc_info:
            client.generate(model="test", prompt="hello")
        assert exc_info.value.status_code == 500

    @patch("httpx.Client")
    def test_get_rate_limit_error(self, mock_client_class):
        """验证：GET 触发限流错误"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        from quantumflow.sdk.exceptions import RateLimitError
        with pytest.raises(RateLimitError):
            client.list_models()

    @patch("httpx.Client")
    def test_get_api_error(self, mock_client_class):
        """验证：GET 触发 API 错误"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        from quantumflow.sdk.exceptions import APIError
        with pytest.raises(APIError) as exc_info:
            client.list_models()
        assert exc_info.value.status_code == 404


class TestAsyncClientInit:
    """异步客户端初始化测试"""

    def test_async_client_default_init(self):
        """验证：异步客户端默认初始化"""
        from quantumflow.sdk.client import AsyncQuantumFlowClient
        client = AsyncQuantumFlowClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0

    def test_async_client_custom_init(self):
        """验证：异步客户端自定义初始化"""
        from quantumflow.sdk.client import AsyncQuantumFlowClient
        client = AsyncQuantumFlowClient(
            base_url="http://example.com:9000",
            api_key="test-key",
            timeout=60.0
        )
        assert client.base_url == "http://example.com:9000"
        assert client.api_key == "test-key"
        assert client.timeout == 60.0


class TestTimeoutHandling:
    """超时处理测试"""

    @patch("httpx.Client")
    def test_post_timeout(self, mock_client_class):
        """验证：POST 请求超时"""
        mock_client_instance = MagicMock()
        mock_client_instance.post.side_effect = httpx.TimeoutException("timeout")
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        from quantumflow.sdk.exceptions import TimeoutError
        with pytest.raises(TimeoutError):
            client.generate(model="test", prompt="hello")

    @patch("httpx.Client")
    def test_get_timeout(self, mock_client_class):
        """验证：GET 请求超时"""
        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = httpx.TimeoutException("timeout")
        mock_client_class.return_value = mock_client_instance

        client = SyncQuantumFlowClient()
        from quantumflow.sdk.exceptions import TimeoutError
        with pytest.raises(TimeoutError):
            client.list_models()


class TestSyncClientContextManager:
    """上下文管理器测试"""

    @patch("httpx.Client")
    def test_context_manager(self, mock_client_class):
        """验证：使用上下文管理器"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        with SyncQuantumFlowClient() as client:
            _ = client.health_check()

        mock_client_instance.close.assert_called_once()