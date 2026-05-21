"""AsyncQuantumFlowClient 异步客户端测试"""
import asyncio
import httpx
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from quantumflow.sdk.client import AsyncQuantumFlowClient, SyncQuantumFlowClient
from quantumflow.sdk.exceptions import APIError, RateLimitError, TimeoutError


class TestAsyncClientInit:
    """异步客户端初始化测试"""

    def test_default_initialization(self):
        """验证：默认初始化"""
        client = AsyncQuantumFlowClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0

    def test_custom_initialization(self):
        """验证：自定义初始化"""
        client = AsyncQuantumFlowClient(
            base_url="http://example.com:9000",
            api_key="test-key",
            timeout=60.0
        )
        assert client.base_url == "http://example.com:9000"
        assert client.api_key == "test-key"
        assert client.timeout == 60.0


class TestAsyncClientRequest:
    """异步请求测试"""

    def test_arequest_method_exists(self):
        """验证：_arequest 方法存在"""
        client = AsyncQuantumFlowClient()
        assert hasattr(client, '_arequest')

    def test_arequest_is_coroutine(self):
        """验证：_arequest 是异步方法"""
        client = AsyncQuantumFlowClient()
        import inspect
        assert inspect.iscoroutinefunction(client._arequest)


class TestAsyncGenerate:
    """异步 generate 测试"""

    def test_generate_is_coroutine(self):
        """验证：generate 是异步方法"""
        client = AsyncQuantumFlowClient()
        import inspect
        assert inspect.iscoroutinefunction(client.generate)


class TestAsyncListModels:
    """异步 list_models 测试"""

    def test_list_models_is_coroutine(self):
        """验证：list_models 是异步方法"""
        client = AsyncQuantumFlowClient()
        import inspect
        assert inspect.iscoroutinefunction(client.list_models)


class TestAsyncHealthCheck:
    """异步 health_check 测试"""

    def test_health_check_is_coroutine(self):
        """验证：health_check 是异步方法"""
        client = AsyncQuantumFlowClient()
        import inspect
        assert inspect.iscoroutinefunction(client.health_check)


class TestAsyncContextManager:
    """异步上下文管理器测试"""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """验证：异步上下文管理器"""
        async with AsyncQuantumFlowClient() as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_async_context_manager_with_health(self):
        """验证：异步上下文管理器中使用"""
        async with AsyncQuantumFlowClient() as client:
            health = await client.health_check()
            assert health is not None


class TestAsyncErrorHandling:
    """异步错误处理测试"""

    @pytest.mark.asyncio
    async def test_async_rate_limit_error(self):
        """验证：异步限流错误"""
        client = AsyncQuantumFlowClient()
        with patch.object(client, '_arequest', new_callable=AsyncMock) as mock:
            mock.side_effect = RateLimitError()
            with pytest.raises(RateLimitError):
                await client.generate(model="test", prompt="hello")

    @pytest.mark.asyncio
    async def test_async_api_error(self):
        """验证：异步 API 错误"""
        client = AsyncQuantumFlowClient()
        with patch.object(client, '_arequest', new_callable=AsyncMock) as mock:
            mock.side_effect = APIError(500, "Server error")
            with pytest.raises(APIError) as exc:
                await client.generate(model="test", prompt="hello")
            assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_async_timeout_error(self):
        """验证：异步超时错误"""
        client = AsyncQuantumFlowClient()
        with patch.object(client, '_arequest', new_callable=AsyncMock) as mock:
            mock.side_effect = TimeoutError("Request timed out")
            with pytest.raises(TimeoutError):
                await client.generate(model="test", prompt="hello")

    @pytest.mark.asyncio
    async def test_async_with_api_key(self):
        """验证：带 api_key 的异步请求"""
        client = AsyncQuantumFlowClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"
        # headers 构建在 _arequest 中，需要 mock httpx 来覆盖
        # 这里验证初始化正确即可
        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_async_client.return_value.__aenter__.return_value.request.return_value = mock_response
            health = await client.health_check()
            assert health["status"] == "healthy"
