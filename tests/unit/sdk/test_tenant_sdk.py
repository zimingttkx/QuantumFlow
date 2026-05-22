"""SDK 租户支持测试"""
import json

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from quantumflow.sdk.client import SyncQuantumFlowClient, AsyncQuantumFlowClient
from quantumflow.sdk.exceptions import APIError, RateLimitError, TimeoutError


class TestSyncClient:
    def test_api_key_header(self):
        """API Key 通过 X-API-Key header 传递"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000", api_key="qk_test_key")
        assert client._client.headers["X-API-Key"] == "qk_test_key"

    def test_tenant_id_header(self):
        """Tenant ID 通过 X-Tenant-ID header 传递"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000", tenant_id="tenant-abc")
        assert client._client.headers["X-Tenant-ID"] == "tenant-abc"

    def test_both_headers(self):
        """同时发送 API Key 和 Tenant ID"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000", api_key="qk_both", tenant_id="tenant-both")
        assert client._client.headers["X-API-Key"] == "qk_both"
        assert client._client.headers["X-Tenant-ID"] == "tenant-both"

    def test_no_headers_when_not_provided(self):
        """未提供时不应设置 header"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000")
        assert "X-API-Key" not in client._client.headers
        assert "X-Tenant-ID" not in client._client.headers

    def test_health_check(self):
        """健康检查发送 GET 请求"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch.object(SyncQuantumFlowClient, '_get', return_value={"status": "healthy"}):
            client = SyncQuantumFlowClient()
            result = client.health_check()
            assert result["status"] == "healthy"

    def test_rate_limit_error(self):
        """429 响应抛出 RateLimitError"""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.Client.post", return_value=mock_response):
            client = SyncQuantumFlowClient()
            with pytest.raises(RateLimitError):
                client._post("/test")

    def test_api_error(self):
        """4xx/5xx 响应抛出 APIError"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.Client.post", return_value=mock_response):
            client = SyncQuantumFlowClient()
            with pytest.raises(APIError):
                client._post("/test")

    def test_timeout_error(self):
        """超时抛出 TimeoutError"""
        import httpx
        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("timeout")):
            client = SyncQuantumFlowClient()
            with pytest.raises(TimeoutError):
                client._post("/test")

    def test_close(self):
        """测试关闭客户端"""
        mock_client = MagicMock()
        client = SyncQuantumFlowClient()
        client._client = mock_client
        client.close()
        mock_client.close.assert_called_once()

    def test_context_manager(self):
        """测试上下文管理器"""
        mock_client = MagicMock()
        with SyncQuantumFlowClient() as client:
            client._client = mock_client
        mock_client.close.assert_called_once()


class TestAsyncClient:
    def test_async_tenant_headers(self):
        """异步客户端设置 header"""
        client = AsyncQuantumFlowClient(base_url="http://localhost:8000", api_key="qk_async", tenant_id="tenant-async")
        assert client.api_key == "qk_async"
        assert client.tenant_id == "tenant-async"

    @pytest.mark.asyncio
    async def test_async_rate_limit_error(self):
        """异步客户端 429 抛出 RateLimitError"""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient.request", AsyncMock(return_value=mock_response)):
            client = AsyncQuantumFlowClient()
            with pytest.raises(RateLimitError):
                await client._arequest("GET", "/test")

    @pytest.mark.asyncio
    async def test_async_api_error(self):
        """异步客户端 4xx/5xx 抛出 APIError"""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        with patch("httpx.AsyncClient.request", AsyncMock(return_value=mock_response)):
            client = AsyncQuantumFlowClient()
            with pytest.raises(APIError):
                await client._arequest("GET", "/test")
