"""WorkerClient gap coverage tests

Tests covering gaps found in test_worker_client.py:
1. _get_client lazy initialization
2. _get_client recreation when existing client is closed
3. close() when _client is None
4. load_model error HTTP responses
5. load_model exception handling
6. get_status error HTTP responses
7. get_status exception handling
8. inference with non-500 HTTP error codes (400, 429)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from quantumflow.scheduler.worker_client import (
    WorkerClient,
    WorkerEndpoint,
    WorkerRegistry,
)


class TestWorkerClientGetClient:
    """Tests for _get_client lazy initialization and recreation."""

    @pytest.fixture
    def client(self):
        return WorkerClient(timeout=10.0)

    @pytest.mark.asyncio
    async def test_get_client_creates_new_when_none(self, client):
        """_get_client creates a new httpx.AsyncClient when _client is None."""
        assert client._client is None

        http_client = await client._get_client()

        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)
        assert client._client is http_client

        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self, client):
        """_get_client reuses the existing client if it is not closed."""
        first = await client._get_client()
        second = await client._get_client()

        assert first is second

        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_recreates_when_closed(self, client):
        """_get_client creates a new client when the existing one is closed."""
        first = await client._get_client()
        await client.close()

        # After close, _client is None
        assert client._client is None

        second = await client._get_client()
        assert second is not None
        assert second is not first

        await client.close()


class TestWorkerClientCloseEdgeCases:
    """Tests for close() edge cases."""

    @pytest.mark.asyncio
    async def test_close_when_client_none_is_safe(self):
        """close() when _client is already None should not raise.

        必须验证:
        1. 不抛异常 (idempotent close)
        2. _client 保持为 None (不要被 close 错误地赋值/恢复成别的对象)
        3. 二次 close 也安全
        """
        client = WorkerClient()
        client._client = None
        await client.close()
        # 真实断言: 状态保持, 二次 close 也安全
        assert client._client is None, (
            f"close() 后 _client 应保持 None, 实际 {client._client!r}"
        )
        # 二次 close 必须也安全
        await client.close()
        assert client._client is None, (
            f"二次 close() 后 _client 应仍为 None, 实际 {client._client!r}"
        )

    @pytest.mark.asyncio
    async def test_close_when_client_already_closed_is_safe(self):
        """close() when the underlying httpx client is already closed should not raise.

        Note: The implementation only sets _client=None when it actually closes
        the client (i.e., when is_closed is False). When already closed, _client
        is retained, but aclose is not called again.
        """
        client = WorkerClient()
        mock_http = MagicMock()
        mock_http.is_closed = True
        mock_http.aclose = AsyncMock()
        client._client = mock_http

        await client.close()

        # _client is not cleared because is_closed=True skips the cleanup block
        assert client._client is mock_http
        mock_http.aclose.assert_not_called()


class TestWorkerClientLoadModelEdgeCases:
    """Tests for load_model error handling."""

    @pytest.fixture
    def client(self):
        return WorkerClient(timeout=10.0)

    @pytest.fixture
    def endpoint(self):
        return WorkerEndpoint(node_id="w1", host="localhost", port=8080)

    @pytest.mark.asyncio
    async def test_load_model_handles_http_error(self, client, endpoint):
        """load_model returns error dict on non-200 HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http):
            result = await client.load_model(
                endpoint=endpoint,
                model_name="test-model",
            )

        assert result["status"] == "error"
        assert result["code"] == 500

    @pytest.mark.asyncio
    async def test_load_model_handles_exception(self, client, endpoint):
        """load_model returns error dict on connection exception."""
        mock_http = MagicMock()
        mock_http.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        mock_http.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http):
            result = await client.load_model(
                endpoint=endpoint,
                model_name="test-model",
            )

        assert result["status"] == "error"
        assert "Connection failed" in result["error"]


class TestWorkerClientGetStatusEdgeCases:
    """Tests for get_status error handling."""

    @pytest.fixture
    def client(self):
        return WorkerClient(timeout=10.0)

    @pytest.fixture
    def endpoint(self):
        return WorkerEndpoint(node_id="w1", host="localhost", port=8080)

    @pytest.mark.asyncio
    async def test_get_status_handles_http_error(self, client, endpoint):
        """get_status returns error dict on non-200 HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http):
            result = await client.get_status(endpoint)

        assert result["status"] == "error"
        assert result["code"] == 503

    @pytest.mark.asyncio
    async def test_get_status_handles_exception(self, client, endpoint):
        """get_status returns error dict on connection exception."""
        mock_http = MagicMock()
        mock_http.get = AsyncMock(side_effect=OSError("Network down"))
        mock_http.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http):
            result = await client.get_status(endpoint)

        assert result["status"] == "error"
        assert "Network down" in result["error"]


class TestWorkerClientInferenceErrorCodes:
    """Tests for inference with various HTTP error status codes."""

    @pytest.fixture
    def client(self):
        return WorkerClient(timeout=10.0)

    @pytest.fixture
    def endpoint(self):
        return WorkerEndpoint(node_id="w1", host="localhost", port=8080)

    @pytest.mark.asyncio
    async def test_inference_handles_400_bad_request(self, client, endpoint):
        """inference returns error on HTTP 400."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="m",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "400" in result["error"]

    @pytest.mark.asyncio
    async def test_inference_handles_429_rate_limit(self, client, endpoint):
        """inference returns error on HTTP 429 rate limit."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"

        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="m",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "429" in result["error"]


class TestWorkerClientTimeoutConfiguration:
    """Tests for timeout configuration."""

    @pytest.mark.asyncio
    async def test_custom_timeout_is_stored(self):
        """WorkerClient stores the timeout value."""
        client = WorkerClient(timeout=45.0)
        assert client.timeout == 45.0

    @pytest.mark.asyncio
    async def test_default_timeout_is_30(self):
        """WorkerClient default timeout is 30.0."""
        client = WorkerClient()
        assert client.timeout == 30.0


class TestWorkerRegistryUpdateNonexistent:
    """Tests for updating status of non-existent workers."""

    @pytest.mark.asyncio
    async def test_update_worker_status_nonexistent_is_safe(self):
        """update_worker_status on non-existent worker should not raise.

        必须验证:
        1. 不抛异常 (静默 no-op)
        2. 不会意外创建/插入新的 worker 记录
        3. 后续 get_worker(nonexistent) 返回 None (而非被错误地建出来)
        """
        registry = WorkerRegistry()
        await registry.update_worker_status("nonexistent", "healthy")
        # 真实断言: 注册表保持空, 不被错误地插入 worker
        count = await registry.get_worker_count()
        assert count == 0, (
            f"update_worker_status 不应创建新 worker, 实际 count={count}"
        )
        result = await registry.get_worker("nonexistent")
        assert result is None, (
            f"nonexistent worker 应仍为 None, 实际被错误地创建成 {result!r}"
        )

    @pytest.mark.asyncio
    async def test_register_duplicate_overwrites(self):
        """Registering the same node_id twice overwrites the previous endpoint."""
        registry = WorkerRegistry()
        ep1 = WorkerEndpoint(node_id="w1", host="10.0.0.1", port=8000)
        ep2 = WorkerEndpoint(node_id="w1", host="10.0.0.2", port=8001)

        await registry.register(ep1)
        await registry.register(ep2)

        worker = await registry.get_worker("w1")
        assert worker.host == "10.0.0.2"
        assert worker.port == 8001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
