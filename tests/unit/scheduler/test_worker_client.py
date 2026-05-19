"""WorkerClient 单元测试

测试 Worker HTTP 客户端的通信功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from quantumflow.scheduler.worker_client import (
    WorkerClient,
    WorkerEndpoint,
    WorkerRegistry,
    get_worker_registry,
)

# ==================== WorkerEndpoint 测试 ====================


class TestWorkerEndpoint:
    """WorkerEndpoint 数据类测试"""

    def test_url_property(self):
        """[核心功能] url 属性正确拼接 host 和 port"""
        endpoint = WorkerEndpoint(
            node_id="worker-1",
            host="192.168.1.100",
            port=8080,
        )
        assert endpoint.url == "http://192.168.1.100:8080"

    def test_url_with_different_ports(self):
        """[边界值] 不同端口号正确拼接"""
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=3000)
        assert endpoint.url == "http://localhost:3000"

    def test_default_status(self):
        """[默认值] status 默认值为 healthy"""
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8080)
        assert endpoint.status == "healthy"

    def test_custom_status(self):
        """[核心功能] status 可自定义"""
        endpoint = WorkerEndpoint(
            node_id="w1",
            host="localhost",
            port=8080,
            status="unhealthy",
        )
        assert endpoint.status == "unhealthy"


# ==================== WorkerClient 测试 ====================


class TestWorkerClient:
    """WorkerClient HTTP 客户端测试"""

    @pytest.fixture
    def client(self):
        """创建 WorkerClient 实例"""
        return WorkerClient(timeout=10.0)

    @pytest.fixture
    def endpoint(self):
        """创建 WorkerEndpoint 实例"""
        return WorkerEndpoint(
            node_id="worker-1",
            host="192.168.1.100",
            port=8080,
        )

    @pytest.mark.asyncio
    async def test_inference_sends_correct_payload(self, client, endpoint):
        """[核心功能] inference 发送正确的 payload 到 Worker"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "req-123",
            "status": "success",
            "results": ["output1"],
            "latency_ms": 100,
        }
        mock_response.text = ""

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-123",
                model_name="test-model",
                prompt="Hello world",
                sampling_params={"temperature": 0.7},
            )

        assert result["status"] == "success"
        assert result["request_id"] == "req-123"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://192.168.1.100:8080/inference"
        assert call_args[1]["json"]["model_name"] == "test-model"
        assert call_args[1]["json"]["prompts"] == ["Hello world"]

    @pytest.mark.asyncio
    async def test_inference_handles_worker_error(self, client, endpoint):
        """[错误处理] Worker 返回错误时正确处理"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-123",
                model_name="test-model",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "500" in result["error"]

    @pytest.mark.asyncio
    async def test_inference_handles_timeout(self, client, endpoint):
        """[错误处理] 请求超时时正确处理"""
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-123",
                model_name="test-model",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_inference_handles_connection_error(self, client, endpoint):
        """[错误处理] 连接错误时正确处理"""
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-123",
                model_name="test-model",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_inference_uses_default_sampling_params(self, client, endpoint):
        """[核心功能] 未提供 sampling_params 时使用默认值"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "req-123",
            "status": "success",
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            await client.inference(
                endpoint=endpoint,
                request_id="req-123",
                model_name="test-model",
                prompt="Hello",
            )

        call_args = mock_client.post.call_args
        default_params = call_args[1]["json"]["sampling_params"]
        assert default_params["temperature"] == 0.7
        assert default_params["top_p"] == 0.9
        assert default_params["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_load_model_success(self, client, endpoint):
        """[核心功能] load_model 成功调用"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "model": "test-model",
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            result = await client.load_model(
                endpoint=endpoint,
                model_name="test-model",
            )

        assert result["status"] == "success"
        assert result["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_get_status_success(self, client, endpoint):
        """[核心功能] get_status 成功获取状态"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "node_id": "worker-1",
            "status": "healthy",
        }

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            result = await client.get_status(endpoint)

        assert result["status"] == "healthy"
        assert result["node_id"] == "worker-1"

    @pytest.mark.asyncio
    async def test_close_closes_client(self, client):
        """[核心功能] close 方法正确关闭客户端"""
        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        client._client = mock_client

        await client.close()

        assert client._client is None
        mock_client.aclose.assert_called_once()


# ==================== WorkerRegistry 测试 ====================


class TestWorkerRegistry:
    """Worker 注册表测试"""

    @pytest.fixture
    def registry(self):
        """创建 WorkerRegistry 实例"""
        return WorkerRegistry()

    @pytest.fixture
    def endpoint(self):
        """创建 WorkerEndpoint 实例"""
        return WorkerEndpoint(
            node_id="worker-1",
            host="192.168.1.100",
            port=8080,
        )

    @pytest.mark.asyncio
    async def test_register_adds_worker(self, registry, endpoint):
        """[核心功能] register 添加 Worker 到注册表"""
        await registry.register(endpoint)

        worker = await registry.get_worker("worker-1")
        assert worker is not None
        assert worker.node_id == "worker-1"
        assert worker.host == "192.168.1.100"
        assert worker.port == 8080

    @pytest.mark.asyncio
    async def test_unregister_removes_worker(self, registry, endpoint):
        """[核心功能] unregister 从注册表移除 Worker"""
        await registry.register(endpoint)
        await registry.unregister("worker-1")

        worker = await registry.get_worker("worker-1")
        assert worker is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_worker(self, registry):
        """[错误处理] 注销不存在的 Worker 不抛异常"""
        await registry.unregister("nonexistent")
        # 不应抛出异常

    @pytest.mark.asyncio
    async def test_get_all_workers(self, registry):
        """[核心功能] get_all_workers 返回所有已注册 Worker"""
        endpoint1 = WorkerEndpoint(node_id="w1", host="localhost", port=8000)
        endpoint2 = WorkerEndpoint(node_id="w2", host="localhost", port=8001)

        await registry.register(endpoint1)
        await registry.register(endpoint2)

        workers = await registry.get_all_workers()
        assert len(workers) == 2

    @pytest.mark.asyncio
    async def test_get_worker_count(self, registry):
        """[核心功能] get_worker_count 返回正确数量"""
        assert await registry.get_worker_count() == 0

        await registry.register(WorkerEndpoint(node_id="w1", host="localhost", port=8000))
        assert await registry.get_worker_count() == 1

        await registry.register(WorkerEndpoint(node_id="w2", host="localhost", port=8001))
        assert await registry.get_worker_count() == 2

    @pytest.mark.asyncio
    async def test_update_worker_status(self, registry, endpoint):
        """[核心功能] update_worker_status 更新 Worker 状态"""
        await registry.register(endpoint)
        await registry.update_worker_status("worker-1", "unhealthy")

        worker = await registry.get_worker("worker-1")
        assert worker.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_get_worker_not_found(self, registry):
        """[边界值] get_worker 对不存在的节点返回 None"""
        worker = await registry.get_worker("nonexistent")
        assert worker is None


class TestGetWorkerRegistry:
    """全局 WorkerRegistry 单例测试"""

    def test_get_worker_registry_returns_same_instance(self):
        """[核心功能] get_worker_registry 返回单例"""
        registry1 = get_worker_registry()
        registry2 = get_worker_registry()
        assert registry1 is registry2
