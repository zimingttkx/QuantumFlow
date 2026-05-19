"""Worker API 路由测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import sys

sys.path.insert(0, '/home/dingziming/PycharmProjects/QuantumFlow')

from quantumflow.worker import WorkerNode, WorkerConfig
from quantumflow.worker.api_routes import create_worker_router
from quantumflow.core.constants import NodeStatus


class TestWorkerAPIRoutes:
    """Worker API 路由测试"""

    @pytest.fixture
    def mock_engine(self):
        """创建模拟引擎"""
        engine = AsyncMock()
        engine.is_ready = True
        engine.is_initialized = True
        engine.loaded_model_names = ["test-model"]
        engine.initialize = AsyncMock(return_value=True)
        engine.load_model = AsyncMock(return_value=True)
        engine.unload_model = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[])
        engine.get_stats = AsyncMock(return_value={"throughput": 10.5})
        engine.is_model_loaded = AsyncMock(return_value=True)
        return engine

    @pytest.fixture
    def worker(self, mock_engine):
        """创建 Worker 实例"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        return WorkerNode(config=config, engine=mock_engine)

    @pytest.fixture
    def client(self, worker):
        """创建测试客户端"""
        app = worker.create_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        """[正常用例] 健康检查端点返回正确状态"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["node_id"] == "test-worker"

    def test_status_endpoint(self, client, worker):
        """[正常用例] 状态端点返回 Worker 状态"""
        response = client.get("/api/v1/worker/status")
        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "test-worker"
        assert data["status"] == NodeStatus.INITIALIZING.value
        assert data["loaded_models"] == ["test-model"]

    def test_node_info_endpoint(self, client, worker):
        """[正常用例] 节点信息端点返回完整信息"""
        response = client.get("/api/v1/worker/node_info")
        assert response.status_code == 200
        data = response.json()
        assert "node_id" in data
        assert "hostname" in data
        assert "ip" in data
        assert "gpu_info" in data

    def test_models_endpoint(self, client, worker):
        """[正常用例] 模型列表端点返回已加载模型"""
        response = client.get("/api/v1/worker/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert data["models"] == ["test-model"]

    def test_load_model_endpoint(self, client, worker, mock_engine):
        """[正常用例] 加载模型端点正常工作"""
        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "new-model",
                "model_path": "/models/new",
                "backend": "huggingface",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model"] == "new-model"

    def test_load_model_endpoint_failure(self, client, worker, mock_engine):
        """[错误用例] 加载模型失败时返回错误状态"""
        mock_engine.load_model = AsyncMock(return_value=False)

        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "fail-model",
                "model_path": "/models/fail",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"

    def test_unload_model_endpoint(self, client, worker, mock_engine):
        """[正常用例] 卸载模型端点正常工作"""
        response = client.post(
            "/api/v1/worker/unload_model",
            json={"model_name": "test-model"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model"] == "test-model"

    def test_stats_endpoint(self, client, worker, mock_engine):
        """[正常用例] 统计端点返回模型统计"""
        response = client.get("/api/v1/worker/stats?model_name=test-model")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert "stats" in data

    def test_inference_endpoint_success(self, client, worker, mock_engine):
        """[正常用例] 推理端点成功执行推理"""
        from quantumflow.inference.engine import InferenceResult

        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-1",
                outputs=["Hello world"],
                prompt_tokens=5,
                completion_tokens=2,
                latency_ms=100,
                finish_reason="stop"
            )
        ])

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-1",
                "model_name": "test-model",
                "prompts": ["Hi"],
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req-1"
        assert data["status"] == "success"
        assert data["results"][0]["output"] == ["Hello world"]

    def test_inference_endpoint_model_not_loaded(self, client, worker, mock_engine):
        """[错误用例] 推理端点在模型未加载时返回错误"""
        mock_engine.is_model_loaded = AsyncMock(return_value=False)

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-1",
                "model_name": "unloaded-model",
                "prompts": ["Hi"],
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"

    def test_inference_endpoint_no_engine(self):
        """[错误用例] 无引擎时推理返回错误状态（HTTP 200）"""
        # 创建一个没有 engine 的 worker
        config = WorkerConfig(node_id="no-engine-worker", port=18081)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-1",
                "model_name": "test-model",
                "prompts": ["Hi"],
            }
        )
        # API 返回 200 和错误状态，因为推理请求被正确处理了，只是执行失败
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "No inference engine available" in data.get("error", "")


class TestWorkerNodeCreateApp:
    """WorkerNode.create_app() 测试"""

    def test_create_app_returns_fastapi(self):
        """[正常用例] create_app 返回 FastAPI 应用"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()

        assert app is not None
        assert app.title == f"QuantumFlow Worker - test-worker"

    def test_app_has_worker_router(self):
        """[正常用例] app 包含 worker 路由"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()

        # 检查路由是否注册
        routes = [route.path for route in app.routes]
        assert "/api/v1/worker/status" in routes
        assert "/api/v1/worker/load_model" in routes
        assert "/api/v1/worker/unload_model" in routes
        assert "/api/v1/worker/inference" in routes
        assert "/api/v1/worker/stats" in routes
        assert "/health" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
