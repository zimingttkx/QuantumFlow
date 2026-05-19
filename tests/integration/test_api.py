"""集成测试"""

import pytest
from fastapi.testclient import TestClient

from quantumflow.api.server import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestHealthEndpoint:
    """健康检查接口测试"""

    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_ready_check(self, client):
        """测试就绪检查"""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_live_check(self, client):
        """测试存活检查"""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["alive"] is True


class TestInferenceEndpoint:
    """推理接口测试"""

    def test_generate(self, client):
        """测试文本生成"""
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "prompt": "Hello, how are you?",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 50,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "generated_text" in data

    def test_batch_generate(self, client):
        """测试批量生成"""
        response = client.post(
            "/api/v1/inference/batch",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "prompts": ["Hello", "Hi there"],
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 50,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2


class TestModelsEndpoint:
    """模型管理接口测试"""

    def test_list_models(self, client):
        """测试列出模型"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_model(self, client):
        """测试获取模型信息"""
        response = client.get("/api/v1/models/Qwen2.5-7B-Instruct")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Qwen2.5-7B-Instruct"

    def test_get_nonexistent_model(self, client):
        """测试获取不存在的模型"""
        response = client.get("/api/v1/models/NonexistentModel")
        assert response.status_code == 404


class TestClusterEndpoint:
    """集群管理接口测试"""

    @pytest.fixture
    def registered_node(self, client):
        """注册一个测试节点"""
        # 通过心跳接口注册节点
        node_info = {
            "node_id": "local-node",
            "hostname": "test-host",
            "ip": "127.0.0.1",
            "port": 8000,
            "gpu_count": 1,
            "gpu_info": [
                {
                    "gpu_id": 0,
                    "name": "Test GPU",
                    "memory_total": 16 * 1024**3,
                    "memory_used": 4 * 1024**3,
                    "utilization": 0.3,
                    "temperature": 45.0,
                }
            ],
            "status": "healthy",
            "cpu_count": 8,
            "memory_total": 32 * 1024**3,
            "memory_available": 16 * 1024**3,
            "disk_total": 512 * 1024**3,
            "disk_available": 256 * 1024**3,
            "current_load": 0.5,
            "labels": {"platform": "Linux", "host": "local"},
            "version": "1.0.0",
            "loaded_models": [],
        }
        client.post("/api/v1/cluster/heartbeat", json=node_info)
        return "local-node"

    def test_cluster_status(self, client):
        """测试集群状态"""
        response = client.get("/api/v1/cluster/status")
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "healthy_nodes" in data

    def test_list_nodes(self, client):
        """测试列出节点"""
        response = client.get("/api/v1/cluster/nodes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_node(self, client, registered_node):
        """测试获取节点信息"""
        response = client.get("/api/v1/cluster/nodes/local-node")
        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "local-node"
