"""集成测试"""

import pytest
import asyncio
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

    def test_get_node(self, client):
        """测试获取节点信息"""
        response = client.get("/api/v1/cluster/nodes/node-1")
        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "node-1"
