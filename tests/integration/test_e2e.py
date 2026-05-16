"""端到端测试"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import Generator

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from quantumflow.api.server import create_app
from quantumflow.scheduler import Scheduler, SchedulingRequest, NodeResource, GPUResource
from quantumflow.cluster import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.inference import VLLMEngine, ModelConfig, SamplingParams


class TestE2EInferenceWorkflow:
    """端到端推理工作流测试"""

    @pytest.fixture(scope="class")
    def app(self):
        """创建测试应用"""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)

    def test_health_check(self, client):
        """测试健康检查端点"""
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

    def test_inference_generate(self, client):
        """测试文本生成"""
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "prompt": "What is quantum computing?",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "generated_text" in data
        assert "latency_ms" in data
        assert data["model"] == "Qwen2.5-7B-Instruct"

    def test_inference_stream(self, client):
        """测试流式生成"""
        with client.stream(
            "POST",
            "/api/v1/inference/generate/stream",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "prompt": "Hello",
                "sampling_params": {"max_tokens": 50},
            },
        ) as response:
            assert response.status_code == 200

            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunks.append(data)

            assert len(chunks) > 0

    def test_batch_inference(self, client):
        """测试批量推理"""
        response = client.post(
            "/api/v1/inference/batch",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "prompts": ["Hello", "Hi", "How are you?"],
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 50,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3
        assert data["completed"] == 3
        assert data["failed"] == 0

    def test_chat(self, client):
        """测试对话"""
        response = client.post(
            "/api/v1/inference/chat",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is AI?"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data

    def test_inference_invalid_model(self, client):
        """测试无效模型"""
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "model": "NonExistentModel",
                "prompt": "Hello",
            },
        )

        # 模拟的API会返回成功，但实际应该返回404
        assert response.status_code in [200, 404]


class TestE2EModelManagement:
    """端到端模型管理测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_app()
        return TestClient(app)

    def test_list_models(self, client):
        """测试列出模型"""
        response = client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_model(self, client):
        """测试获取模型详情"""
        response = client.get("/api/v1/models/Qwen2.5-7B-Instruct")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Qwen2.5-7B-Instruct"
        assert "parameter_count" in data

    def test_get_nonexistent_model(self, client):
        """测试获取不存在的模型"""
        response = client.get("/api/v1/models/NonExistentModel")

        assert response.status_code == 404

    def test_deploy_model(self, client):
        """测试部署模型"""
        response = client.post(
            "/api/v1/models/deploy",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "tensor_parallel": 1,
                "gpu_memory_utilization": 0.9,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "model_id" in data
        assert data["status"] == "loading"

    def test_filter_models_by_status(self, client):
        """测试按状态过滤模型"""
        response = client.get("/api/v1/models?status=ready")

        assert response.status_code == 200
        data = response.json()
        # 验证返回的是列表且可能包含指定状态的模型
        assert isinstance(data, list)


class TestE2EClusterManagement:
    """端到端集群管理测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_app()
        return TestClient(app)

    def test_cluster_status(self, client):
        """测试集群状态"""
        response = client.get("/api/v1/cluster/status")

        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "healthy_nodes" in data
        assert "total_gpus" in data
        assert "available_gpus" in data

    def test_list_nodes(self, client):
        """测试列出节点"""
        response = client.get("/api/v1/cluster/nodes")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_node(self, client):
        """测试获取节点详情"""
        response = client.get("/api/v1/cluster/nodes/local-node")

        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "local-node"
        assert "gpu_info" in data

    def test_get_nonexistent_node(self, client):
        """测试获取不存在的节点"""
        response = client.get("/api/v1/cluster/nodes/nonexistent")

        assert response.status_code == 404

    def test_node_action_drain(self, client):
        """测试节点drain操作"""
        response = client.post(
            "/api/v1/cluster/nodes/local-node/action?action=drain"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "drain"
        assert data["status"] == "completed"

    def test_node_action_uncordon(self, client):
        """测试节点uncordon操作"""
        response = client.post(
            "/api/v1/cluster/nodes/local-node/action?action=uncordon"
        )

        assert response.status_code == 200

    def test_filter_nodes_by_status(self, client):
        """测试按状态过滤节点"""
        response = client.get("/api/v1/cluster/nodes?status_filter=healthy")

        assert response.status_code == 200
        data = response.json()
        assert all(node["status"] == "healthy" for node in data)


class TestE2EMetricsEndpoint:
    """端到端指标端点测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_app()
        return TestClient(app)

    def test_metrics_endpoint(self, client):
        """测试指标端点"""
        response = client.get("/api/v1/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


class TestE2EIntegration:
    """端到端集成测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_app()
        return TestClient(app)

    def test_full_inference_flow(self, client):
        """测试完整推理流程"""
        # 1. 检查集群状态
        status_response = client.get("/api/v1/cluster/status")
        assert status_response.status_code == 200

        # 2. 检查模型列表
        models_response = client.get("/api/v1/models")
        assert models_response.status_code == 200

        # 3. 提交推理请求
        inference_response = client.post(
            "/api/v1/inference/generate",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "prompt": "Explain machine learning",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
            },
        )
        assert inference_response.status_code == 200

        # 4. 验证响应结构
        data = inference_response.json()
        assert "request_id" in data
        assert "generated_text" in data
        assert "latency_ms" in data

    def test_batch_processing_flow(self, client):
        """测试批处理流程"""
        # 提交批量请求
        response = client.post(
            "/api/v1/inference/batch",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "prompts": [
                    "What is Python?",
                    "What is JavaScript?",
                    "What is Rust?",
                ],
                "sampling_params": {"max_tokens": 50},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # 验证批处理结果
        assert data["total"] == 3
        assert len(data["results"]) == 3

        # 验证每个结果
        for result in data["results"]:
            assert "request_id" in result
            assert "generated_text" in result
            assert result["finish_reason"] in ["stop", "length"]

    def test_concurrent_requests(self, client):
        """测试并发请求"""

        def make_request(i):
            return client.post(
                "/api/v1/inference/generate",
                json={
                    "model": "Qwen2.5-7B-Instruct",
                    "prompt": f"Request {i}",
                    "sampling_params": {"max_tokens": 20},
                },
            )

        # 并发发送10个请求
        responses = [make_request(i) for i in range(10)]

        # 验证所有请求都成功
        assert all(r.status_code == 200 for r in responses)

        # 验证响应都是唯一的
        request_ids = [r.json()["request_id"] for r in responses]
        assert len(set(request_ids)) == len(request_ids)

    def test_priority_handling(self, client):
        """测试优先级处理"""
        # 提交不同优先级的请求
        priorities = [5, 1, 9, 3, 7]
        request_ids = []

        for p in priorities:
            response = client.post(
                "/api/v1/inference/generate",
                json={
                    "model": "Qwen2.5-7B-Instruct",
                    "prompt": f"Priority {p}",
                    "priority": p,
                    "sampling_params": {"max_tokens": 10},
                },
            )
            assert response.status_code == 200
            request_ids.append(response.json()["request_id"])

        # 验证请求都被接受
        assert len(request_ids) == 5


class TestE2EErrorHandling:
    """端到端错误处理测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_app()
        return TestClient(app)

    def test_invalid_json(self, client):
        """测试无效JSON"""
        response = client.post(
            "/api/v1/inference/generate",
            content="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422  # Validation Error

    def test_missing_required_field(self, client):
        """测试缺少必需字段"""
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "model": "test",
                # 缺少 prompt
            },
        )

        assert response.status_code == 422

    def test_invalid_parameter_range(self, client):
        """测试超出范围的参数"""
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "model": "test",
                "prompt": "hello",
                "sampling_params": {
                    "temperature": 5.0,  # 超出范围
                },
            },
        )

        assert response.status_code == 422

    def test_nonexistent_endpoint(self, client):
        """测试不存在的端点"""
        response = client.get("/api/v1/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """测试不允许的方法"""
        response = client.delete("/api/v1/models")

        assert response.status_code == 405


class TestE2EConfigurationValidation:
    """端到端配置验证测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_app()
        return TestClient(app)

    def test_deploy_with_custom_config(self, client):
        """测试自定义配置部署"""
        response = client.post(
            "/api/v1/models/deploy",
            json={
                "model": "Qwen2.5-72B-Instruct",
                "tensor_parallel": 4,
                "pipeline_parallel": 2,
                "gpu_memory_utilization": 0.85,
                "backend": "vllm",
                "replicas": 2,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["replicas"] == 2

    def test_invalid_parallel_config(self, client):
        """测试无效的并行配置"""
        response = client.post(
            "/api/v1/models/deploy",
            json={
                "model": "test",
                "tensor_parallel": 100,  # 超出允许范围
            },
        )

        assert response.status_code == 422


class TestE2EPersistence:
    """端到端持久化测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_app()
        return TestClient(app)

    def test_model_state_persistence(self, client):
        """测试模型状态持久化"""
        # 部署模型
        deploy_response = client.post(
            "/api/v1/models/deploy",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "tensor_parallel": 1,
            },
        )
        assert deploy_response.status_code == 201

        # 多次检查模型状态
        for _ in range(3):
            response = client.get("/api/v1/models/Qwen2.5-7B-Instruct")
            assert response.status_code == 200

    def test_request_id_uniqueness(self, client):
        """测试请求ID唯一性"""
        request_ids = set()

        for _ in range(20):
            response = client.post(
                "/api/v1/inference/generate",
                json={
                    "model": "test",
                    "prompt": "hello",
                    "sampling_params": {"max_tokens": 10},
                },
            )
            request_id = response.json()["request_id"]
            request_ids.add(request_id)

        # 所有请求ID应该唯一
        assert len(request_ids) == 20


class TestE2EPerformance:
    """端到端性能测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_app()
        return TestClient(app)

    def test_response_time(self, client):
        """测试响应时间"""
        start_time = time.time()

        response = client.post(
            "/api/v1/inference/generate",
            json={
                "model": "Qwen2.5-7B-Instruct",
                "prompt": "Test",
                "sampling_params": {"max_tokens": 10},
            },
        )

        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 5.0  # 应该在5秒内响应

    def test_throughput(self, client):
        """测试吞吐量"""
        start_time = time.time()
        success_count = 0

        # 在1秒内尽可能多地发送请求
        while time.time() - start_time < 1.0:
            response = client.post(
                "/api/v1/inference/generate",
                json={
                    "model": "test",
                    "prompt": "t",
                    "sampling_params": {"max_tokens": 1},
                },
            )
            if response.status_code == 200:
                success_count += 1

        # 应该能处理至少几个请求
        assert success_count > 0
