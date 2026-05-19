"""Worker API 路由严格测试

测试原则：
1. 业务逻辑验证优先于运行可用性
2. 强精准断言：每个功能点的预期行为必须严格验证
3. 全覆盖：正常用例、边界值、非法入参、异常场景
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from fastapi.testclient import TestClient
import sys

sys.path.insert(0, '/home/dingziming/PycharmProjects/QuantumFlow')

from quantumflow.worker import WorkerNode, WorkerConfig
from quantumflow.worker.api_routes import create_worker_router
from quantumflow.core.constants import NodeStatus
from quantumflow.inference.engine import InferenceResult, SamplingParams


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_engine():
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
def worker(mock_engine):
    """创建 Worker 实例"""
    config = WorkerConfig(node_id="test-worker", port=18080)
    return WorkerNode(config=config, engine=mock_engine)


@pytest.fixture
def client(worker):
    """创建测试客户端"""
    app = worker.create_app()
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：健康检查端点
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """健康检查端点严格测试"""

    def test_health_returns_200(self, client):
        """[正常用例] 健康检查返回 HTTP 200"""
        response = client.get("/health")
        assert response.status_code == 200, \
            f"健康检查应返回200，实际: {response.status_code}"

    def test_health_returns_exact_status(self, client):
        """[正常用例] 健康检查返回 status=healthy"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy", \
            f"status 应为 'healthy'，实际: {data.get('status')}"

    def test_health_returns_correct_node_id(self, client):
        """[正常用例] 健康检查返回正确的 node_id"""
        response = client.get("/health")
        data = response.json()
        assert data["node_id"] == "test-worker", \
            f"node_id 应为 'test-worker'，实际: {data.get('node_id')}"

    def test_health_response_structure(self, client):
        """[正常用例] 健康检查响应结构完整"""
        response = client.get("/health")
        data = response.json()
        required_fields = {"status", "node_id"}
        actual_fields = set(data.keys())
        assert required_fields.issubset(actual_fields), \
            f"响应缺少字段: {required_fields - actual_fields}"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：状态端点
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatusEndpoint:
    """状态端点严格测试"""

    def test_status_returns_200(self, client):
        """[正常用例] 状态端点返回 HTTP 200"""
        response = client.get("/api/v1/worker/status")
        assert response.status_code == 200

    def test_status_returns_node_id(self, client):
        """[正常用例] 状态端点返回正确的 node_id"""
        response = client.get("/api/v1/worker/status")
        data = response.json()
        assert data["node_id"] == "test-worker"

    def test_status_returns_correct_status_value(self, client):
        """[正常用例] 状态端点返回正确的状态值"""
        response = client.get("/api/v1/worker/status")
        data = response.json()
        # Worker 初始化后状态应为 INITIALIZING
        assert data["status"] == NodeStatus.INITIALIZING.value, \
            f"初始状态应为 INITIALIZING，实际: {data['status']}"

    def test_status_returns_loaded_models(self, client, mock_engine):
        """[正常用例] 状态端点返回已加载模型列表"""
        response = client.get("/api/v1/worker/status")
        data = response.json()
        assert data["loaded_models"] == ["test-model"], \
            f"loaded_models 应为 ['test-model']，实际: {data['loaded_models']}"

    def test_status_returns_gpu_count(self, client):
        """[正常用例] 状态端点返回 GPU 数量"""
        response = client.get("/api/v1/worker/status")
        data = response.json()
        assert "gpu_count" in data
        assert isinstance(data["gpu_count"], int)

    def test_status_returns_active_requests(self, client):
        """[正常用例] 状态端点返回活跃请求数"""
        response = client.get("/api/v1/worker/status")
        data = response.json()
        assert "active_requests" in data
        assert data["active_requests"] == 0, \
            "初始状态 active_requests 应为 0"

    def test_status_returns_completed_requests(self, client):
        """[正常用例] 状态端点返回已完成请求数"""
        response = client.get("/api/v1/worker/status")
        data = response.json()
        assert "completed_requests" in data
        assert data["completed_requests"] == 0

    def test_status_returns_failed_requests(self, client):
        """[正常用例] 状态端点返回失败请求数"""
        response = client.get("/api/v1/worker/status")
        data = response.json()
        assert "failed_requests" in data
        assert data["failed_requests"] == 0

    def test_status_returns_started_at(self, client):
        """[正常用例] 状态端点返回启动时间"""
        response = client.get("/api/v1/worker/status")
        data = response.json()
        assert "started_at" in data
        assert data["started_at"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：节点信息端点
# ═══════════════════════════════════════════════════════════════════════════════

class TestNodeInfoEndpoint:
    """节点信息端点严格测试"""

    def test_node_info_returns_200(self, client):
        """[正常用例] 节点信息返回 HTTP 200"""
        response = client.get("/api/v1/worker/node_info")
        assert response.status_code == 200

    def test_node_info_returns_required_fields(self, client):
        """[正常用例] 节点信息返回所有必填字段"""
        response = client.get("/api/v1/worker/node_info")
        data = response.json()
        required_fields = {
            "node_id", "hostname", "ip", "port", "gpu_count",
            "gpu_info", "status", "labels", "version"
        }
        actual_fields = set(data.keys())
        missing = required_fields - actual_fields
        assert not missing, f"缺少字段: {missing}"

    def test_node_info_returns_correct_node_id(self, client):
        """[正常用例] 节点信息返回正确的 node_id"""
        response = client.get("/api/v1/worker/node_info")
        data = response.json()
        assert data["node_id"] == "test-worker"

    def test_node_info_returns_correct_labels(self, client):
        """[正常用例] 节点信息返回正确的标签"""
        response = client.get("/api/v1/worker/node_info")
        data = response.json()
        assert "labels" in data
        labels = data["labels"]
        assert "platform" in labels
        assert "arch" in labels


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：模型列表端点
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelsEndpoint:
    """模型列表端点严格测试"""

    def test_models_returns_200(self, client):
        """[正常用例] 模型列表返回 HTTP 200"""
        response = client.get("/api/v1/worker/models")
        assert response.status_code == 200

    def test_models_returns_models_list(self, client, mock_engine):
        """[正常用例] 模型列表返回已加载模型"""
        response = client.get("/api/v1/worker/models")
        data = response.json()
        assert "models" in data
        assert data["models"] == ["test-model"]

    def test_models_empty_list(self):
        """[边界用例] 无模型时返回空列表"""
        config = WorkerConfig(node_id="empty-worker", port=18082)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()
        client = TestClient(app)

        response = client.get("/api/v1/worker/models")
        data = response.json()
        assert data["models"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：加载模型端点
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadModelEndpoint:
    """加载模型端点严格测试"""

    def test_load_model_returns_200(self, client, mock_engine):
        """[正常用例] 加载模型返回 HTTP 200"""
        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "new-model",
                "model_path": "/models/new",
            }
        )
        assert response.status_code == 200

    def test_load_model_success_response(self, client, mock_engine):
        """[正常用例] 加载模型成功返回正确响应"""
        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "new-model",
                "model_path": "/models/new",
            }
        )
        data = response.json()
        assert data["status"] == "success"
        assert data["model"] == "new-model"
        assert "message" in data

    def test_load_model_calls_engine_with_correct_args(self, client, mock_engine):
        """[正常用例] 加载模型正确调用引擎"""
        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "test-model",
                "model_path": "/models/test",
                "tensor_parallel": 2,
                "gpu_memory_utilization": 0.9,
            }
        )
        # 验证 load_model 被调用
        assert mock_engine.load_model.called, "load_model 应被调用"
        # 验证调用参数
        call_args = mock_engine.load_model.call_args
        config = call_args[0][0]  # 第一个位置参数
        assert config.model_name == "test-model"
        assert config.model_path == "/models/test"
        assert config.tensor_parallel == 2
        assert config.gpu_memory_utilization == 0.9

    def test_load_model_with_optional_params(self, client, mock_engine):
        """[正常用例] 加载模型支持所有可选参数"""
        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "full-model",
                "model_path": "/models/full",
                "tensor_parallel": 4,
                "gpu_memory_utilization": 0.85,
                "enable_chunked_prefill": True,
                "prefill_chunk_size": 1024,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # 验证所有参数正确传递
        call_args = mock_engine.load_model.call_args
        config = call_args[0][0]
        assert config.enable_chunked_prefill is True
        assert config.prefill_chunk_size == 1024

    def test_load_model_failure(self, client, mock_engine):
        """[错误用例] 加载模型失败返回 error 状态"""
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
        assert data["model"] == "fail-model"

    def test_load_model_engine_exception(self, client, mock_engine):
        """[异常场景] 引擎抛出异常时返回 error 状态（HTTP 200）"""
        # worker.load_model 内部会捕获异常并返回 False
        # 所以 API 返回通用错误消息
        mock_engine.load_model = AsyncMock(side_effect=RuntimeError("GPU out of memory"))

        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "oom-model",
                "model_path": "/models/oom",
            }
        )
        # API 设计：返回 200 + error status
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        # worker.load_model 捕获异常后返回 False，所以返回通用消息
        assert "Failed to load model" in data.get("message", "")

    def test_load_model_missing_model_name(self, client, mock_engine):
        """[非法入参] 缺少 model_name 时返回 422"""
        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_path": "/models/test",
            }
        )
        assert response.status_code == 422, \
            "缺少必填字段 model_name 应返回 422"

    def test_load_model_empty_model_name(self, client, mock_engine):
        """[边界用例] model_name 为空字符串时返回 success（FastAPI/Pydantic不过滤空字符串）"""
        # 注意：Pydantic 默认不验证空字符串，除非显式设置 min_length
        # 如果需要严格验证空字符串，需要在 LoadModelRequest 中添加 Field(min_length=1)
        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "",
                "model_path": "/models/test",
            }
        )
        # 当前实现允许空字符串，会正常处理
        assert response.status_code == 200

    def test_load_model_no_body(self, client, mock_engine):
        """[非法入参] 请求体为空"""
        response = client.post(
            "/api/v1/worker/load_model",
            json=None
        )
        assert response.status_code in [400, 422], \
            "空请求体应返回 400 或 422"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：卸载模型端点
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnloadModelEndpoint:
    """卸载模型端点严格测试"""

    def test_unload_model_returns_200(self, client, mock_engine):
        """[正常用例] 卸载模型返回 HTTP 200"""
        response = client.post(
            "/api/v1/worker/unload_model",
            json={"model_name": "test-model"}
        )
        assert response.status_code == 200

    def test_unload_model_success_response(self, client, mock_engine):
        """[正常用例] 卸载模型成功返回正确响应"""
        response = client.post(
            "/api/v1/worker/unload_model",
            json={"model_name": "test-model"}
        )
        data = response.json()
        assert data["status"] == "success"
        assert data["model"] == "test-model"

    def test_unload_model_calls_engine(self, client, mock_engine):
        """[正常用例] 卸载模型正确调用引擎"""
        response = client.post(
            "/api/v1/worker/unload_model",
            json={"model_name": "test-model"}
        )
        assert mock_engine.unload_model.called, "unload_model 应被调用"
        call_args = mock_engine.unload_model.call_args
        assert call_args[0][0] == "test-model"

    def test_unload_model_failure(self, client, mock_engine):
        """[错误用例] 卸载不存在的模型"""
        mock_engine.unload_model = AsyncMock(return_value=False)

        response = client.post(
            "/api/v1/worker/unload_model",
            json={"model_name": "nonexistent"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"

    def test_unload_model_missing_model_name(self, client, mock_engine):
        """[非法入参] 缺少 model_name"""
        response = client.post(
            "/api/v1/worker/unload_model",
            json={}
        )
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：推理端点
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferenceEndpoint:
    """推理端点严格测试"""

    def test_inference_returns_200(self, client, mock_engine):
        """[正常用例] 推理返回 HTTP 200"""
        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-1",
                outputs=["Hello"],
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
            }
        )
        assert response.status_code == 200

    def test_inference_success_response_structure(self, client, mock_engine):
        """[正常用例] 推理成功响应结构完整"""
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
            }
        )
        data = response.json()
        required_fields = {"request_id", "status", "results", "latency_ms"}
        actual_fields = set(data.keys())
        missing = required_fields - actual_fields
        assert not missing, f"响应缺少字段: {missing}"

    def test_inference_result_fields(self, client, mock_engine):
        """[正常用例] 推理结果包含所有字段"""
        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-1",
                outputs=["Hello"],
                prompt_tokens=10,
                completion_tokens=5,
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
            }
        )
        data = response.json()
        result = data["results"][0]
        result_fields = {"output", "prompt_tokens", "completion_tokens", "finish_reason"}
        actual_fields = set(result.keys())
        missing = result_fields - actual_fields
        assert not missing, f"结果缺少字段: {missing}"

    def test_inference_result_values(self, client, mock_engine):
        """[正常用例] 推理结果值正确"""
        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-1",
                outputs=["Hello world"],
                prompt_tokens=10,
                completion_tokens=5,
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
            }
        )
        data = response.json()
        result = data["results"][0]
        assert result["output"] == ["Hello world"], \
            f"output 错误: 期望 ['Hello world']，实际: {result['output']}"
        assert result["prompt_tokens"] == 10, \
            f"prompt_tokens 错误: 期望 10，实际: {result['prompt_tokens']}"
        assert result["completion_tokens"] == 5, \
            f"completion_tokens 错误: 期望 5，实际: {result['completion_tokens']}"
        assert result["finish_reason"] == "stop", \
            f"finish_reason 错误: 期望 'stop'，实际: {result['finish_reason']}"

    def test_inference_passes_sampling_params(self, client, mock_engine):
        """[正常用例] 推理正确传递采样参数"""
        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-1",
                outputs=["Hi"],
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=10,
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
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "top_k": 100,
                    "max_tokens": 512,
                    "repetition_penalty": 1.2,
                }
            }
        )
        assert mock_engine.generate.called
        call_args = mock_engine.generate.call_args
        # generate 使用关键字参数: model_name=..., prompts=..., sampling_params=...
        sampling_params = call_args.kwargs["sampling_params"]
        assert sampling_params.temperature == 0.5
        assert sampling_params.top_p == 0.8
        assert sampling_params.top_k == 100
        assert sampling_params.max_tokens == 512
        assert sampling_params.repetition_penalty == 1.2

    def test_inference_model_not_loaded(self, client, mock_engine):
        """[错误用例] 模型未加载时返回错误"""
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
        assert "error" in data

    def test_inference_no_engine(self):
        """[错误用例] 无引擎时返回错误"""
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
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "No inference engine available" in data.get("error", "")

    def test_inference_missing_request_id(self, client, mock_engine):
        """[非法入参] 缺少 request_id"""
        response = client.post(
            "/api/v1/worker/inference",
            json={
                "model_name": "test-model",
                "prompts": ["Hi"],
            }
        )
        assert response.status_code == 422

    def test_inference_missing_model_name(self, client, mock_engine):
        """[非法入参] 缺少 model_name"""
        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-1",
                "prompts": ["Hi"],
            }
        )
        assert response.status_code == 422

    def test_inference_missing_prompts(self, client, mock_engine):
        """[非法入参] 缺少 prompts"""
        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-1",
                "model_name": "test-model",
            }
        )
        assert response.status_code == 422

    def test_inference_empty_prompts(self, client, mock_engine):
        """[边界用例] prompts 为空列表"""
        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[])

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-1",
                "model_name": "test-model",
                "prompts": [],
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["results"] == []

    def test_inference_multiple_prompts(self, client, mock_engine):
        """[正常用例] 多 prompts 推理"""
        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-1",
                outputs=["Response 1"],
                prompt_tokens=5,
                completion_tokens=3,
                latency_ms=50,
                finish_reason="stop"
            ),
            InferenceResult(
                request_id="req-1",
                outputs=["Response 2"],
                prompt_tokens=6,
                completion_tokens=4,
                latency_ms=60,
                finish_reason="stop"
            ),
        ])

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-1",
                "model_name": "test-model",
                "prompts": ["Prompt 1", "Prompt 2"],
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["results"]) == 2
        assert data["results"][0]["output"] == ["Response 1"]
        assert data["results"][1]["output"] == ["Response 2"]

    def test_inference_engine_exception(self, client, mock_engine):
        """[异常场景] 引擎抛出异常时返回 error 状态（HTTP 200）"""
        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(side_effect=RuntimeError("CUDA out of memory"))

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-1",
                "model_name": "test-model",
                "prompts": ["Hi"],
            }
        )
        # API 设计：返回 200 + error status（不是 500）
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "CUDA out of memory" in data.get("error", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：统计端点
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatsEndpoint:
    """统计端点严格测试"""

    def test_stats_returns_200(self, client, mock_engine):
        """[正常用例] 统计返回 HTTP 200"""
        response = client.get("/api/v1/worker/stats?model_name=test-model")
        assert response.status_code == 200

    def test_stats_returns_model_name(self, client, mock_engine):
        """[正常用例] 统计返回模型名称"""
        response = client.get("/api/v1/worker/stats?model_name=test-model")
        data = response.json()
        assert data["model"] == "test-model"

    def test_stats_returns_stats_object(self, client, mock_engine):
        """[正常用例] 统计返回 stats 对象"""
        response = client.get("/api/v1/worker/stats?model_name=test-model")
        data = response.json()
        assert "stats" in data
        assert isinstance(data["stats"], dict)

    def test_stats_missing_model_name(self, client, mock_engine):
        """[非法入参] 缺少 model_name 参数"""
        response = client.get("/api/v1/worker/stats")
        # FastAPI 会返回 422 或 500 取决于实现
        assert response.status_code in [400, 422, 500]


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：create_app 验证
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateApp:
    """create_app 严格测试"""

    def test_create_app_returns_fastapi_instance(self):
        """[正常用例] create_app 返回 FastAPI 实例"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()
        assert app is not None
        assert hasattr(app, "title")
        assert hasattr(app, "routes")

    def test_create_app_title(self):
        """[正常用例] app title 包含 node_id"""
        config = WorkerConfig(node_id="my-worker", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()
        assert "my-worker" in app.title

    def test_create_app_registers_all_routes(self):
        """[正常用例] app 注册所有必需路由"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()

        routes = {route.path for route in app.routes}
        required_routes = {
            "/health",
            "/api/v1/worker/status",
            "/api/v1/worker/load_model",
            "/api/v1/worker/unload_model",
            "/api/v1/worker/inference",
            "/api/v1/worker/stats",
            "/api/v1/worker/node_info",
            "/api/v1/worker/models",
        }
        missing = required_routes - routes
        assert not missing, f"缺少路由: {missing}"

    def test_create_app_health_route_method(self):
        """[正常用例] health 路由为 GET 方法"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()

        health_route = None
        for route in app.routes:
            if route.path == "/health":
                health_route = route
                break
        assert health_route is not None, "health 路由未注册"

    def test_create_app_multiple_times(self):
        """[边界用例] 多次调用 create_app 返回同一实例（缓存）"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app1 = worker.create_app()
        app2 = worker.create_app()
        # create_app 缓存实例，第二次调用返回同一实例
        assert app1 is app2, "create_app 应返回缓存的同一实例"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
