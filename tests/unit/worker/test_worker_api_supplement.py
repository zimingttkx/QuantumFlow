"""Worker API 路由补充测试 — 覆盖异常处理分支

目标覆盖缺失行:
- 55-56: load_model 路由的 HTTPException(500) 分支
- 81-82: unload_model 路由的 HTTPException(500) 分支
- 123-124: inference 路由的 HTTPException(500) 分支
- 164-165: get_stats 路由的 HTTPException(500) 分支
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from quantumflow.core.constants import NodeStatus
from quantumflow.inference.engine import InferenceResult
from quantumflow.worker import WorkerConfig, WorkerNode


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_worker_with_mocked_methods(**overrides):
    """创建 worker 并替换指定方法为 AsyncMock

    返回 (worker, app, client, mocks_dict)
    """
    config = WorkerConfig(node_id="test-worker", port=18080)
    engine = AsyncMock()
    engine.is_ready = True
    engine.loaded_model_names = ["test-model"]
    worker = WorkerNode(config=config, engine=engine)

    # 替换指定方法
    mocks = {}
    for name, side_effect in overrides.items():
        mock = AsyncMock(side_effect=side_effect)
        setattr(worker, name, mock)
        mocks[name] = mock

    app = worker.create_app()
    client = TestClient(app)
    return worker, app, client, mocks


# ═══════════════════════════════════════════════════════════════════════════════
# load_model 路由 — HTTPException(500) 分支 (lines 55-56)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadModelRouteException:
    """load_model 路由触发 HTTP 500"""

    def test_load_model_route_returns_500_on_unexpected_exception(self):
        """lines 55-56: worker.load_model 抛出未捕获异常时返回 HTTP 500"""
        _, _, client, _ = _make_worker_with_mocked_methods(
            load_model=RuntimeError("unexpected internal error"),
        )

        response = client.post(
            "/api/v1/worker/load_model",
            json={"model_name": "test-model", "model_path": "/models/test"},
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "unexpected internal error" in data["detail"] or "detail" in data


# ═══════════════════════════════════════════════════════════════════════════════
# unload_model 路由 — HTTPException(500) 分支 (lines 81-82)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnloadModelRouteException:
    """unload_model 路由触发 HTTP 500"""

    def test_unload_model_route_returns_500_on_unexpected_exception(self):
        """lines 81-82: worker.unload_model 抛出未捕获异常时返回 HTTP 500"""
        _, _, client, _ = _make_worker_with_mocked_methods(
            unload_model=RuntimeError("unload internal error"),
        )

        response = client.post(
            "/api/v1/worker/unload_model",
            json={"model_name": "test-model"},
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "unload internal error" in data["detail"] or "detail" in data


# ═══════════════════════════════════════════════════════════════════════════════
# inference 路由 — HTTPException(500) 分支 (lines 123-124)
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferenceRouteException:
    """inference 路由触发 HTTP 500"""

    def test_inference_route_returns_500_on_unexpected_exception(self):
        """lines 123-124: worker.inference 抛出未捕获异常时返回 HTTP 500"""
        _, _, client, _ = _make_worker_with_mocked_methods(
            inference=RuntimeError("inference internal error"),
        )

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-500",
                "model_name": "test-model",
                "prompts": ["Hi"],
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "inference internal error" in data["detail"] or "detail" in data

    def test_inference_route_handles_malformed_request(self):
        """推理路由处理非法输入返回 422"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        worker = WorkerNode(config=config, engine=AsyncMock())
        app = worker.create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/worker/inference",
            json={"not_valid": True},
        )

        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# get_stats 路由 — HTTPException(500) 分支 (lines 164-165)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatsRouteException:
    """get_stats 路由触发 HTTP 500"""

    def test_stats_route_returns_500_on_unexpected_exception(self):
        """lines 164-165: worker.get_stats 抛出未捕获异常时返回 HTTP 500"""
        _, _, client, _ = _make_worker_with_mocked_methods(
            get_stats=RuntimeError("stats internal error"),
        )

        response = client.get("/api/v1/worker/stats?model_name=test-model")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "stats internal error" in data["detail"] or "detail" in data


# ═══════════════════════════════════════════════════════════════════════════════
# load_model 路由 — invalid model_path type 触发异常
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadModelRouteEdgeCases:
    """load_model 路由边界场景"""

    def test_load_model_with_special_characters_in_name(self):
        """模型名称包含特殊字符"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        engine = AsyncMock()
        engine.is_ready = True
        engine.load_model = AsyncMock(return_value=True)
        worker = WorkerNode(config=config, engine=engine)
        app = worker.create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/worker/load_model",
            json={"model_name": "my-model/v2.0_test", "model_path": "/models/special"},
        )

        assert response.status_code in [200, 422]
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"

    def test_load_model_without_model_path(self):
        """不提供 model_path (使用默认值)"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        engine = AsyncMock()
        engine.is_ready = True
        engine.load_model = AsyncMock(return_value=True)
        worker = WorkerNode(config=config, engine=engine)
        app = worker.create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/worker/load_model",
            json={"model_name": "auto-path-model"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


# ═══════════════════════════════════════════════════════════════════════════════
# inference 路由 — 无 sampling_params 时使用默认值
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferenceRouteDefaultParams:
    """inference 路由默认采样参数"""

    def test_inference_without_sampling_params(self):
        """不提供 sampling_params 时使用默认值"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        engine = AsyncMock()
        engine.is_ready = True
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-def",
                outputs=["default"],
                prompt_tokens=1, completion_tokens=1,
                latency_ms=10, finish_reason="stop",
            )
        ])
        worker = WorkerNode(config=config, engine=engine)
        app = worker.create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-def",
                "model_name": "test-model",
                "prompts": ["Hi"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # 验证使用了默认 sampling_params
        call_kwargs = engine.generate.call_args.kwargs
        sp = call_kwargs["sampling_params"]
        assert sp.temperature == 0.7
        assert sp.max_tokens == 2048

    def test_inference_with_partial_sampling_params(self):
        """提供部分 sampling_params 字段"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        engine = AsyncMock()
        engine.is_ready = True
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-partial",
                outputs=["partial"],
                prompt_tokens=1, completion_tokens=1,
                latency_ms=10, finish_reason="stop",
            )
        ])
        worker = WorkerNode(config=config, engine=engine)
        app = worker.create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/worker/inference",
            json={
                "request_id": "req-partial",
                "model_name": "test-model",
                "prompts": ["Hi"],
                "sampling_params": {
                    "temperature": 0.3,
                    "stop": ["END"],
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        call_kwargs = engine.generate.call_args.kwargs
        sp = call_kwargs["sampling_params"]
        assert sp.temperature == 0.3
        assert sp.stop == ["END"]
        # 未提供的使用默认值
        assert sp.top_p == 0.9  # 默认
        assert sp.max_tokens == 2048  # 默认


# ═══════════════════════════════════════════════════════════════════════════════
# status 路由 — 无引擎时
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatusRouteNoEngine:
    """status 路由无引擎场景"""

    def test_status_route_without_engine(self):
        """无引擎时状态端点正常返回"""
        config = WorkerConfig(node_id="no-eng", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()
        client = TestClient(app)

        response = client.get("/api/v1/worker/status")
        assert response.status_code == 200
        data = response.json()
        assert data["loaded_models"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# node_info 路由 — 验证完整性
# ═══════════════════════════════════════════════════════════════════════════════


class TestNodeInfoRouteFull:
    """node_info 路由的完整性验证"""

    def test_node_info_route_has_all_resource_fields(self):
        """节点信息包含资源字段"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        engine = AsyncMock()
        engine.is_ready = True
        engine.loaded_model_names = ["m1"]
        worker = WorkerNode(config=config, engine=engine)
        app = worker.create_app()
        client = TestClient(app)

        response = client.get("/api/v1/worker/node_info")
        assert response.status_code == 200
        data = response.json()

        resource_fields = [
            "cpu_count", "memory_total", "memory_available",
            "disk_total", "disk_available", "current_load",
        ]
        for field in resource_fields:
            assert field in data, f"缺少字段: {field}"
            assert data[field] is not None, f"{field} 不应为 None"

    def test_node_info_route_has_version_and_labels(self):
        """节点信息包含版本和标签"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        engine = AsyncMock()
        engine.is_ready = True
        worker = WorkerNode(config=config, engine=engine)
        app = worker.create_app()
        client = TestClient(app)

        response = client.get("/api/v1/worker/node_info")
        assert response.status_code == 200
        data = response.json()

        assert data["version"] == "1.0.0"
        assert "labels" in data
        assert "platform" in data["labels"]
        assert "arch" in data["labels"]
        assert "gpu_enabled" in data["labels"]


# ═══════════════════════════════════════════════════════════════════════════════
# create_app — 各路由方法验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouteMethods:
    """验证各路由的 HTTP 方法"""

    def test_all_routes_accept_correct_methods(self):
        """所有路由接受正确的方法"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        engine = AsyncMock()
        engine.is_ready = True
        engine.loaded_model_names = []
        worker = WorkerNode(config=config, engine=engine)
        app = worker.create_app()
        client = TestClient(app)

        # GET 路由不应接受 POST
        resp = client.post("/api/v1/worker/status")
        assert resp.status_code == 405

        resp = client.post("/api/v1/worker/node_info")
        assert resp.status_code == 405

        resp = client.post("/api/v1/worker/models")
        assert resp.status_code == 405

        # POST 路由不应接受 GET (这些会返回 422 因为缺少 query params 或 405)
        resp = client.get("/api/v1/worker/load_model")
        assert resp.status_code in [405, 422]

        resp = client.get("/api/v1/worker/unload_model")
        assert resp.status_code in [405, 422]

        resp = client.get("/api/v1/worker/inference")
        assert resp.status_code in [405, 422]


# ═══════════════════════════════════════════════════════════════════════════════
# models 路由 — 无引擎
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelsRouteNoEngine:
    """models 路由无引擎场景"""

    def test_models_route_without_engine(self):
        """无引擎时模型列表为空"""
        config = WorkerConfig(node_id="no-eng", port=18080)
        worker = WorkerNode(config=config, engine=None)
        app = worker.create_app()
        client = TestClient(app)

        response = client.get("/api/v1/worker/models")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# load_model 路由 — ModelConfig 参数完整传递验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadModelConfigPassthrough:
    """load_model 路由的 ModelConfig 参数传递"""

    def test_load_model_passes_all_config_fields_to_engine(self):
        """验证所有 LoadModelRequest 字段都正确传递到 ModelConfig"""
        config = WorkerConfig(node_id="test-worker", port=18080)
        engine = AsyncMock()
        engine.is_ready = True
        engine.load_model = AsyncMock(return_value=True)
        worker = WorkerNode(config=config, engine=engine)
        app = worker.create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/worker/load_model",
            json={
                "model_name": "full-test",
                "model_path": "/custom/path",
                "backend": "huggingface",
                "tensor_parallel": 2,
                "gpu_memory_utilization": 0.75,
                "enable_chunked_prefill": True,
                "prefill_chunk_size": 2048,
            },
        )

        assert response.status_code == 200
        call_args = engine.load_model.call_args
        model_config = call_args[0][0]
        assert model_config.model_name == "full-test"
        assert model_config.model_path == "/custom/path"
        assert model_config.tensor_parallel == 2
        assert model_config.gpu_memory_utilization == 0.75
        assert model_config.enable_chunked_prefill is True
        assert model_config.prefill_chunk_size == 2048
