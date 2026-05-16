"""Worker测试"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from quantumflow.worker import WorkerNode, WorkerConfig
from quantumflow.inference.engine import ModelConfig, SamplingParams
from quantumflow.core.constants import NodeStatus


class TestWorkerConfig:
    """WorkerConfig测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = WorkerConfig()
        assert config.node_id.startswith("worker-")
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.heartbeat_interval == 5
        assert config.gpu_enabled is True
        assert config.max_concurrent_requests == 100

    def test_custom_config(self):
        """测试自定义配置"""
        config = WorkerConfig(
            node_id="test-worker",
            host="127.0.0.1",
            port=9090,
            heartbeat_interval=10,
        )
        assert config.node_id == "test-worker"
        assert config.host == "127.0.0.1"
        assert config.port == 9090
        assert config.heartbeat_interval == 10

    def test_config_gpu_disabled(self):
        """测试禁用GPU配置"""
        config = WorkerConfig(gpu_enabled=False)
        assert config.gpu_enabled is False

    def test_config_max_concurrent_requests(self):
        """测试最大并发请求数配置"""
        config = WorkerConfig(max_concurrent_requests=500)
        assert config.max_concurrent_requests == 500


class TestWorkerNode:
    """WorkerNode测试"""

    @pytest.fixture
    def mock_engine(self):
        """创建模拟引擎"""
        engine = AsyncMock()
        engine.is_ready = False
        engine.is_initialized = False
        engine.loaded_model_names = []
        engine.initialize = AsyncMock(return_value=True)
        engine.load_model = AsyncMock(return_value=True)
        engine.unload_model = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[])
        engine.get_stats = AsyncMock(return_value={})
        return engine

    @pytest.fixture
    def worker(self, mock_engine):
        """创建Worker实例"""
        config = WorkerConfig(node_id="test-worker")
        return WorkerNode(config=config, engine=mock_engine)

    def test_worker_creation(self, worker):
        """测试Worker创建"""
        assert worker.config.node_id == "test-worker"
        assert worker.status == NodeStatus.INITIALIZING
        assert len(worker.active_requests) == 0
        assert worker.completed_requests == 0
        assert worker.failed_requests == 0

    def test_node_info_initial(self, worker):
        """测试初始节点信息"""
        info = worker.node_info
        assert info["node_id"] == "test-worker"
        assert "hostname" in info
        assert "ip" in info
        assert "gpu_count" in info
        assert info["status"] == NodeStatus.INITIALIZING.value

    def test_node_info_healthy(self, worker):
        """测试健康节点信息"""
        worker.status = NodeStatus.HEALTHY
        info = worker.node_info
        assert info["status"] == NodeStatus.HEALTHY.value

    @pytest.mark.asyncio
    async def test_start_worker(self, worker, mock_engine):
        """测试启动Worker"""
        await worker.start()

        assert worker.status == NodeStatus.HEALTHY
        assert worker._running is True
        mock_engine.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_worker_without_engine(self):
        """测试无引擎启动Worker"""
        config = WorkerConfig(node_id="no-engine-worker")
        worker = WorkerNode(config=config, engine=None)

        await worker.start()

        assert worker.status == NodeStatus.HEALTHY
        assert worker._running is True

    @pytest.mark.asyncio
    async def test_stop_worker(self, worker):
        """测试停止Worker"""
        await worker.start()
        await worker.stop()

        assert worker.status == NodeStatus.OFFLINE
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_stop_already_stopped_worker(self, worker):
        """测试停止已停止的Worker"""
        worker._running = False
        await worker.stop()

        assert worker.status == NodeStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_load_model(self, worker, mock_engine):
        """测试加载模型"""
        config = ModelConfig(
            model_name="test-model",
            model_path="/models/test",
        )

        result = await worker.load_model(config)

        assert result is True
        mock_engine.load_model.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_load_model_failure(self, worker, mock_engine):
        """测试加载模型失败"""
        mock_engine.load_model = AsyncMock(return_value=False)

        config = ModelConfig(
            model_name="test-model",
            model_path="/models/test",
        )

        result = await worker.load_model(config)
        assert result is False

    @pytest.mark.asyncio
    async def test_load_model_no_engine(self):
        """测试无引擎时加载模型"""
        config = WorkerConfig(node_id="test")
        worker = WorkerNode(config=config, engine=None)

        model_config = ModelConfig(
            model_name="test-model",
            model_path="/models/test",
        )

        result = await worker.load_model(model_config)
        assert result is False

    @pytest.mark.asyncio
    async def test_unload_model(self, worker, mock_engine):
        """测试卸载模型"""
        mock_engine.loaded_model_names = ["test-model"]

        result = await worker.unload_model("test-model")

        assert result is True
        mock_engine.unload_model.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_unload_nonexistent_model(self, worker, mock_engine):
        """测试卸载不存在的模型"""
        result = await worker.unload_model("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_inference_success(self, worker, mock_engine):
        """测试成功推理"""
        from quantumflow.inference.engine import InferenceResult

        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-1",
                outputs=["Hello"],
                prompt_tokens=5,
                completion_tokens=3,
                latency_ms=100,
                finish_reason="stop",
            )
        ])

        result = await worker.inference(
            request_id="req-1",
            model_name="test-model",
            prompts=["Hi"],
            sampling_params=SamplingParams(max_tokens=100),
        )

        assert result["status"] == "success"
        assert "results" in result
        assert worker.completed_requests == 1

    @pytest.mark.asyncio
    async def test_inference_no_engine(self, worker):
        """测试无引擎时的推理"""
        worker.engine = None

        result = await worker.inference(
            request_id="req-1",
            model_name="test-model",
            prompts=["Hi"],
            sampling_params=SamplingParams(),
        )

        assert result["status"] == "error"
        assert "No inference engine" in result["error"]
        assert worker.failed_requests == 1

    @pytest.mark.asyncio
    async def test_inference_model_not_loaded(self, worker, mock_engine):
        """测试模型未加载时的推理"""
        mock_engine.is_model_loaded = AsyncMock(return_value=False)

        result = await worker.inference(
            request_id="req-1",
            model_name="test-model",
            prompts=["Hi"],
            sampling_params=SamplingParams(),
        )

        assert result["status"] == "error"
        assert "not loaded" in result["error"]
        assert worker.failed_requests == 1

    @pytest.mark.asyncio
    async def test_inference_request_tracking(self, worker, mock_engine):
        """测试请求跟踪"""
        from quantumflow.inference.engine import InferenceResult

        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-track",
                outputs=["Test"],
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=10,
                finish_reason="stop",
            )
        ])

        assert len(worker.active_requests) == 0

        task = asyncio.create_task(worker.inference(
            request_id="req-track",
            model_name="test-model",
            prompts=["Test"],
            sampling_params=SamplingParams(),
        ))

        # 等待一小段时间让请求被跟踪
        await asyncio.sleep(0.01)
        assert "req-track" in worker.active_requests

        await task

        assert "req-track" not in worker.active_requests

    @pytest.mark.asyncio
    async def test_get_stats(self, worker, mock_engine):
        """测试获取统计"""
        mock_engine.get_stats = AsyncMock(return_value={
            "num_requests": 100,
            "throughput": 50.0,
        })

        stats = await worker.get_stats("test-model")

        assert "num_requests" in stats
        assert "active_requests" in stats
        assert stats["active_requests"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_no_engine(self, worker):
        """测试无引擎时获取统计"""
        worker.engine = None

        stats = await worker.get_stats("test-model")
        assert stats == {}

    def test_get_labels(self, worker):
        """测试获取节点标签"""
        labels = worker._get_labels()

        assert "platform" in labels
        assert "arch" in labels
        assert "gpu_enabled" in labels

    def test_get_load(self, worker):
        """测试获取负载"""
        load = worker._get_load()
        assert isinstance(load, float)
        assert load >= 0


class TestWorkerIntegration:
    """Worker集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """测试完整工作流"""
        from quantumflow.inference.engine import InferenceResult

        engine = AsyncMock()
        engine.is_ready = True
        engine.initialize = AsyncMock(return_value=True)
        engine.load_model = AsyncMock(return_value=True)
        engine.unload_model = AsyncMock(return_value=True)
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-1",
                outputs=["Test output"],
                prompt_tokens=5,
                completion_tokens=5,
                latency_ms=50,
                finish_reason="stop",
            )
        ])

        config = WorkerConfig(node_id="integration-test")
        worker = WorkerNode(config=config, engine=engine)

        await worker.start()
        assert worker.status == NodeStatus.HEALTHY

        model_config = ModelConfig(
            model_name="test-model",
            model_path="/models/test",
        )
        await worker.load_model(model_config)

        result = await worker.inference(
            request_id="req-1",
            model_name="test-model",
            prompts=["Test prompt"],
            sampling_params=SamplingParams(max_tokens=100),
        )

        assert result["status"] == "success"
        assert worker.completed_requests == 1

        await worker.unload_model("test-model")
        await worker.stop()

        assert worker.status == NodeStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_multiple_models_workflow(self):
        """测试多模型工作流"""
        from quantumflow.inference.engine import InferenceResult

        engine = AsyncMock()
        engine.is_ready = True
        engine.initialize = AsyncMock(return_value=True)
        engine.loaded_model_names = []

        def mock_load(config):
            engine.loaded_model_names.append(config.model_name)
            return True

        def mock_unload(name):
            if name in engine.loaded_model_names:
                engine.loaded_model_names.remove(name)
            return True

        def mock_is_loaded(name):
            return name in engine.loaded_model_names

        engine.load_model = AsyncMock(side_effect=mock_load)
        engine.unload_model = AsyncMock(side_effect=mock_unload)
        engine.is_model_loaded = AsyncMock(side_effect=mock_is_loaded)
        engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req",
                outputs=["output"],
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=10,
                finish_reason="stop",
            )
        ])

        config = WorkerConfig(node_id="multi-model-test")
        worker = WorkerNode(config=config, engine=engine)

        await worker.start()

        models = ["model-a", "model-b", "model-c"]
        for model in models:
            await worker.load_model(ModelConfig(
                model_name=model,
                model_path=f"/models/{model}",
            ))

        assert len(engine.loaded_model_names) == 3

        for model in models:
            await worker.unload_model(model)

        assert len(engine.loaded_model_names) == 0

        await worker.stop()

    @pytest.mark.asyncio
    async def test_concurrent_inference(self):
        """测试并发推理"""
        from quantumflow.inference.engine import InferenceResult

        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # 模拟处理时间
            return [InferenceResult(
                request_id="concurrent",
                outputs=[f"output-{call_count}"],
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=50,
                finish_reason="stop",
            )]

        engine = AsyncMock()
        engine.is_ready = True
        engine.initialize = AsyncMock(return_value=True)
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = mock_generate

        config = WorkerConfig(node_id="concurrent-test")
        worker = WorkerNode(config=config, engine=engine)

        await worker.start()

        tasks = [
            worker.inference(
                request_id=f"req-{i}",
                model_name="test-model",
                prompts=[f"prompt-{i}"],
                sampling_params=SamplingParams(max_tokens=50),
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert all(r["status"] == "success" for r in results)
        assert call_count == 5

        await worker.stop()
