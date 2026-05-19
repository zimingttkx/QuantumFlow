"""Worker测试"""

from unittest.mock import AsyncMock

import pytest

from quantumflow.core.constants import NodeStatus
from quantumflow.inference.engine import ModelConfig, SamplingParams
from quantumflow.worker import WorkerConfig, WorkerNode


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

    @pytest.mark.asyncio
    async def test_start_worker(self, worker, mock_engine):
        """测试启动Worker"""
        await worker.start()
        assert worker.status == NodeStatus.HEALTHY
        assert worker._running is True
        mock_engine.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_worker(self, worker):
        """测试停止Worker"""
        await worker.start()
        await worker.stop()
        assert worker.status == NodeStatus.OFFLINE
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_load_model(self, worker, mock_engine):
        """测试加载模型"""
        config = ModelConfig(model_name="test-model", model_path="/models/test")
        result = await worker.load_model(config)
        assert result is True
        mock_engine.load_model.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_inference_success(self, worker, mock_engine):
        """测试成功推理"""
        from quantumflow.inference.engine import InferenceResult

        mock_engine.is_model_loaded = AsyncMock(return_value=True)
        mock_engine.generate = AsyncMock(
            return_value=[
                InferenceResult(
                    request_id="req-1",
                    outputs=["Hello"],
                    prompt_tokens=5,
                    completion_tokens=3,
                    latency_ms=100,
                    finish_reason="stop",
                )
            ]
        )
        result = await worker.inference(
            request_id="req-1",
            model_name="test-model",
            prompts=["Hi"],
            sampling_params=SamplingParams(max_tokens=100),
        )
        assert result["status"] == "success"
        assert worker.completed_requests == 1

    @pytest.mark.asyncio
    async def test_inference_model_not_loaded(self, worker, mock_engine):
        """测试模型未加载"""
        mock_engine.is_model_loaded = AsyncMock(return_value=False)
        result = await worker.inference(
            request_id="req-1",
            model_name="test-model",
            prompts=["Hi"],
            sampling_params=SamplingParams(),
        )
        assert result["status"] == "error"
        assert worker.failed_requests == 1
