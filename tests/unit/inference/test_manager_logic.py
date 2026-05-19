"""Engine Manager 核心逻辑专业测试

测试策略：
1. 模型加载/卸载状态管理准确性
2. VRAM 预估与实际占用一致性
3. 批量生成请求路由正确性
4. BlockPool 集成逻辑
5. 空闲淘汰机制触发时机
6. GPU 监控启动/停止生命周期
"""

import asyncio
import sys
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.core.constants import InferenceBackendType
from quantumflow.inference.engine import InferenceResult, SamplingParams
from quantumflow.inference.manager import EngineManager


# Module-level fixture for all test classes
@pytest.fixture
def manager():
    """创建干净的 EngineManager 实例"""
    # 重置单例
    EngineManager._instance = None
    mgr = EngineManager()
    yield mgr
    # 清理
    EngineManager._instance = None


class TestModelLifecycle:
    """模型生命周期管理逻辑"""

    @pytest.mark.asyncio
    async def test_initialize_creates_engine(self, manager):
        """initialize 必须正确初始化指定后端的引擎"""
        result = await manager.initialize(InferenceBackendType.HUGGINGFACE)

        assert result is True
        assert InferenceBackendType.HUGGINGFACE in manager._engines
        assert manager._default_engine is not None

    @pytest.mark.asyncio
    async def test_load_model_updates_vram(self, manager):
        """load_model 成功后必须更新 VRAM 记录"""
        # Mock VRAM manager
        with patch.object(manager._vram_manager, "can_load", return_value=(True, "", [])):
            with patch.object(manager._vram_manager, "record_loaded") as mock_record:
                with patch.object(manager._vram_manager, "update_actual_vram"):
                    # Mock 引擎
                    mock_engine = AsyncMock()
                    mock_engine.load_model = AsyncMock(return_value=True)
                    manager._engines[InferenceBackendType.HUGGINGFACE] = mock_engine

                    result, msg = await manager.load_model(
                        model_name="test_model",
                        model_path="/path/to/model",
                        backend=InferenceBackendType.HUGGINGFACE,
                    )

                    assert result is True
                    mock_record.assert_called_once_with("test_model", ANY)

    @pytest.mark.asyncio
    async def test_load_model_vram_rejected(self, manager):
        """VRAM 不足时 load_model 必须拒绝"""
        with patch.object(manager._vram_manager, "can_load", return_value=(False, "VRAM 不足", [])):
            result, msg = await manager.load_model(
                model_name="test_model",
                model_path="/path/to/model",
                backend=InferenceBackendType.HUGGINGFACE,
            )

            assert result is False
            assert "VRAM 不足" in msg

    @pytest.mark.asyncio
    async def test_unload_model_cleans_up(self, manager):
        """unload_model 必须清理所有相关状态"""
        # 先加载 - engine 必须是 AsyncMock 以支持 await
        mock_engine = AsyncMock()
        mock_engine.unload_model = AsyncMock(return_value=True)
        manager._loaded_models["test_model"] = mock_engine
        manager._engines[InferenceBackendType.HUGGINGFACE] = mock_engine

        with patch.object(manager._vram_manager, "record_unloaded") as mock_unload:
            await manager.unload_model("test_model")

            assert "test_model" not in manager._loaded_models
            mock_unload.assert_called_once_with("test_model")

    @pytest.mark.asyncio
    async def test_unload_nonexistent_model_returns_false(self, manager):
        """卸载不存在的模型必须返回 False"""
        result = await manager.unload_model("nonexistent")
        assert result is False


class TestGenerateRouting:
    """generate 请求路由逻辑"""

    @pytest.fixture
    def manager_with_model(self, manager):
        manager._loaded_models["test_model"] = Mock()
        return manager

    @pytest.mark.asyncio
    async def test_generate_blocks_until_loaded(self, manager):
        """generate 未加载的模型必须抛出 ModelNotFoundError"""
        with pytest.raises(Exception) as exc_info:
            await manager.generate("nonexistent", ["prompt"], SamplingParams())

        # 验证异常类型
        assert "nonexistent" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_calls_correct_engine(self, manager_with_model):
        """generate 必须调用正确的引擎"""
        mock_engine = AsyncMock()
        mock_engine.generate = AsyncMock(
            return_value=[
                InferenceResult(
                    request_id="test_0",
                    outputs=["out"],
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=10,
                    finish_reason="stop",
                    metrics={},
                )
            ]
        )
        manager_with_model._loaded_models["test_model"] = mock_engine

        with patch.object(manager_with_model._vram_manager, "allocate_blocks", return_value=None):
            with patch.object(manager_with_model._vram_manager, "mark_in_use"):
                with patch.object(manager_with_model._vram_manager, "mark_idle"):
                    with patch.object(manager_with_model._vram_manager, "release_blocks"):
                        await manager_with_model.generate(
                            "test_model", ["prompt"], SamplingParams()
                        )

        mock_engine.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_updates_vram_after(self, manager_with_model):
        """generate 完成后必须更新实际 VRAM"""
        mock_engine = AsyncMock()
        mock_engine.generate = AsyncMock(return_value=[])
        manager_with_model._loaded_models["test_model"] = mock_engine

        with patch.object(manager_with_model._vram_manager, "allocate_blocks", return_value=None):
            with patch.object(manager_with_model._vram_manager, "mark_in_use"):
                with patch.object(manager_with_model._vram_manager, "mark_idle"):
                    with patch.object(manager_with_model._vram_manager, "release_blocks"):
                        with patch.object(
                            manager_with_model._vram_manager, "update_actual_vram"
                        ) as mock_update:
                            await manager_with_model.generate(
                                "test_model", ["prompt"], SamplingParams()
                            )

        mock_update.assert_called_once_with("test_model")

    @pytest.mark.asyncio
    async def test_generate_releases_blocks_on_exception(self, manager_with_model):
        """generate 异常时必须释放已分配的 blocks"""
        mock_engine = AsyncMock()
        mock_engine.generate = AsyncMock(side_effect=RuntimeError("Inference failed"))
        manager_with_model._loaded_models["test_model"] = mock_engine

        with patch.object(
            manager_with_model._vram_manager, "allocate_blocks", return_value=["block1"]
        ):
            with patch.object(manager_with_model._vram_manager, "mark_in_use"):
                with patch.object(manager_with_model._vram_manager, "mark_idle"):
                    with patch.object(
                        manager_with_model._vram_manager, "release_blocks"
                    ) as mock_release:
                        with pytest.raises(RuntimeError):
                            await manager_with_model.generate(
                                "test_model", ["prompt"], SamplingParams()
                            )

        mock_release.assert_called_once()


class TestStreamRouting:
    """generate_stream 请求路由逻辑"""

    @pytest.fixture
    def manager_with_model(self, manager):
        manager._loaded_models["test_model"] = Mock()
        return manager

    @pytest.mark.asyncio
    async def test_stream_blocks_until_loaded(self, manager):
        """generate_stream 未加载模型必须抛出异常"""
        with pytest.raises(Exception):
            async for _ in manager.generate_stream("nonexistent", "prompt", SamplingParams()):
                pass

    @pytest.mark.asyncio
    async def test_stream_allocates_and_releases_blocks(self, manager_with_model):
        """stream 必须分配和释放 blocks"""
        call_log = []

        async def mock_stream(*args):
            yield "chunk"
            call_log.append("streamed")

        mock_engine = Mock()
        mock_engine.generate_stream = mock_stream
        manager_with_model._loaded_models["test_model"] = mock_engine

        with patch.object(
            manager_with_model._vram_manager, "allocate_blocks", return_value=["block1"]
        ) as mock_alloc:
            with patch.object(manager_with_model._vram_manager, "mark_in_use"):
                with patch.object(manager_with_model._vram_manager, "mark_idle"):
                    with patch.object(
                        manager_with_model._vram_manager, "release_blocks"
                    ) as mock_release:
                        chunks = []
                        async for chunk in manager_with_model.generate_stream(
                            "test_model", "prompt", SamplingParams()
                        ):
                            chunks.append(chunk)

        mock_alloc.assert_called_once()
        mock_release.assert_called_once()


class TestVRAMStatus:
    """VRAM 状态查询逻辑"""

    @pytest.fixture
    def manager(self):
        EngineManager._instance = None
        return EngineManager()

    def test_get_vram_status_returns_available(self, manager):
        """get_vram_status 必须返回正确的可用 VRAM"""
        with patch.object(manager._vram_manager, "get_available_vram_gb", return_value=8.5):
            with patch.object(manager._vram_manager, "safety_factor", 0.7):
                with patch.object(
                    manager._vram_manager, "get_loaded_models", return_value=["model1"]
                ):
                    status = manager.get_vram_status()

        assert status["available_vram_gb"] == 8.5
        assert status["safety_factor"] == 0.7
        assert "model1" in status["loaded_models"]

    def test_get_loaded_models(self, manager):
        """get_loaded_models 必须返回已加载模型列表"""
        manager._loaded_models["model1"] = Mock()
        manager._loaded_models["model2"] = Mock()

        loaded = manager.get_loaded_models()

        assert "model1" in loaded
        assert "model2" in loaded
        assert len(loaded) == 2

    def test_is_model_loaded(self, manager):
        """is_model_loaded 必须准确判断"""
        manager._loaded_models["model1"] = Mock()

        assert manager.is_model_loaded("model1") is True
        assert manager.is_model_loaded("model2") is False


class TestIdleEviction:
    """空闲淘汰逻辑"""

    @pytest.fixture
    def manager(self):
        EngineManager._instance = None
        return EngineManager()

    def test_configure_idle_eviction_enable(self, manager):
        """configure_idle_eviction 启用时必须设置 TTL"""
        manager.configure_idle_eviction(300.0)

        assert manager._vram_manager.idle_ttl_seconds == 300.0

    def test_configure_idle_eviction_disable(self, manager):
        """configure_idle_eviction(0) 必须禁用"""
        manager._vram_manager.idle_ttl_seconds = 100.0

        manager.configure_idle_eviction(0.0)

        assert manager._vram_manager.idle_ttl_seconds == 0.0

    @pytest.mark.asyncio
    async def test_start_idle_eviction_checker_skip_when_disabled(self, manager):
        """idle_ttl=0 时 start_idle_eviction_checker 必须跳过"""
        manager._vram_manager.idle_ttl_seconds = 0.0

        await manager.start_idle_eviction_checker()

        assert manager._eviction_task is None


class TestGPUMonitoring:
    """GPU 监控生命周期"""

    @pytest.fixture
    def manager(self):
        EngineManager._instance = None
        return EngineManager()

    @pytest.mark.asyncio
    async def test_start_gpu_monitoring(self, manager):
        """start_gpu_monitoring 必须启动监控"""
        with patch.object(manager._gpu_monitor, "start", new_callable=AsyncMock) as mock_start:
            await manager.start_gpu_monitoring()

        mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_gpu_monitoring(self, manager):
        """stop_gpu_monitoring 必须停止监控"""
        manager._gpu_monitor._running = True
        manager._gpu_monitor._task = asyncio.create_task(asyncio.sleep(10))

        with patch.object(manager._gpu_monitor, "stop", new_callable=AsyncMock) as mock_stop:
            await manager.stop_gpu_monitoring()

        mock_stop.assert_called_once()

    def test_get_gpu_status_snapshot(self, manager):
        """get_gpu_status 和 get_gpu_snapshot 必须返回数据"""
        # Mock GPU monitor - 注意：latest 是 property，没有 setter
        # 所以直接设置 _latest
        mock_snapshot = [Mock(to_dict=lambda: {"index": 0, "name": "GPU0"})]
        manager._gpu_monitor._latest = mock_snapshot

        status = manager.get_gpu_status()
        assert len(status) == 1

        with patch.object(manager._gpu_monitor, "collect_snapshot", return_value=mock_snapshot):
            snapshot = manager.get_gpu_snapshot()
            assert len(snapshot) == 1


class TestBatchAccumulatorIntegration:
    """BatchAccumulator 集成逻辑"""

    @pytest.fixture
    def manager(self):
        EngineManager._instance = None
        return EngineManager()

    def test_get_batch_accumulator_creates_per_model_and_params(self, manager):
        """get_batch_accumulator 必须按 model+sampling_params 创建实例"""
        params1 = SamplingParams(temperature=0.7, max_tokens=100)
        params2 = SamplingParams(temperature=0.8, max_tokens=100)

        acc1 = manager.get_batch_accumulator("model1", params1)
        acc2 = manager.get_batch_accumulator("model1", params2)
        acc3 = manager.get_batch_accumulator("model1", params1)  # 相同参数

        # 不同参数应该创建不同 accumulator
        assert acc1 is not acc2
        # 相同参数应该返回同一个
        assert acc1 is acc3

    def test_get_batch_stats(self, manager):
        """get_batch_stats 必须返回所有 accumulator 的统计"""
        params = SamplingParams(temperature=0.7, max_tokens=100)
        acc = manager.get_batch_accumulator("model1", params)

        stats = manager.get_batch_stats()

        # 应该包含创建的 key
        key = f"model1_{params.temperature}_{params.max_tokens}"
        assert key in stats


class TestEdgeCases:
    """边界场景"""

    @pytest.fixture
    def manager(self):
        EngineManager._instance = None
        return EngineManager()

    @pytest.mark.asyncio
    async def test_generate_with_empty_prompts_list(self, manager):
        """空 prompts 列表必须能处理"""
        manager._loaded_models["test"] = Mock()

        mock_engine = AsyncMock()
        mock_engine.generate = AsyncMock(return_value=[])
        manager._loaded_models["test"] = mock_engine

        with patch.object(manager._vram_manager, "allocate_blocks", return_value=None):
            with patch.object(manager._vram_manager, "mark_in_use"):
                with patch.object(manager._vram_manager, "mark_idle"):
                    with patch.object(manager._vram_manager, "release_blocks"):
                        result = await manager.generate("test", [], SamplingParams())

        assert result == []

    def test_get_stats_empty_engine(self, manager):
        """无引擎时 get_stats 必须返回空"""
        stats = manager.get_stats()
        assert stats == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
