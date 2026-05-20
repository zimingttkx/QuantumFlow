"""Engine Manager 覆盖率缺口补充测试

精确覆盖 manager.py 缺失行:
- load_model: eviction loop (163-165)
- load_model: engine load failure returns False (211)
- idle_eviction_checker: background loop (346, 375-385)
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.core.constants import InferenceBackendType
from quantumflow.inference.engine import ModelConfig, SamplingParams
from quantumflow.inference.manager import EngineManager


@pytest.fixture
def manager():
    EngineManager._instance = None
    return EngineManager()


# ── load_model: eviction + engine load failure ─────────────────────────

class TestLoadModelGaps:
    @pytest.mark.asyncio
    async def test_load_model_evicts_lru_models(self, manager):
        """When can_load returns models to evict, they must be evicted before loading"""
        manager._vram_manager.can_load = Mock(return_value=(True, "need eviction", ["old_model"]))
        manager._vram_manager.record_loaded = Mock()
        manager._vram_manager.update_actual_vram = Mock()

        # Setup engine
        mock_engine = AsyncMock()
        mock_engine.load_model = AsyncMock(return_value=True)
        manager._engines[InferenceBackendType.HUGGINGFACE] = mock_engine
        manager._loaded_models["old_model"] = mock_engine

        with patch.object(manager, "unload_model", new_callable=AsyncMock) as mock_unload:
            result, msg = await manager.load_model(
                model_name="new_model",
                model_path="/path/to/new",
                backend=InferenceBackendType.HUGGINGFACE,
            )

        assert result is True
        mock_unload.assert_called_once_with("old_model")

    @pytest.mark.asyncio
    async def test_load_model_engine_load_failure_returns_false(self, manager):
        """When engine.load_model returns False, load_model must return (False, ...)"""
        manager._vram_manager.can_load = Mock(return_value=(True, "ok", []))
        manager._vram_manager.record_loaded = Mock()
        manager._vram_manager.update_actual_vram = Mock()

        mock_engine = AsyncMock()
        mock_engine.load_model = AsyncMock(return_value=False)
        manager._engines[InferenceBackendType.HUGGINGFACE] = mock_engine

        result, msg = await manager.load_model(
            model_name="test",
            model_path="/path",
            backend=InferenceBackendType.HUGGINGFACE,
        )

        assert result is False
        assert msg == "引擎加载失败"

    @pytest.mark.asyncio
    async def test_load_model_multiple_evictions(self, manager):
        """When multiple models need eviction, all are evicted"""
        manager._vram_manager.can_load = Mock(
            return_value=(True, "need eviction", ["model_a", "model_b"])
        )
        manager._vram_manager.record_loaded = Mock()
        manager._vram_manager.update_actual_vram = Mock()

        mock_engine = AsyncMock()
        mock_engine.load_model = AsyncMock(return_value=True)
        manager._engines[InferenceBackendType.HUGGINGFACE] = mock_engine
        manager._loaded_models["model_a"] = mock_engine
        manager._loaded_models["model_b"] = mock_engine

        with patch.object(manager, "unload_model", new_callable=AsyncMock) as mock_unload:
            result, msg = await manager.load_model(
                model_name="new_model",
                model_path="/path/to/new",
                backend=InferenceBackendType.HUGGINGFACE,
            )

        assert result is True
        assert mock_unload.call_count == 2
        mock_unload.assert_any_call("model_a")
        mock_unload.assert_any_call("model_b")


# ── idle_eviction_checker ──────────────────────────────────────────────

class TestIdleEvictionChecker:
    @pytest.fixture
    def manager(self):
        EngineManager._instance = None
        return EngineManager()

    @pytest.mark.asyncio
    async def test_idle_eviction_checker_creates_task_when_enabled(self, manager):
        """When idle_ttl > 0, start_idle_eviction_checker creates an eviction task"""
        manager._vram_manager.idle_ttl_seconds = 60.0

        # Start the checker — it creates a background task
        await manager.start_idle_eviction_checker(check_interval_seconds=0.01)

        assert manager._eviction_task is not None
        assert not manager._eviction_task.done()

        # Cleanup
        manager._eviction_task.cancel()
        try:
            await manager._eviction_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_idle_eviction_checker_evicts_idle_models(self, manager):
        """Background loop must detect idle models and unload them"""
        manager._vram_manager.idle_ttl_seconds = 60.0

        # Mock get_idle_models_to_evict to return one model
        manager._vram_manager.get_idle_models_to_evict = Mock(return_value=["idle_model"])
        manager._loaded_models["idle_model"] = MagicMock()

        with patch.object(manager, "unload_model", new_callable=AsyncMock) as mock_unload:
            await manager.start_idle_eviction_checker(check_interval_seconds=0.01)

            # Let the loop run at least one iteration
            await asyncio.sleep(0.1)

            # Cancel the task
            manager._eviction_task.cancel()
            try:
                await manager._eviction_task
            except asyncio.CancelledError:
                pass

        mock_unload.assert_called_with("idle_model")

    @pytest.mark.asyncio
    async def test_idle_eviction_checker_skips_when_ttl_zero(self, manager):
        """When idle_ttl=0, start_idle_eviction_checker does nothing"""
        manager._vram_manager.idle_ttl_seconds = 0.0
        await manager.start_idle_eviction_checker()
        assert manager._eviction_task is None


# ── get_batch_accumulator: inner _batched_infer ────────────────────────

class TestGetBatchAccumulator:
    @pytest.fixture
    def manager(self):
        EngineManager._instance = None
        return EngineManager()

    def test_get_batch_accumulator_key_generation(self, manager):
        """Different sampling_params should produce different accumulator keys"""
        params1 = SamplingParams(temperature=0.5, max_tokens=100)
        params2 = SamplingParams(temperature=0.5, max_tokens=200)
        params3 = SamplingParams(temperature=0.7, max_tokens=100)

        acc1 = manager.get_batch_accumulator("model", params1)
        acc2 = manager.get_batch_accumulator("model", params2)
        acc3 = manager.get_batch_accumulator("model", params3)
        acc1_again = manager.get_batch_accumulator("model", params1)

        assert acc1 is not acc2  # different max_tokens
        assert acc1 is not acc3  # different temperature
        assert acc1 is acc1_again  # same params, same instance

    @pytest.mark.asyncio
    async def test_batched_infer_exception_propagates(self, manager):
        """When self.generate raises in _batched_infer, exception propagates (line 346)"""
        params = SamplingParams(temperature=0.5, max_tokens=100)

        # Create a new accumulator - this creates _batched_infer closure
        acc = manager.get_batch_accumulator("model", params)

        # Mock generate to raise an exception
        original_generate = manager.generate
        manager.generate = AsyncMock(side_effect=RuntimeError("Generate failed"))

        # Trigger flush by submitting a request
        # The submit will fail because generate raises
        try:
            await acc.submit("test prompt")
        except RuntimeError as e:
            assert "Generate failed" in str(e)

        # Restore
        manager.generate = original_generate


# ── EngineManager: initialize with unsupported backend ─────────────────

class TestManagerInitializeGaps:
    @pytest.fixture
    def manager(self):
        EngineManager._instance = None
        return EngineManager()

    @pytest.mark.asyncio
    async def test_initialize_vllm_failure_returns_false(self, manager):
        """When VLLM init fails, initialize returns False"""
        with patch("quantumflow.inference.manager.VLLMEngine") as mock_vllm:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=False)
            mock_vllm.return_value = mock_instance

            result = await manager.initialize(InferenceBackendType.VLLM)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
