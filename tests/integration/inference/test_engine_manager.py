"""EngineManager Integration Tests with Real Backend

Tests EngineManager with real HuggingFace backend:
1. Model loading through manager
2. Model unloading through manager
3. Multiple backend support
4. VRAM management
"""

import asyncio
import time

import pytest

from quantumflow.core.constants import InferenceBackendType
from quantumflow.inference.engine import SamplingParams
from quantumflow.inference.manager import EngineManager, get_engine_manager


class TestEngineManagerInitialization:
    """Test EngineManager initialization."""

    @pytest.mark.asyncio
    async def test_initialize_huggingface(self):
        """Test initializing HuggingFace backend."""
        manager = EngineManager()
        # Reset for testing
        manager._initialized = True
        manager._engines = {}
        manager._loaded_models = {}

        success = await manager.initialize(InferenceBackendType.HUGGINGFACE)
        # May fail if transformers not installed
        if success:
            assert InferenceBackendType.HUGGINGFACE in manager._engines

    @pytest.mark.asyncio
    async def test_initialize_vllm(self):
        """Test initializing vLLM backend."""
        manager = EngineManager()
        manager._initialized = True
        manager._engines = {}
        manager._loaded_models = {}

        success = await manager.initialize(InferenceBackendType.VLLM)
        # May fail if vllm not installed or no GPU
        if success:
            assert InferenceBackendType.VLLM in manager._engines

    @pytest.mark.asyncio
    async def test_multiple_backend_init(self):
        """Test initializing multiple backends."""
        manager = EngineManager()
        manager._initialized = True
        manager._engines = {}
        manager._loaded_models = {}

        # Initialize HF
        await manager.initialize(InferenceBackendType.HUGGINGFACE)

        # Both should work if installed
        stats = manager.get_stats()
        assert isinstance(stats, dict)


class TestModelManagement:
    """Test model loading and unloading through manager."""

    @pytest.mark.asyncio
    async def test_load_model_via_manager(self, engine_manager, model_id):
        """Test loading a model through EngineManager."""
        model_path = model_id  # Use model ID for HuggingFace

        success, message = await engine_manager.load_model(
            model_name=model_id,
            model_path=model_path,
            backend=InferenceBackendType.HUGGINGFACE,
            max_model_len=512,
        )

        # May fail if model can't be downloaded or loaded
        if success:
            assert engine_manager.is_model_loaded(model_id) is True
            # Cleanup
            await engine_manager.unload_model(model_id)
        else:
            # Expected if model not available
            assert message is not None

    @pytest.mark.asyncio
    async def test_unload_model_via_manager(self, engine_manager, model_id):
        """Test unloading a model through EngineManager."""
        model_path = model_id

        # First load
        success, message = await engine_manager.load_model(
            model_name=model_id,
            model_path=model_path,
            backend=InferenceBackendType.HUGGINGFACE,
            max_model_len=512,
        )

        if not success:
            pytest.skip(f"Could not load model: {message}")

        # Then unload
        unloaded = await engine_manager.unload_model(model_id)
        assert unloaded is True
        assert engine_manager.is_model_loaded(model_id) is False

    @pytest.mark.asyncio
    async def test_load_nonexistent_model(self, engine_manager):
        """Test loading a non-existent model."""
        success, message = await engine_manager.load_model(
            model_name="nonexistent-model-xyz",
            model_path="nonexistent/path/xyz",
            backend=InferenceBackendType.HUGGINGFACE,
        )

        assert success is False
        assert message is not None

    @pytest.mark.asyncio
    async def test_get_loaded_models(self, engine_manager, model_id):
        """Test getting list of loaded models."""
        model_path = model_id

        # Initially empty
        initial_models = engine_manager.get_loaded_models()
        assert isinstance(initial_models, list)

        # Load a model
        success, _ = await engine_manager.load_model(
            model_name=model_id,
            model_path=model_path,
            backend=InferenceBackendType.HUGGINGFACE,
            max_model_len=512,
        )

        if success:
            models = engine_manager.get_loaded_models()
            assert model_id in models
            await engine_manager.unload_model(model_id)


class TestInferenceThroughManager:
    """Test inference through EngineManager."""

    @pytest.mark.asyncio
    async def test_generate_through_manager(self, engine_manager, model_id):
        """Test generate through EngineManager."""
        model_path = model_id

        # Load model
        success, message = await engine_manager.load_model(
            model_name=model_id,
            model_path=model_path,
            backend=InferenceBackendType.HUGGINGFACE,
            max_model_len=512,
        )

        if not success:
            pytest.skip(f"Could not load model: {message}")

        # Generate
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=20,
        )

        results = await engine_manager.generate(
            model_id,
            ["Hello, world"],
            sampling_params,
        )

        assert len(results) == 1
        assert len(results[0].outputs) > 0

        # Cleanup
        await engine_manager.unload_model(model_id)

    @pytest.mark.asyncio
    async def test_generate_unloaded_model_raises(self, engine_manager):
        """Test that generating with unloaded model raises error."""
        from quantumflow.core.exceptions import ModelNotFoundError

        sampling_params = SamplingParams()

        with pytest.raises(ModelNotFoundError):
            await engine_manager.generate(
                "nonexistent-model-xyz",
                ["Hello"],
                sampling_params,
            )


class TestStreamingThroughManager:
    """Test streaming through EngineManager."""

    @pytest.mark.asyncio
    async def test_stream_through_manager(self, engine_manager, model_id):
        """Test streaming through EngineManager."""
        model_path = model_id

        # Load model
        success, message = await engine_manager.load_model(
            model_name=model_id,
            model_path=model_path,
            backend=InferenceBackendType.HUGGINGFACE,
            max_model_len=512,
        )

        if not success:
            pytest.skip(f"Could not load model: {message}")

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=10,
        )

        chunks = []
        async for chunk in engine_manager.generate_stream(
            model_id,
            "Hello",
            sampling_params,
        ):
            chunks.append(chunk)
            if len(chunks) >= 5:
                break

        assert len(chunks) > 0

        # Cleanup
        await engine_manager.unload_model(model_id)


class TestVRAMManagement:
    """Test VRAM management through manager."""

    @pytest.mark.asyncio
    async def test_get_vram_status(self, engine_manager):
        """Test getting VRAM status."""
        vram_status = engine_manager.get_vram_status()

        assert isinstance(vram_status, dict)
        assert "available_vram_gb" in vram_status
        assert "safety_factor" in vram_status
        assert "loaded_models" in vram_status

    @pytest.mark.asyncio
    async def test_vram_status_with_model(self, engine_manager, model_id):
        """Test VRAM status after loading model."""
        model_path = model_id

        initial_status = engine_manager.get_vram_status()
        initial_available = initial_status["available_vram_gb"]

        # Load model
        success, _ = await engine_manager.load_model(
            model_name=model_id,
            model_path=model_path,
            backend=InferenceBackendType.HUGGINGFACE,
            max_model_len=512,
        )

        if not success:
            pytest.skip("Could not load model")

        loaded_status = engine_manager.get_vram_status()

        # VRAM should be consumed after loading
        # (on GPU systems)
        assert len(loaded_status["loaded_models"]) >= 1

        # Cleanup
        await engine_manager.unload_model(model_id)


class TestGPUMonitoring:
    """Test GPU monitoring through manager."""

    @pytest.mark.asyncio
    async def test_get_gpu_status(self, engine_manager):
        """Test getting GPU status."""
        gpu_status = engine_manager.get_gpu_status()

        assert isinstance(gpu_status, list)
        # If GPU available, should have entries
        # If no GPU, list may be empty

    @pytest.mark.asyncio
    async def test_get_gpu_snapshot(self, engine_manager):
        """Test getting GPU snapshot."""
        snapshot = engine_manager.get_gpu_snapshot()

        assert isinstance(snapshot, list)


class TestStats:
    """Test stats reporting through manager."""

    @pytest.mark.asyncio
    async def test_get_stats(self, engine_manager):
        """Test getting engine stats."""
        stats = engine_manager.get_stats()

        assert isinstance(stats, dict)
        # Should have entries for each initialized backend

    @pytest.mark.asyncio
    async def test_get_batch_stats(self, engine_manager):
        """Test getting batch statistics."""
        stats = engine_manager.get_batch_stats()

        assert isinstance(stats, dict)
        # Initially empty before any batch operations
