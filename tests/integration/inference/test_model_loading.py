"""Real LLM Model Integration Tests

Tests with real HuggingFace models:
1. Model loading and unloading
2. Synchronous inference
3. Streaming inference
4. Model configuration
5. Engine stats

Note: These tests use CPU mode by default to avoid requiring the `accelerate` package.
To run with GPU, set TEST_MODEL_GPU=true environment variable.
"""

import asyncio
import os
import time
from collections.abc import AsyncIterator

import pytest

from quantumflow.inference.engine import InferenceResult, ModelConfig, SamplingParams


# Check if GPU tests should run
RUN_GPU_TESTS = os.environ.get("TEST_MODEL_GPU", "false").lower() == "true"


class TestModelLoading:
    """Test model loading and unloading with real models."""

    @pytest.mark.asyncio
    async def test_engine_initialize(self, huggingface_engine):
        """Test that engine initializes successfully."""
        assert huggingface_engine is not None
        assert huggingface_engine.is_ready is True
        assert huggingface_engine.backend_type.value == "huggingface"

    @pytest.mark.asyncio
    async def test_load_small_model_cpu(self, huggingface_engine, model_config_cpu):
        """Test loading a small model (tiny-gpt2) in CPU mode."""
        success = await huggingface_engine.load_model(model_config_cpu)
        if not success:
            pytest.skip(f"Failed to load model {model_config_cpu.model_name} in CPU mode - likely missing dependencies")

        assert success is True
        assert await huggingface_engine.is_model_loaded(model_config_cpu.model_name) is True

        # Cleanup
        await huggingface_engine.unload_model(model_config_cpu.model_name)

    @pytest.mark.asyncio
    async def test_load_unload_cycle_cpu(self, huggingface_engine, model_config_cpu):
        """Test loading and unloading a model in CPU mode."""
        # Load
        success = await huggingface_engine.load_model(model_config_cpu)
        if not success:
            pytest.skip(f"Failed to load model {model_config_cpu.model_name}")

        assert await huggingface_engine.is_model_loaded(model_config_cpu.model_name) is True

        # Unload
        unloaded = await huggingface_engine.unload_model(model_config_cpu.model_name)
        assert unloaded is True
        assert await huggingface_engine.is_model_loaded(model_config_cpu.model_name) is False

    @pytest.mark.asyncio
    async def test_unload_nonexistent_model(self, huggingface_engine):
        """Test unloading a model that doesn't exist."""
        result = await huggingface_engine.unload_model("nonexistent-model-xyz")
        assert result is False

    @pytest.mark.asyncio
    async def test_model_config_persistence_cpu(self, huggingface_engine, model_config_cpu):
        """Test that model config is stored correctly after loading."""
        success = await huggingface_engine.load_model(model_config_cpu)
        if not success:
            pytest.skip(f"Failed to load model {model_config_cpu.model_name}")

        stored_config = await huggingface_engine.get_model_config(model_config_cpu.model_name)
        assert stored_config is not None
        assert stored_config.model_name == model_config_cpu.model_name
        assert stored_config.model_path == model_config_cpu.model_path

        # Cleanup
        await huggingface_engine.unload_model(model_config_cpu.model_name)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests disabled (set TEST_MODEL_GPU=true to enable)")
    async def test_load_multiple_models(self, huggingface_engine, model_config):
        """Test loading multiple models sequentially (GPU only)."""
        model_config1 = model_config

        # Create a second model config with different name
        model_config2 = ModelConfig(
            model_name=f"{model_config.model_name}-copy",
            model_path=model_config.model_path,
            tensor_parallel=1,
            max_model_len=512,
            dtype="float16",
            trust_remote_code=True,
        )

        # Load first model
        success1 = await huggingface_engine.load_model(model_config1)
        assert success1 is True

        # Load second model
        success2 = await huggingface_engine.load_model(model_config2)
        # May fail due to memory, but should handle gracefully
        if success2:
            assert await huggingface_engine.is_model_loaded(model_config2.model_name) is True

        # First model should still be loaded
        assert await huggingface_engine.is_model_loaded(model_config1.model_name) is True


class TestSynchronousInference:
    """Test synchronous inference with real models."""

    @pytest.mark.asyncio
    async def test_single_prompt_inference_cpu(self, loaded_engine_cpu, model_config_cpu, sampling_params):
        """Test inference with a single prompt in CPU mode."""
        prompts = ["Hello, my name is"]

        results = await loaded_engine_cpu.generate(
            model_config_cpu.model_name,
            prompts,
            sampling_params,
        )

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, InferenceResult)
        assert result.request_id is not None
        assert len(result.outputs) > 0
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_batch_inference_cpu(self, loaded_engine_cpu, model_config_cpu, sampling_params):
        """Test batch inference with multiple prompts in CPU mode."""
        prompts = [
            "The capital of France is",
            "Once upon a time",
            "In a galaxy far, far away",
        ]

        results = await loaded_engine_cpu.generate(
            model_config_cpu.model_name,
            prompts,
            sampling_params,
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, InferenceResult)
            assert len(result.outputs) > 0
            assert "error" not in result.metrics or result.metrics.get("error") is None

    @pytest.mark.asyncio
    async def test_greedy_decoding_cpu(self, loaded_engine_cpu, model_config_cpu):
        """Test greedy decoding (temperature=0) in CPU mode."""
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy
            top_p=1.0,
            top_k=1,
            max_tokens=20,
        )

        prompts = ["The capital of France is"]

        results = await loaded_engine_cpu.generate(
            model_config_cpu.model_name,
            prompts,
            sampling_params,
        )

        assert len(results) == 1
        # With greedy, should get deterministic output
        assert len(results[0].outputs) > 0

    @pytest.mark.asyncio
    async def test_max_tokens_limit_cpu(self, loaded_engine_cpu, model_config_cpu):
        """Test that max_tokens is respected in CPU mode."""
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=5,  # Very short
        )

        prompts = ["Once upon a time in a land far away there lived a brave knight"]

        results = await loaded_engine_cpu.generate(
            model_config_cpu.model_name,
            prompts,
            sampling_params,
        )

        assert len(results) == 1
        # Completion should be limited by max_tokens
        assert results[0].completion_tokens <= 5

    @pytest.mark.asyncio
    async def test_empty_prompt_cpu(self, loaded_engine_cpu, model_config_cpu, sampling_params):
        """Test handling of empty prompt in CPU mode."""
        prompts = [""]

        results = await loaded_engine_cpu.generate(
            model_config_cpu.model_name,
            prompts,
            sampling_params,
        )

        assert len(results) == 1
        # Should handle gracefully
        assert results[0].completion_tokens >= 0

    @pytest.mark.asyncio
    async def test_very_long_prompt_cpu(self, loaded_engine_cpu, model_config_cpu):
        """Test handling of very long prompt in CPU mode."""
        # Create a very long prompt
        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        prompts = [long_text]

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=10,
        )

        results = await loaded_engine_cpu.generate(
            model_config_cpu.model_name,
            prompts,
            sampling_params,
        )

        assert len(results) == 1
        # Should handle long prompt gracefully
        assert results[0].prompt_tokens > 0


class TestStreamingInference:
    """Test streaming inference with real models."""

    @pytest.mark.asyncio
    async def test_stream_basic_cpu(self, loaded_engine_cpu, model_config_cpu, sampling_params):
        """Test basic streaming inference in CPU mode."""
        prompt = "Once upon a time"

        chunks = []
        async for chunk in loaded_engine_cpu.generate_stream(
            model_config_cpu.model_name,
            prompt,
            sampling_params,
        ):
            chunks.append(chunk)
            if len(chunks) >= 10:  # Limit for test
                break

        assert len(chunks) > 0
        # Chunks should be strings
        for chunk in chunks:
            assert isinstance(chunk, str)

    @pytest.mark.asyncio
    async def test_stream_completes_cpu(self, loaded_engine_cpu, model_config_cpu):
        """Test that streaming completes within reasonable time in CPU mode."""
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=20,
        )

        prompt = "Hello"

        start_time = time.time()
        full_output = ""
        chunk_count = 0

        async for chunk in loaded_engine_cpu.generate_stream(
            model_config_cpu.model_name,
            prompt,
            sampling_params,
        ):
            full_output += chunk
            chunk_count += 1

        elapsed = time.time() - start_time

        assert chunk_count > 0
        assert len(full_output) > 0
        # Should complete reasonably fast for small model
        assert elapsed < 120  # 120 seconds max for CPU model

    @pytest.mark.asyncio
    async def test_stream_with_greedy_cpu(self, loaded_engine_cpu, model_config_cpu):
        """Test streaming with greedy decoding in CPU mode."""
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            max_tokens=10,
        )

        prompt = "The capital of France is"

        chunks = []
        async for chunk in loaded_engine_cpu.generate_stream(
            model_config_cpu.model_name,
            prompt,
            sampling_params,
        ):
            chunks.append(chunk)

        # With greedy, should get consistent chunks
        full_output = "".join(chunks)
        assert len(full_output) > 0


class TestEngineStats:
    """Test engine statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_loaded_model_cpu(self, loaded_engine_cpu, model_config_cpu):
        """Test getting stats for a loaded model in CPU mode."""
        stats = await loaded_engine_cpu.get_stats(model_config_cpu.model_name)

        assert isinstance(stats, dict)
        # On CPU, stats may be empty or have limited info

    @pytest.mark.asyncio
    async def test_get_stats_unloaded_model(self, huggingface_engine):
        """Test getting stats for unloaded model returns empty dict."""
        stats = await huggingface_engine.get_stats("nonexistent-model-xyz")
        assert stats == {}


class TestInferenceLatency:
    """Test inference latency metrics."""

    @pytest.mark.asyncio
    async def test_single_inference_latency_cpu(self, loaded_engine_cpu, model_config_cpu, sampling_params):
        """Test that single inference completes within time limit in CPU mode."""
        prompts = ["Hello, world"]

        start_time = time.time()
        results = await loaded_engine_cpu.generate(
            model_config_cpu.model_name,
            prompts,
            sampling_params,
        )
        elapsed = time.time() - start_time

        assert len(results) == 1
        assert results[0].latency_ms > 0
        # Latency from result should be close to wall time
        assert abs(results[0].latency_ms / 1000 - elapsed) < 2.0  # Within 2 seconds

    @pytest.mark.asyncio
    async def test_batch_inference_scales_cpu(self, loaded_engine_cpu, model_config_cpu, sampling_params):
        """Test that batch inference handles multiple prompts in CPU mode."""
        prompts = [f"Prompt number {i}" for i in range(3)]

        start_time = time.time()
        results = await loaded_engine_cpu.generate(
            model_config_cpu.model_name,
            prompts,
            sampling_params,
        )
        elapsed = time.time() - start_time

        assert len(results) == 3
        # All should complete successfully
        for result in results:
            assert len(result.outputs) > 0
