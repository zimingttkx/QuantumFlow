"""Integration Test Fixtures for LLM Inference Module

Provides fixtures for integration testing with real models:
- Real HuggingFace engine with small test models
- Model loading/unloading tests
- Inference tests with streaming support
"""

import asyncio
import os
from typing import Optional

import pytest

from quantumflow.inference.backends.huggingface import HuggingFaceEngine
from quantumflow.inference.engine import ModelConfig, SamplingParams
from quantumflow.inference.manager import EngineManager


# ==============================================================================
# Model Configuration
# ==============================================================================

# Small test model for fast integration tests
# These models are small enough to download quickly and test inference
SMALL_TEST_MODEL_ID = "gpt2"  # ~124M parameters, ~500MB
TINY_TEST_MODEL_ID = "sshleifer/tiny-gpt2"  # ~14M parameters, ~60MB

# Test model path - can be overridden by environment
TEST_MODEL_ID = os.environ.get("TEST_MODEL_ID", TINY_TEST_MODEL_ID)


def get_test_model_path() -> str:
    """Get the test model path from environment or use default."""
    # For integration tests, we use model_id as path (HuggingFace downloads automatically)
    model_path = os.environ.get("TEST_MODEL_PATH")
    if model_path:
        return model_path
    # Use the model_id directly - HuggingFaceEngine will handle download
    return TEST_MODEL_ID


def is_model_available(model_id: str) -> bool:
    """Check if a model is available or can be downloaded."""
    # Check if model is already cached
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        # Check for model in cache
        for hub_dir in cache_dir.glob("hub/*"):
            if hub_dir.name.startswith("models--"):
                model_name = hub_dir.name.replace("models--", "").replace("--", "/")
                if model_name == model_id:
                    return True
    return True  # Assume available if not cached (will be downloaded)


# Skip marker for when no real model is available
requires_real_model = pytest.mark.skipif(
    not is_model_available(TEST_MODEL_ID),
    reason=f"Real model {TEST_MODEL_ID} not available"
)


# ==============================================================================
# Engine Fixtures
# ==============================================================================

@pytest.fixture
def model_id():
    """Get the test model ID."""
    return TEST_MODEL_ID


@pytest.fixture
def model_path():
    """Get the test model path."""
    return get_test_model_path()


@pytest.fixture
def sampling_params():
    """Create default sampling parameters for testing."""
    return SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=50,  # Small for fast tests
        repetition_penalty=1.0,
    )


@pytest.fixture
async def huggingface_engine():
    """Create and initialize a HuggingFace engine."""
    engine = HuggingFaceEngine()
    success = await engine.initialize()
    if not success:
        pytest.skip("HuggingFace engine initialization failed")

    yield engine

    # Cleanup
    for model_name in list(engine._loaded_models.keys()):
        await engine.unload_model(model_name)


@pytest.fixture
def model_config_cpu(model_path, model_id):
    """Create a CPU model configuration for testing (no GPU required)."""
    return ModelConfig(
        model_name=model_id,
        model_path=model_path,
        tensor_parallel=1,
        pipeline_parallel=1,
        gpu_memory_utilization=0.8,
        max_model_len=512,
        dtype="cpu",  # CPU mode - no accelerate required
        trust_remote_code=True,
        torch_compile=False,
    )


@pytest.fixture
async def loaded_engine_cpu(huggingface_engine, model_config_cpu):
    """Create an engine with a model loaded in CPU mode (no accelerate required)."""
    success = await huggingface_engine.load_model(model_config_cpu)
    if not success:
        pytest.skip(f"Failed to load model {model_config_cpu.model_name} in CPU mode")

    yield huggingface_engine

    # Cleanup
    await huggingface_engine.unload_model(model_config_cpu.model_name)


@pytest.fixture
async def model_config(model_path, model_id):
    """Create a model configuration for testing."""
    return ModelConfig(
        model_name=model_id,
        model_path=model_path,
        tensor_parallel=1,
        pipeline_parallel=1,
        gpu_memory_utilization=0.8,
        max_model_len=512,
        dtype="float16",
        trust_remote_code=True,
        torch_compile=False,  # Disable for faster test startup
    )


@pytest.fixture
async def loaded_engine(huggingface_engine, model_config):
    """Create an engine with a loaded model."""
    success = await huggingface_engine.load_model(model_config)
    if not success:
        pytest.skip(f"Failed to load model {model_config.model_name}")

    yield huggingface_engine

    # Cleanup
    await huggingface_engine.unload_model(model_config.model_name)


@pytest.fixture
async def engine_manager():
    """Create an EngineManager for testing."""
    manager = EngineManager()

    # Reset singleton state for testing
    manager._initialized = True
    manager._engines = {}
    manager._default_engine = None
    manager._loaded_models = {}

    yield manager

    # Cleanup - unload all models
    for model_name in list(manager._loaded_models.keys()):
        await manager.unload_model(model_name)


# ==============================================================================
# Test Prompts
# ==============================================================================

SMALL_PROMPT = "The quick brown fox"
MEDIUM_PROMPT = "Once upon a time, in a land far away, there lived a brave knight who"
LONG_PROMPT = """
In the beginning, the universe was created. This has made a lot of people very angry
and been widely regarded as a bad move. The hitchhiker's guide to the galaxy is a
remarkable book that explores the absurdity of life through humor and wit.
""" * 2  # Repeat to make it longer


@pytest.fixture
def small_prompt():
    """A small prompt for quick tests."""
    return SMALL_PROMPT


@pytest.fixture
def medium_prompt():
    """A medium-length prompt."""
    return MEDIUM_PROMPT


@pytest.fixture
def long_prompt():
    """A long prompt for chunked prefill tests."""
    return LONG_PROMPT


# ==============================================================================
# GPU Check
# ==============================================================================

def is_gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(
    not is_gpu_available(),
    reason="GPU not available"
)


# ==============================================================================
# Test Configuration
# ==============================================================================

@pytest.fixture(autouse=True)
async def reset_globals():
    """Reset any global state before each test."""
    yield
    # Cancel any lingering tasks
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
