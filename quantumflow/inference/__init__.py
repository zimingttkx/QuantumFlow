"""推理模块"""

try:
    from quantumflow.inference.backends.vllm import VLLMEngine
except ImportError:
    VLLMEngine = None  # type: ignore

from quantumflow.inference.engine import (
    InferenceEngine,
    InferenceResult,
    ModelConfig,
    QueuedRequest,
    SamplingParams,
)
from quantumflow.inference.manager import EngineManager, get_engine_manager

__all__ = [
    "InferenceEngine",
    "ModelConfig",
    "SamplingParams",
    "InferenceResult",
    "QueuedRequest",
    "VLLMEngine",
    "EngineManager",
    "get_engine_manager",
]
