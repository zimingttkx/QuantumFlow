"""推理后端

所有后端都依赖 torch/gpu 库。如果未安装，对应的 Engine 为 None。
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantumflow.inference.backends.base import InferenceBackend

__all__ = [
    "HuggingFaceEngine",
    "VLLMEngine",
    "TGIEngine",
    "SGLangEngine",
    "TensorRTLLMEngine",
]

# 尝试导入各后端，失败时设为 None
try:
    from quantumflow.inference.backends.huggingface import HuggingFaceEngine
except ImportError:
    HuggingFaceEngine = None  # type: ignore

try:
    from quantumflow.inference.backends.vllm import VLLMEngine
except ImportError:
    VLLMEngine = None  # type: ignore

try:
    from quantumflow.inference.backends.tgi import TGIEngine
except ImportError:
    TGIEngine = None  # type: ignore

try:
    from quantumflow.inference.backends.sglang import SGLangEngine
except ImportError:
    SGLangEngine = None  # type: ignore

try:
    from quantumflow.inference.backends.tensorrt_llm import TensorRTLLMEngine
except ImportError:
    TensorRTLLMEngine = None  # type: ignore
