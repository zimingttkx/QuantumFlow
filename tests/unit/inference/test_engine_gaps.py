"""Engine 基类覆盖率缺口补充测试

精确覆盖 engine.py 缺失行:
- get_model_config (154)
补充验证:
- is_model_loaded
- loaded_model_names property
- InferenceEngine base class contract
"""

import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.core.constants import InferenceBackendType
from quantumflow.inference.engine import (
    InferenceEngine,
    InferenceResult,
    ModelConfig,
    SamplingParams,
)


class TestModelConfigExtra:
    def test_model_config_all_fields_to_dict(self):
        config = ModelConfig(
            model_name="full-model",
            model_path="/full/path",
            tensor_parallel=4,
            pipeline_parallel=2,
            gpu_memory_utilization=0.75,
            max_model_len=8192,
            dtype="bfloat16",
            quantization="awq",
            trust_remote_code=False,
            block_size=32,
            max_num_batched_tokens=16384,
            max_num_seqs=512,
            enforce_eager=True,
            enable_chunked_prefill=True,
            prefill_chunk_size=1024,
            torch_compile=False,
        )
        d = config.to_dict()
        assert d["model_name"] == "full-model"
        assert d["model_path"] == "/full/path"
        assert d["tensor_parallel_size"] == 4
        assert d["pipeline_parallel_size"] == 2
        assert d["gpu_memory_utilization"] == 0.75
        assert d["max_model_len"] == 8192
        assert d["dtype"] == "bfloat16"
        assert d["quantization"] == "awq"
        assert d["trust_remote_code"] is False


class TestSamplingParamsExtra:
    def test_sampling_params_all_fields_to_dict(self):
        params = SamplingParams(
            temperature=0.5,
            top_p=0.95,
            top_k=100,
            max_tokens=500,
            repetition_penalty=1.1,
            stop=["###"],
            presence_penalty=0.2,
            frequency_penalty=0.3,
            details=True,
        )
        d = params.to_dict()
        assert d["temperature"] == 0.5
        assert d["top_p"] == 0.95
        assert d["top_k"] == 100
        assert d["max_tokens"] == 500
        assert d["repetition_penalty"] == 1.1
        assert d["stop"] == ["###"]
        assert d["presence_penalty"] == 0.2
        assert d["frequency_penalty"] == 0.3
        assert d["details"] is True

    def test_sampling_params_to_dict_defaults(self):
        params = SamplingParams()
        d = params.to_dict()
        assert d["temperature"] == 0.7
        assert d["stop"] is None


class TestInferenceEngineBase:
    """测试 InferenceEngine 基类 property 和 method"""

    def test_is_ready_property(self):
        from quantumflow.inference.backends.vllm import VLLMEngine
        engine = VLLMEngine()
        assert engine.is_ready is False
        engine._is_initialized = True
        assert engine.is_ready is True

    def test_loaded_model_names_property(self):
        from quantumflow.inference.backends.vllm import VLLMEngine
        engine = VLLMEngine()
        assert engine.loaded_model_names == []
        engine._loaded_models["model1"] = ModelConfig(model_name="model1", model_path="/p1")
        engine._loaded_models["model2"] = ModelConfig(model_name="model2", model_path="/p2")
        assert len(engine.loaded_model_names) == 2
        assert "model1" in engine.loaded_model_names
        assert "model2" in engine.loaded_model_names

    @pytest.mark.asyncio
    async def test_get_model_config_returns_model(self):
        from quantumflow.inference.backends.vllm import VLLMEngine
        engine = VLLMEngine()
        config = ModelConfig(model_name="test", model_path="/path")
        engine._loaded_models["test"] = config

        result = await engine.get_model_config("test")
        assert result is config
        assert result.model_name == "test"

    @pytest.mark.asyncio
    async def test_get_model_config_returns_none_for_missing(self):
        from quantumflow.inference.backends.vllm import VLLMEngine
        engine = VLLMEngine()
        result = await engine.get_model_config("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_is_model_loaded(self):
        from quantumflow.inference.backends.vllm import VLLMEngine
        engine = VLLMEngine()
        assert await engine.is_model_loaded("test") is False
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")
        assert await engine.is_model_loaded("test") is True

    @pytest.mark.asyncio
    async def test_get_model_config_empty_loaded_models(self):
        from quantumflow.inference.backends.vllm import VLLMEngine
        engine = VLLMEngine()
        result = await engine.get_model_config("any")
        assert result is None


class TestInferenceResultFields:
    def test_result_default_metrics_is_empty_dict(self):
        result = InferenceResult(
            request_id="r1",
            outputs=["out"],
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=10.0,
            finish_reason="stop",
        )
        assert result.metrics == {}

    def test_result_all_fields_set(self):
        result = InferenceResult(
            request_id="r1",
            outputs=["out1", "out2"],
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=200.5,
            finish_reason="length",
            metrics={"tps": 250.0},
        )
        assert result.request_id == "r1"
        assert result.outputs == ["out1", "out2"]
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.latency_ms == 200.5
        assert result.finish_reason == "length"
        assert result.metrics == {"tps": 250.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
