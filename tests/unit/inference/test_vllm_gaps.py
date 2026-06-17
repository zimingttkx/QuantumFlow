"""vLLM 后端覆盖率缺口补充测试

精确覆盖 vllm.py 缺失行:
- _run_vllm_generate (44)
- initialize generic Exception (64-69)
- load_model Exception (97-99)
- unload_model happy path (103-119)
- generate happy path (143-193)
- generate_stream happy path (217-250)
- get_stats happy path (256-279)
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantumflow.inference.backends.vllm import VLLMEngine, _build_vllm_llm, _run_vllm_generate
from quantumflow.inference.engine import InferenceResult, ModelConfig, SamplingParams


# ── _build_vllm_llm helper ──────────────────────────────────────────────

class TestBuildVLLMLLM:
    def test_build_vllm_llm_sets_env_and_calls_llm(self):
        import os
        with patch.dict("os.environ", {}, clear=True):
            with patch("vllm.LLM") as mock_llm:
                config = ModelConfig(
                    model_name="test", model_path="/path",
                    tensor_parallel=2, gpu_memory_utilization=0.85,
                    max_model_len=4096, dtype="float16",
                    quantization="gptq", trust_remote_code=False,
                    enforce_eager=True,
                )
                _build_vllm_llm(config)

                # Check inside the context since patch.dict restores env on exit
                assert os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN") == "1"
                mock_llm.assert_called_once()
                kwargs = mock_llm.call_args.kwargs
                assert kwargs["model"] == "/path"
                assert kwargs["tensor_parallel_size"] == 2
                assert kwargs["gpu_memory_utilization"] == 0.85
                assert kwargs["max_model_len"] == 4096
                assert kwargs["dtype"] == "float16"
                assert kwargs["quantization"] == "gptq"
                assert kwargs["trust_remote_code"] is False
                assert kwargs["enforce_eager"] is True


class TestRunVLLMGenerate:
    def test_run_vllm_generate_calls_llm_dot_generate(self):
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="results")
        prompts = ["p1", "p2"]
        params = {"temperature": 0.7}

        result = _run_vllm_generate(mock_llm, prompts, params)

        assert result == "results"
        mock_llm.generate.assert_called_once_with(prompts, params)


# ── initialize 异常路径 ────────────────────────────────────────────────

class TestVLLMInitializeExceptions:
    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        engine = VLLMEngine()
        with patch.dict("sys.modules", {"vllm": None}):
            result = await engine.initialize()
        assert result is False
        assert engine.is_ready is False

    @pytest.mark.asyncio
    async def test_initialize_generic_exception(self):
        engine = VLLMEngine()
        import sys
        # Collect existing vllm modules
        vllm_modules = {k: v for k, v in sys.modules.items() if k == "vllm" or k.startswith("vllm.")}

        # Create a fake vllm module that has no __version__ (triggers AttributeError = Exception)
        class FakeVLLM:
            pass
        sys.modules["vllm"] = FakeVLLM()

        try:
            result = await engine.initialize()
            assert result is False
            assert engine.is_ready is False
        finally:
            # Restore vllm modules
            for mod in vllm_modules:
                sys.modules[mod] = vllm_modules[mod]


# ── load_model 异常路径 ────────────────────────────────────────────────

class TestVLLMLoadModelExceptions:
    @pytest.mark.asyncio
    async def test_load_model_exception_caught(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        config = ModelConfig(model_name="test", model_path="/path")
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("Loop boom")):
            result = await engine.load_model(config)
        assert result is False

    @pytest.mark.asyncio
    async def test_load_model_not_initialized(self):
        engine = VLLMEngine()
        config = ModelConfig(model_name="test", model_path="/path")
        result = await engine.load_model(config)
        assert result is False


# ── unload_model happy path ────────────────────────────────────────────

class TestVLLMUnloadModel:
    @pytest.mark.asyncio
    async def test_unload_model_with_cuda_cleanup(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                with patch("gc.collect") as mock_gc:
                    result = await engine.unload_model("test")

        assert result is True
        assert "test" not in engine._llm_instances
        assert "test" not in engine._loaded_models
        mock_gc.assert_called_once()
        mock_empty_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_model_with_cuda_not_available(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        engine._llm_instances["test"] = MagicMock()
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("torch.cuda.is_available", return_value=False):
            with patch("gc.collect") as mock_gc:
                result = await engine.unload_model("test")

        assert result is True
        mock_gc.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_model_cuda_error_suppressed(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        engine._llm_instances["test"] = MagicMock()
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache", side_effect=RuntimeError("CUDA error")):
                result = await engine.unload_model("test")

        assert result is True

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        result = await engine.unload_model("nonexistent")
        assert result is False


# ── generate happy path ────────────────────────────────────────────────

class TestVLLMGenerateHappy:
    def _make_mock_output(self, text, finish_reason="stop",
                          prompt_ids=None, token_ids=None):
        mock_out = MagicMock()
        mock_out.text = text
        mock_out.finish_reason = finish_reason
        mock_out.token_ids = token_ids
        mock_output = MagicMock()
        mock_output.prompt_token_ids = prompt_ids
        mock_output.outputs = [mock_out]
        return mock_output

    @pytest.mark.asyncio
    async def test_generate_happy_path_single_prompt(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        mock_out = self._make_mock_output(
            "Hello world", "stop",
            prompt_ids=[1, 2, 3],
            token_ids=[4, 5]
        )
        mock_llm.generate = MagicMock(return_value=[mock_out])

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp_instance = MagicMock()
            mock_sp.return_value = mock_sp_instance

            results = await engine.generate("test", ["prompt"], SamplingParams(temperature=0.5))

        assert len(results) == 1
        r = results[0]
        assert r.outputs[0] == "Hello world"
        assert r.prompt_tokens == 3
        assert r.completion_tokens == 2
        assert r.finish_reason == "stop"
        assert r.request_id == "test_0"
        assert r.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_happy_path_multiple_prompts(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        outputs = [
            self._make_mock_output(f"out_{i}", "stop",
                                   prompt_ids=[10+i], token_ids=[20+i])
            for i in range(3)
        ]
        mock_llm.generate = MagicMock(return_value=outputs)

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp.return_value = MagicMock()

            results = await engine.generate("test", ["a", "b", "c"], SamplingParams())

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.outputs[0] == f"out_{i}"
            assert r.request_id == f"test_{i}"

    @pytest.mark.asyncio
    async def test_generate_with_none_finish_reason(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        mock_out = self._make_mock_output("text", None, prompt_ids=[1], token_ids=[2])
        mock_llm.generate = MagicMock(return_value=[mock_out])

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp.return_value = MagicMock()

            results = await engine.generate("test", ["prompt"], SamplingParams())

        assert results[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_none_token_ids(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        mock_out = self._make_mock_output("text", "stop", prompt_ids=None, token_ids=None)
        mock_llm.generate = MagicMock(return_value=[mock_out])

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp.return_value = MagicMock()

            results = await engine.generate("test", ["prompt"], SamplingParams())

        assert results[0].prompt_tokens == 0
        assert results[0].completion_tokens == 0

    @pytest.mark.asyncio
    async def test_generate_passes_sampling_params(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        mock_out = self._make_mock_output("ok")
        mock_llm.generate = MagicMock(return_value=[mock_out])

        params = SamplingParams(
            temperature=0.3, top_p=0.8, top_k=100,
            max_tokens=500, repetition_penalty=1.2,
            stop=["END"]
        )

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp.return_value = MagicMock()
            await engine.generate("test", ["p"], params)

        mock_sp.assert_called_once_with(
            temperature=0.3,
            top_p=0.8,
            top_k=100,
            max_tokens=500,
            repetition_penalty=1.2,
            stop=["END"],
        )

    @pytest.mark.asyncio
    async def test_generate_exception_returns_error_results(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp_instance = MagicMock()
            mock_sp.return_value = mock_sp_instance
            mock_llm.generate = MagicMock(side_effect=RuntimeError("GPU OOM"))

            results = await engine.generate("test", ["p1", "p2"], SamplingParams())

        assert len(results) == 2
        for r in results:
            assert r.finish_reason == "error"
            assert "[vLLM错误: GPU OOM]" in r.outputs[0]


# ── generate_stream happy path ─────────────────────────────────────────

class TestVLLMGenerateStream:
    @pytest.mark.asyncio
    async def test_generate_stream_yields_text_chunks(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        mock_out1 = MagicMock()
        mock_out1.text = "Hello "
        mock_output1 = MagicMock()
        mock_output1.outputs = [mock_out1]

        mock_out2 = MagicMock()
        mock_out2.text = "world"
        mock_output2 = MagicMock()
        mock_output2.outputs = [mock_out2]

        mock_llm.generate = MagicMock(return_value=[mock_output1, mock_output2])

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp.return_value = MagicMock()
            chunks = []
            async for chunk in engine.generate_stream("test", "prompt", SamplingParams()):
                chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_generate_stream_empty_text_skipped(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        mock_out = MagicMock()
        mock_out.text = ""
        mock_output = MagicMock()
        mock_output.outputs = [mock_out]
        mock_llm.generate = MagicMock(return_value=[mock_output])

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp.return_value = MagicMock()
            chunks = []
            async for chunk in engine.generate_stream("test", "prompt", SamplingParams()):
                chunks.append(chunk)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_generate_stream_exception_suppressed(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        mock_llm = MagicMock()
        engine._llm_instances["test"] = mock_llm
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("vllm.SamplingParams") as mock_sp:
            mock_sp.return_value = MagicMock()
            mock_llm.generate = MagicMock(side_effect=RuntimeError("Stream error"))
            chunks = []
            async for chunk in engine.generate_stream("test", "prompt", SamplingParams()):
                chunks.append(chunk)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_generate_stream_model_not_loaded(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        chunks = []
        async for chunk in engine.generate_stream("nonexistent", "prompt", SamplingParams()):
            chunks.append(chunk)
        assert chunks == []


# ── get_stats happy path ───────────────────────────────────────────────

class TestVLLMGetStats:
    @pytest.mark.asyncio
    async def test_get_stats_returns_gpu_metrics_when_cuda_available(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        engine._llm_instances["test"] = MagicMock()
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=2 * 1024**3):
                with patch("torch.cuda.memory_reserved", return_value=3 * 1024**3):
                    with patch("pynvml.nvmlInit"):
                        with patch("pynvml.nvmlDeviceGetHandleByIndex"):
                            mock_util = MagicMock()
                            mock_util.gpu = 75
                            mock_util.memory = 50
                            with patch("pynvml.nvmlDeviceGetUtilizationRates", return_value=mock_util):
                                with patch("pynvml.nvmlShutdown"):
                                    stats = await engine.get_stats("test")

        assert stats["gpu_memory_allocated"] == 2.0
        assert stats["gpu_memory_reserved"] == 3.0
        assert stats["gpu_utilization"] == 0.75
        assert stats["gpu_memory_utilization"] == 0.5

    @pytest.mark.asyncio
    async def test_get_stats_pynvml_fails_silently(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        engine._llm_instances["test"] = MagicMock()
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=1 * 1024**3):
                with patch("torch.cuda.memory_reserved", return_value=1 * 1024**3):
                    with patch("pynvml.nvmlInit", side_effect=ImportError):
                        stats = await engine.get_stats("test")

        assert "gpu_memory_allocated" in stats
        assert "gpu_utilization" not in stats

    @pytest.mark.asyncio
    async def test_get_stats_cuda_not_available(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        engine._llm_instances["test"] = MagicMock()
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("torch.cuda.is_available", return_value=False):
            stats = await engine.get_stats("test")

        assert stats == {}

    @pytest.mark.asyncio
    async def test_get_stats_model_not_loaded(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        stats = await engine.get_stats("nonexistent")
        assert stats == {}

    @pytest.mark.asyncio
    async def test_get_stats_torch_exception_suppressed(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        engine._llm_instances["test"] = MagicMock()
        engine._loaded_models["test"] = ModelConfig(model_name="test", model_path="/path")

        with patch("torch.cuda.is_available", side_effect=RuntimeError):
            stats = await engine.get_stats("test")
        assert stats == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
