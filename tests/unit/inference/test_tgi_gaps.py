"""TGI 后端覆盖率缺口补充测试

精确覆盖 tgi.py 缺失行:
- initialize generic Exception (63-65)
- load_model: info returns non-200 (86-87)
- load_model Exception (105-111)
- generate: fallback token estimation (214, 223)
- generate: general Exception (265-271)
- generate_stream: json_decode_error / parse_error (360-363)
- generate_stream: TimeoutError (366)
- get_stats: general Exception (389-390)
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantumflow.inference.backends.tgi import TGIEngine
from quantumflow.inference.engine import ModelConfig, SamplingParams


@pytest.fixture
def engine():
    engine = TGIEngine(base_url="http://localhost:8080")
    engine._client = MagicMock()
    engine._is_initialized = True
    return engine


def _mock_sse_response(chunks, status_code=200):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    async def aiter_lines():
        for chunk in chunks:
            yield chunk
    mock_response.aiter_lines = aiter_lines
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    return mock_context


# ── initialize generic Exception ───────────────────────────────────────

class TestTGIInitGaps:
    @pytest.mark.asyncio
    async def test_initialize_generic_exception(self):
        engine = TGIEngine(base_url="http://localhost:8080")
        with patch("httpx.AsyncClient", side_effect=RuntimeError("Boom")):
            result = await engine.initialize()
        assert result is False
        assert engine._is_initialized is False


# ── load_model: info non-200 + Exception ───────────────────────────────

class TestTGILoadModelGaps:
    @pytest.mark.asyncio
    async def test_load_model_info_non_200(self, engine):
        mock_response = MagicMock()
        mock_response.status_code = 500
        engine._client.get = AsyncMock(return_value=mock_response)

        config = ModelConfig(model_name="test", model_path="/path")
        result = await engine.load_model(config)

        assert result is False

    @pytest.mark.asyncio
    async def test_load_model_exception_returns_false(self, engine):
        engine._client.get = AsyncMock(side_effect=RuntimeError("Server down"))
        config = ModelConfig(model_name="test", model_path="/path")
        result = await engine.load_model(config)
        assert result is False


# ── generate: fallback token estimation + general Exception ────────────

class TestTGIGenerateGaps:
    @pytest.mark.asyncio
    async def test_generate_fallback_prompt_tokens_estimation(self, engine):
        """
        When prompt_tokens is not a list and not (int, float), the fallback
        uses len(prompts[i]) // 4 for estimation.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "generated_text": ["resp1", "resp2"],
            "prompt_tokens": None,  # None triggers the else branch for prompt
            "generated_tokens": None,  # None triggers the else branch for completion
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["hello world", "hi"], SamplingParams())

        assert len(results) == 2
        # prompt_tokens is None, so fallback: len("hello world")//4 = 11//4 = 2
        assert results[0].prompt_tokens == 2
        # generated_tokens is None, so fallback: len("resp1".split()) = 1
        assert results[0].completion_tokens == 1

    @pytest.mark.asyncio
    async def test_generate_fallback_completion_tokens_estimation(self, engine):
        """
        When generated_tokens is missing, uses len(text.split()) as estimation.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "generated_text": "hello world again",
            "prompt_tokens": 5,
            # generated_tokens missing
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["prompt"], SamplingParams())

        assert len(results) == 1
        assert results[0].prompt_tokens == 5
        # len("hello world again".split()) = 3
        assert results[0].completion_tokens == 3

    @pytest.mark.asyncio
    async def test_generate_completion_tokens_string_fallback(self, engine):
        """
        When generated_tokens is present but not int/float (e.g., string),
        uses len(text.split()) as estimation (line 244).
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "generated_text": "hello world",
            "prompt_tokens": 5,
            "generated_tokens": "not_a_number",  # string triggers fallback
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["prompt"], SamplingParams())

        assert len(results) == 1
        # len("hello world".split()) = 2
        assert results[0].completion_tokens == 2

    @pytest.mark.asyncio
    async def test_generate_general_exception(self, engine):
        engine._client.post = AsyncMock(side_effect=RuntimeError("Network error"))
        results = await engine.generate("test", ["p1", "p2"], SamplingParams())
        assert len(results) == 2
        for r in results:
            assert r.finish_reason == "error"
            assert "[TGI错误: Network error]" in r.outputs[0]


# ── generate_stream: json_decode_error + parse_error + TimeoutError ────

class TestTGIStreamGaps:
    @pytest.mark.asyncio
    async def test_stream_json_decode_error_skipped(self, engine):
        sse_chunks = [
            'data: {"token":{"text":"Hello"}}\n\n',
            'data: {invalid-json\n\n',
            'data: {"token":{"text":"World"}}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert "Hello" in chunks
        assert "World" in chunks

    @pytest.mark.asyncio
    async def test_stream_parse_error_continue(self, engine):
        """Verify non-json exceptions during parsing are handled gracefully"""
        sse_chunks = [
            'data: {"token":{"text":"Hello"}}\n\n',
            'data: {"token":[1,2,3]}\n\n',  # This would trigger a different parse error
            'data: {"token":{"text":"World"}}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert "Hello" in chunks
        assert "World" in chunks

    @pytest.mark.asyncio
    async def test_stream_timeout_error(self, engine):
        # Mock stream to return an async context manager that raises TimeoutError
        async def mock_stream(*args, **kwargs):
            raise asyncio.TimeoutError()

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.__aenter__.side_effect = asyncio.TimeoutError()

        engine._client.stream = MagicMock(return_value=mock_context)
        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)
        assert chunks == []


# ── get_stats: general Exception ───────────────────────────────────────

class TestTGIStatsGaps:
    @pytest.mark.asyncio
    async def test_get_stats_general_exception(self, engine):
        engine._client.get = AsyncMock(side_effect=RuntimeError("Stats crash"))
        stats = await engine.get_stats("test")
        assert stats == {"healthy": 0.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
