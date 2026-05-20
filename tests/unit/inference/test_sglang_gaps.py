"""SGLang 后端覆盖率缺口补充测试

精确覆盖 sglang.py 缺失行:
- initialize generic Exception (64-66)
- load_model Exception (100-106)
- generate: repetition_penalty inclusion (156)
- generate: general Exception handler (223-243)
- generate_stream: stop inclusion (278)
- generate_stream: TimeoutError / Exception (314-317)
- get_stats: Exception handler (347-349)
- chat: stop / presence_penalty / frequency_penalty (398, 400, 402)
- chat_stream: repetition_penalty / stop / penalties (480, 482, 484, 486)
- chat_stream: TimeoutError / Exception (512, 518-524)
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.inference.backends.sglang import SGLangEngine
from quantumflow.inference.engine import ModelConfig, SamplingParams


@pytest.fixture
def engine():
    engine = SGLangEngine(base_url="http://localhost:30000")
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

class TestSGLangInitGaps:
    @pytest.mark.asyncio
    async def test_initialize_generic_exception(self):
        engine = SGLangEngine(base_url="http://localhost:30000")
        with patch("httpx.AsyncClient", side_effect=RuntimeError("Boom")):
            result = await engine.initialize()
        assert result is False
        assert engine._is_initialized is False


# ── load_model Exception ───────────────────────────────────────────────

class TestSGLangLoadModelGaps:
    @pytest.mark.asyncio
    async def test_load_model_exception_returns_false(self, engine):
        engine._client.get = AsyncMock(side_effect=RuntimeError("Server down"))
        config = ModelConfig(model_name="test", model_path="/path")
        result = await engine.load_model(config)
        assert result is False


# ── generate: repetition_penalty + general Exception ───────────────────

class TestSGLangGenerateGaps:
    @pytest.mark.asyncio
    async def test_generate_includes_repetition_penalty(self, engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "choices": [{"text": "ok", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(repetition_penalty=1.2)
        await engine.generate("test", ["prompt"], params)

        payload = engine._client.post.call_args.kwargs["json"]
        assert payload.get("repetition_penalty") == 1.2

    @pytest.mark.asyncio
    async def test_generate_general_exception(self, engine):
        engine._client.post = AsyncMock(side_effect=RuntimeError("Network down"))
        results = await engine.generate("test", ["p1", "p2"], SamplingParams())
        assert len(results) == 2
        for r in results:
            assert r.finish_reason == "error"
            assert "[SGLang错误: Network down]" in r.outputs[0]

    @pytest.mark.asyncio
    async def test_generate_bulk_timeout_error(self, engine):
        engine._client.post = AsyncMock(side_effect=asyncio.TimeoutError())
        results = await engine.generate("test", ["p1", "p2"], SamplingParams())
        assert len(results) == 2
        for r in results:
            assert r.finish_reason == "error"
            assert "[SGLang超时]" in r.outputs[0]


# ── generate_stream: stop + TimeoutError + Exception ───────────────────

class TestSGLangStreamGaps:
    @pytest.mark.asyncio
    async def test_generate_stream_includes_stop(self, engine):
        sse_chunks = [
            'data: {"choices":[{"text":"ok"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        params = SamplingParams(stop=["STOP"])
        async for _ in engine.generate_stream("test", "prompt", params):
            pass

        call_args = engine._client.stream.call_args
        payload = call_args.kwargs["json"]
        assert payload.get("stop") == ["STOP"]

    @pytest.mark.asyncio
    async def test_generate_stream_timeout_error(self, engine):
        engine._client.stream = AsyncMock(side_effect=asyncio.TimeoutError())
        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_generate_stream_general_exception(self, engine):
        engine._client.stream = AsyncMock(side_effect=RuntimeError("Stream crash"))
        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)
        assert chunks == []


# ── get_stats: Exception handler ───────────────────────────────────────

class TestSGLangStatsGaps:
    @pytest.mark.asyncio
    async def test_get_stats_general_exception(self, engine):
        engine._client.get = AsyncMock(side_effect=RuntimeError("Stats crash"))
        stats = await engine.get_stats("test")
        assert stats == {"healthy": 0.0, "connected": 0.0}


# ── chat: stop + presence_penalty + frequency_penalty ──────────────────

class TestSGLangChatGaps:
    @pytest.mark.asyncio
    async def test_chat_includes_stop(self, engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(stop=["END"])
        await engine.chat("test", [{"role": "user", "content": "hi"}], params)

        payload = engine._client.post.call_args.kwargs["json"]
        assert payload.get("stop") == ["END"]

    @pytest.mark.asyncio
    async def test_chat_includes_presence_penalty(self, engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(presence_penalty=0.6)
        await engine.chat("test", [{"role": "user", "content": "hi"}], params)

        payload = engine._client.post.call_args.kwargs["json"]
        assert payload.get("presence_penalty") == 0.6

    @pytest.mark.asyncio
    async def test_chat_includes_frequency_penalty(self, engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(frequency_penalty=0.7)
        await engine.chat("test", [{"role": "user", "content": "hi"}], params)

        payload = engine._client.post.call_args.kwargs["json"]
        assert payload.get("frequency_penalty") == 0.7

    @pytest.mark.asyncio
    async def test_chat_excludes_presence_penalty_when_zero(self, engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(presence_penalty=0.0)
        await engine.chat("test", [{"role": "user", "content": "hi"}], params)

        payload = engine._client.post.call_args.kwargs["json"]
        assert "presence_penalty" not in payload


# ── chat_stream: penalties + TimeoutError + Exception ──────────────────

class TestSGLangChatStreamGaps:
    @pytest.mark.asyncio
    async def test_chat_stream_includes_repetition_penalty(self, engine):
        sse_chunks = [
            'data: {"choices":[{"delta":{"content":"ok"},"index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        params = SamplingParams(repetition_penalty=1.3)
        async for _ in engine.chat_stream("test", [{"role": "user", "content": "hi"}], params):
            pass

        payload = engine._client.stream.call_args.kwargs["json"]
        assert payload.get("repetition_penalty") == 1.3

    @pytest.mark.asyncio
    async def test_chat_stream_includes_stop(self, engine):
        sse_chunks = [
            'data: {"choices":[{"delta":{"content":"ok"},"index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        params = SamplingParams(stop=["FINISH"])
        async for _ in engine.chat_stream("test", [{"role": "user", "content": "hi"}], params):
            pass

        payload = engine._client.stream.call_args.kwargs["json"]
        assert payload.get("stop") == ["FINISH"]

    @pytest.mark.asyncio
    async def test_chat_stream_includes_presence_penalty(self, engine):
        sse_chunks = [
            'data: {"choices":[{"delta":{"content":"ok"},"index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        params = SamplingParams(presence_penalty=0.5)
        async for _ in engine.chat_stream("test", [{"role": "user", "content": "hi"}], params):
            pass

        payload = engine._client.stream.call_args.kwargs["json"]
        assert payload.get("presence_penalty") == 0.5

    @pytest.mark.asyncio
    async def test_chat_stream_includes_frequency_penalty(self, engine):
        sse_chunks = [
            'data: {"choices":[{"delta":{"content":"ok"},"index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        params = SamplingParams(frequency_penalty=0.3)
        async for _ in engine.chat_stream("test", [{"role": "user", "content": "hi"}], params):
            pass

        payload = engine._client.stream.call_args.kwargs["json"]
        assert payload.get("frequency_penalty") == 0.3

    @pytest.mark.asyncio
    async def test_chat_stream_timeout_error(self, engine):
        engine._client.stream = AsyncMock(side_effect=asyncio.TimeoutError())
        chunks = []
        async for text in engine.chat_stream("test", [{"role": "user", "content": "hi"}], SamplingParams()):
            chunks.append(text)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_chat_stream_general_exception(self, engine):
        engine._client.stream = AsyncMock(side_effect=RuntimeError("Chat stream crash"))
        chunks = []
        async for text in engine.chat_stream("test", [{"role": "user", "content": "hi"}], SamplingParams()):
            chunks.append(text)
        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
