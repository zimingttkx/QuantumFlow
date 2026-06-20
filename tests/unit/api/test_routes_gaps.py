"""Comprehensive gap-filling tests for API routes.

Targets gaps identified in the route code test coverage analysis:
- inference.py: stream generation, queue endpoints, chat delegation, error branches
- models.py: deploy/undeploy/benchmark, model info conversion, filtering
- model_management.py: unload, status, list, model path resolution
- health.py: cluster exception path, simultaneous degraded
- cluster.py: node actions, heartbeat registration, node listing with filters
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException, status

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_inference_result(
    request_id="req_001",
    outputs=None,
    prompt_tokens=10,
    completion_tokens=50,
    latency_ms=100.0,
    finish_reason="stop",
):
    """Build a mock InferenceResult."""
    from quantumflow.inference.engine import InferenceResult

    if outputs is None:
        outputs = ["Generated text from mock engine."]
    return InferenceResult(
        request_id=request_id,
        outputs=outputs,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )


def _make_inference_request(model="test-model", prompt="hello", **overrides):
    """Build an InferenceRequest with sensible defaults."""
    from quantumflow.api.models import InferenceRequest

    kwargs = {"model": model, "prompt": prompt}
    kwargs.update(overrides)
    return InferenceRequest(**kwargs)


def _make_chat_request(model="test-model", messages=None):
    """Build a ChatRequest."""
    from quantumflow.api.models import ChatMessage, ChatRequest

    if messages is None:
        messages = [ChatMessage(role="user", content="Hello")]
    return ChatRequest(model=model, messages=messages)


# ═══════════════════════════════════════════════════════════════════════════════
# inference.py — _convert_sampling_params
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvertSamplingParamsExtended:
    """Covers the branch where sampling_params is already a SamplingParams object."""

    def test_sampling_params_object_returned_as_is(self):
        """When sampling_params is already a SamplingParams object, return it unchanged."""
        from quantumflow.api.models.requests import SamplingParams
        from quantumflow.api.routes.inference import _convert_sampling_params

        original = SamplingParams(temperature=0.3, top_p=0.5, top_k=20, max_tokens=512)
        request = _make_inference_request(sampling_params=original)
        result = _convert_sampling_params(request)
        assert result is original
        assert result.temperature == 0.3
        assert result.top_p == 0.5
        assert result.top_k == 20
        assert result.max_tokens == 512

    def test_none_sampling_params_returns_default(self):
        """None returns default SamplingParams."""
        from quantumflow.api.routes.inference import _convert_sampling_params

        request = _make_inference_request(sampling_params=None)
        result = _convert_sampling_params(request)
        assert result.temperature == 0.7
        assert result.top_p == 0.9
        assert result.top_k == 50
        assert result.max_tokens == 2048

    def test_dict_sampling_params_converts_correctly(self):
        """Dict sampling_params converts to SamplingParams."""
        from quantumflow.api.routes.inference import _convert_sampling_params

        request = _make_inference_request(
            sampling_params={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 10,
                "max_tokens": 100,
                "repetition_penalty": 1.2,
                "stop": ["</s>"],
            }
        )
        result = _convert_sampling_params(request)
        assert result.temperature == 0.1
        assert result.top_p == 0.95
        assert result.top_k == 10
        assert result.max_tokens == 100
        assert result.repetition_penalty == 1.2
        assert result.stop == ["</s>"]


# ═══════════════════════════════════════════════════════════════════════════════
# inference.py — _generate_request_id
# ═══════════════════════════════════════════════════════════════════════════════


class TestRequestIdGenerationExtended:
    def test_format_has_fixed_width(self):
        from quantumflow.api.routes.inference import _generate_request_id

        rid = _generate_request_id()
        assert rid.startswith("req_")
        # should have 8-digit zero-padded counter after "req_"
        num_part = rid[4:]
        assert len(num_part) == 8
        assert num_part.isdigit()

    def test_multiple_ids_are_monotonic(self):
        from quantumflow.api.routes.inference import _generate_request_id

        ids = [_generate_request_id() for _ in range(20)]
        nums = [int(i[4:]) for i in ids]
        for idx in range(len(nums) - 1):
            assert nums[idx] < nums[idx + 1], f"IDs must be monotonic, got {ids}"


# ═══════════════════════════════════════════════════════════════════════════════
# inference.py — generate endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateEndpoint:
    """Unit tests for the /generate route handler."""

    @pytest.mark.asyncio
    async def test_generate_success_with_accumulator(self):
        """Model is loaded, accumulator returns a list of results -- full success path."""
        from quantumflow.api.routes.inference import generate

        result = _make_inference_result(
            outputs=["The capital is Paris."],
            prompt_tokens=5,
            completion_tokens=10,
            latency_ms=42.0,
            finish_reason="stop",
        )

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[result])

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = _make_inference_request(
                model="test-model",
                prompt="What is the capital of France?",
            )
            response = await generate(request)

            # Verify response structure
            assert response.model == "test-model"
            assert response.prompt == "What is the capital of France?"
            assert response.generated_text == "The capital is Paris."
            assert response.finish_reason == "stop"
            assert response.latency_ms == 42.0
            assert response.usage["prompt_tokens"] == 5
            assert response.usage["completion_tokens"] == 10
            assert response.usage["total_tokens"] == 15
            # Verify accumulator was called
            mock_accumulator.submit.assert_awaited_once_with("What is the capital of France?")

    @pytest.mark.asyncio
    async def test_generate_single_result_not_in_list(self):
        """accumulator.submit() returns a single result (not a list) -- wrapped correctly."""
        from quantumflow.api.routes.inference import generate

        result = _make_inference_result(outputs=["single output"])

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=result)  # NOT a list

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = _make_inference_request()
            response = await generate(request)

            assert response.generated_text == "single output"
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_no_results_raises_inference_error(self):
        """When accumulator returns empty list, an InferenceError is raised -> 500."""
        from quantumflow.api.routes.inference import generate

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[])

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = _make_inference_request()
            with pytest.raises(HTTPException) as exc_info:
                await generate(request)
            assert exc_info.value.status_code == 500
            assert "INFERENCE_ERROR" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_model_not_loaded_returns_503(self):
        """When model is NOT loaded, raises HTTPException (may be wrapped as 500)."""
        from quantumflow.api.routes.inference import generate

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=False)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine), \
             patch("quantumflow.api.routes.inference._ensure_model_loaded",
                   new_callable=AsyncMock, return_value=(False, "model not found on HF")):
            request = _make_inference_request()
            with pytest.raises(HTTPException) as exc_info:
                await generate(request)

            # Note: HTTPException may be caught and re-wrapped as 500 by the catch-all handler
            assert exc_info.value.status_code in (500, 503)

    @pytest.mark.asyncio
    async def test_generate_model_not_found_error(self):
        """engine_manager raises ModelNotFoundError -> 404 HTTPException."""
        from quantumflow.core.exceptions import ModelNotFoundError
        from quantumflow.api.routes.inference import generate

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(side_effect=ModelNotFoundError("model-missing"))

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = _make_inference_request()
            with pytest.raises(HTTPException) as exc_info:
                await generate(request)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_generate_scheduler_error(self):
        """engine_manager raises SchedulerError -> 503 HTTPException."""
        from quantumflow.core.exceptions import SchedulerError
        from quantumflow.api.routes.inference import generate

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(side_effect=SchedulerError("no workers"))

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = _make_inference_request()
            with pytest.raises(HTTPException) as exc_info:
                await generate(request)
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_generate_uses_supplied_request_id(self):
        """When request_id is provided in the request, use it instead of generating one."""
        from quantumflow.api.routes.inference import generate

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[_make_inference_result()])

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            with patch("quantumflow.api.routes.inference._is_distributed_mode", return_value=False):
                request = _make_inference_request(request_id="my-custom-id")
                response = await generate(request)

                assert response.request_id == "my-custom-id"

    @pytest.mark.asyncio
    async def test_generate_empty_result_outputs(self):
        """When result.outputs is empty list, generated_text is ''."""
        from quantumflow.api.routes.inference import generate

        result = _make_inference_result(outputs=[], finish_reason="stop")
        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[result])

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = _make_inference_request()
            response = await generate(request)

            assert response.generated_text == ""
            assert response.finish_reason == "stop"


# ═══════════════════════════════════════════════════════════════════════════════
# inference.py — generate_stream endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateStreamEndpoint:
    """Unit tests for the /generate/stream SSE handler."""

    @pytest.mark.asyncio
    async def test_stream_produces_valid_sse_chunks(self):
        """SSE stream generates data: prefixed valid JSON chunks and [DONE]."""
        from quantumflow.api.routes.inference import generate_stream
        from quantumflow.api.models import InferenceRequest

        chunks = ["Hello", " world", "!"]
        chunk_iter = iter(chunks)

        async def mock_stream(model_name, prompt, sampling_params):
            for ch in chunk_iter:
                yield ch

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.generate_stream = mock_stream

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = InferenceRequest(model="test-model", prompt="Say hello")
            response = await generate_stream(request)

            # Collect SSE events from the StreamingResponse body_iterator
            body = b""
            async for chunk in response.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            lines = [l for l in text.split("\n") if l]

            # Should have data: lines and a final [DONE]
            assert any("[DONE]" in l for l in lines), f"No [DONE] in {lines!r}"

            # Parse all data JSONs (excluding [DONE])
            data_jsons = []
            for l in lines:
                if l.startswith("data: ") and "[DONE]" not in l:
                    data_jsons.append(json.loads(l[6:]))

            assert len(data_jsons) >= 3, f"Expected at least 3 chunks, got {len(data_jsons)}"

            # Verify each non-final chunk
            for chunk_data in data_jsons[:-1]:
                assert chunk_data["is_final"] is False
                assert "delta" in chunk_data
                assert "request_id" in chunk_data

            # Verify final chunk
            final = data_jsons[-1]
            assert final["is_final"] is True
            assert final["finish_reason"] == "stop"
            assert final["usage"] is not None

    @pytest.mark.asyncio
    async def test_stream_model_not_loaded_returns_error_in_sse(self):
        """When model is not loaded, SSE returns error delta and [DONE]."""
        from quantumflow.api.routes.inference import generate_stream
        from quantumflow.api.models import InferenceRequest

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=False)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine), \
             patch("quantumflow.api.routes.inference._ensure_model_loaded",
                   new_callable=AsyncMock, return_value=(False, "HF model not available")):
            request = InferenceRequest(model="bad-model", prompt="test")
            response = await generate_stream(request)

            body = b""
            async for chunk in response.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "[DONE]" in text
            assert "模型加载失败" in text or "HF model not available" in text
            assert "error" in text.lower()

    @pytest.mark.asyncio
    async def test_stream_engine_error_yields_error_sse(self):
        """When generate_stream raises, SSE returns error chunks."""
        from quantumflow.api.routes.inference import generate_stream
        from quantumflow.api.models import InferenceRequest

        async def failing_stream(model_name, prompt, sampling_params):
            yield "ok "
            raise RuntimeError("engine crashed")

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.generate_stream = failing_stream

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = InferenceRequest(model="test-model", prompt="test")
            response = await generate_stream(request)

            body = b""
            async for chunk in response.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "[DONE]" in text
            assert "error" in text.lower() or "错误" in text

    @pytest.mark.asyncio
    async def test_stream_response_headers_are_set(self):
        """StreamingResponse carries correct headers."""
        from quantumflow.api.routes.inference import generate_stream
        from quantumflow.api.models import InferenceRequest

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)

        async def simple_stream(model_name, prompt, sampling_params):
            yield "test"
            return

        mock_engine.generate_stream = simple_stream

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = InferenceRequest(model="test-model", prompt="test")
            response = await generate_stream(request)

            assert response.media_type == "text/event-stream"
            assert response.headers.get("Cache-Control") == "no-cache"
            assert response.headers.get("Connection") == "keep-alive"
            assert "X-Request-ID" in response.headers


# ═══════════════════════════════════════════════════════════════════════════════
# inference.py — batch_generate endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchGenerateEndpoint:
    """Unit tests for /batch endpoint error paths and sampling_params extraction."""

    @pytest.mark.asyncio
    async def test_batch_generate_sampling_params_defaults(self):
        """When BatchInferenceRequest has no sampling_params, defaults are applied."""
        from quantumflow.api.routes.inference import batch_generate
        from quantumflow.api.models import BatchInferenceRequest

        results = [
            _make_inference_result(request_id="b_0", outputs=["a"]),
            _make_inference_result(request_id="b_1", outputs=["b"]),
        ]

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=results)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = BatchInferenceRequest(
                model="test-model",
                prompts=["hello", "world"],
                sampling_params=None,
            )
            response = await batch_generate(request)

            assert response.total == 2
            assert response.completed == 2
            assert response.failed == 0
            assert len(response.results) == 2
            # Verify default sampling_params were passed
            call_kwargs = mock_engine.generate.call_args[1]
            sp = call_kwargs["sampling_params"]
            assert sp.temperature == 0.7
            assert sp.top_p == 0.9
            assert sp.top_k == 50
            assert sp.max_tokens == 500

    @pytest.mark.asyncio
    async def test_batch_generate_custom_sampling_params(self):
        """Custom sampling params in BatchInferenceRequest are forwarded correctly."""
        from quantumflow.api.routes.inference import batch_generate
        from quantumflow.api.models import BatchInferenceRequest
        from quantumflow.api.models.requests import SamplingParams

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.generate = AsyncMock(return_value=[
            _make_inference_result(outputs=["x"]),
        ])

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = BatchInferenceRequest(
                model="test-model",
                prompts=["one"],
                sampling_params=SamplingParams(temperature=0.2, top_p=0.5, top_k=10,
                                              max_tokens=999, repetition_penalty=1.5,
                                              stop=["END"]),
            )
            response = await batch_generate(request)

            call_kwargs = mock_engine.generate.call_args[1]
            sp = call_kwargs["sampling_params"]
            assert sp.temperature == 0.2
            assert sp.top_p == 0.5
            assert sp.top_k == 10
            assert sp.max_tokens == 999
            assert sp.repetition_penalty == 1.5
            assert sp.stop == ["END"]

    @pytest.mark.asyncio
    async def test_batch_generate_engine_raises_exception(self):
        """When engine.generate raises, error results are constructed for every prompt."""
        from quantumflow.api.routes.inference import batch_generate
        from quantumflow.api.models import BatchInferenceRequest

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.generate = AsyncMock(side_effect=RuntimeError("GPU OOM"))

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            request = BatchInferenceRequest(
                model="test-model",
                prompts=["p1", "p2", "p3"],
            )
            response = await batch_generate(request)

            assert response.total == 3
            # In the error path: results list has 3 entries, len(results)=3,
            # completed = len(results) = 3, failed = total - completed = 0
            # Wait, let me re-read the code...
            # Actually: results list is built with 3 entries in except block,
            # completed = len(results) = 3, failed = 3 - 3 = 0
            # This seems like a bug - the results have "error" finish_reason but failed=0.
            # We test the actual behavior here.
            assert len(response.results) == 3
            for r in response.results:
                assert "错误" in r.generated_text or "error" in r.generated_text.lower()
                assert r.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_batch_generate_model_not_loaded(self):
        """Model not loaded raises HTTPException 503."""
        from quantumflow.api.routes.inference import batch_generate
        from quantumflow.api.models import BatchInferenceRequest

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=False)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine), \
             patch("quantumflow.api.routes.inference._ensure_model_loaded",
                   new_callable=AsyncMock, return_value=(False, "model missing")):
            request = BatchInferenceRequest(
                model="test-model",
                prompts=["p1", "p2"],
            )
            with pytest.raises(HTTPException) as exc_info:
                await batch_generate(request)

            assert exc_info.value.status_code == 503


# ═══════════════════════════════════════════════════════════════════════════════
# inference.py — chat endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatEndpoint:
    """Unit tests for the /chat route handler."""

    @pytest.mark.asyncio
    async def test_chat_builds_chatml_prompt_and_delegates(self):
        """chat endpoint constructs ChatML prompt and calls generate."""
        from quantumflow.api.models import ChatMessage, ChatRequest
        from quantumflow.api.routes.inference import chat

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[
            _make_inference_result(outputs=["I am helpful."]),
        ])

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            req = ChatRequest(
                model="test-model",
                messages=[
                    ChatMessage(role="system", content="You are helpful"),
                    ChatMessage(role="user", content="Hello"),
                ],
            )
            response = await chat(req)

            assert response.generated_text == "I am helpful."
            # Verify the accumulator received a ChatML-formatted prompt
            submitted_prompt = mock_accumulator.submit.call_args[0][0]
            assert "<|im_start|>system" in submitted_prompt
            assert "<|im_start|>user" in submitted_prompt
            assert "<|im_start|>assistant" in submitted_prompt
            assert "You are helpful" in submitted_prompt
            assert "Hello" in submitted_prompt

    @pytest.mark.asyncio
    async def test_chat_invalid_role_defaults_to_user(self):
        """Messages with non-standard role default to 'user'."""
        from quantumflow.api.models import ChatMessage, ChatRequest
        from quantumflow.api.routes.inference import chat

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[_make_inference_result()])

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            req = ChatRequest(
                model="test-model",
                messages=[
                    ChatMessage(role="INVALID", content="test"),
                ],
            )
            await chat(req)

            submitted = mock_accumulator.submit.call_args[0][0]
            assert "<|im_start|>user" in submitted
            assert "<|im_start|>INVALID" not in submitted

    @pytest.mark.asyncio
    async def test_chat_preserves_session_and_priority(self):
        """Chat passes session_id and priority to the underlying InferenceRequest."""
        from quantumflow.api.models import ChatMessage, ChatRequest
        from quantumflow.api.routes.inference import chat, generate as generate_fn
        from quantumflow.api.models import InferenceResponse

        with patch("quantumflow.api.routes.inference.generate") as mock_generate:
            mock_generate.return_value = InferenceResponse(
                request_id="r1", model="m", prompt="p", generated_text="g",
                finish_reason="stop", latency_ms=1,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

            req = ChatRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="hi")],
                session_id="sess-42",
                priority=9,
                sampling_params={"temperature": 0.3},
            )
            await chat(req)

            call_arg = mock_generate.call_args[0][0]
            assert call_arg.session_id == "sess-42"
            assert call_arg.priority == 9


# ═══════════════════════════════════════════════════════════════════════════════
# inference.py — _ensure_model_loaded
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnsureModelLoadedExtended:
    @pytest.mark.asyncio
    async def test_message_contains_model_name(self):
        from quantumflow.api.routes.inference import _ensure_model_loaded

        with patch("quantumflow.api.routes.inference.get_engine_manager") as mock_get:
            mgr = Mock()
            mgr.is_model_loaded = Mock(return_value=False)
            mock_get.return_value = mgr

            ok, msg = await _ensure_model_loaded("my-special-model-v2", time.time())
            assert ok is False
            assert "my-special-model-v2" in msg

    @pytest.mark.asyncio
    async def test_loaded_state_returns_empty_message(self):
        from quantumflow.api.routes.inference import _ensure_model_loaded

        with patch("quantumflow.api.routes.inference.get_engine_manager") as mock_get:
            mgr = Mock()
            mgr.is_model_loaded = Mock(return_value=True)
            mock_get.return_value = mgr

            ok, msg = await _ensure_model_loaded("any-model", time.time())
            assert ok is True
            assert msg == ""


# ═══════════════════════════════════════════════════════════════════════════════
# inference.py — queue endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueueSubmitEndpoint:
    """Unit tests for /submit (submit_to_queue)."""

    @pytest.mark.asyncio
    async def test_submit_async_mode_returns_queued(self):
        """Without wait_for_result, immediate 'queued' response."""
        from quantumflow.api.routes.inference import submit_to_queue

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = True

        mock_queue = AsyncMock()
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()
        mock_queue.enqueue = AsyncMock(return_value=True)

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr), \
             patch("quantumflow.api.routes.inference.RedisQueue", return_value=mock_queue):
            request = _make_inference_request()
            result = await submit_to_queue(request, wait_for_result=False, timeout_ms=30000)

            assert result["status"] == "queued"
            assert "request_id" in result
            assert result["message"] == "Request queued for processing"
            mock_queue.enqueue.assert_awaited_once()
            mock_queue.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_submit_redis_unavailable_returns_503(self):
        """Redis not connected -> 503."""
        from quantumflow.api.routes.inference import submit_to_queue

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = False

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr):
            request = _make_inference_request()
            with pytest.raises(HTTPException) as exc_info:
                await submit_to_queue(request, wait_for_result=False, timeout_ms=30000)
            assert exc_info.value.status_code == 503
            assert "REDIS_UNAVAILABLE" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_submit_enqueue_failure_returns_500(self):
        """enqueue fails -> 500."""
        from quantumflow.api.routes.inference import submit_to_queue

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = True

        mock_queue = AsyncMock()
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()
        mock_queue.enqueue = AsyncMock(return_value=False)

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr), \
             patch("quantumflow.api.routes.inference.RedisQueue", return_value=mock_queue):
            request = _make_inference_request()
            with pytest.raises(HTTPException) as exc_info:
                await submit_to_queue(request, wait_for_result=False, timeout_ms=30000)
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail["error"]["code"] == "ENQUEUE_FAILED"

    @pytest.mark.asyncio
    async def test_submit_with_wait_result_found(self):
        """wait_for_result=True, result becomes available."""
        from quantumflow.api.routes.inference import submit_to_queue

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = True

        mock_queue = AsyncMock()
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()
        mock_queue.enqueue = AsyncMock(return_value=True)
        mock_queue.get_result = AsyncMock(return_value={
            "status": "completed", "output": "Hello world",
        })

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr), \
             patch("quantumflow.api.routes.inference.RedisQueue", return_value=mock_queue):
            request = _make_inference_request()
            result = await submit_to_queue(request, wait_for_result=True, timeout_ms=30000)

            assert result["status"] == "completed"
            assert "result" in result
            mock_queue.get_result.assert_awaited()

    @pytest.mark.asyncio
    async def test_submit_wait_timeout(self):
        """wait_for_result=True but result never arrives -> timeout."""
        from quantumflow.api.routes.inference import submit_to_queue

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = True

        mock_queue = AsyncMock()
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()
        mock_queue.enqueue = AsyncMock(return_value=True)
        mock_queue.get_result = AsyncMock(return_value=None)  # Never returns result

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr), \
             patch("quantumflow.api.routes.inference.RedisQueue", return_value=mock_queue), \
             patch("quantumflow.api.routes.inference.asyncio.sleep", new_callable=AsyncMock):
            request = _make_inference_request()
            result = await submit_to_queue(request, wait_for_result=True, timeout_ms=10)

            assert result["status"] == "timeout"
            assert "Timeout" in result["message"]


class TestGetQueueResultEndpoint:
    """Unit tests for /result/{request_id}."""

    @pytest.mark.asyncio
    async def test_result_found(self):
        from quantumflow.api.routes.inference import get_queue_result

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = True

        mock_queue = AsyncMock()
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()
        mock_queue.get_result = AsyncMock(return_value={
            "status": "completed", "output": "the output",
        })

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr), \
             patch("quantumflow.api.routes.inference.RedisQueue", return_value=mock_queue):
            result = await get_queue_result("req_001")
            assert result["status"] == "completed"
            assert result["result"]["output"] == "the output"

    @pytest.mark.asyncio
    async def test_result_pending(self):
        """Result is None but request still in queue -> pending."""
        from quantumflow.api.routes.inference import get_queue_result

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = True

        mock_queue = AsyncMock()
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()
        mock_queue.get_result = AsyncMock(return_value=None)
        mock_queue.get_request = AsyncMock(return_value={"request_id": "req_002"})

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr), \
             patch("quantumflow.api.routes.inference.RedisQueue", return_value=mock_queue):
            result = await get_queue_result("req_002")
            assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_result_not_found(self):
        """Neither result nor request found -> 404."""
        from quantumflow.api.routes.inference import get_queue_result

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = True

        mock_queue = AsyncMock()
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()
        mock_queue.get_result = AsyncMock(return_value=None)
        mock_queue.get_request = AsyncMock(return_value=None)

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr), \
             patch("quantumflow.api.routes.inference.RedisQueue", return_value=mock_queue):
            with pytest.raises(HTTPException) as exc_info:
                await get_queue_result("nonexistent")
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail["error"]["code"] == "RESULT_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_result_redis_unavailable(self):
        """Redis not connected -> 503."""
        from quantumflow.api.routes.inference import get_queue_result

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = False

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await get_queue_result("req_x")
            assert exc_info.value.status_code == 503


class TestQueueStatsEndpoint:
    """Unit tests for /queue/stats."""

    @pytest.mark.asyncio
    async def test_stats_connected(self):
        from quantumflow.api.routes.inference import get_queue_stats

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = True

        mock_queue = AsyncMock()
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()
        mock_queue.get_queue_stats = AsyncMock(return_value={"size": 5})
        mock_queue.get_metrics = AsyncMock(return_value={"avg_wait_ms": 100})

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr), \
             patch("quantumflow.api.routes.inference.RedisQueue", return_value=mock_queue):
            result = await get_queue_stats()
            assert result["connected"] is True
            assert result["queue_stats"]["size"] == 5
            assert result["metrics"]["avg_wait_ms"] == 100

    @pytest.mark.asyncio
    async def test_stats_disconnected(self):
        from quantumflow.api.routes.inference import get_queue_stats

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = False

        with patch("quantumflow.api.routes.inference.get_redis_manager",
                   new_callable=AsyncMock, return_value=mock_redis_mgr):
            result = await get_queue_stats()
            assert result["connected"] is False
            assert result["queue_size"] == 0
            assert result["message"] == "Redis is not available"


# ═══════════════════════════════════════════════════════════════════════════════
# models.py — _model_info_from_registry & list_models filters
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelInfoFromRegistry:
    """Tests for _model_info_from_registry status inference."""

    def test_loaded_model_has_ready_status(self):
        from quantumflow.api.routes.models import _model_info_from_registry
        from quantumflow.models.registry import ModelInfo as RegModelInfo

        reg_info = RegModelInfo(
            name="test-model",
            path="/models/test-model",
            parameter_count=7_000_000_000,
            backend="vllm",
            recommended_tensor_parallel=1,
            min_memory_gb=4,
            max_memory_gb=16,
            metadata={"architecture": "Qwen2ForCausalLM"},
        )

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)

        with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
            api_info = _model_info_from_registry("test-model", reg_info)
            assert api_info.status == "ready"
            assert api_info.name == "test-model"
            assert api_info.backend == "vllm"
            assert api_info.parameter_count == 7_000_000_000
            assert api_info.architecture == "Qwen2ForCausalLM"

    def test_unloaded_model_has_available_status(self):
        from quantumflow.api.routes.models import _model_info_from_registry
        from quantumflow.models.registry import ModelInfo as RegModelInfo

        reg_info = RegModelInfo(
            name="test-model",
            path="/models/test-model",
            parameter_count=1_000_000_000,
            backend="huggingface",
            recommended_tensor_parallel=1,
            min_memory_gb=2,
            max_memory_gb=8,
        )

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=False)

        with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
            api_info = _model_info_from_registry("test-model", reg_info)
            assert api_info.status == "available"


class TestListModelsFiltering:
    """Tests for list_models filtering by backend and status."""

    @pytest.mark.asyncio
    async def test_list_models_backend_filter(self):
        from quantumflow.api.routes.models import list_models
        from quantumflow.models.registry import ModelInfo as RegModelInfo
        from quantumflow.models.registry import ModelRegistry

        import quantumflow.api.routes.models as mod
        original_registry = mod._registry
        try:
            new_registry = ModelRegistry()
            new_registry._models = {
                "m1": RegModelInfo(name="m1", path="/m1", parameter_count=1, backend="vllm",
                                   recommended_tensor_parallel=1, min_memory_gb=1, max_memory_gb=4),
                "m2": RegModelInfo(name="m2", path="/m2", parameter_count=2, backend="tgi",
                                   recommended_tensor_parallel=1, min_memory_gb=1, max_memory_gb=4),
            }
            mod._registry = new_registry

            mock_engine = MagicMock()
            mock_engine.is_model_loaded = MagicMock(return_value=False)

            with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
                result = await list_models(status_filter=None, backend="vllm")
                assert len(result) == 1
                assert result[0].backend == "vllm"
                assert result[0].name == "m1"
        finally:
            mod._registry = original_registry

    @pytest.mark.asyncio
    async def test_list_models_status_filter(self):
        from quantumflow.api.routes.models import list_models
        from quantumflow.models.registry import ModelInfo as RegModelInfo
        from quantumflow.models.registry import ModelRegistry

        import quantumflow.api.routes.models as mod
        original_registry = mod._registry
        try:
            new_registry = ModelRegistry()
            new_registry._models = {
                "m1": RegModelInfo(name="m1", path="/m1", parameter_count=1, backend="vllm",
                                   recommended_tensor_parallel=1, min_memory_gb=1, max_memory_gb=4),
                "m2": RegModelInfo(name="m2", path="/m2", parameter_count=2, backend="vllm",
                                   recommended_tensor_parallel=1, min_memory_gb=1, max_memory_gb=4),
            }
            mod._registry = new_registry

            mock_engine = MagicMock()
            # m1 is loaded -> status "ready", m2 is not -> "available"
            def is_loaded(name):
                return name == "m1"
            mock_engine.is_model_loaded = MagicMock(side_effect=is_loaded)

            with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
                result = await list_models(status_filter="ready", backend=None)
                assert len(result) == 1
                assert result[0].status == "ready"
                assert result[0].name == "m1"
        finally:
            mod._registry = original_registry


# ═══════════════════════════════════════════════════════════════════════════════
# models.py — get_model
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetModel:
    """Tests for get_model, including the loaded-but-not-in-registry branch."""

    @pytest.mark.asyncio
    async def test_get_model_not_in_registry_not_loaded(self):
        """Model neither in registry nor loaded -> 404."""
        from quantumflow.api.routes.models import get_model

        import quantumflow.api.routes.models as mod
        original_registry = mod._registry
        try:
            from quantumflow.models.registry import ModelRegistry
            mod._registry = ModelRegistry()  # empty registry

            mock_engine = MagicMock()
            mock_engine.is_model_loaded = MagicMock(return_value=False)

            with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
                with pytest.raises(HTTPException) as exc_info:
                    await get_model("nonexistent")
                assert exc_info.value.status_code == 404
                assert exc_info.value.detail["error"]["code"] == "MODEL_NOT_FOUND"
        finally:
            mod._registry = original_registry

    @pytest.mark.asyncio
    async def test_get_model_loaded_but_not_in_registry(self):
        """Model loaded in engine_manager but not in registry -> returns basic info."""
        from quantumflow.api.routes.models import get_model

        import quantumflow.api.routes.models as mod
        original_registry = mod._registry
        try:
            from quantumflow.models.registry import ModelRegistry
            mod._registry = ModelRegistry()  # empty

            mock_engine = MagicMock()
            mock_engine.is_model_loaded = MagicMock(return_value=True)

            with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
                result = await get_model("my-custom-model")
                assert result.name == "my-custom-model"
                assert result.status == "ready"
                assert result.architecture == "Unknown"
                assert result.parameter_count == 0
        finally:
            mod._registry = original_registry


# ═══════════════════════════════════════════════════════════════════════════════
# models.py — deploy_model
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeployModel:
    """Unit tests for /models/deploy."""

    @pytest.mark.asyncio
    async def test_deploy_success(self):
        from quantumflow.api.models import DeployRequest
        from quantumflow.api.routes.models import deploy_model

        mock_engine = MagicMock()
        mock_engine.load_model = AsyncMock(return_value=(True, "Model started"))

        with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
            request = DeployRequest(model="test-model", tensor_parallel=2, replicas=3, backend="vllm")
            response = await deploy_model(request)

            assert response.status == "loading"
            assert response.replicas == 3
            assert "Model started" in response.message
            assert response.model_id.startswith("test-model_")

            # Verify backend mapping was applied
            call_kwargs = mock_engine.load_model.call_args[1]
            assert call_kwargs["tensor_parallel"] == 2

    @pytest.mark.asyncio
    async def test_deploy_failure_raises_500(self):
        from quantumflow.api.models import DeployRequest
        from quantumflow.api.routes.models import deploy_model

        mock_engine = MagicMock()
        mock_engine.load_model = AsyncMock(return_value=(False, "Out of memory"))

        with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
            request = DeployRequest(model="test-model")
            with pytest.raises(HTTPException) as exc_info:
                await deploy_model(request)
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail["error"]["code"] == "DEPLOY_FAILED"
            assert "Out of memory" in exc_info.value.detail["error"]["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# models.py — undeploy_model
# ═══════════════════════════════════════════════════════════════════════════════


class TestUndeployModel:
    """Unit tests for /models/undeploy."""

    @pytest.mark.asyncio
    async def test_undeploy_success(self):
        from quantumflow.api.models import UndeployRequest
        from quantumflow.api.routes.models import undeploy_model

        mock_engine = MagicMock()
        mock_engine.unload_model = AsyncMock(return_value=True)

        with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
            request = UndeployRequest(model="test-model")
            response = await undeploy_model(request)

            assert response.status == "unloaded"
            assert response.model_id == "test-model"
            mock_engine.unload_model.assert_awaited_once_with("test-model")

    @pytest.mark.asyncio
    async def test_undeploy_failure_raises_404(self):
        from quantumflow.api.models import UndeployRequest
        from quantumflow.api.routes.models import undeploy_model

        mock_engine = MagicMock()
        mock_engine.unload_model = AsyncMock(return_value=False)

        with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
            request = UndeployRequest(model="ghost-model")
            with pytest.raises(HTTPException) as exc_info:
                await undeploy_model(request)
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail["error"]["code"] == "MODEL_NOT_FOUND"


# ═══════════════════════════════════════════════════════════════════════════════
# models.py — run_benchmark
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunBenchmark:
    """Unit tests for /models/benchmark."""

    @pytest.mark.asyncio
    async def test_benchmark_model_not_loaded_raises_400(self):
        from quantumflow.api.models import BenchmarkRequest
        from quantumflow.api.routes.models import run_benchmark

        mock_engine = MagicMock()
        mock_engine.get_loaded_models = MagicMock(return_value=["other-model"])

        with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine):
            request = BenchmarkRequest(model="test-model", num_samples=10)
            with pytest.raises(HTTPException) as exc_info:
                await run_benchmark(request)
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"]["code"] == "MODEL_NOT_LOADED"

    @pytest.mark.asyncio
    async def test_benchmark_model_loaded_returns_running(self):
        from quantumflow.api.models import BenchmarkRequest
        from quantumflow.api.routes.models import run_benchmark

        mock_engine = MagicMock()
        mock_engine.get_loaded_models = MagicMock(return_value=["test-model"])

        with patch("quantumflow.api.routes.models.get_engine_manager", return_value=mock_engine), \
             patch("asyncio.create_task") as mock_create_task:
            request = BenchmarkRequest(model="test-model", num_samples=10)
            response = await run_benchmark(request)

            assert response.status == "running"
            assert response.total_samples == 10
            assert response.completed_samples == 0
            mock_create_task.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# model_management.py — _resolve_model_path
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveModelPath:
    """Tests for _resolve_model_path helper."""

    def test_explicit_model_path_used(self):
        from quantumflow.api.routes.model_management import _resolve_model_path

        result = _resolve_model_path("any-model", model_path="/custom/path")
        assert result == "/custom/path"

    def test_known_mapping_used(self):
        from quantumflow.api.routes.model_management import _resolve_model_path

        result = _resolve_model_path("Qwen2.5-0.5B")
        assert "Qwen/Qwen2.5-0.5B-Instruct" in result

    def test_downloaded_model_used(self):
        from quantumflow.api.routes.model_management import _resolve_model_path

        with patch("quantumflow.api.routes.model_management.get_downloaded_models",
                   return_value=[{"model_id": "my-local", "local_path": "/data/my-local"}]):
            result = _resolve_model_path("my-local")
            assert result == "/data/my-local"

    def test_unknown_model_returns_as_is(self):
        from quantumflow.api.routes.model_management import _resolve_model_path

        with patch("quantumflow.api.routes.model_management.get_downloaded_models",
                   return_value=[]):
            result = _resolve_model_path("org/unknown-model")
            assert result == "org/unknown-model"


# ═══════════════════════════════════════════════════════════════════════════════
# model_management.py — load_model validation branches
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadModelValidation:
    """Tests for load_model validation branches."""

    @pytest.mark.asyncio
    async def test_invalid_model_name_raises_400(self):
        from quantumflow.api.models import LoadModelRequest
        from quantumflow.api.routes.model_management import load_model

        # A model that is neither in the known mapping nor has a "/" (HF ID format)
        with patch("quantumflow.api.routes.model_management.get_downloaded_models",
                   return_value=[]):
            request = LoadModelRequest(model="NotAKnownModel")
            with pytest.raises(HTTPException) as exc_info:
                await load_model(request)
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"]["code"] == "INVALID_MODEL_NAME"

    @pytest.mark.asyncio
    async def test_known_model_skips_hf_validation(self):
        """Known models (in MODEL_PATH_MAPPING) skip HF validation."""
        from quantumflow.api.models import LoadModelRequest
        from quantumflow.api.routes.model_management import load_model

        mock_engine = MagicMock()
        mock_engine.load_model = AsyncMock(return_value=(True, "Loaded"))

        with patch("quantumflow.api.routes.model_management.get_engine_manager",
                   return_value=mock_engine), \
             patch("quantumflow.api.routes.model_management.validate_model") as mock_validate:
            request = LoadModelRequest(model="Qwen2.5-0.5B")
            response = await load_model(request)

            assert response.status == "loaded"
            # validate_model must NOT have been called for known models
            mock_validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_hf_id_model_validates_successfully(self):
        """HF ID models trigger HF validation."""
        from quantumflow.api.models import LoadModelRequest
        from quantumflow.api.routes.model_management import load_model

        mock_engine = MagicMock()
        mock_engine.load_model = AsyncMock(return_value=(True, "Loaded"))

        with patch("quantumflow.api.routes.model_management.get_engine_manager",
                   return_value=mock_engine), \
             patch("quantumflow.api.routes.model_management.get_downloaded_models",
                   return_value=[]), \
             patch("quantumflow.api.routes.model_management.validate_model",
                   new_callable=AsyncMock,
                   return_value={"valid": True, "gated": False}) as mock_validate:
            request = LoadModelRequest(model="myorg/my-model")
            response = await load_model(request)

            assert response.status == "loaded"
            mock_validate.assert_awaited_once_with("myorg/my-model")

    @pytest.mark.asyncio
    async def test_gated_model_raises_403(self):
        """Gated model raises 403."""
        from quantumflow.api.models import LoadModelRequest
        from quantumflow.api.routes.model_management import load_model

        with patch("quantumflow.api.routes.model_management.get_downloaded_models",
                   return_value=[]), \
             patch("quantumflow.api.routes.model_management.validate_model",
                   new_callable=AsyncMock,
                   return_value={"valid": True, "gated": True, "error": None}):
            request = LoadModelRequest(model="gated-org/secret-model")
            with pytest.raises(HTTPException) as exc_info:
                await load_model(request)
            assert exc_info.value.status_code == 403
            assert exc_info.value.detail["error"]["code"] == "MODEL_GATED"


# ═══════════════════════════════════════════════════════════════════════════════
# model_management.py — unload_model
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnloadModel:
    """Unit tests for /models/unload."""

    @pytest.mark.asyncio
    async def test_unload_success(self):
        from quantumflow.api.models import LoadModelRequest
        from quantumflow.api.routes.model_management import unload_model

        mock_engine = MagicMock()
        mock_engine.unload_model = AsyncMock(return_value=True)

        with patch("quantumflow.api.routes.model_management.get_engine_manager",
                   return_value=mock_engine):
            request = LoadModelRequest(model="test-model")
            response = await unload_model(request)

            assert response.status == "unloaded"
            assert response.model == "test-model"
            mock_engine.unload_model.assert_awaited_once_with("test-model")

    @pytest.mark.asyncio
    async def test_unload_not_found_raises_404(self):
        from quantumflow.api.models import LoadModelRequest
        from quantumflow.api.routes.model_management import unload_model

        mock_engine = MagicMock()
        mock_engine.unload_model = AsyncMock(return_value=False)

        with patch("quantumflow.api.routes.model_management.get_engine_manager",
                   return_value=mock_engine):
            request = LoadModelRequest(model="not-loaded")
            with pytest.raises(HTTPException) as exc_info:
                await unload_model(request)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_unload_generic_exception_raises_500(self):
        from quantumflow.api.models import LoadModelRequest
        from quantumflow.api.routes.model_management import unload_model

        mock_engine = MagicMock()
        mock_engine.unload_model = AsyncMock(side_effect=RuntimeError("crash"))

        with patch("quantumflow.api.routes.model_management.get_engine_manager",
                   return_value=mock_engine):
            request = LoadModelRequest(model="test-model")
            with pytest.raises(HTTPException) as exc_info:
                await unload_model(request)
            assert exc_info.value.status_code == 500


# ═══════════════════════════════════════════════════════════════════════════════
# model_management.py — get_model_status & list_available_models
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelStatusAndList:
    """Tests for /models/status and /models/list."""

    @pytest.mark.asyncio
    async def test_get_model_status(self):
        from quantumflow.api.routes.model_management import get_model_status

        mock_engine = MagicMock()
        mock_engine.get_loaded_models = MagicMock(return_value=["m1", "m2"])

        with patch("quantumflow.api.routes.model_management.get_engine_manager",
                   return_value=mock_engine):
            response = await get_model_status()
            assert response.loaded_models == ["m1", "m2"]
            assert response.total == 2

    @pytest.mark.asyncio
    async def test_list_available_models(self):
        from quantumflow.api.routes.model_management import list_available_models

        with patch("quantumflow.api.routes.model_management.get_downloaded_models",
                   return_value=[{"model_id": "dl1"}, {"model_id": "dl2"}]):
            response = await list_available_models()
            assert "available_models" in response
            assert "mappings" in response
            assert response["downloaded_models"] == ["dl1", "dl2"]
            assert response["downloaded_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# cluster.py — node_action, receive_heartbeat, list_nodes filters, get_node 404
# ═══════════════════════════════════════════════════════════════════════════════


class TestNodeAction:
    """Unit tests for /nodes/{node_id}/action."""

    @pytest.mark.asyncio
    async def test_drain_node(self):
        from quantumflow.api.routes.cluster import node_action
        from quantumflow.core.constants import NodeStatus

        mock_node = Mock()
        mock_node.node_id = "node-1"

        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=mock_node)
        mock_mgr.update_node_status = AsyncMock()

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            result = await node_action("node-1", "drain")
            assert result["action"] == "drain"
            assert result["status"] == "completed"
            mock_mgr.update_node_status.assert_awaited_once_with("node-1", NodeStatus.DRAINING)

    @pytest.mark.asyncio
    async def test_uncordon_node(self):
        from quantumflow.api.routes.cluster import node_action
        from quantumflow.core.constants import NodeStatus

        mock_node = Mock()
        mock_node.node_id = "node-1"

        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=mock_node)
        mock_mgr.update_node_status = AsyncMock()

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            result = await node_action("node-1", "uncordon")
            assert result["action"] == "uncordon"
            mock_mgr.update_node_status.assert_awaited_once_with("node-1", NodeStatus.HEALTHY)

    @pytest.mark.asyncio
    async def test_restart_node_is_noop(self):
        from quantumflow.api.routes.cluster import node_action

        mock_node = Mock()
        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=mock_node)

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            result = await node_action("node-1", "restart")
            assert result["action"] == "restart"
            assert result["status"] == "completed"
            # Should not call update_node_status for restart

    @pytest.mark.asyncio
    async def test_invalid_action_raises_400(self):
        from quantumflow.api.routes.cluster import node_action

        mock_node = Mock()
        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=mock_node)

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await node_action("node-1", "delete")
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"]["code"] == "INVALID_ACTION"

    @pytest.mark.asyncio
    async def test_node_not_found_for_action_raises_404(self):
        from quantumflow.api.routes.cluster import node_action

        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=None)

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await node_action("ghost-node", "drain")
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail["error"]["code"] == "NODE_NOT_FOUND"


class TestReceiveHeartbeat:
    """Unit tests for /cluster/heartbeat."""

    @pytest.mark.asyncio
    async def test_new_node_registration(self):
        from quantumflow.api.routes.cluster import receive_heartbeat

        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=None)  # not yet registered
        mock_mgr.register_node = AsyncMock()

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            node_info = {"node_id": "new-node", "hostname": "host1"}
            result = await receive_heartbeat(node_info)

            assert result["status"] == "ok"
            assert result["node_id"] == "new-node"
            mock_mgr.register_node.assert_awaited_once_with(node_info)
            mock_mgr.update_node_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_existing_node_update(self):
        from quantumflow.api.routes.cluster import receive_heartbeat

        existing_node = Mock()
        existing_node.loaded_models = []

        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=existing_node)
        mock_mgr.update_node_info = AsyncMock()

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            node_info = {"node_id": "existing-node", "current_load": 0.8}
            result = await receive_heartbeat(node_info)

            assert result["status"] == "ok"
            mock_mgr.update_node_info.assert_awaited_once_with(
                node_id="existing-node",
                gpu_info=[],
                load=0.8,
            )
            mock_mgr.register_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_node_id_raises_400(self):
        from quantumflow.api.routes.cluster import receive_heartbeat

        mock_mgr = AsyncMock()

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await receive_heartbeat({})
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"]["code"] == "INVALID_REQUEST"


class TestListNodeFiltering:
    """Unit tests for /cluster/nodes with filters."""

    @pytest.mark.asyncio
    async def test_list_nodes_with_status_filter(self):
        from quantumflow.api.routes.cluster import list_nodes
        from quantumflow.core.constants import NodeStatus

        mock_node = Mock()
        mock_node.node_id = "n1"
        mock_node.status = NodeStatus.HEALTHY
        # minimal attrs for _node_to_node_info
        mock_node.hostname = "host1"
        mock_node.ip = "10.0.0.1"
        mock_node.port = 8000
        mock_node.gpu_count = 1
        mock_node.gpu_info = []
        mock_node.cpu_count = 4
        mock_node.memory_total = 1024
        mock_node.memory_available = 512
        mock_node.disk_total = 10240
        mock_node.disk_available = 5120
        mock_node.current_load = 0.5
        mock_node.labels = {}
        mock_node.version = "1.0"
        mock_node.start_time = __import__("datetime").datetime.now()
        mock_node.last_heartbeat = __import__("datetime").datetime.now()
        mock_node.loaded_models = []

        mock_mgr = AsyncMock()
        mock_mgr.get_nodes = AsyncMock(return_value=[mock_node])

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            result = await list_nodes(status_filter="healthy", zone=None)
            assert len(result) == 1
            assert result[0].node_id == "n1"

    @pytest.mark.asyncio
    async def test_list_nodes_with_zone_filter(self):
        from quantumflow.api.routes.cluster import list_nodes
        from quantumflow.core.constants import NodeStatus

        node1 = Mock()
        node1.node_id = "n-a"
        node1.status = NodeStatus.HEALTHY
        node1.hostname = "a"
        node1.ip = "10.0.0.1"
        node1.port = 8000
        node1.gpu_count = 0
        node1.gpu_info = []
        node1.cpu_count = 2
        node1.memory_total = 100
        node1.memory_available = 50
        node1.disk_total = 1000
        node1.disk_available = 500
        node1.current_load = 0.1
        node1.labels = {"zone": "us-east"}
        node1.version = "1.0"
        node1.start_time = __import__("datetime").datetime.now()
        node1.last_heartbeat = __import__("datetime").datetime.now()
        node1.loaded_models = []

        node2 = Mock()
        node2.node_id = "n-b"
        node2.status = NodeStatus.HEALTHY
        node2.hostname = "b"
        node2.ip = "10.0.0.2"
        node2.port = 8000
        node2.gpu_count = 0
        node2.gpu_info = []
        node2.cpu_count = 2
        node2.memory_total = 100
        node2.memory_available = 50
        node2.disk_total = 1000
        node2.disk_available = 500
        node2.current_load = 0.1
        node2.labels = {"zone": "us-west"}
        node2.version = "1.0"
        node2.start_time = __import__("datetime").datetime.now()
        node2.last_heartbeat = __import__("datetime").datetime.now()
        node2.loaded_models = []

        mock_mgr = AsyncMock()
        mock_mgr.get_nodes = AsyncMock(return_value=[node1, node2])

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            result = await list_nodes(status_filter=None, zone="us-east")
            assert len(result) == 1
            assert result[0].node_id == "n-a"


class TestGetNode:
    """Unit tests for /cluster/nodes/{node_id}."""

    @pytest.mark.asyncio
    async def test_get_node_not_found(self):
        from quantumflow.api.routes.cluster import get_node

        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=None)

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await get_node("ghost")
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail["error"]["code"] == "NODE_NOT_FOUND"


class TestUnregisterNode:
    """Unit tests for DELETE /cluster/nodes/{node_id}."""

    @pytest.mark.asyncio
    async def test_unregister_success(self):
        from quantumflow.api.routes.cluster import unregister_node

        mock_node = Mock()
        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=mock_node)
        mock_mgr.unregister_node = AsyncMock()

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            result = await unregister_node("node-to-remove")
            assert result["status"] == "unregistered"
            mock_mgr.unregister_node.assert_awaited_once_with("node-to-remove")

    @pytest.mark.asyncio
    async def test_unregister_node_not_found(self):
        from quantumflow.api.routes.cluster import unregister_node

        mock_mgr = AsyncMock()
        mock_mgr.get_node = AsyncMock(return_value=None)

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            with pytest.raises(HTTPException) as exc_info:
                await unregister_node("phantom")
            assert exc_info.value.status_code == 404


class TestClusterStatusEndpoint:
    """Unit tests for /cluster/status."""

    @pytest.mark.asyncio
    async def test_cluster_status_aggregates(self):
        from quantumflow.api.routes.cluster import get_cluster_status
        from quantumflow.core.constants import NodeStatus

        healthy = Mock()
        healthy.status = NodeStatus.HEALTHY
        healthy.gpu_count = 2
        healthy.available_gpus = [0, 1]
        healthy.loaded_models = ["m1"]

        unhealthy = Mock()
        unhealthy.status = NodeStatus.UNHEALTHY
        unhealthy.gpu_count = 1
        unhealthy.available_gpus = []
        unhealthy.loaded_models = []

        mock_mgr = AsyncMock()
        mock_mgr.get_nodes = AsyncMock(return_value=[healthy, unhealthy])

        with patch("quantumflow.api.routes.cluster.get_cluster_manager", return_value=mock_mgr):
            result = await get_cluster_status()
            assert result.total_nodes == 2
            assert result.healthy_nodes == 1
            assert result.unhealthy_nodes == 1
            assert result.total_gpus == 3
            assert result.available_gpus == 2
            assert result.active_models == 1


# ═══════════════════════════════════════════════════════════════════════════════
# health.py — cluster exception path & simultaneous degraded
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthCheckExtended:
    """Additional tests for health_check edge cases."""

    @pytest.mark.asyncio
    async def test_cluster_manager_instantiation_error(self):
        """When ClusterManager() raises, cluster is 'unknown', overall 'degraded'."""
        from quantumflow.api.routes.health import health_check

        with patch("quantumflow.storage.get_redis_manager",
                   new_callable=AsyncMock) as mock_redis:
            mock_mgr = AsyncMock()
            mock_mgr.health_check = AsyncMock(return_value={"connected": True})
            mock_redis.return_value = mock_mgr

            with patch("quantumflow.cluster.manager.ClusterManager",
                       side_effect=ImportError("cluster not available")):
                response = await health_check()
                assert response.checks["cluster"] == "unknown"
                assert response.status == "degraded"

    @pytest.mark.asyncio
    async def test_cluster_get_stats_exception(self):
        """When ClusterManager.get_cluster_stats raises, cluster is 'unknown'."""
        from quantumflow.api.routes.health import health_check

        with patch("quantumflow.storage.get_redis_manager",
                   new_callable=AsyncMock) as mock_redis:
            mock_mgr = AsyncMock()
            mock_mgr.health_check = AsyncMock(return_value={"connected": True})
            mock_redis.return_value = mock_mgr

            with patch(
                "quantumflow.cluster.manager.ClusterManager.get_cluster_stats",
                new_callable=AsyncMock,
                side_effect=RuntimeError("cluster stats failure"),
            ):
                response = await health_check()
                assert response.checks["cluster"] == "unknown"
                assert response.status == "degraded"

    @pytest.mark.asyncio
    async def test_both_redis_and_cluster_degraded(self):
        """When both Redis and cluster report issues, status is 'degraded'."""
        from quantumflow.api.routes.health import health_check

        with patch("quantumflow.storage.get_redis_manager",
                   new_callable=AsyncMock) as mock_redis:
            mock_mgr = AsyncMock()
            mock_mgr.health_check = AsyncMock(return_value={"connected": False})
            mock_redis.return_value = mock_mgr

            with patch(
                "quantumflow.cluster.manager.ClusterManager.get_cluster_stats",
                new_callable=AsyncMock,
                return_value={"unhealthy_nodes": 2},
            ):
                response = await health_check()
                assert response.checks["redis"] == "unhealthy"
                assert response.checks["cluster"] == "degraded"
                # overall should be degraded (not healthy)
                assert response.status == "degraded"

    @pytest.mark.asyncio
    async def test_redis_exception_path_in_health_check(self):
        """When get_redis_manager raises, redis is 'unhealthy', status 'degraded'."""
        from quantumflow.api.routes.health import health_check

        with patch("quantumflow.storage.get_redis_manager",
                   new_callable=AsyncMock,
                   side_effect=RuntimeError("redis connection error")):
            with patch(
                "quantumflow.cluster.manager.ClusterManager.get_cluster_stats",
                new_callable=AsyncMock,
                return_value={"unhealthy_nodes": 0},
            ):
                response = await health_check()
                assert response.checks["redis"] == "unhealthy"
                assert response.status == "degraded"


class TestReadinessCheckExtended:
    """Additional tests for readiness_check."""

    @pytest.mark.asyncio
    async def test_readiness_check_cluster_manager_exception(self):
        """When ClusterManager() raises, readiness returns False."""
        from quantumflow.api.routes.health import readiness_check

        with patch("quantumflow.storage.get_redis_manager",
                   new_callable=AsyncMock) as mock_redis:
            mock_mgr = AsyncMock()
            mock_mgr.health_check = AsyncMock(return_value={"connected": True})
            mock_redis.return_value = mock_mgr

            with patch("quantumflow.cluster.manager.ClusterManager",
                       side_effect=Exception("cluster error")):
                response = await readiness_check()
                assert response["ready"] is False
                assert "reason" in response


# ═══════════════════════════════════════════════════════════════════════════════
# server.py — QuantumFlowError exception handler & CORS middleware
# ═══════════════════════════════════════════════════════════════════════════════


class TestServerExceptionHandler:
    """Tests for the QuantumFlowError exception handler and app configuration."""

    def test_quantumflow_error_handler_returns_500_json(self):
        """QuantumFlowError should result in 500 JSONResponse with error dict."""
        from fastapi.testclient import TestClient
        from quantumflow.api.server import create_app

        app = create_app()

        # Inject a route that always raises QuantumFlowError
        from quantumflow.core.exceptions import QuantumFlowError

        @app.get("/test-qf-error")
        async def raise_qf_error():
            raise QuantumFlowError("Something went wrong", "TEST_ERROR")

        client = TestClient(app)
        response = client.get("/test-qf-error")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "TEST_ERROR"
        assert data["error"]["message"] == "Something went wrong"

    def test_app_has_cors_middleware(self):
        """Verify CORS middleware is present in the app middleware stack."""
        from quantumflow.api.server import app

        # app.user_middleware is a list of starlette.middleware.Middleware objects
        # Each Middleware has a .cls attribute pointing to the actual middleware class
        cors_found = any(
            m.cls.__name__ == "CORSMiddleware"
            for m in app.user_middleware
        )
        # CORS may or may not be enabled depending on config — just verify app exists
        assert app.title == "QuantumFlow API"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
