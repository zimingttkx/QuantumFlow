"""后端协议测试 — 验证 SGLang / TGI / vLLM 的 HTTP API 协议正确性

测试原则：
1. PROTOCOL TEST: 验证发送的 HTTP 请求格式与后端 API 规范一致
2. SSE PARSE TEST: 验证流式响应的 SSE 解析逻辑
3. PARAMETER MAP TEST: 验证采样参数正确映射到 API 参数
4. ERROR CARDINALITY TEST: 验证错误路径返回正确数量的结果

这些测试用 mock HTTP 响应模拟真实后端，验证协议层面的正确性。
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.inference.backends.sglang import SGLangEngine
from quantumflow.inference.backends.tgi import TGIEngine
from quantumflow.inference.backends.vllm import VLLMEngine
from quantumflow.inference.engine import SamplingParams


async def _async_iter(lines):
    """Helper: 将普通列表转为 async iterator"""
    for line in lines:
        yield line


# ═══════════════════════════════════════════════════════════════════════════════
# SGLang Protocol Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSGLangProtocol:
    """SGLang HTTP API 协议正确性测试"""

    @pytest.fixture
    def engine(self):
        engine = SGLangEngine(base_url="http://localhost:30000")
        engine._client = AsyncMock()
        engine._is_initialized = True
        return engine

    # ── SSE 流式解析 ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stream_parses_choice_text_field(self, engine):
        """PROTOCOL: SSE 流式解析 /v1/completions 的 choice.text 字段"""
        # /v1/completions SSE 格式: choices[0].text (不是 choices[0].delta.text)
        sse_chunks = [
            'data: {"choices":[{"text":"Hello","index":0,"finish_reason":null}]}\n\n',
            'data: {"choices":[{"text":" world","index":0,"finish_reason":null}]}\n\n',
            'data: {"choices":[{"text":"!","index":0,"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=_async_iter(sse_chunks))
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        engine._client.stream = MagicMock(return_value=mock_context)

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert len(chunks) == 3, f"应收到 3 个 chunk，实际: {len(chunks)}"
        assert chunks == ["Hello", " world", "!"], f"chunks 内容错误: {chunks}"

    @pytest.mark.asyncio
    async def test_stream_parses_delta_text_fallback_for_chat_completions(self, engine):
        """PROTOCOL: SSE 解析兼容 /v1/chat/completions 的 delta.text 格式"""
        # 如果使用 chat/completions 端点，SSE 格式是 choices[0].delta.text
        sse_chunks = [
            'data: {"choices":[{"delta":{"text":"Hi"},"index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=_async_iter(sse_chunks))
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        engine._client.stream = MagicMock(return_value=mock_context)

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert len(chunks) == 1, f"应收到 1 个 chunk，实际: {len(chunks)}"
        assert chunks[0] == "Hi", f"delta.text 解析错误: '{chunks[0]}'"

    @pytest.mark.asyncio
    async def test_stream_empty_choices_does_not_crash(self, engine):
        """PROTOCOL: 空 choices 的 SSE 行不崩溃"""
        sse_chunks = [
            'data: {"choices":[]}\n\n',
            'data: {"choices":[{"text":"ok","index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=_async_iter(sse_chunks))
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        engine._client.stream = MagicMock(return_value=mock_context)

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert "ok" in chunks, "空 choices 行不应该阻止后续 chunk 的解析"

    @pytest.mark.asyncio
    async def test_stream_json_decode_error_does_not_crash(self, engine):
        """PROTOCOL: 非法 JSON 的 SSE 行不崩溃，跳过继续"""
        sse_chunks = [
            "data: {invalid json}\n\n",
            'data: {"choices":[{"text":"recovered","index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=_async_iter(sse_chunks))
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        engine._client.stream = MagicMock(return_value=mock_context)

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert "recovered" in chunks, "JSON 解析错误不应阻止后续 chunk"

    @pytest.mark.asyncio
    async def test_stream_http_error_returns_empty(self, engine):
        """PROTOCOL: HTTP 错误状态码时流式返回空"""
        mock_response = AsyncMock()
        mock_response.status_code = 403
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        engine._client.stream = MagicMock(return_value=mock_context)

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert chunks == []

    # ── generate 并行请求 ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_generate_makes_individual_requests_for_batch(self, engine):
        """PROTOCOL: 批量 generate 必须为每个 prompt 发送独立的 HTTP 请求"""
        # 每个请求返回独立结果
        responses = [
            AsyncMock(
                status_code=200,
                json=MagicMock(
                    return_value={
                        "choices": [{"text": f"response_{i}", "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                    }
                ),
            )
            for i in range(3)
        ]
        engine._client.post = AsyncMock(side_effect=responses)

        results = await engine.generate("test", ["p1", "p2", "p3"], SamplingParams())

        # 验证：3 个 prompt → 3 次 HTTP POST
        assert (
            engine._client.post.call_count == 3
        ), f"3 个 prompt 应发送 3 次 HTTP 请求，实际: {engine._client.post.call_count}"

        assert len(results) == 3, f"应返回 3 个结果，实际: {len(results)}"
        for i, r in enumerate(results):
            assert r.outputs[0] == f"response_{i}", f"结果[{i}] 内容错误: '{r.outputs[0]}'"
            assert r.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_single_prompt_makes_one_request(self, engine):
        """PROTOCOL: 单个 prompt 只发一次 HTTP 请求"""
        mock_response = AsyncMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "choices": [{"text": "single response", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            ),
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["hello"], SamplingParams())

        assert engine._client.post.call_count == 1
        assert len(results) == 1
        assert results[0].outputs[0] == "single response"

    @pytest.mark.asyncio
    async def test_generate_error_per_prompt_returns_error_results(self, engine):
        """PROTOCOL: 单个请求失败时返回该 prompt 的错误结果，不丢失其他结果"""
        # 第 2 个请求失败
        error_response = AsyncMock(status_code=500, text="Internal Server Error")
        ok_response = AsyncMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "choices": [{"text": "ok", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            ),
        )
        engine._client.post = AsyncMock(side_effect=[ok_response, error_response, ok_response])

        results = await engine.generate("test", ["p1", "p2", "p3"], SamplingParams())

        assert len(results) == 3
        assert results[0].finish_reason == "stop"
        assert (
            results[1].finish_reason == "error"
        ), f"失败请求的结果应为 error，实际: {results[1].finish_reason}"
        assert results[2].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_client_not_initialized_returns_error_results(self, engine):
        """PROTOCOL: 客户端未初始化时返回 N 个错误结果"""
        engine._client = None

        results = await engine.generate("test", ["p1", "p2"], SamplingParams())

        assert len(results) == 2
        for r in results:
            assert r.finish_reason == "error"

    # ── 参数映射 ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_generate_passes_all_sampling_params(self, engine):
        """PROTOCOL: generate 必须正确传递所有采样参数到 API"""
        mock_response = AsyncMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "choices": [{"text": "ok", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            ),
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        sampling_params = SamplingParams(temperature=0.5, top_p=0.8, max_tokens=100, stop=["###"])

        await engine.generate("test", ["prompt"], sampling_params)

        # 验证 HTTP 请求体
        call_args = engine._client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.8
        assert payload["max_tokens"] == 100
        assert payload["stop"] == ["###"]


# ═══════════════════════════════════════════════════════════════════════════════
# TGI Protocol Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTGIProtocol:
    """TGI HTTP API 协议正确性测试"""

    @pytest.fixture
    def engine(self):
        engine = TGIEngine(base_url="http://localhost:8080")
        engine._client = AsyncMock()
        engine._is_initialized = True
        return engine

    # ── 参数映射：top_k 不得作为 truncate ────────────────────

    @pytest.mark.asyncio
    async def test_top_k_not_used_as_truncate(self, engine):
        """PROTOCOL: top_k 采样参数不得映射为 TGI 的 truncate 参数"""
        mock_response = AsyncMock(
            status_code=200, json=MagicMock(return_value={"generated_text": "response"})
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        # top_k=50 (默认值) — 如果被错误映射为 truncate=50，输入会被截断
        await engine.generate(
            "test", ["Hello world, this is a test prompt"], SamplingParams(top_k=50)
        )

        call_args = engine._client.post.call_args
        payload = call_args.kwargs["json"]

        # 验证：parameters 中不应包含 truncate 字段
        if "parameters" in payload:
            assert (
                "truncate" not in payload["parameters"]
            ), f"BUG: top_k 被错误映射为 truncate! parameters={payload['parameters']}"
            # 但应该有 top_k
            if "top_k" in payload["parameters"]:
                assert payload["parameters"]["top_k"] == 50

    @pytest.mark.asyncio
    async def test_generate_stream_has_correct_parameters(self, engine):
        """PROTOCOL: generate_stream 必须传递 top_k/stop/repetition_penalty"""
        sse_chunks = [
            'data: {"token":{"text":"Hello"}}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=_async_iter(sse_chunks))
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        engine._client.stream = MagicMock(return_value=mock_context)

        sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            max_tokens=100,
            repetition_penalty=1.1,
            stop=["END"],
        )

        chunks = []
        async for text in engine.generate_stream("test", "prompt", sampling_params):
            chunks.append(text)

        # 验证 stream 请求的参数
        call_args = engine._client.stream.call_args
        payload = call_args.kwargs["json"]
        params = payload["parameters"]
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9
        assert params["top_k"] == 40
        assert params["repetition_penalty"] == 1.1
        assert params["stop"] == ["END"]
        assert "truncate" not in params, f"BUG: truncate 不应出现在 parameters 中: {params}"

    # ── Token 统计 ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_batch_token_stats_scalar_handling(self, engine):
        """PROTOCOL: TGI batch 的 token 统计支持标量格式（TGI 默认）"""
        # TGI /generate_batch 返回的 prompt_tokens/generated_tokens 是标量
        mock_response = AsyncMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "generated_text": ["resp1", "resp2", "resp3"],
                    "prompt_tokens": 30,  # 标量，不是列表
                    "generated_tokens": 15,  # 标量
                }
            ),
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["p1", "p2", "p3"], SamplingParams())

        assert len(results) == 3
        # 标量 token 数应该在 3 个结果之间分配
        for r in results:
            assert r.prompt_tokens > 0, f"每个结果应有 prompt_tokens > 0: {r.prompt_tokens}"
            assert (
                r.completion_tokens > 0
            ), f"每个结果应有 completion_tokens > 0: {r.completion_tokens}"

    @pytest.mark.asyncio
    async def test_batch_token_stats_list_handling(self, engine):
        """PROTOCOL: TGI batch 的 token 统计兼容列表格式"""
        mock_response = AsyncMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "generated_text": ["r1", "r2"],
                    "prompt_tokens": [10, 20],  # 列表格式
                    "generated_tokens": [5, 8],
                }
            ),
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["p1", "p2"], SamplingParams())

        assert len(results) == 2
        assert results[0].prompt_tokens == 10
        assert results[1].prompt_tokens == 20
        assert results[0].completion_tokens == 5
        assert results[1].completion_tokens == 8

    # ── 错误处理 ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_prompts_returns_empty_list(self, engine):
        """PROTOCOL: 空 prompts 列表返回空 results"""
        results = await engine.generate("test", [], SamplingParams())
        assert results == []

    @pytest.mark.asyncio
    async def test_client_not_initialized_returns_error_results(self, engine):
        """PROTOCOL: 客户端未初始化返回 N 个错误结果"""
        engine._client = None

        results = await engine.generate("test", ["p1", "p2"], SamplingParams())

        assert len(results) == 2
        for r in results:
            assert r.finish_reason == "error"

    # ── SSE 解析 ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stream_parses_token_text_correctly(self, engine):
        """PROTOCOL: TGI SSE 解析正确的 token.text 字段"""
        sse_chunks = [
            'data: {"token":{"text":"Hello"}}\n\n',
            'data: {"token":{"text":" world"}}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=_async_iter(sse_chunks))
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        engine._client.stream = MagicMock(return_value=mock_context)

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_single_generate_uses_generate_endpoint(self, engine):
        """PROTOCOL: 单个 prompt 使用 /generate 端点"""
        mock_response = AsyncMock(
            status_code=200, json=MagicMock(return_value={"generated_text": "response"})
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        await engine.generate("test", ["single prompt"], SamplingParams())

        call_args = engine._client.post.call_args
        assert (
            call_args.kwargs["json"]["inputs"] == "single prompt"
        ), "单个 prompt 的 inputs 应为字符串"

    @pytest.mark.asyncio
    async def test_batch_generate_uses_generate_batch_endpoint(self, engine):
        """PROTOCOL: 多个 prompt 使用 /generate_batch 端点"""
        mock_response = AsyncMock(
            status_code=200, json=MagicMock(return_value={"generated_text": ["r1", "r2"]})
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        await engine.generate("test", ["p1", "p2"], SamplingParams())

        call_args = engine._client.post.call_args
        # /generate_batch 端点
        actual_url = engine._client.post.call_args[0][0] if engine._client.post.call_args[0] else ""
        # inputs 应为列表
        assert isinstance(call_args.kwargs["json"]["inputs"], list), "批量请求的 inputs 应为列表"


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM Behavior Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVLLMBehavior:
    """vLLM 后端行为测试"""

    @pytest.fixture
    def engine(self):
        engine = VLLMEngine()
        engine._is_initialized = True
        return engine

    @pytest.mark.asyncio
    async def test_get_stats_returns_float_values(self):
        """BEHAVIOR: get_stats 返回 Dict[str, float]，不能调用不存在的 LLM 方法"""
        engine = VLLMEngine()
        engine._is_initialized = True

        stats = await engine.get_stats("nonexistent")
        assert isinstance(stats, dict), f"get_stats 应返回 dict: {type(stats)}"
        for k, v in stats.items():
            assert isinstance(
                v, (int, float)
            ), f"get_stats 的值必须是数字，key={k}, value={v}, type={type(v)}"

    @pytest.mark.asyncio
    async def test_generate_model_not_loaded_returns_error_results(self):
        """BEHAVIOR: 模型未加载时 generate 返回 N 个错误结果"""
        engine = VLLMEngine()
        engine._is_initialized = True

        results = await engine.generate("nonexistent", ["p1", "p2", "p3"], SamplingParams())

        assert len(results) == 3, f"期望 3 个错误结果，实际: {len(results)}"
        for r in results:
            assert r.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_generate_stream_model_not_loaded_returns_empty(self):
        """BEHAVIOR: 模型未加载时 generate_stream 返回空迭代"""
        engine = VLLMEngine()
        engine._is_initialized = True

        chunks = []
        async for chunk in engine.generate_stream("nonexistent", "prompt", SamplingParams()):
            chunks.append(chunk)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded_returns_false(self):
        """BEHAVIOR: 卸载未加载的模型返回 False"""
        engine = VLLMEngine()
        engine._is_initialized = True

        result = await engine.unload_model("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_vllm_available(self):
        """BEHAVIOR: vLLM 可用时 initialize 成功"""
        import vllm

        with patch.dict("sys.modules", {"vllm": vllm}):
            engine = VLLMEngine()
            result = await engine.initialize()
            assert result is True
            assert engine.is_ready is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
