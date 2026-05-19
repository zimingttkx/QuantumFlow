"""SGLang 后端行为测试 — 严格验证 SGLangEngine 的业务逻辑正确性

测试原则：
1. 业务逻辑验证优先于运行可用性
2. 强精准断言：每个功能点的预期返回值必须严格比对
3. 全覆盖：正常用例、边界值、非法入参、异常场景
4. 错误定位精准：区分运行报错、逻辑错误、流程偏差
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.inference.backends.sglang import SGLangEngine
from quantumflow.inference.engine import ModelConfig, SamplingParams

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine():
    """创建已初始化的 SGLang 引擎"""
    engine = SGLangEngine(base_url="http://localhost:30000")
    engine._client = MagicMock()  # 使用 MagicMock 而非 AsyncMock
    engine._is_initialized = True
    return engine


@pytest.fixture
def model_config():
    """标准模型配置"""
    return ModelConfig(
        model_name="test-sglang-model",
        model_path="sglang-test/path",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _mock_sse_response(chunks, status_code=200):
    """创建模拟 SSE 响应"""
    mock_response = MagicMock()
    mock_response.status_code = status_code

    async def aiter_lines():
        for chunk in chunks:
            yield chunk

    # 注意：aiter_lines() 被调用时返回 async generator
    mock_response.aiter_lines = aiter_lines

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    return mock_context


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：初始化与生命周期
# ═══════════════════════════════════════════════════════════════════════════════


class TestSGLangInitialization:
    """验证 SGLang 引擎初始化和生命周期管理"""

    @pytest.mark.asyncio
    async def test_initialize_success_with_healthy_response(self):
        """[正常用例] 健康检查成功时 initialize 返回 True"""
        engine = SGLangEngine(base_url="http://localhost:30000")

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await engine.initialize()

        assert result is True, "健康检查返回200时应初始化成功"
        assert engine._is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure_when_health_check_fails(self):
        """[错误用例] 健康检查失败时 initialize 返回 False"""
        engine = SGLangEngine(base_url="http://localhost:30000")

        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await engine.initialize()

        assert result is False, "健康检查失败时应初始化失败"

    @pytest.mark.asyncio
    async def test_initialize_failure_when_httpx_not_installed(self):
        """[异常用例] httpx 未安装时 initialize 返回 False"""
        engine = SGLangEngine(base_url="http://localhost:30000")

        with patch.dict("sys.modules", {"httpx": None}):
            result = await engine.initialize()

        assert result is False, "httpx导入失败时应返回False"

    @pytest.mark.asyncio
    async def test_close_closes_http_client(self, engine):
        """[正常用例] close() 正确关闭 HTTP 客户端"""
        engine._client = MagicMock()
        engine._client.aclose = AsyncMock()

        await engine.close()

        assert engine._client.aclose.called, "close() 应关闭客户端"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：模型加载与卸载
# ═══════════════════════════════════════════════════════════════════════════════


class TestSGLangModelLifecycle:
    """验证模型的加载/卸载状态管理"""

    @pytest.mark.asyncio
    async def test_load_model_tracks_model_from_v1_models(self, engine, model_config):
        """[正常用例] load_model 从 /v1/models 端点验证并跟踪模型"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "data": [
                    {"id": "test-sglang-model", "object": "model"},
                ]
            }
        )
        engine._client.get = AsyncMock(return_value=mock_response)

        result = await engine.load_model(model_config)

        assert result is True, "模型ID匹配时应加载成功"
        assert model_config.model_name in engine._loaded_models

    @pytest.mark.asyncio
    async def test_load_model_without_client_returns_false(self, engine, model_config):
        """[错误用例] 客户端未初始化时 load_model 返回 False"""
        engine._client = None

        result = await engine.load_model(model_config)

        assert result is False, "客户端未初始化时应返回False"

    @pytest.mark.asyncio
    async def test_load_model_accepts_any_when_server_running(self, engine, model_config):
        """[边界用例] 服务器运行但模型列表为空时，容错假设模型可加载"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"data": []})
        engine._client.get = AsyncMock(return_value=mock_response)

        result = await engine.load_model(model_config)

        assert result is True

    @pytest.mark.asyncio
    async def test_unload_model_removes_from_tracking(self, engine, model_config):
        """[正常用例] unload_model 正确移除模型跟踪"""
        engine._loaded_models[model_config.model_name] = model_config

        result = await engine.unload_model(model_config.model_name)

        assert result is True
        assert model_config.model_name not in engine._loaded_models

    @pytest.mark.asyncio
    async def test_unload_nonexistent_model_returns_false(self, engine):
        """[错误用例] 卸载未加载的模型返回 False"""
        result = await engine.unload_model("nonexistent")

        assert result is False, "卸载不存在的模型应返回False"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：generate() 业务逻辑
# ═══════════════════════════════════════════════════════════════════════════════


class TestSGLangGenerateLogic:
    """验证 generate() 的业务逻辑正确性"""

    @pytest.mark.asyncio
    async def test_generate_makes_parallel_requests_for_multiple_prompts(self, engine):
        """[正常用例] 多个prompt时并行发送HTTP请求"""
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

        assert (
            engine._client.post.call_count == 3
        ), f"3个prompt应发送3次HTTP POST，实际: {engine._client.post.call_count}"
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_generate_request_payload_format(self, engine):
        """[正常用例] generate() 发送正确格式的请求体到 /v1/completions"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "choices": [{"text": "ok", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["STOP"],
        )

        await engine.generate("test", ["prompt"], params)

        call_args = engine._client.post.call_args
        payload = call_args.kwargs["json"]

        assert payload["model"] == "test", "model字段应为model_name"
        assert payload["prompt"] == "prompt", "prompt字段应为输入文本"
        assert payload["max_tokens"] == 100, "max_tokens映射错误"
        assert payload["temperature"] == 0.7, "temperature映射错误"
        assert payload["top_p"] == 0.9, "top_p映射错误"
        assert payload["stop"] == ["STOP"], "stop序列映射错误"

    @pytest.mark.asyncio
    async def test_generate_parses_response_correctly(self, engine):
        """[正常用例] generate() 正确解析 /v1/completions 响应格式"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "choices": [{"text": "Hello world", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            }
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["Hi"], SamplingParams())

        assert len(results) == 1
        result = results[0]
        assert (
            result.outputs[0] == "Hello world"
        ), f"outputs[0]应为'Hello world'，实际: {result.outputs[0]}"
        assert result.prompt_tokens == 5, "prompt_tokens解析错误"
        assert result.completion_tokens == 10, "completion_tokens解析错误"
        assert result.finish_reason == "length", "finish_reason应为length"

    @pytest.mark.asyncio
    async def test_generate_handles_empty_choices(self, engine):
        """[错误用例] 响应choices为空时返回错误结果"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["prompt"], SamplingParams())

        assert len(results) == 1
        assert results[0].finish_reason == "error", "空choices时应返回error"
        assert "[SGLang错误: 空响应]" in results[0].outputs[0]

    @pytest.mark.asyncio
    async def test_generate_http_error_includes_status_code(self, engine):
        """[错误用例] HTTP错误时错误消息包含状态码"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["prompt"], SamplingParams())

        assert len(results) == 1
        assert "500" in results[0].outputs[0]

    @pytest.mark.asyncio
    async def test_generate_empty_prompts_returns_empty_list(self, engine):
        """[边界用例] 空prompts列表返回空结果"""
        results = await engine.generate("test", [], SamplingParams())

        assert results == []

    @pytest.mark.asyncio
    async def test_generate_client_not_initialized_returns_error_results(self, engine):
        """[错误用例] 客户端未初始化时返回错误结果"""
        engine._client = None

        results = await engine.generate("test", ["p1", "p2"], SamplingParams())

        assert len(results) == 2
        for r in results:
            assert r.finish_reason == "error"
            assert "[SGLang错误" in r.outputs[0]

    @pytest.mark.asyncio
    async def test_generate_timeout_returns_error_result(self, engine):
        """[异常用例] 请求超时时返回超时错误"""
        engine._client.post = AsyncMock(side_effect=asyncio.TimeoutError())

        results = await engine.generate("test", ["prompt"], SamplingParams())

        assert len(results) == 1
        assert results[0].finish_reason == "error"
        assert "[SGLang超时]" in results[0].outputs[0]

    @pytest.mark.asyncio
    async def test_generate_each_prompt_result_has_correct_index(self, engine):
        """[正常用例] 批量请求中每个结果的request_id正确对应prompt索引"""
        responses = [
            AsyncMock(
                status_code=200,
                json=MagicMock(
                    return_value={
                        "choices": [{"text": f"resp{i}", "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                    }
                ),
            )
            for i in range(3)
        ]
        engine._client.post = AsyncMock(side_effect=responses)

        results = await engine.generate("test", ["p1", "p2", "p3"], SamplingParams())

        assert results[0].request_id == "test_0"
        assert results[1].request_id == "test_1"
        assert results[2].request_id == "test_2"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：generate_stream() 业务逻辑
# ═══════════════════════════════════════════════════════════════════════════════


class TestSGLangGenerateStreamLogic:
    """验证 generate_stream() 的 SSE 解析逻辑"""

    @pytest.mark.asyncio
    async def test_stream_parses_completions_format_text_field(self, engine):
        """[正常用例] SSE流解析 /v1/completions 的 choices[0].text 字段"""
        sse_chunks = [
            'data: {"choices":[{"text":"Hello","index":0,"finish_reason":null}]}\n\n',
            'data: {"choices":[{"text":" world","index":0,"finish_reason":null}]}\n\n',
            'data: {"choices":[{"text":"!","index":0,"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert len(chunks) == 3, f"应返回3个chunk，实际: {len(chunks)}"
        assert chunks == ["Hello", " world", "!"], f"chunks内容错误: {chunks}"

    @pytest.mark.asyncio
    async def test_stream_parses_delta_format_for_chat_completions(self, engine):
        """[正常用例] SSE流兼容 /v1/chat/completions 的 choices[0].delta.text 字段"""
        sse_chunks = [
            'data: {"choices":[{"delta":{"text":"Hi"},"index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert len(chunks) == 1
        assert chunks[0] == "Hi"

    @pytest.mark.asyncio
    async def test_stream_handles_empty_choices(self, engine):
        """[边界用例] 空choices的SSE行被跳过"""
        sse_chunks = [
            'data: {"choices":[]}\n\n',
            'data: {"choices":[{"text":"ok","index":0}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert "ok" in chunks, "空choices行应被跳过"

    @pytest.mark.asyncio
    async def test_stream_skips_invalid_json(self, engine):
        """[错误处理] 非法JSON行被跳过，不中断流"""
        sse_chunks = [
            'data: {"choices":[{"text":"A"}]}\n\n',
            "data: {invalid}\n\n",
            'data: {"choices":[{"text":"B"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert "A" in chunks and "B" in chunks, "非法JSON行不应中断流"

    @pytest.mark.asyncio
    async def test_stream_stops_at_done(self, engine):
        """[正常用例] 收到[DONE]后停止迭代"""
        sse_chunks = [
            'data: {"choices":[{"text":"A"}]}\n\n',
            "data: [DONE]\n\n",
            'data: {"choices":[{"text":"B"}]}\n\n',
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert len(chunks) == 1
        assert "A" in chunks
        assert "B" not in chunks, "[DONE]后不应继续处理"

    @pytest.mark.asyncio
    async def test_stream_http_error_returns_empty(self, engine):
        """[错误用例] HTTP错误时流式返回空"""
        engine._client.stream = MagicMock(return_value=_mock_sse_response([], status_code=403))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_client_not_initialized_returns_empty(self, engine):
        """[错误用例] 客户端未初始化时流式返回空"""
        engine._client = None

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_request_includes_stream_true(self, engine):
        """[正常用例] 流式请求必须设置 stream=True"""
        sse_chunks = [
            'data: {"choices":[{"text":"ok"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        async for _ in engine.generate_stream("test", "prompt", SamplingParams()):
            pass

        call_args = engine._client.stream.call_args
        payload = call_args.kwargs["json"]

        assert payload.get("stream") is True, "流式请求必须设置stream=True"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：chat() 业务逻辑 (SGLang特有)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSGLangChatLogic:
    """验证 chat() 方法的业务逻辑（SGLang特有接口）"""

    @pytest.mark.asyncio
    async def test_chat_request_payload_format(self, engine):
        """[正常用例] chat() 发送正确格式到 /v1/chat/completions"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hi"}]

        await engine.chat("test", messages, SamplingParams())

        call_args = engine._client.post.call_args
        payload = call_args.kwargs["json"]

        assert payload["model"] == "test"
        assert payload["messages"] == messages
        assert payload["max_tokens"] == 2048
        assert payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_parses_response_correctly(self, engine):
        """[正常用例] chat() 正确解析 /v1/chat/completions 响应"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "I am helpful"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 5},
            }
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        result = await engine.chat("test", [{"role": "user", "content": "Hello"}], SamplingParams())

        assert result.outputs[0] == "I am helpful"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 5
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_request_id_format(self, engine):
        """[正常用例] chat() 的request_id包含_chat后缀"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        )
        engine._client.post = AsyncMock(return_value=mock_response)

        result = await engine.chat("test-model", [], SamplingParams())

        assert result.request_id == "test-model_chat"

    @pytest.mark.asyncio
    async def test_chat_http_error_returns_error_result(self, engine):
        """[错误用例] chat HTTP错误时返回错误结果"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        engine._client.post = AsyncMock(return_value=mock_response)

        result = await engine.chat("test", [], SamplingParams())

        assert result.finish_reason == "error"
        assert "400" in result.outputs[0]

    @pytest.mark.asyncio
    async def test_chat_client_not_initialized_returns_error(self, engine):
        """[错误用例] 客户端未初始化时chat返回错误结果"""
        engine._client = None

        result = await engine.chat("test", [], SamplingParams())

        assert result.finish_reason == "error"
        assert "[SGLang错误" in result.outputs[0]

    @pytest.mark.asyncio
    async def test_chat_timeout_returns_error_result(self, engine):
        """[异常用例] chat请求超时时返回超时错误"""
        engine._client.post = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await engine.chat("test", [], SamplingParams())

        assert result.finish_reason == "error"
        assert "[SGLang错误" in result.outputs[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：get_stats() 业务逻辑
# ═══════════════════════════════════════════════════════════════════════════════


class TestSGLangStats:
    """验证 get_stats() 的返回值格式"""

    @pytest.mark.asyncio
    async def test_get_stats_returns_dict(self, engine):
        """[正常用例] get_stats返回字典"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        engine._client.get = AsyncMock(return_value=mock_response)

        stats = await engine.get_stats("test-model")

        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_get_stats_returns_healthy_on_success(self, engine):
        """[正常用例] /v1/models返回200时stats包含healthy=1.0"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        engine._client.get = AsyncMock(return_value=mock_response)

        stats = await engine.get_stats("test-model")

        assert stats.get("healthy") == 1.0

    @pytest.mark.asyncio
    async def test_get_stats_client_not_initialized_returns_empty_dict(self, engine):
        """[错误用例] 客户端未初始化时返回空字典"""
        engine._client = None

        stats = await engine.get_stats("test-model")

        assert stats == {}

    @pytest.mark.asyncio
    async def test_get_stats_http_error_returns_empty_dict(self, engine):
        """[错误用例] HTTP错误时返回空字典"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        engine._client.get = AsyncMock(return_value=mock_response)

        stats = await engine.get_stats("test-model")

        assert stats == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：错误消息格式一致性
# ═══════════════════════════════════════════════════════════════════════════════


class TestSGLangErrorMessages:
    """验证错误消息格式的一致性"""

    @pytest.mark.asyncio
    async def test_all_error_results_contain_sglang_prefix(self, engine):
        """[错误格式] 所有错误结果应包含[SGLang错误]或[SGLang超时]前缀"""
        engine._client = None

        results = await engine.generate("test", ["p1"], SamplingParams())

        assert "[SGLang错误" in results[0].outputs[0] or "[SGLang超时]" in results[0].outputs[0]

    @pytest.mark.asyncio
    async def test_error_results_have_error_finish_reason(self, engine):
        """[错误格式] 错误结果的finish_reason必须为error"""
        engine._client = None

        results = await engine.generate("test", ["p1"], SamplingParams())

        assert results[0].finish_reason == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
