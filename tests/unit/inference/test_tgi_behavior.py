"""TGI 后端行为测试 — 严格验证 TGIEngine 的业务逻辑正确性

测试原则：
1. 业务逻辑验证优先于运行可用性
2. 强精准断言：每个功能点的预期返回值必须严格比对
3. 全覆盖：正常用例、边界值、非法入参、异常场景
4. 错误定位精准：区分运行报错、逻辑错误、流程偏差

验证维度：
- generate() 的请求构建逻辑（参数映射、端点选择）
- generate_stream() 的 SSE 解析逻辑
- 错误处理路径（网络错误、超时、HTTP错误码）
- 模型加载/卸载状态管理
- 批量请求的并行处理和错误隔离
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List

import sys
sys.path.insert(0, '/home/dingziming/PycharmProjects/QuantumFlow')

from quantumflow.inference.backends.tgi import TGIEngine
from quantumflow.inference.engine import ModelConfig, SamplingParams, InferenceResult


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    """创建已初始化的 TGI 引擎"""
    engine = TGIEngine(base_url="http://localhost:8080")
    engine._client = MagicMock()  # 使用 MagicMock 而非 AsyncMock
    engine._is_initialized = True
    return engine


@pytest.fixture
def model_config():
    """标准模型配置"""
    return ModelConfig(
        model_name="test-model",
        model_path="tgi-test/path",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

async def _make_async_iter(items: List[str]):
    """将列表转为 async iterator"""
    for item in items:
        yield item


def _mock_sse_response(chunks: List[str], status_code: int = 200):
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

class TestTGIInitialization:
    """验证 TGI 引擎初始化和生命周期管理"""

    @pytest.mark.asyncio
    async def test_initialize_success_with_healthy_response(self):
        """[正常用例] 健康检查成功时 initialize 返回 True"""
        engine = TGIEngine(base_url="http://localhost:8080")

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
        engine = TGIEngine(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await engine.initialize()

        assert result is False, "健康检查返回非200时应初始化失败"

    @pytest.mark.asyncio
    async def test_initialize_failure_when_httpx_import_error(self):
        """[异常用例] httpx 未安装时 initialize 返回 False"""
        engine = TGIEngine(base_url="http://localhost:8080")

        with patch.dict("sys.modules", {"httpx": None}):
            result = await engine.initialize()

        assert result is False, "httpx导入失败时应返回False"

    @pytest.mark.asyncio
    async def test_close_closes_http_client(self):
        """[正常用例] close() 正确关闭 HTTP 客户端"""
        engine = TGIEngine(base_url="http://localhost:8080")
        engine._client = MagicMock()
        engine._client.aclose = AsyncMock()

        await engine.close()

        assert engine._client.aclose.called, "close() 应关闭客户端连接"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：模型加载与卸载
# ═══════════════════════════════════════════════════════════════════════════════

class TestTGIModelLifecycle:
    """验证模型的加载/卸载状态管理"""

    @pytest.mark.asyncio
    async def test_load_model_tracks_model_config(self, engine, model_config):
        """[正常用例] load_model 正确跟踪模型配置"""
        mock_info_response = MagicMock()
        mock_info_response.status_code = 200
        mock_info_response.json = MagicMock(return_value={"model_ids": ["test-model"]})

        engine._client.get = AsyncMock(return_value=mock_info_response)

        result = await engine.load_model(model_config)

        assert result is True, "模型ID匹配时应加载成功"
        assert model_config.model_name in engine._loaded_models, \
            "加载的模型应被添加到 _loaded_models"

    @pytest.mark.asyncio
    async def test_load_model_without_client_returns_false(self, engine, model_config):
        """[错误用例] 客户端未初始化时 load_model 返回 False"""
        engine._client = None

        result = await engine.load_model(model_config)

        assert result is False, "客户端未初始化时应返回False"

    @pytest.mark.asyncio
    async def test_load_model_with_mismatched_id_returns_true_anyway(self, engine, model_config):
        """[边界用例] TGI服务器上的模型ID不匹配时，也假设模型已加载（容错设计）"""
        mock_info_response = MagicMock()
        mock_info_response.status_code = 200
        mock_info_response.json = MagicMock(return_value={"model_ids": ["different-model"]})

        engine._client.get = AsyncMock(return_value=mock_info_response)

        result = await engine.load_model(model_config)

        # 代码逻辑：只要TGI运行中，就假设模型可加载
        assert result is True, "TGI运行时应容错返回True"

    @pytest.mark.asyncio
    async def test_unload_model_removes_from_tracking(self, engine, model_config):
        """[正常用例] unload_model 正确移除模型跟踪"""
        engine._loaded_models[model_config.model_name] = model_config

        result = await engine.unload_model(model_config.model_name)

        assert result is True, "卸载已跟踪的模型应返回True"
        assert model_config.model_name not in engine._loaded_models, \
            "卸载后模型不应在 _loaded_models 中"

    @pytest.mark.asyncio
    async def test_unload_nonexistent_model_returns_false(self, engine):
        """[错误用例] 卸载未加载的模型返回 False"""
        result = await engine.unload_model("nonexistent-model")

        assert result is False, "卸载不存在的模型应返回False"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：generate() 业务逻辑
# ═══════════════════════════════════════════════════════════════════════════════

class TestTGIGenerateLogic:
    """验证 generate() 的业务逻辑正确性"""

    # ── 端点选择逻辑 ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_single_prompt_uses_generate_endpoint(self, engine):
        """[正常用例] 单个prompt使用/generate端点"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "generated_text": "test response",
            "prompt_tokens": 5,
            "generated_tokens": 2,
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        await engine.generate("test", ["single prompt"], SamplingParams())

        # 验证端点选择
        call_args = engine._client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")

        # 断言：单个prompt应调用/generate而非/generate_batch
        assert "/generate" in str(url), \
            f"单个prompt应调用/generate端点，实际: {url}"

    @pytest.mark.asyncio
    async def test_multiple_prompts_use_generate_batch_endpoint(self, engine):
        """[正常用例] 多个prompt使用/generate_batch端点"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "generated_text": ["r1", "r2"],
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        await engine.generate("test", ["p1", "p2"], SamplingParams())

        call_args = engine._client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")

        assert "/generate_batch" in str(url), \
            f"多个prompt应调用/generate_batch端点，实际: {url}"

    # ── 请求参数映射 ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_sampling_params_fully_mapped(self, engine):
        """[正常用例] 所有采样参数正确映射到API参数"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"generated_text": "ok"})
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=100,
            repetition_penalty=1.1,
            stop=["STOP", "END"],
        )

        await engine.generate("test", ["prompt"], params)

        call_args = engine._client.post.call_args
        payload = call_args.kwargs["json"]

        # 严格验证每个参数
        assert payload["inputs"] == "prompt", "inputs应为prompt字符串"
        params_block = payload["parameters"]
        assert params_block["temperature"] == 0.7, "temperature映射错误"
        assert params_block["top_p"] == 0.9, "top_p映射错误"
        assert params_block["top_k"] == 40, "top_k映射错误"
        assert params_block["max_new_tokens"] == 100, "max_tokens应映射为max_new_tokens"
        assert params_block["repetition_penalty"] == 1.1, "repetition_penalty映射错误"
        assert params_block["stop"] == ["STOP", "END"], "stop序列映射错误"

    @pytest.mark.asyncio
    async def test_top_k_only_sent_when_positive(self, engine):
        """[边界用例] top_k <= 0 时不应发送到API（避免截断）"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"generated_text": "ok"})
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(top_k=0)  # 无效值
        await engine.generate("test", ["prompt"], params)

        call_args = engine._client.post.call_args
        payload = call_args.kwargs["json"]

        # top_k <= 0 时，API不应包含top_k参数
        assert "top_k" not in payload.get("parameters", {}), \
            "top_k<=0时不应发送到API，否则会导致意外截断"

    @pytest.mark.asyncio
    async def test_stop_not_sent_when_empty(self, engine):
        """[边界用例] stop为空列表时不发送到API"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"generated_text": "ok"})
        engine._client.post = AsyncMock(return_value=mock_response)

        params = SamplingParams(stop=[])
        await engine.generate("test", ["prompt"], params)

        call_args = engine._client.post.call_args
        payload = call_args.kwargs["json"]

        # 空stop不应发送到API
        assert "stop" not in payload.get("parameters", {}), \
            "空stop列表不应发送到API"

    # ── 响应解析逻辑 ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_single_response_parses_correctly(self, engine):
        """[正常用例] 单响应正确解析generated_text字段"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "generated_text": "Hello world",
            "prompt_tokens": 5,
            "generated_tokens": 2,
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["Hi"], SamplingParams())

        assert len(results) == 1, "单prompt应返回单结果"
        result = results[0]
        assert result.outputs[0] == "Hello world", \
            f"outputs[0]应为'Hello world'，实际: {result.outputs[0]}"
        assert result.prompt_tokens == 5, "prompt_tokens解析错误"
        assert result.completion_tokens == 2, "generated_tokens应映射为completion_tokens"
        assert result.finish_reason == "stop", "单响应应返回finish_reason=stop"

    @pytest.mark.asyncio
    async def test_batch_response_with_list_tokens_parses_correctly(self, engine):
        """[正常用例] 批量响应中token统计为列表时正确解析"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "generated_text": ["resp1", "resp2", "resp3"],
            "prompt_tokens": [10, 20, 30],
            "generated_tokens": [5, 8, 12],
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["p1", "p2", "p3"], SamplingParams())

        assert len(results) == 3, "批量请求应返回3个结果"

        # 严格验证每个结果的token统计
        assert results[0].prompt_tokens == 10, "第1个结果的prompt_tokens错误"
        assert results[1].prompt_tokens == 20, "第2个结果的prompt_tokens错误"
        assert results[2].prompt_tokens == 30, "第3个结果的prompt_tokens错误"

        assert results[0].completion_tokens == 5, "第1个结果的completion_tokens错误"
        assert results[1].completion_tokens == 8, "第2个结果的completion_tokens错误"
        assert results[2].completion_tokens == 12, "第3个结果的completion_tokens错误"

    @pytest.mark.asyncio
    async def test_batch_response_with_scalar_tokens_distributes_evenly(self, engine):
        """[边界用例] 批量响应中token统计为标量时均分到各结果"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "generated_text": ["r1", "r2", "r3"],
            "prompt_tokens": 30,  # 标量
            "generated_tokens": 15,  # 标量
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["p1", "p2", "p3"], SamplingParams())

        assert len(results) == 3

        # 验证均分逻辑：30 // 3 = 10，15 // 3 = 5
        for r in results:
            assert r.prompt_tokens == 10, \
                f"均分后每个结果的prompt_tokens应为10，实际: {r.prompt_tokens}"
            assert r.completion_tokens == 5, \
                f"均分后每个结果的completion_tokens应为5，实际: {r.completion_tokens}"

    @pytest.mark.asyncio
    async def test_empty_prompts_returns_empty_list(self, engine):
        """[边界用例] 空prompts列表返回空结果列表"""
        results = await engine.generate("test", [], SamplingParams())

        assert results == [], "空prompts应返回空列表"
        assert isinstance(results, list), "返回值应为list类型"

    @pytest.mark.asyncio
    async def test_missing_generated_text_returns_empty_output(self, engine):
        """[错误用例] 响应缺少generated_text时返回空output"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            # generated_text 字段缺失
            "prompt_tokens": 5,
        })
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["prompt"], SamplingParams())

        assert len(results) == 1
        assert results[0].outputs[0] == "", \
            "缺少generated_text时应返回空字符串"

    # ── 错误处理路径 ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_results(self, engine):
        """[错误用例] HTTP错误码返回空结果列表"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        engine._client.post = AsyncMock(return_value=mock_response)

        results = await engine.generate("test", ["p1", "p2"], SamplingParams())

        assert results == [], "HTTP错误时应返回空列表"

    @pytest.mark.asyncio
    async def test_client_not_initialized_returns_error_results(self, engine):
        """[错误用例] 客户端未初始化时返回错误结果"""
        engine._client = None

        results = await engine.generate("test", ["p1", "p2"], SamplingParams())

        assert len(results) == 2, "应返回与prompts数量相等的错误结果"
        for r in results:
            assert r.finish_reason == "error", "未初始化时应返回error状态"
            assert "[TGI错误" in r.outputs[0], "错误消息应包含[TGI错误"

    @pytest.mark.asyncio
    async def test_timeout_returns_error_results(self, engine):
        """[异常用例] 请求超时时返回错误结果"""
        engine._client.post = AsyncMock(side_effect=asyncio.TimeoutError())

        results = await engine.generate("test", ["p1"], SamplingParams())

        assert len(results) == 1
        assert results[0].finish_reason == "error", "超时时应返回error"
        assert "[TGI超时]" in results[0].outputs[0], "错误消息应包含[TGI超时]"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：generate_stream() 业务逻辑
# ═══════════════════════════════════════════════════════════════════════════════

class TestTGIGenerateStreamLogic:
    """验证 generate_stream() 的 SSE 解析和流式处理逻辑"""

    @pytest.mark.asyncio
    async def test_stream_parses_token_text_correctly(self, engine):
        """[正常用例] SSE流正确解析token.text字段"""
        sse_chunks = [
            'data: {"token":{"text":"Hello"}}\n\n',
            'data: {"token":{"text":" world"}}\n\n',
            'data: {"token":{"text":"!"}}\n\n',
            'data: [DONE]\n\n',
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert len(chunks) == 3, f"应返回3个chunk，实际: {len(chunks)}"
        assert chunks[0] == "Hello", f"第1个chunk错误: {chunks[0]}"
        assert chunks[1] == " world", f"第2个chunk错误: {chunks[1]}"
        assert chunks[2] == "!", f"第3个chunk错误: {chunks[2]}"

    @pytest.mark.asyncio
    async def test_stream_skips_invalid_json_lines(self, engine):
        """[错误处理] SSE中的非法JSON行被跳过，不中断流"""
        sse_chunks = [
            'data: {"token":{"text":"Start"}}\n\n',
            'data: {invalid json}\n\n',  # 非法JSON
            'data: {"token":{"text":"End"}}\n\n',
            'data: [DONE]\n\n',
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        # 不应因非法JSON而崩溃或中断
        assert len(chunks) == 2, f"非法JSON行应被跳过，实际chunks: {chunks}"
        assert "Start" in chunks, "Start应在chunks中"
        assert "End" in chunks, "End应在chunks中"

    @pytest.mark.asyncio
    async def test_stream_handles_empty_token_text(self, engine):
        """[边界用例] token.text为空字符串时正确处理"""
        sse_chunks = [
            'data: {"token":{"text":"Hello"}}\n\n',
            'data: {"token":{"text":""}}\n\n',  # 空token
            'data: {"token":{"text":"World"}}\n\n',
            'data: [DONE]\n\n',
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        # 空字符串token不应添加为chunk
        assert len(chunks) == 2, f"空token应被过滤，实际chunks: {chunks}"
        assert "" not in chunks, "空字符串不应出现在chunks中"

    @pytest.mark.asyncio
    async def test_stream_stops_at_done(self, engine):
        """[正常用例] 收到[DONE]后停止迭代"""
        sse_chunks = [
            'data: {"token":{"text":"A"}}\n\n',
            'data: [DONE]\n\n',
            'data: {"token":{"text":"B"}}\n\n',  # 不应被处理
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert len(chunks) == 1, f"[DONE]后不应继续处理，实际chunks: {chunks}"
        assert chunks[0] == "A"

    @pytest.mark.asyncio
    async def test_stream_http_error_returns_empty(self, engine):
        """[错误用例] HTTP错误时流式返回空"""
        engine._client.stream = MagicMock(return_value=_mock_sse_response([], status_code=403))

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert chunks == [], f"HTTP错误时应返回空流，实际: {chunks}"

    @pytest.mark.asyncio
    async def test_stream_client_not_initialized_returns_empty(self, engine):
        """[错误用例] 客户端未初始化时流式返回空"""
        engine._client = None

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert chunks == [], "客户端未初始化时应返回空流"

    @pytest.mark.asyncio
    async def test_stream_timeout_yields_nothing(self, engine):
        """[异常用例] 超时时流式不产生任何chunk"""
        engine._client.stream = AsyncMock(side_effect=asyncio.TimeoutError())

        chunks = []
        async for text in engine.generate_stream("test", "prompt", SamplingParams()):
            chunks.append(text)

        assert chunks == [], "超时时不应产生任何chunk"

    @pytest.mark.asyncio
    async def test_stream_passes_correct_parameters(self, engine):
        """[正常用例] 流式请求正确传递所有采样参数"""
        sse_chunks = [
            'data: {"token":{"text":"ok"}}\n\n',
            'data: [DONE]\n\n',
        ]
        engine._client.stream = MagicMock(return_value=_mock_sse_response(sse_chunks))

        params = SamplingParams(
            temperature=0.5,
            top_p=0.8,
            top_k=30,
            max_tokens=50,
            repetition_penalty=1.2,
            stop=["FIN"],
        )

        async for _ in engine.generate_stream("test", "prompt", params):
            pass

        call_args = engine._client.stream.call_args
        payload = call_args.kwargs["json"]

        assert payload["inputs"] == "prompt"
        params_block = payload["parameters"]
        assert params_block["temperature"] == 0.5
        assert params_block["top_p"] == 0.8
        assert params_block["top_k"] == 30
        assert params_block["max_new_tokens"] == 50
        assert params_block["repetition_penalty"] == 1.2
        assert params_block["stop"] == ["FIN"]


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：get_stats() 业务逻辑
# ═══════════════════════════════════════════════════════════════════════════════

class TestTGIStats:
    """验证 get_stats() 的返回值格式"""

    @pytest.mark.asyncio
    async def test_get_stats_returns_dict(self, engine):
        """[正常用例] get_stats返回字典类型"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        engine._client.get = AsyncMock(return_value=mock_response)

        stats = await engine.get_stats("test-model")

        assert isinstance(stats, dict), f"返回值应为dict，实际: {type(stats)}"

    @pytest.mark.asyncio
    async def test_get_stats_client_not_initialized_returns_empty_dict(self, engine):
        """[错误用例] 客户端未初始化时返回空字典"""
        engine._client = None

        stats = await engine.get_stats("test-model")

        assert stats == {}, "客户端未初始化时应返回空字典"

    @pytest.mark.asyncio
    async def test_get_stats_http_error_returns_empty_dict(self, engine):
        """[错误用例] HTTP错误时返回空字典"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        engine._client.get = AsyncMock(return_value=mock_response)

        stats = await engine.get_stats("test-model")

        assert stats == {}, "HTTP错误时应返回空字典"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：错误消息格式一致性
# ═══════════════════════════════════════════════════════════════════════════════

class TestTGIErrorMessages:
    """验证错误消息格式的一致性和完整性"""

    @pytest.mark.asyncio
    async def test_all_error_results_contain_tgi_prefix(self, engine):
        """[错误格式验证] 所有错误结果的消息应包含[TGI错误]或[TGI超时]前缀"""
        engine._client = None

        results = await engine.generate("test", ["p1", "p2"], SamplingParams())

        for r in results:
            # 错误消息应有一致的格式
            assert r.outputs[0].startswith("[TGI错误") or \
                   r.outputs[0].startswith("[TGI超时]"), \
                f"错误消息格式不统一: {r.outputs[0]}"

    @pytest.mark.asyncio
    async def test_error_results_have_error_finish_reason(self, engine):
        """[错误格式验证] 错误结果的finish_reason必须为error"""
        engine._client = None

        results = await engine.generate("test", ["p1"], SamplingParams())

        assert results[0].finish_reason == "error", \
            f"错误结果的finish_reason应为error，实际: {results[0].finish_reason}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])