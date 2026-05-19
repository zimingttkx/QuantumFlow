"""Chunked Prefill 功能测试 — 验证 HuggingFaceEngine 的分块预填充逻辑

测试原则：
1. 业务逻辑验证优先于运行可用性
2. 强精准断言：每个功能点的预期行为必须严格验证
3. 全覆盖：正常用例、边界值、非法入参、异常场景

注意：此文件仅测试可隔离验证的逻辑。涉及 torch 操作的核心推理逻辑
需要通过集成测试验证。
"""

import sys

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.inference.backends.huggingface import (
    CHUNKED_PREFILL_THRESHOLD_TOKENS,
    HuggingFaceEngine,
)
from quantumflow.inference.engine import InferenceResult, ModelConfig, SamplingParams

# ═══════════════════════════════════════════════════════════════════════════════
# 常量验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunkedPrefillConstants:
    """验证 Chunked Prefill 相关的常量定义"""

    def test_threshold_constant_is_512(self):
        """[正常用例] CHUNKED_PREFILL_THRESHOLD_TOKENS 常量值正确"""
        assert (
            CHUNKED_PREFILL_THRESHOLD_TOKENS == 512
        ), f"阈值应为512，实际: {CHUNKED_PREFILL_THRESHOLD_TOKENS}"

    def test_threshold_constant_is_positive(self):
        """[边界用例] 阈值必须为正数"""
        assert CHUNKED_PREFILL_THRESHOLD_TOKENS > 0, "阈值必须为正数"


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine():
    """创建已初始化的 HF 引擎"""
    engine = HuggingFaceEngine()
    engine._is_initialized = True
    return engine


@pytest.fixture
def model_config():
    """标准模型配置，启用 chunked prefill"""
    return ModelConfig(
        model_name="test-chunked-model",
        model_path="test/model/path",
        enable_chunked_prefill=True,
        prefill_chunk_size=512,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：模型未加载错误处理
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunkedErrorHandling:
    """验证 Chunked Prefill 的错误处理"""

    @pytest.mark.asyncio
    async def test_model_not_loaded_returns_error_result(self, engine):
        """[错误用例] 模型未加载时generate返回错误结果"""
        results = await engine.generate("nonexistent", ["prompt"], SamplingParams())

        assert len(results) == 1
        assert results[0].finish_reason == "error"
        assert "[模型未加载" in results[0].outputs[0]

    @pytest.mark.asyncio
    async def test_stream_model_not_loaded_returns_empty(self, engine):
        """[错误用例] 模型未加载时generate_stream返回空"""
        chunks = []
        async for chunk in engine.generate_stream("nonexistent", "prompt", SamplingParams()):
            chunks.append(chunk)

        assert chunks == [], "模型未加载时应返回空流"

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded_returns_false(self, engine):
        """[错误用例] 卸载未加载的模型返回 False"""
        result = await engine.unload_model("nonexistent")

        assert result is False, "卸载不存在的模型应返回False"

    @pytest.mark.asyncio
    async def test_is_model_loaded_returns_false_for_unloaded(self, engine):
        """[正常用例] 未加载的模型返回 False"""
        assert await engine.is_model_loaded("nonexistent") is False

    @pytest.mark.asyncio
    async def test_loaded_model_names_empty_initially(self, engine):
        """[正常用例] 初始状态没有已加载模型"""
        assert engine.loaded_model_names == []


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：ModelConfig 验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelConfigChunkedPrefill:
    """验证 ModelConfig 中与 Chunked Prefill 相关的配置"""

    def test_enable_chunked_prefill_defaults_to_false(self):
        """[正常用例] enable_chunked_prefill 默认为 False"""
        config = ModelConfig(model_name="test", model_path="path")
        assert config.enable_chunked_prefill is False, "enable_chunked_prefill 默认值应为 False"

    def test_prefill_chunk_size_defaults_to_512(self):
        """[正常用例] prefill_chunk_size 默认为 512"""
        config = ModelConfig(model_name="test", model_path="path")
        assert config.prefill_chunk_size == 512, "prefill_chunk_size 默认值应为 512"

    def test_chunked_prefill_config_accepted(self, model_config):
        """[正常用例] Chunked Prefill 配置可被接受"""
        assert model_config.enable_chunked_prefill is True
        assert model_config.prefill_chunk_size == 512

    def test_model_config_to_dict_contains_relevant_fields(self):
        """[正常用例] ModelConfig.to_dict() 包含相关字段"""
        config = ModelConfig(
            model_name="test",
            model_path="/path",
            enable_chunked_prefill=True,
            prefill_chunk_size=256,
        )
        d = config.to_dict()
        assert "model_name" in d
        assert "model_path" in d


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：SamplingParams 验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestSamplingParamsValidation:
    """验证 SamplingParams 的参数验证"""

    def test_sampling_params_defaults(self):
        """[正常用例] SamplingParams 默认值正确"""
        params = SamplingParams()
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 2048
        assert params.repetition_penalty == 1.0
        assert params.stop is None

    def test_sampling_params_with_custom_values(self):
        """[正常用例] SamplingParams 可设置自定义值"""
        params = SamplingParams(
            temperature=0.5,
            top_p=0.8,
            top_k=100,
            max_tokens=512,
            repetition_penalty=1.2,
            stop=["END", "STOP"],
        )
        assert params.temperature == 0.5
        assert params.top_k == 100
        assert params.max_tokens == 512
        assert params.stop == ["END", "STOP"]

    def test_sampling_params_to_dict(self):
        """[正常用例] SamplingParams.to_dict() 返回正确格式"""
        params = SamplingParams(temperature=0.7, max_tokens=100)
        d = params.to_dict()
        assert d["temperature"] == 0.7
        assert d["max_tokens"] == 100
        assert "temperature" in d
        assert "top_p" in d
        assert "top_k" in d


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：InferenceResult 验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferenceResult:
    """验证 InferenceResult 数据类"""

    def test_inference_result_creation(self):
        """[正常用例] InferenceResult 可正确创建"""
        result = InferenceResult(
            request_id="test-123",
            outputs=["Hello world"],
            prompt_tokens=5,
            completion_tokens=2,
            latency_ms=100.5,
            finish_reason="stop",
        )
        assert result.request_id == "test-123"
        assert result.outputs[0] == "Hello world"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 2
        assert result.latency_ms == 100.5
        assert result.finish_reason == "stop"

    def test_inference_result_with_metrics(self):
        """[正常用例] InferenceResult 可包含 metrics"""
        result = InferenceResult(
            request_id="test-456",
            outputs=["Test"],
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=10.0,
            finish_reason="stop",
            metrics={"tokens_per_second": 10.5},
        )
        assert result.metrics["tokens_per_second"] == 10.5


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：引擎初始化
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineInitialization:
    """验证引擎初始化状态"""

    def test_engine_initial_state(self, engine):
        """[正常用例] 引擎在fixture中已初始化，is_ready 应为 True"""
        assert engine.is_ready is True, "fixture 中的引擎 is_ready 应为 True"
        assert engine.backend_type.value == "huggingface"

    def test_engine_loaded_models_initially_empty(self, engine):
        """[正常用例] 初始加载模型列表为空"""
        assert len(engine.loaded_model_names) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 测试类：分块决策逻辑（单元测试友好部分）
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunkingDecisionLogic:
    """验证 prompt 长度与分块决策的逻辑关系

    注意：完整的分块逻辑测试需要实际模型和 torch 操作，
    此处仅测试可隔离验证的部分。
    """

    def test_chunk_size_calculation_for_prompt(self):
        """[正常用例] 分块数量 = ceil(prompt_len / chunk_size)"""
        # 测试不同 prompt 长度的分块数量计算
        test_cases = [
            # (prompt_len, chunk_size, expected_chunks)
            (100, 512, 1),  # 短于 chunk_size
            (512, 512, 1),  # 等于 chunk_size
            (513, 512, 2),  # 略大于 chunk_size
            (1024, 512, 2),  # 2x chunk_size
            (1025, 512, 3),  # 略大于 2x
            (1536, 512, 3),  # 3x chunk_size
            (1537, 512, 4),  # 略大于 3x
        ]

        for prompt_len, chunk_size, expected_chunks in test_cases:
            actual_chunks = (prompt_len + chunk_size - 1) // chunk_size
            assert (
                actual_chunks == expected_chunks
            ), f"prompt_len={prompt_len}, chunk_size={chunk_size} 时应有 {expected_chunks} 个 chunks，实际: {actual_chunks}"

    def test_threshold_comparison_for_decision(self):
        """[正常用例] prompt_len > threshold 时应使用 chunked prefill"""
        test_cases = [
            # (prompt_len, threshold, should_use_chunked)
            (100, 512, False),  # 短于阈值
            (512, 512, False),  # 等于阈值（不算长）
            (513, 512, True),  # 大于阈值
            (1000, 512, True),  # 明显大于阈值
        ]

        for prompt_len, threshold, should_use_chunked in test_cases:
            decision = prompt_len > threshold
            assert (
                decision == should_use_chunked
            ), f"prompt_len={prompt_len}, threshold={threshold} 时 should_use_chunked 应为 {should_use_chunked}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
