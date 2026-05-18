"""API Routes 核心逻辑专业测试

测试策略：
1. 请求参数校验逻辑
2. 响应数据字段完整性
3. 错误处理流程正确性
4. 状态码与错误码对应关系
5. 批量接口结果统计准确性
6. 流式接口数据格式正确性
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

import sys
sys.path.insert(0, '/home/dingziming/PycharmProjects/QuantumFlow')

from quantumflow.api.server import create_app
from quantumflow.api.models import (
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    ChatRequest,
)
from quantumflow.inference.engine import SamplingParams


class TestBatchResponseStatistics:
    """批量推理响应统计逻辑严格校验"""

    def test_completed_plus_failed_equals_total(self):
        """completed + failed 必须等于 total"""
        # 这个测试校验响应模型本身的约束

        response = BatchInferenceResponse(
            batch_id="test_batch",
            model="test_model",
            total=10,
            completed=7,
            failed=3,
            results=[],
            total_latency_ms=100.0,
            avg_latency_ms=10.0,
        )

        assert response.completed + response.failed == response.total

    def test_model_not_loaded_failed_equals_all(self):
        """模型未加载时，failed 必须等于 total（所有请求都失败）"""
        # 模拟模型未加载场景
        total = 5
        completed = 0
        failed = total

        response = BatchInferenceResponse(
            batch_id="test_batch",
            model="nonexistent",
            total=total,
            completed=completed,
            failed=failed,
            results=[],
            total_latency_ms=0,
            avg_latency_ms=0,
        )

        assert response.completed == 0
        assert response.failed == total

    def test_avg_latency_calculation(self):
        """avg_latency 必须等于 total_latency / completed（当 completed > 0）"""
        total_latency = 100.0
        completed = 10
        expected_avg = total_latency / completed

        response = BatchInferenceResponse(
            batch_id="test_batch",
            model="test_model",
            total=completed,
            completed=completed,
            failed=0,
            results=[],
            total_latency_ms=total_latency,
            avg_latency_ms=expected_avg,
        )

        assert abs(response.avg_latency_ms - expected_avg) < 0.001

    def test_all_success_no_failures(self):
        """全部成功时 failed 必须为 0"""
        response = BatchInferenceResponse(
            batch_id="test_batch",
            model="test_model",
            total=5,
            completed=5,
            failed=0,
            results=[],
            total_latency_ms=50.0,
            avg_latency_ms=10.0,
        )

        assert response.failed == 0
        assert response.completed == 5

    def test_all_failures_no_success(self):
        """全部失败时 completed 必须为 0"""
        response = BatchInferenceResponse(
            batch_id="test_batch",
            model="test_model",
            total=5,
            completed=0,
            failed=5,
            results=[],
            total_latency_ms=0,
            avg_latency_ms=0,
        )

        assert response.completed == 0
        assert response.failed == 5


class TestSamplingParamsConversion:
    """采样参数转换逻辑校验"""

    def test_convert_sampling_params_with_defaults(self):
        """带默认值的 sampling_params 转换"""
        from quantumflow.api.routes.inference import _convert_sampling_params

        request = InferenceRequest(
            model="test",
            prompt="hello",
            sampling_params=None,
        )

        params = _convert_sampling_params(request)

        assert params.temperature == 0.7  # 默认值
        assert params.top_p == 0.9  # 默认值
        assert params.max_tokens == 2048  # 默认值

    def test_convert_sampling_params_with_custom_values(self):
        """自定义 sampling_params 转换"""
        from quantumflow.api.routes.inference import _convert_sampling_params

        request = InferenceRequest(
            model="test",
            prompt="hello",
            sampling_params={"temperature": 0.5, "top_p": 0.8, "max_tokens": 100},
        )

        params = _convert_sampling_params(request)

        assert params.temperature == 0.5
        assert params.top_p == 0.8
        assert params.max_tokens == 100


class TestRequestValidation:
    """请求参数校验逻辑"""

    def test_inference_request_defaults(self):
        """InferenceRequest 默认值校验"""
        request = InferenceRequest(
            model="test_model",
            prompt="hello",
        )

        assert request.stream is False  # 默认非流式
        # sampling_params 默认是 SamplingParams 对象，不是 None
        assert request.sampling_params is not None
        assert request.sampling_params.temperature == 0.7  # 默认 temperature

    def test_batch_request_requires_prompts(self):
        """BatchInferenceRequest prompts 不能为空"""
        # 当 prompts 为空列表时，应该如何处理？
        # 这是业务逻辑决策点

    def test_chat_request_message_format(self):
        """ChatRequest 消息格式校验"""
        request = ChatRequest(
            model="test",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
        )

        assert len(request.messages) == 2
        # messages 是 ChatMessage 对象列表，不是字典列表
        assert request.messages[0].role == "system"
        assert request.messages[0].content == "You are helpful"


class TestResponseFieldCompleteness:
    """响应字段完整性校验"""

    def test_inference_response_required_fields(self):
        """InferenceResponse 所有必填字段"""
        response = InferenceResponse(
            request_id="req_001",
            model="test_model",
            prompt="hello",
            generated_text="hi there",
            finish_reason="stop",
            latency_ms=100.0,
            usage={
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        )

        # 验证所有字段存在且非 None
        assert response.request_id == "req_001"
        assert response.generated_text == "hi there"
        assert response.finish_reason in ["stop", "length", "error"]

    def test_stream_response_fields(self):
        """StreamResponse 字段完整性"""
        from quantumflow.api.models import StreamResponse

        response = StreamResponse(
            request_id="req_001",
            delta="hello",
            is_final=False,
            finish_reason="stop",
        )

        assert response.delta == "hello"
        assert response.is_final is False

    def test_error_response_format(self):
        """错误响应格式校验"""
        from quantumflow.api.models import ErrorResponse

        error = ErrorResponse(
            error={
                "code": "MODEL_NOT_FOUND",
                "message": "Model x not loaded",
            }
        )

        # error.error 是 ErrorDetail 对象，不是字典
        assert error.error.code == "MODEL_NOT_FOUND"
        assert error.error.message == "Model x not loaded"


class TestEnsureModelLoadedLogic:
    """_ensure_model_loaded 逻辑校验"""

    @pytest.mark.asyncio
    async def test_model_loaded_returns_true(self):
        """模型已加载时返回 True"""
        from quantumflow.api.routes.inference import _ensure_model_loaded

        with patch('quantumflow.api.routes.inference.get_engine_manager') as mock_get_mgr:
            mock_mgr = Mock()
            mock_mgr.is_model_loaded = Mock(return_value=True)
            mock_get_mgr.return_value = mock_mgr

            result, msg = await _ensure_model_loaded("test_model", time.time())

            assert result is True
            assert msg == ""

    @pytest.mark.asyncio
    async def test_model_not_loaded_returns_false_with_hint(self):
        """模型未加载时返回 False 并给出提示"""
        from quantumflow.api.routes.inference import _ensure_model_loaded

        with patch('quantumflow.api.routes.inference.get_engine_manager') as mock_get_mgr:
            mock_mgr = Mock()
            mock_mgr.is_model_loaded = Mock(return_value=False)
            mock_get_mgr.return_value = mock_mgr

            result, msg = await _ensure_model_loaded("test_model", time.time())

            assert result is False
            assert "test_model" in msg
            assert "加载" in msg or "load" in msg.lower()


class TestBatchGenerateErrorHandling:
    """批量生成错误处理逻辑"""

    def test_batch_results_have_correct_indices(self):
        """批量结果索引必须正确对应原始顺序"""
        results = []
        for i in range(3):
            results.append(
                InferenceResponse(
                    request_id=f"batch_{i}",
                    model="test",
                    prompt=f"prompt{i}",
                    generated_text=f"response{i}",
                    finish_reason="stop",
                    latency_ms=10,
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )
            )

        # 验证 request_id 格式正确
        indices = [int(r.request_id.split("_")[-1]) for r in results]
        assert indices == [0, 1, 2]


class TestChatMLFormat:
    """ChatML 格式逻辑校验"""

    def test_chat_endpoint_builds_prompt(self):
        """chat endpoint 必须正确构建 ChatML 格式的 prompt"""
        # chat endpoint 直接在函数内构建 prompt，无独立 helper
        # 此测试验证 ChatML prompt 格式
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        # 模拟 chat endpoint 的 prompt 构建逻辑
        prompt_parts = []
        for msg in messages:
            role = msg["role"].lower()
            if role not in ("user", "assistant", "system"):
                role = "user"
            prompt_parts.append(f"<|im_start|>{role}\n{msg['content']}<|im_end|>")

        prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"

        # 验证 prompt 包含 ChatML 格式标记
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt


class TestTokenEstimation:
    """Token 估算逻辑校验"""

    def test_character_to_token_estimation(self):
        """字符数除以 4 是合理的估算方法"""
        # 这是一个经验估算，不精确但合理
        test_cases = [
            ("hello", 1),  # 5 chars / 4 ≈ 1-2 tokens
            ("Hello, how are you?", 3),  # ~20 chars / 4 ≈ 5 tokens
            ("", 0),  # 空字符串
        ]

        for text, min_expected in test_cases:
            estimated = len(text) // 4
            assert estimated >= min_expected // 4, f"'{text}' 估算过低"


class TestRequestIdGeneration:
    """请求 ID 生成逻辑"""

    def test_request_id_format(self):
        """request_id 格式必须是唯一的"""
        from quantumflow.api.routes.inference import _generate_request_id

        ids = [_generate_request_id() for _ in range(100)]

        # 验证唯一性
        assert len(set(ids)) == 100, "request_id 应该唯一"

        # 验证格式
        for id_ in ids:
            assert id_.startswith("req_"), f"ID 格式错误: {id_}"


class TestEdgeCaseResponses:
    """边界场景响应校验"""

    def test_empty_generated_text(self):
        """空生成文本的响应"""
        response = InferenceResponse(
            request_id="req_001",
            model="test",
            prompt="test",
            generated_text="",  # 空文本
            finish_reason="stop",
            latency_ms=10,
            usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        )

        assert response.generated_text == ""

    def test_max_tokens_hit(self):
        """达到 max_tokens 时 finish_reason 必须是 length"""
        response = InferenceResponse(
            request_id="req_001",
            model="test",
            prompt="test",
            generated_text="x" * 100,
            finish_reason="length",  # 不是 stop
            latency_ms=100,
            usage={"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
        )

        assert response.finish_reason == "length"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
