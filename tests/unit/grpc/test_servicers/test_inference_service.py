"""InferenceService 单元测试

严格测试推理服务的业务逻辑：
1. 请求验证（边界值、非法输入）
2. 模型加载状态检查
3. 推理参数传递
4. 响应状态和内容正确性
5. 错误处理
"""

import uuid
from unittest.mock import MagicMock, Mock

import pytest

from quantumflow.grpc.generated import quantumflow_pb2
from quantumflow.grpc.services.inference import InferenceServiceServicer


import grpc


class MockServicerContext:
    """Mock gRPC ServicerContext"""

    def __init__(self):
        self._aborted = False
        self._abort_code = None
        self._abort_message = None

    def abort(self, code, message):
        self._aborted = True
        self._abort_code = code
        self._abort_message = message
        raise Exception(f"abort: {code}, {message}")


class TestInferenceServiceValidation:
    """InferenceService 请求验证测试"""

    @pytest.fixture
    def servicer(self):
        """创建服务实例（无引擎管理器，模拟模式）"""
        return InferenceServiceServicer()

    def test_rejects_empty_request_id(self, servicer):
        """拒绝空的 request_id"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="",
            model_name="llama-2-7b",
            prompt="Hello",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Inference(request, context)

        assert "abort" in str(exc_info.value)
        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_model_name(self, servicer):
        """拒绝空的 model_name"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="",
            prompt="Hello",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Inference(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_negative_max_tokens(self, servicer):
        """拒绝负数 max_tokens"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            max_tokens=-1,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Inference(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_max_tokens_exceeding_limit(self, servicer):
        """拒绝超过上限的 max_tokens"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            max_tokens=10000,  # 超过 8192
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Inference(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_invalid_temperature(self, servicer):
        """拒绝无效的 temperature"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            temperature=5.0,  # 超出 [0, 2]
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Inference(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_invalid_top_p(self, servicer):
        """拒绝无效的 top_p"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            top_p=1.5,  # 超出 (0, 1]
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Inference(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_accepts_zero_top_p(self, servicer):
        """top_p=0 被接受"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            top_p=0,
        )
        context = MockServicerContext()

        # 不应该抛异常
        response = servicer.Inference(request, context)
        assert response.request_id == request.request_id

    def test_accepts_zero_repetition_penalty(self, servicer):
        """repetition_penalty=0 被接受"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            repetition_penalty=0,
        )
        context = MockServicerContext()

        # 不应该抛异常
        response = servicer.Inference(request, context)
        assert response.request_id == request.request_id


class TestInferenceServiceSuccess:
    """InferenceService 成功场景测试"""

    @pytest.fixture
    def servicer_with_mock_engine(self):
        """创建带 Mock 引擎管理器的服务"""
        mock_engine = MagicMock()
        mock_engine.is_model_loaded.return_value = True
        mock_engine.generate.return_value = MagicMock(
            text="Generated text",
            tokens_generated=10,
        )
        return InferenceServiceServicer(engine_manager=mock_engine)

    def test_returns_success_response(self, servicer_with_mock_engine):
        """返回成功响应"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello, world!",
            max_tokens=100,
            temperature=0.7,
        )
        context = MockServicerContext()

        response = servicer_with_mock_engine.Inference(request, context)

        assert response.status == quantumflow_pb2.STATUS_SUCCESS
        assert response.request_id == request.request_id
        assert len(response.text) > 0
        assert response.tokens_generated > 0
        assert response.latency_ms >= 0

    def test_passes_correct_parameters_to_engine(self, servicer_with_mock_engine):
        """正确传递参数给引擎"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            max_tokens=50,
            temperature=0.9,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2,
        )
        context = MockServicerContext()

        servicer_with_mock_engine.Inference(request, context)

        servicer_with_mock_engine.engine_manager.generate.assert_called_once()
        call_kwargs = servicer_with_mock_engine.engine_manager.generate.call_args[1]
        assert call_kwargs["model_name"] == "llama-2-7b"
        assert call_kwargs["prompt"] == "Hello"
        assert call_kwargs["max_tokens"] == 50
        assert abs(call_kwargs["temperature"] - 0.9) < 0.01
        assert abs(call_kwargs["top_p"] - 0.95) < 0.01

    def test_handles_extra_params(self, servicer_with_mock_engine):
        """处理额外参数"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            extra_params={"custom_param": "value"},
        )
        context = MockServicerContext()

        servicer_with_mock_engine.Inference(request, context)

        call_kwargs = servicer_with_mock_engine.engine_manager.generate.call_args[1]
        assert "extra_params" in call_kwargs
        assert call_kwargs["extra_params"]["custom_param"] == "value"

    def test_model_not_loaded_returns_not_found(self, servicer_with_mock_engine):
        """模型未加载返回 NOT_FOUND"""
        servicer_with_mock_engine.engine_manager.is_model_loaded.return_value = False

        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="non-existent-model",
            prompt="Hello",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer_with_mock_engine.Inference(request, context)

        assert context._abort_code == grpc.StatusCode.NOT_FOUND


class TestInferenceServiceEdgeCases:
    """InferenceService 边界情况测试"""

    @pytest.fixture
    def servicer(self):
        return InferenceServiceServicer()

    def test_empty_prompt_allowed(self, servicer):
        """空 prompt 被接受"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="",
        )
        context = MockServicerContext()

        # 不应该抛异常
        response = servicer.Inference(request, context)
        # 响应状态取决于实际实现

    def test_max_tokens_zero_allowed(self, servicer):
        """max_tokens=0 被接受"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            max_tokens=0,
        )
        context = MockServicerContext()

        response = servicer.Inference(request, context)
        assert response.request_id == request.request_id

    def test_temperature_zero_allowed(self, servicer):
        """temperature=0 被接受"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            temperature=0.0,
        )
        context = MockServicerContext()

        response = servicer.Inference(request, context)
        assert response.request_id == request.request_id

    def test_boundary_temperature_values(self, servicer):
        """边界 temperature 值"""
        for temp in [0.0, 1.0, 2.0]:
            request = quantumflow_pb2.InferenceRequest(
                request_id=str(uuid.uuid4()),
                model_name="llama-2-7b",
                prompt="Hello",
                temperature=temp,
            )
            context = MockServicerContext()

            # 不应该抛异常
            response = servicer.Inference(request, context)
            assert response.request_id == request.request_id


class TestInferenceServiceBuildSamplingParams:
    """InferenceService._build_sampling_params 测试"""

    @pytest.fixture
    def servicer(self):
        return InferenceServiceServicer()

    def test_builds_basic_params(self, servicer):
        """构建基本参数"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="test",
            model_name="test-model",
            prompt="Hello",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        params = servicer._build_sampling_params(request)

        assert params["max_tokens"] == 100
        assert abs(params["temperature"] - 0.7) < 0.01
        assert abs(params["top_p"] - 0.9) < 0.01

    def test_includes_top_k_when_positive(self, servicer):
        """top_k > 0 时包含"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="test",
            model_name="test-model",
            prompt="Hello",
            top_k=50,
        )

        params = servicer._build_sampling_params(request)

        assert "top_k" in params
        assert params["top_k"] == 50

    def test_excludes_top_k_when_zero(self, servicer):
        """top_k = 0 时不包含"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="test",
            model_name="test-model",
            prompt="Hello",
            top_k=0,
        )

        params = servicer._build_sampling_params(request)

        assert "top_k" not in params

    def test_includes_repetition_penalty_when_not_one(self, servicer):
        """repetition_penalty != 1.0 时包含"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="test",
            model_name="test-model",
            prompt="Hello",
            repetition_penalty=1.2,
        )

        params = servicer._build_sampling_params(request)

        assert "repetition_penalty" in params
        assert abs(params["repetition_penalty"] - 1.2) < 0.01

    def test_excludes_repetition_penalty_when_one(self, servicer):
        """repetition_penalty = 1.0 时不包含"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="test",
            model_name="test-model",
            prompt="Hello",
            repetition_penalty=1.0,
        )

        params = servicer._build_sampling_params(request)

        assert "repetition_penalty" not in params

    def test_includes_extra_params(self, servicer):
        """包含额外参数"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="test",
            model_name="test-model",
            prompt="Hello",
            extra_params={"key1": "value1", "key2": "value2"},
        )

        params = servicer._build_sampling_params(request)

        assert "extra_params" in params
        assert params["extra_params"]["key1"] == "value1"
        assert params["extra_params"]["key2"] == "value2"


class TestBatchInference:
    """BatchInference 测试"""

    @pytest.fixture
    def servicer(self):
        mock_engine = MagicMock()
        mock_engine.is_model_loaded.return_value = True
        mock_engine.batch_generate.return_value = [
            MagicMock(text=f"Result {i}", tokens_generated=10)
            for i in range(3)
        ]
        return InferenceServiceServicer(engine_manager=mock_engine)

    def test_empty_batch_returns_empty_results(self, servicer):
        """空批量请求返回空结果"""
        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=[],
        )
        context = MockServicerContext()

        response = servicer.BatchInference(request, context)

        assert response.batch_id == request.batch_id
        assert len(response.results) == 0
        assert response.status == quantumflow_pb2.STATUS_SUCCESS

    def test_batch_returns_all_results(self, servicer):
        """批量返回所有结果"""
        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
        )
        context = MockServicerContext()

        response = servicer.BatchInference(request, context)

        assert len(response.results) == 3
        assert response.status == quantumflow_pb2.STATUS_SUCCESS
        assert response.batch_id == request.batch_id

    def test_batch_uses_correct_model(self, servicer):
        """批量使用正确的模型"""
        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="mixtral-8x7b",
            prompts=["Prompt 1"],
        )
        context = MockServicerContext()

        servicer.BatchInference(request, context)

        servicer.engine_manager.batch_generate.assert_called_once()
        call_args = servicer.engine_manager.batch_generate.call_args
        assert call_args[1]["model_name"] == "mixtral-8x7b"

    def test_batch_passes_max_tokens(self, servicer):
        """批量传递 max_tokens"""
        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=["Prompt 1"],
            max_tokens=50,
        )
        context = MockServicerContext()

        servicer.BatchInference(request, context)

        call_args = servicer.engine_manager.batch_generate.call_args
        assert call_args[1]["max_tokens"] == 50

    def test_batch_model_not_loaded(self, servicer):
        """模型未加载"""
        servicer.engine_manager.is_model_loaded.return_value = False

        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="non-existent",
            prompts=["Prompt 1"],
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.BatchInference(request, context)

        assert context._abort_code == grpc.StatusCode.NOT_FOUND


class TestInferenceServiceExceptionHandling:
    """InferenceService 异常处理测试"""

    @pytest.fixture
    def servicer_with_mock_engine(self):
        """创建带 Mock 引擎管理器的服务"""
        mock_engine = MagicMock()
        mock_engine.is_model_loaded.return_value = True
        return InferenceServiceServicer(engine_manager=mock_engine)

    def test_inference_exception_returns_error_response(self, servicer_with_mock_engine):
        """Inference 异常返回错误响应"""
        servicer_with_mock_engine.engine_manager.generate.side_effect = RuntimeError("Engine error")

        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
        )
        context = MockServicerContext()

        response = servicer_with_mock_engine.Inference(request, context)

        assert response.status == quantumflow_pb2.STATUS_ERROR
        assert "Engine error" in response.error_message

    def test_inference_model_loaded_check_raises_exception(self, servicer_with_mock_engine):
        """模型加载检查时引擎抛出异常"""
        servicer_with_mock_engine.engine_manager.is_model_loaded.side_effect = Exception("Check failed")

        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
        )
        context = MockServicerContext()

        # 由于 is_model_loaded 抛出异常后被捕获，返回 True
        response = servicer_with_mock_engine.Inference(request, context)
        # 默认返回模拟响应
        assert response.request_id == request.request_id

    def test_batch_inference_exception_returns_error(self, servicer_with_mock_engine):
        """BatchInference 异常返回错误响应"""
        servicer_with_mock_engine.engine_manager.batch_generate.side_effect = RuntimeError("Batch error")

        request = quantumflow_pb2.BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=["Prompt 1", "Prompt 2"],
        )
        context = MockServicerContext()

        response = servicer_with_mock_engine.BatchInference(request, context)

        assert response.status == quantumflow_pb2.STATUS_ERROR


class TestInferenceServiceValidateRequest:
    """_validate_request 边界测试"""

    @pytest.fixture
    def servicer(self):
        return InferenceServiceServicer()

    def test_rejects_negative_repetition_penalty(self, servicer):
        """拒绝负数 repetition_penalty"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            repetition_penalty=-0.1,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Inference(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT


class TestInferenceServiceStreamSimulated:
    """InferenceStream 模拟模式测试"""

    @pytest.fixture
    def servicer(self):
        return InferenceServiceServicer()

    def test_inference_stream_simulated_yields_responses(self, servicer):
        """模拟流式推理返回响应"""
        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello world",
            max_tokens=3,
        )
        context = MockServicerContext()

        responses = list(servicer.InferenceStream(request, context))

        assert len(responses) > 0
        # 最后一个响应的 status 应该是 SUCCESS
        assert responses[-1].status in [quantumflow_pb2.STATUS_SUCCESS, quantumflow_pb2.STATUS_PROCESSING]

    def test_inference_stream_validation_error(self, servicer):
        """流式推理验证错误"""
        request = quantumflow_pb2.InferenceRequest(
            request_id="",  # 空 request_id
            model_name="llama-2-7b",
            prompt="Hello",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            list(servicer.InferenceStream(request, context))

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT


class TestInferenceServiceStreamWithEngine:
    """InferenceStream 带引擎测试"""

    @pytest.fixture
    def servicer_with_mock_engine(self):
        """创建带 Mock 引擎管理器的服务"""
        mock_engine = MagicMock()
        mock_engine.is_model_loaded.return_value = True
        return InferenceServiceServicer(engine_manager=mock_engine)

    def test_inference_stream_exception_returns_error(self, servicer_with_mock_engine):
        """流式推理异常返回错误响应"""
        servicer_with_mock_engine.engine_manager.generate_stream.side_effect = RuntimeError("Stream error")

        request = quantumflow_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            max_tokens=5,
        )
        context = MockServicerContext()

        responses = list(servicer_with_mock_engine.InferenceStream(request, context))

        # 应该有至少一个错误响应
        assert any(r.status == quantumflow_pb2.STATUS_ERROR for r in responses)
