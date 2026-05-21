"""ModelManagementService 单元测试

严格测试模型管理服务的业务逻辑：
1. 请求验证
2. 模型加载
3. 模型卸载
4. 模型列表
"""

import uuid
from unittest.mock import MagicMock

import pytest

from quantumflow.grpc.generated import quantumflow_pb2
from quantumflow.grpc.services.model_management import ModelManagementServiceServicer


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


class TestModelManagementServiceValidation:
    """ModelManagementService 请求验证测试"""

    @pytest.fixture
    def servicer(self):
        return ModelManagementServiceServicer()

    def test_rejects_empty_model_name_on_load(self, servicer):
        """拒绝空的 model_name（加载）"""
        request = quantumflow_pb2.LoadModelRequest(
            model_name="",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.LoadModel(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_unspecified_backend(self, servicer):
        """拒绝 UNSPECIFIED backend"""
        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-7b",
            backend=quantumflow_pb2.MODEL_BACKEND_UNSPECIFIED,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.LoadModel(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_invalid_utilization_too_low(self, servicer):
        """拒绝过低的 gpu_memory_utilization"""
        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-7b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
            gpu_memory_utilization=-0.1,  # 负数
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.LoadModel(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_invalid_utilization_too_high(self, servicer):
        """拒绝过高的 gpu_memory_utilization"""
        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-7b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
            gpu_memory_utilization=1.5,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.LoadModel(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_model_name_on_unload(self, servicer):
        """拒绝空的 model_name（卸载）"""
        request = quantumflow_pb2.UnloadModelRequest(model_name="")
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.UnloadModel(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT


class TestModelManagementServiceLoadModel:
    """模型加载测试"""

    @pytest.fixture
    def servicer(self):
        mock_engine = MagicMock()
        return ModelManagementServiceServicer(engine_manager=mock_engine)

    def test_load_model_success(self, servicer):
        """加载成功"""
        servicer.engine_manager.load_model.return_value = {
            "memory_used": 14 * 1024**3,
        }

        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-7b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        context = MockServicerContext()

        response = servicer.LoadModel(request, context)

        assert response.success is True
        assert response.model_name == "llama-2-7b"

    def test_load_model_passes_correct_backend(self, servicer):
        """传递正确的 backend"""
        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-7b",
            backend=quantumflow_pb2.MODEL_BACKEND_HUGGINGFACE,
        )
        context = MockServicerContext()

        servicer.LoadModel(request, context)

        servicer.engine_manager.load_model.assert_called_once()
        call_kwargs = servicer.engine_manager.load_model.call_args[1]
        assert call_kwargs["backend"] == "huggingface"

    def test_load_model_passes_tensor_parallel_size(self, servicer):
        """传递 tensor_parallel_size"""
        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-70b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
            tensor_parallel_size=4,
        )
        context = MockServicerContext()

        servicer.LoadModel(request, context)

        call_kwargs = servicer.engine_manager.load_model.call_args[1]
        assert call_kwargs["tensor_parallel_size"] == 4

    def test_load_model_resource_unavailable(self, servicer):
        """资源不足"""
        from quantumflow.grpc.exceptions import ResourceUnavailableError

        servicer.engine_manager.load_model.side_effect = ResourceUnavailableError(
            resource="GPU memory",
            required=80,
            available=40,
        )

        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-70b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
        )
        context = MockServicerContext()

        response = servicer.LoadModel(request, context)

        assert response.success is False
        assert "memory" in response.message.lower()


class TestModelManagementServiceUnloadModel:
    """模型卸载测试"""

    @pytest.fixture
    def servicer(self):
        mock_engine = MagicMock()
        return ModelManagementServiceServicer(engine_manager=mock_engine)

    def test_unload_model_success(self, servicer):
        """卸载成功"""
        servicer.engine_manager.unload_model.return_value = 14 * 1024**3

        request = quantumflow_pb2.UnloadModelRequest(model_name="llama-2-7b")
        context = MockServicerContext()

        response = servicer.UnloadModel(request, context)

        assert response.success is True
        assert response.memory_freed_bytes == 14 * 1024**3

    def test_unload_model_not_loaded(self, servicer):
        """模型未加载"""
        from quantumflow.grpc.exceptions import ModelNotLoadedError

        servicer.engine_manager.unload_model.side_effect = ModelNotLoadedError(
            model_name="non-existent",
        )

        request = quantumflow_pb2.UnloadModelRequest(model_name="non-existent")
        context = MockServicerContext()

        response = servicer.UnloadModel(request, context)

        assert response.success is False


class TestModelManagementServiceListModels:
    """模型列表测试"""

    @pytest.fixture
    def servicer(self):
        mock_engine = MagicMock()
        mock_engine.list_loaded_models.return_value = [
            {
                "name": "llama-2-7b",
                "backend": "vllm",
                "size_bytes": 13 * 1024**3,
                "tensor_parallel_size": 1,
                "is_loaded": True,
            },
            {
                "name": "mixtral-8x7b",
                "backend": "vllm",
                "size_bytes": 40 * 1024**3,
                "tensor_parallel_size": 2,
                "is_loaded": True,
            },
        ]
        return ModelManagementServiceServicer(engine_manager=mock_engine)

    def test_list_models_returns_all(self, servicer):
        """列出所有模型"""
        request = quantumflow_pb2.ListModelsRequest()
        context = MockServicerContext()

        response = servicer.ListModels(request, context)

        assert len(response.models) == 2

    def test_list_models_contains_correct_info(self, servicer):
        """列表包含正确信息"""
        request = quantumflow_pb2.ListModelsRequest()
        context = MockServicerContext()

        response = servicer.ListModels(request, context)

        model_names = [m.name for m in response.models]
        assert "llama-2-7b" in model_names
        assert "mixtral-8x7b" in model_names

    def test_list_models_contains_backend_info(self, servicer):
        """列表包含 backend 信息"""
        request = quantumflow_pb2.ListModelsRequest()
        context = MockServicerContext()

        response = servicer.ListModels(request, context)

        for model in response.models:
            assert model.backend == "vllm"


class TestModelManagementServiceSimulatedMode:
    """模拟模式测试"""

    @pytest.fixture
    def servicer(self):
        return ModelManagementServiceServicer()

    def test_load_model_returns_simulated_success(self, servicer):
        """模拟加载成功"""
        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-7b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
        )
        context = MockServicerContext()

        response = servicer.LoadModel(request, context)

        assert response.success is True
        assert response.model_name == "llama-2-7b"

    def test_list_models_returns_simulated_list(self, servicer):
        """模拟列出模型"""
        request = quantumflow_pb2.ListModelsRequest()
        context = MockServicerContext()

        response = servicer.ListModels(request, context)

        assert len(response.models) == 2
        model_names = [m.name for m in response.models]
        assert "llama-2-7b" in model_names


class TestModelManagementServiceExceptions:
    """通用异常处理测试"""

    @pytest.fixture
    def servicer(self):
        mock_engine = MagicMock()
        return ModelManagementServiceServicer(engine_manager=mock_engine)

    def test_load_model_generic_exception(self, servicer):
        """加载时通用异常"""
        servicer.engine_manager.load_model.side_effect = RuntimeError("Unexpected error")

        request = quantumflow_pb2.LoadModelRequest(
            model_name="llama-2-7b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
        )
        context = MockServicerContext()

        response = servicer.LoadModel(request, context)

        assert response.success is False
        assert "Failed to load model" in response.message

    def test_unload_model_generic_exception(self, servicer):
        """卸载时通用异常"""
        servicer.engine_manager.unload_model.side_effect = RuntimeError("Unexpected error")

        request = quantumflow_pb2.UnloadModelRequest(model_name="llama-2-7b")
        context = MockServicerContext()

        response = servicer.UnloadModel(request, context)

        assert response.success is False
        assert "Failed to unload model" in response.message

    def test_list_models_generic_exception(self, servicer):
        """列出模型时通用异常"""
        servicer.engine_manager.list_loaded_models.side_effect = RuntimeError("Unexpected error")

        request = quantumflow_pb2.ListModelsRequest()
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.ListModels(request, context)


class TestModelManagementServiceSimulatedModeUnload:
    """模拟模式卸载测试"""

    @pytest.fixture
    def servicer(self):
        return ModelManagementServiceServicer()

    def test_unload_model_simulated(self, servicer):
        """模拟卸载（无 engine_manager）"""
        request = quantumflow_pb2.UnloadModelRequest(model_name="llama-2-7b")
        context = MockServicerContext()

        response = servicer.UnloadModel(request, context)

        assert response.success is True
        assert response.memory_freed_bytes == 14 * 1024**3
