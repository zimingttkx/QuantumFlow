"""ModelManagementService gRPC 服务实现

提供模型管理功能：
- 加载模型 (LoadModel)
- 卸载模型 (UnloadModel)
- 列出模型 (ListModels)
"""

from typing import List

import grpc

from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc
from quantumflow.grpc.exceptions import (
    InvalidRequestError,
    ModelNotLoadedError,
    ResourceUnavailableError,
)
from quantumflow.grpc.services.base import BaseService


class ModelManagementServiceServicer(
    BaseService, quantumflow_pb2_grpc.ModelManagementServiceServicer
):
    """模型管理服务实现

    提供模型加载、卸载和列表功能。
    """

    def __init__(self, engine_manager=None):
        """
        Args:
            engine_manager: EngineManager 实例
        """
        super().__init__(engine_manager=engine_manager)

    def LoadModel(
        self,
        request: quantumflow_pb2.LoadModelRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.LoadModelResponse:
        """加载模型

        Args:
            request: LoadModelRequest 消息
            context: gRPC 上下文

        Returns:
            LoadModelResponse 消息
        """
        # 验证请求
        if not request.model_name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_name cannot be empty")

        if request.backend == quantumflow_pb2.MODEL_BACKEND_UNSPECIFIED:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "backend cannot be UNSPECIFIED")

        if request.gpu_memory_utilization < 0 or request.gpu_memory_utilization > 1:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "gpu_memory_utilization must be in [0, 1]",
            )

        try:
            if self.engine_manager:
                # 实际加载模型
                result = self.engine_manager.load_model(
                    model_name=request.model_name,
                    backend=self._backend_to_string(request.backend),
                    tensor_parallel_size=request.tensor_parallel_size,
                    gpu_memory_utilization=request.gpu_memory_utilization,
                    backend_config=dict(request.backend_config) if request.backend_config else {},
                )

                return quantumflow_pb2.LoadModelResponse(
                    success=True,
                    model_name=request.model_name,
                    message="Model loaded successfully",
                    memory_allocated=quantumflow_pb2.GPUMemory(
                        used_bytes=result.get("memory_used", 0),
                    ),
                )
            else:
                # 模拟加载成功
                return quantumflow_pb2.LoadModelResponse(
                    success=True,
                    model_name=request.model_name,
                    message="Model loaded successfully (simulated)",
                    memory_allocated=quantumflow_pb2.GPUMemory(
                        total_bytes=16 * 1024**3,
                        used_bytes=14 * 1024**3,
                    ),
                )

        except ResourceUnavailableError as e:
            return quantumflow_pb2.LoadModelResponse(
                success=False,
                model_name=request.model_name,
                message=e.message,
            )

        except Exception as e:
            return quantumflow_pb2.LoadModelResponse(
                success=False,
                model_name=request.model_name,
                message=f"Failed to load model: {str(e)}",
            )

    def UnloadModel(
        self,
        request: quantumflow_pb2.UnloadModelRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.UnloadModelResponse:
        """卸载模型

        Args:
            request: UnloadModelRequest 消息
            context: gRPC 上下文

        Returns:
            UnloadModelResponse 消息
        """
        if not request.model_name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_name cannot be empty")

        try:
            if self.engine_manager:
                memory_freed = self.engine_manager.unload_model(request.model_name)
            else:
                # 模拟卸载
                memory_freed = 14 * 1024**3

            return quantumflow_pb2.UnloadModelResponse(
                success=True,
                message="Model unloaded successfully",
                memory_freed_bytes=memory_freed,
            )

        except ModelNotLoadedError as e:
            return quantumflow_pb2.UnloadModelResponse(
                success=False,
                message=e.message,
                memory_freed_bytes=0,
            )

        except Exception as e:
            return quantumflow_pb2.UnloadModelResponse(
                success=False,
                message=f"Failed to unload model: {str(e)}",
                memory_freed_bytes=0,
            )

    def ListModels(
        self,
        request: quantumflow_pb2.ListModelsRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.ListModelsResponse:
        """列出模型

        Args:
            request: ListModelsRequest 消息
            context: gRPC 上下文

        Returns:
            ListModelsResponse 消息
        """
        try:
            models: List[quantumflow_pb2.ModelInfo] = []

            if self.engine_manager:
                # 获取已加载模型列表
                loaded_models = self.engine_manager.list_loaded_models()

                for model in loaded_models:
                    models.append(quantumflow_pb2.ModelInfo(
                        name=model.get("name", ""),
                        backend=model.get("backend", ""),
                        size_bytes=model.get("size_bytes", 0),
                        tensor_parallel_size=model.get("tensor_parallel_size", 1),
                        is_loaded=model.get("is_loaded", True),
                    ))
            else:
                # 模拟模型列表
                models.append(quantumflow_pb2.ModelInfo(
                    name="llama-2-7b",
                    backend="vllm",
                    size_bytes=13 * 1024**3,
                    tensor_parallel_size=1,
                    is_loaded=True,
                ))
                models.append(quantumflow_pb2.ModelInfo(
                    name="llama-2-70b",
                    backend="vllm",
                    size_bytes=70 * 1024**3,
                    tensor_parallel_size=4,
                    is_loaded=False,
                ))

            return quantumflow_pb2.ListModelsResponse(models=models)

        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Failed to list models: {str(e)}")

    def _backend_to_string(self, backend: quantumflow_pb2.ModelBackend) -> str:
        """将 ModelBackend 枚举转换为字符串

        Args:
            backend: ModelBackend 枚举值

        Returns:
            字符串
        """
        mapping = {
            quantumflow_pb2.MODEL_BACKEND_VLLM: "vllm",
            quantumflow_pb2.MODEL_BACKEND_HUGGINGFACE: "huggingface",
            quantumflow_pb2.MODEL_BACKEND_TGI: "tgi",
            quantumflow_pb2.MODEL_BACKEND_SGLANG: "sglang",
        }
        return mapping.get(backend, "unknown")
