"""InferenceService gRPC 服务实现

提供推理功能：
- 同步推理 (Inference)
- 流式推理 (InferenceStream)
- 批量推理 (BatchInference)
"""

import time
from typing import Iterator, List

import grpc

from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc
from quantumflow.grpc.exceptions import (
    InvalidRequestError,
    ModelNotLoadedError,
    InternalServerError,
)
from quantumflow.grpc.services.base import BaseService


class InferenceServiceServicer(BaseService, quantumflow_pb2_grpc.InferenceServiceServicer):
    """推理服务实现

    提供同步推理、流式推理和批量推理功能。
    """

    def __init__(self, engine_manager=None, cluster_manager=None):
        """
        Args:
            engine_manager: EngineManager 实例，用于执行推理
            cluster_manager: ClusterManager 实例，用于查找可用节点
        """
        super().__init__(engine_manager=engine_manager, cluster_manager=cluster_manager)

    def Inference(
        self,
        request: quantumflow_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.InferenceResponse:
        """同步推理

        Args:
            request: InferenceRequest 消息
            context: gRPC 上下文

        Returns:
            InferenceResponse 消息
        """
        start_time = time.perf_counter()

        # 验证请求
        try:
            self._validate_request(request)
        except InvalidRequestError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, e.message)

        # 检查模型是否已加载
        if not self._is_model_loaded(request.model_name):
            error_msg = f"Model '{request.model_name}' is not loaded"
            context.abort(grpc.StatusCode.NOT_FOUND, error_msg)

        try:
            # 构建采样参数
            sampling_params = self._build_sampling_params(request)

            # 执行推理
            if self.engine_manager:
                result = self.engine_manager.generate(
                    model_name=request.model_name,
                    prompt=request.prompt,
                    **sampling_params,
                )
                text = result.text if hasattr(result, 'text') else str(result)
                tokens_generated = result.tokens_generated if hasattr(result, 'tokens_generated') else len(text.split())
            else:
                # 模拟响应（当没有真正的引擎管理器时）
                text = f"Generated text for prompt: {request.prompt[:50]}..."
                tokens_generated = len(text.split())

            duration_ms = (time.perf_counter() - start_time) * 1000

            return quantumflow_pb2.InferenceResponse(
                request_id=request.request_id,
                status=quantumflow_pb2.STATUS_SUCCESS,
                text=text,
                tokens_generated=tokens_generated,
                latency_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return quantumflow_pb2.InferenceResponse(
                request_id=request.request_id,
                status=quantumflow_pb2.STATUS_ERROR,
                error_message=str(e),
                latency_ms=duration_ms,
            )

    def InferenceStream(
        self,
        request: quantumflow_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[quantumflow_pb2.InferenceResponse]:
        """流式推理

        Args:
            request: InferenceRequest 消息
            context: gRPC 上下文

        Yields:
            InferenceResponse 消息（每个 token 一条）
        """
        start_time = time.perf_counter()

        # 验证请求
        try:
            self._validate_request(request)
        except InvalidRequestError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, e.message)

        # 检查模型是否已加载
        if not self._is_model_loaded(request.model_name):
            error_msg = f"Model '{request.model_name}' is not loaded"
            context.abort(grpc.StatusCode.NOT_FOUND, error_msg)

        try:
            # 构建采样参数
            sampling_params = self._build_sampling_params(request)

            # 执行流式推理
            if self.engine_manager:
                results = self.engine_manager.generate_stream(
                    model_name=request.model_name,
                    prompt=request.prompt,
                    **sampling_params,
                )
            else:
                # 模拟流式响应
                words = request.prompt.split()
                results = [f"token_{i}" for i in range(request.max_tokens)]

            # Yield 每个 token
            for i, token in enumerate(results):
                duration_ms = (time.perf_counter() - start_time) * 1000
                text = token.text if hasattr(token, 'text') else str(token)
                yield quantumflow_pb2.InferenceResponse(
                    request_id=request.request_id,
                    status=quantumflow_pb2.STATUS_PROCESSING,
                    text=text,
                    tokens_generated=i + 1,
                    latency_ms=duration_ms,
                )

            # 最后一条表示完成
            duration_ms = (time.perf_counter() - start_time) * 1000
            yield quantumflow_pb2.InferenceResponse(
                request_id=request.request_id,
                status=quantumflow_pb2.STATUS_SUCCESS,
                text="",
                tokens_generated=request.max_tokens,
                latency_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            yield quantumflow_pb2.InferenceResponse(
                request_id=request.request_id,
                status=quantumflow_pb2.STATUS_ERROR,
                error_message=str(e),
                latency_ms=duration_ms,
            )

    def BatchInference(
        self,
        request: quantumflow_pb2.BatchInferenceRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.BatchInferenceResponse:
        """批量推理

        Args:
            request: BatchInferenceRequest 消息
            context: gRPC 上下文

        Returns:
            BatchInferenceResponse 消息
        """
        start_time = time.perf_counter()

        # 验证请求
        if not request.batch_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "batch_id cannot be empty")

        if not request.prompts:
            return quantumflow_pb2.BatchInferenceResponse(
                batch_id=request.batch_id,
                status=quantumflow_pb2.STATUS_SUCCESS,
                results=[],
                total_latency_ms=0,
            )

        # 检查模型是否已加载
        if not self._is_model_loaded(request.model_name):
            context.abort(grpc.StatusCode.NOT_FOUND, f"Model '{request.model_name}' is not loaded")

        try:
            results: List[quantumflow_pb2.InferenceResponse] = []

            if self.engine_manager:
                # 执行批量推理
                batch_results = self.engine_manager.batch_generate(
                    model_name=request.model_name,
                    prompts=request.prompts,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                for result in batch_results:
                    text = result.text if hasattr(result, 'text') else str(result)
                    tokens = result.tokens_generated if hasattr(result, 'tokens_generated') else len(text.split())
                    results.append(quantumflow_pb2.InferenceResponse(
                        request_id=f"{request.batch_id}_{len(results)}",
                        status=quantumflow_pb2.STATUS_SUCCESS,
                        text=text,
                        tokens_generated=tokens,
                    ))
            else:
                # 模拟批量响应
                for i, prompt in enumerate(request.prompts):
                    text = f"Generated {i}: {prompt[:30]}..."
                    results.append(quantumflow_pb2.InferenceResponse(
                        request_id=f"{request.batch_id}_{i}",
                        status=quantumflow_pb2.STATUS_SUCCESS,
                        text=text,
                        tokens_generated=len(text.split()),
                    ))

            duration_ms = (time.perf_counter() - start_time) * 1000

            return quantumflow_pb2.BatchInferenceResponse(
                batch_id=request.batch_id,
                status=quantumflow_pb2.STATUS_SUCCESS,
                results=results,
                total_latency_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return quantumflow_pb2.BatchInferenceResponse(
                batch_id=request.batch_id,
                status=quantumflow_pb2.STATUS_ERROR,
                total_latency_ms=duration_ms,
            )

    def _validate_request(self, request: quantumflow_pb2.InferenceRequest) -> None:
        """验证推理请求

        Args:
            request: InferenceRequest 消息

        Raises:
            InvalidRequestError: 请求无效
        """
        if not request.request_id:
            raise InvalidRequestError(field="request_id", reason="Request ID cannot be empty")

        if not request.model_name:
            raise InvalidRequestError(field="model_name", reason="Model name cannot be empty")

        if request.max_tokens < 0:
            raise InvalidRequestError(field="max_tokens", reason="max_tokens must be non-negative", value=request.max_tokens)

        if request.max_tokens > 8192:
            raise InvalidRequestError(field="max_tokens", reason="max_tokens exceeds maximum (8192)", value=request.max_tokens)

        if request.temperature < 0 or request.temperature > 2:
            raise InvalidRequestError(field="temperature", reason="temperature must be in [0, 2]", value=request.temperature)

        if request.top_p < 0 or request.top_p > 1:
            raise InvalidRequestError(field="top_p", reason="top_p must be in [0, 1]", value=request.top_p)

        if request.repetition_penalty < 0:
            raise InvalidRequestError(field="repetition_penalty", reason="repetition_penalty must be >= 0", value=request.repetition_penalty)

    def _is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载

        Args:
            model_name: 模型名称

        Returns:
            bool
        """
        if self.engine_manager:
            try:
                return self.engine_manager.is_model_loaded(model_name)
            except Exception:
                pass
        # 默认返回 True（允许模拟响应）
        return True

    def _build_sampling_params(self, request: quantumflow_pb2.InferenceRequest) -> dict:
        """构建采样参数

        Args:
            request: InferenceRequest 消息

        Returns:
            采样参数字典
        """
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        if request.top_k > 0:
            params["top_k"] = request.top_k

        if request.repetition_penalty != 1.0:
            params["repetition_penalty"] = request.repetition_penalty

        # 添加额外参数
        if request.extra_params:
            params["extra_params"] = dict(request.extra_params)

        return params
