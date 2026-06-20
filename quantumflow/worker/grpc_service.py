"""Worker gRPC Service

Worker 节点上的 gRPC 服务，用于接收 Controller 的推理请求。
"""

import time
from typing import Iterator

import grpc

from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc
from quantumflow.grpc.services.base import BaseService


class WorkerGrpcService(
    BaseService, quantumflow_pb2_grpc.InferenceServiceServicer
):
    """Worker gRPC 推理服务

    在 Worker 节点上运行，接收 Controller 分发的推理请求。
    """

    def __init__(self, worker_node=None):
        """
        Args:
            worker_node: WorkerNode 实例
        """
        super().__init__(engine_manager=None)
        self.worker_node = worker_node

    def Inference(
        self,
        request: quantumflow_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.InferenceResponse:
        """处理推理请求

        Args:
            request: InferenceRequest 消息
            context: gRPC 上下文

        Returns:
            InferenceResponse 消息
        """
        start_time = time.perf_counter()

        # 验证请求
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id cannot be empty")
            return quantumflow_pb2.InferenceResponse()

        if not request.model_name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_name cannot be empty")
            return quantumflow_pb2.InferenceResponse()

        try:
            # 如果有 WorkerNode，使用它执行推理
            if self.worker_node and self.worker_node.engine_manager:
                result = self.worker_node.engine_manager.generate(
                    model_name=request.model_name,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )

                text = result.text if hasattr(result, 'text') else str(result)
                tokens_generated = result.tokens_generated if hasattr(result, 'tokens_generated') else len(text.split())
            else:
                # 模拟响应
                text = f"Worker inference: {request.prompt[:50]}..."
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
        """处理流式推理请求

        Args:
            request: InferenceRequest 消息
            context: gRPC 上下文

        Yields:
            InferenceResponse 消息
        """
        start_time = time.perf_counter()
        actual_tokens_generated = 0

        try:
            if self.worker_node and self.worker_node.engine_manager:
                results = self.worker_node.engine_manager.generate_stream(
                    model_name=request.model_name,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            else:
                # 模拟流式响应
                words = request.prompt.split()
                results = [f"token_{i}" for i in range(request.max_tokens)]

            for i, token in enumerate(results):
                duration_ms = (time.perf_counter() - start_time) * 1000
                text = token.text if hasattr(token, 'text') else str(token)
                actual_tokens_generated += 1

                yield quantumflow_pb2.InferenceResponse(
                    request_id=request.request_id,
                    status=quantumflow_pb2.STATUS_PROCESSING,
                    text=text,
                    tokens_generated=actual_tokens_generated,
                    latency_ms=duration_ms,
                )

            # 最后一条 — use actual token count, not request.max_tokens
            duration_ms = (time.perf_counter() - start_time) * 1000
            yield quantumflow_pb2.InferenceResponse(
                request_id=request.request_id,
                status=quantumflow_pb2.STATUS_SUCCESS,
                text="",
                tokens_generated=actual_tokens_generated,
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
        """处理批量推理请求

        Args:
            request: BatchInferenceRequest 消息
            context: gRPC 上下文

        Returns:
            BatchInferenceResponse 消息
        """
        start_time = time.perf_counter()

        if not request.batch_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "batch_id cannot be empty")
            return quantumflow_pb2.BatchInferenceResponse()

        if not request.prompts:
            return quantumflow_pb2.BatchInferenceResponse(
                batch_id=request.batch_id,
                status=quantumflow_pb2.STATUS_SUCCESS,
                results=[],
                total_latency_ms=0,
            )

        try:
            results = []

            if self.worker_node and self.worker_node.engine_manager:
                batch_results = self.worker_node.engine_manager.batch_generate(
                    model_name=request.model_name,
                    prompts=list(request.prompts),
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )

                for i, result in enumerate(batch_results):
                    text = result.text if hasattr(result, 'text') else str(result)
                    tokens = result.tokens_generated if hasattr(result, 'tokens_generated') else len(text.split())
                    results.append(quantumflow_pb2.InferenceResponse(
                        request_id=f"{request.batch_id}_{i}",
                        status=quantumflow_pb2.STATUS_SUCCESS,
                        text=text,
                        tokens_generated=tokens,
                    ))
            else:
                # 模拟批量响应
                for i, prompt in enumerate(request.prompts):
                    text = f"Batch {i}: {prompt[:30]}..."
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
                error_message=str(e),
                total_latency_ms=duration_ms,
            )
