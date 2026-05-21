"""SchedulerService gRPC 服务实现

提供调度功能：
- 提交调度请求 (SubmitRequest)
- 取消请求 (Cancel)
- 获取状态 (GetStatus)
- 取消请求流 (CancelRequestStream)
"""

import time
from typing import Iterator, List

import grpc

from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc
from quantumflow.grpc.exceptions import (
    InvalidRequestError,
    SchedulingError,
    ResourceUnavailableError,
)
from quantumflow.grpc.services.base import BaseService


class SchedulerServiceServicer(BaseService, quantumflow_pb2_grpc.SchedulerServiceServicer):
    """调度服务实现

    提供调度请求提交、取消和状态查询功能。
    """

    def __init__(self, scheduler=None, engine_manager=None):
        """
        Args:
            scheduler: 调度器实例
            engine_manager: 引擎管理器（用于获取节点信息）
        """
        super().__init__(scheduler=scheduler, engine_manager=engine_manager)

    def SubmitRequest(
        self,
        request: quantumflow_pb2.SchedulingRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.SchedulingResponse:
        """提交调度请求

        Args:
            request: SchedulingRequest 消息
            context: gRPC 上下文

        Returns:
            SchedulingResponse 消息
        """
        # 验证请求
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id cannot be empty")

        if not request.model_name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_name cannot be empty")

        if request.priority < 0 or request.priority > 10:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "priority must be in range [0, 10]")

        try:
            if self.scheduler:
                # 使用调度器进行调度
                result = self.scheduler.schedule(
                    request_id=request.request_id,
                    model_name=request.model_name,
                    backend=request.backend,
                    tensor_parallel_size=request.tensor_parallel_size,
                    gpu_memory_required_gb=request.gpu_memory_required_gb,
                    priority=request.priority,
                    mode=request.mode,
                )
                return result
            else:
                # 模拟调度成功
                return quantumflow_pb2.SchedulingResponse(
                    request_id=request.request_id,
                    scheduled=True,
                    assigned_node_id="worker-001",
                    assigned_host="192.168.1.100",
                    assigned_port=50051,
                    status=quantumflow_pb2.STATUS_PENDING,
                )

        except ResourceUnavailableError as e:
            return quantumflow_pb2.SchedulingResponse(
                request_id=request.request_id,
                scheduled=False,
                error_message=e.message,
                status=quantumflow_pb2.STATUS_ERROR,
            )

        except SchedulingError as e:
            return quantumflow_pb2.SchedulingResponse(
                request_id=request.request_id,
                scheduled=False,
                error_message=e.message,
                status=quantumflow_pb2.STATUS_ERROR,
            )

        except Exception as e:
            return quantumflow_pb2.SchedulingResponse(
                request_id=request.request_id,
                scheduled=False,
                error_message=f"Scheduling failed: {str(e)}",
                status=quantumflow_pb2.STATUS_ERROR,
            )

    def Cancel(
        self,
        request: quantumflow_pb2.CancelRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.CancelResponse:
        """取消请求

        Args:
            request: CancelRequest 消息
            context: gRPC 上下文

        Returns:
            CancelResponse 消息
        """
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id cannot be empty")

        try:
            if self.scheduler:
                success = self.scheduler.cancel(request.request_id)
            else:
                # 模拟取消
                success = True

            if success:
                return quantumflow_pb2.CancelResponse(
                    success=True,
                    message="Request cancelled successfully",
                )
            else:
                return quantumflow_pb2.CancelResponse(
                    success=False,
                    message="Request not found or already completed",
                )

        except Exception as e:
            return quantumflow_pb2.CancelResponse(
                success=False,
                message=f"Cancellation failed: {str(e)}",
            )

    def GetStatus(
        self,
        request: quantumflow_pb2.GetSchedulingStatusRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.GetSchedulingStatusResponse:
        """获取调度状态

        Args:
            request: GetSchedulingStatusRequest 消息
            context: gRPC 上下文

        Returns:
            GetSchedulingStatusResponse 消息
        """
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id cannot be empty")

        try:
            if self.scheduler:
                result = self.scheduler.get_status(request.request_id)
                return result
            else:
                # 模拟状态
                return quantumflow_pb2.GetSchedulingStatusResponse(
                    request_id=request.request_id,
                    status=quantumflow_pb2.STATUS_PENDING,
                )

        except Exception as e:
            return quantumflow_pb2.GetSchedulingStatusResponse(
                request_id=request.request_id,
                status=quantumflow_pb2.STATUS_ERROR,
                error_message=f"Failed to get status: {str(e)}",
            )

    def CancelRequestStream(
        self,
        request_iterator: Iterator[quantumflow_pb2.CancelRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[quantumflow_pb2.CancelResponse]:
        """取消请求流（双向流式）

        Args:
            request_iterator: CancelRequest 迭代器
            context: gRPC 上下文

        Yields:
            CancelResponse 消息
        """
        for request in request_iterator:
            if not request.request_id:
                yield quantumflow_pb2.CancelResponse(
                    success=False,
                    message="request_id cannot be empty",
                )
                continue

            try:
                if self.scheduler:
                    success = self.scheduler.cancel(request.request_id)
                else:
                    success = True

                yield quantumflow_pb2.CancelResponse(
                    success=success,
                    message="Request cancelled" if success else "Request not found",
                )

            except Exception as e:
                yield quantumflow_pb2.CancelResponse(
                    success=False,
                    message=f"Cancellation failed: {str(e)}",
                )
