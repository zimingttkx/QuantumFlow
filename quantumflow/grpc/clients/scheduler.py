"""gRPC SchedulerClient

提供调度功能的 gRPC 客户端。
"""

from typing import Iterator, List, Optional

import grpc

from quantumflow.grpc.channels.pool import GrpcChannelPool, get_default_pool
from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc


class SchedulerClient:
    """调度客户端

    提供对 gRPC SchedulerService 的访问。
    """

    def __init__(
        self,
        target: str,
        timeout: float = 30.0,
        pool: Optional[GrpcChannelPool] = None,
    ):
        """
        Args:
            target: 服务器地址 (host:port)
            timeout: 默认超时时间
            pool: 连接池（可选）
        """
        self.target = target
        self.timeout = timeout
        self._pool = pool or get_default_pool()
        self._channel = self._pool.get_channel(target, timeout)
        self._stub = quantumflow_pb2_grpc.SchedulerServiceStub(self._channel.get_channel())

    def submit_request(
        self,
        request: quantumflow_pb2.SchedulingRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.SchedulingResponse:
        """提交调度请求

        Args:
            request: SchedulingRequest 消息
            timeout: 超时时间（可选）

        Returns:
            SchedulingResponse 消息
        """
        timeout = timeout or self.timeout
        return self._stub.SubmitRequest(request, timeout=timeout)

    def cancel(
        self,
        request: quantumflow_pb2.CancelRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.CancelResponse:
        """取消请求

        Args:
            request: CancelRequest 消息
            timeout: 超时时间（可选）

        Returns:
            CancelResponse 消息
        """
        timeout = timeout or self.timeout
        return self._stub.Cancel(request, timeout=timeout)

    def get_status(
        self,
        request: quantumflow_pb2.GetSchedulingStatusRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.GetSchedulingStatusResponse:
        """获取调度状态

        Args:
            request: GetSchedulingStatusRequest 消息
            timeout: 超时时间（可选）

        Returns:
            GetSchedulingStatusResponse 消息
        """
        timeout = timeout or self.timeout
        return self._stub.GetStatus(request, timeout=timeout)

    def cancel_stream(
        self,
        request_iterator: Iterator[quantumflow_pb2.CancelRequest],
        timeout: Optional[float] = None,
    ) -> Iterator[quantumflow_pb2.CancelResponse]:
        """取消请求（流式）

        Args:
            request_iterator: CancelRequest 迭代器
            timeout: 超时时间（可选）

        Returns:
            CancelResponse 消息迭代器
        """
        timeout = timeout or self.timeout
        return self._stub.CancelRequestStream(request_iterator, timeout=timeout)

    def close(self) -> None:
        """关闭客户端"""
        self._pool.remove_channel(self.target)
