"""gRPC InferenceClient

提供推理功能的 gRPC 客户端。
"""

from typing import Iterator, List, Optional

import grpc

from quantumflow.grpc.channels.pool import GrpcChannelPool, get_default_pool
from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc


class InferenceClient:
    """推理客户端

    提供对 gRPC InferenceService 的访问。
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
        self._stub = quantumflow_pb2_grpc.InferenceServiceStub(self._channel.get_channel())

    def inference(
        self,
        request: quantumflow_pb2.InferenceRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.InferenceResponse:
        """同步推理

        Args:
            request: InferenceRequest 消息
            timeout: 超时时间（可选）

        Returns:
            InferenceResponse 消息
        """
        timeout = timeout if timeout is not None else self.timeout
        return self._stub.Inference(request, timeout=timeout)

    def inference_stream(
        self,
        request: quantumflow_pb2.InferenceRequest,
        timeout: Optional[float] = None,
    ) -> Iterator[quantumflow_pb2.InferenceResponse]:
        """流式推理

        Args:
            request: InferenceRequest 消息
            timeout: 超时时间（可选）

        Returns:
            InferenceResponse 消息迭代器
        """
        timeout = timeout if timeout is not None else self.timeout
        return self._stub.InferenceStream(request, timeout=timeout)

    def batch_inference(
        self,
        request: quantumflow_pb2.BatchInferenceRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.BatchInferenceResponse:
        """批量推理

        Args:
            request: BatchInferenceRequest 消息
            timeout: 超时时间（可选）

        Returns:
            BatchInferenceResponse 消息
        """
        timeout = timeout if timeout is not None else self.timeout
        return self._stub.BatchInference(request, timeout=timeout)

    def close(self) -> None:
        """关闭客户端"""
        self._pool.remove_channel(self.target)


class InferenceClientPool:
    """推理客户端池

    管理多个 InferenceClient 实例，用于负载均衡。
    """

    def __init__(self, targets: List[str], timeout: float = 30.0):
        """
        Args:
            targets: 服务器地址列表
            timeout: 默认超时时间
        """
        self.targets = targets
        self.timeout = timeout
        self._clients = [InferenceClient(t, timeout) for t in targets]
        self._current = 0
        self._lock = threading.Lock()

    def get_client(self) -> InferenceClient:
        """获取下一个客户端（轮询）

        Returns:
            InferenceClient 实例
        """
        with self._lock:
            client = self._clients[self._current]
            self._current = (self._current + 1) % len(self._clients)
            return client

    def inference(
        self,
        request: quantumflow_pb2.InferenceRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.InferenceResponse:
        """同步推理（负载均衡）"""
        client = self.get_client()
        return client.inference(request, timeout)

    def inference_stream(
        self,
        request: quantumflow_pb2.InferenceRequest,
        timeout: Optional[float] = None,
    ) -> Iterator[quantumflow_pb2.InferenceResponse]:
        """流式推理（负载均衡）"""
        client = self.get_client()
        return client.inference_stream(request, timeout)

    def batch_inference(
        self,
        request: quantumflow_pb2.BatchInferenceRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.BatchInferenceResponse:
        """批量推理（负载均衡）"""
        client = self.get_client()
        return client.batch_inference(request, timeout)

    def close_all(self) -> None:
        """关闭所有客户端"""
        for client in self._clients:
            client.close()


import threading
