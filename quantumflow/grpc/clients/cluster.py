"""gRPC ClusterClient

提供集群管理功能的 gRPC 客户端。
"""

from typing import Iterator, List, Optional

import grpc

from quantumflow.grpc.channels.pool import GrpcChannelPool, get_default_pool
from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc


class ClusterClient:
    """集群管理客户端

    提供对 gRPC ClusterService 的访问。
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
        self._stub = quantumflow_pb2_grpc.ClusterServiceStub(self._channel.get_channel())

    def register_node(
        self,
        request: quantumflow_pb2.RegisterNodeRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.RegisterNodeResponse:
        """注册节点

        Args:
            request: RegisterNodeRequest 消息
            timeout: 超时时间（可选）

        Returns:
            RegisterNodeResponse 消息
        """
        timeout = timeout or self.timeout
        return self._stub.RegisterNode(request, timeout=timeout)

    def deregister_node(
        self,
        request: quantumflow_pb2.DeregisterNodeRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.DeregisterNodeResponse:
        """注销节点

        Args:
            request: DeregisterNodeRequest 消息
            timeout: 超时时间（可选）

        Returns:
            DeregisterNodeResponse 消息
        """
        timeout = timeout or self.timeout
        return self._stub.DeregisterNode(request, timeout=timeout)

    def heartbeat(
        self,
        request: quantumflow_pb2.HeartbeatRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.HeartbeatResponse:
        """发送心跳

        Args:
            request: HeartbeatRequest 消息
            timeout: 超时时间（可选）

        Returns:
            HeartbeatResponse 消息
        """
        timeout = timeout or self.timeout
        return self._stub.Heartbeat(request, timeout=timeout)

    def list_nodes(
        self,
        request: quantumflow_pb2.ListNodesRequest,
        timeout: Optional[float] = None,
    ) -> quantumflow_pb2.ListNodesResponse:
        """列出节点

        Args:
            request: ListNodesRequest 消息
            timeout: 超时时间（可选）

        Returns:
            ListNodesResponse 消息
        """
        timeout = timeout or self.timeout
        return self._stub.ListNodes(request, timeout=timeout)

    def update_node_resources_stream(
        self,
        request_iterator: Iterator[quantumflow_pb2.NodeResources],
        timeout: Optional[float] = None,
    ) -> Iterator[quantumflow_pb2.NodeResources]:
        """更新节点资源（流式）

        Args:
            request_iterator: NodeResources 迭代器
            timeout: 超时时间（可选）

        Returns:
            NodeResources 消息迭代器
        """
        timeout = timeout or self.timeout
        return self._stub.UpdateNodeResources(request_iterator, timeout=timeout)

    def close(self) -> None:
        """关闭客户端"""
        self._pool.remove_channel(self.target)
