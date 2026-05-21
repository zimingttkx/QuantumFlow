"""ClusterService gRPC 服务实现

提供集群管理功能：
- 节点注册 (RegisterNode)
- 节点注销 (DeregisterNode)
- 心跳 (Heartbeat)
- 节点列表 (ListNodes)
- 节点资源更新 (UpdateNodeResources)
"""

import time
from typing import Iterator, List

import grpc

from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc
from quantumflow.grpc.exceptions import (
    InvalidRequestError,
    NodeNotFoundError,
    AlreadyExistsError,
)
from quantumflow.grpc.services.base import BaseService


class ClusterServiceServicer(BaseService, quantumflow_pb2_grpc.ClusterServiceServicer):
    """集群管理服务实现

    提供节点注册、注销、心跳和列表功能。
    """

    def __init__(self, cluster_manager=None):
        """
        Args:
            cluster_manager: ClusterManager 实例
        """
        super().__init__(cluster_manager=cluster_manager)

    def RegisterNode(
        self,
        request: quantumflow_pb2.RegisterNodeRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.RegisterNodeResponse:
        """节点注册

        Args:
            request: RegisterNodeRequest 消息
            context: gRPC 上下文

        Returns:
            RegisterNodeResponse 消息
        """
        # 验证请求
        if not request.node_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "node_id cannot be empty")

        if not request.host:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "host cannot be empty")

        if request.port <= 0 or request.port > 65535:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "port must be in valid range (1-65535)")

        try:
            if self.cluster_manager:
                # 实际注册到集群管理器
                node_id = self.cluster_manager.register_node(
                    node_id=request.node_id,
                    host=request.host,
                    port=request.port,
                    gpus=self._convert_gpus(request.gpus),
                    capabilities=dict(request.capabilities) if request.capabilities else {},
                )
            else:
                # 模拟注册
                node_id = request.node_id

            return quantumflow_pb2.RegisterNodeResponse(
                success=True,
                message="Node registered successfully",
                assigned_id=node_id,
            )

        except AlreadyExistsError as e:
            return quantumflow_pb2.RegisterNodeResponse(
                success=False,
                message=e.message,
            )

        except Exception as e:
            return quantumflow_pb2.RegisterNodeResponse(
                success=False,
                message=f"Registration failed: {str(e)}",
            )

    def DeregisterNode(
        self,
        request: quantumflow_pb2.DeregisterNodeRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.DeregisterNodeResponse:
        """节点注销

        Args:
            request: DeregisterNodeRequest 消息
            context: gRPC 上下文

        Returns:
            DeregisterNodeResponse 消息
        """
        if not request.node_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "node_id cannot be empty")

        try:
            if self.cluster_manager:
                self.cluster_manager.deregister_node(
                    node_id=request.node_id,
                    reason=request.reason,
                )

            return quantumflow_pb2.DeregisterNodeResponse(
                success=True,
                message="Node deregistered successfully",
            )

        except NodeNotFoundError as e:
            return quantumflow_pb2.DeregisterNodeResponse(
                success=False,
                message=e.message,
            )

        except Exception as e:
            return quantumflow_pb2.DeregisterNodeResponse(
                success=False,
                message=f"Deregistration failed: {str(e)}",
            )

    def Heartbeat(
        self,
        request: quantumflow_pb2.HeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.HeartbeatResponse:
        """心跳

        Args:
            request: HeartbeatRequest 消息
            context: gRPC 上下文

        Returns:
            HeartbeatResponse 消息
        """
        if not request.node_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "node_id cannot be empty")

        try:
            pending_tasks: List[str] = []

            if self.cluster_manager:
                # 更新心跳
                self.cluster_manager.update_heartbeat(
                    node_id=request.node_id,
                    resources=self._convert_resources(request.resources) if request.HasField("resources") else None,
                )
                # 获取待处理任务
                pending_tasks = self.cluster_manager.get_pending_tasks(request.node_id) or []

            return quantumflow_pb2.HeartbeatResponse(
                success=True,
                server_time=int(time.time()),
                pending_tasks=pending_tasks,
            )

        except NodeNotFoundError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Node '{request.node_id}' not found")

        except Exception as e:
            return quantumflow_pb2.HeartbeatResponse(
                success=False,
                server_time=int(time.time()),
                pending_tasks=[],
            )

    def ListNodes(
        self,
        request: quantumflow_pb2.ListNodesRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.ListNodesResponse:
        """列出节点

        Args:
            request: ListNodesRequest 消息
            context: gRPC 上下文

        Returns:
            ListNodesResponse 消息
        """
        try:
            nodes: List[quantumflow_pb2.NodeResources] = []

            if self.cluster_manager:
                # 获取节点列表
                raw_nodes = self.cluster_manager.list_nodes(
                    status=request.filter_status if request.filter_status != quantumflow_pb2.NODE_STATUS_UNSPECIFIED else None,
                    model=request.filter_model or None,
                )

                for node in raw_nodes:
                    nodes.append(self._convert_to_node_resources(node))
            else:
                # 模拟节点列表
                if request.filter_status == quantumflow_pb2.NODE_STATUS_UNSPECIFIED or request.filter_status == quantumflow_pb2.NODE_STATUS_ACTIVE:
                    nodes.append(quantumflow_pb2.NodeResources(
                        node_id="worker-001",
                        host="192.168.1.100",
                        port=8001,
                        status=quantumflow_pb2.NODE_STATUS_ACTIVE,
                    ))

            return quantumflow_pb2.ListNodesResponse(nodes=nodes)

        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Failed to list nodes: {str(e)}")

    def UpdateNodeResources(
        self,
        request_iterator: Iterator[quantumflow_pb2.NodeResources],
        context: grpc.ServicerContext,
    ) -> Iterator[quantumflow_pb2.NodeResources]:
        """节点资源更新（双向流式）

        Args:
            request_iterator: NodeResources 迭代器
            context: gRPC 上下文

        Yields:
            NodeResources 消息
        """
        try:
            for resources in request_iterator:
                if not resources.node_id:
                    continue

                # 更新节点资源
                if self.cluster_manager:
                    self.cluster_manager.update_node_resources(resources)

                # 返回更新后的资源（可以包含调度决策等）
                yield resources

        except Exception as e:
            # 流式处理中发生错误
            pass

    def _convert_gpus(self, gpus: List[quantumflow_pb2.GPUInfo]) -> List[dict]:
        """转换 GPU 信息

        Args:
            gpus: GPU 信息列表

        Returns:
            dict 列表
        """
        result = []
        for gpu in gpus:
            result.append({
                "index": gpu.index,
                "name": gpu.name,
                "memory": {
                    "total": gpu.memory.total_bytes if gpu.HasField("memory") else 0,
                    "available": gpu.memory.available_bytes if gpu.HasField("memory") else 0,
                    "used": gpu.memory.used_bytes if gpu.HasField("memory") else 0,
                } if gpu.HasField("memory") else None,
                "utilization": gpu.utilization,
            })
        return result

    def _convert_resources(self, resources: quantumflow_pb2.NodeResources) -> dict:
        """转换资源信息

        Args:
            resources: NodeResources 消息

        Returns:
            dict
        """
        return {
            "node_id": resources.node_id,
            "host": resources.host,
            "port": resources.port,
            "status": resources.status,
            "gpus": self._convert_gpus(resources.gpus),
        }

    def _convert_to_node_resources(self, node: dict) -> quantumflow_pb2.NodeResources:
        """将 dict 转换为 NodeResources 消息

        Args:
            node: 节点信息 dict

        Returns:
            NodeResources 消息
        """
        gpus = []
        if "gpus" in node:
            for gpu in node["gpus"]:
                gpu_msg = quantumflow_pb2.GPUInfo(
                    index=gpu.get("index", 0),
                    name=gpu.get("name", ""),
                    utilization=gpu.get("utilization", 0.0),
                )
                if "memory" in gpu and gpu["memory"]:
                    gpu_msg.memory.CopyFrom(quantumflow_pb2.GPUMemory(
                        total_bytes=gpu["memory"].get("total", 0),
                        available_bytes=gpu["memory"].get("available", 0),
                        used_bytes=gpu["memory"].get("used", 0),
                    ))
                gpus.append(gpu_msg)

        return quantumflow_pb2.NodeResources(
            node_id=node.get("node_id", ""),
            host=node.get("host", ""),
            port=node.get("port", 0),
            status=node.get("status", quantumflow_pb2.NODE_STATUS_UNSPECIFIED),
            gpus=gpus,
        )
