"""Worker gRPC Client

Worker 节点使用的 gRPC 客户端，用于连接 Controller。
"""

import threading
import time
from typing import Iterator, Optional

import grpc

from quantumflow.grpc.channels.pool import GrpcChannelPool, get_default_pool
from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc


class WorkerGrpcClient:
    """Worker gRPC 客户端

    用于 Worker 节点连接 Controller，执行：
    - 心跳发送
    - 模型加载/卸载通知
    - 推理结果上报
    """

    def __init__(
        self,
        controller_endpoint: str,
        node_id: str,
        timeout: float = 30.0,
        pool: Optional[GrpcChannelPool] = None,
    ):
        """
        Args:
            controller_endpoint: Controller 地址 (host:port)
            node_id: 当前节点 ID
            timeout: 超时时间
            pool: 连接池（可选）
        """
        self.controller_endpoint = controller_endpoint
        self.node_id = node_id
        self.timeout = timeout
        self._pool = pool or get_default_pool()
        self._channel = self._pool.get_channel(controller_endpoint, timeout)

        # 创建 Stub
        self._cluster_stub = quantumflow_pb2_grpc.ClusterServiceStub(self._channel.get_channel())
        self._model_stub = quantumflow_pb2_grpc.ModelManagementServiceStub(self._channel.get_channel())
        self._scheduler_stub = quantumflow_pb2_grpc.SchedulerServiceStub(self._channel.get_channel())

        # 状态
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def send_heartbeat(
        self,
        resources: quantumflow_pb2.NodeResources,
        pending_tasks: Optional[list] = None,
    ) -> quantumflow_pb2.HeartbeatResponse:
        """发送心跳

        Args:
            resources: 节点资源信息
            pending_tasks: 待处理任务列表

        Returns:
            HeartbeatResponse 消息
        """
        request = quantumflow_pb2.HeartbeatRequest(
            node_id=self.node_id,
            resources=resources,
        )

        try:
            response = self._cluster_stub.Heartbeat(request, timeout=self.timeout)
            return response
        except grpc.RpcError as e:
            raise GrpcWorkerError(
                code=e.code(),
                message=f"Heartbeat failed: {e.details()}",
            )

    def register_node(
        self,
        host: str,
        port: int,
        gpus: list = None,
        capabilities: dict = None,
    ) -> quantumflow_pb2.RegisterNodeResponse:
        """注册节点

        Args:
            host: 节点地址
            port: 端口
            gpus: GPU 列表
            capabilities: 能力映射

        Returns:
            RegisterNodeResponse 消息
        """
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id=self.node_id,
            host=host,
            port=port,
            gpus=gpus or [],
            capabilities=capabilities or {},
        )

        try:
            response = self._cluster_stub.RegisterNode(request, timeout=self.timeout)
            return response
        except grpc.RpcError as e:
            raise GrpcWorkerError(
                code=e.code(),
                message=f"Node registration failed: {e.details()}",
            )

    def deregister_node(self, reason: str = "") -> quantumflow_pb2.DeregisterNodeResponse:
        """注销节点

        Args:
            reason: 注销原因

        Returns:
            DeregisterNodeResponse 消息
        """
        request = quantumflow_pb2.DeregisterNodeRequest(
            node_id=self.node_id,
            reason=reason,
        )

        try:
            response = self._cluster_stub.DeregisterNode(request, timeout=self.timeout)
            return response
        except grpc.RpcError as e:
            raise GrpcWorkerError(
                code=e.code(),
                message=f"Node deregistration failed: {e.details()}",
            )

    def report_model_loaded(
        self,
        model_name: str,
        backend: str,
        memory_used: int,
    ) -> None:
        """报告模型已加载

        Args:
            model_name: 模型名称
            backend: 后端类型
            memory_used: 已使用显存（字节）
        """
        # 这个功能可以通过心跳附带，不需要单独调用
        pass

    def report_model_unloaded(self, model_name: str) -> None:
        """报告模型已卸载

        Args:
            model_name: 模型名称
        """
        pass

    def fetch_tasks(self) -> list:
        """获取待处理任务

        Returns:
            任务列表
        """
        # 通过心跳响应返回待处理任务
        return []

    def start_heartbeat_loop(self, interval_seconds: float = 5.0) -> None:
        """启动心跳循环

        Args:
            interval_seconds: 心跳间隔（秒）
        """
        with self._lock:
            if self._running:
                return

            self._running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(interval_seconds,),
                daemon=True,
            )
            self._heartbeat_thread.start()

    def stop_heartbeat_loop(self) -> None:
        """停止心跳循环"""
        with self._lock:
            self._running = False
            if self._heartbeat_thread:
                self._heartbeat_thread.join(timeout=10.0)
                self._heartbeat_thread = None

    def _heartbeat_loop(self, interval_seconds: float) -> None:
        """心跳循环（内部方法）"""
        while self._running:
            try:
                # 构建资源信息
                resources = quantumflow_pb2.NodeResources(
                    node_id=self.node_id,
                    status=quantumflow_pb2.NODE_STATUS_ACTIVE,
                )

                # 发送心跳
                self.send_heartbeat(resources)

            except Exception as e:
                # 记录错误但继续运行
                pass

            time.sleep(interval_seconds)

    def close(self) -> None:
        """关闭客户端"""
        self.stop_heartbeat_loop()
        self._pool.remove_channel(self.controller_endpoint)


class GrpcWorkerError(Exception):
    """gRPC Worker 错误"""

    def __init__(self, code: grpc.StatusCode, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code.name}] {message}")
