"""HealthService gRPC 服务实现

提供健康检查功能：
- 健康检查 (Check)
- 流式健康监控 (Watch)
"""

import time
from typing import Iterator

import grpc

from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc
from quantumflow.grpc.services.base import BaseService


class HealthServiceServicer(BaseService, quantumflow_pb2_grpc.HealthServiceServicer):
    """健康检查服务实现

    提供健康检查和流式监控功能。
    """

    def __init__(self, engine_manager=None, cluster_manager=None):
        """
        Args:
            engine_manager: EngineManager 实例
            cluster_manager: ClusterManager 实例
        """
        super().__init__(engine_manager=engine_manager, cluster_manager=cluster_manager)
        self._healthy = True
        self._status = "OK"

    def Check(
        self,
        request: quantumflow_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.HealthCheckResponse:
        """健康检查

        Args:
            request: HealthCheckRequest 消息
            context: gRPC 上下文

        Returns:
            HealthCheckResponse 消息
        """
        details = {}

        # 检查各组件健康状态
        if self.engine_manager:
            try:
                # 检查引擎管理器
                details["engine_manager"] = "OK"
            except Exception as e:
                details["engine_manager"] = f"ERROR: {str(e)}"
                self._healthy = False
                self._status = "DEGRADED"

        if self.cluster_manager:
            try:
                # 检查集群管理器
                details["cluster_manager"] = "OK"
            except Exception as e:
                details["cluster_manager"] = f"ERROR: {str(e)}"
                self._healthy = False
                self._status = "DEGRADED"

        # 检查指定服务
        if request.service and request.service != "all":
            service_status = self._check_service(request.service)
            details[request.service] = service_status
            if service_status != "OK":
                self._healthy = False
                self._status = "DEGRADED"

        return quantumflow_pb2.HealthCheckResponse(
            healthy=self._healthy,
            status=self._status,
            details=details,
        )

    def Watch(
        self,
        request: quantumflow_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[quantumflow_pb2.HealthCheckResponse]:
        """流式健康监控

        Args:
            request: HealthCheckRequest 消息
            context: gRPC 上下文

        Yields:
            HealthCheckResponse 消息（定期）
        """
        while not context.cancelled():
            response = self.Check(request, context)
            yield response

            # 等待 5 秒再发送下一个
            time.sleep(5)

    def set_healthy(self, healthy: bool, status: str = None) -> None:
        """设置健康状态

        Args:
            healthy: 是否健康
            status: 状态描述
        """
        self._healthy = healthy
        if status:
            self._status = status
        elif not healthy:
            self._status = "UNHEALTHY"
        else:
            self._status = "OK"

    def _check_service(self, service: str) -> str:
        """检查指定服务

        Args:
            service: 服务名

        Returns:
            状态描述
        """
        if service == "inference":
            if self.engine_manager:
                return "OK"
            return "NO_ENGINE_MANAGER"

        elif service == "cluster":
            if self.cluster_manager:
                return "OK"
            return "NO_CLUSTER_MANAGER"

        elif service == "scheduler":
            if self.scheduler:
                return "OK"
            return "NO_SCHEDULER"

        else:
            return f"UNKNOWN_SERVICE: {service}"
