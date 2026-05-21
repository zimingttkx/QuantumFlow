"""gRPC Server 封装

提供 gRPC 服务器的统一管理：
- 服务注册
- 拦截器配置
- 启动/停止
- 健康检查集成
"""

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set

import grpc

from quantumflow.grpc import interceptors
from quantumflow.grpc.services import (
    InferenceServiceServicer,
    ClusterServiceServicer,
    SchedulerServiceServicer,
    ModelManagementServiceServicer,
    HealthServiceServicer,
    MetricsServiceServicer,
)


class GrpcServer:
    """gRPC 服务器封装

    提供统一的服务注册、拦截器配置和生命周期管理。
    """

    def __init__(
        self,
        port: int = 50051,
        max_workers: int = 10,
        reflection_enabled: bool = True,
    ):
        """
        Args:
            port: gRPC 服务器端口
            max_workers: 工作线程数
            reflection_enabled: 是否启用 gRPC Reflection（用于 grpcurl 等工具）
        """
        self.port = port
        self.max_workers = max_workers
        self.reflection_enabled = reflection_enabled
        self._server: Optional[grpc.Server] = None
        self._servicers: Dict[str, any] = {}
        self._interceptors: List[grpc.ServerInterceptor] = []
        self._lock = threading.Lock()
        self._started: bool = False

        # 创建服务器
        self._create_server()

    def _create_server(self) -> None:
        """创建 gRPC 服务器"""
        self._server = grpc.server(
            thread_pool=ThreadPoolExecutor(max_workers=self.max_workers),
            interceptors=self._interceptors if self._interceptors else None,
        )

    def add_inference_service(
        self,
        engine_manager=None,
        cluster_manager=None,
    ) -> None:
        """添加推理服务

        Args:
            engine_manager: 引擎管理器
            cluster_manager: 集群管理器
        """
        servicer = InferenceServiceServicer(
            engine_manager=engine_manager,
            cluster_manager=cluster_manager,
        )
        self._servicers["inference"] = servicer

    def add_cluster_service(self, cluster_manager=None) -> None:
        """添加集群管理服务

        Args:
            cluster_manager: 集群管理器
        """
        servicer = ClusterServiceServicer(cluster_manager=cluster_manager)
        self._servicers["cluster"] = servicer

    def add_scheduler_service(self, scheduler=None, engine_manager=None) -> None:
        """添加调度服务

        Args:
            scheduler: 调度器
            engine_manager: 引擎管理器
        """
        servicer = SchedulerServiceServicer(
            scheduler=scheduler,
            engine_manager=engine_manager,
        )
        self._servicers["scheduler"] = servicer

    def add_model_management_service(self, engine_manager=None) -> None:
        """添加模型管理服务

        Args:
            engine_manager: 引擎管理器
        """
        servicer = ModelManagementServiceServicer(engine_manager=engine_manager)
        self._servicers["model_management"] = servicer

    def add_health_service(
        self,
        engine_manager=None,
        cluster_manager=None,
        scheduler=None,
    ) -> None:
        """添加健康检查服务

        Args:
            engine_manager: 引擎管理器
            cluster_manager: 集群管理器
            scheduler: 调度器
        """
        servicer = HealthServiceServicer(
            engine_manager=engine_manager,
            cluster_manager=cluster_manager,
        )
        self._servicers["health"] = servicer

    def add_metrics_service(
        self,
        engine_manager=None,
        cluster_manager=None,
    ) -> None:
        """添加指标服务

        Args:
            engine_manager: 引擎管理器
            cluster_manager: 集群管理器
        """
        servicer = MetricsServiceServicer(
            engine_manager=engine_manager,
            cluster_manager=cluster_manager,
        )
        self._servicers["metrics"] = servicer

    def add_logging_interceptor(self) -> None:
        """添加日志拦截器"""
        interceptor = interceptors.LoggingInterceptor()
        self._interceptors.append(interceptor)

    def add_auth_interceptor(
        self,
        allowed_tokens: Optional[Dict[str, str]] = None,
        bypass_methods: Optional[Set[str]] = None,
    ) -> None:
        """添加认证拦截器

        Args:
            allowed_tokens: token -> user_id 映射
            bypass_methods: 跳过认证的方法集合
        """
        interceptor = interceptors.AuthInterceptor(
            allowed_tokens=allowed_tokens,
            bypass_methods=bypass_methods,
        )
        self._interceptors.append(interceptor)

    def add_rate_limit_interceptor(
        self,
        qps: int = 100,
        burst: int = 200,
        per_method: bool = False,
    ) -> None:
        """添加限流拦截器

        Args:
            qps: 每秒请求数
            burst: 突发容量
            per_method: 是否按方法限流
        """
        interceptor = interceptors.RateLimitInterceptor(
            qps=qps,
            burst=burst,
            per_method=per_method,
        )
        self._interceptors.append(interceptor)

    def add_metrics_interceptor(self) -> None:
        """添加监控拦截器"""
        interceptor = interceptors.MetricsInterceptor()
        self._interceptors.append(interceptor)

    def start(self) -> None:
        """启动 gRPC 服务器"""
        with self._lock:
            if self._started:
                return

            # 如果有拦截器但服务器未创建拦截器，则重新创建服务器
            if self._interceptors and not hasattr(self._server, '_interceptors_added'):
                self._server = grpc.server(
                    thread_pool=ThreadPoolExecutor(max_workers=self.max_workers),
                    interceptors=self._interceptors,
                )

            if self._server is None:
                raise RuntimeError("Server not created")

            # 注册所有服务
            self._register_services()

            # 启用 Reflection（如果启用）
            if self.reflection_enabled:
                try:
                    from grpc_reflection.v1alpha import reflection
                    service_names = [
                        "quantumflow.v1.InferenceService",
                        "quantumflow.v1.ClusterService",
                        "quantumflow.v1.SchedulerService",
                        "quantumflow.v1.ModelManagementService",
                        "quantumflow.v1.HealthService",
                        "quantumflow.v1.MetricsService",
                    ]
                    reflection.enable_server_reflection(service_names, self._server)
                except ImportError:
                    pass  # Reflection 不可用

            # 绑定端口
            self._server.add_insecure_port(f"[::]:{self.port}")

            # 启动服务器
            self._server.start()
            self._started = True
            print(f"gRPC server started on port {self.port}")

            # 注册关闭钩子
            atexit.register(self.stop)

    def _register_services(self) -> None:
        """注册所有服务到服务器"""
        from quantumflow.grpc.generated import (
            quantumflow_pb2_grpc as pb2_grpc,
        )

        if "inference" in self._servicers:
            pb2_grpc.add_InferenceServiceServicer_to_server(
                self._servicers["inference"],
                self._server,
            )

        if "cluster" in self._servicers:
            pb2_grpc.add_ClusterServiceServicer_to_server(
                self._servicers["cluster"],
                self._server,
            )

        if "scheduler" in self._servicers:
            pb2_grpc.add_SchedulerServiceServicer_to_server(
                self._servicers["scheduler"],
                self._server,
            )

        if "model_management" in self._servicers:
            pb2_grpc.add_ModelManagementServiceServicer_to_server(
                self._servicers["model_management"],
                self._server,
            )

        if "health" in self._servicers:
            pb2_grpc.add_HealthServiceServicer_to_server(
                self._servicers["health"],
                self._server,
            )

        if "metrics" in self._servicers:
            pb2_grpc.add_MetricsServiceServicer_to_server(
                self._servicers["metrics"],
                self._server,
            )

    def stop(self, grace: float = 5.0) -> None:
        """停止 gRPC 服务器

        Args:
            grace: 优雅关闭等待时间（秒）
        """
        with self._lock:
            if self._server is not None:
                self._server.stop(grace=grace)
                self._server = None
                print("gRPC server stopped")

    def wait_for_termination(self) -> None:
        """等待服务器终止"""
        if self._server is not None:
            self._server.wait_for_termination()

    def is_started(self) -> bool:
        """检查服务器是否已启动"""
        return self._started

    @property
    def port(self) -> int:
        """获取端口号"""
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        """设置端口号"""
        if value < 1 or value > 65535:
            raise ValueError("Port must be between 1 and 65535")
        self._port = value


class GrpcServerManager:
    """gRPC 服务器管理器

    管理多个 gRPC 服务器实例，支持分布式部署。
    """

    def __init__(self):
        self._servers: Dict[str, GrpcServer] = {}
        self._lock = threading.Lock()

    def create_server(
        self,
        name: str,
        port: int = 50051,
        **kwargs,
    ) -> GrpcServer:
        """创建 gRPC 服务器

        Args:
            name: 服务器名称
            port: 端口号
            **kwargs: 其他参数

        Returns:
            GrpcServer 实例
        """
        with self._lock:
            if name in self._servers:
                raise ValueError(f"Server '{name}' already exists")

            server = GrpcServer(port=port, **kwargs)
            self._servers[name] = server
            return server

    def get_server(self, name: str) -> Optional[GrpcServer]:
        """获取服务器

        Args:
            name: 服务器名称

        Returns:
            GrpcServer 实例或 None
        """
        return self._servers.get(name)

    def start_server(self, name: str) -> None:
        """启动服务器

        Args:
            name: 服务器名称
        """
        server = self.get_server(name)
        if server is None:
            raise ValueError(f"Server '{name}' not found")
        server.start()

    def stop_server(self, name: str, grace: float = 5.0) -> None:
        """停止服务器

        Args:
            name: 服务器名称
            grace: 优雅关闭等待时间
        """
        server = self.get_server(name)
        if server is not None:
            server.stop(grace=grace)

    def stop_all(self, grace: float = 5.0) -> None:
        """停止所有服务器"""
        with self._lock:
            for server in self._servers.values():
                server.stop(grace=grace)
            self._servers.clear()
