"""gRPC 服务基类

提供所有 gRPC 服务的公共功能：
- 错误处理
- 请求验证
- 上下文管理
"""

from typing import Any, Optional

import grpc

from quantumflow.grpc.exceptions import (
    GrpcQuantumFlowError,
    InvalidRequestError,
    InternalServerError,
)


class BaseService:
    """gRPC 服务基类

    提供公共的错误处理和验证方法。
    """

    def __init__(self, engine_manager=None, cluster_manager=None, scheduler=None):
        """
        Args:
            engine_manager: 引擎管理器（用于推理和模型管理）
            cluster_manager: 集群管理器（用于节点管理）
            scheduler: 调度器（用于调度服务）
        """
        self.engine_manager = engine_manager
        self.cluster_manager = cluster_manager
        self.scheduler = scheduler

    def _validate_request_id(self, request_id: str, field_name: str = "request_id") -> None:
        """验证请求 ID

        Args:
            request_id: 请求 ID
            field_name: 字段名（用于错误消息）

        Raises:
            InvalidRequestError: 请求 ID 无效
        """
        if not request_id:
            raise InvalidRequestError(field=field_name, reason="Request ID cannot be empty")

    def _validate_model_name(self, model_name: str) -> None:
        """验证模型名

        Args:
            model_name: 模型名称

        Raises:
            InvalidRequestError: 模型名无效
        """
        if not model_name:
            raise InvalidRequestError(field="model_name", reason="Model name cannot be empty")

    def _validate_node_id(self, node_id: str) -> None:
        """验证节点 ID

        Args:
            node_id: 节点 ID

        Raises:
            InvalidRequestError: 节点 ID 无效
        """
        if not node_id:
            raise InvalidRequestError(field="node_id", reason="Node ID cannot be empty")

    def _handle_exception(self, e: Exception) -> GrpcQuantumFlowError:
        """将异常转换为 gRPC 异常

        Args:
            e: 异常对象

        Returns:
            GrpcQuantumFlowError
        """
        if isinstance(e, GrpcQuantumFlowError):
            return e
        elif isinstance(e, ValueError):
            return InvalidRequestError(field="unknown", reason=str(e))
        else:
            # 未知异常，包装为内部错误
            return InternalServerError(reason=str(e), cause=e)

    def _create_rpc_error(
        self, code: grpc.StatusCode, message: str
    ) -> GrpcQuantumFlowError:
        """创建 gRPC 异常

        Args:
            code: 状态码
            message: 错误消息

        Returns:
            GrpcQuantumFlowError
        """
        return GrpcQuantumFlowError(code=code, message=message)
