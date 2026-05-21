"""gRPC 异常定义

提供 gRPC 相关的自定义异常类，用于统一处理 gRPC 错误。
"""

from typing import Optional


class GrpcQuantumFlowError(Exception):
    """gRPC 异常基类

    所有 gRPC 相关异常都继承自此类。

    Attributes:
        code: gRPC 状态码
        message: 错误消息
    """

    def __init__(
        self,
        code: "grpc.StatusCode",
        message: str,
        cause: Optional[Exception] = None,
    ):
        self.code = code
        self.message = message
        self.cause = cause
        super().__init__(f"[{code.name}] {message}")
        # 设置异常链
        if cause is not None:
            self.__cause__ = cause

    def __repr__(self) -> str:
        return f"GrpcQuantumFlowError(code={self.code.name}, message='{self.message}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GrpcQuantumFlowError):
            return False
        return self.code == other.code and self.message == other.message

    def __hash__(self) -> int:
        return hash((self.code, self.message))

    @classmethod
    def from_rpc_error(cls, rpc_error: "grpc.RpcError") -> "GrpcQuantumFlowError":
        """从 grpc.RpcError 转换为 GrpcQuantumFlowError

        Args:
            rpc_error: gRPC RpcError 对象

        Returns:
            GrpcQuantumFlowError 实例
        """
        code = rpc_error.code() if callable(rpc_error.code) else rpc_error.code
        details = rpc_error.details() if callable(rpc_error.details) else str(rpc_error)
        return cls(code=code, message=details)


class NodeNotFoundError(GrpcQuantumFlowError):
    """节点未找到异常

    当请求的节点不存在于集群中时抛出。
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(
            code=_get_status_code("NOT_FOUND"),
            message=f"Node '{node_id}' not found in cluster",
        )


class ModelNotLoadedError(GrpcQuantumFlowError):
    """模型未加载异常

    当请求的模型尚未加载到任何节点时抛出。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(
            code=_get_status_code("NOT_FOUND"),
            message=f"Model '{model_name}' is not loaded",
        )


class SchedulingError(GrpcQuantumFlowError):
    """调度失败异常

    当请求无法被调度时抛出（例如资源不足）。
    """

    def __init__(
        self,
        reason: str,
        requested: Optional[str] = None,
        available: Optional[str] = None,
    ):
        self.reason = reason
        self.requested = requested
        self.available = available

        msg = reason
        if requested and available:
            msg = f"{reason} (requested: {requested}, available: {available})"

        super().__init__(
            code=_get_status_code("UNAVAILABLE"),
            message=msg,
        )


class ResourceUnavailableError(GrpcQuantumFlowError):
    """资源不可用异常

    当所需的 GPU 内存、计算资源等不可用时抛出。
    """

    def __init__(
        self,
        resource: str,
        required: Optional[float] = None,
        available: Optional[float] = None,
    ):
        self.resource = resource
        self.required = required
        self.available = available

        msg = f"Resource '{resource}' is unavailable"
        if required is not None and available is not None:
            msg = f"Resource '{resource}' insufficient: required {required}, available {available}"

        super().__init__(
            code=_get_status_code("RESOURCE_EXHAUSTED"),
            message=msg,
        )


class AuthenticationError(GrpcQuantumFlowError):
    """认证失败异常

    当客户端提供的认证信息无效时抛出。
    """

    def __init__(self, reason: str = "Invalid or missing authentication credentials"):
        super().__init__(
            code=_get_status_code("UNAUTHENTICATED"),
            message=reason,
        )


class PermissionDeniedError(GrpcQuantumFlowError):
    """权限不足异常

    当客户端没有执行操作的权限时抛出。
    """

    def __init__(self, operation: str, required_permission: str):
        self.operation = operation
        self.required_permission = required_permission
        super().__init__(
            code=_get_status_code("PERMISSION_DENIED"),
            message=f"Permission denied for operation '{operation}', requires '{required_permission}'",
        )


class InvalidRequestError(GrpcQuantumFlowError):
    """无效请求异常

    当请求参数验证失败时抛出。
    """

    def __init__(self, field: str, reason: str, value: Optional[object] = None):
        self.field = field
        self.reason = reason
        self.value = value

        msg = f"Invalid request field '{field}': {reason}"
        if value is not None:
            msg = f"{msg} (value: {value!r})"

        super().__init__(
            code=_get_status_code("INVALID_ARGUMENT"),
            message=msg,
        )


class DeadlineExceededError(GrpcQuantumFlowError):
    """超时异常

    当操作超过截止时间时抛出。
    """

    def __init__(self, operation: str, timeout_seconds: float):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            code=_get_status_code("DEADLINE_EXCEEDED"),
            message=f"Operation '{operation}' exceeded deadline of {timeout_seconds}s",
        )


class ServiceUnavailableError(GrpcQuantumFlowError):
    """服务不可用异常

    当服务暂时不可用时抛出（例如维护中）。
    """

    def __init__(self, service: str, reason: Optional[str] = None):
        self.service = service
        self.reason = reason

        msg = f"Service '{service}' is unavailable"
        if reason:
            msg = f"{msg}: {reason}"

        super().__init__(
            code=_get_status_code("UNAVAILABLE"),
            message=msg,
        )


class InternalServerError(GrpcQuantumFlowError):
    """内部服务器错误异常

    当服务器端发生未预期的错误时抛出。
    """

    def __init__(self, reason: str, cause: Optional[Exception] = None):
        super().__init__(
            code=_get_status_code("INTERNAL"),
            message=f"Internal server error: {reason}",
            cause=cause,
        )


class AlreadyExistsError(GrpcQuantumFlowError):
    """资源已存在异常

    当尝试创建的资源已存在时抛出。
    """

    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(
            code=_get_status_code("ALREADY_EXISTS"),
            message=f"{resource_type} '{resource_id}' already exists",
        )


def _get_status_code(code_name: str) -> "grpc.StatusCode":
    """获取 gRPC 状态码

    Args:
        code_name: 状态码名称（如 "NOT_FOUND"）

    Returns:
        grpc.StatusCode 枚举值
    """
    import grpc

    try:
        return getattr(grpc.StatusCode, code_name)
    except AttributeError:
        return grpc.StatusCode.INTERNAL


def get_status_code_from_exception(exc: Exception) -> "grpc.StatusCode":
    """从异常获取对应的 gRPC 状态码

    Args:
        exc: 异常对象

    Returns:
        对应的 gRPC 状态码
    """
    import grpc

    if isinstance(exc, GrpcQuantumFlowError):
        return exc.code

    # 处理标准 gRPC 异常
    if isinstance(exc, grpc.RpcError):
        return exc.code() if callable(exc.code) else exc.code

    # 默认返回内部错误
    return grpc.StatusCode.INTERNAL
