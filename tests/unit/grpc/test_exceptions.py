"""gRPC 异常测试

严格测试所有异常类的:
1. 错误码映射正确性
2. 错误消息格式
3. 属性访问
4. 异常比较和哈希
5. 从 RpcError 转换
"""

import pytest

from quantumflow.grpc.exceptions import (
    AlreadyExistsError,
    AuthenticationError,
    DeadlineExceededError,
    GrpcQuantumFlowError,
    InternalServerError,
    InvalidRequestError,
    ModelNotLoadedError,
    NodeNotFoundError,
    PermissionDeniedError,
    ResourceUnavailableError,
    SchedulingError,
    ServiceUnavailableError,
    get_status_code_from_exception,
)


class TestGrpcQuantumFlowError:
    """GrpcQuantumFlowError 基类测试"""

    def test_error_has_code_and_message(self):
        """异常包含 code 和 message"""
        import grpc

        error = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        assert error.code == grpc.StatusCode.NOT_FOUND
        assert error.message == "Node not found"

    def test_error_equality_with_non_grpc_error(self):
        """与普通异常比较返回 False"""
        import grpc

        error = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        assert error != ValueError("Not found")
        assert error != "Not found"
        assert error != 123
        assert error != None

    def test_from_rpc_error_with_callable_code_and_details(self):
        """from_rpc_error 处理可调用的 code() 和 details()"""
        import grpc
        from unittest.mock import MagicMock

        mock_rpc = MagicMock()
        mock_rpc.code = MagicMock(return_value=grpc.StatusCode.NOT_FOUND)
        mock_rpc.details = MagicMock(return_value="Node not found")

        error = GrpcQuantumFlowError.from_rpc_error(mock_rpc)

        assert error.code == grpc.StatusCode.NOT_FOUND
        assert error.message == "Node not found"

    def test_from_rpc_error_with_direct_code(self):
        """from_rpc_error 处理直接的 code 属性"""
        import grpc
        from unittest.mock import MagicMock

        mock_rpc = MagicMock()
        mock_rpc.code = grpc.StatusCode.INTERNAL
        mock_rpc.details = "Internal error"

        error = GrpcQuantumFlowError.from_rpc_error(mock_rpc)

        assert error.code == grpc.StatusCode.INTERNAL

    def test_error_string_format(self):
        """错误字符串格式正确"""
        import grpc

        error = GrpcQuantumFlowError(grpc.StatusCode.INVALID_ARGUMENT, "Bad request")
        error_str = str(error)
        assert "[INVALID_ARGUMENT]" in error_str
        assert "Bad request" in error_str

    def test_error_repr(self):
        """repr 格式正确"""
        import grpc

        error = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        repr_str = repr(error)
        assert "GrpcQuantumFlowError" in repr_str
        assert "NOT_FOUND" in repr_str
        assert "Node not found" in repr_str

    def test_error_equality(self):
        """相同 code/message 的错误相等"""
        import grpc

        error1 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        error2 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        assert error1 == error2

    def test_error_inequality_different_code(self):
        """不同 code 的错误不相等"""
        import grpc

        error1 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        error2 = GrpcQuantumFlowError(grpc.StatusCode.INTERNAL, "Node not found")
        assert error1 != error2

    def test_error_inequality_different_message(self):
        """不同 message 的错误不相等"""
        import grpc

        error1 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node 1")
        error2 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node 2")
        assert error1 != error2

    def test_error_hash(self):
        """错误可哈希（用于 set/dict）"""
        import grpc

        error1 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        error2 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")

        error_set = {error1, error2}
        assert len(error_set) == 1

    def test_error_chain_with_cause(self):
        """异常可以链接"""
        import grpc

        cause = ValueError("original error")
        error = GrpcQuantumFlowError(
            grpc.StatusCode.INTERNAL, "wrapped", cause=cause
        )
        assert error.__cause__ == cause

    def test_error_without_cause(self):
        """无 cause 的异常"""
        import grpc

        error = GrpcQuantumFlowError(grpc.StatusCode.INTERNAL, "error")
        assert error.cause is None


class TestNodeNotFoundError:
    """NodeNotFoundError 测试"""

    def test_error_message_contains_node_id(self):
        """错误消息包含 node_id"""
        error = NodeNotFoundError(node_id="node-123")
        assert "node-123" in str(error)
        assert error.node_id == "node-123"

    def test_error_code_is_not_found(self):
        """错误码是 NOT_FOUND"""
        import grpc

        error = NodeNotFoundError(node_id="node-123")
        assert error.code == grpc.StatusCode.NOT_FOUND

    def test_error_equality(self):
        """相同 node_id 的错误相等"""
        error1 = NodeNotFoundError(node_id="node-123")
        error2 = NodeNotFoundError(node_id="node-123")
        assert error1 == error2


class TestModelNotLoadedError:
    """ModelNotLoadedError 测试"""

    def test_error_message_contains_model_name(self):
        """错误消息包含 model_name"""
        error = ModelNotLoadedError(model_name="llama-2-70b")
        assert "llama-2-70b" in str(error)
        assert error.model_name == "llama-2-70b"

    def test_error_code_is_not_found(self):
        """错误码是 NOT_FOUND"""
        import grpc

        error = ModelNotLoadedError(model_name="llama-2-70b")
        assert error.code == grpc.StatusCode.NOT_FOUND


class TestSchedulingError:
    """SchedulingError 测试"""

    def test_error_with_reason(self):
        """包含调度失败原因"""
        error = SchedulingError(reason="No available GPU with enough memory")
        assert "No available GPU with enough memory" in str(error)

    def test_error_with_resource_details(self):
        """包含资源详情"""
        error = SchedulingError(
            reason="Insufficient memory",
            requested="80GB",
            available="40GB",
        )
        error_str = str(error)
        assert "80GB" in error_str
        assert "40GB" in error_str

    def test_error_code_is_unavailable(self):
        """错误码是 UNAVAILABLE"""
        import grpc

        error = SchedulingError(reason="No resources")
        assert error.code == grpc.StatusCode.UNAVAILABLE

    def test_error_attributes(self):
        """属性正确访问"""
        error = SchedulingError(
            reason="test",
            requested="10GB",
            available="5GB",
        )
        assert error.reason == "test"
        assert error.requested == "10GB"
        assert error.available == "5GB"


class TestResourceUnavailableError:
    """ResourceUnavailableError 测试"""

    def test_error_with_resource_type(self):
        """包含资源类型信息"""
        error = ResourceUnavailableError(resource="GPU", required=8, available=0)
        assert "GPU" in str(error)
        assert error.resource == "GPU"

    def test_error_code_is_resource_exhausted(self):
        """错误码是 RESOURCE_EXHAUSTED"""
        import grpc

        error = ResourceUnavailableError(resource="GPU memory", required=80, available=0)
        assert error.code == grpc.StatusCode.RESOURCE_EXHAUSTED

    def test_error_message_format(self):
        """消息格式正确"""
        error = ResourceUnavailableError(resource="GPU", required=8, available=2)
        error_str = str(error)
        assert "required 8" in error_str
        assert "available 2" in error_str


class TestAuthenticationError:
    """AuthenticationError 测试"""

    def test_default_message(self):
        """默认错误消息"""
        error = AuthenticationError()
        assert "authentication" in str(error).lower()

    def test_custom_message(self):
        """自定义错误消息"""
        error = AuthenticationError(reason="Token expired")
        assert "Token expired" in str(error)

    def test_error_code_is_unauthenticated(self):
        """错误码是 UNAUTHENTICATED"""
        import grpc

        error = AuthenticationError()
        assert error.code == grpc.StatusCode.UNAUTHENTICATED


class TestPermissionDeniedError:
    """PermissionDeniedError 测试"""

    def test_error_message_contains_operation(self):
        """错误消息包含操作名"""
        error = PermissionDeniedError(operation="delete_model", required_permission="admin")
        assert "delete_model" in str(error)

    def test_error_message_contains_permission(self):
        """错误消息包含权限名"""
        error = PermissionDeniedError(operation="delete_model", required_permission="admin")
        assert "admin" in str(error)

    def test_error_code_is_permission_denied(self):
        """错误码是 PERMISSION_DENIED"""
        import grpc

        error = PermissionDeniedError(operation="test", required_permission="read")
        assert error.code == grpc.StatusCode.PERMISSION_DENIED

    def test_attributes_accessible(self):
        """属性可访问"""
        error = PermissionDeniedError(operation="test", required_permission="admin")
        assert error.operation == "test"
        assert error.required_permission == "admin"


class TestInvalidRequestError:
    """InvalidRequestError 测试"""

    def test_error_message_contains_field(self):
        """错误消息包含字段名"""
        error = InvalidRequestError(field="temperature", reason="out of range", value=5.0)
        assert "temperature" in str(error)
        assert error.field == "temperature"

    def test_error_message_contains_reason(self):
        """错误消息包含原因"""
        error = InvalidRequestError(field="temperature", reason="out of range")
        assert "out of range" in str(error)

    def test_error_message_contains_value(self):
        """错误消息包含值"""
        error = InvalidRequestError(field="temperature", reason="out of range", value=5.0)
        assert "5.0" in str(error)

    def test_error_code_is_invalid_argument(self):
        """错误码是 INVALID_ARGUMENT"""
        import grpc

        error = InvalidRequestError(field="test", reason="invalid")
        assert error.code == grpc.StatusCode.INVALID_ARGUMENT


class TestDeadlineExceededError:
    """DeadlineExceededError 测试"""

    def test_error_message_contains_operation(self):
        """错误消息包含操作名"""
        error = DeadlineExceededError(operation="model_load", timeout_seconds=300)
        assert "model_load" in str(error)

    def test_error_message_contains_timeout(self):
        """错误消息包含超时时间"""
        error = DeadlineExceededError(operation="test", timeout_seconds=300)
        assert "300" in str(error)

    def test_error_code_is_deadline_exceeded(self):
        """错误码是 DEADLINE_EXCEEDED"""
        import grpc

        error = DeadlineExceededError(operation="test", timeout_seconds=60)
        assert error.code == grpc.StatusCode.DEADLINE_EXCEEDED


class TestServiceUnavailableError:
    """ServiceUnavailableError 测试"""

    def test_error_message_contains_service(self):
        """错误消息包含服务名"""
        error = ServiceUnavailableError(service="inference", reason="maintenance")
        assert "inference" in str(error)
        assert error.service == "inference"

    def test_error_message_contains_reason(self):
        """错误消息包含原因"""
        error = ServiceUnavailableError(service="inference", reason="maintenance")
        assert "maintenance" in str(error)

    def test_error_without_reason(self):
        """无原因的错误"""
        error = ServiceUnavailableError(service="inference")
        assert "inference" in str(error)

    def test_error_code_is_unavailable(self):
        """错误码是 UNAVAILABLE"""
        import grpc

        error = ServiceUnavailableError(service="test")
        assert error.code == grpc.StatusCode.UNAVAILABLE


class TestInternalServerError:
    """InternalServerError 测试"""

    def test_error_message_contains_reason(self):
        """错误消息包含原因"""
        error = InternalServerError(reason="database connection failed")
        assert "database connection failed" in str(error)

    def test_error_with_cause(self):
        """包含原始异常"""
        cause = RuntimeError("original")
        error = InternalServerError(reason="wrapped", cause=cause)
        assert error.cause is cause

    def test_error_code_is_internal(self):
        """错误码是 INTERNAL"""
        import grpc

        error = InternalServerError(reason="test")
        assert error.code == grpc.StatusCode.INTERNAL


class TestAlreadyExistsError:
    """AlreadyExistsError 测试"""

    def test_error_message_contains_resource_type(self):
        """错误消息包含资源类型"""
        error = AlreadyExistsError(resource_type="Node", resource_id="worker-001")
        assert "Node" in str(error)
        assert error.resource_type == "Node"

    def test_error_message_contains_resource_id(self):
        """错误消息包含资源 ID"""
        error = AlreadyExistsError(resource_type="Node", resource_id="worker-001")
        assert "worker-001" in str(error)
        assert error.resource_id == "worker-001"

    def test_error_code_is_already_exists(self):
        """错误码是 ALREADY_EXISTS"""
        import grpc

        error = AlreadyExistsError(resource_type="test", resource_id="123")
        assert error.code == grpc.StatusCode.ALREADY_EXISTS


class TestGetStatusCodeFromException:
    """get_status_code_from_exception 测试"""

    def test_from_grpc_quantum_flow_error(self):
        """从 GrpcQuantumFlowError 获取状态码"""
        import grpc

        error = NodeNotFoundError(node_id="test")
        code = get_status_code_from_exception(error)
        assert code == grpc.StatusCode.NOT_FOUND

    def test_from_grpc_rpc_error(self):
        """从 grpc.RpcError 获取状态码"""
        import grpc

        # 创建 mock RpcError
        rpc_error = grpc.RpcError()
        rpc_error._code = grpc.StatusCode.UNAUTHENTICATED
        rpc_error._details = "Invalid token"

        # Mock the methods
        def get_code():
            return rpc_error._code

        def get_details():
            return rpc_error._details

        type(rpc_error).code = property(lambda self: get_code())
        type(rpc_error).details = property(lambda self: get_details())

        code = get_status_code_from_exception(rpc_error)
        assert code == grpc.StatusCode.UNAUTHENTICATED

    def test_from_unknown_exception(self):
        """从未知异常获取默认状态码"""
        import grpc

        error = ValueError("unknown error")
        code = get_status_code_from_exception(error)
        assert code == grpc.StatusCode.INTERNAL


class TestGetStatusCode:
    """_get_status_code 函数测试"""

    def test_get_existing_status_code(self):
        """获取存在的状态码"""
        import grpc
        from quantumflow.grpc.exceptions import _get_status_code

        code = _get_status_code("NOT_FOUND")
        assert code == grpc.StatusCode.NOT_FOUND

    def test_get_nonexistent_status_code_returns_internal(self):
        """获取不存在的状态码返回 INTERNAL"""
        import grpc
        from quantumflow.grpc.exceptions import _get_status_code

        code = _get_status_code("NONEXISTENT_CODE")
        assert code == grpc.StatusCode.INTERNAL


class TestExceptionMapping:
    """异常到 gRPC 状态码映射测试"""

    @pytest.mark.parametrize("error_class,expected_status", [
        (NodeNotFoundError, "NOT_FOUND"),
        (ModelNotLoadedError, "NOT_FOUND"),
        (SchedulingError, "UNAVAILABLE"),
        (ResourceUnavailableError, "RESOURCE_EXHAUSTED"),
        (AuthenticationError, "UNAUTHENTICATED"),
        (PermissionDeniedError, "PERMISSION_DENIED"),
        (InvalidRequestError, "INVALID_ARGUMENT"),
        (DeadlineExceededError, "DEADLINE_EXCEEDED"),
        (ServiceUnavailableError, "UNAVAILABLE"),
        (InternalServerError, "INTERNAL"),
        (AlreadyExistsError, "ALREADY_EXISTS"),
    ])
    def test_exception_maps_to_correct_status(self, error_class, expected_status):
        """每种异常对应正确的 gRPC 状态码"""
        import grpc

        if error_class == NodeNotFoundError:
            error = error_class(node_id="test")
        elif error_class == ModelNotLoadedError:
            error = error_class(model_name="test")
        elif error_class == SchedulingError:
            error = error_class(reason="test")
        elif error_class == ResourceUnavailableError:
            error = error_class(resource="test", required=1, available=0)
        elif error_class == AuthenticationError:
            error = error_class()
        elif error_class == PermissionDeniedError:
            error = error_class(operation="test", required_permission="read")
        elif error_class == InvalidRequestError:
            error = error_class(field="test", reason="invalid")
        elif error_class == DeadlineExceededError:
            error = error_class(operation="test", timeout_seconds=60)
        elif error_class == ServiceUnavailableError:
            error = error_class(service="test")
        elif error_class == InternalServerError:
            error = error_class(reason="test")
        elif error_class == AlreadyExistsError:
            error = error_class(resource_type="test", resource_id="123")

        expected = getattr(grpc.StatusCode, expected_status)
        assert error.code == expected
