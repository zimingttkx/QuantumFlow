"""gRPC 服务基类测试

测试 BaseService 的公共功能。
"""

import pytest
import grpc

from quantumflow.grpc.services.base import BaseService
from quantumflow.grpc.exceptions import (
    GrpcQuantumFlowError,
    InvalidRequestError,
    InternalServerError,
)


class TestBaseServiceValidation:
    """请求验证测试"""

    def test_validate_request_id_empty(self):
        """验证空 request_id 抛出异常"""
        service = BaseService()

        with pytest.raises(InvalidRequestError) as exc_info:
            service._validate_request_id("")

        assert "request_id" in str(exc_info.value)

    def test_validate_request_id_valid(self):
        """验证有效 request_id"""
        service = BaseService()

        # 不应该抛出异常
        service._validate_request_id("req-123")

    def test_validate_model_name_empty(self):
        """验证空 model_name 抛出异常"""
        service = BaseService()

        with pytest.raises(InvalidRequestError) as exc_info:
            service._validate_model_name("")

        assert "model_name" in str(exc_info.value)

    def test_validate_model_name_valid(self):
        """验证有效 model_name"""
        service = BaseService()

        # 不应该抛出异常
        service._validate_model_name("gpt-3")

    def test_validate_node_id_empty(self):
        """验证空 node_id 抛出异常"""
        service = BaseService()

        with pytest.raises(InvalidRequestError) as exc_info:
            service._validate_node_id("")

        assert "node_id" in str(exc_info.value)

    def test_validate_node_id_valid(self):
        """验证有效 node_id"""
        service = BaseService()

        # 不应该抛出异常
        service._validate_node_id("node-123")


class TestBaseServiceExceptionHandling:
    """异常处理测试"""

    def test_handle_grpc_quantum_flow_error(self):
        """处理 GrpcQuantumFlowError"""
        service = BaseService()

        error = NodeNotFoundError("node-123")
        result = service._handle_exception(error)

        assert isinstance(result, grpc.RpcError)

    def test_handle_value_error(self):
        """处理 ValueError"""
        service = BaseService()

        error = ValueError("Invalid value")
        result = service._handle_exception(error)

        assert isinstance(result, grpc.RpcError)

    def test_handle_unknown_error(self):
        """处理未知异常"""
        service = BaseService()

        error = RuntimeError("Unknown error")
        result = service._handle_exception(error)

        assert isinstance(result, grpc.RpcError)

    def test_create_rpc_error(self):
        """创建 RpcError"""
        service = BaseService()

        error = service._create_rpc_error(
            grpc.StatusCode.NOT_FOUND,
            "Not found"
        )

        assert isinstance(error, grpc.RpcError)


class NodeNotFoundError(GrpcQuantumFlowError):
    """测试用异常"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(
            code=grpc.StatusCode.NOT_FOUND,
            message=f"Node '{node_id}' not found",
        )
