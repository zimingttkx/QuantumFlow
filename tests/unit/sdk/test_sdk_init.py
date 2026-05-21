"""QuantumFlow SDK 初始化测试"""
import pytest


def test_import_quantumflow_sdk():
    """验证：可以导入 QuantumFlow SDK"""
    from quantumflow.sdk import QuantumFlowSDK
    assert QuantumFlowSDK is not None


def test_import_exceptions():
    """验证：可以导入 SDK 异常类"""
    from quantumflow.sdk.exceptions import QuantumFlowError, APIError, RateLimitError
    assert QuantumFlowError is not None
    assert APIError is not None
    assert RateLimitError is not None


def test_client_import():
    """验证：可以导入客户端"""
    from quantumflow.sdk import AsyncQuantumFlowClient, SyncQuantumFlowClient
    assert AsyncQuantumFlowClient is not None
    assert SyncQuantumFlowClient is not None
