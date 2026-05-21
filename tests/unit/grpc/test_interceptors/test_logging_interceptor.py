"""gRPC 日志拦截器测试

严格测试日志拦截器的业务逻辑：
- 方法名提取
- Metadata 提取
- 耗时计算
"""

import pytest
import time
import grpc
from unittest.mock import MagicMock


class MockHandlerCallDetails:
    """模拟 HandlerCallDetails"""

    def __init__(
        self,
        method: str = b"/quantumflow.v1.InferenceService/Inference",
        metadata: list = None,
        peer: str = None,
    ):
        self.method = method
        self._metadata = metadata or []
        self._peer = peer

    @property
    def invocation_metadata(self):
        return self._metadata

    @property
    def peer(self):
        if self._peer is None:
            return None
        return self._peer


class TestLoggingInterceptorBasics:
    """日志拦截器基础测试"""

    def test_default_initialization(self):
        """默认初始化"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()

        assert interceptor.log_level == 20  # INFO
        assert interceptor.include_metadata is True
        assert interceptor.include_call_duration is True

    def test_custom_initialization(self):
        """自定义初始化"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor(
            log_level=10,  # DEBUG
            include_metadata=False,
            include_call_duration=False,
        )

        assert interceptor.log_level == 10
        assert interceptor.include_metadata is False
        assert interceptor.include_call_duration is False


class TestLoggingInterceptorMetadata:
    """Metadata 提取测试"""

    def test_extract_metadata_empty(self):
        """提取空 metadata"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        details = MockHandlerCallDetails(metadata=[])

        metadata = interceptor._extract_metadata(details)

        assert metadata == {}

    def test_extract_metadata_with_values(self):
        """提取有值的 metadata"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        metadata_items = [
            MagicMock(key="x-request-id", value="req-123"),
            MagicMock(key="x-user-id", value="user-456"),
        ]
        details = MockHandlerCallDetails(metadata=metadata_items)

        metadata = interceptor._extract_metadata(details)

        assert metadata.get("x-request-id") == "req-123"
        assert metadata.get("x-user-id") == "user-456"

    def test_extract_metadata_handles_exception(self):
        """metadata 提取异常处理"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()

        # 模拟 metadata 抛出异常
        details = MagicMock()
        details.invocation_metadata = MagicMock(side_effect=Exception("test"))

        metadata = interceptor._extract_metadata(details)

        # 应该返回空字典而不是崩溃
        assert metadata == {}


class TestLoggingInterceptorPeerInfo:
    """调用者信息提取测试"""

    def test_peer_info_normal(self):
        """正常 peer 信息"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        details = MockHandlerCallDetails(peer="192.168.1.100:50000")

        peer = interceptor._get_peer_info(details)

        assert peer == "192.168.1.100:50000"

    def test_peer_info_missing(self):
        """缺少 peer 信息"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        details = MockHandlerCallDetails(peer=None)

        peer = interceptor._get_peer_info(details)

        assert peer == "unknown"


class TestLoggingInterceptorSuccessLogging:
    """成功日志测试"""

    def test_log_success_format(self):
        """成功日志格式"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        interceptor._log_success(
            method_name="/TestService/Method",
            peer="127.0.0.1:50000",
            duration=0.05,
            metadata={"x-request-id": "req-123"},
        )

        # 验证日志调用
        mock_logger.log.assert_called_once()

    def test_log_success_without_metadata(self):
        """不带 metadata 的成功日志"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor(include_metadata=False)
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        interceptor._log_success(
            method_name="/TestService/Method",
            peer="127.0.0.1:50000",
            duration=0.01,
            metadata={"x-request-id": "req-123"},
        )

        call_args = mock_logger.log.call_args
        extra = call_args[1]["extra"]

        # metadata 不应该被记录
        assert extra.get("metadata") is None


class TestLoggingInterceptorErrorLogging:
    """错误日志测试"""

    def test_log_error_format(self):
        """错误日志格式"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        # 创建一个模拟的 RpcError
        error = MagicMock()
        error.code = MagicMock(return_value=grpc.StatusCode.NOT_FOUND)
        error.details = MagicMock(return_value="Node not found")

        interceptor._log_error(
            method_name="/TestService/Method",
            peer="127.0.0.1:50000",
            duration=0.01,
            error=error,
            metadata={},
        )

        mock_logger.warning.assert_called_once()

    def test_log_unexpected_error(self):
        """意外错误日志"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        error = ValueError("Unexpected value error")

        interceptor._log_unexpected_error(
            method_name="/TestService/Method",
            peer="127.0.0.1:50000",
            duration=0.001,
            error=error,
            metadata={},
        )

        mock_logger.error.assert_called_once()


class TestLoggingInterceptorDuration:
    """耗时计算测试"""

    def test_duration_calculation(self):
        """耗时计算"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        start_time = time.perf_counter()
        time.sleep(0.1)  # 模拟 100ms 延迟
        duration = time.perf_counter() - start_time

        interceptor._log_success(
            method_name="/TestService/Method",
            peer="127.0.0.1:50000",
            duration=duration,
            metadata={},
        )

        call_args = mock_logger.log.call_args
        extra = call_args[1]["extra"]

        # 延迟应该约等于 100ms（允许一些误差）
        assert 90 <= extra.get("duration_ms", 0) <= 200


class TestLoggingInterceptorInterceptService:
    """intercept_service 异步测试"""

    @pytest.mark.asyncio
    async def test_intercept_service_success(self):
        """成功调用"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        details = MockHandlerCallDetails(
            method=b"/TestService/Method",
            metadata=[],
            peer="127.0.0.1:50000",
        )
        from unittest.mock import AsyncMock
        continuation = AsyncMock(return_value=MagicMock())

        result = await interceptor.intercept_service(continuation, details)

        continuation.assert_called_once_with(details)

    @pytest.mark.asyncio
    async def test_intercept_service_with_grpc_error(self):
        """gRPC 错误"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        details = MockHandlerCallDetails(
            method=b"/TestService/Method",
            metadata=[],
            peer="127.0.0.1:50000",
        )
        from unittest.mock import AsyncMock

        # 创建一个 mock RpcError 接口的对象
        class MockRpcError(grpc.RpcError):
            def __init__(self, code, details):
                self._code = code
                self._details = details

            def code(self):
                return self._code

            def details(self):
                return self._details

        def raise_rpc_error(details):
            raise MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found")

        continuation = AsyncMock(side_effect=raise_rpc_error)

        # 执行调用并验证异常被抛出
        with pytest.raises(grpc.RpcError):
            await interceptor.intercept_service(continuation, details)

    @pytest.mark.asyncio
    async def test_intercept_service_with_unexpected_error(self):
        """意外错误"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        details = MockHandlerCallDetails(
            method=b"/TestService/Method",
            metadata=[],
            peer="127.0.0.1:50000",
        )
        from unittest.mock import AsyncMock
        continuation = AsyncMock(side_effect=ValueError("Unexpected"))

        with pytest.raises(ValueError):
            await interceptor.intercept_service(continuation, details)

    @pytest.mark.asyncio
    async def test_intercept_service_with_string_method(self):
        """字符串方法名"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        details = MockHandlerCallDetails(
            method="/TestService/Method",
            metadata=[],
            peer="127.0.0.1:50000",
        )
        from unittest.mock import AsyncMock
        continuation = AsyncMock(return_value=MagicMock())

        result = await interceptor.intercept_service(continuation, details)

        continuation.assert_called_once_with(details)

    def test_get_peer_info_with_exception(self):
        """_get_peer_info 异常处理"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        details = MagicMock()
        details.peer = property(lambda self: (_ for _ in ()).throw(Exception("peer error")))

        # Accessing peer should raise an exception
        try:
            details.peer
        except Exception:
            pass  # Expected

        # The _get_peer_info should return "unknown" on exception
        # We need to set up the mock differently
        details = MagicMock()
        del details.peer  # Remove the property

        result = interceptor._get_peer_info(details)
        assert result == "unknown"

    def test_extract_metadata_with_empty_key(self):
        """提取 metadata 空 key"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()
        metadata_items = [
            MagicMock(key="", value="empty-key"),
        ]
        details = MockHandlerCallDetails(metadata=metadata_items)

        metadata = interceptor._extract_metadata(details)

        assert metadata.get("") == "empty-key"

    def test_extract_metadata_with_iteration_error(self):
        """metadata 迭代异常处理"""
        from quantumflow.grpc.interceptors.logging import LoggingInterceptor

        interceptor = LoggingInterceptor()

        # 创建一个在迭代时抛出异常的 metadata
        def raise_on_iter():
            raise Exception("iteration error")

        mock_metadata = MagicMock()
        mock_metadata.__iter__ = raise_on_iter
        details = MagicMock()
        details.invocation_metadata = mock_metadata

        metadata = interceptor._extract_metadata(details)

        # 应该返回空字典而不是崩溃
        assert metadata == {}


class TestUnaryUnaryLoggingInterceptor:
    """UnaryUnaryLoggingInterceptor 测试"""

    def test_init(self):
        """初始化"""
        from quantumflow.grpc.interceptors.logging import UnaryUnaryLoggingInterceptor

        interceptor = UnaryUnaryLoggingInterceptor()

        assert interceptor.logger is not None

    def test_init_with_logger_name(self):
        """带 logger name 初始化"""
        from quantumflow.grpc.interceptors.logging import UnaryUnaryLoggingInterceptor

        interceptor = UnaryUnaryLoggingInterceptor(logger_name="test.logger")

        assert interceptor.logger is not None

    def test_log_call_started(self):
        """记录调用开始"""
        from quantumflow.grpc.interceptors.logging import UnaryUnaryLoggingInterceptor

        interceptor = UnaryUnaryLoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        interceptor.log_call_started("/TestService/Method")

        mock_logger.info.assert_called_once()

    def test_log_call_completed(self):
        """记录调用完成"""
        from quantumflow.grpc.interceptors.logging import UnaryUnaryLoggingInterceptor

        interceptor = UnaryUnaryLoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        interceptor.log_call_completed("/TestService/Method", 1.5)

        mock_logger.info.assert_called_once()

    def test_log_call_failed(self):
        """记录调用失败"""
        from quantumflow.grpc.interceptors.logging import UnaryUnaryLoggingInterceptor

        interceptor = UnaryUnaryLoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        interceptor.log_call_failed("/TestService/Method", ValueError("test error"))

        mock_logger.error.assert_called_once()


class TestUnaryStreamLoggingInterceptor:
    """UnaryStreamLoggingInterceptor 测试"""

    def test_init(self):
        """初始化"""
        from quantumflow.grpc.interceptors.logging import UnaryStreamLoggingInterceptor

        interceptor = UnaryStreamLoggingInterceptor()

        assert interceptor.logger is not None

    def test_init_with_logger_name(self):
        """带 logger name 初始化"""
        from quantumflow.grpc.interceptors.logging import UnaryStreamLoggingInterceptor

        interceptor = UnaryStreamLoggingInterceptor(logger_name="test.logger")

        assert interceptor.logger is not None

    def test_log_call_started(self):
        """记录调用开始"""
        from quantumflow.grpc.interceptors.logging import UnaryStreamLoggingInterceptor

        interceptor = UnaryStreamLoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        interceptor.log_call_started("/TestService/Method")

        mock_logger.info.assert_called_once()

    def test_log_call_completed(self):
        """记录调用完成"""
        from quantumflow.grpc.interceptors.logging import UnaryStreamLoggingInterceptor

        interceptor = UnaryStreamLoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        interceptor.log_call_completed("/TestService/Method", 2.5)

        mock_logger.info.assert_called_once()

    def test_log_call_failed(self):
        """记录调用失败"""
        from quantumflow.grpc.interceptors.logging import UnaryStreamLoggingInterceptor

        interceptor = UnaryStreamLoggingInterceptor()
        mock_logger = MagicMock()
        interceptor.logger = mock_logger

        interceptor.log_call_failed("/TestService/Method", RuntimeError("test error"))

        mock_logger.error.assert_called_once()
