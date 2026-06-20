"""gRPC 日志拦截器

记录所有 gRPC 调用的详细信息，包括：
- 方法名
- 调用者信息
- 调用耗时
- 状态码
- 错误信息（如果失败）
"""

import time
from typing import Callable, Dict, Optional

import grpc

from quantumflow.utils.logging import get_logger

logger = get_logger(__name__)


class LoggingInterceptor(grpc.ServerInterceptor):
    """日志拦截器 - 记录所有 gRPC 调用

    功能:
    - 记录方法名、调用者 IP
    - 记录调用耗时
    - 记录状态码和错误信息
    - 区分成功和失败调用
    """

    def __init__(
        self,
        logger_name: Optional[str] = None,
        log_level: int = 20,  # INFO
        include_metadata: bool = True,
        include_call_duration: bool = True,
    ):
        """
        Args:
            logger_name: 自定义日志记录器名称
            log_level: 日志级别 (logging.INFO = 20, logging.DEBUG = 10)
            include_metadata: 是否记录 metadata
            include_call_duration: 是否记录调用耗时
        """
        self.logger = get_logger(logger_name or __name__)
        self.log_level = log_level
        self.include_metadata = include_metadata
        self.include_call_duration = include_call_duration

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ):
        """拦截 gRPC 调用并记录日志

        Args:
            continuation: 继续处理调用的函数
            handler_call_details: 调用详情

        Returns:
            RpcMethodHandler
        """
        method_name = handler_call_details.method.decode() if isinstance(handler_call_details.method, bytes) else handler_call_details.method
        start_time = time.perf_counter()

        # 提取调用者信息
        peer = self._get_peer_info(handler_call_details)
        call_metadata = self._extract_metadata(handler_call_details)

        # 记录调用开始
        self.logger.log(
            self.log_level,
            f"gRPC call started: {method_name} from {peer}",
            extra={
                "method": method_name,
                "peer": peer,
                "metadata": call_metadata if self.include_metadata else None,
            },
        )

        try:
            # 执行调用
            response = continuation(handler_call_details)
            duration = time.perf_counter() - start_time

            # 记录成功
            self._log_success(method_name, peer, duration, call_metadata)
            return response

        except grpc.RpcError as e:
            duration = time.perf_counter() - start_time
            self._log_error(method_name, peer, duration, e, call_metadata)
            raise

        except Exception as e:
            duration = time.perf_counter() - start_time
            self._log_unexpected_error(method_name, peer, duration, e, call_metadata)
            raise

    def _get_peer_info(self, handler_call_details: grpc.HandlerCallDetails) -> str:
        """获取调用者信息"""
        try:
            # 从 context 获取 peer 信息（如果可用）
            # 注意：在 ServerInterceptor 中我们无法直接访问 context
            # 这里返回占位符，实际 peer 信息需要在 handler 中获取
            return handler_call_details.peer or "unknown"
        except Exception:
            return "unknown"

    def _extract_metadata(
        self, handler_call_details: grpc.HandlerCallDetails
    ) -> Dict[str, str]:
        """提取 metadata"""
        metadata = {}
        try:
            if handler_call_details.invocation_metadata:
                for item in handler_call_details.invocation_metadata:
                    metadata[item.key] = item.value
        except Exception:
            pass
        return metadata

    def _log_success(
        self,
        method_name: str,
        peer: str,
        duration: float,
        metadata: Dict[str, str],
    ) -> None:
        """记录成功调用"""
        log_data = {
            "method": method_name,
            "peer": peer,
            "status": "OK",
            "duration_ms": round(duration * 1000, 2),
        }
        if self.include_metadata:
            log_data["metadata"] = metadata

        self.logger.log(
            self.log_level,
            f"gRPC call completed: {method_name} ({duration*1000:.2f}ms)",
            extra=log_data,
        )

    def _log_error(
        self,
        method_name: str,
        peer: str,
        duration: float,
        error: grpc.RpcError,
        metadata: Dict[str, str],
    ) -> None:
        """记录错误调用"""
        status_code = error.code() if callable(error.code) else error.code
        error_message = error.details() if callable(error.details) else str(error)

        log_data = {
            "method": method_name,
            "peer": peer,
            "status": "ERROR",
            "status_code": status_code.name if status_code else "UNKNOWN",
            "error_message": error_message,
            "duration_ms": round(duration * 1000, 2),
        }
        if self.include_metadata:
            log_data["metadata"] = metadata

        # 错误日志使用 WARNING 级别
        self.logger.warning(
            f"gRPC call failed: {method_name} - {status_code.name if status_code else 'UNKNOWN'}: {error_message}",
            extra=log_data,
        )

    def _log_unexpected_error(
        self,
        method_name: str,
        peer: str,
        duration: float,
        error: Exception,
        metadata: Dict[str, str],
    ) -> None:
        """记录意外错误"""
        log_data = {
            "method": method_name,
            "peer": peer,
            "status": "UNEXPECTED_ERROR",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration_ms": round(duration * 1000, 2),
        }
        if self.include_metadata:
            log_data["metadata"] = metadata

        self.logger.error(
            f"gRPC unexpected error: {method_name} - {type(error).__name__}: {error}",
            extra=log_data,
        )


class UnaryUnaryLoggingInterceptor:
    """Unary-Unary 调用日志拦截器（同步版本占位符）

    注意: 同步版本的 ServerInterceptor 在标准 grpc 库中不可用
    此拦截器仅作为接口定义，需要配合 aio 使用
    """

    def __init__(self, logger_name: Optional[str] = None):
        self.logger = get_logger(logger_name or __name__)

    def log_call_started(self, method_name: str) -> None:
        """记录调用开始"""
        self.logger.info(f"gRPC UnaryUnary call started: {method_name}")

    def log_call_completed(self, method_name: str, duration_ms: float) -> None:
        """记录调用完成"""
        self.logger.info(
            f"gRPC UnaryUnary call completed: {method_name} ({duration_ms:.2f}ms)"
        )

    def log_call_failed(self, method_name: str, error: Exception) -> None:
        """记录调用失败"""
        self.logger.error(
            f"gRPC UnaryUnary call failed: {method_name} - {type(error).__name__}: {error}"
        )


class UnaryStreamLoggingInterceptor:
    """Unary-Stream 调用日志拦截器（同步版本占位符）

    注意: 同步版本的 ServerInterceptor 在标准 grpc 库中不可用
    此拦截器仅作为接口定义，需要配合 aio 使用
    """

    def __init__(self, logger_name: Optional[str] = None):
        self.logger = get_logger(logger_name or __name__)

    def log_call_started(self, method_name: str) -> None:
        """记录调用开始"""
        self.logger.info(f"gRPC UnaryStream call started: {method_name}")

    def log_call_completed(self, method_name: str, duration_ms: float) -> None:
        """记录调用完成"""
        self.logger.info(
            f"gRPC UnaryStream call completed: {method_name} ({duration_ms:.2f}ms)"
        )

    def log_call_failed(self, method_name: str, error: Exception) -> None:
        """记录调用失败"""
        self.logger.error(
            f"gRPC UnaryStream call failed: {method_name} - {type(error).__name__}: {error}"
        )
