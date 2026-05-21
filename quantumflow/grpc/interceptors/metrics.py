"""gRPC 监控拦截器

提供 Prometheus 指标收集功能：
- 请求计数
- 请求延迟
- 活跃请求数
- 按方法分类的指标
"""

import time
from typing import Callable, Dict, Optional

import grpc
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
GRPC_REQUESTS_TOTAL = Counter(
    "grpc_requests_total",
    "Total number of gRPC requests",
    ["method", "status"],
)

GRPC_REQUEST_DURATION_SECONDS = Histogram(
    "grpc_request_duration_seconds",
    "gRPC request duration in seconds",
    ["method"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

GRPC_REQUESTS_IN_PROGRESS = Gauge(
    "grpc_requests_in_progress",
    "Number of gRPC requests currently in progress",
    ["method"],
)

GRPC_REQUESTS_FAILED_TOTAL = Counter(
    "grpc_requests_failed_total",
    "Total number of failed gRPC requests",
    ["method", "error_code"],
)


class MetricsInterceptor(grpc.ServerInterceptor):
    """监控拦截器 - 记录 Prometheus 指标

    功能:
    - 记录请求总数（按方法和状态分类）
    - 记录请求延迟
    - 记录活跃请求数
    - 记录失败请求数（按错误码分类）
    """

    def __init__(
        self,
        requests_total: Optional[Counter] = None,
        request_duration: Optional[Histogram] = None,
        requests_in_progress: Optional[Gauge] = None,
        requests_failed: Optional[Counter] = None,
        include_call_details: bool = True,
    ):
        """
        Args:
            requests_total: 自定义请求计数器
            request_duration: 自定义延迟直方图
            requests_in_progress: 自定义活跃请求Gauge
            requests_failed: 自定义失败计数器
            include_call_details: 是否记录更多调用详情
        """
        self.requests_total = requests_total or GRPC_REQUESTS_TOTAL
        self.request_duration = request_duration or GRPC_REQUEST_DURATION_SECONDS
        self.requests_in_progress = requests_in_progress or GRPC_REQUESTS_IN_PROGRESS
        self.requests_failed = requests_failed or GRPC_REQUESTS_FAILED_TOTAL
        self.include_call_details = include_call_details

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """拦截并记录指标"""
        method_name = self._get_method_name(handler_call_details)

        # 增加活跃请求数
        self.requests_in_progress.labels(method=method_name).inc()

        start_time = time.perf_counter()

        try:
            response = continuation(handler_call_details)

            # 计算延迟
            duration = time.perf_counter() - start_time

            # 记录成功指标
            self._record_success(method_name, duration)

            return response

        except grpc.RpcError as e:
            duration = time.perf_counter() - start_time
            status_code = e.code() if callable(e.code) else grpc.StatusCode.UNKNOWN
            error_code = status_code.name if status_code else "UNKNOWN"

            # 记录失败指标
            self._record_failure(method_name, duration, error_code)

            raise

        except Exception:
            duration = time.perf_counter() - start_time
            self._record_failure(method_name, duration, "UNEXPECTED_ERROR")
            raise

        finally:
            # 减少活跃请求数
            self.requests_in_progress.labels(method=method_name).dec()

    def _get_method_name(self, handler_call_details: grpc.HandlerCallDetails) -> str:
        """获取方法名"""
        method = handler_call_details.method
        if isinstance(method, bytes):
            return method.decode()
        return method or "unknown"

    def _record_success(self, method_name: str, duration: float) -> None:
        """记录成功请求指标"""
        self.requests_total.labels(method=method_name, status="OK").inc()
        self.request_duration.labels(method=method_name).observe(duration)

    def _record_failure(self, method_name: str, duration: float, error_code: str) -> None:
        """记录失败请求指标"""
        self.requests_total.labels(method=method_name, status="ERROR").inc()
        self.request_duration.labels(method=method_name).observe(duration)
        self.requests_failed.labels(method=method_name, error_code=error_code).inc()


class AsyncMetricsInterceptor:
    """异步版本的监控拦截器

    用于 aio.grpc.Server。
    """

    def __init__(self):
        self.requests_total = GRPC_REQUESTS_TOTAL
        self.request_duration = GRPC_REQUEST_DURATION_SECONDS
        self.requests_in_progress = GRPC_REQUESTS_IN_PROGRESS
        self.requests_failed = GRPC_REQUESTS_FAILED_TOTAL

    async def intercept_service(
        self,
        continuation: Callable,
        handler_call_details,
    ) -> any:
        """拦截并记录指标（异步版本）"""
        method_name = self._get_method_name(handler_call_details)

        self.requests_in_progress.labels(method=method_name).inc()

        start_time = time.perf_counter()

        try:
            response = await continuation(handler_call_details)
            duration = time.perf_counter() - start_time

            self._record_success(method_name, duration)

            return response

        except grpc.RpcError as e:
            duration = time.perf_counter() - start_time
            status_code = e.code() if callable(e.code) else grpc.StatusCode.UNKNOWN
            error_code = status_code.name if status_code else "UNKNOWN"

            self._record_failure(method_name, duration, error_code)

            raise

        except Exception:
            duration = time.perf_counter() - start_time
            self._record_failure(method_name, duration, "UNEXPECTED_ERROR")
            raise

        finally:
            self.requests_in_progress.labels(method=method_name).dec()

    def _get_method_name(self, handler_call_details) -> str:
        """获取方法名"""
        method = handler_call_details.method
        if isinstance(method, bytes):
            return method.decode()
        return method or "unknown"

    def _record_success(self, method_name: str, duration: float) -> None:
        """记录成功请求指标"""
        self.requests_total.labels(method=method_name, status="OK").inc()
        self.request_duration.labels(method=method_name).observe(duration)

    def _record_failure(self, method_name: str, duration: float, error_code: str) -> None:
        """记录失败请求指标"""
        self.requests_total.labels(method=method_name, status="ERROR").inc()
        self.request_duration.labels(method=method_name).observe(duration)
        self.requests_failed.labels(method=method_name, error_code=error_code).inc()
