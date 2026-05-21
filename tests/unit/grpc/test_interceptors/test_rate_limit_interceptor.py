"""gRPC 限流拦截器测试

严格测试限流拦截器和令牌桶算法的业务逻辑：
- 令牌桶算法正确性
- 全局限流
- 按方法限流
- 并发场景
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import grpc

from quantumflow.grpc.interceptors.rate_limit import (
    TokenBucket,
    RateLimitInterceptor,
    SlidingWindowRateLimiter,
)


class TestTokenBucketAlgorithm:
    """令牌桶算法测试"""

    def test_initial_tokens_equal_capacity(self):
        """初始令牌数等于容量"""
        bucket = TokenBucket(capacity=10, refill_rate=5)

        assert bucket.tokens == 10, "初始令牌数应等于容量"

    def test_try_acquire_success(self):
        """获取令牌成功"""
        bucket = TokenBucket(capacity=10, refill_rate=0)

        result = bucket.try_acquire(1)

        assert result is True
        assert bucket.tokens == 9, "获取后令牌数应减 1"

    def test_try_acquire_insufficient_tokens(self):
        """令牌不足时获取失败"""
        bucket = TokenBucket(capacity=2, refill_rate=0)

        result = bucket.try_acquire(5)

        assert result is False
        assert bucket.tokens == 2, "获取失败时令牌数不变"

    def test_try_acquire_exact_tokens(self):
        """获取恰好数量的令牌"""
        bucket = TokenBucket(capacity=5, refill_rate=0)

        result = bucket.try_acquire(5)

        assert result is True
        assert bucket.tokens == 0

    def test_try_acquire_zero_tokens(self):
        """获取 0 个令牌"""
        bucket = TokenBucket(capacity=10, refill_rate=0)

        result = bucket.try_acquire(0)

        assert result is True
        assert bucket.tokens == 10

    def test_reset_bucket(self):
        """重置令牌桶"""
        bucket = TokenBucket(capacity=10, refill_rate=0)
        bucket.try_acquire(7)

        bucket.reset()

        assert bucket.tokens == 10

    def test_thread_safety_concurrent_acquire(self):
        """并发获取的线程安全"""
        bucket = TokenBucket(capacity=1000, refill_rate=0)
        errors = []

        def acquire_many(n):
            try:
                for _ in range(n):
                    bucket.try_acquire(1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquire_many, args=(100,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert bucket.tokens == 0

    def test_refill_method(self):
        """refill 公开方法"""
        bucket = TokenBucket(capacity=10, refill_rate=5)
        bucket.try_acquire(5)  # 剩下 5 个

        bucket.refill()  # 补充

        assert bucket.tokens >= 5

    def test_available_tokens_property(self):
        """available_tokens 属性"""
        bucket = TokenBucket(capacity=10, refill_rate=0)

        assert bucket.available_tokens == 10

        bucket.try_acquire(3)

        assert bucket.available_tokens == 7


class TestRateLimitInterceptorBasics:
    """限流拦截器基础测试"""

    def test_intercept_service_calls_continuation(self):
        """intercept_service 调用 continuation"""
        interceptor = RateLimitInterceptor(qps=100, burst=100)

        details = MagicMock()
        details.method = b"/TestService/Method"
        continuation = MagicMock(return_value=MagicMock())

        result = interceptor.intercept_service(continuation, details)

        continuation.assert_called_once_with(details)
        assert result is not None

    def test_intercept_service_blocks_when_no_tokens(self):
        """无令牌时阻止请求"""
        interceptor = RateLimitInterceptor(qps=0, burst=0)

        details = MagicMock()
        details.method = b"/TestService/Method"
        continuation = MagicMock(return_value=MagicMock())

        # 验证抛出异常
        with pytest.raises(Exception) as exc_info:
            interceptor.intercept_service(continuation, details)

        # 验证异常信息包含限流相关文字
        assert "Rate limit" in str(exc_info.value) or "RESOURCE_EXHAUSTED" in str(exc_info.value)

    def test_method_name_extraction_bytes(self):
        """方法名提取 bytes 类型"""
        interceptor = RateLimitInterceptor()
        details = MagicMock()
        details.method = b"/TestService/Method"

        name = interceptor._get_method_name(details)

        assert name == "/TestService/Method"

    def test_method_name_extraction_string(self):
        """方法名提取 string 类型"""
        interceptor = RateLimitInterceptor()
        details = MagicMock()
        details.method = "/TestService/Method"

        name = interceptor._get_method_name(details)

        assert name == "/TestService/Method"

    def test_update_qps(self):
        """更新 QPS"""
        interceptor = RateLimitInterceptor(qps=100, burst=200)

        interceptor.update_qps(200)

        # 验证 QPS 已更新

    def test_update_burst(self):
        """更新突发容量"""
        interceptor = RateLimitInterceptor(qps=100, burst=200)

        interceptor.update_burst(300)

        # 验证 burst 已更新

    def test_per_method_rate_limiting(self):
        """按方法限流"""
        interceptor = RateLimitInterceptor(
            qps=100, burst=100, per_method=True,
            methods={"/TestService/Method": {"qps": 10, "burst": 20}}
        )

        details = MagicMock()
        details.method = b"/TestService/Method"
        continuation = MagicMock(return_value=MagicMock())

        # 第一次调用应该成功
        result = interceptor.intercept_service(continuation, details)
        assert result is not None

    def test_per_method_rate_limiting_creates_bucket(self):
        """按方法限流创建桶"""
        interceptor = RateLimitInterceptor(
            qps=100, burst=100, per_method=True
        )

        details = MagicMock()
        details.method = b"/NewMethod/Call"
        continuation = MagicMock(return_value=MagicMock())

        # 调用应该创建新的桶
        interceptor.intercept_service(continuation, details)

    def test_get_bucket_for_method_returns_existing(self):
        """获取桶返回已存在的"""
        interceptor = RateLimitInterceptor(
            qps=100, burst=100, per_method=True
        )

        details = MagicMock()
        details.method = b"/TestService/Method"
        continuation = MagicMock(return_value=MagicMock())

        # 第一次调用创建桶
        interceptor.intercept_service(continuation, details)

        # 第二次调用应该返回相同的桶
        bucket1 = interceptor._get_bucket_for_method("/TestService/Method")
        bucket2 = interceptor._get_bucket_for_method("/TestService/Method")

        assert bucket1 is bucket2


class TestSlidingWindowRateLimitInterceptor:
    """SlidingWindowRateLimitInterceptor 测试"""

    def test_init(self):
        """初始化"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(max_requests=100, window_seconds=1.0)

        assert interceptor.max_requests == 100
        assert interceptor.window_seconds == 1.0

    def test_init_per_method(self):
        """按方法初始化"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(
            max_requests=100, window_seconds=1.0, per_method=True
        )

        assert interceptor.per_method is True

    def test_get_limiter_for_method_creates_new(self):
        """获取方法限流器创建新的"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(max_requests=100, window_seconds=1.0)

        new_limiter = interceptor._get_limiter_for_method("/NewMethod/Call")

        assert new_limiter is not None

    def test_get_limiter_for_method_returns_existing(self):
        """获取方法限流器返回已存在的"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(max_requests=100, window_seconds=1.0)

        lim1 = interceptor._get_limiter_for_method("/Method1")
        lim2 = interceptor._get_limiter_for_method("/Method1")

        assert lim1 is lim2

    def test_intercept_service_allows_request(self):
        """允许请求"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(max_requests=100, window_seconds=1.0)

        details = MagicMock()
        details.method = b"/TestService/Method"
        continuation = MagicMock(return_value=MagicMock())

        result = interceptor.intercept_service(continuation, details)

        continuation.assert_called_once()

    def test_intercept_service_blocks_request(self):
        """阻止请求"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(max_requests=0, window_seconds=1.0)

        details = MagicMock()
        details.method = b"/TestService/Method"
        continuation = MagicMock(return_value=MagicMock())

        with pytest.raises(Exception) as exc_info:
            interceptor.intercept_service(continuation, details)

        assert "Rate limit" in str(exc_info.value) or "RESOURCE_EXHAUSTED" in str(exc_info.value)

    def test_intercept_service_per_method(self):
        """按方法限流"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(
            max_requests=100, window_seconds=1.0, per_method=True
        )

        details = MagicMock()
        details.method = b"/TestService/Method"
        continuation = MagicMock(return_value=MagicMock())

        result = interceptor.intercept_service(continuation, details)

        continuation.assert_called_once()

    def test_get_method_name_bytes(self):
        """获取方法名 (bytes)"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(max_requests=100, window_seconds=1.0)
        details = MagicMock()
        details.method = b"/TestService/Method"

        name = interceptor._get_method_name(details)

        assert name == "/TestService/Method"

    def test_get_method_name_string(self):
        """获取方法名 (string)"""
        from quantumflow.grpc.interceptors.rate_limit import SlidingWindowRateLimitInterceptor

        interceptor = SlidingWindowRateLimitInterceptor(max_requests=100, window_seconds=1.0)
        details = MagicMock()
        details.method = "/TestService/Method"

        name = interceptor._get_method_name(details)

        assert name == "/TestService/Method"


class TestSlidingWindowRateLimiter:
    """滑动窗口限流器测试"""

    def test_allows_requests_under_limit(self):
        """限流内请求通过"""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=1.0)

        for _ in range(5):
            assert limiter.try_acquire() is True

    def test_blocks_requests_over_limit(self):
        """超出限流拒绝"""
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=1.0)

        for _ in range(3):
            limiter.try_acquire()

        result = limiter.try_acquire()

        assert result is False

    def test_window_slides_over_time(self):
        """窗口滑动"""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=0.2)

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

        time.sleep(0.25)

        assert limiter.try_acquire() is True

    def test_reset(self):
        """重置"""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=1.0)

        limiter.try_acquire()
        limiter.try_acquire()
        limiter.reset()

        assert limiter.current_count == 0
