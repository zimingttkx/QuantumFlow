"""gRPC 限流拦截器

基于令牌桶算法的限流实现：
- 支持全局限流
- 支持按方法限流
- 可配置的 QPS 和突发容量
"""

import threading
import time
from typing import Callable, Dict, Optional, Set

import grpc


class TokenBucket:
    """令牌桶算法实现

    Attributes:
        capacity: 桶的最大容量
        refill_rate: 每秒补充的令牌数
        tokens: 当前令牌数
        last_refill: 上次补充时间
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: 桶的最大容量（初始令牌数）
            refill_rate: 每秒补充的令牌数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self, tokens: int = 1) -> bool:
        """尝试获取令牌

        Args:
            tokens: 要获取的令牌数

        Returns:
            True 如果获取成功，False 如果令牌不足
        """
        with self._lock:
            self._refill_unlocked()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill_unlocked(self) -> None:
        """内部方法：在持有锁的情况下补充令牌"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def refill(self) -> None:
        """补充令牌（公开方法）"""
        with self._lock:
            self._refill_unlocked()

    @property
    def available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self._lock:
            self._refill_unlocked()
            return self.tokens

    def reset(self) -> None:
        """重置令牌桶"""
        with self._lock:
            self.tokens = float(self.capacity)
            self.last_refill = time.monotonic()


class RateLimitInterceptor(grpc.ServerInterceptor):
    """限流拦截器 - 基于令牌桶算法

    功能:
    - 全局限流
    - 按方法限流
    - 可配置的 QPS 和突发容量
    - 令牌桶自动补充
    """

    def __init__(
        self,
        qps: int = 100,
        burst: int = 200,
        per_method: bool = False,
        methods: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """
        Args:
            qps: 每秒允许的请求数（默认 100）
            burst: 突发容量（默认 200）
            per_method: 是否按方法分别限流
            methods: 方法特定的限流配置 {"method_name": {"qps": 50, "burst": 100}}
        """
        self.per_method = per_method
        self.methods = methods or {}

        if per_method:
            # 按方法的限流桶
            self._method_buckets: Dict[str, TokenBucket] = {}
            self._global_bucket: Optional[TokenBucket] = None
        else:
            # 全局限流桶
            self._global_bucket = TokenBucket(capacity=burst, refill_rate=qps)
            self._method_buckets = {}

        self._lock = threading.Lock()

    def _get_bucket_for_method(self, method_name: str) -> TokenBucket:
        """获取方法对应的令牌桶"""
        if method_name in self._method_buckets:
            return self._method_buckets[method_name]

        # 方法特定的配置优先
        if method_name in self.methods:
            config = self.methods[method_name]
            bucket = TokenBucket(
                capacity=config.get("burst", 100),
                refill_rate=config.get("qps", 50),
            )
        else:
            # 默认使用全局配置
            if self._global_bucket:
                return self._global_bucket
            # per_method=True 但没有全局桶
            bucket = TokenBucket(capacity=100, refill_rate=50)

        with self._lock:
            self._method_buckets[method_name] = bucket

        return bucket

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """拦截并检查限流"""
        method_name = self._get_method_name(handler_call_details)
        bucket = self._get_bucket_for_method(method_name)

        if not bucket.try_acquire(1):
            # 限流触发
            raise grpc.RpcError(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"Rate limit exceeded for method {method_name}. Try again later.",
            )

        return continuation(handler_call_details)

    def _get_method_name(self, handler_call_details: grpc.HandlerCallDetails) -> str:
        """获取方法名"""
        method = handler_call_details.method
        if isinstance(method, bytes):
            return method.decode()
        return method or "unknown"

    def update_qps(self, qps: int) -> None:
        """更新 QPS 配置

        Args:
            qps: 新的 QPS 值
        """
        if self._global_bucket:
            self._global_bucket.refill_rate = qps

    def update_burst(self, burst: int) -> None:
        """更新突发容量

        Args:
            burst: 新的突发容量值
        """
        if self._global_bucket:
            self._global_bucket.capacity = burst


class SlidingWindowRateLimiter:
    """滑动窗口限流器

    更精确的限流实现，基于滑动窗口算法。
    """

    def __init__(self, max_requests: int, window_seconds: float):
        """
        Args:
            max_requests: 窗口内的最大请求数
            window_seconds: 窗口大小（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list = []
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """尝试获取许可"""
        with self._lock:
            now = time.monotonic()

            # 清理过期的请求记录
            cutoff = now - self.window_seconds
            self.requests = [t for t in self.requests if t > cutoff]

            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True

            return False

    @property
    def current_count(self) -> int:
        """当前窗口内的请求数"""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            return len([t for t in self.requests if t > cutoff])

    def reset(self) -> None:
        """重置限流器"""
        with self._lock:
            self.requests = []


class SlidingWindowRateLimitInterceptor(grpc.ServerInterceptor):
    """滑动窗口限流拦截器

    使用滑动窗口算法进行更精确的限流。
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 1.0,
        per_method: bool = False,
    ):
        """
        Args:
            max_requests: 窗口内的最大请求数
            window_seconds: 窗口大小（秒）
            per_method: 是否按方法分别限流
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.per_method = per_method

        if per_method:
            self._method_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        else:
            self._global_limiter = SlidingWindowRateLimiter(max_requests, window_seconds)
            self._method_limiters = {}

        self._lock = threading.Lock()

    def _get_limiter_for_method(self, method_name: str) -> SlidingWindowRateLimiter:
        """获取方法对应的限流器"""
        if method_name in self._method_limiters:
            return self._method_limiters[method_name]

        limiter = SlidingWindowRateLimiter(self.max_requests, self.window_seconds)

        with self._lock:
            self._method_limiters[method_name] = limiter

        return limiter

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """拦截并检查限流"""
        method_name = self._get_method_name(handler_call_details)

        if self.per_method:
            limiter = self._get_limiter_for_method(method_name)
        else:
            limiter = self._global_limiter

        if not limiter.try_acquire():
            raise grpc.RpcError(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"Rate limit exceeded for method {method_name}. Try again later.",
            )

        return continuation(handler_call_details)

    def _get_method_name(self, handler_call_details: grpc.HandlerCallDetails) -> str:
        """获取方法名"""
        method = handler_call_details.method
        if isinstance(method, bytes):
            return method.decode()
        return method or "unknown"
