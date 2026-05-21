"""REST API 限流中间件"""

import threading
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class TokenBucket:
    """令牌桶算法实现 - 与 quantumflow/grpc/interceptors/rate_limit.py 保持一致"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self, tokens: int = 1) -> bool:
        """尝试获取令牌"""
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

    @property
    def available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self._lock:
            self._refill_unlocked()
            return self.tokens


class RateLimitMiddleware(BaseHTTPMiddleware):
    """REST API 限流中间件

    基于令牌桶算法，支持：
    - 全局限流
    - 按端点限流（per_endpoint=True）
    - 可配置的 QPS 和突发容量
    """

    def __init__(
        self,
        app,
        qps: int = 100,
        burst: int = 200,
        per_endpoint: bool = False,
    ):
        super().__init__(app)
        self.per_endpoint = per_endpoint
        self.bucket = TokenBucket(capacity=burst, refill_rate=float(qps))
        self.endpoint_buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, path: str) -> TokenBucket:
        """获取对应端点的令牌桶"""
        if not self.per_endpoint:
            return self.bucket
        if path not in self.endpoint_buckets:
            self.endpoint_buckets[path] = TokenBucket(
                capacity=self.bucket.capacity,
                refill_rate=self.bucket.refill_rate,
            )
        return self.endpoint_buckets[path]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        bucket = self._get_bucket(path)
        if not bucket.try_acquire(1):
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "请求过于频繁，请稍后再试",
                    }
                },
            )
        return await call_next(request)
