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


class TenantRateLimiter:
    """租户级别限流器

    使用令牌桶算法，为每个租户维护独立的限流状态。
    """

    def __init__(self, max_buckets: int = 10000):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._default_qps = 60
        self._default_burst = 120
        self._max_buckets = max_buckets

    def _get_bucket(self, tenant_id: str, qps: int, burst: int) -> TokenBucket:
        """获取或创建租户的令牌桶"""
        with self._lock:
            if tenant_id not in self._buckets:
                # Evict oldest if at capacity (simple LRU-ish eviction)
                if len(self._buckets) >= self._max_buckets:
                    # Remove first key (not truly LRU but prevents unbounded growth)
                    oldest = next(iter(self._buckets))
                    del self._buckets[oldest]
                self._buckets[tenant_id] = TokenBucket(
                    capacity=burst,
                    refill_rate=float(qps)
                )
            return self._buckets[tenant_id]

    def check_limit(
        self,
        tenant_id: str,
        qps: int | None = None,
        burst: int | None = None
    ) -> bool:
        """检查是否超过限流

        Args:
            tenant_id: 租户 ID
            qps: 租户自定义 QPS (使用默认值如果 None)
            burst: 租户自定义突发容量 (使用默认值如果 None)

        Returns:
            True: 未超限，可以处理
            False: 已超限，拒绝请求
        """
        qps = qps if qps is not None else self._default_qps
        burst = burst if burst is not None else self._default_burst
        bucket = self._get_bucket(tenant_id, qps, burst)
        return bucket.try_acquire(1)

    def get_remaining(self, tenant_id: str) -> float:
        """获取剩余令牌数"""
        with self._lock:
            if tenant_id not in self._buckets:
                return float(self._default_burst)
            return self._buckets[tenant_id].available_tokens

    def remove_bucket(self, tenant_id: str) -> bool:
        """移除租户的令牌桶 (用于清理)

        Args:
            tenant_id: 租户 ID

        Returns:
            bool: 是否成功移除
        """
        with self._lock:
            if tenant_id in self._buckets:
                del self._buckets[tenant_id]
                return True
            return False


# 全局限流器实例
_global_limiter = TenantRateLimiter()


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
        per_tenant: bool = False,
    ):
        super().__init__(app)
        self.per_endpoint = per_endpoint
        self.per_tenant = per_tenant
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

        # 检查租户上下文 (如果启用)
        if self.per_tenant:
            from quantumflow.api.middleware.auth import TenantContext
            tenant = TenantContext.get_tenant()
            if tenant:
                # 使用租户配额
                allowed = _global_limiter.check_limit(
                    tenant.id,
                    qps=tenant.quota.requests_per_minute,
                    burst=tenant.quota.concurrent_requests
                )
                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "code": "TENANT_RATE_LIMIT_EXCEEDED",
                                "message": f"租户 {tenant.name} 请求频率超限",
                                "tenant_id": tenant.id,
                            }
                        },
                    )

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
