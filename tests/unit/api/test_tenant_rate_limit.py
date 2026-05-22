"""租户限流单元测试"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from quantumflow.api.middlewares.rate_limit import (
    TenantRateLimiter,
    TokenBucket,
    RateLimitMiddleware,
)
from quantumflow.api.models.tenant import Tenant, TenantStatus, QuotaConfig


def test_token_bucket_basic():
    """测试令牌桶基本功能"""
    bucket = TokenBucket(capacity=10, refill_rate=1.0)  # 1 token/sec

    # 消耗所有令牌
    for _ in range(10):
        assert bucket.try_acquire(1) is True

    # 应该被拒绝
    assert bucket.try_acquire(1) is False


def test_token_bucket_refill():
    """测试令牌补充"""
    bucket = TokenBucket(capacity=5, refill_rate=10.0)  # 10 tokens/sec

    # 消耗所有令牌
    for _ in range(5):
        bucket.try_acquire(1)

    # 等待补充 - 稍微多等一点以确保令牌补充
    time.sleep(0.15)  # 应该补充 1.5 tokens

    assert bucket.try_acquire(1) is True


def test_tenant_rate_limiter_single():
    """测试单一租户限流"""
    limiter = TenantRateLimiter()

    # 前 10 个请求应该通过
    for _ in range(10):
        assert limiter.check_limit("tenant-1", qps=60, burst=10) is True

    # 第 11 个应该被拒绝
    assert limiter.check_limit("tenant-1", qps=60, burst=10) is False


def test_tenant_rate_limiter_multiple():
    """测试多租户隔离"""
    limiter = TenantRateLimiter()

    # 租户 1 消耗完自己的配额
    for _ in range(10):
        limiter.check_limit("tenant-1", qps=60, burst=10)

    # 租户 1 被限流
    assert limiter.check_limit("tenant-1", qps=60, burst=10) is False

    # 租户 2 不受影响
    assert limiter.check_limit("tenant-2", qps=60, burst=10) is True


def test_tenant_rate_limiter_different_quota():
    """测试不同配额"""
    limiter = TenantRateLimiter()

    # 租户 1: 小配额
    for _ in range(5):
        limiter.check_limit("small", qps=60, burst=5)

    assert limiter.check_limit("small", qps=60, burst=5) is False

    # 租户 2: 大配额
    for _ in range(10):
        limiter.check_limit("large", qps=60, burst=100)

    assert limiter.check_limit("large", qps=60, burst=100) is True


def test_token_bucket_available_tokens():
    """测试 available_tokens 属性"""
    bucket = TokenBucket(capacity=10, refill_rate=1.0)
    assert bucket.available_tokens == 10.0

    for _ in range(5):
        bucket.try_acquire(1)
    assert bucket.available_tokens < 10.0
    assert bucket.available_tokens >= 4.0


def test_get_remaining_nonexistent():
    """测试获取不存在租户的剩余令牌"""
    limiter = TenantRateLimiter()
    remaining = limiter.get_remaining("nonexistent")
    assert remaining > 0


def test_get_remaining_after_usage():
    """测试使用后的剩余令牌"""
    limiter = TenantRateLimiter()
    limiter.check_limit("tenant-1", qps=60, burst=10)
    limiter.check_limit("tenant-1", qps=60, burst=10)
    remaining = limiter.get_remaining("tenant-1")
    assert remaining < 10.0


def test_remove_bucket_success():
    """测试成功移除令牌桶"""
    limiter = TenantRateLimiter()
    limiter.check_limit("tenant-del", qps=60, burst=10)
    assert limiter.remove_bucket("tenant-del") is True
    assert limiter.get_remaining("tenant-del") == limiter._default_burst  # 桶已移除


def test_remove_bucket_nonexistent():
    """测试移除不存在的令牌桶"""
    limiter = TenantRateLimiter()
    assert limiter.remove_bucket("nonexistent") is False


def test_check_limit_default_values():
    """测试使用默认 qps/burst 值"""
    limiter = TenantRateLimiter()
    limiter._default_qps = 60
    limiter._default_burst = 5

    for _ in range(5):
        assert limiter.check_limit("default-tenant") is True

    assert limiter.check_limit("default-tenant") is False


def test_eviction_at_max_buckets():
    """测试超过 max_buckets 时的驱逐"""
    limiter = TenantRateLimiter(max_buckets=3)

    # 创建 3 个租户桶
    limiter.check_limit("t1", qps=60, burst=10)
    limiter.check_limit("t2", qps=60, burst=10)
    limiter.check_limit("t3", qps=60, burst=10)

    # 第 4 个应该驱逐最旧的
    limiter.check_limit("t4", qps=60, burst=10)
    assert "t4" in limiter._buckets
    # 最多 3 个桶
    assert len(limiter._buckets) <= 3


def test_rate_limit_middleware_per_tenant():
    """测试 per_tenant 模式的限流中间件（无租户上下文时正常放行）"""
    app_mock = MagicMock()
    middleware = RateLimitMiddleware(app_mock, per_tenant=True)

    request = MagicMock()
    request.url.path = "/api/v1/inference/generate"
    call_next = AsyncMock(return_value=MagicMock(status_code=200))

    with patch("quantumflow.api.middleware.auth.TenantContext") as mock_ctx:
        mock_ctx.get_tenant.return_value = None
        import asyncio
        response = asyncio.run(middleware.dispatch(request, call_next))
        assert response.status_code == 200


def test_rate_limit_middleware_tenant_exceeded():
    """测试租户限流超限返回 429"""
    app_mock = MagicMock()
    middleware = RateLimitMiddleware(app_mock, per_tenant=True)

    quota = QuotaConfig(requests_per_minute=60, concurrent_requests=5)
    tenant = Tenant(
        id="limited-tenant",
        name="Limited",
        api_key_hash="hash",
        api_key_prefix="prefix",
        status=TenantStatus.ACTIVE,
        quota=quota,
    )

    request = MagicMock()
    request.url.path = "/api/v1/inference/generate"
    call_next = AsyncMock(return_value=MagicMock(status_code=200))

    with patch("quantumflow.api.middleware.auth.TenantContext") as mock_ctx:
        mock_ctx.get_tenant.return_value = tenant
        with patch("quantumflow.api.middlewares.rate_limit._global_limiter") as mock_limiter:
            mock_limiter.check_limit.return_value = False
            import asyncio
            response = asyncio.run(middleware.dispatch(request, call_next))
            assert response.status_code == 429
            import json
            data = json.loads(response.body.decode())
            assert data["error"]["code"] == "TENANT_RATE_LIMIT_EXCEEDED"


def test_rate_limit_middleware_per_endpoint_bucket():
    """测试 per_endpoint 模式的令牌桶"""
    app_mock = MagicMock()
    middleware = RateLimitMiddleware(app_mock, qps=10, burst=5, per_endpoint=True)

    bucket = middleware._get_bucket("/api/v1/inference/generate")
    assert bucket is not None
    # 同一端点应返回相同桶
    bucket2 = middleware._get_bucket("/api/v1/inference/generate")
    assert bucket is bucket2


def test_rate_limit_middleware_global_exceeded():
    """测试全局限流超限返回 429"""
    app_mock = MagicMock()
    middleware = RateLimitMiddleware(app_mock, qps=10, burst=1)

    # 消耗令牌
    request = MagicMock()
    request.url.path = "/api/v1/inference/generate"
    call_next = AsyncMock(return_value=MagicMock(status_code=200))

    # 首先消耗完配额
    middleware.bucket.try_acquire(1)

    import asyncio
    response = asyncio.run(middleware.dispatch(request, call_next))
    assert response.status_code == 429
    import json
    data = json.loads(response.body.decode())
    assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
