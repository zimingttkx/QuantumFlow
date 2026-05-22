"""租户认证中间件单元测试"""

import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quantumflow.api.middleware.auth import (
    TenantAuthMiddleware,
    TenantContext,
    generate_api_key,
    hash_api_key,
    _tenant_cache,
    _cache_lock,
)
from quantumflow.api.models.tenant import Tenant, TenantStatus, QuotaConfig


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(TenantAuthMiddleware)

    @app.get("/test")
    async def test_endpoint():
        tenant = TenantContext.get_tenant()
        if tenant:
            return {"tenant_id": tenant.id}
        return {"error": "no tenant"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_endpoint_no_auth(client):
    """健康检查端点无需认证"""
    response = client.get("/health")
    assert response.status_code == 200


def test_missing_api_key(client):
    """缺少 API Key 返回 401（有认证基础设施时）"""
    with patch.object(TenantAuthMiddleware, "_has_auth_infrastructure", return_value=True):
        response = client.get("/test")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "MISSING_API_KEY"


def test_dev_mode_no_auth_infrastructure(client):
    """无认证基础设施且无 API Key 时，开发模式放行"""
    with patch.object(TenantAuthMiddleware, "_has_auth_infrastructure", return_value=False):
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "default"


def test_invalid_api_key(client):
    """无效 API Key 返回 401"""
    with patch.object(TenantAuthMiddleware, "_has_auth_infrastructure", return_value=True):
        response = client.get("/test", headers={"X-API-Key": "invalid_key"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "INVALID_API_KEY"


def test_generate_api_key():
    """测试 API Key 生成"""
    plain, hashed, prefix = generate_api_key()
    assert plain.startswith("qk_")
    assert len(plain) > 20
    assert len(hashed) == 64  # SHA256 hex
    assert prefix == plain[:8]


def test_hash_api_key():
    """测试 API Key 哈希"""
    key = "qk_test123"
    hashed = hash_api_key(key)
    assert len(hashed) == 64
    assert hashed == hash_api_key(key)


def test_tenant_context():
    """测试租户上下文"""
    TenantContext.clear()
    assert TenantContext.get_tenant() is None

    # Create a mock tenant
    quota = QuotaConfig()
    tenant = Tenant(
        id="test-tenant-id",
        name="test-tenant",
        api_key_hash="hash123",
        api_key_prefix="prefix123",
        status=TenantStatus.ACTIVE,
        quota=quota,
    )

    TenantContext.set_tenant(tenant)
    assert TenantContext.get_tenant() == tenant

    TenantContext.clear()
    assert TenantContext.get_tenant() is None


def test_generate_api_key_uniqueness():
    """测试生成的 API Key 唯一性"""
    keys = set()
    for _ in range(100):
        plain, hashed, prefix = generate_api_key()
        # plain key should be unique
        assert plain not in keys
        keys.add(plain)
        # hashed should be unique
        assert hashed not in keys
        keys.add(hashed)


def test_tenant_status_suspended(client):
    """暂停租户返回 403"""
    with patch("quantumflow.api.middleware.auth.TenantAuthMiddleware._authenticate") as mock_auth:
        quota = QuotaConfig()
        suspended_tenant = Tenant(
            id="suspended-tenant",
            name="Suspended Tenant",
            api_key_hash="hash123",
            api_key_prefix="prefix123",
            status=TenantStatus.SUSPENDED,
            quota=quota,
        )
        mock_auth.return_value = suspended_tenant

        response = client.get("/test", headers={"X-API-Key": "qk_validkey123"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "TENANT_SUSPENDED"


def test_tenant_status_deleted(client):
    """删除租户返回 403"""
    with patch("quantumflow.api.middleware.auth.TenantAuthMiddleware._authenticate") as mock_auth:
        quota = QuotaConfig()
        deleted_tenant = Tenant(
            id="deleted-tenant",
            name="Deleted Tenant",
            api_key_hash="hash123",
            api_key_prefix="prefix123",
            status=TenantStatus.DELETED,
            quota=quota,
        )
        mock_auth.return_value = deleted_tenant

        response = client.get("/test", headers={"X-API-Key": "qk_validkey123"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "TENANT_DELETED"


def test_exempt_paths():
    """测试豁免路径"""
    middleware = TenantAuthMiddleware(FastAPI())

    # 这些路径应该豁免
    assert middleware._is_exempt("/")
    assert middleware._is_exempt("/health")
    assert middleware._is_exempt("/api/v1/health")
    assert middleware._is_exempt("/api/v1/health/ready")
    assert middleware._is_exempt("/api/v1/health/live")
    assert middleware._is_exempt("/docs")
    assert middleware._is_exempt("/openapi.json")
    assert middleware._is_exempt("/redoc")

    # 这些路径不应该豁免（包括租户管理端点，需要认证）
    assert not middleware._is_exempt("/api/v1/tenants")
    assert not middleware._is_exempt("/api/v1/tenants/create")
    assert not middleware._is_exempt("/api/v1/inference")
    assert not middleware._is_exempt("/api/v1/models")
    assert not middleware._is_exempt("/test")


def test_successful_auth_with_mock_tenant(client):
    """成功认证并获取租户信息"""
    with patch("quantumflow.api.middleware.auth.TenantAuthMiddleware._authenticate") as mock_auth:
        quota = QuotaConfig()
        active_tenant = Tenant(
            id="active-tenant-id",
            name="Active Tenant",
            api_key_hash="hash123",
            api_key_prefix="prefix123",
            status=TenantStatus.ACTIVE,
            quota=quota,
        )
        mock_auth.return_value = active_tenant

        response = client.get("/test", headers={"X-API-Key": "qk_validkey123"})
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "active-tenant-id"


def test_has_auth_infrastructure_with_cache():
    """有缓存租户时返回 True"""
    tenant = Tenant(
        id="cached",
        name="Cached",
        api_key_hash="hash_cached",
        api_key_prefix="prefix",
        status=TenantStatus.ACTIVE,
        quota=QuotaConfig(),
    )
    with _cache_lock:
        _tenant_cache["hash_cached"] = tenant
    try:
        middleware = TenantAuthMiddleware(FastAPI())
        assert middleware._has_auth_infrastructure() is True
    finally:
        with _cache_lock:
            _tenant_cache.clear()


def test_has_auth_infrastructure_empty():
    """无缓存无 Redis 时返回 False"""
    with _cache_lock:
        _tenant_cache.clear()
    middleware = TenantAuthMiddleware(FastAPI())
    assert middleware._has_auth_infrastructure() is False


def test_default_tenant():
    """默认租户创建"""
    middleware = TenantAuthMiddleware(FastAPI())
    tenant = middleware._default_tenant()
    assert tenant.id == "default"
    assert tenant.name == "默认租户"
    assert tenant.status == TenantStatus.ACTIVE


def test_authenticate_cache_hit():
    """缓存命中返回租户"""
    quota = QuotaConfig()
    tenant = Tenant(
        id="cached-tenant",
        name="Cached Tenant",
        api_key_hash=hash_api_key("qk_test_key_12345"),
        api_key_prefix="qk_test_",
        status=TenantStatus.ACTIVE,
        quota=quota,
    )
    tenant._cache_time = time.time()
    key_hash = tenant.api_key_hash
    with _cache_lock:
        _tenant_cache[key_hash] = tenant

    import asyncio
    middleware = TenantAuthMiddleware(FastAPI())
    result = asyncio.run(middleware._authenticate("qk_test_key_12345"))
    assert result is not None
    assert result.id == "cached-tenant"


def test_authenticate_cache_expired():
    """缓存过期返回 None"""
    quota = QuotaConfig()
    tenant = Tenant(
        id="expired-tenant",
        name="Expired",
        api_key_hash=hash_api_key("qk_expired_key"),
        api_key_prefix="qk_expi_",
        status=TenantStatus.ACTIVE,
        quota=quota,
    )
    tenant._cache_time = time.time() - 600  # 10 分钟前
    key_hash = tenant.api_key_hash
    with _cache_lock:
        _tenant_cache[key_hash] = tenant

    import asyncio
    middleware = TenantAuthMiddleware(FastAPI())
    result = asyncio.run(middleware._authenticate("qk_expired_key"))
    assert result is None


def test_authenticate_from_redis():
    """从 Redis 加载租户"""
    import asyncio
    middleware = TenantAuthMiddleware(FastAPI())

    with _cache_lock:
        _tenant_cache.clear()

    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = {
        b"id": b"redis-tenant",
        b"name": b"Redis Tenant",
        b"api_key_hash": hash_api_key("qk_redis_key").encode(),
        b"api_key_prefix": b"qk_redis",
        b"status": b"active",
        b"quota_requests_per_minute": b"60",
        b"quota_requests_per_day": b"10000",
        b"quota_max_tokens": b"8192",
        b"quota_gpu_memory": b"4096",
        b"quota_concurrent": b"10",
        b"priority": b"5",
    }

    with patch("quantumflow.api.middleware.auth._get_redis", return_value=mock_redis):
        result = asyncio.run(middleware._authenticate("qk_redis_key"))
        assert result is not None
        assert result.id == "redis-tenant"
        assert result.name == "Redis Tenant"


def test_deserialize_tenant():
    """直接测试反序列化"""
    middleware = TenantAuthMiddleware(FastAPI())
    data = {
        b"id": b"deserialized",
        b"name": b"Deserialized",
        b"api_key_hash": b"hash123",
        b"api_key_prefix": b"prefix12",
        b"status": b"active",
        b"priority": b"7",
        b"quota_requests_per_minute": b"30",
        b"quota_requests_per_day": b"5000",
        b"quota_max_tokens": b"4096",
        b"quota_gpu_memory": b"2048",
        b"quota_concurrent": b"5",
    }
    tenant = middleware._deserialize_tenant(data)
    assert tenant.id == "deserialized"
    assert tenant.name == "Deserialized"
    assert tenant.priority == 7
    assert tenant.quota.requests_per_minute == 30
    assert tenant.quota.concurrent_requests == 5
    assert hasattr(tenant, "_cache_time")


def test_deserialize_tenant_defaults():
    """反序列化缺少字段时使用默认值"""
    middleware = TenantAuthMiddleware(FastAPI())
    data = {
        b"id": b"minimal",
        b"name": b"Minimal",
        b"api_key_hash": b"minhash",
        b"api_key_prefix": b"minimal_",
        b"status": b"active",
    }
    tenant = middleware._deserialize_tenant(data)
    assert tenant.quota.requests_per_minute == 60
    assert tenant.quota.concurrent_requests == 10
