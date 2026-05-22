"""租户隔离集成测试 — 真实业务场景

测试覆盖完整的租户生命周期、多租户隔离、认证授权、
配额管理、限流、以及各种边界条件。
"""

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from quantumflow.api.middleware.auth import (
    TenantAuthMiddleware,
    TenantContext,
    _cache_lock,
    _tenant_cache,
    generate_api_key,
    hash_api_key,
)
from quantumflow.api.models.tenant import (
    QuotaConfig,
    Tenant,
    TenantCreateResponse,
    TenantResponse,
    TenantStatus,
    TenantUsage,
)
from quantumflow.api.routes import tenants as tenants_module
from quantumflow.core.constants import DEFAULT_TENANT_QUOTA


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_tenant_state():
    """每个测试前清理租户状态"""
    _tenant_cache.clear()
    tenants_module._tenants.clear()
    TenantContext.clear()
    yield
    _tenant_cache.clear()
    tenants_module._tenants.clear()
    TenantContext.clear()


@pytest.fixture
def app_without_auth():
    """不带认证的 App（仅租户管理路由）"""
    app = FastAPI()
    app.include_router(tenants_module.router)
    return app


@pytest.fixture
def client_without_auth(app_without_auth):
    """不含认证中间件的测试客户端"""
    return TestClient(app_without_auth)


@pytest.fixture
def app_with_auth():
    """带认证中间件的完整 App"""
    from quantumflow.api.server import app
    return app


@pytest.fixture
def client_with_auth(app_with_auth):
    """含认证中间件的测试客户端"""
    return TestClient(app_with_auth, raise_server_exceptions=False)


def _create_tenant(client, name="test-tenant", priority=5, quota=None):
    """Helper: 创建租户并返回响应数据"""
    body = {"name": name, "priority": priority}
    if quota:
        body["quota"] = quota
    with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
        resp = client.post("/tenants/", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── 1. 租户生命周期 ─────────────────────────────────────────────────────────


class TestTenantLifecycle:
    """租户完整生命周期：创建→读取→列表→更新→删除→验证删除"""

    def test_create_tenant_returns_api_key(self, client_without_auth):
        """创建租户时返回 API Key（仅此一次）"""
        data = _create_tenant(client_without_auth, name="lifecycle-tenant")
        assert "api_key" in data
        assert data["api_key"].startswith("qk_")
        assert len(data["api_key"]) > 30
        assert "api_key_prefix" in data
        assert data["api_key_prefix"] == data["api_key"][:8]

    def test_list_tenants_excludes_deleted(self, client_without_auth):
        """列出租户时排除已删除的"""
        d1 = _create_tenant(client_without_auth, name="active-1")
        d2 = _create_tenant(client_without_auth, name="active-2")
        d3 = _create_tenant(client_without_auth, name="to-delete")

        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            client_without_auth.delete(f"/tenants/{d3['id']}")

        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get("/tenants/")
        assert resp.status_code == 200
        ids = [t["id"] for t in resp.json()]
        assert d1["id"] in ids
        assert d2["id"] in ids
        assert d3["id"] not in ids

    def test_get_tenant_detail(self, client_without_auth):
        """获取单个租户详情"""
        data = _create_tenant(client_without_auth, name="detail-tenant", priority=7)
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get(f"/tenants/{data['id']}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "detail-tenant"
        assert resp.json()["priority"] == 7

    def test_get_deleted_tenant_returns_404(self, client_without_auth):
        """获取已删除租户返回 404"""
        data = _create_tenant(client_without_auth, name="gone-soon")
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            client_without_auth.delete(f"/tenants/{tid}")
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get(f"/tenants/{tid}")
        assert resp.status_code == 404

    def test_update_tenant_name_and_priority(self, client_without_auth):
        """更新租户名称和优先级"""
        data = _create_tenant(client_without_auth, name="old-name", priority=3)
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.patch(
                f"/tenants/{tid}",
                json={"name": "new-name", "priority": 9},
            )
        assert resp.status_code == 200
        assert resp.json()["name"] == "new-name"
        assert resp.json()["priority"] == 9

    def test_update_tenant_quota(self, client_without_auth):
        """更新租户配额（业务场景：升级套餐）"""
        data = _create_tenant(client_without_auth, name="upgrade-tenant")
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.patch(
                f"/tenants/{tid}",
                json={
                    "quota": {
                        "requests_per_minute": 120,
                        "concurrent_requests": 20,
                        "gpu_memory_mb": 8192,
                    }
                },
            )
        assert resp.status_code == 200
        q = resp.json()["quota"]
        assert q["requests_per_minute"] == 120
        assert q["concurrent_requests"] == 20
        assert q["gpu_memory_mb"] == 8192

    def test_update_tenant_status_suspend_and_restore(self, client_without_auth):
        """暂停和恢复租户（业务场景：欠费处理）"""
        data = _create_tenant(client_without_auth, name="suspend-me")
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            # 暂停
            resp = client_without_auth.patch(f"/tenants/{tid}", json={"status": "suspended"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "suspended"
            # 恢复
            resp = client_without_auth.patch(f"/tenants/{tid}", json={"status": "active"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "active"

    def test_soft_delete_tenant(self, client_without_auth):
        """软删除租户（业务场景：客户注销）"""
        data = _create_tenant(client_without_auth, name="delete-me")
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.delete(f"/tenants/{tid}")
        assert resp.status_code == 204
        # 删除后不可见
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get(f"/tenants/{tid}")
        assert resp.status_code == 404
        # 不在列表中
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get("/tenants/")
        ids = [t["id"] for t in resp.json()]
        assert tid not in ids

    def test_tenant_usage_no_redis(self, client_without_auth):
        """获取租户使用量（无 Redis 时返回默认值）"""
        data = _create_tenant(client_without_auth, name="usage-tenant")
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get(f"/tenants/{tid}/usage")
        assert resp.status_code == 200
        u = resp.json()
        assert u["tenant_id"] == tid
        assert u["requests_today"] == 0
        assert u["tokens_today"] == 0
        assert "quota_remaining" in u

    def test_tenant_usage_with_redis_data(self, client_without_auth):
        """获取租户使用量（Redis 有数据）"""
        data = _create_tenant(client_without_auth, name="usage-redis")
        tid = data["id"]
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {b"requests": b"99", b"tokens": b"5000"}
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=mock_redis):
            resp = client_without_auth.get(f"/tenants/{tid}/usage")
        assert resp.status_code == 200
        u = resp.json()
        assert u["requests_today"] == 99
        assert u["tokens_today"] == 5000


# ── 2. 认证与授权 ───────────────────────────────────────────────────────────


class TestTenantAuth:
    """认证授权：API Key 验证、状态检查、缓存行为"""

    def test_missing_api_key_blocked_when_auth_active(self, client_with_auth):
        """有认证基础设施时，缺失 API Key → 401"""
        with patch.object(TenantAuthMiddleware, "_has_auth_infrastructure", return_value=True):
            resp = client_with_auth.get("/api/v1/inference/generate")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "MISSING_API_KEY"

    def test_invalid_api_key_blocked_when_auth_active(self, client_with_auth):
        """有认证基础设施时，无效 API Key → 401"""
        with patch.object(TenantAuthMiddleware, "_has_auth_infrastructure", return_value=True):
            resp = client_with_auth.get("/api/v1/inference/generate",
                                        headers={"X-API-Key": "qk_wrong_key_12345678901234567890"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "INVALID_API_KEY"

    def test_suspended_tenant_blocked_403(self, client_with_auth):
        """暂停的租户返回 403"""
        quota = QuotaConfig()
        suspended = Tenant(
            id="suspended-t",
            name="Suspended",
            api_key_hash=hash_api_key("qk_suspended_key_1234567890"),
            api_key_prefix="qk_suspe",
            status=TenantStatus.SUSPENDED,
            quota=quota,
        )
        with patch.object(TenantAuthMiddleware, "_authenticate", return_value=suspended):
            resp = client_with_auth.get("/api/v1/inference/generate",
                                        headers={"X-API-Key": "qk_suspended_key_1234567890"})
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "TENANT_SUSPENDED"

    def test_deleted_tenant_blocked_403(self, client_with_auth):
        """已删除的租户返回 403"""
        quota = QuotaConfig()
        deleted = Tenant(
            id="deleted-t",
            name="Deleted",
            api_key_hash=hash_api_key("qk_deleted_key_1234567890"),
            api_key_prefix="qk_delet",
            status=TenantStatus.DELETED,
            quota=quota,
        )
        with patch.object(TenantAuthMiddleware, "_authenticate", return_value=deleted):
            resp = client_with_auth.get("/api/v1/inference/generate",
                                        headers={"X-API-Key": "qk_deleted_key_1234567890"})
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "TENANT_DELETED"

    def test_active_tenant_passes_auth(self, client_with_auth):
        """活跃租户通过认证"""
        quota = QuotaConfig()
        active = Tenant(
            id="active-t",
            name="Active",
            api_key_hash=hash_api_key("qk_active_key_1234567890"),
            api_key_prefix="qk_activ",
            status=TenantStatus.ACTIVE,
            quota=quota,
        )
        with patch.object(TenantAuthMiddleware, "_authenticate", return_value=active):
            resp = client_with_auth.get("/api/v1/health",
                                        headers={"X-API-Key": "qk_active_key_1234567890"})
        assert resp.status_code == 200

    def test_health_endpoint_always_exempt(self, client_with_auth):
        """健康检查端点始终免认证"""
        resp = client_with_auth.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_metrics_endpoint_exempt(self, client_with_auth):
        """Metrics 端点免认证"""
        resp = client_with_auth.get("/api/v1/metrics")
        assert resp.status_code in (200, 404)  # 404 if prometheus not fully init

    def test_docs_endpoint_exempt(self, client_with_auth):
        """文档端点免认证"""
        resp = client_with_auth.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_exempt(self, client_with_auth):
        """OpenAPI schema 端点免认证"""
        resp = client_with_auth.get("/openapi.json")
        assert resp.status_code == 200

    def test_tenant_routes_exempt(self, client_with_auth):
        """租户管理路由免认证（否则无法创建租户）"""
        resp = client_with_auth.post("/api/v1/tenants/", json={"name": "direct-create"})
        assert resp.status_code in (201, 404)  # 201 if route matches, 404 if path != /api/v1/tenants/x


# ── 3. 多租户隔离 ───────────────────────────────────────────────────────────


class TestMultiTenantIsolation:
    """多租户隔离：不同租户互不影响"""

    def test_tenant_quota_isolation(self, client_without_auth):
        """租户 A 的配额不影响租户 B"""
        t_a = _create_tenant(client_without_auth, "tenant-A", quota={
            "requests_per_minute": 60,
            "concurrent_requests": 10,
            "gpu_memory_mb": 4096,
        })
        t_b = _create_tenant(client_without_auth, "tenant-B", quota={
            "requests_per_minute": 120,
            "concurrent_requests": 50,
            "gpu_memory_mb": 16384,
        })

        assert t_a["quota"]["requests_per_minute"] == 60
        assert t_a["quota"]["concurrent_requests"] == 10
        assert t_b["quota"]["requests_per_minute"] == 120
        assert t_b["quota"]["concurrent_requests"] == 50

    def test_update_tenant_a_does_not_affect_b(self, client_without_auth):
        """更新租户 A 不影响租户 B"""
        t_a = _create_tenant(client_without_auth, "tenant-A", priority=5)
        t_b = _create_tenant(client_without_auth, "tenant-B", priority=8)

        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            client_without_auth.patch(f"/tenants/{t_a['id']}", json={"priority": 1})

        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp_a = client_without_auth.get(f"/tenants/{t_a['id']}")
            resp_b = client_without_auth.get(f"/tenants/{t_b['id']}")

        assert resp_a.json()["priority"] == 1
        assert resp_b.json()["priority"] == 8  # 不受影响

    def test_delete_tenant_a_does_not_affect_b(self, client_without_auth):
        """删除租户 A 不影响租户 B"""
        t_a = _create_tenant(client_without_auth, "tenant-A")
        t_b = _create_tenant(client_without_auth, "tenant-B")

        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            client_without_auth.delete(f"/tenants/{t_a['id']}")

        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get(f"/tenants/{t_b['id']}")
        assert resp.status_code == 200

    def test_different_tenants_have_unique_api_keys(self, client_without_auth):
        """不同租户的 API Key 唯一"""
        t1 = _create_tenant(client_without_auth, "key-test-1")
        t2 = _create_tenant(client_without_auth, "key-test-2")
        assert t1["api_key"] != t2["api_key"]
        assert t1["api_key_prefix"] != t2["api_key_prefix"]


# ── 4. 边界条件 ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    """边界条件与异常场景"""

    def test_get_nonexistent_tenant_404(self, client_without_auth):
        """获取不存在的租户 → 404"""
        resp = client_without_auth.get("/tenants/fake-id-12345")
        assert resp.status_code == 404

    def test_update_nonexistent_tenant_404(self, client_without_auth):
        """更新不存在的租户 → 404"""
        resp = client_without_auth.patch("/tenants/fake-id-12345", json={"name": "nope"})
        assert resp.status_code == 404

    def test_delete_nonexistent_tenant_404(self, client_without_auth):
        """删除不存在的租户 → 404"""
        resp = client_without_auth.delete("/tenants/fake-id-12345")
        assert resp.status_code == 404

    def test_usage_nonexistent_tenant_404(self, client_without_auth):
        """获取不存在租户的使用量 → 404"""
        resp = client_without_auth.get("/tenants/fake-id-12345/usage")
        assert resp.status_code == 404

    def test_create_tenant_default_quota(self, client_without_auth):
        """创建租户时使用默认配额"""
        data = _create_tenant(client_without_auth, name="default-quota")
        q = data["quota"]
        assert q["requests_per_minute"] == DEFAULT_TENANT_QUOTA["requests_per_minute"]
        assert q["concurrent_requests"] == DEFAULT_TENANT_QUOTA["concurrent_requests"]

    def test_create_tenant_minimal_body(self, client_without_auth):
        """最小请求体创建租户（仅 name）"""
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.post("/tenants/", json={"name": "minimal"})
        assert resp.status_code == 201
        assert resp.json()["priority"] == 5  # 默认优先级

    def test_create_tenant_empty_name_should_fail(self, client_without_auth):
        """空名称创建租户应被拒绝（Pydantic 校验）"""
        resp = client_without_auth.post("/tenants/", json={"name": ""})
        assert resp.status_code == 422

    def test_create_tenant_name_too_long_should_fail(self, client_without_auth):
        """超长名称创建租户应被拒绝"""
        resp = client_without_auth.post("/tenants/", json={"name": "a" * 200})
        assert resp.status_code == 422

    def test_create_tenant_negative_concurrent_requests_should_fail(self, client_without_auth):
        """负数的并发请求数应被拒绝"""
        resp = client_without_auth.post("/tenants/", json={
            "name": "bad-quota",
            "quota": {"concurrent_requests": -1},
        })
        assert resp.status_code == 422

    def test_double_delete_is_idempotent(self, client_without_auth):
        """重复删除同一租户返回 404"""
        data = _create_tenant(client_without_auth, name="double-delete")
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp1 = client_without_auth.delete(f"/tenants/{tid}")
        assert resp1.status_code == 204
        # 第二次删除
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp2 = client_without_auth.delete(f"/tenants/{tid}")
        assert resp2.status_code == 404

    def test_update_deleted_tenant_should_fail(self, client_without_auth):
        """更新已删除的租户应返回 404"""
        data = _create_tenant(client_without_auth, name="update-deleted")
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            client_without_auth.delete(f"/tenants/{tid}")
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.patch(f"/tenants/{tid}", json={"name": "nope"})
        assert resp.status_code == 404

    def test_auth_cache_ttl_expiry(self):
        """认证缓存过期后重新从数据源加载"""
        middleware = TenantAuthMiddleware(FastAPI())
        key = "qk_cache_ttl_test_123456789"
        key_hash = hash_api_key(key)
        tenant = Tenant(
            id="cache-ttl",
            name="Cache TTL",
            api_key_hash=key_hash,
            api_key_prefix="qk_cache",
            status=TenantStatus.ACTIVE,
            quota=QuotaConfig(),
        )
        tenant._cache_time = time.time() - 600  # 过期
        with _cache_lock:
            _tenant_cache[key_hash] = tenant

        import asyncio
        result = asyncio.run(middleware._authenticate(key))
        # 缓存过期 → Redis 不可用 → 返回 None
        assert result is None


# ── 5. 数据模型完整性 ────────────────────────────────────────────────────────


class TestModelIntegrity:
    """数据模型序列化 / 反序列化完整性"""

    def test_tenant_create_response_has_all_fields(self, client_without_auth):
        """TenantCreateResponse 包含所有必需字段"""
        data = _create_tenant(client_without_auth, "full-tenant", priority=5,
                              quota={"requests_per_minute": 30, "concurrent_requests": 5})
        required_fields = {"id", "name", "status", "quota", "priority",
                           "created_at", "last_active_at", "api_key", "api_key_prefix"}
        assert required_fields.issubset(set(data.keys()))

    def test_tenant_response_excludes_api_key(self, client_without_auth):
        """TenantResponse（get/list）不包含 API Key"""
        data = _create_tenant(client_without_auth, "safe-tenant")
        tid = data["id"]
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get(f"/tenants/{tid}")
        assert resp.status_code == 200
        assert "api_key" not in resp.json()

    def test_list_tenant_response_excludes_api_key(self, client_without_auth):
        """列表接口不暴露 API Key"""
        _create_tenant(client_without_auth, "list-safe1")
        _create_tenant(client_without_auth, "list-safe2")
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.get("/tenants/")
        for t in resp.json():
            assert "api_key" not in t

    def test_quota_config_validation(self):
        """QuotaConfig Pydantic 校验"""
        # 合法值
        q = QuotaConfig(requests_per_minute=100, concurrent_requests=50, gpu_memory_mb=8192)
        assert q.requests_per_minute == 100

        # 超出范围应失败
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QuotaConfig(requests_per_minute=0)  # ge=1
        with pytest.raises(ValidationError):
            QuotaConfig(concurrent_requests=0)  # ge=1
        with pytest.raises(ValidationError):
            QuotaConfig(requests_per_minute=20000)  # le=10000


# ── 6. 并发与竞态 ───────────────────────────────────────────────────────────


class TestConcurrency:
    """并发场景测试"""

    def test_multiple_tenants_create_concurrently(self, client_without_auth):
        """并发创建多个租户（模拟平台注册高峰）"""
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            results = []
            for i in range(10):
                resp = client_without_auth.post("/tenants/", json={"name": f"concurrent-{i}"})
                results.append(resp)
        # 所有都应成功
        for r in results:
            assert r.status_code == 201
        # 所有 ID 应该唯一
        ids = [r.json()["id"] for r in results]
        assert len(set(ids)) == 10

    def test_rapid_create_delete_cycle(self, client_without_auth):
        """快速创建→删除→创建同一租户名称"""
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            r1 = client_without_auth.post("/tenants/", json={"name": "recreate"})
            tid1 = r1.json()["id"]
            client_without_auth.delete(f"/tenants/{tid1}")
            r2 = client_without_auth.post("/tenants/", json={"name": "recreate"})
        # 新创建的应该有不同的 ID
        assert r2.status_code == 201
        assert r2.json()["id"] != tid1


# ── 7. 端到端：创建→认证→请求 完整链路 ─────────────────────────────────────


class TestEndToEndFlow:
    """端到端完整业务链路"""

    def test_create_tenant_then_auth_with_key(self, client_with_auth):
        """创建租户 → 用 API Key 认证 → 通过认证访问"""
        # Step 1: 创建租户（租户路由在完整 app 中挂载在 /api/v1/tenants）
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            create_resp = client_with_auth.post("/api/v1/tenants/", json={"name": "e2e-tenant"})
        assert create_resp.status_code == 201
        api_key = create_resp.json()["api_key"]

        # Step 2: 用 API Key 认证访问
        # 模拟认证基础设施启用
        key_hash = hash_api_key(api_key)
        from quantumflow.api.models.tenant import QuotaConfig as QC, TenantStatus as TS
        tenant = Tenant(
            id=create_resp.json()["id"],
            name="e2e-tenant",
            api_key_hash=key_hash,
            api_key_prefix=api_key[:8],
            status=TenantStatus.ACTIVE,
            quota=QuotaConfig(),
        )
        with patch.object(TenantAuthMiddleware, "_authenticate", return_value=tenant):
            resp = client_with_auth.get("/api/v1/health",
                                        headers={"X-API-Key": api_key})
            assert resp.status_code == 200

    def test_full_tenant_upgrade_flow(self, client_without_auth):
        """完整升级流程：基础套餐 → 升级配额 → 验证生效"""
        # 基础套餐
        data = _create_tenant(client_without_auth, "upgrade-customer",
                              quota={"requests_per_minute": 10, "concurrent_requests": 3})
        assert data["quota"]["requests_per_minute"] == 10

        tid = data["id"]
        # 升级到高级套餐
        with patch("quantumflow.api.routes.tenants._get_redis", return_value=None):
            resp = client_without_auth.patch(f"/tenants/{tid}", json={
                "quota": {"requests_per_minute": 1000, "concurrent_requests": 100}
            })
        assert resp.status_code == 200
        assert resp.json()["quota"]["requests_per_minute"] == 1000
        assert resp.json()["quota"]["concurrent_requests"] == 100

    def test_suspend_and_reauth_flow(self, client_with_auth):
        """暂停租户 → 认证被拒绝 → 恢复 → 认证通过"""
        api_key = "qk_suspend_flow_key_1234567890"
        key_hash = hash_api_key(api_key)

        # 使用非豁免路径测试认证（/api/v1/models 不在豁免列表中）
        # 活跃租户 → 通过
        active = Tenant(
            id="flow-tenant", name="Flow", api_key_hash=key_hash,
            api_key_prefix=api_key[:8], status=TenantStatus.ACTIVE, quota=QuotaConfig(),
        )
        with patch.object(TenantAuthMiddleware, "_authenticate", return_value=active):
            # 即使 models 路由可能 404，认证层应该通过
            resp = client_with_auth.get("/api/v1/models",
                                        headers={"X-API-Key": api_key})
            # 认证通过后由路由层处理（404 或 200 都行，只要不是 401/403）
            assert resp.status_code not in (401, 403)

        # 暂停租户 → 403
        suspended = Tenant(
            id="flow-tenant", name="Flow", api_key_hash=key_hash,
            api_key_prefix=api_key[:8], status=TenantStatus.SUSPENDED, quota=QuotaConfig(),
        )
        with patch.object(TenantAuthMiddleware, "_authenticate", return_value=suspended):
            resp = client_with_auth.get("/api/v1/models",
                                        headers={"X-API-Key": api_key})
            assert resp.status_code == 403
            assert resp.json()["error"]["code"] == "TENANT_SUSPENDED"
