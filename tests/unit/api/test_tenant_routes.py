"""租户管理 API 路由测试"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def clear_tenants():
    """Clear in-memory tenant storage before each test"""
    from quantumflow.api.routes import tenants
    tenants._tenants.clear()
    yield
    tenants._tenants.clear()


@pytest.fixture
def app():
    from quantumflow.api.routes import tenants
    app = FastAPI()
    app.include_router(tenants.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_create_tenant(client):
    """测试创建租户"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None  # 禁用 Redis
        response = client.post(
            "/tenants/",
            json={"name": "test-tenant", "priority": 8}
        )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-tenant"
    assert data["priority"] == 8
    assert "id" in data


def test_list_tenants(client):
    """测试列出租户"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None
        # 先创建一个
        client.post("/tenants/", json={"name": "tenant-1"})
        response = client.get("/tenants/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_tenant_not_found(client):
    """测试获取不存在的租户"""
    response = client.get("/tenants/nonexistent-id")
    assert response.status_code == 404


def test_update_tenant(client):
    """测试更新租户"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None
        # 创建
        create_resp = client.post("/tenants/", json={"name": "original"})
        tenant_id = create_resp.json()["id"]

        # 更新
        response = client.patch(
            f"/tenants/{tenant_id}",
            json={"name": "updated", "priority": 10}
        )

    assert response.status_code == 200
    assert response.json()["name"] == "updated"
    assert response.json()["priority"] == 10


def test_delete_tenant(client):
    """测试删除租户"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None
        # 创建
        create_resp = client.post("/tenants/", json={"name": "to-delete"})
        tenant_id = create_resp.json()["id"]

        # 删除
        response = client.delete(f"/tenants/{tenant_id}")
        assert response.status_code == 204

        # 验证已删除
        response = client.get(f"/tenants/{tenant_id}")
        assert response.status_code == 404


def test_get_tenant_usage(client):
    """测试获取租户使用量"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None
        # Create tenant
        create_resp = client.post("/tenants/", json={"name": "usage-tenant"})
        tenant_id = create_resp.json()["id"]

        # Get usage
        response = client.get(f"/tenants/{tenant_id}/usage")
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == tenant_id
        assert "requests_today" in data


def test_get_tenant_success(client):
    """测试成功获取租户"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None
        create_resp = client.post("/tenants/", json={"name": "success-tenant"})
        tenant_id = create_resp.json()["id"]

        response = client.get(f"/tenants/{tenant_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "success-tenant"


def test_update_nonexistent_tenant(client):
    """测试更新不存在的租户返回 404"""
    response = client.patch("/tenants/nonexistent", json={"name": "updated"})
    assert response.status_code == 404


def test_update_tenant_quota(client):
    """测试更新租户配额"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None
        create_resp = client.post("/tenants/", json={"name": "quota-tenant"})
        tenant_id = create_resp.json()["id"]

        response = client.patch(
            f"/tenants/{tenant_id}",
            json={"quota": {"requests_per_minute": 30, "concurrent_requests": 5}}
        )
        assert response.status_code == 200
        assert response.json()["quota"]["requests_per_minute"] == 30
        assert response.json()["quota"]["concurrent_requests"] == 5


def test_update_tenant_status(client):
    """测试更新租户状态"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None
        create_resp = client.post("/tenants/", json={"name": "status-tenant"})
        tenant_id = create_resp.json()["id"]

        response = client.patch(
            f"/tenants/{tenant_id}",
            json={"status": "suspended"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "suspended"


def test_delete_nonexistent_tenant(client):
    """测试删除不存在的租户返回 404"""
    response = client.delete("/tenants/nonexistent")
    assert response.status_code == 404


def test_get_usage_nonexistent_tenant(client):
    """测试获取不存在租户的使用量返回 404"""
    response = client.get("/tenants/nonexistent/usage")
    assert response.status_code == 404


def test_create_tenant_with_redis(client):
    """测试创建租户并同步到 Redis（含 ID 索引）"""
    mock_redis = MagicMock()
    with patch("quantumflow.api.routes.tenants._get_redis", return_value=mock_redis):
        response = client.post(
            "/tenants/",
            json={"name": "redis-tenant", "priority": 7}
        )
    assert response.status_code == 201
    mock_redis.hset.assert_called_once()
    mock_redis.set.assert_called_once()
    mock_redis.sadd.assert_called_once()


def test_serialize_tenant():
    """测试租户序列化"""
    from quantumflow.api.models.tenant import Tenant, QuotaConfig
    from quantumflow.api.routes.tenants import _serialize_tenant

    tenant = Tenant(
        name="serialize-test",
        api_key_hash="abc123",
        api_key_prefix="qk_abc12",
        quota=QuotaConfig(requests_per_minute=30, concurrent_requests=5),
        priority=6,
    )
    result = _serialize_tenant(tenant)
    assert result["name"] == "serialize-test"
    assert result["api_key_hash"] == "abc123"
    assert result["quota_requests_per_minute"] == "30"
    assert result["quota_concurrent"] == "5"
    assert result["priority"] == "6"
    assert "created_at" in result
    assert "updated_at" in result


def test_create_tenant_with_custom_quota(client):
    """测试创建租户时自定义配额"""
    with patch("quantumflow.api.routes.tenants._get_redis") as mock_redis:
        mock_redis.return_value = None
        response = client.post(
            "/tenants/",
            json={
                "name": "custom-quota",
                "quota": {
                    "requests_per_minute": 120,
                    "concurrent_requests": 20,
                    "gpu_memory_mb": 8192,
                }
            }
        )
    assert response.status_code == 201
    data = response.json()
    assert data["quota"]["requests_per_minute"] == 120
    assert data["quota"]["concurrent_requests"] == 20
    assert data["quota"]["gpu_memory_mb"] == 8192


def test_get_tenant_usage_with_redis(client):
    """测试从 Redis 获取使用量数据"""
    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = {
        b"requests": b"42",
        b"tokens": b"12345",
    }
    with patch("quantumflow.api.routes.tenants._get_redis", return_value=mock_redis):
        create_resp = client.post("/tenants/", json={"name": "usage-redis"})
        tenant_id = create_resp.json()["id"]

        response = client.get(f"/tenants/{tenant_id}/usage")
        assert response.status_code == 200
        data = response.json()
        assert data["requests_today"] == 42
        assert data["tokens_today"] == 12345


def test_update_tenant_with_redis(client):
    """测试更新租户并同步到 Redis（含 ID 索引）"""
    mock_redis = MagicMock()
    with patch("quantumflow.api.routes.tenants._get_redis", return_value=mock_redis):
        create_resp = client.post("/tenants/", json={"name": "redis-update"})
        tenant_id = create_resp.json()["id"]

        # Reset mock to clear create calls
        mock_redis.hset.reset_mock()
        mock_redis.set.reset_mock()
        mock_redis.sadd.reset_mock()

        response = client.patch(
            f"/tenants/{tenant_id}",
            json={"name": "redis-updated", "status": "suspended"}
        )
        assert response.status_code == 200
        mock_redis.hset.assert_called()
        mock_redis.set.assert_called()
        mock_redis.sadd.assert_called()


def test_delete_tenant_with_redis(client):
    """测试删除租户并同步到 Redis"""
    mock_redis = MagicMock()
    mock_redis.hset = MagicMock()
    with patch("quantumflow.api.routes.tenants._get_redis", return_value=mock_redis):
        create_resp = client.post("/tenants/", json={"name": "redis-delete"})
        tenant_id = create_resp.json()["id"]

        response = client.delete(f"/tenants/{tenant_id}")
        assert response.status_code == 204
        # delete 调用 hset 来更新 status
        mock_redis.hset.assert_called()


def test__get_redis_no_redis():
    """测试 _get_redis 函数（Redis 不可用时返回 None）"""
    from quantumflow.api.routes.tenants import _get_redis
    result = _get_redis()
    assert result is None
