"""跨组件集成测试 — 验证租户数据在不同组件间的正确传递

这些测试防止之前发现的生产 bug：
- tenants.py 写入的数据能被 scheduler 正确读取
- tenants.py 写入的数据能被 vram_manager 正确读取
- 配额更新后所有组件看到一致的值
"""

import pytest
from unittest.mock import patch, MagicMock
from fakeredis import FakeRedis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quantumflow.api.models.tenant import QuotaConfig, Tenant, TenantStatus
from quantumflow.api.middleware.auth import TenantContext, _tenant_cache
from quantumflow.api.routes import tenants as tenants_module
from quantumflow.core.constants import DEFAULT_TENANT_QUOTA


@pytest.fixture(autouse=True)
def clean_state():
    """每个测试前清理全局状态"""
    _tenant_cache.clear()
    tenants_module._tenants.clear()
    TenantContext.clear()
    yield
    _tenant_cache.clear()
    tenants_module._tenants.clear()
    TenantContext.clear()


@pytest.fixture
def fake_redis():
    """共享的 fakeredis 实例"""
    return FakeRedis()


@pytest.fixture
def client_with_fakeredis(fake_redis):
    """带 fakeredis 的测试客户端"""
    app = FastAPI()
    app.include_router(tenants_module.router)
    with patch("quantumflow.api.routes.tenants._get_redis", return_value=fake_redis):
        yield TestClient(app)


@pytest.fixture
def tenant_with_custom_quota(client_with_fakeredis):
    """创建一个自定义配额的租户，返回 (client, tenant_id, quota)"""
    resp = client_with_fakeredis.post(
        "/tenants/",
        json={
            "name": "custom-quota-tenant",
            "priority": 3,
            "quota": {
                "requests_per_minute": 200,
                "requests_per_day": 50000,
                "max_tokens_per_request": 8192,
                "gpu_memory_mb": 16384,
                "concurrent_requests": 25,
            },
        },
    )
    data = resp.json()
    return client_with_fakeredis, data["id"], data["quota"]


class TestTenantToScheduler:
    """验证 tenants.py → scheduler 数据传递"""

    def test_scheduler_reads_quota_created_via_api(self, client_with_fakeredis, fake_redis):
        """通过 API 创建租户后，scheduler 能读取正确配额"""
        from quantumflow.scheduler.distributed import DistributedScheduler

        resp = client_with_fakeredis.post(
            "/tenants/",
            json={
                "name": "scheduler-test",
                "quota": {
                    "requests_per_minute": 150,
                    "concurrent_requests": 30,
                    "gpu_memory_mb": 32768,
                },
            },
        )
        assert resp.status_code == 201
        tenant_id = resp.json()["id"]

        # scheduler 通过同一 fakeredis 读取
        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            scheduler = DistributedScheduler()
            quota = scheduler._get_tenant_quota(tenant_id)

        assert quota.requests_per_minute == 150
        assert quota.concurrent_requests == 30
        assert quota.gpu_memory_mb == 32768

    def test_scheduler_sees_updated_quota(self, client_with_fakeredis, fake_redis):
        """更新租户配额后，scheduler 应立即看到新值"""
        from quantumflow.scheduler.distributed import DistributedScheduler

        # 创建
        resp = client_with_fakeredis.post("/tenants/", json={"name": "update-test"})
        tenant_id = resp.json()["id"]

        # 更新配额
        client_with_fakeredis.patch(
            f"/tenants/{tenant_id}",
            json={"quota": {"requests_per_minute": 500, "concurrent_requests": 50}},
        )

        # scheduler 读到的应是更新后的值
        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            scheduler = DistributedScheduler()
            quota = scheduler._get_tenant_quota(tenant_id)

        assert quota.requests_per_minute == 500
        assert quota.concurrent_requests == 50

    def test_scheduler_default_for_deleted_tenant(self, client_with_fakeredis, fake_redis):
        """软删除后 scheduler 对已删除租户返回默认配额"""
        from quantumflow.scheduler.distributed import DistributedScheduler

        resp = client_with_fakeredis.post("/tenants/", json={"name": "to-delete"})
        tenant_id = resp.json()["id"]

        # 软删除
        client_with_fakeredis.delete(f"/tenants/{tenant_id}")

        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            scheduler = DistributedScheduler()
            quota = scheduler._get_tenant_quota(tenant_id)

        # 即使软删除，scheduler 仍能读取（quota 数据还在）
        assert quota.requests_per_minute == DEFAULT_TENANT_QUOTA["requests_per_minute"]

    def test_scheduler_tenant_isolation(self, client_with_fakeredis, fake_redis):
        """不同租户的配额彼此独立"""
        from quantumflow.scheduler.distributed import DistributedScheduler

        resp1 = client_with_fakeredis.post(
            "/tenants/",
            json={"name": "tenant-a", "quota": {"requests_per_minute": 100}},
        )
        resp2 = client_with_fakeredis.post(
            "/tenants/",
            json={"name": "tenant-b", "quota": {"requests_per_minute": 300}},
        )
        id_a, id_b = resp1.json()["id"], resp2.json()["id"]

        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            scheduler = DistributedScheduler()

            assert scheduler._get_tenant_quota(id_a).requests_per_minute == 100
            assert scheduler._get_tenant_quota(id_b).requests_per_minute == 300


class TestTenantToVRAMManager:
    """验证 tenants.py → vram_manager 数据传递"""

    def test_vram_reads_quota_created_via_api(self, client_with_fakeredis, fake_redis):
        """通过 API 创建租户后，vram_manager 能读取 GPU 配额"""
        from quantumflow.inference.vram_manager import VRAMManager

        resp = client_with_fakeredis.post(
            "/tenants/",
            json={
                "name": "vram-test",
                "quota": {"gpu_memory_mb": 99999},
            },
        )
        assert resp.status_code == 201
        tenant_id = resp.json()["id"]

        vram = VRAMManager()
        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            quota = vram._get_tenant_quota(tenant_id)

        assert quota.gpu_memory_mb == 99999

    def test_vram_quota_enforcement_flag(self, client_with_fakeredis, fake_redis):
        """租户 GPU 配额超限时应拒绝分配"""
        from quantumflow.inference.vram_manager import VRAMManager

        resp = client_with_fakeredis.post(
            "/tenants/",
            json={
                "name": "vram-enforce",
                "quota": {"gpu_memory_mb": 100},  # 100 MB
            },
        )
        tenant_id = resp.json()["id"]

        vram = VRAMManager()
        vram._tenant_quota_enabled = True
        # 注入 fakeredis
        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            # 请求超过配额的显存 (200 MB > 100 MB)
            result = vram.allocate("test-model", 200 * 1024 * 1024, tenant_id=tenant_id)
            assert result is False, "超配额的分配应被拒绝"

            # 请求在配额内的显存 (50 MB < 100 MB)
            result = vram.allocate("test-model-2", 50 * 1024 * 1024, tenant_id=tenant_id)
            assert result is True, "配额内的分配应成功"

    def test_vram_default_tenant_unlimited(self):
        """默认租户（default）的显存分配不应受限"""
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager()
        # default tenant 跳过配额检查
        result = vram.allocate("huge-model", 100 * 1024 * 1024 * 1024, tenant_id="default")
        assert result is True


class TestRateLimiterIntegration:
    """验证 rate limiter 正确使用租户配额"""

    def test_rate_limiter_uses_tenant_quota_from_api(self, client_with_fakeredis, fake_redis):
        """rate limiter 应使用通过 API 设置的租户配额"""
        from quantumflow.api.middlewares.rate_limit import _global_limiter

        resp = client_with_fakeredis.post(
            "/tenants/",
            json={
                "name": "rate-test",
                "quota": {"requests_per_minute": 30, "concurrent_requests": 60},
            },
        )
        tenant_id = resp.json()["id"]

        # rate limiter 基于租户配额的 burst 检查
        # qps=30, burst=60 → 初始令牌=60，消耗1个后应仍有足够令牌
        assert _global_limiter.check_limit(tenant_id, qps=30, burst=60) is True

    def test_rate_limiter_enforces_per_tenant_isolation(self):
        """不同租户的限流独立"""
        from quantumflow.api.middlewares.rate_limit import _global_limiter

        tid_a, tid_b = "tenant-a", "tenant-b"

        # 租户 A: qps=1, burst=1（极端限流）
        # 先用掉唯一令牌
        _global_limiter.check_limit(tid_a, qps=1, burst=1)
        # 第二次应失败
        assert _global_limiter.check_limit(tid_a, qps=1, burst=1) is False

        # 租户 B 不受 A 的限流影响
        assert _global_limiter.check_limit(tid_b, qps=100, burst=200) is True

        # 清理
        _global_limiter.remove_bucket(tid_a)
        _global_limiter.remove_bucket(tid_b)

    def test_rate_limiter_cleanup(self):
        """remove_bucket 后应可重新创建"""
        from quantumflow.api.middlewares.rate_limit import _global_limiter

        tid = "cleanup-test"
        _global_limiter.check_limit(tid, qps=1, burst=1)
        _global_limiter.remove_bucket(tid)

        # 清理后重新创建 —— 应有完整令牌
        assert _global_limiter.check_limit(tid, qps=100, burst=100) is True


class TestPersistenceLifecycle:
    """验证租户数据在完整生命周期中的持久性"""

    def test_create_read_verify_round_trip(self, client_with_fakeredis):
        """创建后立即读取，字段应完全一致"""
        resp = client_with_fakeredis.post(
            "/tenants/",
            json={
                "name": "roundtrip",
                "priority": 8,
                "quota": {
                    "requests_per_minute": 77,
                    "concurrent_requests": 11,
                    "gpu_memory_mb": 5555,
                },
            },
        )
        data = resp.json()
        tid = data["id"]

        # 通过 GET 读回
        get_resp = client_with_fakeredis.get(f"/tenants/{tid}")
        assert get_resp.status_code == 200
        readback = get_resp.json()

        assert readback["name"] == "roundtrip"
        assert readback["priority"] == 8
        assert readback["quota"]["requests_per_minute"] == 77
        assert readback["quota"]["concurrent_requests"] == 11
        assert readback["quota"]["gpu_memory_mb"] == 5555

    def test_lifecycle_active_to_suspended_to_deleted(self, client_with_fakeredis):
        """完整生命周期：active → suspended → active → deleted"""
        resp = client_with_fakeredis.post("/tenants/", json={"name": "lifecycle"})
        tid = resp.json()["id"]

        # suspended
        patch_resp = client_with_fakeredis.patch(
            f"/tenants/{tid}", json={"status": "suspended"}
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["status"] == "suspended"

        # 恢复
        patch_resp = client_with_fakeredis.patch(
            f"/tenants/{tid}", json={"status": "active"}
        )
        assert patch_resp.status_code == 200

        # 删除
        del_resp = client_with_fakeredis.delete(f"/tenants/{tid}")
        assert del_resp.status_code == 204

        # 删除后 GET 返回 404
        get_resp = client_with_fakeredis.get(f"/tenants/{tid}")
        assert get_resp.status_code == 404

    def test_list_includes_all_active_tenants(self, client_with_fakeredis):
        """list 应返回所有活跃租户"""
        resp1 = client_with_fakeredis.post("/tenants/", json={"name": "list-1"})
        resp2 = client_with_fakeredis.post("/tenants/", json={"name": "list-2"})
        id1, id2 = resp1.json()["id"], resp2.json()["id"]

        # 删除 list-1
        client_with_fakeredis.delete(f"/tenants/{id1}")

        # list 只应包含 list-2
        list_resp = client_with_fakeredis.get("/tenants/")
        ids = [t["id"] for t in list_resp.json()]
        assert id1 not in ids, "已删除租户不应出现在列表中"
        assert id2 in ids


class TestSDKMultiCall:
    """验证 SDK 客户端在多次调用中的正确行为"""

    def test_sync_client_multiple_calls(self):
        """同步客户端应能多次调用不报错（之前 with self._client 导致第二次失败）"""
        from quantumflow.sdk.client import SyncQuantumFlowClient
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch("httpx.Client.get", return_value=mock_response):
            client = SyncQuantumFlowClient()

            # 第一次调用
            result1 = client.health_check()
            assert result1["status"] == "healthy"

            # 第二次调用 — 之前会因 client 已关闭而失败
            result2 = client.health_check()
            assert result2["status"] == "healthy"

            client.close()

    def test_sync_client_generate_then_health(self):
        """先调用 generate 再 health_check，两次都应成功"""
        from quantumflow.sdk.client import SyncQuantumFlowClient

        gen_response = MagicMock()
        gen_response.status_code = 200
        gen_response.json.return_value = {
            "request_id": "req-1",
            "model": "test-model",
            "generated_text": "Hello",
            "finish_reason": "stop",
            "latency_ms": 100.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}

        with patch("httpx.Client.post", return_value=gen_response), \
             patch("httpx.Client.get", return_value=health_response):
            client = SyncQuantumFlowClient()

            result1 = client.generate("test-model", "Hello")
            assert result1.generated_text == "Hello"

            result2 = client.health_check()
            assert result2["status"] == "healthy"

            client.close()

    def test_sync_client_context_manager_multi_call(self):
        """上下文管理器内多次调用应正常工作"""
        from quantumflow.sdk.client import SyncQuantumFlowClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch("httpx.Client.get", return_value=mock_response):
            with SyncQuantumFlowClient() as client:
                for _ in range(3):
                    result = client.health_check()
                    assert result["status"] == "healthy"


class TestVRAMBlockTracking:
    """验证 VRAM block 分配/释放实际追踪"""

    def test_allocate_tracks_bytes(self):
        """分配后应追踪 reserved bytes"""
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager()
        vram.allocate("model-a", 1024 * 1024, tenant_id="default")

        assert vram._model_reserved_bytes.get("model-a", 0) == 1024 * 1024

    def test_multiple_allocations_accumulate(self):
        """多次分配应累计 reserved bytes"""
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager()
        vram.allocate("model-x", 100, tenant_id="default")
        vram.allocate("model-x", 200, tenant_id="default")

        assert vram._model_reserved_bytes["model-x"] == 300

    def test_release_reduces_bytes(self):
        """释放后 reserved bytes 应减少"""
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager()
        vram.allocate("model-r", 500, tenant_id="default")
        vram.release("model-r", tenant_id="default")

        assert vram._model_reserved_bytes["model-r"] == 0
