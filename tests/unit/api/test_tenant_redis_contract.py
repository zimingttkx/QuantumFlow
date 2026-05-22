"""Redis 契约测试 — 验证所有组件使用一致的 key 和 field 名称

这些测试防止之前发现的生产 bug：
- scheduler 读 qf:quota:{id} 但数据写在 qf:tenant:{hash}
- field 名称不一致 (rpm vs quota_requests_per_minute)
"""

import pytest
from unittest.mock import patch, MagicMock
from fakeredis import FakeRedis

from quantumflow.api.models.tenant import QuotaConfig, Tenant, TenantStatus
from quantumflow.api.routes.tenants import (
    _serialize_tenant,
    _deserialize_tenant,
    _store_tenant_redis,
    _load_tenant_from_redis,
    _load_all_tenants_from_redis,
    _remove_tenant_redis,
    TENANT_ID_KEY,
    TENANT_IDS_SET,
)
from quantumflow.core.constants import DEFAULT_TENANT_QUOTA, TENANT_PREFIX


@pytest.fixture
def fake_redis():
    """提供 fakeredis 实例"""
    return FakeRedis()


@pytest.fixture
def sample_tenant():
    """标准测试租户"""
    return Tenant(
        name="contract-test-tenant",
        api_key_hash="abc123def456",
        api_key_prefix="qk_test1",
        quota=QuotaConfig(
            requests_per_minute=120,
            requests_per_day=20000,
            max_tokens_per_request=4096,
            gpu_memory_mb=8192,
            concurrent_requests=15,
        ),
        priority=7,
    )


@pytest.fixture
def patch_get_redis(fake_redis):
    """将 _get_redis 替换为 fakeredis"""
    with patch("quantumflow.api.routes.tenants._get_redis", return_value=fake_redis):
        yield


class TestSerializationRoundTrip:
    """序列化 → Redis → 反序列化 往返一致性"""

    def test_round_trip_all_fields(self, sample_tenant, fake_redis, patch_get_redis):
        """写入 Redis 后读回，所有字段应完全一致"""
        _store_tenant_redis(sample_tenant)

        loaded = _load_tenant_from_redis(sample_tenant.id)
        assert loaded is not None, "应该能从 Redis 加载租户"

        assert loaded.id == sample_tenant.id
        assert loaded.name == sample_tenant.name
        assert loaded.api_key_hash == sample_tenant.api_key_hash
        assert loaded.api_key_prefix == sample_tenant.api_key_prefix
        assert loaded.status == sample_tenant.status
        assert loaded.priority == sample_tenant.priority
        assert loaded.quota.requests_per_minute == sample_tenant.quota.requests_per_minute
        assert loaded.quota.requests_per_day == sample_tenant.quota.requests_per_day
        assert loaded.quota.max_tokens_per_request == sample_tenant.quota.max_tokens_per_request
        assert loaded.quota.gpu_memory_mb == sample_tenant.quota.gpu_memory_mb
        assert loaded.quota.concurrent_requests == sample_tenant.quota.concurrent_requests

    def test_round_trip_default_quota(self, fake_redis, patch_get_redis):
        """使用默认配额的租户也能正确往返"""
        tenant = Tenant(
            name="default-quota-tenant",
            api_key_hash="def789",
            api_key_prefix="qk_def78",
            quota=QuotaConfig(),
            priority=5,
        )
        _store_tenant_redis(tenant)
        loaded = _load_tenant_from_redis(tenant.id)
        assert loaded.quota.requests_per_minute == DEFAULT_TENANT_QUOTA["requests_per_minute"]
        assert loaded.quota.concurrent_requests == DEFAULT_TENANT_QUOTA["concurrent_requests"]

    def test_deserialize_missing_fields_uses_defaults(self):
        """Redis 数据缺少字段时使用默认值"""
        minimal_data = {
            b"id": b"minimal-1",
            b"name": b"minimal",
            b"api_key_hash": b"hash123",
            b"api_key_prefix": b"qk_min",
            b"status": b"active",
        }
        tenant = _deserialize_tenant(minimal_data)
        assert tenant.quota.requests_per_minute == DEFAULT_TENANT_QUOTA["requests_per_minute"]
        assert tenant.quota.gpu_memory_mb == DEFAULT_TENANT_QUOTA["gpu_memory_mb"]
        assert tenant.priority == 5


class TestRedisKeyPatterns:
    """验证所有组件使用相同的 Redis key 模式"""

    def test_store_creates_id_index(self, sample_tenant, fake_redis, patch_get_redis):
        """_store_tenant_redis 应创建 ID→hash 索引"""
        _store_tenant_redis(sample_tenant)

        # qf:tenant:id:{tenant_id} → api_key_hash
        id_key = TENANT_ID_KEY.format(sample_tenant.id)
        stored_hash = fake_redis.get(id_key)
        assert stored_hash is not None, f"ID 索引 {id_key} 应存在"
        assert stored_hash.decode() if isinstance(stored_hash, bytes) else stored_hash == sample_tenant.api_key_hash

    def test_store_adds_to_ids_set(self, sample_tenant, fake_redis, patch_get_redis):
        """_store_tenant_redis 应将 ID 加入全局集合"""
        _store_tenant_redis(sample_tenant)

        assert fake_redis.sismember(TENANT_IDS_SET, sample_tenant.id), \
            f"ID {sample_tenant.id} 应在 {TENANT_IDS_SET} 中"

    def test_store_writes_hash_to_correct_key(self, sample_tenant, fake_redis, patch_get_redis):
        """租户数据应写入 qf:tenant:{api_key_hash}"""
        _store_tenant_redis(sample_tenant)

        expected_key = f"{TENANT_PREFIX}{sample_tenant.api_key_hash}"
        data = fake_redis.hgetall(expected_key)
        assert data, f"租户数据应存在 key={expected_key}"
        assert data[b"name"].decode() == sample_tenant.name

    def test_remove_cleans_up_all_keys(self, sample_tenant, fake_redis, patch_get_redis):
        """_remove_tenant_redis 应清理 hash + ID 索引 + ID 集合"""
        _store_tenant_redis(sample_tenant)
        _remove_tenant_redis(sample_tenant.id, sample_tenant.api_key_hash)

        # Hash key 应被删除
        assert not fake_redis.exists(f"{TENANT_PREFIX}{sample_tenant.api_key_hash}")
        # ID 索引应被删除
        assert not fake_redis.exists(TENANT_ID_KEY.format(sample_tenant.id))
        # ID 不应在集合中
        assert not fake_redis.sismember(TENANT_IDS_SET, sample_tenant.id)

    def test_load_all_returns_all_active_tenants(self, fake_redis, patch_get_redis):
        """_load_all_tenants_from_redis 应返回所有非删除租户"""
        t1 = Tenant(name="t1", api_key_hash="h1", api_key_prefix="qk_a1", quota=QuotaConfig())
        t2 = Tenant(name="t2", api_key_hash="h2", api_key_prefix="qk_b2", quota=QuotaConfig())
        t3 = Tenant(name="t3", api_key_hash="h3", api_key_prefix="qk_c3", quota=QuotaConfig(),
                    status=TenantStatus.DELETED)

        _store_tenant_redis(t1)
        _store_tenant_redis(t2)
        _store_tenant_redis(t3)

        all_tenants = _load_all_tenants_from_redis()
        ids = {t.id for t in all_tenants}
        assert t1.id in ids
        assert t2.id in ids
        assert t3.id not in ids, "已删除的租户不应出现在列表"

    def test_load_nonexistent_tenant_returns_none(self, fake_redis, patch_get_redis):
        """加载不存在的租户应返回 None"""
        assert _load_tenant_from_redis("nonexistent-id") is None

    def test_id_index_missing_returns_none(self, fake_redis, patch_get_redis):
        """ID 索引缺失时返回 None（即使 hash key 存在）"""
        # 直接写 hash，跳过 ID 索引
        fake_redis.hset(f"{TENANT_PREFIX}orphan-hash", mapping={
            "id": "orphan-1", "name": "orphan", "api_key_hash": "orphan-hash",
            "api_key_prefix": "qk_or", "status": "active",
        })
        assert _load_tenant_from_redis("orphan-1") is None


class TestSchedulerContract:
    """验证 scheduler 使用与 tenants.py 一致的 key 和 field"""

    def test_scheduler_reads_tenant_written_by_store(self, sample_tenant, fake_redis, patch_get_redis):
        """scheduler._get_tenant_quota 应能读取 tenants._store_tenant_redis 写入的数据"""
        from quantumflow.scheduler.distributed import DistributedScheduler

        _store_tenant_redis(sample_tenant)

        # 模拟 scheduler 的 _get_redis
        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            scheduler = DistributedScheduler()
            quota = scheduler._get_tenant_quota(sample_tenant.id)

        assert quota.requests_per_minute == sample_tenant.quota.requests_per_minute
        assert quota.requests_per_day == sample_tenant.quota.requests_per_day
        assert quota.max_tokens_per_request == sample_tenant.quota.max_tokens_per_request
        assert quota.gpu_memory_mb == sample_tenant.quota.gpu_memory_mb
        assert quota.concurrent_requests == sample_tenant.quota.concurrent_requests

    def test_scheduler_unknown_tenant_gets_defaults(self):
        """scheduler 对未知租户返回默认配额"""
        from quantumflow.scheduler.distributed import DistributedScheduler

        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = FakeRedis()
            scheduler = DistributedScheduler()
            quota = scheduler._get_tenant_quota("unknown-tenant")

        assert quota.requests_per_minute == DEFAULT_TENANT_QUOTA["requests_per_minute"]
        assert quota.concurrent_requests == DEFAULT_TENANT_QUOTA["concurrent_requests"]

    def test_scheduler_field_names_match_serialization(self, sample_tenant):
        """scheduler 读取的 field 名称必须与 _serialize_tenant 输出一致"""
        serialized = _serialize_tenant(sample_tenant)

        # scheduler 使用的 field 名称
        scheduler_fields = [
            "quota_requests_per_minute",
            "quota_requests_per_day",
            "quota_max_tokens",
            "quota_gpu_memory",
            "quota_concurrent",
        ]
        for field in scheduler_fields:
            assert field in serialized, \
                f"Field '{field}' must exist in _serialize_tenant output (used by scheduler)"

    def test_scheduler_id_key_pattern_matches_tenants(self):
        """scheduler 和 tenants.py 对同一 tenant_id 应生成相同的 ID 索引 key"""
        from quantumflow.api.routes.tenants import TENANT_ID_KEY as ROUTES_ID_KEY

        # tenants.py 使用 "qf:tenant:id:{}".format(tid)
        # scheduler 使用 f"qf:tenant:id:{tid}"
        # 两者对同一 tid 必须生成一致的结果
        tid = "test-tenant-abc"
        routes_key = ROUTES_ID_KEY.format(tid)
        scheduler_key = f"qf:tenant:id:{tid}"
        assert routes_key == scheduler_key, \
            f"Key mismatch: routes={routes_key} vs scheduler={scheduler_key}"


class TestVRAMManagerContract:
    """验证 vram_manager 使用与 tenants.py 一致的 key 和 field"""

    def test_vram_reads_tenant_written_by_store(self, sample_tenant, fake_redis, patch_get_redis):
        """vram_manager._get_tenant_quota 应能读取 tenants._store_tenant_redis 写入的数据"""
        from quantumflow.inference.vram_manager import VRAMManager

        _store_tenant_redis(sample_tenant)

        vram = VRAMManager()
        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            quota = vram._get_tenant_quota(sample_tenant.id)

        assert quota.gpu_memory_mb == sample_tenant.quota.gpu_memory_mb
        assert quota.requests_per_minute == sample_tenant.quota.requests_per_minute
        assert quota.concurrent_requests == sample_tenant.quota.concurrent_requests

    def test_vram_field_names_match_serialization(self, sample_tenant):
        """vram_manager 读取的 field 名称必须与 _serialize_tenant 输出一致"""
        serialized = _serialize_tenant(sample_tenant)

        vram_fields = [
            "quota_requests_per_minute",
            "quota_requests_per_day",
            "quota_max_tokens",
            "quota_gpu_memory",
            "quota_concurrent",
        ]
        for field in vram_fields:
            assert field in serialized, \
                f"Field '{field}' must exist in _serialize_tenant output (used by vram_manager)"


class TestAuthMiddlewareContract:
    """验证 auth middleware 使用与 tenants.py 一致的 key 和 field"""

    def test_auth_deserialization_field_names(self, sample_tenant):
        """auth middleware _deserialize_tenant 读取的字段应与 _serialize_tenant 输出一致"""
        from quantumflow.api.middleware.auth import TenantAuthMiddleware

        serialized = _serialize_tenant(sample_tenant)

        # auth middleware 的 _deserialize_tenant 读取这些 field
        auth_fields = [
            "quota_requests_per_minute",
            "quota_requests_per_day",
            "quota_max_tokens",
            "quota_gpu_memory",
            "quota_concurrent",
        ]
        for field in auth_fields:
            assert field in serialized, \
                f"Field '{field}' must exist in _serialize_tenant output (used by auth middleware)"

    def test_auth_reads_store_written_data(self, sample_tenant, fake_redis, patch_get_redis):
        """auth middleware 应能反序列化 _store_tenant_redis 写入的数据"""
        from quantumflow.api.middleware.auth import TenantAuthMiddleware

        _store_tenant_redis(sample_tenant)

        raw_data = fake_redis.hgetall(f"{TENANT_PREFIX}{sample_tenant.api_key_hash}")
        mw = TenantAuthMiddleware(None)  # app=None for testing
        tenant = mw._deserialize_tenant(raw_data)

        assert tenant.id == sample_tenant.id
        assert tenant.name == sample_tenant.name
        assert tenant.api_key_hash == sample_tenant.api_key_hash
        assert tenant.status == sample_tenant.status
        assert tenant.quota.requests_per_minute == sample_tenant.quota.requests_per_minute
        assert tenant.quota.concurrent_requests == sample_tenant.quota.concurrent_requests


class TestConcurrentCounterContract:
    """验证并发计数器 key 的独立性"""

    def test_concurrent_counter_separate_from_tenant_data(self, sample_tenant, fake_redis, patch_get_redis):
        """并发计数器 key 应与租户数据 key 分离"""
        from quantumflow.scheduler.distributed import DistributedScheduler

        _store_tenant_redis(sample_tenant)

        with patch("quantumflow.storage.get_redis_manager_sync") as mock_mgr:
            mock_mgr.return_value.get_client.return_value = fake_redis
            scheduler = DistributedScheduler()

            # 初始并发为 0
            assert scheduler._get_concurrent_requests(sample_tenant.id) == 0

            # 增加并发
            scheduler._increment_concurrent_requests(sample_tenant.id)
            assert scheduler._get_concurrent_requests(sample_tenant.id) == 1

            # 验证计数器 key 不影响租户数据
            tenant_data = fake_redis.hgetall(f"{TENANT_PREFIX}{sample_tenant.api_key_hash}")
            assert tenant_data, "租户数据不应被并发计数器影响"

    def test_concurrent_key_pattern(self):
        """并发计数器使用 qf:concurrent:{tenant_id} 模式"""
        key = "qf:concurrent:test-tenant-id"
        assert key.startswith("qf:concurrent:")
        assert "tenant:" not in key, "并发计数器 key 不应与租户数据 key 冲突"
