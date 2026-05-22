"""租户模型单元测试"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from quantumflow.api.models.tenant import (
    QuotaConfig,
    Tenant,
    TenantCreate,
    TenantResponse,
    TenantStatus,
    TenantUpdate,
)


def test_quota_config_defaults():
    """测试配额配置默认值"""
    quota = QuotaConfig()
    assert quota.requests_per_minute == 60
    assert quota.requests_per_day == 10000
    assert quota.gpu_memory_mb == 4096


def test_quota_config_validation():
    """测试配额配置验证"""
    with pytest.raises(ValidationError):
        QuotaConfig(requests_per_minute=0)  # 必须 >= 1

    with pytest.raises(ValidationError):
        QuotaConfig(gpu_memory_mb=-100)  # 必须 >= 0


def test_tenant_auto_generated_id():
    """测试租户自动生成 ID"""
    tenant = Tenant(name="test", api_key_hash="hash", api_key_prefix="qk_")
    assert tenant.id is not None
    assert len(tenant.id) == 36  # UUID format


def test_tenant_status_defaults():
    """测试租户状态默认值"""
    tenant = Tenant(name="test", api_key_hash="hash", api_key_prefix="qk_")
    assert tenant.status == TenantStatus.ACTIVE


def test_tenant_create_with_quota():
    """测试带配额创建租户"""
    quota = QuotaConfig(requests_per_minute=100, gpu_memory_mb=8192)
    create = TenantCreate(name="acme", quota=quota)
    assert create.quota.requests_per_minute == 100


def test_tenant_update_partial():
    """测试部分更新"""
    update = TenantUpdate(priority=8)
    assert update.priority == 8
    assert update.name is None
    assert update.quota is None


def test_tenant_response():
    """测试租户响应模型"""
    quota = QuotaConfig()
    response = TenantResponse(
        id="test-id",
        name="test",
        status=TenantStatus.ACTIVE,
        quota=quota,
        priority=5,
        created_at=datetime.now(timezone.utc),
        last_active_at=None
    )
    assert response.id == "test-id"
    assert response.name == "test"
    assert response.status == TenantStatus.ACTIVE


def test_tenant_update_priority_validation():
    """测试更新请求优先级验证"""
    # 优先级超出范围应该报错
    with pytest.raises(ValidationError):
        TenantUpdate(priority=15)  # 应该 <= 10

    with pytest.raises(ValidationError):
        TenantUpdate(priority=-1)  # 应该 >= 0
