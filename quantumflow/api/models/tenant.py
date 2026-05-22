"""租户数据模型"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TenantStatus(str, Enum):
    """租户状态"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class QuotaConfig(BaseModel):
    """资源配额配置"""
    requests_per_minute: int = Field(default=60, ge=1, le=10000)
    requests_per_day: int = Field(default=10000, ge=1)
    max_tokens_per_request: int = Field(default=8192, ge=1, le=32768)
    gpu_memory_mb: int = Field(default=4096, ge=0)  # 0 = 无限制
    concurrent_requests: int = Field(default=10, ge=1, le=100)


class Tenant(BaseModel):
    """租户模型"""
    id: str = Field(default_factory=lambda: str(uuid4()), description="租户唯一ID")
    name: str = Field(..., min_length=1, max_length=100, description="租户名称")
    api_key_hash: str = Field(..., description="API Key 哈希值 (bcrypt)")
    api_key_prefix: str = Field(..., description="API Key 前缀 (用于展示)")
    status: TenantStatus = Field(default=TenantStatus.ACTIVE)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    priority: int = Field(default=5, ge=0, le=10, description="调度优先级")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "acme-corp",
                "quota": {
                    "requests_per_minute": 100,
                    "gpu_memory_mb": 8192
                },
                "priority": 8
            }
        }
    }


class TenantCreate(BaseModel):
    """创建租户请求"""
    name: str = Field(..., min_length=1, max_length=100)
    quota: QuotaConfig | None = None
    priority: int = Field(default=5, ge=0, le=10)


class TenantUpdate(BaseModel):
    """更新租户请求"""
    name: str | None = Field(default=None, min_length=1, max_length=100)
    quota: QuotaConfig | None = None
    priority: int | None = Field(default=None, ge=0, le=10)
    status: TenantStatus | None = None


class TenantResponse(BaseModel):
    """租户响应 (不包含敏感信息)"""
    id: str
    name: str
    status: TenantStatus
    quota: QuotaConfig
    priority: int
    created_at: datetime
    last_active_at: datetime | None


class TenantCreateResponse(TenantResponse):
    """租户创建响应 (包含 API Key，仅返回一次)"""
    api_key: str
    api_key_prefix: str


class TenantUsage(BaseModel):
    """租户使用量"""
    tenant_id: str
    requests_today: int = 0
    tokens_today: int = 0
    concurrent_requests: int = 0
    gpu_memory_mb: float = 0.0
    quota_remaining: dict[str, Any] = {}
