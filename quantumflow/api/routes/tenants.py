"""租户管理 API 路由"""

import json
import structlog
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from quantumflow.api.models.tenant import (
    QuotaConfig,
    Tenant,
    TenantCreate,
    TenantCreateResponse,
    TenantResponse,
    TenantStatus,
    TenantUpdate,
    TenantUsage,
)
from quantumflow.api.middleware.auth import generate_api_key
from quantumflow.core.constants import DEFAULT_TENANT_QUOTA, TENANT_PREFIX

logger = structlog.get_logger().bind(component="tenant_routes")

router = APIRouter(prefix="/tenants", tags=["Tenants"])

# 内存存储 (L1 缓存，Redis 为 L2 持久存储)
_tenants: dict[str, Tenant] = {}

# Redis key patterns
TENANT_ID_KEY = "qf:tenant:id:{}"  # tenant_id → hashed_api_key
TENANT_IDS_SET = "qf:tenant:ids"   # set of all tenant IDs


def _get_redis():
    try:
        from quantumflow.storage import get_redis_manager_sync
        return get_redis_manager_sync().get_client()
    except Exception:
        return None


def _serialize_tenant(tenant: Tenant) -> dict:
    """序列化租户为 Redis Hash"""
    return {
        "id": tenant.id,
        "name": tenant.name,
        "api_key_hash": tenant.api_key_hash,
        "api_key_prefix": tenant.api_key_prefix,
        "status": tenant.status.value,
        "quota_requests_per_minute": str(tenant.quota.requests_per_minute),
        "quota_requests_per_day": str(tenant.quota.requests_per_day),
        "quota_max_tokens": str(tenant.quota.max_tokens_per_request),
        "quota_gpu_memory": str(tenant.quota.gpu_memory_mb),
        "quota_concurrent": str(tenant.quota.concurrent_requests),
        "priority": str(tenant.priority),
        "created_at": tenant.created_at.isoformat(),
        "updated_at": tenant.updated_at.isoformat(),
    }


def _deserialize_tenant(data: dict) -> Tenant:
    """从 Redis Hash 反序列化租户"""
    quota = QuotaConfig(
        requests_per_minute=int(data.get(b"quota_requests_per_minute", DEFAULT_TENANT_QUOTA["requests_per_minute"])),
        requests_per_day=int(data.get(b"quota_requests_per_day", DEFAULT_TENANT_QUOTA["requests_per_day"])),
        max_tokens_per_request=int(data.get(b"quota_max_tokens", DEFAULT_TENANT_QUOTA["max_tokens_per_request"])),
        gpu_memory_mb=int(data.get(b"quota_gpu_memory", DEFAULT_TENANT_QUOTA["gpu_memory_mb"])),
        concurrent_requests=int(data.get(b"quota_concurrent", DEFAULT_TENANT_QUOTA["concurrent_requests"])),
    )
    return Tenant(
        id=data[b"id"].decode(),
        name=data[b"name"].decode(),
        api_key_hash=data[b"api_key_hash"].decode(),
        api_key_prefix=data[b"api_key_prefix"].decode(),
        status=TenantStatus(data[b"status"].decode()),
        quota=quota,
        priority=int(data.get(b"priority", 5)),
    )


def _store_tenant_redis(tenant: Tenant) -> None:
    """将租户写入 Redis（含 ID 索引）"""
    redis = _get_redis()
    if not redis:
        return
    redis.hset(f"{TENANT_PREFIX}{tenant.api_key_hash}", mapping=_serialize_tenant(tenant))
    redis.set(TENANT_ID_KEY.format(tenant.id), tenant.api_key_hash)
    redis.sadd(TENANT_IDS_SET, tenant.id)


def _remove_tenant_redis(tenant_id: str, api_key_hash: str) -> None:
    """从 Redis 移除租户"""
    redis = _get_redis()
    if not redis:
        return
    redis.delete(f"{TENANT_PREFIX}{api_key_hash}")
    redis.delete(TENANT_ID_KEY.format(tenant_id))
    redis.srem(TENANT_IDS_SET, tenant_id)


def _load_tenant_from_redis(tenant_id: str) -> Tenant | None:
    """通过 ID 从 Redis 加载租户"""
    redis = _get_redis()
    if not redis:
        return None
    hashed_key = redis.get(TENANT_ID_KEY.format(tenant_id))
    if not hashed_key:
        return None
    data = redis.hgetall(f"{TENANT_PREFIX}{hashed_key.decode() if isinstance(hashed_key, bytes) else hashed_key}")
    if not data:
        return None
    return _deserialize_tenant(data)


def _load_all_tenants_from_redis() -> list[Tenant]:
    """从 Redis 加载所有租户"""
    redis = _get_redis()
    if not redis:
        return []
    ids = redis.smembers(TENANT_IDS_SET)
    if not ids:
        return []
    tenants = []
    for tid in ids:
        tid_str = tid.decode() if isinstance(tid, bytes) else tid
        t = _load_tenant_from_redis(tid_str)
        if t and t.status != TenantStatus.DELETED:
            tenants.append(t)
    return tenants


@router.post("/", response_model=TenantCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(create: TenantCreate) -> TenantCreateResponse:
    """创建新租户（返回 API Key，仅此一次）"""
    # 生成 API Key
    plain_key, hashed_key, prefix = generate_api_key()

    # 创建租户
    tenant = Tenant(
        name=create.name,
        api_key_hash=hashed_key,
        api_key_prefix=prefix,
        quota=create.quota or QuotaConfig(),
        priority=create.priority,
    )

    # 保存到内存（L1）和 Redis（L2）
    _tenants[tenant.id] = tenant
    _store_tenant_redis(tenant)

    logger.info("tenant_created", tenant_id=tenant.id, name=tenant.name)

    return TenantCreateResponse(
        id=tenant.id,
        name=tenant.name,
        status=tenant.status,
        quota=tenant.quota,
        priority=tenant.priority,
        created_at=tenant.created_at,
        last_active_at=tenant.last_active_at,
        api_key=plain_key,
        api_key_prefix=prefix,
    )


@router.get("/", response_model=list[TenantResponse])
async def list_tenants() -> list[TenantResponse]:
    """列出所有租户（合并内存和 Redis 数据）"""
    seen: set[str] = set()
    results: list[TenantResponse] = []

    # 从 L1 缓存加载
    for t in _tenants.values():
        if t.status != TenantStatus.DELETED and t.id not in seen:
            seen.add(t.id)
            results.append(_tenant_to_response(t))

    # 从 Redis 加载（补充内存中没有的）
    for t in _load_all_tenants_from_redis():
        if t.id not in seen:
            seen.add(t.id)
            _tenants[t.id] = t  # 回填缓存
            results.append(_tenant_to_response(t))

    return results


def _tenant_to_response(t: Tenant) -> TenantResponse:
    return TenantResponse(
        id=t.id,
        name=t.name,
        status=t.status,
        quota=t.quota,
        priority=t.priority,
        created_at=t.created_at,
        last_active_at=t.last_active_at,
    )


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(tenant_id: str) -> TenantResponse:
    """获取租户详情（优先内存，回退 Redis）"""
    t = _tenants.get(tenant_id)
    if not t:
        t = _load_tenant_from_redis(tenant_id)
        if t:
            _tenants[tenant_id] = t  # 回填缓存
    if not t or t.status == TenantStatus.DELETED:
        raise HTTPException(status_code=404, detail="租户不存在")
    return _tenant_to_response(t)


@router.patch("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(tenant_id: str, update: TenantUpdate) -> TenantResponse:
    """更新租户"""
    t = _tenants.get(tenant_id)
    if not t:
        t = _load_tenant_from_redis(tenant_id)
        if t:
            _tenants[tenant_id] = t
    if not t or t.status == TenantStatus.DELETED:
        raise HTTPException(status_code=404, detail="租户不存在")

    if update.name is not None:
        t.name = update.name
    if update.quota is not None:
        t.quota = update.quota
    if update.priority is not None:
        t.priority = update.priority
    if update.status is not None:
        t.status = update.status

    t.updated_at = datetime.now(timezone.utc)
    _store_tenant_redis(t)

    logger.info("tenant_updated", tenant_id=t.id)
    return _tenant_to_response(t)


@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tenant(tenant_id: str):
    """删除租户 (软删除)"""
    t = _tenants.get(tenant_id)
    if not t:
        t = _load_tenant_from_redis(tenant_id)
        if t:
            _tenants[tenant_id] = t
    if not t or t.status == TenantStatus.DELETED:
        raise HTTPException(status_code=404, detail="租户不存在")

    t.status = TenantStatus.DELETED
    t.updated_at = datetime.now(timezone.utc)

    # 同步到 Redis — 保留 hash 数据但更新状态，保持 ID 索引
    redis = _get_redis()
    if redis:
        redis.hset(f"{TENANT_PREFIX}{t.api_key_hash}", "status", "deleted")

    logger.info("tenant_deleted", tenant_id=t.id)


@router.get("/{tenant_id}/usage", response_model=TenantUsage)
async def get_tenant_usage(tenant_id: str) -> TenantUsage:
    """获取租户使用量"""
    tenant = _tenants.get(tenant_id)
    if not tenant:
        tenant = _load_tenant_from_redis(tenant_id)
        if tenant:
            _tenants[tenant_id] = tenant
    if not tenant or tenant.status == TenantStatus.DELETED:
        raise HTTPException(status_code=404, detail="租户不存在")

    # 从 Redis 获取使用量
    redis = _get_redis()
    requests_today = 0
    tokens_today = 0
    concurrent = 0

    if redis:
        usage_key = f"qf:usage:{tenant_id}:today"
        data = redis.hgetall(usage_key)
        requests_today = int(data.get(b"requests", 0) if data else 0)
        tokens_today = int(data.get(b"tokens", 0) if data else 0)

    return TenantUsage(
        tenant_id=tenant_id,
        requests_today=requests_today,
        tokens_today=tokens_today,
        concurrent_requests=concurrent,
        gpu_memory_mb=0.0,
        quota_remaining={
            "requests_per_minute": tenant.quota.requests_per_minute,
            "requests_per_day_remaining": max(0, tenant.quota.requests_per_day - requests_today),
        },
    )
