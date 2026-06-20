"""租户认证中间件"""

import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict
from contextvars import ContextVar
from typing import Callable

import structlog
from fastapi import Request, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from quantumflow.api.models.tenant import QuotaConfig, Tenant, TenantStatus
from quantumflow.core.constants import DEFAULT_TENANT_QUOTA, TENANT_PREFIX
from quantumflow.storage import get_redis_manager

logger = structlog.get_logger().bind(component="tenant_auth")

# API Key Header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# 内存缓存 (生产环境应使用 Redis)
_tenant_cache: dict[str, Tenant] = {}
_tenant_cache_times: dict[str, float] = {}
_cache_lock = threading.Lock()
_cache_ttl = 300  # 5 minutes


def generate_api_key() -> tuple[str, str, str]:
    """生成 API Key

    Returns:
        (plain_key, hashed_key, prefix): 明文密钥 (仅返回一次)、哈希值和前缀
    """
    plain_key = f"qk_{secrets.token_urlsafe(32)}"
    hashed_key = hashlib.sha256(plain_key.encode()).hexdigest()
    prefix = plain_key[:8]
    return plain_key, hashed_key, prefix


def hash_api_key(plain_key: str) -> str:
    """哈希 API Key"""
    return hashlib.sha256(plain_key.encode()).hexdigest()


def _get_redis():
    """获取 Redis 客户端"""
    try:
        from quantumflow.storage import get_redis_manager_sync
        return get_redis_manager_sync().get_client()
    except Exception:
        return None


_TENANT_VAR: ContextVar[Tenant | None] = ContextVar("tenant", default=None)


class TenantContext:
    """租户上下文 (async-safe context variable)"""

    @classmethod
    def set_tenant(cls, tenant: Tenant | None) -> None:
        _TENANT_VAR.set(tenant)

    @classmethod
    def get_tenant(cls) -> Tenant | None:
        return _TENANT_VAR.get()

    @classmethod
    def clear(cls) -> None:
        _TENANT_VAR.set(None)


class TenantAuthMiddleware(BaseHTTPMiddleware):
    """租户认证中间件

    功能:
    1. 从 X-API-Key header 提取 API Key
    2. 验证 API Key 并加载租户信息
    3. 检查租户状态 (active/suspended)
    4. 设置租户上下文供后续处理
    """

    # 不需要认证的路径
    EXEMPT_PATHS = {
        "/",
        "/health",
        "/api/v1/health",
        "/api/v1/health/ready",
        "/api/v1/health/live",
        "/api/v1/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    def __init__(self, app, redis_enabled: bool = True):
        super().__init__(app)
        self.redis_enabled = redis_enabled

    def _has_auth_infrastructure(self) -> bool:
        """检查是否存在认证基础设施（Redis 或缓存租户）"""
        with _cache_lock:
            if _tenant_cache:
                return True
        redis = _get_redis()
        return redis is not None

    @staticmethod
    def _default_tenant() -> Tenant:
        """返回开发/测试模式的默认租户"""
        return Tenant(
            id="default",
            name="默认租户",
            api_key_hash="",
            api_key_prefix="",
            status=TenantStatus.ACTIVE,
            quota=QuotaConfig(**DEFAULT_TENANT_QUOTA),
            priority=5,
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查是否需要认证
        if self._is_exempt(request.url.path):
            return await call_next(request)

        # 提取 API Key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            # 无租户数据且 Redis 不可用 → 开发/测试模式，放行
            if not self._has_auth_infrastructure():
                TenantContext.set_tenant(self._default_tenant())
                try:
                    return await call_next(request)
                finally:
                    TenantContext.clear()

            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "MISSING_API_KEY",
                        "message": "缺少 API Key，请使用 X-API-Key header"
                    }
                }
            )

        # 验证并获取租户
        tenant = await self._authenticate(api_key)
        if not tenant:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "INVALID_API_KEY",
                        "message": "无效的 API Key"
                    }
                }
            )

        # 检查租户状态
        if tenant.status != TenantStatus.ACTIVE:
            error_code = "TENANT_SUSPENDED" if tenant.status == TenantStatus.SUSPENDED else "TENANT_DELETED"
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": error_code,
                        "message": f"租户已被{'暂停' if tenant.status == TenantStatus.SUSPENDED else '删除'}"
                    }
                }
            )

        # 设置租户上下文
        TenantContext.set_tenant(tenant)

        try:
            response = await call_next(request)
            return response
        finally:
            TenantContext.clear()

    def _is_exempt(self, path: str) -> bool:
        """检查路径是否免认证"""
        if path in self.EXEMPT_PATHS:
            return True
        return False

    async def _authenticate(self, api_key: str) -> Tenant | None:
        """验证 API Key 并返回租户"""
        key_hash = hash_api_key(api_key)

        # 先检查内存缓存
        with _cache_lock:
            if key_hash in _tenant_cache:
                cache_time = _tenant_cache_times.get(key_hash, 0)
                if time.time() - cache_time < _cache_ttl:
                    return _tenant_cache[key_hash]

        # 从 Redis 加载
        if self.redis_enabled:
            redis = _get_redis()
            if redis:
                tenant_data = redis.hgetall(f"{TENANT_PREFIX}{key_hash}")
                if tenant_data:
                    return self._deserialize_tenant(tenant_data)

        return None

    def _deserialize_tenant(self, data: dict) -> Tenant:
        """反序列化租户数据

        兼容 decode_responses=True (str keys) 和 decode_responses=False (bytes keys)。
        """
        def _get(key: str, default=None):
            """同时尝试 str 和 bytes 键名"""
            if key in data:
                return data[key]
            if isinstance(key, str) and key.encode() in data:
                return data[key.encode()]
            return default

        quota = QuotaConfig(
            requests_per_minute=int(_get("quota_requests_per_minute", 60)),
            requests_per_day=int(_get("quota_requests_per_day", 10000)),
            max_tokens_per_request=int(_get("quota_max_tokens", 8192)),
            gpu_memory_mb=int(_get("quota_gpu_memory", 4096)),
            concurrent_requests=int(_get("quota_concurrent", 10)),
        )
        def _get_field(field: str, default=""):
            val = _get(field, default)
            if isinstance(val, bytes):
                return val.decode()
            return val

        tenant = Tenant(
            id=_get_field("id"),
            name=_get_field("name"),
            api_key_hash=_get_field("api_key_hash"),
            api_key_prefix=_get_field("api_key_prefix"),
            status=TenantStatus(_get_field("status")),
            quota=quota,
            priority=int(_get("priority", 5)),
        )
        now = time.time()
        with _cache_lock:
            _tenant_cache[tenant.api_key_hash] = tenant
            _tenant_cache_times[tenant.api_key_hash] = now
        return tenant
