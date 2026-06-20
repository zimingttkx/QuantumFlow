"""gRPC 认证拦截器

提供 API Key / Token 认证功能：
- 验证 Authorization header
- 支持 Bearer token 格式
- 可配置的白名单（跳过特定方法的认证）
- 过期 token 支持
"""

import threading
import time
from typing import Callable, Dict, Optional, Set

import grpc


class AuthInterceptor(grpc.ServerInterceptor):
    """认证拦截器 - 验证 API Key/Token

    功能:
    - 验证 Authorization header
    - 支持 Bearer token 格式
    - 可配置跳过认证的方法列表
    - 支持 token 过期检查（需要传入过期检查函数）
    """

    def __init__(
        self,
        allowed_tokens: Optional[Dict[str, str]] = None,
        token_validator: Optional[Callable[[str], bool]] = None,
        bypass_methods: Optional[Set[str]] = None,
        require_auth_for_all: bool = False,
    ):
        """
        Args:
            allowed_tokens: token -> user_id 的映射字典
            token_validator: 可选的 token 验证函数，接受 token 返回 bool
            bypass_methods: 跳过认证的方法名集合（如 {"/HealthService/Check"}）
            require_auth_for_all: 是否对所有方法要求认证（True=默认拒绝，False=默认允许）
        """
        self.allowed_tokens = allowed_tokens or {}
        self.token_validator = token_validator
        self.bypass_methods = bypass_methods or set()
        self.require_auth_for_all = require_auth_for_all

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """拦截并验证认证信息"""
        method_name = self._get_method_name(handler_call_details)

        # 检查是否跳过认证
        if method_name in self.bypass_methods:
            return continuation(handler_call_details)

        # 提取 token
        token = self._extract_token(handler_call_details)

        # 验证 token
        if not self._validate_token(token):
            # 认证失败
            if self.require_auth_for_all:
                return self._create_aborting_handler(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Invalid or missing authentication credentials",
                )

        return continuation(handler_call_details)

    def _create_aborting_handler(
        self, status_code: grpc.StatusCode, details: str
    ) -> grpc.RpcMethodHandler:
        """创建一个会中止调用的 RpcMethodHandler"""
        def aborting_handler(request, context):
            context.abort(status_code, details)

        return grpc.unary_unary_rpc_method_handler(
            aborting_handler,
            request_deserializer=None,
            response_serializer=None,
        )

    def _get_method_name(self, handler_call_details: grpc.HandlerCallDetails) -> str:
        """获取方法名"""
        method = handler_call_details.method
        if isinstance(method, bytes):
            return method.decode()
        return method or ""

    def _extract_token(self, handler_call_details: grpc.HandlerCallDetails) -> Optional[str]:
        """从 metadata 中提取 token

        支持两种格式:
        - Authorization: Bearer <token>
        - Authorization: <token> (无 Bearer 前缀)
        """
        try:
            if handler_call_details.invocation_metadata:
                for item in handler_call_details.invocation_metadata:
                    if item.key.lower() == "authorization":
                        value = item.value
                        # 移除 "Bearer " 前缀
                        if value.startswith("Bearer "):
                            return value[7:]
                        return value
        except Exception:
            pass
        return None

    def _validate_token(self, token: Optional[str]) -> bool:
        """验证 token

        Returns:
            True 如果 token 有效或为空（不要求认证）
        """
        # 无 token
        if not token:
            if self.require_auth_for_all:
                return False
            return True

        # 如果有 token_validator，使用它验证
        if self.token_validator:
            return self.token_validator(token)

        # 否则检查是否在白名单中
        if token in self.allowed_tokens:
            return True

        return False

    def get_user_id(self, token: str) -> Optional[str]:
        """根据 token 获取用户 ID

        Args:
            token: 认证 token

        Returns:
            用户 ID，如果 token 无效则返回 None
        """
        return self.allowed_tokens.get(token)


class TokenBucket:
    """令牌桶算法实现

    用于限流等场景。
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: 桶的最大容量
            refill_rate: 每秒补充的令牌数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self, tokens: int = 1) -> bool:
        """尝试获取令牌

        Args:
            tokens: 要获取的令牌数

        Returns:
            True 如果获取成功，False 如果令牌不足
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self._lock:
            self._refill()
            return self._tokens
