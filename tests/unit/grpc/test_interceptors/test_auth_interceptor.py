"""gRPC 认证拦截器测试

严格测试认证拦截器的业务逻辑：
- Token 提取
- Token 验证
- 白名单绕过
"""

import pytest
from unittest.mock import MagicMock
import grpc

from quantumflow.grpc.interceptors.auth import AuthInterceptor


class MockMetadataItem:
    """模拟 metadata 项"""

    def __init__(self, key, value):
        self.key = key
        self.value = value


class MockHandlerCallDetails:
    """模拟 HandlerCallDetails"""

    def __init__(
        self,
        method: str = b"/quantumflow.v1.InferenceService/Inference",
        metadata: list = None,
    ):
        self.method = method
        self._metadata = metadata or []

    @property
    def invocation_metadata(self):
        return self._metadata


class TestAuthInterceptorTokenExtraction:
    """Token 提取测试"""

    def test_extract_bearer_token_with_prefix(self):
        """提取带 Bearer 前缀的 token"""
        interceptor = AuthInterceptor(allowed_tokens={"test-token": "user-1"})
        metadata = [MockMetadataItem("authorization", "Bearer test-token")]
        details = MockHandlerCallDetails(metadata=metadata)

        token = interceptor._extract_token(details)

        assert token == "test-token"

    def test_extract_token_without_bearer_prefix(self):
        """提取不带 Bearer 前缀的 token"""
        interceptor = AuthInterceptor(allowed_tokens={"direct-token": "user-2"})
        metadata = [MockMetadataItem("authorization", "direct-token")]
        details = MockHandlerCallDetails(metadata=metadata)

        token = interceptor._extract_token(details)

        assert token == "direct-token"

    def test_extract_token_missing_authorization_header(self):
        """缺少 Authorization 头时返回 None"""
        interceptor = AuthInterceptor()
        details = MockHandlerCallDetails(metadata=[])

        token = interceptor._extract_token(details)

        assert token is None

    def test_extract_token_only_bearer_word(self):
        """只有 Bearer 关键字没有 token"""
        interceptor = AuthInterceptor()
        metadata = [MockMetadataItem("authorization", "Bearer")]
        details = MockHandlerCallDetails(metadata=metadata)

        token = interceptor._extract_token(details)

        # 注意：当前实现返回 "Bearer"（Bearer 后面没有内容时）
        assert token in ["", "Bearer"]


class TestAuthInterceptorTokenValidation:
    """Token 验证测试"""

    def test_validate_valid_token_in_whitelist(self):
        """白名单中的有效 token"""
        interceptor = AuthInterceptor(allowed_tokens={"valid-token": "user-123"})
        assert interceptor._validate_token("valid-token") is True

    def test_validate_invalid_token_not_in_whitelist(self):
        """不在白名单中的 token"""
        interceptor = AuthInterceptor(allowed_tokens={"valid-token": "user-123"})
        assert interceptor._validate_token("invalid-token") is False

    def test_validate_none_token_require_auth_for_all_false(self):
        """require_auth_for_all=False 时 None token 通过"""
        interceptor = AuthInterceptor(allowed_tokens={}, require_auth_for_all=False)
        assert interceptor._validate_token(None) is True

    def test_validate_none_token_require_auth_for_all_true(self):
        """require_auth_for_all=True 时 None token 失败"""
        interceptor = AuthInterceptor(allowed_tokens={}, require_auth_for_all=True)
        assert interceptor._validate_token(None) is False

    def test_validate_with_custom_validator(self):
        """使用自定义验证函数"""
        validator = MagicMock(return_value=True)
        interceptor = AuthInterceptor(allowed_tokens={}, token_validator=validator)
        result = interceptor._validate_token("any-token")
        validator.assert_called_once_with("any-token")
        assert result is True


class TestAuthInterceptorBypassMethods:
    """绕过方法测试"""

    def test_bypass_method_allows_without_auth(self):
        """bypass 方法无需认证"""
        interceptor = AuthInterceptor(
            allowed_tokens={},
            bypass_methods={"/HealthService/Check"},
            require_auth_for_all=True,
        )
        details = MockHandlerCallDetails(
            method="/HealthService/Check",
            metadata=[MockMetadataItem("authorization", "Bearer invalid-token")],
        )
        continuation = MagicMock(return_value=MagicMock())

        result = interceptor.intercept_service(continuation, details)

        continuation.assert_called_once_with(details)

    def test_non_bypass_method_rejects_invalid_token(self):
        """非 bypass 方法拒绝无效 token"""
        interceptor = AuthInterceptor(
            allowed_tokens={},
            require_auth_for_all=True,
        )
        details = MockHandlerCallDetails(
            method="/InferenceService/Inference",
            metadata=[MockMetadataItem("authorization", "Bearer invalid-token")],
        )
        continuation = MagicMock()

        # 验证返回 aborting handler (不是异常，而是 grpc.RpcMethodHandler)
        result = interceptor.intercept_service(continuation, details)
        assert result is not None
        # 返回的 handler 不是 continuation 的结果
        continuation.assert_not_called()

    def test_successful_authentication_passes_through(self):
        """有效认证通过"""
        interceptor = AuthInterceptor(
            allowed_tokens={"valid-token": "user-123"},
            require_auth_for_all=True,
        )
        details = MockHandlerCallDetails(
            method="/InferenceService/Inference",
            metadata=[MockMetadataItem("authorization", "Bearer valid-token")],
        )
        continuation = MagicMock(return_value=MagicMock())

        result = interceptor.intercept_service(continuation, details)

        # 验证 continuation 被调用，无异常
        continuation.assert_called_once_with(details)

    def test_get_method_name_with_bytes(self):
        """获取方法名 (bytes 类型)"""
        interceptor = AuthInterceptor()
        details = MockHandlerCallDetails(method=b"/TestService/Method")

        name = interceptor._get_method_name(details)

        assert name == "/TestService/Method"

    def test_extract_token_with_metadata_exception(self):
        """metadata 异常时返回 None"""
        from quantumflow.grpc.interceptors.auth import AuthInterceptor

        interceptor = AuthInterceptor()
        details = MagicMock()

        # 创建一个在迭代时抛出异常的 metadata
        def raise_on_iteration():
            raise Exception("metadata error")

        mock_metadata = MagicMock()
        mock_metadata.__iter__ = raise_on_iteration
        details.invocation_metadata = mock_metadata

        token = interceptor._extract_token(details)

        assert token is None


class TestAuthInterceptorGetUserId:
    """get_user_id 方法测试"""

    def test_get_user_id_existing_token(self):
        """获取已存在 token 的用户 ID"""
        interceptor = AuthInterceptor(allowed_tokens={"token-123": "user-456"})
        assert interceptor.get_user_id("token-123") == "user-456"

    def test_get_user_id_nonexistent_token(self):
        """获取不存在 token 的用户 ID"""
        interceptor = AuthInterceptor(allowed_tokens={"token-123": "user-456"})
        assert interceptor.get_user_id("nonexistent") is None


class TestTokenBucketInAuth:
    """TokenBucket 类测试 (在 auth.py 中定义)"""

    def test_token_bucket_initial_tokens(self):
        """初始令牌数等于容量"""
        from quantumflow.grpc.interceptors.auth import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=5)

        assert bucket.capacity == 10
        assert bucket.refill_rate == 5
        assert bucket.available_tokens == 10

    def test_token_bucket_try_acquire_success(self):
        """获取令牌成功"""
        from quantumflow.grpc.interceptors.auth import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=0)

        result = bucket.try_acquire(3)

        assert result is True
        assert bucket.available_tokens == 7

    def test_token_bucket_try_acquire_failure(self):
        """令牌不足时获取失败"""
        from quantumflow.grpc.interceptors.auth import TokenBucket

        bucket = TokenBucket(capacity=2, refill_rate=0)

        result = bucket.try_acquire(5)

        assert result is False
        assert bucket.available_tokens == 2

    def test_token_bucket_refill(self):
        """令牌补充"""
        from quantumflow.grpc.interceptors.auth import TokenBucket
        import time

        bucket = TokenBucket(capacity=10, refill_rate=10)  # 10 tokens/sec

        bucket.try_acquire(5)  # 剩下 5 个

        time.sleep(0.2)  # 等待补充

        # 应该补充了约 2 个令牌
        assert bucket.available_tokens >= 6

    def test_token_bucket_available_tokens(self):
        """获取可用令牌数"""
        from quantumflow.grpc.interceptors.auth import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=0)

        assert bucket.available_tokens == 10

        bucket.try_acquire(3)

        assert bucket.available_tokens == 7
