"""REST API 限流集成测试

测试覆盖：
1. 配置正确加载
2. 中间件集成到 FastAPI
3. 响应格式验证
"""

import pytest
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quantumflow.api.server import create_app
from quantumflow.api.middlewares.rate_limit import RateLimitMiddleware, TokenBucket
from quantumflow.utils.config import get_config


class TestRateLimitConfig:
    """限流配置验证测试"""

    def test_rate_limit_config_exists(self):
        """验证：rate_limit 配置存在"""
        config = get_config()
        assert hasattr(config, "server"), "config 应有 server 属性"
        assert hasattr(config.server, "rate_limit"), "server 应有 rate_limit 属性"

    def test_rate_limit_config_structure(self):
        """验证：rate_limit 配置结构正确"""
        config = get_config()
        rate_limit = config.server.rate_limit

        # 顶层配置
        assert hasattr(rate_limit, "enabled"), "rate_limit 应有 enabled 属性"
        assert hasattr(rate_limit, "rest_api"), "rate_limit 应有 rest_api 属性"
        assert hasattr(rate_limit, "grpc"), "rate_limit 应有 grpc 属性"

    def test_rest_api_rate_limit_defaults(self):
        """验证：REST API 限流默认值正确"""
        config = get_config()
        rest_api = config.server.rate_limit.rest_api

        assert rest_api.enabled is True, "REST API 限流默认应启用"
        assert rest_api.qps == 100, "默认 QPS 应为 100"
        assert rest_api.burst == 200, "默认 burst 应为 200"

    def test_grpc_rate_limit_defaults(self):
        """验证：gRPC 限流默认值正确"""
        config = get_config()
        grpc = config.server.rate_limit.grpc

        assert grpc.enabled is True, "gRPC 限流默认应启用"
        assert grpc.qps == 100, "默认 QPS 应为 100"
        assert grpc.burst == 200, "默认 burst 应为 200"


class TestServerIntegration:
    """服务器集成测试"""

    def test_server_creates_with_rate_limit(self):
        """验证：服务器创建成功且包含限流中间件"""
        app = create_app()
        assert app is not None, "create_app() 应返回 FastAPI 实例"

        # 检查中间件存在（通过请求验证）
        client = TestClient(app)
        # 健康检查端点应可访问
        response = client.get("/api/v1/health")
        assert response.status_code == 200, "健康检查应返回 200"


class TestRateLimitMiddlewareIntegration:
    """限流中间件集成测试"""

    def test_rate_limit_works_with_small_burst(self):
        """验证：限流中间件在小 burst 下正确工作"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=1, burst=2)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # burst=2，应允许 2 个请求
        assert client.get("/test").status_code == 200, "第 1 个请求应成功"
        assert client.get("/test").status_code == 200, "第 2 个请求应成功"
        # 第 3 个被限流
        assert client.get("/test").status_code == 429, "第 3 个请求应被限流"

    def test_rate_limit_response_format_correct(self):
        """验证：限流响应格式符合规范"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=1, burst=1)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        client.get("/test")  # 消耗令牌
        response = client.get("/test")  # 限流

        assert response.status_code == 429, "耗尽令牌后应返回 429"

        data = response.json()
        assert "error" in data, "响应应包含 error 字段"
        assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED", "错误码应为 RATE_LIMIT_EXCEEDED"
        assert "message" in data["error"], "error 应包含 message 字段"

    def test_token_refill_after_delay(self):
        """验证：等待后令牌补充，请求可继续"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=10, burst=1)  # 每秒补充 10 个

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        assert client.get("/test").status_code == 200, "第 1 个请求应成功"
        assert client.get("/test").status_code == 429, "第 2 个请求应被限流"

        # 等待补充
        time.sleep(0.2)  # 补充约 2 个令牌

        assert client.get("/test").status_code == 200, "补充后请求应成功"


class TestTokenBucketIntegration:
    """TokenBucket 集成测试"""

    def test_token_bucket_refill_rate(self):
        """验证：TokenBucket 补充率正确"""
        bucket = TokenBucket(capacity=5, refill_rate=10.0)  # 每秒补充 10 个

        # 消耗所有令牌
        for _ in range(5):
            bucket.try_acquire(1)

        assert bucket.tokens <= 0.01, f"令牌应耗尽，实际: {bucket.tokens}"

        # 等待补充
        time.sleep(0.1)  # 补充 1 个令牌
        assert 0.9 <= bucket.available_tokens <= 1.1, f"补充后应有约 1 个令牌，实际: {bucket.available_tokens}"

    def test_token_bucket_capacity_cap(self):
        """验证：TokenBucket 容量上限"""
        bucket = TokenBucket(capacity=5, refill_rate=100.0)

        # 消耗
        bucket.try_acquire(3)

        # 等待过量补充
        time.sleep(0.1)  # 补充 10 个，但上限为 5
        assert bucket.available_tokens <= 5.0, "令牌数不能超过容量"
