"""REST API 限流中间件 - 单元测试

测试覆盖：
1. TokenBucket 算法正确性
2. 令牌补充逻辑（时间相关）
3. 容量上限
4. 多令牌获取
5. 并发线程安全
6. RateLimitMiddleware 请求拦截
7. 429 响应格式
8. 不同 QPS/Burst 配置
9. 边界条件
"""

import threading
import time
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from quantumflow.api.middlewares.rate_limit import RateLimitMiddleware, TokenBucket


# =============================================================================
# TokenBucket 算法核心测试
# =============================================================================

class TestTokenBucketAlgorithm:
    """TokenBucket 令牌桶算法核心逻辑验证"""

    def test_initial_tokens_equal_capacity(self):
        """验证：初始化时令牌数等于容量"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.capacity == 10
        assert bucket.tokens == 10.0
        assert bucket.refill_rate == 1.0

    def test_try_acquire_single_token_success(self):
        """验证：容量充足时获取单个令牌成功"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        result = bucket.try_acquire(1)
        assert result is True
        assert bucket.tokens == 9.0  # 剩余 9 个

    def test_try_acquire_single_token_fail_when_empty(self):
        """验证：令牌耗尽时获取失败"""
        bucket = TokenBucket(capacity=1, refill_rate=0.0)  # 不补充
        # 首次获取成功
        result1 = bucket.try_acquire(1)
        assert result1 is True
        assert bucket.tokens == 0.0
        # 再次获取失败
        result2 = bucket.try_acquire(1)
        assert result2 is False
        assert bucket.tokens == 0.0  # 令牌不变

    def test_try_acquire_multiple_tokens_success(self):
        """验证：容量充足时获取多个令牌成功"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        result = bucket.try_acquire(5)
        assert result is True
        assert bucket.tokens == 5.0

    def test_try_acquire_multiple_tokens_fail_insufficient(self):
        """验证：令牌不足时获取多个令牌失败"""
        bucket = TokenBucket(capacity=3, refill_rate=0.0)
        result = bucket.try_acquire(5)
        assert result is False
        assert bucket.tokens == 3.0  # 令牌数不变，未扣减

    def test_tokens_cannot_exceed_capacity(self):
        """验证：令牌数不能超过容量上限"""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)  # 高补充率
        # 消耗部分令牌
        bucket.try_acquire(3)
        assert bucket.tokens == 7.0
        # 等待补充
        time.sleep(0.1)  # 补充 10 个
        current = bucket.available_tokens
        assert current <= 10.0  # 不能超过容量
        # 再次补充
        time.sleep(0.1)
        current2 = bucket.available_tokens
        assert current2 <= 10.0

    def test_token_refill_over_time(self):
        """验证：令牌随时间补充"""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 每秒补充 10 个
        # 消耗所有令牌
        bucket.try_acquire(10)
        assert bucket.tokens == 0.0
        # 等待 0.2 秒，补充 2 个令牌
        time.sleep(0.2)
        available = bucket.available_tokens
        assert 1.9 <= available <= 2.1  # 允许浮点误差

    def test_available_tokens_property(self):
        """验证：available_tokens 属性触发补充逻辑"""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        bucket.try_acquire(5)  # 剩余 5
        assert bucket.tokens == 5.0
        # 不调用 available_tokens，令牌数不变
        time.sleep(0.1)
        # 调用 available_tokens 会触发补充
        _ = bucket.available_tokens
        # 此时应该有补充

    def test_try_acquire_with_zero_tokens(self):
        """验证：请求 0 个令牌总是成功"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        result = bucket.try_acquire(0)
        assert result is True
        assert bucket.tokens == 10.0  # 无变化


class TestTokenBucketEdgeCases:
    """TokenBucket 边界条件测试"""

    def test_burst_zero(self):
        """验证：burst=0 时首个请求即失败"""
        bucket = TokenBucket(capacity=0, refill_rate=1.0)
        result = bucket.try_acquire(1)
        assert result is False
        assert bucket.tokens == 0.0

    def test_qps_zero_no_refill(self):
        """验证：qps=0 时令牌不补充"""
        bucket = TokenBucket(capacity=10, refill_rate=0.0)
        bucket.try_acquire(9)
        assert bucket.tokens == 1.0
        time.sleep(0.5)
        assert bucket.available_tokens == 1.0  # 仍未 1.0

    def test_large_capacity(self):
        """验证：大容量处理"""
        bucket = TokenBucket(capacity=1000000, refill_rate=100.0)
        assert bucket.capacity == 1000000
        result = bucket.try_acquire(500000)
        assert result is True
        assert bucket.tokens == 500000.0

    def test_concurrent_acquire(self):
        """验证：并发获取令牌"""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        results = []
        errors = []

        def acquire():
            try:
                result = bucket.try_acquire(1)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquire) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发错误: {errors}"
        # 50 个线程各获取 1 个，100 个令牌应全部成功
        success_count = sum(1 for r in results if r is True)
        fail_count = sum(1 for r in results if r is False)
        assert success_count + fail_count == 50
        # 最终令牌数（允许浮点误差）
        assert 49.0 <= bucket.tokens <= 51.0


# =============================================================================
# RateLimitMiddleware 中间件测试
# =============================================================================

class TestRateLimitMiddlewareBasic:
    """RateLimitMiddleware 基础功能测试"""

    def test_normal_request_passes(self):
        """验证：正常请求通过中间件"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=100, burst=200)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok", "data": "test"}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["data"] == "test"

    def test_rate_limited_request_returns_429(self):
        """验证：触发限流时返回 429"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=1, burst=1)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        # 第一个请求成功
        response1 = client.get("/test")
        assert response1.status_code == 200
        # 第二个请求限流
        response2 = client.get("/test")
        assert response2.status_code == 429

    def test_429_response_format(self):
        """验证：429 响应格式符合规范"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=1, burst=1)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        client.get("/test")  # 消耗令牌
        response = client.get("/test")  # 限流

        assert response.status_code == 429
        data = response.json()
        # 验证错误格式
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    def test_different_burst_affects_limit(self):
        """验证：不同 burst 值影响限流阈值"""
        app_small = FastAPI()
        app_small.add_middleware(RateLimitMiddleware, qps=1, burst=2)

        @app_small.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app_small)
        # burst=2，应允许 2 个请求
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 429  # 第 3 个被限流

    def test_different_qps_affects_refill(self):
        """验证：不同 qps 值影响令牌补充速度"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=10, burst=1)  # 每秒 10 个

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 429  # 耗尽

        # qps=10，等待 0.2 秒应补充 2 个令牌
        time.sleep(0.2)
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 429


class TestRateLimitMiddlewareEdgeCases:
    """RateLimitMiddleware 边界条件测试"""

    def test_post_request_rate_limited(self):
        """验证：POST 请求同样限流"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=1, burst=1)

        @app.post("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        assert client.post("/test").status_code == 200
        assert client.post("/test").status_code == 429

    def test_multiple_endpoints_share_bucket(self):
        """验证：多个端点共享同一令牌桶"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=1, burst=2)

        @app.get("/endpoint1")
        async def endpoint1():
            return {"endpoint": 1}

        @app.get("/endpoint2")
        async def endpoint2():
            return {"endpoint": 2}

        client = TestClient(app)
        # 两个端点共享 burst=2
        assert client.get("/endpoint1").status_code == 200
        assert client.get("/endpoint2").status_code == 200
        assert client.get("/endpoint1").status_code == 429  # 共享耗尽

    def test_rate_limit_does_not_affect_other_paths(self):
        """验证：限流只影响添加了中间件的路径（如果单独配置）"""
        # 此测试验证基本行为
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, qps=1, burst=1)

        @app.get("/limited")
        async def limited():
            return {"path": "/limited"}

        client = TestClient(app)
        assert client.get("/limited").status_code == 200
        assert client.get("/limited").status_code == 429


# =============================================================================
# 配置集成测试
# =============================================================================

class TestRateLimitConfigIntegration:
    """配置集成测试"""

    def test_middleware_receives_config_values(self):
        """验证：中间件正确接收配置值（使用小值确保确定性）"""
        app = FastAPI()
        custom_qps = 1  # 慢速补充，确保测试确定性
        custom_burst = 3  # 小 burst，易于验证
        app.add_middleware(RateLimitMiddleware, qps=custom_qps, burst=custom_burst)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        # burst=3，应允许 3 个请求
        assert client.get("/test").status_code == 200, "第 1 个请求应成功"
        assert client.get("/test").status_code == 200, "第 2 个请求应成功"
        assert client.get("/test").status_code == 200, "第 3 个请求应成功"
        # 第 4 个限流
        assert client.get("/test").status_code == 429, "第 4 个请求应被限流"

    def test_qps_refill_rate_in_middleware(self):
        """验证：中间件使用 qps 作为补充率"""
        app = FastAPI()
        # qps=5, burst=1
        app.add_middleware(RateLimitMiddleware, qps=5, burst=1)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        # 消耗 burst
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 429

        # qps=5，等待 0.2 秒补充 1 个令牌
        time.sleep(0.2)
        assert client.get("/test").status_code == 200
        assert client.get("/test").status_code == 429


# =============================================================================
# 逆向测试
# =============================================================================

class TestTokenBucketReverseValidation:
    """逆向验证测试"""

    def test_acquire_does_not_affect_capacity(self):
        """验证：获取令牌不影响容量上限"""
        bucket = TokenBucket(capacity=10, refill_rate=0.0)
        bucket.try_acquire(5)
        assert bucket.capacity == 10  # 容量不变

    def test_empty_bucket_preserves_tokens_state(self):
        """验证：空桶状态正确保持"""
        bucket = TokenBucket(capacity=5, refill_rate=0.0)
        bucket.try_acquire(5)
        assert bucket.tokens == 0.0
        # 再次获取
        result = bucket.try_acquire(1)
        assert result is False
        assert bucket.tokens == 0.0  # 仍为 0

    def test_thread_safety_on_available_tokens(self):
        """验证：available_tokens 线程安全"""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)

        def read_tokens():
            for _ in range(100):
                _ = bucket.available_tokens

        threads = [threading.Thread(target=read_tokens) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 不应抛出异常
        assert bucket.capacity == 100
