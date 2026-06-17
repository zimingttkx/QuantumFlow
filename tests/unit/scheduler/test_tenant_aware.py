"""租户感知调度测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from quantumflow.scheduler.distributed import DistributedScheduler
from quantumflow.scheduler.strategy.base import SchedulingRequest


def test_scheduling_request_with_tenant_id():
    """测试带租户 ID 的调度请求"""
    request = SchedulingRequest(
        request_id="req-1",
        model="test-model",
        prompt="test",
        tenant_id="tenant-abc",
        priority=8
    )
    assert request.tenant_id == "tenant-abc"


def test_default_tenant_id():
    """测试默认租户 ID"""
    request = SchedulingRequest(
        request_id="req-1",
        model="test-model",
        prompt="test"
    )
    assert request.tenant_id == "default"


@pytest.mark.asyncio
async def test_submit_respects_concurrent_limit():
    """测试提交请求时尊重并发限制

    Bug fix (H-C5): 原本用 check-then-incr 存在 TOCTOU 竞态。
    修复后: 用原子 check-and-incr (Lua 脚本),返回 -1 表示超限。
    本测试 mock 原子 incr 方法返回 -1 来模拟超限场景。
    """
    with patch("quantumflow.scheduler.distributed.get_redis_manager") as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value.get_client.return_value = mock_client
        mock_client.get.return_value = b"10"  # 10 concurrent requests

        scheduler = DistributedScheduler()
        scheduler._get_tenant_quota = MagicMock(return_value=MagicMock(
            concurrent_requests=10,
            gpu_memory_mb=0
        ))
        # Bug fix (H-C5): 原子 incr 模拟超限
        scheduler._try_increment_concurrent_requests = MagicMock(return_value=-1)
        scheduler._get_concurrent_requests = AsyncMock(return_value=10)

        request = MagicMock()
        request.model = "test-model"
        request.prompt = "test"

        with pytest.raises(Exception) as exc_info:
            await scheduler.submit(request, tenant_id="test-tenant")
        assert "并发请求数超限" in str(exc_info.value)


@pytest.mark.asyncio
async def test_submit_with_valid_tenant():
    """测试有效租户的请求提交"""
    with patch("quantumflow.scheduler.distributed.get_redis_manager") as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value.get_client.return_value = mock_client
        mock_client.get.return_value = None
        mock_client.hgetall.return_value = {}

        scheduler = DistributedScheduler()
        scheduler._get_tenant_quota = MagicMock(return_value=MagicMock(
            concurrent_requests=10,
            gpu_memory_mb=0
        ))
        scheduler._get_concurrent_requests = AsyncMock(return_value=0)

        request = MagicMock()
        request.request_id = "req-123"
        request.model = "test-model"
        request.prompt = "test"
        request.priority = 5
        request.tenant_id = "test-tenant"
        request.prompt_tokens = 0
        request.max_tokens = 512

        # Mock the pending_queue
        scheduler.pending_queue = AsyncMock()
        scheduler._request_counter = 0
        scheduler._use_redis = False  # Force fallback to memory queue

        result = await scheduler.submit(request, tenant_id="test-tenant")
        assert result == "req-123"


@pytest.mark.asyncio
async def test_submit_increments_counter():
    """测试提交请求后计数器递增

    Bug fix (H-C5): 原本用 _increment_concurrent_requests (无原子保证)。
    修复后: 用 _try_increment_concurrent_requests 原子 check-and-incr。
    """
    scheduler = DistributedScheduler()
    scheduler._get_tenant_quota = MagicMock(return_value=MagicMock(
        concurrent_requests=10,
        gpu_memory_mb=0
    ))
    scheduler._get_concurrent_requests = AsyncMock(return_value=0)
    # Bug fix (H-C5): 用原子 incr 替代
    scheduler._try_increment_concurrent_requests = MagicMock(return_value=1)

    # Mock Redis queue
    scheduler._use_redis = True
    scheduler._redis_queue = AsyncMock()
    scheduler._redis_queue.enqueue = AsyncMock(return_value=True)

    request = MagicMock()
    request.request_id = "req-123"
    request.model = "test-model"
    request.prompt = "test"
    request.priority = 5
    request.tenant_id = "test-tenant"
    request.prompt_tokens = 0
    request.max_tokens = 512

    result = await scheduler.submit(request, tenant_id="test-tenant")
    assert result == "req-123"

    # Bug fix (H-C5): 验证原子 incr 被调用
    scheduler._try_increment_concurrent_requests.assert_called_once_with(
        "test-tenant", 10
    )
