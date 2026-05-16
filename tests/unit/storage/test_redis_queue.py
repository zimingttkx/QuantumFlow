"""Storage测试"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock

from quantumflow.storage import RedisQueue, QueuedRequest, QueuePriority


class TestQueuedRequest:
    """QueuedRequest测试"""

    def test_create_request(self):
        """测试创建请求"""
        request = QueuedRequest(
            request_id="req-123",
            model_name="test-model",
            prompt="Hello world",
            priority=5,
            created_at=datetime.now(),
        )
        assert request.request_id == "req-123"
        assert request.model_name == "test-model"
        assert request.prompt == "Hello world"
        assert request.priority == 5

    def test_to_json(self):
        """测试JSON序列化"""
        request = QueuedRequest(
            request_id="req-123",
            model_name="test-model",
            prompt="Hello",
            priority=5,
            created_at=datetime.now(),
        )
        json_str = request.to_json()
        data = json.loads(json_str)
        assert data["request_id"] == "req-123"
        assert data["model_name"] == "test-model"

    def test_from_json(self):
        """测试JSON反序列化"""
        json_str = json.dumps({
            "request_id": "req-456",
            "model_name": "model-x",
            "prompt": "Test prompt",
            "priority": 7,
            "created_at": datetime.now().isoformat(),
            "scheduled_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "metadata": {},
        })
        request = QueuedRequest.from_json(json_str)
        assert request.request_id == "req-456"
        assert request.model_name == "model-x"


class TestQueuePriority:
    """QueuePriority测试"""

    def test_priority_values(self):
        """测试优先级值"""
        assert QueuePriority.LOW.value == 0
        assert QueuePriority.NORMAL.value == 5
        assert QueuePriority.HIGH.value == 7
        assert QueuePriority.CRITICAL.value == 9


class TestRedisQueue:
    """RedisQueue测试"""

    @pytest.fixture
    def mock_redis(self):
        """创建模拟Redis"""
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock(return_value=True)
        redis_mock.setex = AsyncMock(return_value=True)
        redis_mock.zadd = AsyncMock(return_value=1)
        redis_mock.zpopmin = AsyncMock(return_value=[])
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.zcard = AsyncMock(return_value=0)
        redis_mock.close = AsyncMock()
        return redis_mock

    @pytest.fixture
    def queue(self, mock_redis):
        """创建RedisQueue实例"""
        queue = RedisQueue(redis_url="redis://localhost:6379/0")
        queue._redis = mock_redis
        queue._connected = True
        return queue

    def test_create_queue(self):
        """测试创建队列"""
        queue = RedisQueue()
        assert queue.redis_url == "redis://localhost:6379/0"
        assert queue.default_ttl == 3600
        assert queue.max_retries == 3
        assert queue.is_connected is False

    def test_create_request(self):
        """测试创建请求"""
        request = RedisQueue.create_request(
            model_name="test-model",
            prompt="Test prompt",
            priority=5,
        )
        assert request.model_name == "test-model"
        assert request.prompt == "Test prompt"
        assert request.priority == 5
        assert request.request_id is not None

    @pytest.mark.asyncio
    async def test_enqueue(self, queue, mock_redis):
        """测试入队"""
        request = QueuedRequest(
            request_id="req-123",
            model_name="test-model",
            prompt="Hello",
            priority=5,
            created_at=datetime.now(),
        )
        result = await queue.enqueue(request)
        assert result is True
        mock_redis.setex.assert_called()
        mock_redis.zadd.assert_called()

    @pytest.mark.asyncio
    async def test_dequeue_empty(self, queue, mock_redis):
        """测试空队列出队"""
        mock_redis.zpopmin = AsyncMock(return_value=[])
        result = await queue.dequeue(timeout=0)
        assert result is None

    @pytest.mark.asyncio
    async def test_queue_size(self, queue, mock_redis):
        """测试获取队列大小"""
        mock_redis.zcard = AsyncMock(return_value=10)
        size = await queue.queue_size()
        assert size == 10

    @pytest.mark.asyncio
    async def test_store_result(self, queue, mock_redis):
        """测试存储结果"""
        result = await queue.store_result("req-123", {"output": "test"})
        assert result is True
