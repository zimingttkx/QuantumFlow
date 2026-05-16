"""Storage测试"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

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
        assert request.result is None
        assert request.error is None
        assert request.metadata == {}

    def test_create_request_with_metadata(self):
        """测试带元数据的请求"""
        request = QueuedRequest(
            request_id="req-456",
            model_name="model-x",
            prompt="Test",
            priority=7,
            created_at=datetime.now(),
            metadata={"user_id": "user-1", "session": "session-1"},
        )

        assert request.metadata["user_id"] == "user-1"
        assert request.metadata["session"] == "session-1"

    def test_to_json(self):
        """测试JSON序列化"""
        now = datetime.now()
        request = QueuedRequest(
            request_id="req-123",
            model_name="test-model",
            prompt="Hello",
            priority=5,
            created_at=now,
        )

        json_str = request.to_json()
        data = json.loads(json_str)

        assert data["request_id"] == "req-123"
        assert data["model_name"] == "test-model"
        assert data["prompt"] == "Hello"
        assert data["priority"] == 5
        assert data["created_at"] == now.isoformat()

    def test_to_json_with_result(self):
        """测试带结果的JSON序列化"""
        now = datetime.now()
        request = QueuedRequest(
            request_id="req-789",
            model_name="test",
            prompt="Hi",
            priority=3,
            created_at=now,
            result={"output": "Hello, World!"},
            scheduled_at=now,
            completed_at=now,
        )

        json_str = request.to_json()
        data = json.loads(json_str)

        assert data["result"]["output"] == "Hello, World!"
        assert data["scheduled_at"] is not None
        assert data["completed_at"] is not None

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
        assert request.prompt == "Test prompt"
        assert request.priority == 7

    def test_from_json_with_error(self):
        """测试带错误的反序列化"""
        json_str = json.dumps({
            "request_id": "req-error",
            "model_name": "model",
            "prompt": "test",
            "priority": 5,
            "created_at": datetime.now().isoformat(),
            "scheduled_at": None,
            "completed_at": None,
            "result": None,
            "error": "Model not found",
            "metadata": {"retry_count": 1},
        })

        request = QueuedRequest.from_json(json_str)

        assert request.error == "Model not found"
        assert request.metadata["retry_count"] == 1

    def test_roundtrip_serialization(self):
        """测试往返序列化"""
        original = QueuedRequest(
            request_id="req-roundtrip",
            model_name="model-t",
            prompt="Test roundtrip",
            priority=8,
            created_at=datetime.now(),
            metadata={"key": "value"},
        )

        json_str = original.to_json()
        restored = QueuedRequest.from_json(json_str)

        assert restored.request_id == original.request_id
        assert restored.model_name == original.model_name
        assert restored.prompt == original.prompt
        assert restored.priority == original.priority
        assert restored.metadata == original.metadata


class TestQueuePriority:
    """QueuePriority测试"""

    def test_priority_values(self):
        """测试优先级值"""
        assert QueuePriority.LOW.value == 0
        assert QueuePriority.NORMAL.value == 5
        assert QueuePriority.HIGH.value == 7
        assert QueuePriority.CRITICAL.value == 9

    def test_priority_ordering(self):
        """测试优先级排序"""
        priorities = [
            QueuePriority.CRITICAL,
            QueuePriority.HIGH,
            QueuePriority.NORMAL,
            QueuePriority.LOW,
        ]

        values = [p.value for p in priorities]
        assert values == sorted(values, reverse=True)


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
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.zcard = AsyncMock(return_value=0)
        redis_mock.zcount = AsyncMock(return_value=0)
        redis_mock.hincrby = AsyncMock(return_value=1)
        redis_mock.hgetall = AsyncMock(return_value={})
        redis_mock.close = AsyncMock()
        redis_mock.zrange = AsyncMock(return_value=[])
        redis_mock.bzpopmin = AsyncMock(return_value=None)
        return redis_mock

    @pytest.fixture
    def queue(self, mock_redis):
        """创建RedisQueue实例"""
        with patch("redis.asyncio.Redis", return_value=mock_redis):
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

    def test_create_queue_custom_config(self):
        """测试自定义配置"""
        queue = RedisQueue(
            redis_url="redis://localhost:6380/1",
            default_ttl=7200,
            max_retries=5,
        )
        assert queue.redis_url == "redis://localhost:6380/1"
        assert queue.default_ttl == 7200
        assert queue.max_retries == 5

    def test_create_request(self):
        """测试创建请求的静态方法"""
        request = RedisQueue.create_request(
            model_name="test-model",
            prompt="Test prompt",
            priority=5,
            custom_field="value",
        )

        assert request.model_name == "test-model"
        assert request.prompt == "Test prompt"
        assert request.priority == 5
        assert request.metadata["custom_field"] == "value"
        assert request.request_id is not None
        assert request.created_at is not None

    def test_queue_not_connected(self):
        """测试未连接状态"""
        queue = RedisQueue()
        assert queue.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_redis):
        """测试连接成功"""
        with patch("redis.asyncio.Redis", return_value=mock_redis):
            queue = RedisQueue()
            result = await queue.connect()

            assert result is True
            assert queue.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """测试连接失败"""
        with patch("redis.asyncio.Redis", side_effect=Exception("Connection failed")):
            queue = RedisQueue()
            result = await queue.connect()

            assert result is False
            assert queue.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, queue, mock_redis):
        """测试断开连接"""
        await queue.disconnect()

        assert queue.is_connected is False
        mock_redis.close.assert_called_once()

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
    async def test_enqueue_not_connected(self):
        """测试未连接时入队"""
        queue = RedisQueue()
        request = QueuedRequest(
            request_id="req-123",
            model_name="test-model",
            prompt="Hello",
            priority=5,
            created_at=datetime.now(),
        )

        result = await queue.enqueue(request)
        assert result is False

    @pytest.mark.asyncio
    async def test_enqueue_with_priority(self, queue, mock_redis):
        """测试不同优先级入队"""
        for priority in [0, 5, 7, 9]:
            request = QueuedRequest(
                request_id=f"req-p{priority}",
                model_name="model",
                prompt="test",
                priority=priority,
                created_at=datetime.now(),
            )
            result = await queue.enqueue(request)
            assert result is True

        assert mock_redis.zadd.call_count == 4

    @pytest.mark.asyncio
    async def test_dequeue_empty(self, queue, mock_redis):
        """测试空队列出队"""
        mock_redis.zpopmin = AsyncMock(return_value=[])

        result = await queue.dequeue(timeout=0)

        assert result is None

    @pytest.mark.asyncio
    async def test_dequeue_with_data(self, queue, mock_redis):
        """测试有数据的出队"""
        request = QueuedRequest(
            request_id="req-456",
            model_name="test",
            prompt="Hello",
            priority=5,
            created_at=datetime.now(),
        )

        mock_redis.zpopmin = AsyncMock(return_value=[("req-456", -5.0)])
        mock_redis.get = AsyncMock(return_value=request.to_json())

        result = await queue.dequeue(timeout=0)

        assert result is not None
        assert result.request_id == "req-456"
        assert result.scheduled_at is not None

    @pytest.mark.asyncio
    async def test_dequeue_blocking(self, queue, mock_redis):
        """测试阻塞出队"""
        mock_redis.bzpopmin = AsyncMock(return_value=None)

        result = await queue.dequeue(timeout=1)

        assert result is None
        mock_redis.bzpopmin.assert_called()

    @pytest.mark.asyncio
    async def test_dequeue_batch(self, queue, mock_redis):
        """测试批量出队"""
        requests = [
            QueuedRequest(
                request_id=f"req-batch-{i}",
                model_name="model",
                prompt=f"prompt-{i}",
                priority=5,
                created_at=datetime.now(),
            )
            for i in range(3)
        ]

        # 设置模拟返回值
        mock_redis.zpopmin = AsyncMock(side_effect=[
            [(f"req-batch-{i}", -5.0) for i in range(3)],
            [],
        ])
        mock_redis.get = AsyncMock(side_effect=[r.to_json() for r in requests])

        results = await queue.dequeue_batch(batch_size=5)

        assert len(results) == 3
        assert all(r.request_id.startswith("req-batch-") for r in results)

    @pytest.mark.asyncio
    async def test_requeue(self, queue, mock_redis):
        """测试重新入队"""
        request = QueuedRequest(
            request_id="req-retry",
            model_name="model",
            prompt="test",
            priority=5,
            created_at=datetime.now(),
        )

        result = await queue.requeue(request)

        assert result is True
        assert mock_redis.zadd.called

    @pytest.mark.asyncio
    async def test_requeue_max_retries(self, queue, mock_redis):
        """测试超过最大重试次数"""
        request = QueuedRequest(
            request_id="req-max-retry",
            model_name="model",
            prompt="test",
            priority=5,
            created_at=datetime.now(),
            metadata={"retry_count": 3},
        )

        result = await queue.requeue(request, increment_retry=True)

        assert result is False

    @pytest.mark.asyncio
    async def test_queue_size(self, queue, mock_redis):
        """测试获取队列大小"""
        mock_redis.zcard = AsyncMock(return_value=10)

        size = await queue.queue_size()

        assert size == 10
        mock_redis.zcard.assert_called()

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, queue, mock_redis):
        """测试获取队列统计"""
        mock_redis.zcard = AsyncMock(return_value=5)
        mock_redis.zcount = AsyncMock(return_value=0)

        stats = await queue.get_queue_stats()

        assert "queue_size" in stats
        assert "priority_counts" in stats
        assert stats["connected"] is True
        assert stats["queue_size"] == 5

    @pytest.mark.asyncio
    async def test_store_result(self, queue, mock_redis):
        """测试存储结果"""
        result = await queue.store_result("req-123", {"output": "test"})

        assert result is True
        mock_redis.setex.assert_called()

    @pytest.mark.asyncio
    async def test_store_result_not_connected(self):
        """测试未连接时存储结果"""
        queue = RedisQueue()
        result = await queue.store_result("req-123", {"output": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_get_result(self, queue, mock_redis):
        """测试获取结果"""
        mock_redis.get = AsyncMock(return_value='{"output": "test"}')

        result = await queue.get_result("req-123")

        assert result == {"output": "test"}

    @pytest.mark.asyncio
    async def test_get_result_not_found(self, queue, mock_redis):
        """测试获取不存在的结果"""
        mock_redis.get = AsyncMock(return_value=None)

        result = await queue.get_result("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_request(self, queue, mock_redis):
        """测试获取请求详情"""
        request = QueuedRequest(
            request_id="req-get",
            model_name="model",
            prompt="test",
            priority=5,
            created_at=datetime.now(),
        )

        mock_redis.get = AsyncMock(return_value=request.to_json())

        result = await queue.get_request("req-get")

        assert result is not None
        assert result.request_id == "req-get"

    @pytest.mark.asyncio
    async def test_clear_queue(self, queue, mock_redis):
        """测试清空队列"""
        mock_redis.zrange = AsyncMock(return_value=["req-1", "req-2", "req-3"])

        result = await queue.clear_queue()

        assert result is True
        mock_redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_get_metrics(self, queue, mock_redis):
        """测试获取指标"""
        mock_redis.hgetall = AsyncMock(return_value={
            "enqueued": "100",
            "completed": "90",
            "failed": "10",
        })

        metrics = await queue.get_metrics()

        assert metrics["enqueued"] == 100
        assert metrics["completed"] == 90
        assert metrics["failed"] == 10


class TestRedisQueueIntegration:
    """RedisQueue集成测试"""

    @pytest.mark.asyncio
    async def test_full_enqueue_dequeue_cycle(self):
        """测试完整的入队-出队周期"""
        # 这个测试需要真实的Redis连接
        # 在CI环境中跳过
        pytest.skip("需要Redis服务器")

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """测试优先级排序"""
        pytest.skip("需要Redis服务器")

    @pytest.mark.asyncio
    async def test_result_persistence(self):
        """测试结果持久化"""
        pytest.skip("需要Redis服务器")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """测试并发操作"""
        pytest.skip("需要Redis服务器")
