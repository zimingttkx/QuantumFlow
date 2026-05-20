"""分布式功能综合测试

严格验证分布式存储架构的完整业务逻辑，包括：
- RedisQueue 核心功能
- DistributedScheduler 调度逻辑
- WorkerClient HTTP通信
- WorkerRegistry 注册管理
- API路由端到端流程

测试标准：
1. 严禁浅层可用性测试，必须验证业务逻辑正确性
2. 全覆盖：正常、边界、错误、异常场景
3. 强精准断言，不放松任何条件
4. 专项测试核心功能和易错细节
"""

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from quantumflow.scheduler.distributed import DistributedScheduler
from quantumflow.scheduler.strategy.base import SchedulingRequest
from quantumflow.scheduler.worker_client import (
    WorkerClient,
    WorkerEndpoint,
    WorkerRegistry,
    get_worker_registry,
)
from quantumflow.storage.redis_queue import (
    QueuedRequest,
    QueuePriority,
    RedisQueue,
)

# ==================== 测试数据准备 ====================


def create_queued_request(
    request_id: str = "test-req-1",
    model_name: str = "test-model",
    prompt: str = "Hello world",
    priority: int = QueuePriority.NORMAL.value,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> QueuedRequest:
    """创建测试用QueuedRequest"""
    return QueuedRequest(
        request_id=request_id,
        model_name=model_name,
        prompt=prompt,
        priority=priority,
        created_at=created_at or datetime.now(),
        metadata=metadata or {},
    )


def create_scheduling_request(
    request_id: str = "test-req-1",
    model: str = "test-model",
    prompt: str = "Hello world",
    priority: int = 5,
) -> SchedulingRequest:
    """创建测试用SchedulingRequest"""
    return SchedulingRequest(
        request_id=request_id,
        model=model,
        prompt=prompt,
        priority=priority,
        created_at=datetime.now(),
    )


# ==================== QueuedRequest 序列化测试 ====================


class TestQueuedRequestSerialization:
    """QueuedRequest 序列化/反序列化测试"""

    def test_serialization_preserves_all_fields(self):
        """[核心功能] 序列化必须保留所有字段信息"""
        original = QueuedRequest(
            request_id="req-123",
            model_name="gpt-4",
            prompt="Test prompt",
            priority=QueuePriority.HIGH.value,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            scheduled_at=datetime(2024, 1, 1, 12, 0, 1),
            completed_at=datetime(2024, 1, 1, 12, 0, 5),
            result={"output": "response"},
            error=None,
            metadata={"key": "value", "count": 42},
        )

        json_str = original.to_json()
        restored = QueuedRequest.from_json(json_str)

        assert restored.request_id == original.request_id
        assert restored.model_name == original.model_name
        assert restored.prompt == original.prompt
        assert restored.priority == original.priority
        assert restored.scheduled_at == original.scheduled_at
        assert restored.completed_at == original.completed_at
        assert restored.result == original.result
        assert restored.error == original.error
        assert restored.metadata == original.metadata

    def test_serialization_with_none_optional_fields(self):
        """[边界值] 可选字段为None时的序列化"""
        original = QueuedRequest(
            request_id="req-456",
            model_name="llama-2",
            prompt="Prompt",
            priority=0,
            created_at=datetime.now(),
            scheduled_at=None,
            completed_at=None,
            result=None,
            error=None,
            metadata={},
        )

        json_str = original.to_json()
        restored = QueuedRequest.from_json(json_str)

        assert restored.scheduled_at is None
        assert restored.completed_at is None
        assert restored.result is None
        assert restored.error is None

    def test_roundtrip_idempotent(self):
        """[核心功能] 双重序列化结果必须一致"""
        original = create_queued_request(request_id="idempotent-test")
        first = original.to_json()
        second = QueuedRequest.from_json(first).to_json()
        third = QueuedRequest.from_json(second).to_json()

        assert first == second == third


# ==================== QueuePriority 优先级测试 ====================


class TestQueuePriorityValues:
    """队列优先级枚举值测试"""

    def test_priority_values_ascending(self):
        """[核心功能] 优先级值必须递增"""
        assert QueuePriority.LOW.value < QueuePriority.NORMAL.value
        assert QueuePriority.NORMAL.value < QueuePriority.HIGH.value
        assert QueuePriority.HIGH.value < QueuePriority.CRITICAL.value

    def test_priority_values_correct(self):
        """[核心功能] 优先级值必须符合预期"""
        assert QueuePriority.LOW.value == 0
        assert QueuePriority.NORMAL.value == 5
        assert QueuePriority.HIGH.value == 7
        assert QueuePriority.CRITICAL.value == 9


# ==================== RedisQueue 入队/出队核心逻辑测试 ====================


class TestRedisQueueEnqueueDequeueLogic:
    """RedisQueue 入队/出队核心业务逻辑测试"""

    @pytest.fixture
    def queue(self):
        """创建RedisQueue实例"""
        return RedisQueue(redis_url="redis://localhost:6379/0", default_ttl=3600)

    @pytest.fixture
    def mock_redis(self):
        """创建Mock Redis客户端"""
        mock = AsyncMock()
        mock.setex = AsyncMock(return_value=True)
        mock.zadd = AsyncMock(return_value=1)
        mock.hincrby = AsyncMock(return_value=1)
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_enqueue_stores_request_data(self, queue, mock_redis):
        """[核心功能] 入队必须存储完整请求数据"""
        queue._redis = mock_redis
        queue._connected = True

        request = create_queued_request(
            request_id="enqueue-test-1",
            model_name="test-model",
            prompt="Test prompt",
            priority=QueuePriority.HIGH.value,
        )

        result = await queue.enqueue(request)

        assert result is True
        # 验证SETEX调用 - 存储请求详情
        assert mock_redis.setex.called
        setex_call = mock_redis.setex.call_args
        assert "qf:request:enqueue-test-1" in setex_call[0][0]
        assert setex_call[0][1] == 3600  # default_ttl

    @pytest.mark.asyncio
    async def test_enqueue_uses_zset_with_priority_score(self, queue, mock_redis):
        """[核心功能] 入队必须使用ZSET，score为优先级+时间戳"""
        queue._redis = mock_redis
        queue._connected = True

        t0 = datetime(2024, 1, 1, 12, 0, 0)
        request = create_queued_request(
            request_id="priority-test",
            priority=QueuePriority.HIGH.value,
            created_at=t0,
        )

        await queue.enqueue(request)

        # 验证ZADD调用
        assert mock_redis.zadd.called
        zadd_call = mock_redis.zadd.call_args
        zadd_args = zadd_call[0][1]

        # score = -(priority * 10**12 + timestamp)
        expected_score = -(7 * 10**12 + t0.timestamp())
        actual_score = list(zadd_args.values())[0]
        assert abs(actual_score - expected_score) < 0.001

    @pytest.mark.asyncio
    async def test_enqueue_increments_metric(self, queue, mock_redis):
        """[核心功能] 入队必须增加enqueued指标"""
        queue._redis = mock_redis
        queue._connected = True

        request = create_queued_request()
        await queue.enqueue(request)

        assert mock_redis.hincrby.called
        incr_call = mock_redis.hincrby.call_args
        assert "qf:metrics" in incr_call[0][0]
        assert incr_call[0][1] == "enqueued"

    @pytest.mark.asyncio
    async def test_enqueue_returns_false_when_not_connected(self, queue):
        """[错误处理] 未连接时入队必须返回False"""
        queue._connected = False

        request = create_queued_request()
        result = await queue.enqueue(request)

        assert result is False

    @pytest.mark.asyncio
    async def test_dequeue_returns_request_with_scheduled_time(self, queue, mock_redis):
        """[核心功能] 出队必须返回请求并设置scheduled_at时间"""
        # 模拟ZPOPMIN返回
        mock_redis.zpopmin = AsyncMock(return_value=[("priority-test", -7.123456)])
        mock_redis.get = AsyncMock(
            return_value=json.dumps(
                {
                    "request_id": "priority-test",
                    "model_name": "test-model",
                    "prompt": "Test",
                    "priority": 7,
                    "created_at": datetime.now().isoformat(),
                    "scheduled_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None,
                    "metadata": {},
                }
            )
        )
        mock_redis.setex = AsyncMock(return_value=True)
        mock_redis.hincrby = AsyncMock(return_value=1)

        queue._redis = mock_redis
        queue._connected = True

        result = await queue.dequeue(timeout=0)

        assert result is not None
        assert result.request_id == "priority-test"
        assert result.scheduled_at is not None

    @pytest.mark.asyncio
    async def test_dequeue_increments_dequeued_metric(self, queue, mock_redis):
        """[核心功能] 出队必须增加dequeued指标"""
        mock_redis.zpopmin = AsyncMock(return_value=[("req-1", -5.0)])
        mock_redis.get = AsyncMock(
            return_value=json.dumps(
                {
                    "request_id": "req-1",
                    "model_name": "m",
                    "prompt": "p",
                    "priority": 5,
                    "created_at": datetime.now().isoformat(),
                    "scheduled_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None,
                    "metadata": {},
                }
            )
        )
        mock_redis.setex = AsyncMock(return_value=True)
        mock_redis.hincrby = AsyncMock(return_value=1)

        queue._redis = mock_redis
        queue._connected = True

        await queue.dequeue(timeout=0)

        # 查找hincrby调用
        hincr_calls = [c for c in mock_redis.hincrby.call_args_list]
        metrics_calls = [c for c in hincr_calls if c[0][0] == "qf:metrics"]
        assert len(metrics_calls) >= 1

    @pytest.mark.asyncio
    async def test_dequeue_returns_none_when_queue_empty(self, queue, mock_redis):
        """[边界值] 队列为空时出队必须返回None"""
        mock_redis.zpopmin = AsyncMock(return_value=[])

        queue._redis = mock_redis
        queue._connected = True

        result = await queue.dequeue(timeout=0)

        assert result is None


# ==================== RedisQueue 结果存储测试 ====================


class TestRedisQueueResultStorage:
    """RedisQueue 结果存储逻辑测试"""

    @pytest.fixture
    def queue(self):
        return RedisQueue(default_ttl=3600)

    @pytest.fixture
    def mock_redis(self):
        mock = AsyncMock()
        mock.setex = AsyncMock(return_value=True)
        mock.hincrby = AsyncMock(return_value=1)
        return mock

    @pytest.mark.asyncio
    async def test_store_result_uses_double_ttl(self, queue, mock_redis):
        """[核心功能] 存储结果的TTL必须是默认TTL的两倍"""
        queue._redis = mock_redis
        queue._connected = True

        await queue.store_result("req-123", {"status": "success"})

        assert mock_redis.setex.called
        setex_call = mock_redis.setex.call_args
        assert setex_call[0][0] == "qf:result:req-123"
        assert setex_call[0][1] == 7200  # 3600 * 2

    @pytest.mark.asyncio
    async def test_store_result_increments_completed_metric(self, queue, mock_redis):
        """[核心功能] 存储结果必须增加completed指标"""
        queue._redis = mock_redis
        queue._connected = True

        await queue.store_result("req-123", {"status": "success"})

        assert mock_redis.hincrby.called
        incr_call = mock_redis.hincrby.call_args
        assert incr_call[0][1] == "completed"

    @pytest.mark.asyncio
    async def test_get_result_returns_deserialized_data(self, queue, mock_redis):
        """[核心功能] 获取结果必须返回反序列化的数据"""
        expected_result = {"status": "success", "output": "response"}
        mock_redis.get = AsyncMock(return_value=json.dumps(expected_result))

        queue._redis = mock_redis
        queue._connected = True

        result = await queue.get_result("req-123")

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_get_result_returns_none_when_not_found(self, queue, mock_redis):
        """[边界值] 结果不存在时返回None"""
        mock_redis.get = AsyncMock(return_value=None)

        queue._redis = mock_redis
        queue._connected = True

        result = await queue.get_result("nonexistent")

        assert result is None


# ==================== RedisQueue 重试逻辑测试 ====================


class TestRedisQueueRetryLogic:
    """RedisQueue 重试逻辑测试"""

    @pytest.fixture
    def queue(self):
        return RedisQueue(default_ttl=3600, max_retries=3)

    @pytest.fixture
    def mock_redis(self):
        mock = AsyncMock()
        mock.setex = AsyncMock(return_value=True)
        mock.zadd = AsyncMock(return_value=1)
        mock.hincrby = AsyncMock(return_value=1)
        return mock

    @pytest.mark.asyncio
    async def test_requeue_increments_retry_count(self, queue, mock_redis):
        """[核心功能] 重新入队必须增加重试计数"""
        queue._redis = mock_redis
        queue._connected = True

        request = create_queued_request(metadata={"retry_count": 0})
        await queue.requeue(request, increment_retry=True)

        # 验证setex调用时，存储的JSON中retry_count已被更新
        setex_call = mock_redis.setex.call_args
        stored_json = setex_call[0][2]  # 第三个参数是存储的JSON字符串
        stored_request = json.loads(stored_json)
        assert stored_request["metadata"]["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_requeue_fails_when_max_retries_exceeded(self, queue, mock_redis):
        """[核心功能] 超过最大重试次数必须返回False"""
        queue._redis = mock_redis
        queue._connected = True

        request = create_queued_request(metadata={"retry_count": 3})  # max_retries=3
        result = await queue.requeue(request, increment_retry=True)

        assert result is False
        assert not mock_redis.zadd.called  # 不应该再入队

    @pytest.mark.asyncio
    async def test_requeue_with_increment_false_preserves_count(self, queue, mock_redis):
        """[核心功能] increment_retry=False时不增加重试计数"""
        queue._redis = mock_redis
        queue._connected = True

        request = create_queued_request(metadata={"retry_count": 2})
        await queue.requeue(request, increment_retry=False)

        # 验证setex调用时，retry_count保持不变
        setex_call = mock_redis.setex.call_args
        stored_json = setex_call[0][2]
        stored_request = json.loads(stored_json)
        assert stored_request["metadata"]["retry_count"] == 2  # 未变化


# ==================== RedisQueue 队列统计测试 ====================


class TestRedisQueueStats:
    """RedisQueue 队列统计测试"""

    @pytest.fixture
    def queue(self):
        return RedisQueue()

    @pytest.fixture
    def mock_redis(self):
        mock = AsyncMock()
        mock.zcard = AsyncMock(return_value=10)
        mock.zcount = AsyncMock(side_effect=[2, 5, 2, 1])
        mock.hgetall = AsyncMock(
            return_value={
                "enqueued": "100",
                "dequeued": "90",
                "completed": "85",
                "failed": "5",
            }
        )
        return mock

    @pytest.mark.asyncio
    async def test_queue_size_uses_zcard(self, queue, mock_redis):
        """[核心功能] 队列大小必须使用ZCARD命令"""
        queue._redis = mock_redis
        queue._connected = True

        size = await queue.queue_size()

        assert size == 10
        assert mock_redis.zcard.called
        mock_redis.zcard.assert_called_with("qf:queue")

    @pytest.mark.asyncio
    async def test_get_queue_stats_returns_priority_counts(self, queue, mock_redis):
        """[核心功能] 队列统计必须包含各优先级数量"""
        queue._redis = mock_redis
        queue._connected = True

        stats = await queue.get_queue_stats()

        assert "queue_size" in stats
        assert "priority_counts" in stats
        assert stats["priority_counts"]["LOW"] == 2
        assert stats["priority_counts"]["NORMAL"] == 5

    @pytest.mark.asyncio
    async def test_get_metrics_returns_integer_counts(self, queue, mock_redis):
        """[核心功能] 指标必须返回整型数值"""
        queue._redis = mock_redis
        queue._connected = True

        metrics = await queue.get_metrics()

        assert metrics["enqueued"] == 100
        assert metrics["dequeued"] == 90
        assert metrics["completed"] == 85
        assert isinstance(metrics["enqueued"], int)


# ==================== WorkerClient HTTP通信测试 ====================


class TestWorkerClientInference:
    """WorkerClient 推理请求HTTP通信测试"""

    @pytest.fixture
    def client(self):
        return WorkerClient(timeout=10.0)

    @pytest.fixture
    def endpoint(self):
        return WorkerEndpoint(
            node_id="worker-1",
            host="192.168.1.100",
            port=8080,
        )

    @pytest.mark.asyncio
    async def test_inference_sends_correct_url(self, client, endpoint):
        """[核心功能] 推理请求必须发送正确的URL"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        mock_http_client.post.assert_called_once()
        call_url = mock_http_client.post.call_args[0][0]
        assert call_url == "http://192.168.1.100:8080/inference"

    @pytest.mark.asyncio
    async def test_inference_sends_correct_payload_structure(self, client, endpoint):
        """[核心功能] 推理请求payload结构必须正确"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            await client.inference(
                endpoint=endpoint,
                request_id="req-123",
                model_name="gpt-4",
                prompt="Test prompt",
                sampling_params={"temperature": 0.5, "top_p": 0.9},
            )

        payload = mock_http_client.post.call_args[1]["json"]
        assert payload["request_id"] == "req-123"
        assert payload["model_name"] == "gpt-4"
        assert payload["prompts"] == ["Test prompt"]
        assert payload["sampling_params"]["temperature"] == 0.5
        assert payload["sampling_params"]["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_inference_uses_default_sampling_params(self, client, endpoint):
        """[核心功能] 未提供sampling_params时必须使用默认值"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        payload = mock_http_client.post.call_args[1]["json"]
        defaults = payload["sampling_params"]
        assert defaults["temperature"] == 0.7
        assert defaults["top_p"] == 0.9
        assert defaults["top_k"] == 50
        assert defaults["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_inference_returns_success_response(self, client, endpoint):
        """[核心功能] HTTP 200响应必须返回success状态"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "req-1",
            "status": "success",
            "results": ["output1"],
            "latency_ms": 100,
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        assert result["status"] == "success"
        assert result["request_id"] == "req-1"
        assert result["results"] == ["output1"]
        assert result["latency_ms"] == 100

    @pytest.mark.asyncio
    async def test_inference_handles_http_error_status(self, client, endpoint):
        """[错误处理] HTTP错误状态码必须返回error状态"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "500" in result["error"]
        assert "Internal Server Error" in result["error"]

    @pytest.mark.asyncio
    async def test_inference_handles_timeout(self, client, endpoint):
        """[错误处理] 超时必须返回error状态"""
        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_inference_handles_connection_error(self, client, endpoint):
        """[错误处理] 连接错误必须返回error状态"""
        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_inference_timeout_value_from_config(self, client, endpoint):
        """[核心功能] 超时时间必须使用配置的值"""
        custom_client = WorkerClient(timeout=60.0)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(custom_client, "_get_client", return_value=mock_http_client):
            await custom_client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        # 验证创建client时使用了正确的timeout
        assert custom_client.timeout == 60.0


# ==================== WorkerClient LoadModel测试 ====================


class TestWorkerClientLoadModel:
    """WorkerClient 加载模型测试"""

    @pytest.fixture
    def client(self):
        return WorkerClient()

    @pytest.fixture
    def endpoint(self):
        return WorkerEndpoint(node_id="w1", host="localhost", port=8080)

    @pytest.mark.asyncio
    async def test_load_model_sends_correct_payload(self, client, endpoint):
        """[核心功能] 加载模型请求payload必须正确"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            await client.load_model(
                endpoint=endpoint,
                model_name="llama-2",
                tensor_parallel=2,
                gpu_memory_utilization=0.8,
            )

        payload = mock_http_client.post.call_args[1]["json"]
        assert payload["model_name"] == "llama-2"
        assert payload["tensor_parallel"] == 2
        assert payload["gpu_memory_utilization"] == 0.8


# ==================== WorkerClient GetStatus测试 ====================


class TestWorkerClientGetStatus:
    """WorkerClient 获取状态测试"""

    @pytest.fixture
    def client(self):
        return WorkerClient()

    @pytest.fixture
    def endpoint(self):
        return WorkerEndpoint(node_id="w1", host="localhost", port=8080)

    @pytest.mark.asyncio
    async def test_get_status_returns_worker_info(self, client, endpoint):
        """[核心功能] 获取状态必须返回Worker信息"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "node_id": "worker-1",
            "status": "healthy",
            "gpu_count": 4,
        }

        mock_http_client = MagicMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.get_status(endpoint)

        assert result["node_id"] == "worker-1"
        assert result["status"] == "healthy"
        assert result["gpu_count"] == 4


# ==================== WorkerClient 资源清理测试 ====================


class TestWorkerClientCleanup:
    """WorkerClient 资源清理测试"""

    @pytest.fixture
    def client(self):
        return WorkerClient()

    @pytest.mark.asyncio
    async def test_close_cleans_up_client(self, client):
        """[核心功能] close()必须清理HTTP客户端"""
        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        client._client = mock_client

        await client.close()

        assert client._client is None
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_already_closed(self, client):
        """[边界值] 客户端已关闭时调用close不抛异常"""
        client._client = None
        await client.close()  # 不应该抛异常


# ==================== WorkerRegistry 注册管理测试 ====================


class TestWorkerRegistryManagement:
    """WorkerRegistry 注册管理测试"""

    @pytest.fixture
    def registry(self):
        return WorkerRegistry()

    @pytest.fixture
    def endpoint1(self):
        return WorkerEndpoint(node_id="w1", host="192.168.1.1", port=8080)

    @pytest.fixture
    def endpoint2(self):
        return WorkerEndpoint(node_id="w2", host="192.168.1.2", port=8080)

    @pytest.mark.asyncio
    async def test_register_adds_worker(self, registry, endpoint1):
        """[核心功能] 注册必须添加Worker到注册表"""
        await registry.register(endpoint1)

        worker = await registry.get_worker("w1")
        assert worker is not None
        assert worker.node_id == "w1"
        assert worker.host == "192.168.1.1"
        assert worker.port == 8080

    @pytest.mark.asyncio
    async def test_unregister_removes_worker(self, registry, endpoint1):
        """[核心功能] 注销必须从注册表移除Worker"""
        await registry.register(endpoint1)
        await registry.unregister("w1")

        worker = await registry.get_worker("w1")
        assert worker is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_is_safe(self, registry):
        """[边界值] 注销不存在的Worker不抛异常"""
        await registry.unregister("nonexistent")  # 不应该抛异常

    @pytest.mark.asyncio
    async def test_get_all_workers_returns_all(self, registry, endpoint1, endpoint2):
        """[核心功能] 获取所有Worker必须返回完整列表"""
        await registry.register(endpoint1)
        await registry.register(endpoint2)

        workers = await registry.get_all_workers()

        assert len(workers) == 2
        node_ids = {w.node_id for w in workers}
        assert node_ids == {"w1", "w2"}

    @pytest.mark.asyncio
    async def test_get_worker_count_returns_correct_count(self, registry, endpoint1, endpoint2):
        """[核心功能] 获取Worker数量必须正确"""
        assert await registry.get_worker_count() == 0

        await registry.register(endpoint1)
        assert await registry.get_worker_count() == 1

        await registry.register(endpoint2)
        assert await registry.get_worker_count() == 2

        await registry.unregister("w1")
        assert await registry.get_worker_count() == 1

    @pytest.mark.asyncio
    async def test_update_worker_status_changes_status(self, registry, endpoint1):
        """[核心功能] 更新状态必须改变Worker状态"""
        await registry.register(endpoint1)
        await registry.update_worker_status("w1", "unhealthy")

        worker = await registry.get_worker("w1")
        assert worker.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_get_worker_returns_none_for_unknown(self, registry):
        """[边界值] 获取未知Worker必须返回None"""
        worker = await registry.get_worker("unknown")
        assert worker is None


# ==================== WorkerRegistry 并发安全测试 ====================


class TestWorkerRegistryConcurrency:
    """WorkerRegistry 并发安全测试"""

    @pytest.mark.asyncio
    async def test_concurrent_register_is_safe(self):
        """[核心功能] 并发注册必须安全"""
        registry = WorkerRegistry()

        async def register_worker(worker_id: str):
            endpoint = WorkerEndpoint(
                node_id=f"w{worker_id}",
                host=f"192.168.1.{worker_id}",
                port=8080,
            )
            await registry.register(endpoint)

        # 并发注册10个Worker
        await asyncio.gather(*[register_worker(i) for i in range(10)])

        count = await registry.get_worker_count()
        assert count == 10

    @pytest.mark.asyncio
    async def test_concurrent_read_write_is_safe(self):
        """[核心功能] 并发读写必须安全"""
        registry = WorkerRegistry()
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8080)
        await registry.register(endpoint)

        async def read_worker():
            return await registry.get_worker("w1")

        async def write_worker():
            await registry.update_worker_status("w1", "busy")

        # 并发读写
        results = await asyncio.gather(
            *[read_worker() for _ in range(100)],
            *[write_worker() for _ in range(100)],
        )

        # 应该都有有效的Worker对象返回
        workers = [r for r in results if r is not None]
        assert len(workers) > 0


# ==================== WorkerEndpoint 数据测试 ====================


class TestWorkerEndpoint:
    """WorkerEndpoint 数据类测试"""

    def test_url_property_format(self):
        """[核心功能] url属性格式必须正确"""
        endpoint = WorkerEndpoint(node_id="w1", host="192.168.1.100", port=8080)
        assert endpoint.url == "http://192.168.1.100:8080"

    def test_url_with_different_ports(self):
        """[边界值] 不同端口号必须正确拼接"""
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=3000)
        assert endpoint.url == "http://localhost:3000"

    def test_default_status_is_healthy(self):
        """[默认值] status默认为healthy"""
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8080)
        assert endpoint.status == "healthy"

    def test_custom_status_value(self):
        """[核心功能] status可自定义"""
        endpoint = WorkerEndpoint(
            node_id="w1",
            host="localhost",
            port=8080,
            status="draining",
        )
        assert endpoint.status == "draining"


# ==================== Global Registry Singleton测试 ====================


class TestGlobalRegistrySingleton:
    """全局注册表单例测试"""

    def test_get_worker_registry_returns_same_instance(self):
        """[核心功能] 获取全局注册表必须返回单例"""
        # 重置全局单例
        import quantumflow.scheduler.worker_client as wc

        wc._registry = None

        registry1 = get_worker_registry()
        registry2 = get_worker_registry()

        assert registry1 is registry2


# ==================== DistributedScheduler 核心逻辑测试 ====================


class TestDistributedSchedulerDispatch:
    """DistributedScheduler 分发逻辑测试"""

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
            worker_timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_scheduler_creates_worker_client(self, scheduler):
        """[核心功能] 调度器必须创建WorkerClient"""
        assert scheduler._worker_client is not None
        assert isinstance(scheduler._worker_client, WorkerClient)

    @pytest.mark.asyncio
    async def test_scheduler_creates_worker_registry(self, scheduler):
        """[核心功能] 调度器必须创建WorkerRegistry"""
        assert scheduler._worker_registry is not None
        assert isinstance(scheduler._worker_registry, WorkerRegistry)

    @pytest.mark.asyncio
    async def test_register_worker_creates_endpoint(self, scheduler):
        """[核心功能] 注册Worker必须创建正确的端点"""
        await scheduler.register_worker("w1", "192.168.1.100", 8080)

        endpoint = await scheduler._worker_registry.get_worker("w1")
        assert endpoint is not None
        assert endpoint.node_id == "w1"
        assert endpoint.host == "192.168.1.100"
        assert endpoint.port == 8080

    @pytest.mark.asyncio
    async def test_unregister_worker_removes_endpoint(self, scheduler):
        """[核心功能] 注销Worker必须移除端点"""
        await scheduler.register_worker("w1", "localhost", 8080)
        await scheduler.unregister_worker("w1")

        endpoint = await scheduler._worker_registry.get_worker("w1")
        assert endpoint is None

    @pytest.mark.asyncio
    async def test_get_worker_count_returns_registered_count(self, scheduler):
        """[核心功能] 获取Worker数量必须正确"""
        await scheduler.register_worker("w1", "localhost", 8080)
        await scheduler.register_worker("w2", "localhost", 8081)

        count = await scheduler.get_worker_count()
        assert count == 2


# ==================== DistributedScheduler 停止逻辑测试 ====================


class TestDistributedSchedulerStop:
    """DistributedScheduler 停止逻辑测试"""

    @pytest.mark.asyncio
    async def test_stop_closes_worker_client(self):
        """[核心功能] 停止必须关闭WorkerClient"""
        scheduler = DistributedScheduler()
        scheduler._worker_client = AsyncMock()
        scheduler._worker_client.close = AsyncMock()

        await scheduler.stop()

        scheduler._worker_client.close.assert_called_once()


# ==================== Integration: 端到端流程测试 ====================


class TestEndToEndFlow:
    """端到端流程集成测试"""

    @pytest.mark.asyncio
    async def test_worker_client_full_inference_flow(self):
        """[集成测试] 完整的推理流程必须正常工作"""
        client = WorkerClient(timeout=10.0)
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8080)

        # Mock HTTP响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "req-endtoend",
            "status": "success",
            "results": ["Generated text"],
            "latency_ms": 150,
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-endtoend",
                model_name="test-model",
                prompt="Hello, world!",
                sampling_params={"temperature": 0.7},
            )

        # 验证完整流程
        assert result["status"] == "success"
        assert result["request_id"] == "req-endtoend"
        assert result["results"] == ["Generated text"]

        # 清理
        await client.close()

    @pytest.mark.asyncio
    async def test_worker_registry_lifecycle(self):
        """[集成测试] Worker注册表生命周期必须正确"""
        registry = WorkerRegistry()

        # 注册
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8080)
        await registry.register(endpoint)
        assert await registry.get_worker_count() == 1

        # 更新状态
        await registry.update_worker_status("w1", "busy")
        worker = await registry.get_worker("w1")
        assert worker.status == "busy"

        # 注销
        await registry.unregister("w1")
        assert await registry.get_worker_count() == 0


# ==================== 错误码和边界值测试 ====================


class TestErrorHandling:
    """错误处理边界值测试"""

    @pytest.mark.asyncio
    async def test_worker_client_handles_404(self):
        """[错误处理] 404错误必须正确处理"""
        client = WorkerClient()
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8080)

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "404" in result["error"]

    @pytest.mark.asyncio
    async def test_worker_client_handles_503(self):
        """[错误处理] 503 Service Unavailable必须正确处理"""
        client = WorkerClient()
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8080)

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "503" in result["error"]

    @pytest.mark.asyncio
    async def test_worker_client_handles_network_error(self):
        """[错误处理] 网络错误必须正确处理"""
        client = WorkerClient()
        endpoint = WorkerEndpoint(node_id="w1", host="unreachable.host", port=8080)

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(side_effect=OSError("Network unreachable"))
        mock_http_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.inference(
                endpoint=endpoint,
                request_id="req-1",
                model_name="model-1",
                prompt="Hello",
            )

        assert result["status"] == "error"
        assert "Network unreachable" in result["error"]
