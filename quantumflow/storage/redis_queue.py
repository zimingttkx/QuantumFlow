"""Redis队列存储实现"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import redis.asyncio as redis
import structlog

logger = structlog.get_logger().bind(component="redis_queue")


class QueuePriority(Enum):
    """队列优先级"""

    LOW = 0
    NORMAL = 5
    HIGH = 7
    CRITICAL = 9


@dataclass
class QueuedRequest:
    """队列中的请求"""

    request_id: str
    model_name: str
    prompt: str
    priority: int
    created_at: datetime
    scheduled_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为JSON"""
        return json.dumps(
            {
                "request_id": self.request_id,
                "model_name": self.model_name,
                "prompt": self.prompt,
                "priority": self.priority,
                "created_at": self.created_at.isoformat(),
                "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "result": self.result,
                "error": self.error,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "QueuedRequest":
        """从JSON反序列化"""
        obj = json.loads(data)
        return cls(
            request_id=obj["request_id"],
            model_name=obj["model_name"],
            prompt=obj["prompt"],
            priority=obj["priority"],
            created_at=datetime.fromisoformat(obj["created_at"]),
            scheduled_at=(
                datetime.fromisoformat(obj["scheduled_at"]) if obj.get("scheduled_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(obj["completed_at"]) if obj.get("completed_at") else None
            ),
            result=obj.get("result"),
            error=obj.get("error"),
            metadata=obj.get("metadata", {}),
        )


class RedisQueue:
    """
    Redis队列存储

    特性：
    - 支持优先级队列
    - 支持请求追踪
    - 支持超时重试
    - 支持结果存储
    """

    # Redis键前缀
    PREFIX = "qf:"
    QUEUE_KEY = f"{PREFIX}queue"
    QUEUE_PRIORITY_SUFFIX = ":priority"
    RESULT_KEY = f"{PREFIX}result:"
    REQUEST_KEY = f"{PREFIX}request:"
    METRICS_KEY = f"{PREFIX}metrics"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        max_retries: int = 3,
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_retries = max_retries

        self._redis: redis.Redis | None = None
        self._connected = False

        logger.info(
            "redis_queue_created",
            redis_url=redis_url,
            default_ttl=default_ttl,
        )

    async def connect(self) -> bool:
        """连接Redis"""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            self._connected = True

            logger.info("redis_connected", redis_url=self.redis_url)
            return True

        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            self._connected = False
            return False

    async def disconnect(self):
        """断开Redis连接"""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("redis_disconnected")

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected and self._redis is not None

    # ==================== 队列操作 ====================

    async def enqueue(self, request: QueuedRequest) -> bool:
        """
        入队请求

        Args:
            request: 请求对象

        Returns:
            是否成功
        """
        if not self.is_connected:
            logger.error("redis_not_connected")
            return False

        try:
            # 存储请求详情
            request_key = f"{self.REQUEST_KEY}{request.request_id}"
            await self._redis.setex(
                request_key,
                self.default_ttl,
                request.to_json(),
            )

            # 加入优先级队列 (使用有序集合，score为优先级)
            # 使用负数优先级，因为Redis ZSET默认升序
            score = -(request.priority + request.created_at.timestamp())
            await self._redis.zadd(
                self.QUEUE_KEY,
                {request.request_id: score},
            )

            # 增加队列大小指标
            await self._increment_metric("enqueued")

            logger.debug(
                "request_enqueued",
                request_id=request.request_id,
                priority=request.priority,
            )
            return True

        except Exception as e:
            logger.error(
                "enqueue_failed",
                request_id=request.request_id,
                error=str(e),
            )
            return False

    async def dequeue(self, timeout: int = 0) -> QueuedRequest | None:
        """
        出队请求（阻塞）

        Args:
            timeout: 阻塞超时时间（秒），0表示非阻塞

        Returns:
            请求对象或None
        """
        if not self.is_connected:
            logger.error("redis_not_connected")
            return None

        try:
            # 使用ZPOPMIN获取最高优先级请求
            if timeout > 0:
                # 阻塞模式
                result = await self._redis.bzpopmin(
                    self.QUEUE_KEY,
                    timeout=timeout,
                )
                if result is None:
                    return None
                request_id = result[1]
            else:
                # 非阻塞模式
                result = await self._redis.zpopmin(
                    self.QUEUE_KEY,
                    count=1,
                )
                if not result:
                    return None
                request_id = result[0][0]

            # 获取请求详情
            request_key = f"{self.REQUEST_KEY}{request_id}"
            data = await self._redis.get(request_key)

            if not data:
                logger.warning(
                    "request_not_found",
                    request_id=request_id,
                )
                return None

            request = QueuedRequest.from_json(data)
            request.scheduled_at = datetime.now()

            # 更新为已调度状态
            await self._redis.setex(
                request_key,
                self.default_ttl,
                request.to_json(),
            )

            await self._increment_metric("dequeued")

            logger.debug(
                "request_dequeued",
                request_id=request.request_id,
            )
            return request

        except Exception as e:
            logger.error("dequeue_failed", error=str(e))
            return None

    async def dequeue_batch(self, batch_size: int = 10) -> list[QueuedRequest]:
        """
        批量出队

        Args:
            batch_size: 批量大小

        Returns:
            请求列表
        """
        requests = []

        for _ in range(batch_size):
            request = await self.dequeue(timeout=0)
            if request is None:
                break
            requests.append(request)

        return requests

    async def requeue(self, request: QueuedRequest, increment_retry: bool = True) -> bool:
        """
        重新入队（失败重试）

        Args:
            request: 请求对象
            increment_retry: 是否增加重试计数

        Returns:
            是否成功
        """
        if not self.is_connected:
            return False

        try:
            # 增加重试计数
            if increment_retry:
                request.metadata["retry_count"] = request.metadata.get("retry_count", 0) + 1

                # 检查是否超过最大重试次数
                if request.metadata.get("retry_count", 0) > self.max_retries:
                    await self._increment_metric("retries_exceeded")
                    return False

            # 重新入队
            return await self.enqueue(request)

        except Exception as e:
            logger.error(
                "requeue_failed",
                request_id=request.request_id,
                error=str(e),
            )
            return False

    # ==================== 结果存储 ====================

    async def store_result(self, request_id: str, result: dict[str, Any]) -> bool:
        """
        存储请求结果

        Args:
            request_id: 请求ID
            result: 结果数据

        Returns:
            是否成功
        """
        if not self.is_connected:
            return False

        try:
            result_key = f"{self.RESULT_KEY}{request_id}"
            await self._redis.setex(
                result_key,
                self.default_ttl * 2,  # 结果保留更长时间
                json.dumps(result),
            )

            await self._increment_metric("completed")

            logger.debug(
                "result_stored",
                request_id=request_id,
            )
            return True

        except Exception as e:
            logger.error(
                "store_result_failed",
                request_id=request_id,
                error=str(e),
            )
            return False

    async def get_result(self, request_id: str) -> dict[str, Any] | None:
        """
        获取请求结果

        Args:
            request_id: 请求ID

        Returns:
            结果数据或None
        """
        if not self.is_connected:
            return None

        try:
            result_key = f"{self.RESULT_KEY}{request_id}"
            data = await self._redis.get(result_key)

            if data:
                return json.loads(data)
            return None

        except Exception as e:
            logger.error(
                "get_result_failed",
                request_id=request_id,
                error=str(e),
            )
            return None

    async def get_request(self, request_id: str) -> QueuedRequest | None:
        """
        获取请求详情

        Args:
            request_id: 请求ID

        Returns:
            请求对象或None
        """
        if not self.is_connected:
            return None

        try:
            request_key = f"{self.REQUEST_KEY}{request_id}"
            data = await self._redis.get(request_key)

            if data:
                return QueuedRequest.from_json(data)
            return None

        except Exception as e:
            logger.error(
                "get_request_failed",
                request_id=request_id,
                error=str(e),
            )
            return None

    # ==================== 队列查询 ====================

    async def queue_size(self) -> int:
        """获取队列大小"""
        if not self.is_connected:
            return 0

        try:
            return await self._redis.zcard(self.QUEUE_KEY)
        except Exception as e:
            logger.error("queue_size_failed", error=str(e))
            return 0

    async def get_queue_stats(self) -> dict[str, Any]:
        """获取队列统计"""
        if not self.is_connected:
            return {}

        try:
            queue_size = await self.queue_size()

            # 获取各优先级队列大小
            priority_counts = {}
            for priority in QueuePriority:
                count = await self._redis.zcount(
                    self.QUEUE_KEY,
                    -(priority.value + 1),
                    -priority.value,
                )
                priority_counts[priority.name] = count

            return {
                "queue_size": queue_size,
                "priority_counts": priority_counts,
                "connected": self._connected,
            }

        except Exception as e:
            logger.error("get_queue_stats_failed", error=str(e))
            return {}

    async def clear_queue(self) -> bool:
        """清空队列"""
        if not self.is_connected:
            return False

        try:
            # 获取所有请求ID
            request_ids = await self._redis.zrange(self.QUEUE_KEY, 0, -1)

            # 删除所有请求详情
            for request_id in request_ids:
                await self._redis.delete(f"{self.REQUEST_KEY}{request_id}")

            # 清空队列
            await self._redis.delete(self.QUEUE_KEY)

            logger.info("queue_cleared", count=len(request_ids))
            return True

        except Exception as e:
            logger.error("clear_queue_failed", error=str(e))
            return False

    # ==================== 指标 ====================

    async def _increment_metric(self, metric: str):
        """增加指标"""
        if not self.is_connected:
            return

        try:
            await self._redis.hincrby(self.METRICS_KEY, metric, 1)
        except Exception as e:
            logger.warning("increment_metric_failed", error=str(e))

    async def get_metrics(self) -> dict[str, int]:
        """获取指标"""
        if not self.is_connected:
            return {}

        try:
            metrics = await self._redis.hgetall(self.METRICS_KEY)
            return {k: int(v) for k, v in metrics.items()}
        except Exception as e:
            logger.error("get_metrics_failed", error=str(e))
            return {}

    # ==================== 工具方法 ====================

    @staticmethod
    def create_request(
        model_name: str,
        prompt: str,
        priority: int = QueuePriority.NORMAL.value,
        **metadata,
    ) -> QueuedRequest:
        """
        创建请求对象

        Args:
            model_name: 模型名称
            prompt: 提示
            priority: 优先级
            **metadata: 额外元数据

        Returns:
            QueuedRequest对象
        """
        return QueuedRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            priority=priority,
            created_at=datetime.now(),
            metadata=metadata,
        )
