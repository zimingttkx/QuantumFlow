"""分布式调度器

使用Redis队列实现Controller和Worker之间的解耦。
Controller将推理请求放入Redis队列，Worker从队列中拉取任务执行。
"""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from quantumflow.scheduler.scheduler import QueueItem, Scheduler
from quantumflow.scheduler.strategy.base import (
    SchedulingRequest,
    SchedulingResult,
)
from quantumflow.scheduler.worker_client import (
    WorkerClient,
    WorkerEndpoint,
    get_worker_registry,
)
from quantumflow.storage.connection import get_redis_manager
from quantumflow.storage.redis_queue import QueuedRequest, RedisQueue

logger = structlog.get_logger().bind(component="distributed_scheduler")


class DistributedScheduler(Scheduler):
    """
    分布式调度器

    特性：
    - 使用Redis队列替代内存队列
    - 支持任务持久化（Redis持久化）
    - Controller和Worker解耦
    - 支持分布式部署
    - 真实的HTTP通信到Worker节点
    """

    def __init__(
        self,
        default_strategy: str = "adaptive",
        loop_interval_ms: int = 100,
        max_retries: int = 3,
        redis_url: str = "redis://localhost:6379/0",
        worker_timeout: float = 30.0,
    ):
        super().__init__(
            default_strategy=default_strategy,
            loop_interval_ms=loop_interval_ms,
            max_retries=max_retries,
        )

        self.redis_url = redis_url
        self._redis_queue: RedisQueue | None = None
        self._use_redis = True

        # Worker 通信
        self._worker_client = WorkerClient(timeout=worker_timeout)
        self._worker_registry = get_worker_registry()

        logger.info(
            "distributed_scheduler_created",
            redis_url=redis_url,
        )

    def _get_tenant_quota(self, tenant_id: str) -> "QuotaConfig":
        """获取租户配额（从 Redis 加载，key 与 tenants.py 对齐）"""
        from quantumflow.api.models.tenant import QuotaConfig
        from quantumflow.core.constants import DEFAULT_TENANT_QUOTA, TENANT_PREFIX

        redis = None
        try:
            from quantumflow.storage import get_redis_manager_sync
            redis = get_redis_manager_sync().get_client()
        except Exception:
            pass

        if redis:
            # 通过 ID→hash 映射找到租户数据
            hashed_key = redis.get(f"qf:tenant:id:{tenant_id}")
            if hashed_key:
                if isinstance(hashed_key, bytes):
                    hashed_key = hashed_key.decode()
                data = redis.hgetall(f"{TENANT_PREFIX}{hashed_key}")
                if data:
                    return QuotaConfig(
                        requests_per_minute=int(data.get(b"quota_requests_per_minute", DEFAULT_TENANT_QUOTA["requests_per_minute"])),
                        requests_per_day=int(data.get(b"quota_requests_per_day", DEFAULT_TENANT_QUOTA["requests_per_day"])),
                        max_tokens_per_request=int(data.get(b"quota_max_tokens", DEFAULT_TENANT_QUOTA["max_tokens_per_request"])),
                        gpu_memory_mb=int(data.get(b"quota_gpu_memory", DEFAULT_TENANT_QUOTA["gpu_memory_mb"])),
                        concurrent_requests=int(data.get(b"quota_concurrent", DEFAULT_TENANT_QUOTA["concurrent_requests"])),
                    )

        return QuotaConfig(**DEFAULT_TENANT_QUOTA)

    def _get_concurrent_requests(self, tenant_id: str) -> int:
        """获取租户当前并发请求数"""
        try:
            from quantumflow.storage import get_redis_manager_sync
            redis = get_redis_manager_sync().get_client()
        except Exception:
            return 0

        if redis:
            key = f"qf:concurrent:{tenant_id}"
            count = redis.get(key)
            return int(count) if count else 0
        return 0

    def _increment_concurrent_requests(self, tenant_id: str) -> None:
        """增加租户并发请求计数"""
        try:
            from quantumflow.storage import get_redis_manager_sync
            redis = get_redis_manager_sync().get_client()
        except Exception:
            return

        if redis:
            key = f"qf:concurrent:{tenant_id}"
            redis.incr(key)

    async def start(self):
        """启动分布式调度器"""
        if self._running:
            logger.warning("scheduler_already_running")
            return

        # 初始化Redis连接
        redis_mgr = await get_redis_manager()
        if redis_mgr.is_connected:
            self._redis_queue = RedisQueue(redis_url=self.redis_url)
            await self._redis_queue.connect()
            logger.info("distributed_scheduler_redis_connected")
        else:
            logger.warning("distributed_scheduler_redis_not_available")
            self._use_redis = False

        self._running = True
        self._scheduling_task = asyncio.create_task(self._scheduling_loop())

        logger.info("distributed_scheduler_started", strategy=self.default_strategy)

    async def stop(self):
        """停止分布式调度器"""
        self._running = False

        if self._scheduling_task:
            self._scheduling_task.cancel()
            try:
                await self._scheduling_task
            except asyncio.CancelledError:
                pass

        # 关闭Worker客户端
        await self._worker_client.close()

        if self._redis_queue:
            await self._redis_queue.disconnect()

        logger.info("distributed_scheduler_stopped")

    async def submit(
        self,
        request: SchedulingRequest,
        tenant_id: str | None = None,
    ) -> str:
        """
        提交推理请求到Redis队列

        Args:
            request: 调度请求
            tenant_id: 租户 ID（如果为 None，则从 request.tenant_id 获取）

        Returns:
            request_id: 请求ID

        Raises:
            Exception: 当租户并发请求数超限时
        """
        # 获取租户 ID
        if tenant_id is None:
            tenant_id = getattr(request, "tenant_id", "default")

        # 检查租户并发限制
        quota = self._get_tenant_quota(tenant_id)
        current_concurrent = self._get_concurrent_requests(tenant_id)

        if current_concurrent >= quota.concurrent_requests:
            error_msg = f"租户 {tenant_id} 并发请求数超限: {current_concurrent}/{quota.concurrent_requests}"
            logger.warning(
                "tenant_concurrent_limit_exceeded",
                tenant_id=tenant_id,
                current=current_concurrent,
                limit=quota.concurrent_requests,
            )
            raise Exception(error_msg)

        self.stats["total_requests"] += 1

        logger.info(
            "request_submitted",
            request_id=request.request_id,
            model=request.model,
            priority=request.priority,
            tenant_id=tenant_id,
        )

        if self._use_redis and self._redis_queue:
            # 使用Redis队列
            queued_request = QueuedRequest(
                request_id=request.request_id,
                model_name=request.model,
                prompt=request.prompt,
                priority=request.priority,
                created_at=datetime.now(),
                metadata={
                    "prompt_tokens": getattr(request, "prompt_tokens", 0),
                    "max_tokens": getattr(request, "max_tokens", 512),
                    "tenant_id": tenant_id,
                },
            )
            success = await self._redis_queue.enqueue(queued_request)
            if success:
                self.stats["pending_requests"] += 1
                self._increment_concurrent_requests(tenant_id)
                return request.request_id
            else:
                logger.error("redis_enqueue_failed", request_id=request.request_id)
                # 回退到内存队列
                self._use_redis = False

        # 回退到内存队列
        self.stats["pending_requests"] += 1
        priority = request.priority
        self._request_counter += 1
        await self.pending_queue.put((priority, request.created_at, self._request_counter, request))

        return request.request_id

    async def _scheduling_loop(self):
        """调度循环"""
        logger.info("distributed_scheduling_loop_started")

        while self._running:
            try:
                await asyncio.sleep(self.loop_interval_ms / 1000.0)

                # 获取可用节点
                if not self.available_nodes:
                    continue

                # 从Redis队列或内存队列处理请求
                if self._use_redis and self._redis_queue:
                    await self._process_batch_from_redis()
                else:
                    await self._process_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("scheduling_loop_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("distributed_scheduling_loop_stopped")

    async def _process_batch_from_redis(self):
        """从Redis队列批量处理请求"""
        if not self._redis_queue:
            return

        # 批量获取请求
        batch_size = min(await self._redis_queue.queue_size(), 10)
        if batch_size == 0:
            return

        requests_batch = []
        for _ in range(batch_size):
            request = await self._redis_queue.dequeue(timeout=0)
            if request is None:
                break

            # 转换为SchedulingRequest
            scheduling_request = SchedulingRequest(
                request_id=request.request_id,
                model=request.model_name,
                prompt=request.prompt,
                priority=request.priority,
                created_at=request.created_at,
            )
            requests_batch.append(scheduling_request)

        if not requests_batch:
            return

        # 处理每个请求
        for request in requests_batch:
            result = await self._schedule_request(request)

            if result.success:
                self.stats["successful_requests"] += 1
                await self._dispatch(request, result)
            else:
                self.stats["failed_requests"] += 1
                await self._handle_scheduling_failure_redis(request, result)

    async def _handle_scheduling_failure_redis(
        self, request: SchedulingRequest, result: SchedulingResult
    ):
        """处理Redis队列中调度失败的请求"""
        request_id = request.request_id

        # 重试逻辑
        request.retry_count += 1

        if request.retry_count < self.max_retries:
            logger.warning(
                "scheduling_retry",
                request_id=request_id,
                reason=result.reason,
                retry_count=request.retry_count,
            )
            # 重新入队
            queued_request = QueuedRequest(
                request_id=request.request_id,
                model_name=request.model,
                prompt=request.prompt,
                priority=request.priority,
                created_at=request.created_at,
                metadata={"retry_count": request.retry_count},
            )
            await self._redis_queue.requeue(queued_request)
        else:
            logger.error(
                "scheduling_failed",
                request_id=request_id,
                reason=result.reason,
                max_retries=self.max_retries,
            )
            self.stats["failed_requests"] += 1
            self.stats["pending_requests"] -= 1

            # 存储错误结果
            await self._redis_queue.store_result(
                request_id,
                {"status": "error", "reason": result.reason},
            )

    async def _dispatch(self, request: SchedulingRequest, result: SchedulingResult):
        """分发请求到Worker（真实HTTP调用）"""
        request_id = request.request_id

        # 更新运行状态
        self.running_requests[request_id] = QueueItem(
            request=request,
            scheduled_at=datetime.now(),
            result=result,
        )

        self.stats["pending_requests"] -= 1

        logger.info(
            "request_dispatched",
            request_id=request_id,
            nodes=result.assigned_nodes,
            strategy=result.strategy_used,
        )

        # 获取分配的Worker节点
        if not result.assigned_nodes:
            logger.error("no_nodes_assigned", request_id=request_id)
            await self._handle_dispatch_failure(request, "No nodes assigned")
            return

        # 选择第一个可用的Worker节点
        node_id = result.assigned_nodes[0]
        worker_endpoint = await self._worker_registry.get_worker(node_id)

        if not worker_endpoint:
            logger.error("worker_not_found", request_id=request_id, node_id=node_id)
            await self._handle_dispatch_failure(request, f"Worker {node_id} not found")
            return

        # 构建采样参数
        metadata = getattr(request, "metadata", {}) or {}
        sampling_params = {
            "temperature": metadata.get("temperature", 0.7),
            "top_p": metadata.get("top_p", 0.9),
            "top_k": metadata.get("top_k", 50),
            "max_tokens": metadata.get("max_tokens", 512),
            "repetition_penalty": metadata.get("repetition_penalty", 1.0),
        }

        # 异步发送到Worker（不阻塞调度循环）
        asyncio.create_task(self._send_to_worker(request, worker_endpoint, sampling_params))

    async def _send_to_worker(
        self,
        request: SchedulingRequest,
        worker_endpoint: WorkerEndpoint,
        sampling_params: dict[str, Any],
    ):
        """发送请求到Worker并处理响应"""
        request_id = request.request_id

        try:
            logger.info(
                "sending_to_worker",
                request_id=request_id,
                worker_url=worker_endpoint.url,
                model=request.model,
            )

            # 实际HTTP调用到Worker
            response = await self._worker_client.inference(
                endpoint=worker_endpoint,
                request_id=request_id,
                model_name=request.model,
                prompt=request.prompt,
                sampling_params=sampling_params,
            )

            # 处理响应
            if response.get("status") == "success":
                logger.info(
                    "worker_inference_success",
                    request_id=request_id,
                    worker_url=worker_endpoint.url,
                )
                # 存储成功结果
                if self._redis_queue:
                    await self._redis_queue.store_result(
                        request_id,
                        {
                            "status": "success",
                            "result": response.get("results"),
                            "latency_ms": response.get("latency_ms"),
                        },
                    )
                # 清理运行状态
                self.running_requests.pop(request_id, None)
            else:
                error = response.get("error", "Unknown error")
                logger.error(
                    "worker_inference_failed",
                    request_id=request_id,
                    error=error,
                )
                await self._handle_dispatch_failure(request, error)

        except Exception as e:
            logger.error(
                "worker_communication_error",
                request_id=request_id,
                worker_url=worker_endpoint.url,
                error=str(e),
            )
            await self._handle_dispatch_failure(request, str(e))

    async def _handle_dispatch_failure(self, request: SchedulingRequest, error: str):
        """处理分发失败（重试或标记失败）"""
        request_id = request.request_id
        retry_count = getattr(request, "retry_count", 0)

        if retry_count < self.max_retries:
            logger.warning(
                "dispatch_retry",
                request_id=request_id,
                error=error,
                retry_count=retry_count + 1,
            )
            # 重新入队等待重试
            if self._redis_queue:
                queued_request = QueuedRequest(
                    request_id=request.request_id,
                    model_name=request.model,
                    prompt=request.prompt,
                    priority=request.priority,
                    created_at=request.created_at,
                    metadata={"retry_count": retry_count + 1},
                )
                await self._redis_queue.requeue(queued_request)
        else:
            logger.error(
                "dispatch_failed",
                request_id=request_id,
                error=error,
                max_retries=self.max_retries,
            )
            # 存储失败结果
            if self._redis_queue:
                await self._redis_queue.store_result(
                    request_id,
                    {"status": "error", "reason": error},
                )
            # 清理运行状态
            self.running_requests.pop(request_id, None)
            self.stats["failed_requests"] += 1

    # ==================== Worker 管理 ====================

    async def register_worker(
        self,
        node_id: str,
        host: str,
        port: int,
    ) -> bool:
        """
        注册Worker节点

        Args:
            node_id: 节点ID
            host: 主机地址
            port: 端口

        Returns:
            是否成功
        """
        endpoint = WorkerEndpoint(
            node_id=node_id,
            host=host,
            port=port,
            status="healthy",
        )
        await self._worker_registry.register(endpoint)

        logger.info(
            "worker_registered",
            node_id=node_id,
            host=host,
            port=port,
        )
        return True

    async def unregister_worker(self, node_id: str) -> bool:
        """
        注销Worker节点

        Args:
            node_id: 节点ID

        Returns:
            是否成功
        """
        await self._worker_registry.unregister(node_id)
        logger.info("worker_unregistered", node_id=node_id)
        return True

    async def get_worker_count(self) -> int:
        """获取已注册Worker数量"""
        return await self._worker_registry.get_worker_count()

    def get_queue_stats(self) -> dict[str, Any]:
        """获取队列统计"""
        stats = {
            **self.stats,
            "running_size": len(self.running_requests),
            "available_nodes": len(self.available_nodes),
        }

        if self._use_redis and self._redis_queue:
            stats["queue_size"] = asyncio.run(self._redis_queue.queue_size())
            stats["redis_enabled"] = True
        else:
            stats["queue_size"] = self.pending_queue.qsize()
            stats["redis_enabled"] = False

        return stats


# 全局单例
_scheduler: DistributedScheduler | None = None


def get_scheduler() -> DistributedScheduler:
    """获取全局调度器"""
    global _scheduler
    if _scheduler is None:
        _scheduler = DistributedScheduler()
    return _scheduler


async def init_scheduler(
    default_strategy: str = "adaptive",
    redis_url: str = "redis://localhost:6379/0",
) -> DistributedScheduler:
    """初始化调度器"""
    global _scheduler
    _scheduler = DistributedScheduler(redis_url=redis_url)
    await _scheduler.start()
    return _scheduler


async def close_scheduler():
    """关闭调度器"""
    global _scheduler
    if _scheduler:
        await _scheduler.stop()
        _scheduler = None
