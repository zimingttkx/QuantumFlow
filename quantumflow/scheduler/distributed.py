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
        # Bug fix (M-R2): 不再因一次失败永久禁用 Redis。
        # 改为: 连续失败 N 次才降级,成功后自动恢复。
        self._redis_consecutive_failures = 0
        self._redis_disable_threshold = 5  # 连续 5 次失败才禁用
        self._redis_disabled_at: float | None = None
        # 失败后多久才再次尝试 Redis(指数退避上限 60s)
        self._redis_retry_after_ts: float = 0.0

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

    async def _get_concurrent_requests(self, tenant_id: str) -> int:
        """获取租户当前并发请求数（异步，与 _decrement_concurrent_requests 配对）"""
        try:
            from quantumflow.storage import get_redis_manager_sync
            redis = get_redis_manager_sync().get_client()
        except Exception:
            return 0

        if redis:
            key = f"qf:concurrent:{tenant_id}"
            try:
                # redis-py 同步客户端在 to_thread 中调用
                import asyncio
                count = await asyncio.to_thread(redis.get, key)
                return int(count) if count else 0
            except Exception:
                return 0
        return 0

    # Lua 脚本: 原子"check-limit + increment",返回新的计数值,超限返回 -1
    # Bug fix (H-C5): 原本 check-then-incr 有 TOCTOU 竞态,N 个并发请求会
    # 全部读到 limit-1,然后都通过自增,实际并发 = limit + N,远超 quota。
    _INCR_WITH_LIMIT_LUA = """
    local current = tonumber(redis.call('GET', KEYS[1]) or '0')
    local limit = tonumber(ARGV[1])
    if current >= limit then
        return -1
    else
        return redis.call('INCR', KEYS[1])
    end
    """

    def _try_increment_concurrent_requests(self, tenant_id: str, limit: int) -> int:
        """原子地"check-limit + increment"。

        Returns:
            > 0: 新计数值(成功,可放行)
            -1:  超限(拒绝)
            0:   Redis 不可用(放行,走兜底)
        """
        try:
            from quantumflow.storage import get_redis_manager_sync
            redis = get_redis_manager_sync().get_client()
        except Exception:
            return 0

        if redis:
            key = f"qf:concurrent:{tenant_id}"
            try:
                result = redis.eval(self._INCR_WITH_LIMIT_LUA, 1, key, str(limit))
                return int(result)
            except Exception:
                return 0
        return 0

    def _increment_concurrent_requests(self, tenant_id: str) -> None:
        """增加租户并发请求计数(无上限检查,仅用于已确认通过的场景)"""
        try:
            from quantumflow.storage import get_redis_manager_sync
            redis = get_redis_manager_sync().get_client()
        except Exception:
            return

        if redis:
            key = f"qf:concurrent:{tenant_id}"
            redis.incr(key)

    def _decrement_concurrent_requests(self, tenant_id: str) -> None:
        """减少租户并发请求计数（不会降到 0 以下）"""
        try:
            from quantumflow.storage import get_redis_manager_sync
            redis = get_redis_manager_sync().get_client()
        except Exception:
            return

        if redis:
            key = f"qf:concurrent:{tenant_id}"
            # 使用 Lua 脚本保证原子性：decrement 但不低于 0
            try:
                redis.eval(
                    """
                    local current = tonumber(redis.call('GET', KEYS[1]) or '0')
                    if current > 0 then
                        return redis.call('DECR', KEYS[1])
                    else
                        redis.call('SET', KEYS[1], '0')
                        return 0
                    end
                    """,
                    1,
                    key,
                )
            except Exception:
                # 回退：直接 DECR（即使可能短暂为 -1）
                try:
                    redis.decr(key)
                except Exception:
                    pass

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
        # Bug fix (H-C5): 原本 check-then-incr 有 TOCTOU 竞态,改为原子
        # check-and-incr (Redis Lua 脚本)。
        # 必须在 event loop 中运行 Lua 脚本(redis.eval 是阻塞调用)
        import asyncio
        new_count = await asyncio.to_thread(
            self._try_increment_concurrent_requests,
            tenant_id,
            quota.concurrent_requests,
        )

        if new_count == -1:
            # 超限 — 回滚(虽然 Lua 已经原子不增,但稳妥起见检查一下)
            # 注意: -1 意味着 Lua 没增,这里不需要 DECR
            current_concurrent = await self._get_concurrent_requests(tenant_id)
            error_msg = (
                f"租户 {tenant_id} 并发请求数超限: "
                f"{current_concurrent}/{quota.concurrent_requests}"
            )
            logger.warning(
                "tenant_concurrent_limit_exceeded",
                tenant_id=tenant_id,
                current=current_concurrent,
                limit=quota.concurrent_requests,
            )
            raise Exception(error_msg)

        # new_count > 0: 原子自增成功,继续
        # new_count == 0: Redis 不可用,放行(走兜底,与原行为兼容)

        self.stats["total_requests"] += 1

        logger.info(
            "request_submitted",
            request_id=request.request_id,
            model=request.model,
            priority=request.priority,
            tenant_id=tenant_id,
            concurrent=new_count,
        )

        import time
        # Bug fix (M-R2): 检查退避策略 — 连续失败期间临时禁用,超时后自动恢复
        _in_backoff = self._redis_retry_after_ts > time.time()
        should_try_redis = (
            self._use_redis
            and self._redis_queue is not None
            and not _in_backoff
        )

        if should_try_redis:
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
                # Bug fix (M-R2): 成功后重置失败计数
                self._redis_consecutive_failures = 0
                self._redis_disabled_at = None
                self._redis_retry_after_ts = 0.0
                return request.request_id
            else:
                logger.error("redis_enqueue_failed", request_id=request.request_id)
                # 回滚原子 incr (enqueue 失败,占用应释放)
                await asyncio.to_thread(self._decrement_concurrent_requests, tenant_id)
                # Bug fix (M-R2): 不再永久降级,使用退避策略
                self._redis_consecutive_failures += 1
                if self._redis_consecutive_failures >= self._redis_disable_threshold:
                    self._redis_disabled_at = time.time()
                    backoff = min(
                        60.0,
                        2 ** (self._redis_consecutive_failures - self._redis_disable_threshold),
                    )
                    self._redis_retry_after_ts = time.time() + backoff
                    logger.warning(
                        "redis_temporarily_disabled",
                        consecutive_failures=self._redis_consecutive_failures,
                        retry_after_seconds=backoff,
                    )

        # 回退到内存队列(Redis 不可用 / 失败 / 退避中)
        # Bug fix (H-R7): 当因退避跳过 Redis 直接走内存队列时，并发计数器
        # 已在 submit 开头原子自增，此处需回滚，避免计数器泄漏。
        # 注意：redis_enqueue_failed 路径已在 line 342 回滚过，此处仅处理
        # 退避路径（_in_backoff=True 且未尝试 Redis enqueue）。
        if _in_backoff and new_count > 0:
            await asyncio.to_thread(self._decrement_concurrent_requests, tenant_id)
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
                await self._dispatch(request, result)
            else:
                await self._handle_scheduling_failure_redis(request, result)

    async def _handle_scheduling_failure_redis(
        self, request: SchedulingRequest, result: SchedulingResult
    ):
        """处理Redis队列中调度失败的请求"""
        request_id = request.request_id
        tenant_id = getattr(request, "tenant_id", "default")

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
            # 配额回收
            await asyncio.to_thread(self._decrement_concurrent_requests, tenant_id)

            # 存储错误结果
            await self._redis_queue.store_result(
                request_id,
                {"status": "error", "reason": result.reason},
            )

    async def _dispatch(self, request: SchedulingRequest, result: SchedulingResult):
        """分发请求到Worker（真实HTTP调用）

        Bug fix (C-R1): 覆盖父类 _dispatch 但遗漏了 _reserve_for_request，
        导致分布式模式下 GPU 可被超卖。修复：在发送前调用 _reserve_for_request，
        在失败/错误路径调用 _release_for_request。
        """
        request_id = request.request_id

        # 1. 预留 GPU 显存（原子，防止超卖）
        reserved = self._reserve_for_request(request, result)
        if not reserved:
            logger.warning(
                "reservation_failed_after_scheduling",
                request_id=request_id,
            )
            await self._handle_dispatch_failure(
                request, "GPU overcommitted (reservation failed)"
            )
            return

        # 2. 更新运行状态
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

        # 3. 获取分配的Worker节点
        if not result.assigned_nodes:
            logger.error("no_nodes_assigned", request_id=request_id)
            self._release_for_request(request_id)
            await self._handle_dispatch_failure(request, "No nodes assigned")
            return

        # 4. 选择第一个可用的Worker节点
        node_id = result.assigned_nodes[0]
        worker_endpoint = await self._worker_registry.get_worker(node_id)

        if not worker_endpoint:
            logger.error("worker_not_found", request_id=request_id, node_id=node_id)
            self._release_for_request(request_id)
            await self._handle_dispatch_failure(request, f"Worker {node_id} not found")
            return

        # 5. 构建采样参数
        metadata = getattr(request, "metadata", {}) or {}
        sampling_params = {
            "temperature": metadata.get("temperature", 0.7),
            "top_p": metadata.get("top_p", 0.9),
            "top_k": metadata.get("top_k", 50),
            "max_tokens": metadata.get("max_tokens", 512),
            "repetition_penalty": metadata.get("repetition_penalty", 1.0),
        }

        # 6. 异步发送到Worker（不阻塞调度循环）
        asyncio.create_task(self._send_to_worker(request, worker_endpoint, sampling_params))

    async def _send_to_worker(
        self,
        request: SchedulingRequest,
        worker_endpoint: WorkerEndpoint,
        sampling_params: dict[str, Any],
    ):
        """发送请求到Worker并处理响应"""
        request_id = request.request_id
        tenant_id = getattr(request, "tenant_id", "default")

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
                self.stats["successful_requests"] += 1
                # 释放 GPU 预留
                self._release_for_request(request_id)
                # 配额回收
                await asyncio.to_thread(self._decrement_concurrent_requests, tenant_id)
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
        """处理分发失败（重试或标记失败）

        Bug fix (M-R8): retry_count 从未在分布式分发路径中递增，
        导致 _handle_dispatch_failure 永远读到 0，请求无限重试。
        修复：在检查前递增 request.retry_count。

        Bug fix (M-R9): 重新入队时未递增 pending_requests，
        导致统计计数不准确。修复：入队时递增 pending_requests。
        """
        request_id = request.request_id
        tenant_id = getattr(request, "tenant_id", "default")
        # Bug fix (M-R8): 递增 retry_count
        request.retry_count += 1
        retry_count = request.retry_count

        if retry_count < self.max_retries:
            logger.warning(
                "dispatch_retry",
                request_id=request_id,
                error=error,
                retry_count=retry_count,
            )
            # 重新入队等待重试
            # 释放当前预留（重试时会重新 _reserve_for_request）
            self._release_for_request(request_id)
            if self._redis_queue:
                queued_request = QueuedRequest(
                    request_id=request.request_id,
                    model_name=request.model,
                    prompt=request.prompt,
                    priority=request.priority,
                    created_at=request.created_at,
                    metadata={"retry_count": retry_count},
                )
                await self._redis_queue.requeue(queued_request)
                # Bug fix (M-R9): 重新入队时递增 pending_requests
                self.stats["pending_requests"] += 1
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
            # 释放 GPU 预留（Bug fix: 防止预留泄漏）
            self._release_for_request(request_id)
            # 配额回收
            await asyncio.to_thread(self._decrement_concurrent_requests, tenant_id)

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
        """同步获取队列统计（默认 API，向后兼容）

        适用于监控 / CLI / 健康检查端点。
        在有 Redis 时返回 ``redis_enabled=True`` 与 ``queue_size=0``（占位），
        因为同步上下文中无法 await Redis。真实数据请用 :meth:`get_queue_stats_async`。
        """
        stats = {
            **self.stats,
            "running_size": len(self.running_requests),
            "available_nodes": len(self.available_nodes),
        }

        if self._use_redis and self._redis_queue:
            # 同步入口无法 await；用本地 queue size 兜底
            stats["queue_size"] = self.pending_queue.qsize()
            stats["redis_enabled"] = True
        else:
            stats["queue_size"] = self.pending_queue.qsize()
            stats["redis_enabled"] = False

        return stats

    async def get_queue_stats_async(self) -> dict[str, Any]:
        """异步获取队列统计（可在 event loop 中使用）

        在 Redis 启用时正确 await ``queue_size()``。
        """
        stats = self.get_queue_stats()
        if self._use_redis and self._redis_queue:
            stats["queue_size"] = await self._redis_queue.queue_size()
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
    _scheduler = DistributedScheduler(
        default_strategy=default_strategy,
        redis_url=redis_url,
    )
    await _scheduler.start()
    return _scheduler


async def close_scheduler():
    """关闭调度器"""
    global _scheduler
    if _scheduler:
        await _scheduler.stop()
        _scheduler = None
