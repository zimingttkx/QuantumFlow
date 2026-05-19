"""Worker任务抓取器

Worker从Redis队列拉取任务执行，实现与Controller的解耦。
"""

import asyncio
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
import structlog

from quantumflow.storage.redis_queue import RedisQueue, QueuedRequest
from quantumflow.storage.connection import get_redis_manager
from quantumflow.inference.engine import InferenceEngine, SamplingParams

logger = structlog.get_logger().bind(component="task_fetcher")


@dataclass
class TaskFetcherConfig:
    """任务抓取器配置"""

    node_id: str
    redis_url: str = "redis://localhost:6379/0"
    poll_interval_ms: int = 100  # 轮询间隔
    batch_size: int = 1  # 每次抓取数量
    max_retries: int = 3  # 最大重试次数
    active_tasks: int = 10  # 最大并发任务数


class TaskFetcher:
    """
    任务抓取器

    Worker使用此组件从Redis队列拉取任务并执行。
    实现与Controller的解耦，支持分布式部署。

    使用方式：
    1. 创建TaskFetcher并配置
    2. 注册任务处理函数
    3. 启动抓取循环
    4. 停止时优雅关闭
    """

    def __init__(
        self,
        config: TaskFetcherConfig,
        engine: Optional[InferenceEngine] = None,
    ):
        self.config = config
        self.engine = engine

        self._redis_queue: Optional[RedisQueue] = None
        self._running = False
        self._fetch_task: Optional[asyncio.Task] = None
        self._active_tasks: Dict[str, asyncio.Task] = {}

        # 任务处理函数
        self._task_handlers: Dict[str, Callable] = {}

        # 统计
        self.stats = {
            "tasks_pulled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
        }

        logger.info(
            "task_fetcher_created",
            node_id=config.node_id,
            poll_interval_ms=config.poll_interval_ms,
            batch_size=config.batch_size,
        )

    def register_handler(self, model_name: str, handler: Callable):
        """
        注册任务处理器

        Args:
            model_name: 模型名称（* 表示处理所有模型）
            handler: 异步处理函数，签名为 handler(request: QueuedRequest) -> bool
        """
        self._task_handlers[model_name] = handler
        logger.info("task_handler_registered", model_name=model_name)

    async def start(self):
        """启动任务抓取器"""
        if self._running:
            logger.warning("task_fetcher_already_running")
            return

        # 连接Redis
        redis_mgr = await get_redis_manager()
        if redis_mgr.is_connected:
            self._redis_queue = RedisQueue(redis_url=self.config.redis_url)
            await self._redis_queue.connect()
            logger.info("task_fetcher_redis_connected")
        else:
            logger.error("task_fetcher_redis_not_available")
            return

        self._running = True
        self._fetch_task = asyncio.create_task(self._fetch_loop())

        logger.info("task_fetcher_started", node_id=self.config.node_id)

    async def stop(self):
        """停止任务抓取器"""
        if not self._running:
            return

        self._running = False

        # 停止抓取循环
        if self._fetch_task:
            self._fetch_task.cancel()
            try:
                await self._fetch_task
            except asyncio.CancelledError:
                pass

        # 等待活跃任务完成（带超时）
        if self._active_tasks:
            logger.info("waiting_for_active_tasks", count=len(self._active_tasks))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks.values(), return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("active_tasks_timeout")

        # 断开Redis
        if self._redis_queue:
            await self._redis_queue.disconnect()

        logger.info("task_fetcher_stopped", stats=self.stats)

    async def _fetch_loop(self):
        """抓取循环"""
        logger.info("fetch_loop_started")

        while self._running:
            try:
                # 等待一段时候再抓取
                await asyncio.sleep(self.config.poll_interval_ms / 1000.0)

                # 检查是否有空闲槽位
                if len(self._active_tasks) >= self.config.active_tasks:
                    continue

                # 从Redis队列获取任务
                if self._redis_queue:
                    for _ in range(self.config.batch_size):
                        if len(self._active_tasks) >= self.config.active_tasks:
                            break

                        request = await self._redis_queue.dequeue(timeout=1)
                        if request:
                            await self._process_task(request)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("fetch_loop_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("fetch_loop_stopped")

    async def _process_task(self, request: QueuedRequest):
        """处理任务"""
        request_id = request.request_id

        logger.info(
            "task_received",
            request_id=request_id,
            model_name=request.model_name,
            priority=request.priority,
        )

        self.stats["tasks_pulled"] += 1

        # 创建任务
        task = asyncio.create_task(self._execute_task(request))
        self._active_tasks[request_id] = task

        # 添加完成回调
        task.add_done_callback(
            lambda t, req_id=request_id: self._on_task_done(req_id, t)
        )

    async def _execute_task(self, request: QueuedRequest) -> bool:
        """执行任务"""
        request_id = request.request_id

        try:
            # 找到处理器
            handler = self._task_handlers.get(
                request.model_name
            ) or self._task_handlers.get("*")

            if not handler:
                # 默认处理：使用引擎执行推理
                success = await self._default_handler(request)
            else:
                success = await handler(request)

            if success:
                self.stats["tasks_completed"] += 1
                # 存储成功结果
                if self._redis_queue:
                    await self._redis_queue.store_result(
                        request_id,
                        {"status": "success"},
                    )
            else:
                self.stats["tasks_failed"] += 1
                # 重新入队
                await self._retry_task(request)

            return success

        except Exception as e:
            logger.error(
                "task_execution_error",
                request_id=request_id,
                error=str(e),
            )
            self.stats["tasks_failed"] += 1
            await self._retry_task(request)
            return False

    async def _default_handler(self, request: QueuedRequest) -> bool:
        """默认任务处理器：使用推理引擎执行"""
        if not self.engine:
            logger.error("no_engine_available", request_id=request.request_id)
            return False

        try:
            # 检查模型是否已加载
            if not await self.engine.is_model_loaded(request.model_name):
                logger.warning(
                    "model_not_loaded",
                    request_id=request.request_id,
                    model=request.model_name,
                )
                return False

            # 构建采样参数
            metadata = request.metadata or {}
            sampling_params = SamplingParams(
                temperature=metadata.get("temperature", 0.7),
                top_p=metadata.get("top_p", 0.9),
                top_k=metadata.get("top_k", 50),
                max_tokens=metadata.get("max_tokens", 512),
                repetition_penalty=metadata.get("repetition_penalty", 1.0),
            )

            # 执行推理
            results = await self.engine.generate(
                model_name=request.model_name,
                prompts=[request.prompt],
                sampling_params=sampling_params,
            )

            if results and len(results) > 0:
                logger.info(
                    "task_completed",
                    request_id=request.request_id,
                    output_length=len(results[0].outputs[0]) if results[0].outputs else 0,
                )
                return True
            else:
                return False

        except Exception as e:
            logger.error(
                "default_handler_error",
                request_id=request.request_id,
                error=str(e),
            )
            return False

    async def _retry_task(self, request: QueuedRequest):
        """重试任务"""
        if not self._redis_queue:
            return

        metadata = request.metadata or {}
        retry_count = metadata.get("retry_count", 0)

        if retry_count < self.config.max_retries:
            metadata["retry_count"] = retry_count + 1
            request.metadata = metadata

            await self._redis_queue.requeue(request, increment_retry=False)
            self.stats["tasks_retried"] += 1

            logger.info(
                "task_requeued",
                request_id=request.request_id,
                retry_count=retry_count + 1,
            )
        else:
            # 超过最大重试次数，存储失败结果
            await self._redis_queue.store_result(
                request.request_id,
                {"status": "error", "reason": "max_retries_exceeded"},
            )

            logger.error(
                "task_max_retries_exceeded",
                request_id=request.request_id,
            )

    def _on_task_done(self, request_id: str, task: asyncio.Task):
        """任务完成回调"""
        self._active_tasks.pop(request_id, None)

        if task.exception():
            logger.error(
                "task_failed",
                request_id=request_id,
                error=str(task.exception()),
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "active_tasks": len(self._active_tasks),
            "running": self._running,
        }
