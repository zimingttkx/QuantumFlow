"""调度器核心"""

from typing import Dict, List, Optional, Callable, Any
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import structlog

from quantumflow.scheduler.strategy.base import (
    SchedulingRequest,
    SchedulingResult,
    NodeResource,
    SchedulingStrategy,
)
from quantumflow.scheduler.strategy.gang import GangSchedulingStrategy
from quantumflow.scheduler.strategy.pack import PackSchedulingStrategy
from quantumflow.scheduler.strategy.adaptive import AdaptiveSchedulingStrategy
from quantumflow.core.exceptions import (
    SchedulerError,
    SchedulerNodeUnavailableError,
)
from quantumflow.utils.config import get_config

logger = structlog.get_logger().bind(component="scheduler")


@dataclass
class QueueItem:
    """队列项"""

    request: SchedulingRequest
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    result: Optional[SchedulingResult] = None
    retry_count: int = 0


class Scheduler:
    """
    核心调度器

    负责：
    - 请求队列管理
    - 节点资源管理
    - 调度策略选择
    - 请求分发
    """

    def __init__(
        self,
        default_strategy: str = "adaptive",
        loop_interval_ms: int = 100,
        max_retries: int = 3,
    ):
        self.config = get_config()

        # 调度配置
        self.default_strategy = default_strategy
        self.loop_interval_ms = loop_interval_ms
        self.max_retries = max_retries

        # 策略注册
        self.strategies: Dict[str, SchedulingStrategy] = {}
        self._init_strategies()

        # 自适应策略
        self.adaptive_strategy = AdaptiveSchedulingStrategy(
            strategies=self.strategies
        )

        # 请求队列（使用计数器确保可排序）
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._request_counter: int = 0
        self.running_requests: Dict[str, QueueItem] = {}

        # 节点状态
        self.available_nodes: Dict[str, NodeResource] = {}
        self.node_update_callbacks: List[Callable] = []

        # 调度循环
        self._scheduling_task: Optional[asyncio.Task] = None
        self._running = False

        # 统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "pending_requests": 0,
        }

    def _init_strategies(self):
        """初始化调度策略"""
        # Gang调度 - 大模型
        self.strategies["gang"] = GangSchedulingStrategy(
            config=self.config.scheduler.model_dump()
        )

        # Pack调度 - 小模型
        self.strategies["pack"] = PackSchedulingStrategy(
            config=self.config.scheduler.model_dump()
        )

    async def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("scheduler_already_running")
            return

        self._running = True
        self._scheduling_task = asyncio.create_task(self._scheduling_loop())

        logger.info("scheduler_started", strategy=self.default_strategy)

    async def stop(self):
        """停止调度器"""
        self._running = False

        if self._scheduling_task:
            self._scheduling_task.cancel()
            try:
                await self._scheduling_task
            except asyncio.CancelledError:
                pass

        logger.info("scheduler_stopped")

    async def submit(
        self,
        request: SchedulingRequest,
    ) -> str:
        """
        提交推理请求

        Args:
            request: 调度请求

        Returns:
            request_id: 请求ID
        """
        self.stats["total_requests"] += 1
        self.stats["pending_requests"] += 1

        logger.info(
            "request_submitted",
            request_id=request.request_id,
            model=request.model,
            priority=request.priority,
        )

        # 加入优先级队列 (优先级, 创建时间, 序号, 请求)
        # 序号确保相同优先级和时间戳时可以比较
        priority = request.priority
        self._request_counter += 1
        await self.pending_queue.put((priority, request.created_at, self._request_counter, request))

        return request.request_id

    async def _scheduling_loop(self):
        """调度循环"""
        logger.info("scheduling_loop_started")

        while self._running:
            try:
                await asyncio.sleep(self.loop_interval_ms / 1000.0)

                # 获取可用节点
                if not self.available_nodes:
                    continue

                # 批量处理队列中的请求
                await self._process_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("scheduling_loop_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("scheduling_loop_stopped")

    async def _process_batch(self):
        """批量处理请求"""
        # 获取待处理请求
        batch_size = min(self.pending_queue.qsize(), 10)
        requests_batch = []

        for _ in range(batch_size):
            try:
                priority, created_at, counter, request = self.pending_queue.get_nowait()
                requests_batch.append(request)
            except asyncio.QueueEmpty:
                break

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
                await self._handle_scheduling_failure(request, result)

    async def _schedule_request(
        self, request: SchedulingRequest
    ) -> SchedulingResult:
        """调度单个请求"""
        # 获取可用节点列表
        nodes = list(self.available_nodes.values())

        if not nodes:
            return SchedulingResult(
                success=False,
                reason="No available nodes",
            )

        # 根据策略选择节点
        strategy = self._get_strategy(request, nodes)

        if not strategy:
            return SchedulingResult(
                success=False,
                reason=f"Strategy not found: {self.default_strategy}",
            )

        result = strategy.select_nodes(request, nodes)

        return result

    def _get_strategy(
        self, request: SchedulingRequest, nodes: List[NodeResource]
    ) -> Optional[SchedulingStrategy]:
        """获取调度策略"""
        if self.default_strategy == "adaptive":
            return self.adaptive_strategy

        return self.strategies.get(self.default_strategy)

    async def _dispatch(
        self, request: SchedulingRequest, result: SchedulingResult
    ):
        """分发请求到Worker"""
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

        # TODO: 实际发送到Worker
        # 这里暂时模拟
        # 注意：fire-and-forget 模式下，任务失败会被静默忽略
        # 后续应接入真正的 Worker 通信并跟踪任务状态
        task = asyncio.create_task(self._simulate_execution(request_id))
        # 临时方案：添加任务完成回调以便观察状态
        task.add_done_callback(
            lambda t: logger.debug("simulated_execution_done", request_id=request_id)
            if not t.exception()
            else logger.error("simulated_execution_failed", request_id=request_id, error=str(t.exception()))
        )

    async def _handle_scheduling_failure(
        self, request: SchedulingRequest, result: SchedulingResult
    ):
        """处理调度失败"""
        request_id = request.request_id

        # 重试逻辑：递增 retry_count
        request.retry_count += 1

        if request.retry_count < self.max_retries:
            logger.warning(
                "scheduling_retry",
                request_id=request_id,
                reason=result.reason,
                retry_count=request.retry_count,
            )
            # 重新加入队列
            self._request_counter += 1
            await self.pending_queue.put(
                (request.priority, request.created_at, self._request_counter, request)
            )
        else:
            logger.error(
                "scheduling_failed",
                request_id=request_id,
                reason=result.reason,
                max_retries=self.max_retries,
            )
            self.stats["failed_requests"] += 1
            self.stats["pending_requests"] -= 1

    async def _simulate_execution(self, request_id: str):
        """模拟请求执行"""
        await asyncio.sleep(0.5)  # 模拟延迟

        if request_id in self.running_requests:
            del self.running_requests[request_id]
            logger.info("request_completed", request_id=request_id)

    # ==================== 节点管理 ====================

    async def register_node(self, node: NodeResource):
        """注册节点"""
        self.available_nodes[node.node_id] = node

        logger.info(
            "node_registered",
            node_id=node.node_id,
            gpu_count=node.gpu_count,
            status=node.status,
        )

        # 触发回调
        for callback in self.node_update_callbacks:
            try:
                await callback("register", node)
            except Exception as e:
                logger.error("node_callback_error", error=str(e))

    async def unregister_node(self, node_id: str):
        """注销节点"""
        if node_id in self.available_nodes:
            node = self.available_nodes.pop(node_id)

            logger.info("node_unregistered", node_id=node_id)

            # 触发回调
            for callback in self.node_update_callbacks:
                try:
                    await callback("unregister", node)
                except Exception as e:
                    logger.error("node_callback_error", error=str(e))

    async def update_node(self, node: NodeResource):
        """更新节点状态"""
        self.available_nodes[node.node_id] = node

        logger.debug(
            "node_updated",
            node_id=node.node_id,
            gpu_count=node.gpu_count,
            load=node.load,
        )

    def on_node_update(self, callback: Callable):
        """注册节点更新回调"""
        self.node_update_callbacks.append(callback)

    # ==================== 查询接口 ====================

    def get_pending_requests(self) -> List[SchedulingRequest]:
        """获取待调度请求"""
        requests = []
        for _, _, _, request in self.pending_queue._queue:  # type: ignore
            if isinstance(request, SchedulingRequest):
                requests.append(request)
        return requests

    def get_running_requests(self) -> Dict[str, QueueItem]:
        """获取运行中的请求"""
        return self.running_requests.copy()

    def get_stats(self) -> Dict[str, Any]:
        """获取调度统计"""
        return {
            **self.stats,
            "queue_size": self.pending_queue.qsize(),
            "running_size": len(self.running_requests),
            "available_nodes": len(self.available_nodes),
        }
