"""集群管理器"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import structlog

from quantumflow.core.constants import NodeStatus
from quantumflow.scheduler.strategy.base import GPUResource, NodeResource

logger = structlog.get_logger().bind(component="cluster_manager")


@dataclass
class GPUInfo:
    """GPU信息"""

    gpu_id: int
    name: str
    memory_total: int
    memory_used: int
    utilization: float
    temperature: float

    def to_resource(self, node_id: str) -> GPUResource:
        """转换为GPUResource"""
        return GPUResource(
            gpu_id=self.gpu_id,
            memory_total=self.memory_total,
            memory_used=self.memory_used,
            utilization=self.utilization,
            temperature=self.temperature,
            node_id=node_id,
        )


@dataclass
class Node:
    """计算节点"""

    node_id: str
    hostname: str
    ip: str
    port: int
    gpu_count: int
    gpu_info: list[GPUInfo]
    status: NodeStatus
    labels: dict[str, str] = field(default_factory=dict)
    version: str = "1.0.0"
    last_heartbeat: datetime = field(default_factory=datetime.now)

    # 额外属性
    cpu_count: int = 0
    memory_total: int = 0
    memory_available: int = 0
    disk_total: int = 0
    disk_available: int = 0
    current_load: float = 0.0
    loaded_models: list[str] = field(default_factory=list)

    def to_resource(self) -> NodeResource:
        """转换为NodeResource"""
        return NodeResource(
            node_id=self.node_id,
            hostname=self.hostname,
            ip=self.ip,
            status=self.status.value,
            gpu_count=self.gpu_count,
            gpus=[gpu.to_resource(self.node_id) for gpu in self.gpu_info],
            cpu_count=self.cpu_count,
            memory_total=self.memory_total,
            memory_available=self.memory_available,
            disk_total=self.disk_total,
            disk_available=self.disk_available,
            load=self.current_load,
            labels=self.labels,
            loaded_models=self.loaded_models,
            version=self.version,
            last_heartbeat=self.last_heartbeat,
        )

    @property
    def available_gpus(self) -> list[GPUInfo]:
        """获取可用GPU"""
        return [gpu for gpu in self.gpu_info if gpu.memory_used < gpu.memory_total * 0.95]

    @property
    def is_healthy(self) -> bool:
        """节点是否健康"""
        return self.status == NodeStatus.HEALTHY


class ClusterManager:
    """
    集群管理器

    负责：
    - 节点注册和注销
    - 心跳监控
    - 节点状态管理
    - 服务发现
    """

    def __init__(
        self,
        heartbeat_interval: int = 5,
        heartbeat_timeout: int = 30,
    ):
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout

        # 节点存储
        self.nodes: dict[str, Node] = {}
        self._lock = asyncio.Lock()

        # 心跳任务
        self._heartbeat_tasks: dict[str, asyncio.Task] = {}

        # 事件回调
        self._event_handlers: dict[str, list[Callable]] = {
            "node_joined": [],
            "node_left": [],
            "node_health_changed": [],
            "node_heartbeat": [],
        }

        # 运行状态
        self._running = False

        logger.info(
            "cluster_manager_created",
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )

    async def start(self):
        """启动集群管理器"""
        self._running = True
        logger.info("cluster_manager_started")

    async def stop(self):
        """停止集群管理器"""
        self._running = False

        # 停止所有心跳任务
        for task in self._heartbeat_tasks.values():
            task.cancel()

        logger.info("cluster_manager_stopped")

    # ==================== 节点管理 ====================

    async def register_node(self, node_info: dict) -> Node:
        """
        注册新节点

        Args:
            node_info: 节点信息字典

        Returns:
            Node: 注册的节点
        """
        async with self._lock:
            node = Node(
                node_id=node_info["node_id"],
                hostname=node_info["hostname"],
                ip=node_info["ip"],
                port=node_info["port"],
                gpu_count=node_info["gpu_count"],
                gpu_info=[GPUInfo(**gpu) for gpu in node_info.get("gpu_info", [])],
                status=NodeStatus.HEALTHY,
                labels=node_info.get("labels", {}),
                version=node_info.get("version", "1.0.0"),
                cpu_count=node_info.get("cpu_count", 0),
                memory_total=node_info.get("memory_total", 0),
                memory_available=node_info.get("memory_available", 0),
                loaded_models=node_info.get("loaded_models", []),
            )

            self.nodes[node.node_id] = node

        # 启动心跳监控
        self._start_heartbeat_monitor(node.node_id)

        # 触发事件
        await self._emit_event("node_joined", node)

        logger.info(
            "node_registered",
            node_id=node.node_id,
            hostname=node.hostname,
            gpu_count=node.gpu_count,
        )

        return node

    async def unregister_node(self, node_id: str):
        """注销节点"""
        async with self._lock:
            node = self.nodes.pop(node_id, None)

        if node:
            # 停止心跳监控
            if node_id in self._heartbeat_tasks:
                self._heartbeat_tasks[node_id].cancel()
                del self._heartbeat_tasks[node_id]

            # 触发事件
            await self._emit_event("node_left", node)

            logger.info("node_unregistered", node_id=node_id)

    async def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """更新节点状态"""
        async with self._lock:
            if node_id not in self.nodes:
                return False

            old_status = self.nodes[node_id].status
            self.nodes[node_id].status = status

            if old_status != status:
                await self._emit_event(
                    "node_health_changed", self.nodes[node_id], old_status, status
                )

            logger.info(
                "node_status_updated",
                node_id=node_id,
                old_status=old_status.value,
                new_status=status.value,
            )

            return True

    async def update_node_info(self, node_id: str, gpu_info: list[dict], load: float = None):
        """更新节点信息"""
        async with self._lock:
            if node_id not in self.nodes:
                return

            self.nodes[node_id].gpu_info = [GPUInfo(**gpu) for gpu in gpu_info]
            self.nodes[node_id].last_heartbeat = datetime.now()

            if load is not None:
                self.nodes[node_id].current_load = load

    async def add_loaded_model(self, node_id: str, model_name: str):
        """添加已加载模型"""
        async with self._lock:
            if node_id in self.nodes:
                if model_name not in self.nodes[node_id].loaded_models:
                    self.nodes[node_id].loaded_models.append(model_name)

    async def remove_loaded_model(self, node_id: str, model_name: str):
        """移除已加载模型"""
        async with self._lock:
            if node_id in self.nodes:
                if model_name in self.nodes[node_id].loaded_models:
                    self.nodes[node_id].loaded_models.remove(model_name)

    # ==================== 查询接口 ====================

    async def get_node(self, node_id: str) -> Node | None:
        """获取节点"""
        return self.nodes.get(node_id)

    async def get_node_resource(self, node_id: str) -> NodeResource | None:
        """获取节点资源信息"""
        node = self.nodes.get(node_id)
        return node.to_resource() if node else None

    async def get_nodes(
        self,
        status: NodeStatus | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[Node]:
        """获取节点列表"""
        nodes = list(self.nodes.values())

        if status:
            nodes = [n for n in nodes if n.status == status]

        if labels:
            nodes = [n for n in nodes if all(n.labels.get(k) == v for k, v in labels.items())]

        return nodes

    async def get_node_resources(
        self,
        status: NodeStatus | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[NodeResource]:
        """获取节点资源列表"""
        nodes = await self.get_nodes(status=status, labels=labels)
        return [n.to_resource() for n in nodes]

    async def get_healthy_nodes(self) -> list[Node]:
        """获取健康节点"""
        return await self.get_nodes(status=NodeStatus.HEALTHY)

    async def find_best_nodes(
        self, required_gpus: int, labels: dict[str, str] | None = None
    ) -> list[Node]:
        """查找最优节点组合"""
        healthy_nodes = await self.get_healthy_nodes()

        if labels:
            healthy_nodes = [
                n for n in healthy_nodes if all(n.labels.get(k) == v for k, v in labels.items())
            ]

        # 按可用GPU数量排序
        sorted_nodes = sorted(
            healthy_nodes,
            key=lambda n: (len(n.available_gpus), n.current_load),
            reverse=True,
        )

        selected = []
        remaining_gpus = required_gpus

        for node in sorted_nodes:
            if remaining_gpus <= 0:
                break

            available = len(node.available_gpus)
            if available > 0:
                selected.append(node)
                remaining_gpus -= available

        return selected

    async def get_cluster_stats(self) -> dict:
        """获取集群统计"""
        nodes = list(self.nodes.values())

        total_gpus = sum(n.gpu_count for n in nodes)
        available_gpus = sum(len(n.available_gpus) for n in nodes)

        return {
            "total_nodes": len(nodes),
            "healthy_nodes": sum(1 for n in nodes if n.status == NodeStatus.HEALTHY),
            "unhealthy_nodes": sum(1 for n in nodes if n.status == NodeStatus.UNHEALTHY),
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "total_models": len({m for n in nodes for m in n.loaded_models}),
        }

    # ==================== 心跳监控 ====================

    def _start_heartbeat_monitor(self, node_id: str):
        """启动心跳监控"""
        task = asyncio.create_task(self._heartbeat_loop(node_id))
        self._heartbeat_tasks[node_id] = task

    async def _heartbeat_loop(self, node_id: str):
        """心跳检测循环"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                node = self.nodes.get(node_id)
                if not node:
                    break

                # 节点已经不健康，停止监控
                if node.status != NodeStatus.HEALTHY:
                    break

                # 检查是否超时
                elapsed = (datetime.now() - node.last_heartbeat).total_seconds()
                if elapsed > self.heartbeat_timeout:
                    await self._handle_node_timeout(node_id)
                    # 超时处理后，节点状态变为 OFFLINE，下次循环会退出

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("heartbeat_error", node_id=node_id, error=str(e))

    async def _handle_node_timeout(self, node_id: str):
        """处理节点超时"""
        logger.warning("node_timeout", node_id=node_id)
        await self.update_node_status(node_id, NodeStatus.OFFLINE)

        # 触发事件
        await self._emit_event(
            "node_health_changed",
            self.nodes.get(node_id),
            NodeStatus.HEALTHY,
            NodeStatus.OFFLINE,
        )

    # ==================== 事件系统 ====================

    def on(self, event: str, handler: Callable):
        """注册事件处理器"""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    async def _emit_event(self, event: str, *args, **kwargs):
        """触发事件"""
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error("event_handler_error", event_name=event, error=str(e))
