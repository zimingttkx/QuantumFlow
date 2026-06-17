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
    start_time: datetime = field(default_factory=datetime.now)  # 节点启动时间

    # 额外属性
    cpu_count: int = 0
    memory_total: int = 0
    memory_available: int = 0
    disk_total: int = 0
    disk_available: int = 0
    current_load: float = 0.0
    loaded_models: list[str] = field(default_factory=list)

    # ========== 容灾相关字段 ==========
    # 副本角色: primary, secondary
    replica_role: str = "secondary"
    # VRAM 快照（用于状态同步）
    vram_snapshot: dict = field(default_factory=dict)
    # GPU 健康状态: {gpu_id: "healthy|degraded|fail"}
    gpu_health: dict = field(default_factory=dict)
    # 模型健康状态: {model_name: "healthy|degraded|fail"}
    model_health: dict = field(default_factory=dict)
    # 通信健康状态: {node_id: "healthy|timeout|fail"}
    communication_health: dict = field(default_factory=dict)
    # 最后一次副本同步时间
    last_replica_sync: datetime = field(default_factory=datetime.now)
    # 故障计数（用于抖动判定）
    failure_count: int = 0
    # 连续失败次数
    consecutive_failures: int = 0
    # 失败原因列表（最近 N 条）
    failure_reasons: list[str] = field(default_factory=list)
    # Worker 服务端口（供 scheduler 直接构造 endpoint）
    worker_port: int = 8000

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
            load=self._compute_load(),
            labels=self.labels,
            loaded_models=self.loaded_models,
            version=self.version,
            last_heartbeat=self.last_heartbeat,
        )

    def _compute_load(self) -> float:
        """基于 GPU 显存使用率 + 利用率计算真实负载"""
        if not self.gpu_info:
            return self.current_load
        mem_used = sum(g.memory_used for g in self.gpu_info)
        mem_total = sum(g.memory_total for g in self.gpu_info) or 1
        mem_ratio = mem_used / mem_total
        util_ratio = (
            sum(g.utilization for g in self.gpu_info) / len(self.gpu_info)
            if self.gpu_info
            else 0.0
        )
        return max(0.0, min(1.0, 0.6 * mem_ratio + 0.4 * util_ratio))

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

        # 停止所有心跳任务并等待它们完成
        for task in self._heartbeat_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

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

        # 过滤掉 GPU 显存剩余 < 10% 的节点（与 scheduler 保持一致）
        healthy_nodes = [n for n in healthy_nodes if any(g.memory_free_percent > 0.1 for g in n.gpu_info)]

        # 按 (可用 GPU 数, 真实负载升序) 排序
        def node_score(n: Node) -> tuple:
            available = sum(1 for g in n.gpu_info if g.memory_free_percent > 0.1)
            return (available, -n._compute_load())

        sorted_nodes = sorted(healthy_nodes, key=node_score, reverse=True)

        selected = []
        remaining_gpus = required_gpus

        for node in sorted_nodes:
            if remaining_gpus <= 0:
                break

            available = sum(1 for g in node.gpu_info if g.memory_free_percent > 0.1)
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
        # 注意: update_node_status 已经触发了 node_health_changed 事件，
        # 这里不需要再次触发

    # ==================== 容灾相关方法 ====================

    async def update_node_health(
        self,
        node_id: str,
        health_status: str,
        reason: str | None = None,
    ) -> bool:
        """
        更新节点健康状态

        Args:
            node_id: 节点 ID
            health_status: 健康状态 ("healthy", "degraded", "unhealthy")
            reason: 状态变更原因

        Returns:
            是否更新成功
        """
        async with self._lock:
            if node_id not in self.nodes:
                return False

            node = self.nodes[node_id]

            if reason:
                node.failure_reasons.append(reason)

            if health_status == "unhealthy":
                node.consecutive_failures += 1
                node.failure_count += 1
            elif health_status == "healthy":
                node.consecutive_failures = 0

            logger.info(
                "node_health_updated",
                node_id=node_id,
                health_status=health_status,
                reason=reason,
                consecutive_failures=node.consecutive_failures,
            )

            return True

    async def get_replica_role(self, node_id: str) -> str | None:
        """
        获取节点副本角色

        Args:
            node_id: 节点 ID

        Returns:
            副本角色 ("primary", "secondary")，节点不存在返回 None
        """
        node = self.nodes.get(node_id)
        return node.replica_role if node else None

    async def set_replica_role(self, node_id: str, role: str) -> bool:
        """
        设置节点副本角色

        Args:
            node_id: 节点 ID
            role: 副本角色 ("primary", "secondary")

        Returns:
            是否设置成功
        """
        async with self._lock:
            if node_id not in self.nodes:
                return False

            old_role = self.nodes[node_id].replica_role
            self.nodes[node_id].replica_role = role

            logger.info(
                "node_replica_role_updated",
                node_id=node_id,
                old_role=old_role,
                new_role=role,
            )

            return True

    async def update_vram_snapshot(self, node_id: str, snapshot: dict) -> bool:
        """
        更新节点 VRAM 快照

        Args:
            node_id: 节点 ID
            snapshot: VRAM 快照数据

        Returns:
            是否更新成功
        """
        async with self._lock:
            if node_id not in self.nodes:
                return False

            self.nodes[node_id].vram_snapshot = snapshot
            self.nodes[node_id].last_replica_sync = datetime.now()
            return True

    async def update_gpu_health(
        self, node_id: str, gpu_id: int, status: str
    ) -> bool:
        """
        更新 GPU 健康状态

        Args:
            node_id: 节点 ID
            gpu_id: GPU ID
            status: 健康状态 ("healthy", "degraded", "fail")

        Returns:
            是否更新成功
        """
        async with self._lock:
            if node_id not in self.nodes:
                return False

            self.nodes[node_id].gpu_health[str(gpu_id)] = status
            return True

    async def update_model_health(
        self, node_id: str, model_name: str, status: str
    ) -> bool:
        """
        更新模型健康状态

        Args:
            node_id: 节点 ID
            model_name: 模型名称
            status: 健康状态 ("healthy", "degraded", "fail")

        Returns:
            是否更新成功
        """
        async with self._lock:
            if node_id not in self.nodes:
                return False

            self.nodes[node_id].model_health[model_name] = status
            return True

    async def update_communication_health(
        self, node_id: str, peer_node_id: str, status: str
    ) -> bool:
        """
        更新节点间通信健康状态

        Args:
            node_id: 本节点 ID
            peer_node_id: 对端节点 ID
            status: 健康状态 ("healthy", "timeout", "fail")

        Returns:
            是否更新成功
        """
        async with self._lock:
            if node_id not in self.nodes:
                return False

            self.nodes[node_id].communication_health[peer_node_id] = status
            return True

    async def get_node_failover_info(self, node_id: str) -> dict | None:
        """
        获取节点容灾相关信息

        Args:
            node_id: 节点 ID

        Returns:
            容灾信息字典，节点不存在返回 None
        """
        node = self.nodes.get(node_id)
        if not node:
            return None

        return {
            "node_id": node.node_id,
            "replica_role": node.replica_role,
            "vram_snapshot": node.vram_snapshot,
            "gpu_health": node.gpu_health,
            "model_health": node.model_health,
            "communication_health": node.communication_health,
            "last_replica_sync": node.last_replica_sync.isoformat(),
            "failure_count": node.failure_count,
            "consecutive_failures": node.consecutive_failures,
            "failure_reasons": node.failure_reasons[-10:],  # 最近 10 条
        }

    async def get_primary_nodes(self) -> list[Node]:
        """
        获取所有主节点

        Returns:
            主节点列表
        """
        return [n for n in self.nodes.values() if n.replica_role == "primary"]

    async def get_secondary_nodes(self) -> list[Node]:
        """
        获取所有备用节点

        Returns:
            备用节点列表
        """
        return [n for n in self.nodes.values() if n.replica_role == "secondary"]

    async def get_nodes_with_model(self, model_name: str) -> list[Node]:
        """
        获取加载了指定模型的节点

        Args:
            model_name: 模型名称

        Returns:
            节点列表
        """
        return [n for n in self.nodes.values() if model_name in n.loaded_models]

    async def reset_failure_count(self, node_id: str) -> bool:
        """
        重置节点故障计数

        Args:
            node_id: 节点 ID

        Returns:
            是否重置成功
        """
        async with self._lock:
            if node_id not in self.nodes:
                return False

            self.nodes[node_id].failure_count = 0
            self.nodes[node_id].consecutive_failures = 0
            self.nodes[node_id].failure_reasons = []
            return True

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
