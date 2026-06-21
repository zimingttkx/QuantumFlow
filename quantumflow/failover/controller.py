"""故障转移控制器

核心功能：
- 监听 ClusterManager 事件，检测故障
- 决策是否需要故障转移
- 执行故障转移流程
- 协调 ReplicaManager 和 LeaderElection
- 防止脑裂
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from quantumflow.cluster.manager import ClusterManager, NodeStatus
from quantumflow.failover.health_checker import HealthChecker
from quantumflow.failover.leader_election import LeaderElection
from quantumflow.failover.models import (
    FailoverEvent,
    FailoverState,
    HealthStatus,
    NodeFailoverState,
    ReplicaRole,
)
from quantumflow.failover.policy import FailoverPolicy, ReplicaPolicy
from quantumflow.failover.replica_manager import ReplicaManager
from quantumflow.failover.state_store import NodeStateStore

logger = structlog.get_logger().bind(component="failover_controller")


@dataclass
class FailoverDecision:
    """故障转移决策"""

    should_failover: bool
    reason: str
    failed_node: str
    target_nodes: list[str]
    affected_models: list[str]
    risk_level: str  # low, medium, high, critical


class FailoverController:
    """
    故障转移控制器

    核心职责：
    1. 监听 ClusterManager 事件，检测故障
    2. 决策是否需要故障转移
    3. 执行故障转移流程
    4. 协调 ReplicaManager 和 LeaderElection
    5. 防止脑裂
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        state_store: NodeStateStore,
        replica_manager: ReplicaManager,
        health_checker: HealthChecker,
        node_id: str,
        failover_policy: FailoverPolicy | None = None,
        replica_policy: ReplicaPolicy | None = None,
    ):
        self._cluster_manager = cluster_manager
        self._state_store = state_store
        self._replica_manager = replica_manager
        self._health_checker = health_checker
        self._node_id = node_id

        self._failover_policy = failover_policy or FailoverPolicy()
        self._replica_policy = replica_policy or ReplicaPolicy()

        # Leader 选举器
        self._leader_election: LeaderElection | None = None

        # 运行状态
        self._running = False
        self._failover_in_progress = False
        self._event_handlers: dict[str, list[callable]] = {}

        logger.info(
            "failover_controller_created",
            node_id=self._node_id,
            policy=self._failover_policy.to_dict(),
        )

    async def initialize(self) -> None:
        """初始化故障转移控制器"""
        # 创建 Leader 选举器
        self._leader_election = LeaderElection(
            node_id=self._node_id,
            cluster_manager=self._cluster_manager,
            state_store=self._state_store,
            replica_policy=self._replica_policy,
        )

        # 设置 Leader 回调
        self._leader_election.set_on_become_leader(self._on_become_leader)
        self._leader_election.set_on_lose_leader(self._on_lose_leader)

        # 注册集群事件监听
        self._cluster_manager.on("node_health_changed", self._on_node_health_changed)
        self._cluster_manager.on("node_left", self._on_node_left)

        logger.info("failover_controller_initialized")

    async def start(self) -> None:
        """启动故障转移控制器"""
        if self._running:
            logger.warning("failover_controller_already_running")
            return

        self._running = True

        # 启动 Leader 选举
        if self._leader_election:
            await self._leader_election.start()

        # 启动健康检测
        await self._health_checker.start()

        logger.info("failover_controller_started", node_id=self._node_id)

    async def stop(self) -> None:
        """停止故障转移控制器"""
        self._running = False

        # 停止健康检测
        await self._health_checker.stop()

        # 停止 Leader 选举
        if self._leader_election:
            await self._leader_election.stop()

        logger.info("failover_controller_stopped")

    # ==================== 事件处理 ====================

    async def _on_become_leader(self, node_id: str, term: int) -> None:
        """成为 Leader 时的回调"""
        logger.info("became_leader", node_id=node_id, term=term)
        await self._emit_event("leader_changed", {"is_leader": True, "node_id": node_id})

    async def _on_lose_leader(self, node_id: str) -> None:
        """失去 Leader 地位时的回调"""
        logger.info("lost_leadership", node_id=node_id)
        await self._emit_event("leader_changed", {"is_leader": False, "node_id": node_id})

    async def _on_node_health_changed(
        self, node: Any, old_status: Any, new_status: Any
    ) -> None:
        """节点健康状态变化时的回调"""
        logger.info(
            "node_health_changed_event",
            node_id=node.node_id,
            old_status=old_status.value if hasattr(old_status, 'value') else old_status,
            new_status=new_status.value if hasattr(new_status, 'value') else new_status,
        )

        # 如果节点变为不健康，触发故障检测
        if new_status == NodeStatus.UNHEALTHY:
            await self._handle_unhealthy_node(node.node_id)

    async def _on_node_left(self, node: Any) -> None:
        """节点离开时的回调"""
        logger.info("node_left_event", node_id=node.node_id)
        await self._handle_node_failure(node.node_id, "node_left")

    async def _handle_unhealthy_node(self, node_id: str) -> None:
        """处理不健康节点"""
        if not self._failover_policy.auto_failover_enabled:
            logger.info("auto_failover_disabled")
            return

        # 执行健康检查
        result = await self._health_checker.check_node_health(node_id)

        if result.is_unhealthy():
            await self._handle_node_failure(node_id, "health_check_failed")

    async def _handle_node_failure(self, node_id: str, reason: str) -> None:
        """处理节点故障"""
        if self._failover_in_progress:
            logger.warning("failover_in_progress_skip", node_id=node_id)
            return

        # 检查自己是否是 Leader
        if self._leader_election and not await self._leader_election.check_is_leader():
            logger.info("not_leader_skip_failover", node_id=node_id)
            return

        # 执行故障转移
        await self.initiate_failover(node_id, reason)

    # ==================== 故障转移接口 ====================

    async def initiate_failover(self, failed_node_id: str, reason: str) -> bool:
        """
        发起故障转移

        Args:
            failed_node_id: 故障节点 ID
            reason: 故障原因

        Returns:
            是否成功发起故障转移
        """
        if self._failover_in_progress:
            logger.warning("failover_already_in_progress")
            return False

        self._failover_in_progress = True

        try:
            logger.info(
                "initiating_failover",
                failed_node_id=failed_node_id,
                reason=reason,
            )

            # 1. 获取故障节点上的所有模型
            failed_node = await self._cluster_manager.get_node(failed_node_id)
            if not failed_node:
                logger.error("failed_node_not_found", node_id=failed_node_id)
                return False

            affected_models = list(failed_node.loaded_models)

            # 2. 决策是否需要故障转移
            decision = await self._make_failover_decision(
                failed_node_id, reason, affected_models
            )

            if not decision.should_failover:
                logger.info(
                    "failover_not_needed",
                    node_id=failed_node_id,
                    reason=decision.reason,
                )
                return False

            # 3. 对每个模型执行故障转移
            failover_event = FailoverEvent(
                event_id=self._state_store.generate_event_id(),
                event_type="node_fail",
                source_node=failed_node_id,
                target_node=None,
                reason=reason,
                timestamp=datetime.now(),
                success=True,
                details={"models": affected_models, "risk_level": decision.risk_level},
            )

            for model_name in decision.affected_models:
                try:
                    target_node = await self._failover_model(model_name, failed_node_id)
                    if target_node:
                        failover_event.target_node = target_node
                        failover_event.target_nodes[model_name] = target_node
                except Exception as e:
                    logger.error(
                        "model_failover_failed",
                        model_name=model_name,
                        error=str(e),
                    )
                    failover_event.success = False

            # 4. 保存故障转移事件
            await self._state_store.save_failover_event(failover_event)

            logger.info(
                "failover_completed",
                event_id=failover_event.event_id,
                success=failover_event.success,
            )

            # 5. 触发回调
            await self._emit_event("failover_completed", {
                "event": failover_event,
                "decision": decision,
            })

            return failover_event.success

        finally:
            self._failover_in_progress = False

    async def _make_failover_decision(
        self, failed_node_id: str, reason: str, affected_models: list[str]
    ) -> FailoverDecision:
        """制定故障转移决策"""
        # 检查故障节点是否是主节点
        failed_node = await self._cluster_manager.get_node(failed_node_id)
        is_primary = failed_node.replica_role == "primary" if failed_node else False

        # 确定目标节点
        target_nodes = []
        for model_name in affected_models:
            locations = await self._replica_manager.get_model_locations(model_name)
            target_nodes.extend(locations.keys())

        target_nodes = list(set(target_nodes) - {failed_node_id})

        # 评估风险等级
        risk_level = "low"
        if not target_nodes:
            risk_level = "critical"
        elif not is_primary:
            risk_level = "low"
        elif len(affected_models) > 3:
            risk_level = "high"

        return FailoverDecision(
            should_failover=is_primary or len(affected_models) > 0,
            reason=reason,
            failed_node=failed_node_id,
            target_nodes=target_nodes,
            affected_models=affected_models,
            risk_level=risk_level,
        )

    async def _failover_model(
        self, model_name: str, failed_node_id: str
    ) -> str | None:
        """对单个模型执行故障转移"""
        logger.info(
            "failover_model",
            model_name=model_name,
            failed_node=failed_node_id,
        )

        # 1. 选举新的主节点
        new_primary = await self._replica_manager.elect_new_primary(model_name)

        if new_primary is None:
            logger.warning(
                "no_new_primary_available",
                model_name=model_name,
            )
            return None

        # 2. 提升新主节点
        success = await self._replica_manager.set_primary_node(model_name, new_primary)

        if success:
            logger.info(
                "model_primary_promoted",
                model_name=model_name,
                new_primary=new_primary,
            )

        # 3. 尝试在新主节点上同步模型
        if success and new_primary != failed_node_id:
            sync_result = await self._replica_manager.sync_replica(model_name, new_primary)
            if not sync_result.success:
                logger.warning(
                    "model_sync_failed_after_failover",
                    model_name=model_name,
                    target_node=new_primary,
                    error=sync_result.error_message,
                )

        return new_primary if success else None

    async def elect_new_primary(self, model_name: str) -> str | None:
        """
        为指定模型选举新的主节点

        Args:
            model_name: 模型名称

        Returns:
            新主节点 ID
        """
        return await self._replica_manager.elect_new_primary(model_name)

    async def promote_to_primary(self, node_id: str, model_name: str) -> bool:
        """
        将节点提升为主节点

        Args:
            node_id: 节点 ID
            model_name: 模型名称

        Returns:
            是否成功
        """
        return await self._replica_manager.set_primary_node(model_name, node_id)

    # ==================== 状态查询 ====================

    async def get_failover_state(self, node_id: str) -> NodeFailoverState | None:
        """
        获取节点故障转移状态

        Args:
            node_id: 节点 ID

        Returns:
            节点故障转移状态
        """
        # 从状态存储获取
        state = await self._state_store.load_node_state(node_id)

        if state:
            return state

        # 如果不存在，从 ClusterManager 构建
        node = await self._cluster_manager.get_node(node_id)
        if not node:
            return None

        # 综合健康检查
        health_result = await self._health_checker.check_node_health(node_id)

        # 构建状态
        state = NodeFailoverState(
            node_id=node_id,
            role=ReplicaRole(node.replica_role),
            state=FailoverState.NORMAL,
            health=health_result.status,
            term=0,
            last_heartbeat=node.last_heartbeat,
            failure_reasons=health_result.reasons,
            gpu_status={k: HealthStatus(v) for k, v in node.gpu_health.items()},
            model_status={k: HealthStatus(v) for k, v in node.model_health.items()},
        )

        # 保存状态
        await self._state_store.save_node_state(state)

        return state

    async def get_cluster_failover_status(self) -> dict[str, Any]:
        """
        获取集群故障转移整体状态

        Returns:
            集群状态字典
        """
        # 获取当前 Leader
        leader_id, leader_term = await self._state_store.get_leader()

        # 获取所有节点状态
        node_states = await self._state_store.get_all_node_states()

        # 获取所有副本信息
        replicas = await self._replica_manager.get_all_replicas()

        # 统计
        healthy_count = sum(
            1 for s in node_states if s.health == HealthStatus.HEALTHY
        )
        degraded_count = sum(
            1 for s in node_states if s.health == HealthStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for s in node_states if s.health == HealthStatus.UNHEALTHY
        )

        return {
            "is_leader": await self._leader_election.check_is_leader() if self._leader_election else False,
            "leader_id": leader_id,
            "leader_term": leader_term,
            "total_nodes": len(node_states),
            "healthy_nodes": healthy_count,
            "degraded_nodes": degraded_count,
            "unhealthy_nodes": unhealthy_count,
            "replicas": [r.to_dict() for r in replicas],
            "failover_in_progress": self._failover_in_progress,
        }

    # ==================== 手动故障转移 ====================

    async def manual_failover(
        self, failed_node_id: str, target_node_id: str | None = None
    ) -> bool:
        """
        手动故障转移

        Args:
            failed_node_id: 故障节点 ID
            target_node_id: 目标节点 ID（如果为 None，自动选择）

        Returns:
            是否成功
        """
        if self._failover_policy.require_manual_confirmation:
            logger.info(
                "manual_failover_confirmation_received",
                failed_node_id=failed_node_id,
                target_node_id=target_node_id,
            )

        # 获取故障节点上的模型
        failed_node = await self._cluster_manager.get_node(failed_node_id)
        if not failed_node:
            return False

        for model_name in failed_node.loaded_models:
            if target_node_id:
                await self._replica_manager.set_primary_node(model_name, target_node_id)
            else:
                await self._replica_manager.elect_new_primary(model_name)

        return True

    # ==================== 事件系统 ====================

    def on(self, event: str, handler: callable) -> None:
        """注册事件处理器"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def _emit_event(self, event: str, data: Any) -> None:
        """触发事件"""
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error("event_handler_error", event_name=event, error=str(e))

    # ==================== 检查方法 ====================

    async def check_node_health(self, node_id: str) -> HealthStatus:
        """
        检查节点健康状态

        Args:
            node_id: 节点 ID

        Returns:
            健康状态
        """
        result = await self._health_checker.check_node_health(node_id)
        return result.status

    async def is_leader(self) -> bool:
        """判断当前节点是否为 Leader"""
        if self._leader_election:
            return await self._leader_election.check_is_leader()
        return False

    def get_leader_election(self) -> LeaderElection | None:
        """获取 Leader 选举器"""
        return self._leader_election
