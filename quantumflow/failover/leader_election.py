"""Leader 选举模块

实现 Raft-like 的 Leader 选举机制，支持：
- Leader 选举
- 分布式锁（脑裂防护）
- 租约管理
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable

import structlog

from quantumflow.cluster.manager import ClusterManager
from quantumflow.failover.policy import ReplicaPolicy
from quantumflow.failover.state_store import NodeStateStore

logger = structlog.get_logger().bind(component="leader_election")


class LeaderElection:
    """
    Leader 选举器

    使用 Raft-like 算法实现 Leader 选举：
    - Term (任期) 用于区分不同的选举周期
    - 投票基于 Term 和节点健康状态
    - 多数派确保唯一 Leader
    - 分布式锁防止脑裂
    """

    def __init__(
        self,
        node_id: str,
        cluster_manager: ClusterManager,
        state_store: NodeStateStore,
        replica_policy: ReplicaPolicy | None = None,
    ):
        self._node_id = node_id
        self._cluster_manager = cluster_manager
        self._state_store = state_store
        self._replica_policy = replica_policy or ReplicaPolicy()

        # 选举状态
        self._current_term = 0
        self._voted_for: str | None = None
        self._is_leader = False
        self._is_candidate = False

        # 锁状态
        self._lock_acquired = False
        self._lock_expires_at: datetime | None = None

        # 任务
        self._election_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._running = False

        # 回调
        self._on_become_leader: Callable | None = None
        self._on_lose_leader: Callable | None = None

        # 投票计数
        self._vote_count = 0
        self._vote_received_from: set[str] = set()

        logger.info("leader_election_created", node_id=self._node_id)

    @property
    def node_id(self) -> str:
        """获取节点 ID"""
        return self._node_id

    @property
    def is_leader(self) -> bool:
        """是否当前是 Leader"""
        return self._is_leader

    @property
    def current_term(self) -> int:
        """获取当前 Term"""
        return self._current_term

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running

    def set_on_become_leader(self, callback: Callable) -> None:
        """设置成为 Leader 时的回调"""
        self._on_become_leader = callback

    def set_on_lose_leader(self, callback: Callable) -> None:
        """设置失去 Leader 地位时的回调"""
        self._on_lose_leader = callback

    async def start(self) -> None:
        """启动 Leader 选举"""
        if self._running:
            logger.warning("leader_election_already_running")
            return

        self._running = True
        self._election_task = asyncio.create_task(self._election_loop())
        logger.info("leader_election_started", node_id=self._node_id)

    async def stop(self) -> None:
        """停止 Leader 选举"""
        self._running = False

        if self._election_task:
            self._election_task.cancel()
            try:
                await self._election_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # 释放锁
        if self._lock_acquired:
            await self._state_store.release_lock("leader", self._node_id)
            self._lock_acquired = False

        logger.info("leader_election_stopped", node_id=self._node_id)

    async def _election_loop(self) -> None:
        """选举循环"""
        logger.info("election_loop_started")

        while self._running:
            try:
                # 等待选举超时
                election_timeout = self._replica_policy.election_timeout_seconds
                await asyncio.sleep(election_timeout)

                # 尝试成为 Candidate 并发起选举
                await self._start_election()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("election_loop_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("election_loop_stopped")

    async def _start_election(self) -> None:
        """发起选举"""
        if self._is_candidate:
            return

        self._is_candidate = True
        self._current_term += 1
        self._voted_for = self._node_id
        self._vote_count = 1  # 投自己一票
        self._vote_received_from = {self._node_id}

        logger.info(
            "election_started",
            node_id=self._node_id,
            term=self._current_term,
        )

        # 获取所有健康节点
        nodes = await self._cluster_manager.get_healthy_nodes()
        total_nodes = len(nodes) + 1  # 加上自己

        # 并发发送投票请求
        vote_tasks = []
        for node in nodes:
            if node.node_id != self._node_id:
                vote_tasks.append(
                    self._request_vote(node.node_id, total_nodes)
                )

        if vote_tasks:
            await asyncio.gather(*vote_tasks, return_exceptions=True)

        # 检查是否获得多数票
        if self._vote_count > total_nodes // 2:
            await self._become_leader()
        else:
            self._is_candidate = False
            logger.info(
                "election_failed",
                node_id=self._node_id,
                term=self._current_term,
                votes=self._vote_count,
                needed=total_nodes // 2 + 1,
            )

    async def _request_vote(self, target_node: str, total_nodes: int) -> None:
        """向其他节点请求投票"""
        try:
            # 模拟发送投票请求
            # 实际应该通过 WorkerClient 或 gRPC 发送
            logger.debug(
                "vote_request_sent",
                from_node=self._node_id,
                to_node=target_node,
                term=self._current_term,
            )

            # 模拟投票响应
            # 在实际实现中，应该等待目标节点的响应
            await asyncio.sleep(0.1)

            # 模拟：总是投票给请求者（如果请求者 Term 更新）
            # 实际应该检查节点健康状态、是否已投票等因素
            if self._running and self._current_term > 0:
                self._vote_count += 1
                self._vote_received_from.add(target_node)

                logger.debug(
                    "vote_received",
                    from_node=target_node,
                    term=self._current_term,
                    total_votes=self._vote_count,
                )

        except Exception as e:
            logger.error(
                "vote_request_failed",
                target_node=target_node,
                error=str(e),
            )

    async def _become_leader(self) -> None:
        """成为 Leader"""
        if self._is_leader:
            return

        logger.info(
            "becoming_leader",
            node_id=self._node_id,
            term=self._current_term,
        )

        # 尝试获取分布式锁（脑裂防护）
        lock_acquired = await self._state_store.acquire_lock(
            "leader",
            self._node_id,
            ttl_seconds=self._replica_policy.lock_ttl_seconds,
        )

        if not lock_acquired:
            logger.warning(
                "lock_acquisition_failed",
                node_id=self._node_id,
                term=self._current_term,
            )
            self._is_candidate = False
            self._is_leader = False
            return

        self._lock_acquired = True
        self._lock_expires_at = datetime.now() + timedelta(
            seconds=self._replica_policy.lock_ttl_seconds
        )
        self._is_leader = True
        self._is_candidate = False

        # 更新状态存储中的 Leader 信息
        await self._state_store.set_leader(self._node_id, self._current_term)

        # 启动心跳任务
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # 触发回调
        if self._on_become_leader:
            try:
                if asyncio.iscoroutinefunction(self._on_become_leader):
                    await self._on_become_leader(self._node_id, self._current_term)
                else:
                    self._on_become_leader(self._node_id, self._current_term)
            except Exception as e:
                logger.error("on_become_leader_callback_failed", error=str(e))

        logger.info(
            "become_leader_success",
            node_id=self._node_id,
            term=self._current_term,
        )

    async def _heartbeat_loop(self) -> None:
        """Leader 心跳循环"""
        heartbeat_interval = self._replica_policy.health_check_interval_seconds

        while self._running and self._is_leader:
            try:
                await asyncio.sleep(heartbeat_interval)

                # 续约分布式锁
                if self._lock_expires_at:
                    time_until_expiry = (self._lock_expires_at - datetime.now()).total_seconds()
                    if time_until_expiry < self._replica_policy.lock_ttl_seconds / 2:
                        # 续约
                        renewed = await self._state_store.extend_lock(
                            "leader",
                            self._node_id,
                            ttl_seconds=self._replica_policy.lock_ttl_seconds,
                        )
                        if renewed:
                            self._lock_expires_at = datetime.now() + timedelta(
                                seconds=self._replica_policy.lock_ttl_seconds
                            )
                        else:
                            logger.warning("lock_renewal_failed")
                            await self._lose_leadership()
                            return

                # 更新 Leader 信息
                await self._state_store.set_leader(self._node_id, self._current_term)

                # 检查节点健康状态
                node = await self._cluster_manager.get_node(self._node_id)
                if not node or node.status.value != "healthy":
                    logger.warning("leader_became_unhealthy")
                    await self._lose_leadership()
                    return

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("heartbeat_loop_error", error=str(e))

    async def _lose_leadership(self) -> None:
        """失去 Leader 地位"""
        if not self._is_leader:
            return

        logger.info("losing_leadership", node_id=self._node_id)

        self._is_leader = False
        self._lock_acquired = False

        # 释放锁
        await self._state_store.release_lock("leader", self._node_id)

        # 停止心跳任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # 触发回调
        if self._on_lose_leader:
            try:
                if asyncio.iscoroutinefunction(self._on_lose_leader):
                    await self._on_lose_leader(self._node_id)
                else:
                    self._on_lose_leader(self._node_id)
            except Exception as e:
                logger.error("on_lose_leader_callback_failed", error=str(e))

    async def renew_leadership(self) -> bool:
        """
        续约 Leadership

        Returns:
            是否成功续约
        """
        if not self._is_leader:
            return False

        try:
            renewed = await self._state_store.extend_lock(
                "leader",
                self._node_id,
                ttl_seconds=self._replica_policy.lock_ttl_seconds,
            )

            if renewed:
                self._lock_expires_at = datetime.now() + timedelta(
                    seconds=self._replica_policy.lock_ttl_seconds
                )
                logger.debug("leadership_renewed", node_id=self._node_id)

            return renewed
        except Exception as e:
            logger.error("leadership_renewal_failed", error=str(e))
            return False

    async def vote_for_candidate(
        self, candidate_id: str, candidate_term: int
    ) -> tuple[bool, int]:
        """
        投票给候选者

        Args:
            candidate_id: 候选者 ID
            candidate_term: 候选者的 Term

        Returns:
            (是否投票, current_term)
        """
        # 如果候选者的 Term 更旧，不投票
        if candidate_term < self._current_term:
            return False, self._current_term

        # 如果候选者的 Term 更新，切换到该 Term
        if candidate_term > self._current_term:
            await self._step_down(candidate_term)

        # 如果还没有投票给任何人，或者投给了这个候选者，投票
        if self._voted_for is None or self._voted_for == candidate_id:
            self._voted_for = candidate_id
            logger.info(
                "voted_for_candidate",
                voter=self._node_id,
                candidate=candidate_id,
                term=candidate_term,
            )
            return True, self._current_term

        return False, self._current_term

    async def _step_down(self, new_term: int) -> None:
        """切换到新的 Term"""
        if new_term <= self._current_term:
            return

        logger.info(
            "stepping_down",
            node_id=self._node_id,
            old_term=self._current_term,
            new_term=new_term,
        )

        self._current_term = new_term
        self._voted_for = None
        self._is_candidate = False

        if self._is_leader:
            await self._lose_leadership()

    async def get_current_leader(self) -> tuple[str | None, int]:
        """
        获取当前 Leader

        Returns:
            (leader_node_id, term)
        """
        return await self._state_store.get_leader()

    async def check_is_leader(self) -> bool:
        """判断当前节点是否为 Leader"""
        if not self._is_leader:
            return False

        # 检查锁是否还在
        if not self._lock_acquired:
            return False

        if self._lock_expires_at and datetime.now() > self._lock_expires_at:
            logger.warning("leader_lock_expired", node_id=self._node_id)
            await self._lose_leadership()
            return False

        return True

    # ==================== 分布式锁接口 ====================

    async def acquire_lock(
        self, resource: str, ttl_seconds: int | None = None
    ) -> bool:
        """
        获取分布式锁

        Args:
            resource: 资源名称
            ttl_seconds: 锁过期时间

        Returns:
            是否成功获取锁
        """
        if ttl_seconds is None:
            ttl_seconds = self._replica_policy.lock_ttl_seconds

        return await self._state_store.acquire_lock(
            resource, self._node_id, ttl_seconds
        )

    async def release_lock(self, resource: str) -> bool:
        """
        释放分布式锁

        Args:
            resource: 资源名称

        Returns:
            是否成功释放锁
        """
        return await self._state_store.release_lock(resource, self._node_id)

    async def extend_lock(
        self, resource: str, ttl_seconds: int | None = None
    ) -> bool:
        """
        延长锁的过期时间

        Args:
            resource: 资源名称
            ttl_seconds: 新的过期时间

        Returns:
            是否成功延长
        """
        if ttl_seconds is None:
            ttl_seconds = self._replica_policy.lock_ttl_seconds

        return await self._state_store.extend_lock(
            resource, self._node_id, ttl_seconds
        )
