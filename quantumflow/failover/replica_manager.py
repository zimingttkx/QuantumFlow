"""模型副本管理器

管理模型副本的完整生命周期：
- 副本创建、删除、同步
- 副本状态追踪
- 副本选择与调度
"""

import asyncio
import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from quantumflow.cluster.manager import ClusterManager
from quantumflow.failover.models import ModelReplica, ReplicaRole
from quantumflow.failover.policy import ReplicaPolicy
from quantumflow.failover.state_store import NodeStateStore

logger = structlog.get_logger().bind(component="replica_manager")


@dataclass
class ReplicaCreateResult:
    """副本创建结果"""

    success: bool
    replica: ModelReplica | None = None
    error_message: str | None = None
    bytes_copied: int = 0
    duration_seconds: float = 0.0


@dataclass
class SyncResult:
    """同步结果"""

    success: bool
    model_name: str
    target_node: str
    source_node: str | None = None
    is_incremental: bool = False
    error_message: str | None = None
    duration_seconds: float = 0.0


class ReplicaManager:
    """
    模型副本管理器

    核心功能：
    - 副本生命周期管理（创建、删除、同步）
    - 副本状态追踪
    - 副本完整性验证
    - 副本选择与调度
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        state_store: NodeStateStore,
        replica_policy: ReplicaPolicy | None = None,
    ):
        self._cluster_manager = cluster_manager
        self._state_store = state_store
        self._replica_policy = replica_policy or ReplicaPolicy()

        # 副本任务锁（防止同一模型的并发副本操作）
        self._replica_locks: dict[str, asyncio.Lock] = {}

        # 副本状态缓存
        self._replica_cache: dict[str, ModelReplica] = {}

        logger.info("replica_manager_created", policy=self._replica_policy.to_dict())

    async def _get_replica_lock(self, model_name: str) -> asyncio.Lock:
        """获取副本操作锁"""
        if model_name not in self._replica_locks:
            self._replica_locks[model_name] = asyncio.Lock()
        return self._replica_locks[model_name]

    # ==================== 副本生命周期管理 ====================

    async def create_replica(
        self,
        model_name: str,
        source_node: str,
        target_node: str,
        model_path: str | None = None,
    ) -> ReplicaCreateResult:
        """
        创建模型副本

        从源节点复制模型到目标节点

        Args:
            model_name: 模型名称
            source_node: 源节点 ID
            target_node: 目标节点 ID
            model_path: 模型路径（如果为 None，从节点信息获取）

        Returns:
            副本创建结果
        """
        start_time = datetime.now()

        async with await self._get_replica_lock(model_name):
            try:
                # 验证源节点存在且有模型
                source_node_info = await self._cluster_manager.get_node(source_node)
                if not source_node_info:
                    return ReplicaCreateResult(
                        success=False,
                        error_message=f"Source node {source_node} not found",
                    )

                if model_name not in source_node_info.loaded_models:
                    return ReplicaCreateResult(
                        success=False,
                        error_message=f"Model {model_name} not loaded on source node {source_node}",
                    )

                # 获取模型路径
                if model_path is None:
                    model_path = await self._get_model_path(source_node, model_name)
                    if model_path is None:
                        return ReplicaCreateResult(
                            success=False,
                            error_message=f"Cannot determine model path for {model_name}",
                        )

                # 检查目标节点
                target_node_info = await self._cluster_manager.get_node(target_node)
                if not target_node_info:
                    return ReplicaCreateResult(
                        success=False,
                        error_message=f"Target node {target_node} not found",
                    )

                # 执行复制（模拟，实际需要通过 WorkerClient 传输）
                bytes_copied = await self._copy_model(
                    model_path, target_node, model_name
                )

                # 更新目标节点信息
                await self._cluster_manager.add_loaded_model(target_node, model_name)

                # 计算 checksum
                checksum = await self._calculate_checksum(model_path)

                # 更新/创建副本索引
                replica = await self._update_replica_index(
                    model_name=model_name,
                    model_path=model_path,
                    primary_node=source_node,
                    secondary_node=target_node,
                    checksum=checksum,
                )

                duration = (datetime.now() - start_time).total_seconds()

                logger.info(
                    "replica_created",
                    model_name=model_name,
                    source_node=source_node,
                    target_node=target_node,
                    bytes_copied=bytes_copied,
                    duration=duration,
                )

                return ReplicaCreateResult(
                    success=True,
                    replica=replica,
                    bytes_copied=bytes_copied,
                    duration_seconds=duration,
                )

            except Exception as e:
                logger.error(
                    "replica_create_failed",
                    model_name=model_name,
                    source_node=source_node,
                    target_node=target_node,
                    error=str(e),
                )
                return ReplicaCreateResult(
                    success=False,
                    error_message=str(e),
                )

    async def remove_replica(self, model_name: str, node_id: str) -> bool:
        """
        移除模型副本

        Args:
            model_name: 模型名称
            node_id: 节点 ID

        Returns:
            是否成功
        """
        async with await self._get_replica_lock(model_name):
            try:
                # 验证节点存在
                node_info = await self._cluster_manager.get_node(node_id)
                if not node_info:
                    logger.warning(
                        "remove_replica_node_not_found",
                        model_name=model_name,
                        node_id=node_id,
                    )
                    return False

                # 移除节点上的模型
                await self._cluster_manager.remove_loaded_model(node_id, model_name)

                # 如果是主节点，需要先选举新的主节点
                replica_index = await self._state_store.load_replica_index(model_name)
                if replica_index and replica_index.primary_node == node_id:
                    # 找到备节点提升为主节点
                    secondary_nodes = [
                        n for n in replica_index.secondary_nodes if n != node_id
                    ]
                    if secondary_nodes:
                        await self.set_primary_node(model_name, secondary_nodes[0])
                    else:
                        # 没有备节点，删除副本索引
                        await self._state_store.delete_replica_index(model_name)
                        logger.info(
                            "replica_index_deleted_no_nodes",
                            model_name=model_name,
                        )
                        return True

                # 从 secondary_nodes 列表移除
                if replica_index:
                    replica_index.secondary_nodes = [
                        n for n in replica_index.secondary_nodes if n != node_id
                    ]
                    replica_index.replica_count = len(replica_index.secondary_nodes) + (
                        1 if replica_index.primary_node else 0
                    )
                    await self._state_store.save_replica_index(replica_index)

                logger.info(
                    "replica_removed",
                    model_name=model_name,
                    node_id=node_id,
                )
                return True

            except Exception as e:
                logger.error(
                    "replica_remove_failed",
                    model_name=model_name,
                    node_id=node_id,
                    error=str(e),
                )
                return False

    async def sync_replica(self, model_name: str, node_id: str) -> SyncResult:
        """
        同步模型副本（增量同步）

        Args:
            model_name: 模型名称
            node_id: 目标节点 ID

        Returns:
            同步结果
        """
        start_time = datetime.now()

        async with await self._get_replica_lock(model_name):
            try:
                # 获取副本索引
                replica_index = await self._state_store.load_replica_index(model_name)
                if not replica_index:
                    return SyncResult(
                        success=False,
                        model_name=model_name,
                        target_node=node_id,
                        error_message=f"No replica index for {model_name}",
                    )

                # 获取源节点（主节点）
                source_node = replica_index.primary_node
                if not source_node:
                    return SyncResult(
                        success=False,
                        model_name=model_name,
                        target_node=node_id,
                        error_message="No primary node available",
                    )

                # 获取模型路径
                model_path = await self._get_model_path(source_node, model_name)
                if model_path is None:
                    return SyncResult(
                        success=False,
                        model_name=model_name,
                        target_node=node_id,
                        error_message=f"Cannot determine model path",
                    )

                # 计算 checksum 并比较
                source_checksum = await self._calculate_checksum(model_path)
                is_incremental = source_checksum == replica_index.checksum

                # 执行同步
                await self._copy_model(model_path, node_id, model_name)

                # 更新副本索引
                replica_index.checksum = source_checksum
                replica_index.last_sync_at = datetime.now()
                replica_index.sync_state = "synced"
                await self._state_store.save_replica_index(replica_index)

                duration = (datetime.now() - start_time).total_seconds()

                logger.info(
                    "replica_synced",
                    model_name=model_name,
                    target_node=node_id,
                    source_node=source_node,
                    is_incremental=is_incremental,
                    duration=duration,
                )

                return SyncResult(
                    success=True,
                    model_name=model_name,
                    target_node=node_id,
                    source_node=source_node,
                    is_incremental=is_incremental,
                    duration_seconds=duration,
                )

            except Exception as e:
                logger.error(
                    "replica_sync_failed",
                    model_name=model_name,
                    node_id=node_id,
                    error=str(e),
                )
                return SyncResult(
                    success=False,
                    model_name=model_name,
                    target_node=node_id,
                    error_message=str(e),
                )

    # ==================== 副本状态管理 ====================

    async def get_replica_status(self, model_name: str) -> ModelReplica | None:
        """
        获取副本状态

        Args:
            model_name: 模型名称

        Returns:
            副本信息，不存在返回 None
        """
        # 先检查缓存
        if model_name in self._replica_cache:
            return self._replica_cache[model_name]

        # 从状态存储加载
        replica = await self._state_store.load_replica_index(model_name)
        if replica:
            self._replica_cache[model_name] = replica

        return replica

    async def verify_replica_integrity(
        self, model_name: str, node_id: str
    ) -> bool:
        """
        验证副本完整性（校验和）

        Args:
            model_name: 模型名称
            node_id: 节点 ID

        Returns:
            是否完整
        """
        try:
            # 获取副本索引
            replica_index = await self._state_store.load_replica_index(model_name)
            if not replica_index:
                return False

            # 获取节点上的模型路径
            model_path = await self._get_model_path(node_id, model_name)
            if model_path is None:
                return False

            # 计算 checksum
            checksum = await self._calculate_checksum(model_path)

            # 比较
            is_intact = checksum == replica_index.checksum

            if not is_intact:
                logger.warning(
                    "replica_integrity_failed",
                    model_name=model_name,
                    node_id=node_id,
                    expected_checksum=replica_index.checksum,
                    actual_checksum=checksum,
                )

            return is_intact

        except Exception as e:
            logger.error(
                "replica_integrity_check_failed",
                model_name=model_name,
                node_id=node_id,
                error=str(e),
            )
            return False

    # ==================== 副本选择与调度 ====================

    async def select_replica_for_inference(
        self,
        model_name: str,
        preferred_role: ReplicaRole = ReplicaRole.PRIMARY,
    ) -> str | None:
        """
        选择推理用副本节点

        Args:
            model_name: 模型名称
            preferred_role: 优先角色

        Returns:
            节点 ID，不可用返回 None
        """
        try:
            replica_index = await self._state_store.load_replica_index(model_name)
            if not replica_index:
                return None

            # 优先选择主节点
            if preferred_role == ReplicaRole.PRIMARY and replica_index.primary_node:
                node = await self._cluster_manager.get_node(replica_index.primary_node)
                if node and node.status.value == "healthy":
                    return replica_index.primary_node

            # 选择备节点
            for secondary_node in replica_index.secondary_nodes:
                node = await self._cluster_manager.get_node(secondary_node)
                if node and node.status.value == "healthy":
                    # 检查模型健康状态
                    model_health = node.model_health.get(model_name, "healthy")
                    if model_health in ("healthy", "degraded"):
                        return secondary_node

            # 如果首选不可用，尝试其他角色
            if preferred_role == ReplicaRole.PRIMARY:
                return await self.select_replica_for_inference(
                    model_name, ReplicaRole.SECONDARY
                )

            return None

        except Exception as e:
            logger.error(
                "select_replica_failed",
                model_name=model_name,
                error=str(e),
            )
            return None

    async def redistribute_replicas(self, model_name: str) -> bool:
        """
        重新分布副本（负载均衡时）

        Args:
            model_name: 模型名称

        Returns:
            是否成功
        """
        try:
            replica_index = await self._state_store.load_replica_index(model_name)
            if not replica_index:
                return False

            # 获取所有健康节点
            healthy_nodes = await self._cluster_manager.get_healthy_nodes()
            all_node_ids = {n.node_id for n in healthy_nodes}

            # 计算需要添加的副本数
            current_replicas = {replica_index.primary_node} | set(
                replica_index.secondary_nodes
            )
            target_count = min(
                self._replica_policy.default_replica_count,
                len(all_node_ids),
            )

            # 需要添加的节点
            nodes_to_add = all_node_ids - current_replicas
            nodes_to_add = list(nodes_to_add)[
                : target_count - len(current_replicas) + 1
            ]

            # 添加副本
            for target_node in nodes_to_add:
                if replica_index.primary_node:
                    await self.create_replica(
                        model_name,
                        source_node=replica_index.primary_node,
                        target_node=target_node,
                    )

            logger.info(
                "replicas_redistributed",
                model_name=model_name,
                nodes_added=len(nodes_to_add),
            )
            return True

        except Exception as e:
            logger.error(
                "replica_redistribute_failed",
                model_name=model_name,
                error=str(e),
            )
            return False

    # ==================== 主节点管理 ====================

    async def set_primary_node(self, model_name: str, node_id: str) -> bool:
        """
        设置模型的主节点

        Args:
            model_name: 模型名称
            node_id: 节点 ID

        Returns:
            是否成功
        """
        try:
            replica_index = await self._state_store.load_replica_index(model_name)
            if not replica_index:
                return False

            # 更新主节点
            old_primary = replica_index.primary_node

            # 如果旧主节点存在且不是故障节点，加入备节点列表
            if old_primary and old_primary != node_id:
                if old_primary not in replica_index.secondary_nodes:
                    replica_index.secondary_nodes.append(old_primary)

            # 如果新主节点在备节点列表中，移除
            if node_id in replica_index.secondary_nodes:
                replica_index.secondary_nodes.remove(node_id)

            replica_index.primary_node = node_id

            # 更新 ClusterManager 中的节点角色
            await self._cluster_manager.set_replica_role(node_id, "primary")
            for secondary in replica_index.secondary_nodes:
                await self._cluster_manager.set_replica_role(secondary, "secondary")

            await self._state_store.save_replica_index(replica_index)

            logger.info(
                "primary_node_updated",
                model_name=model_name,
                old_primary=old_primary,
                new_primary=node_id,
            )
            return True

        except Exception as e:
            logger.error(
                "set_primary_node_failed",
                model_name=model_name,
                node_id=node_id,
                error=str(e),
            )
            return False

    async def elect_new_primary(self, model_name: str) -> str | None:
        """
        为模型选举新的主节点

        Args:
            model_name: 模型名称

        Returns:
            新主节点 ID，没有可用节点返回 None
        """
        try:
            replica_index = await self._state_store.load_replica_index(model_name)
            if not replica_index:
                return None

            # 优先选择健康状况最好的备节点
            best_node = None
            best_health = None

            for node_id in replica_index.secondary_nodes:
                node = await self._cluster_manager.get_node(node_id)
                if node and node.status.value == "healthy":
                    # 选择连续失败次数最少的节点
                    if best_health is None or node.consecutive_failures < best_health:
                        best_node = node_id
                        best_health = node.consecutive_failures

            if best_node:
                await self.set_primary_node(model_name, best_node)
                return best_node

            return None

        except Exception as e:
            logger.error(
                "elect_new_primary_failed",
                model_name=model_name,
                error=str(e),
            )
            return None

    # ==================== 辅助方法 ====================

    async def _update_replica_index(
        self,
        model_name: str,
        model_path: str,
        primary_node: str,
        secondary_node: str | None = None,
        checksum: str = "",
    ) -> ModelReplica:
        """更新副本索引"""
        existing = await self._state_store.load_replica_index(model_name)

        if existing:
            # 更新现有索引
            if primary_node and primary_node != existing.primary_node:
                # 主节点变更，原主节点降为备节点
                if existing.primary_node and existing.primary_node != primary_node:
                    if existing.primary_node not in existing.secondary_nodes:
                        existing.secondary_nodes.append(existing.primary_node)
                existing.primary_node = primary_node

            if secondary_node and secondary_node not in existing.secondary_nodes:
                existing.secondary_nodes.append(secondary_node)

            existing.replica_count = len(existing.secondary_nodes) + (
                1 if existing.primary_node else 0
            )
            existing.checksum = checksum
            existing.last_sync_at = datetime.now()
            existing.sync_state = "synced"

            await self._state_store.save_replica_index(existing)
            self._replica_cache[model_name] = existing
            return existing
        else:
            # 创建新索引
            secondary_nodes = [secondary_node] if secondary_node else []
            replica = ModelReplica(
                model_name=model_name,
                model_path=model_path,
                primary_node=primary_node,
                secondary_nodes=secondary_nodes,
                replica_count=len(secondary_nodes) + 1,
                sync_state="synced",
                last_sync_at=datetime.now(),
                checksum=checksum,
                version=1,
            )

            await self._state_store.save_replica_index(replica)
            self._replica_cache[model_name] = replica
            return replica

    async def _get_model_path(self, node_id: str, model_name: str) -> str | None:
        """获取节点上模型的路径"""
        # 实际应该通过 WorkerClient 查询
        # 这里简化处理，返回模型名称作为路径
        # TODO: 实现从 WorkerClient 获取模型路径
        return f"/models/{model_name}"

    async def _copy_model(
        self, model_path: str, target_node: str, model_name: str
    ) -> int:
        """
        复制模型到目标节点

        实际实现需要通过 WorkerClient 传输模型文件
        这里简化处理，模拟复制操作

        Returns:
            复制的字节数
        """
        # 模拟复制
        await asyncio.sleep(0.1)

        # 估算模型大小（实际应该从节点获取）
        estimated_size = 7 * 1024 * 1024 * 1024  # 7GB

        logger.debug(
            "model_copied",
            model_path=model_path,
            target_node=target_node,
            model_name=model_name,
        )

        return estimated_size

    async def _calculate_checksum(self, model_path: str) -> str:
        """
        计算模型目录的 checksum

        用于验证副本完整性

        Returns:
            SHA256 checksum
        """
        # 简化处理：使用模型路径作为 checksum 的基础
        # 实际应该计算所有模型文件的 checksum
        hash_obj = hashlib.sha256()
        hash_obj.update(model_path.encode())
        hash_obj.update(str(datetime.now()).encode())  # 加入时间戳
        return hash_obj.hexdigest()[:16]

    async def get_all_replicas(self) -> list[ModelReplica]:
        """获取所有模型副本信息"""
        return await self._state_store.get_all_replica_indexes()

    async def get_model_locations(self, model_name: str) -> dict[str, str]:
        """
        获取模型所在的所有节点

        Returns:
            {node_id: role} 字典
        """
        locations = {}
        replica = await self.get_replica_status(model_name)

        if replica:
            if replica.primary_node:
                locations[replica.primary_node] = "primary"
            for secondary in replica.secondary_nodes:
                locations[secondary] = "secondary"

        return locations
