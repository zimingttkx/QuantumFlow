"""Failover 扩展测试 V3 - 覆盖剩余未覆盖代码

针对剩余未覆盖的代码分支：
- replica_manager 异常处理
- state_store get_all_replica_indexes 异常
- leader_election 剩余边界情况
- health_checker 剩余循环代码
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.failover.health_checker import HealthChecker, GPUHealthResult
from quantumflow.failover.models import ModelReplica, HealthStatus
from quantumflow.failover.policy import ReplicaPolicy
from quantumflow.failover.replica_manager import ReplicaManager
from quantumflow.failover.state_store import NodeStateStore


class TestReplicaManagerExceptionPaths:
    """ReplicaManager 异常路径测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        manager.get_healthy_nodes = AsyncMock(return_value=[])
        manager.remove_loaded_model = AsyncMock()
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.load_replica_index = AsyncMock()
        store.save_replica_index = AsyncMock(return_value=True)
        store.delete_replica_index = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def replica_manager(self, mock_cluster_manager, mock_state_store):
        """创建 ReplicaManager 实例"""
        policy = ReplicaPolicy()
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_remove_replica_exception(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试移除副本时异常"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            status=NodeStatus.HEALTHY,
        ))
        # 让 remove_loaded_model 抛出异常
        mock_cluster_manager.remove_loaded_model = AsyncMock(
            side_effect=Exception("Remove failed")
        )

        result = await replica_manager.remove_replica("test-model", "node-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_remove_replica_primary_no_secondary(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试移除主节点且没有从节点"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            status=NodeStatus.HEALTHY,
        ))

        replica = ModelReplica(
            model_name="test-model",
            model_path="/models/test",
            primary_node="node-1",
            secondary_nodes=[],  # 没有从节点
            replica_count=1,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc123",
            version=1,
        )
        mock_state_store.load_replica_index = AsyncMock(return_value=replica)

        result = await replica_manager.remove_replica("test-model", "node-1")

        assert result is True
        mock_state_store.delete_replica_index.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_redistribute_replicas_exception(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试重新分布副本时异常"""
        replica = ModelReplica(
            model_name="test-model",
            model_path="/models/test",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc123",
            version=1,
        )
        mock_state_store.load_replica_index = AsyncMock(return_value=replica)
        mock_cluster_manager.get_healthy_nodes = AsyncMock(return_value=[
            MagicMock(node_id="node-1", status=NodeStatus.HEALTHY),
            MagicMock(node_id="node-2", status=NodeStatus.HEALTHY),
            MagicMock(node_id="node-3", status=NodeStatus.HEALTHY),
            MagicMock(node_id="node-4", status=NodeStatus.HEALTHY),
        ])
        # 让 create_replica 抛出异常
        replica_manager.create_replica = AsyncMock(side_effect=Exception("Create failed"))
        # 需要让 target_count > current_replicas 才会触发 create_replica 调用
        replica_manager._replica_policy.default_replica_count = 3

        result = await replica_manager.redistribute_replicas("test-model")

        assert result is False


class TestStateStoreReplicaIndexes:
    """StateStore 副本索引测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_get_all_replica_indexes_empty(self, store_with_mock_redis):
        """测试获取所有副本索引为空"""
        store, mock_redis = store_with_mock_redis
        mock_redis.keys.return_value = []

        result = await store.get_all_replica_indexes()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_replica_indexes_with_data(self, store_with_mock_redis):
        """测试获取所有副本索引有数据"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        replica_data = {
            "model_name": "test-model",
            "model_path": "/models/test",
            "primary_node": "node-1",
            "secondary_nodes": ["node-2"],
            "replica_count": 2,
            "sync_state": "synced",
            "last_sync_at": now.isoformat(),
            "checksum": "abc123",
            "version": 1,
        }

        mock_redis.keys.return_value = ["qf:failover:replica:index:test-model"]
        mock_redis.get.return_value = json.dumps(replica_data)

        result = await store.get_all_replica_indexes()

        assert len(result) == 1
        assert result[0].model_name == "test-model"

    @pytest.mark.asyncio
    async def test_get_all_replica_indexes_exception(self, store_with_mock_redis):
        """测试获取所有副本索引异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.keys.side_effect = Exception("Redis error")

        result = await store.get_all_replica_indexes()

        assert result == []


class TestLeaderElectionRemaining:
    """LeaderElection 剩余测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        store = MagicMock(spec=NodeStateStore)
        policy = ReplicaPolicy()
        return manager, store, policy

    @pytest.fixture
    def leader_election(self, mock_dependencies):
        manager, store, policy = mock_dependencies
        return LeaderElection(
            node_id="node-1",
            cluster_manager=manager,
            state_store=store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_start_election_already_candidate(self, leader_election):
        """测试已经是候选者时跳过选举"""
        leader_election._is_candidate = True
        original_term = leader_election._current_term

        await leader_election._start_election()

        # Term 不应该增加
        assert leader_election._current_term == original_term

    @pytest.mark.asyncio
    async def test_become_leader_callback_sync(self, leader_election, mock_dependencies):
        """测试成为 Leader 时同步回调"""
        manager, store, policy = mock_dependencies

        leader_election._is_candidate = True
        leader_election._current_term = 1

        # 同步回调（不是 async）
        callback_called = False
        def sync_callback(node_id, term):
            nonlocal callback_called
            callback_called = True

        leader_election.set_on_become_leader(sync_callback)

        # 不应该抛出异常
        await leader_election._become_leader()

        assert callback_called is True

        # 清理
        leader_election._running = False
        if leader_election._heartbeat_task:
            leader_election._heartbeat_task.cancel()
            try:
                await leader_election._heartbeat_task
            except asyncio.CancelledError:
                pass


class TestHealthCheckerLoopPaths:
    """HealthChecker 循环路径测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_healthy_nodes = AsyncMock(return_value=[])
        manager.get_node = AsyncMock()
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds()
        policy = ReplicaPolicy(health_check_interval_seconds=0.05)
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_health_check_loop_exception_in_node_check(self, health_checker, mock_cluster_manager):
        """测试健康检测循环中节点检查异常"""
        healthy_node = MagicMock()
        healthy_node.node_id = "node-1"

        mock_cluster_manager.get_healthy_nodes = AsyncMock(return_value=[healthy_node])
        mock_cluster_manager.get_node = AsyncMock(side_effect=Exception("Node error"))

        health_checker._running = True
        health_checker.check_node_health = AsyncMock(return_value=MagicMock(
            status=HealthStatus.HEALTHY,
            is_healthy=MagicMock(return_value=True),
            is_degraded=MagicMock(return_value=False),
            is_unhealthy=MagicMock(return_value=False),
        ))

        task = asyncio.create_task(health_checker._health_check_loop())

        await asyncio.sleep(0.1)
        health_checker._running = False

        try:
            await task
        except asyncio.CancelledError:
            pass


class TestReplicaManagerSetPrimary:
    """设置主节点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.load_replica_index = AsyncMock()
        store.save_replica_index = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def replica_manager(self, mock_cluster_manager, mock_state_store):
        """创建 ReplicaManager 实例"""
        policy = ReplicaPolicy()
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_set_primary_node_no_replica(self, replica_manager, mock_state_store):
        """测试设置主节点但没有副本索引"""
        mock_state_store.load_replica_index = AsyncMock(return_value=None)

        result = await replica_manager.set_primary_node("test-model", "node-2")

        assert result is False

    @pytest.mark.asyncio
    async def test_set_primary_node_success(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试成功设置主节点"""
        replica = ModelReplica(
            model_name="test-model",
            model_path="/models/test",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc123",
            version=1,
        )
        mock_state_store.load_replica_index = AsyncMock(return_value=replica)
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-2",
            status=NodeStatus.HEALTHY,
        ))

        result = await replica_manager.set_primary_node("test-model", "node-2")

        assert result is True


# 导入需要的模块
import asyncio
import json
from quantumflow.failover.leader_election import LeaderElection
from quantumflow.failover.health_checker import HealthThresholds
