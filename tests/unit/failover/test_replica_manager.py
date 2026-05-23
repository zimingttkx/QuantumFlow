"""ReplicaManager 副本管理器测试

验证副本管理的业务逻辑正确性：
- 副本创建流程与数据一致性
- 副本删除与主节点变更
- 副本同步与增量同步判定
- 副本选择调度策略
- 副本完整性验证
- 主节点选举逻辑
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus
from quantumflow.failover.models import ModelReplica, ReplicaRole
from quantumflow.failover.policy import ReplicaPolicy
from quantumflow.failover.replica_manager import (
    ReplicaManager,
    ReplicaCreateResult,
    SyncResult,
)
from quantumflow.failover.state_store import NodeStateStore


class TestReplicaCreation:
    """副本创建测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        manager.add_loaded_model = AsyncMock()
        manager.set_replica_role = AsyncMock()
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.load_replica_index = AsyncMock(return_value=None)
        store.save_replica_index = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def replica_manager(self, mock_cluster_manager, mock_state_store):
        """创建 ReplicaManager 实例"""
        policy = ReplicaPolicy(default_replica_count=2)
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_create_replica_success(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试成功创建副本"""
        source_node = MagicMock()
        source_node.node_id = "node-1"
        source_node.loaded_models = ["qwen2.5-7b"]

        target_node = MagicMock()
        target_node.node_id = "node-2"

        mock_cluster_manager.get_node.side_effect = [source_node, target_node]

        result = await replica_manager.create_replica(
            model_name="qwen2.5-7b",
            source_node="node-1",
            target_node="node-2",
        )

        assert result.success is True
        assert result.replica is not None
        assert result.replica.primary_node == "node-1"
        assert "node-2" in result.replica.secondary_nodes
        mock_cluster_manager.add_loaded_model.assert_called_once_with("node-2", "qwen2.5-7b")

    @pytest.mark.asyncio
    async def test_create_replica_source_node_not_found(self, replica_manager, mock_cluster_manager):
        """测试源节点不存在"""
        mock_cluster_manager.get_node.return_value = None

        result = await replica_manager.create_replica(
            model_name="qwen2.5-7b",
            source_node="nonexistent",
            target_node="node-2",
        )

        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_create_replica_model_not_on_source(self, replica_manager, mock_cluster_manager):
        """测试模型不在源节点上"""
        source_node = MagicMock()
        source_node.node_id = "node-1"
        source_node.loaded_models = ["other-model"]  # 不是目标模型

        mock_cluster_manager.get_node.return_value = source_node

        result = await replica_manager.create_replica(
            model_name="qwen2.5-7b",
            source_node="node-1",
            target_node="node-2",
        )

        assert result.success is False
        assert "not loaded" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_create_replica_target_node_not_found(self, replica_manager, mock_cluster_manager):
        """测试目标节点不存在"""
        source_node = MagicMock()
        source_node.node_id = "node-1"
        source_node.loaded_models = ["qwen2.5-7b"]

        def get_node_side_effect(node_id):
            if node_id == "node-1":
                return source_node
            return None

        mock_cluster_manager.get_node.side_effect = get_node_side_effect

        result = await replica_manager.create_replica(
            model_name="qwen2.5-7b",
            source_node="node-1",
            target_node="nonexistent",
        )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_create_replica_updates_existing_index(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试创建副本时更新已存在的索引"""
        existing_replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-3"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="old_checksum",
            version=1,
        )
        mock_state_store.load_replica_index = AsyncMock(return_value=existing_replica)

        source_node = MagicMock()
        source_node.node_id = "node-1"
        source_node.loaded_models = ["qwen2.5-7b"]

        target_node = MagicMock()
        target_node.node_id = "node-2"

        def get_node_side_effect(node_id):
            if node_id == "node-1":
                return source_node
            elif node_id == "node-2":
                return target_node
            return None

        mock_cluster_manager.get_node.side_effect = get_node_side_effect

        result = await replica_manager.create_replica(
            model_name="qwen2.5-7b",
            source_node="node-1",
            target_node="node-2",
        )

        assert result.success is True
        # 验证 secondary_nodes 更新
        saved_replica = mock_state_store.save_replica_index.call_args[0][0]
        assert "node-2" in saved_replica.secondary_nodes
        assert "node-3" in saved_replica.secondary_nodes


class TestReplicaRemoval:
    """副本删除测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        manager.remove_loaded_model = AsyncMock()
        manager.set_replica_role = AsyncMock()
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
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
        )

    @pytest.mark.asyncio
    async def test_remove_replica_success(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试成功移除副本"""
        node = MagicMock()
        node.node_id = "node-2"
        mock_cluster_manager.get_node.return_value = node

        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2", "node-3"],
            replica_count=3,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        result = await replica_manager.remove_replica("qwen2.5-7b", "node-2")

        assert result is True
        mock_cluster_manager.remove_loaded_model.assert_called_once_with("node-2", "qwen2.5-7b")

    @pytest.mark.asyncio
    async def test_remove_replica_primary_node(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试移除主节点（需要重新选举）"""
        node = MagicMock()
        node.node_id = "node-2"
        mock_cluster_manager.get_node.return_value = node

        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2", "node-3"],
            replica_count=3,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        result = await replica_manager.remove_replica("qwen2.5-7b", "node-1")

        assert result is True
        # 验证选举了新主节点
        mock_state_store.save_replica_index.assert_called()

    @pytest.mark.asyncio
    async def test_remove_replica_last_node(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试移除最后一个节点（删除副本索引）"""
        node = MagicMock()
        node.node_id = "node-1"
        mock_cluster_manager.get_node.return_value = node

        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=[],  # 没有备节点
            replica_count=1,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        result = await replica_manager.remove_replica("qwen2.5-7b", "node-1")

        assert result is True
        mock_state_store.delete_replica_index.assert_called_once_with("qwen2.5-7b")

    @pytest.mark.asyncio
    async def test_remove_replica_node_not_found(self, replica_manager, mock_cluster_manager):
        """测试节点不存在"""
        mock_cluster_manager.get_node.return_value = None

        result = await replica_manager.remove_replica("qwen2.5-7b", "nonexistent")

        assert result is False


class TestReplicaSync:
    """副本同步测试"""

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
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
        )

    @pytest.mark.asyncio
    async def test_sync_replica_success(self, replica_manager, mock_state_store):
        """测试成功同步副本"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="old_checksum",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        result = await replica_manager.sync_replica("qwen2.5-7b", "node-2")

        assert result.success is True
        assert result.is_incremental is False  # checksum 会变化
        assert result.target_node == "node-2"

    @pytest.mark.asyncio
    async def test_sync_replica_no_primary(self, replica_manager, mock_state_store):
        """测试同步失败（无主节点）"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="",  # 无主节点
            secondary_nodes=["node-2"],
            replica_count=1,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        result = await replica_manager.sync_replica("qwen2.5-7b", "node-2")

        assert result.success is False
        assert "No primary" in result.error_message

    @pytest.mark.asyncio
    async def test_sync_replica_no_index(self, replica_manager, mock_state_store):
        """测试同步失败（无副本索引）"""
        mock_state_store.load_replica_index.return_value = None

        result = await replica_manager.sync_replica("nonexistent-model", "node-2")

        assert result.success is False


class TestReplicaSelection:
    """副本选择测试"""

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
        return store

    @pytest.fixture
    def replica_manager(self, mock_cluster_manager, mock_state_store):
        """创建 ReplicaManager 实例"""
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
        )

    @pytest.mark.asyncio
    async def test_select_primary_for_inference(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试选择主节点进行推理"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        primary_node = MagicMock()
        primary_node.node_id = "node-1"
        primary_node.status = NodeStatus.HEALTHY

        mock_cluster_manager.get_node.return_value = primary_node

        selected = await replica_manager.select_replica_for_inference(
            "qwen2.5-7b", ReplicaRole.PRIMARY
        )

        assert selected == "node-1"

    @pytest.mark.asyncio
    async def test_select_secondary_when_primary_unhealthy(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试主节点不健康时选择备节点"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        primary_node = MagicMock()
        primary_node.node_id = "node-1"
        primary_node.status = NodeStatus.UNHEALTHY  # 不健康

        secondary_node = MagicMock()
        secondary_node.node_id = "node-2"
        secondary_node.status = NodeStatus.HEALTHY
        secondary_node.model_health = {"qwen2.5-7b": "healthy"}

        def get_node_side_effect(node_id):
            if node_id == "node-1":
                return primary_node
            elif node_id == "node-2":
                return secondary_node
            return None

        mock_cluster_manager.get_node.side_effect = get_node_side_effect

        selected = await replica_manager.select_replica_for_inference(
            "qwen2.5-7b", ReplicaRole.PRIMARY
        )

        assert selected == "node-2"

    @pytest.mark.asyncio
    async def test_select_no_available_node(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试没有可用节点"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        unhealthy_node = MagicMock()
        unhealthy_node.node_id = "node-1"
        unhealthy_node.status = NodeStatus.UNHEALTHY

        mock_cluster_manager.get_node.return_value = unhealthy_node

        selected = await replica_manager.select_replica_for_inference(
            "qwen2.5-7b", ReplicaRole.PRIMARY
        )

        assert selected is None

    @pytest.mark.asyncio
    async def test_select_fallback_to_secondary(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试回退到备节点策略"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2", "node-3"],
            replica_count=3,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        primary_node = MagicMock()
        primary_node.node_id = "node-1"
        primary_node.status = NodeStatus.HEALTHY

        secondary_node = MagicMock()
        secondary_node.node_id = "node-2"
        secondary_node.status = NodeStatus.HEALTHY
        secondary_node.model_health = {"qwen2.5-7b": "healthy"}

        def get_node_side_effect(node_id):
            if node_id == "node-1":
                return primary_node
            elif node_id == "node-2":
                return secondary_node
            return None

        mock_cluster_manager.get_node.side_effect = get_node_side_effect

        # 首选备节点
        selected = await replica_manager.select_replica_for_inference(
            "qwen2.5-7b", ReplicaRole.SECONDARY
        )

        assert selected == "node-2"


class TestPrimaryElection:
    """主节点选举测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        manager.set_replica_role = AsyncMock()
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
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
        )

    @pytest.mark.asyncio
    async def test_elect_new_primary_success(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试成功选举新主节点"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2", "node-3"],
            replica_count=3,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        secondary2 = MagicMock()
        secondary2.node_id = "node-2"
        secondary2.status = NodeStatus.HEALTHY
        secondary2.consecutive_failures = 0

        secondary3 = MagicMock()
        secondary3.node_id = "node-3"
        secondary3.status = NodeStatus.HEALTHY
        secondary3.consecutive_failures = 2  # 失败次数多

        def get_node_side_effect(node_id):
            if node_id == "node-2":
                return secondary2
            elif node_id == "node-3":
                return secondary3
            return None

        mock_cluster_manager.get_node.side_effect = get_node_side_effect

        elected = await replica_manager.elect_new_primary("qwen2.5-7b")

        # 应该选择失败次数最少的
        assert elected == "node-2"

    @pytest.mark.asyncio
    async def test_elect_new_primary_no_secondary(self, replica_manager, mock_state_store):
        """测试没有备节点时选举失败"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=[],  # 没有备节点
            replica_count=1,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        elected = await replica_manager.elect_new_primary("qwen2.5-7b")

        assert elected is None

    @pytest.mark.asyncio
    async def test_set_primary_node_updates_roles(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试设置主节点更新角色"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        result = await replica_manager.set_primary_node("qwen2.5-7b", "node-2")

        assert result is True
        # 验证 set_replica_role 被调用
        assert mock_cluster_manager.set_replica_role.call_count == 2


class TestReplicaIntegrity:
    """副本完整性验证测试"""

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
        return store

    @pytest.fixture
    def replica_manager(self, mock_cluster_manager, mock_state_store):
        """创建 ReplicaManager 实例"""
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
        )

    @pytest.mark.asyncio
    async def test_verify_integrity_success(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试验证完整性成功"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc123",  # 预期的 checksum
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        node = MagicMock()
        node.node_id = "node-2"
        mock_cluster_manager.get_node.return_value = node

        # Mock _calculate_checksum to return matching checksum
        with patch.object(replica_manager, '_calculate_checksum', return_value='abc123'):
            result = await replica_manager.verify_replica_integrity("qwen2.5-7b", "node-2")

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_integrity_mismatch(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试验证完整性失败（checksum 不匹配）"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="original_checksum",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        node = MagicMock()
        node.node_id = "node-2"
        mock_cluster_manager.get_node.return_value = node

        # Mock _calculate_checksum to return different checksum
        with patch.object(replica_manager, '_calculate_checksum', return_value='different_checksum'):
            result = await replica_manager.verify_replica_integrity("qwen2.5-7b", "node-2")

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_integrity_no_index(self, replica_manager, mock_state_store):
        """测试无副本索引时验证失败"""
        mock_state_store.load_replica_index.return_value = None

        result = await replica_manager.verify_replica_integrity("nonexistent", "node-1")

        assert result is False


class TestModelLocations:
    """模型位置查询测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.load_replica_index = AsyncMock()
        return store

    @pytest.fixture
    def replica_manager(self, mock_cluster_manager, mock_state_store):
        """创建 ReplicaManager 实例"""
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
        )

    @pytest.mark.asyncio
    async def test_get_model_locations(self, replica_manager, mock_state_store):
        """测试获取模型位置"""
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2", "node-3"],
            replica_count=3,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )
        mock_state_store.load_replica_index.return_value = replica

        locations = await replica_manager.get_model_locations("qwen2.5-7b")

        assert locations["node-1"] == "primary"
        assert "node-2" in locations
        assert "node-3" in locations
        assert locations["node-2"] == "secondary"

    @pytest.mark.asyncio
    async def test_get_model_locations_no_replica(self, replica_manager, mock_state_store):
        """测试无副本时返回空"""
        mock_state_store.load_replica_index.return_value = None

        locations = await replica_manager.get_model_locations("nonexistent")

        assert locations == {}
