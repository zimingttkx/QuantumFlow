"""ReplicaManager 扩展测试 - 提高覆盖率

补充测试以覆盖未覆盖的代码分支：
- 副本创建异常处理
- 副本同步边界情况
- 获取副本状态
- 副本完整性验证
- 副本选择逻辑
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus
from quantumflow.failover.models import ModelReplica, ReplicaRole, HealthStatus
from quantumflow.failover.policy import ReplicaPolicy
from quantumflow.failover.replica_manager import (
    ReplicaManager,
    ReplicaCreateResult,
    SyncResult,
)
from quantumflow.failover.state_store import NodeStateStore


class TestCreateReplicaEdgeCases:
    """副本创建边界情况测试"""

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
    async def test_create_replica_model_path_not_found(self, replica_manager, mock_cluster_manager):
        """测试创建副本时模型路径无法确定"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            status=NodeStatus.HEALTHY,
            loaded_models=["test-model"],  # 模型已加载
        ))

        # 模拟 _get_model_path 返回 None
        replica_manager._get_model_path = AsyncMock(return_value=None)

        result = await replica_manager.create_replica(
            model_name="test-model",
            source_node="node-1",
            target_node="node-2",
        )

        assert result.success is False
        assert "Cannot determine model path" in result.error_message

    @pytest.mark.asyncio
    async def test_create_replica_target_node_not_found(self, replica_manager, mock_cluster_manager):
        """测试创建副本时目标节点不存在"""
        def get_node_side_effect(node_id):
            if node_id == "node-1":
                return MagicMock(
                    node_id="node-1",
                    status=NodeStatus.HEALTHY,
                    loaded_models=["test-model"],
                )
            return None

        mock_cluster_manager.get_node = AsyncMock(side_effect=get_node_side_effect)
        replica_manager._get_model_path = AsyncMock(return_value="/models/test")
        replica_manager._copy_model = AsyncMock(return_value=1024)
        replica_manager._calculate_checksum = AsyncMock(return_value="abc123")
        replica_manager._update_replica_index = AsyncMock()

        result = await replica_manager.create_replica(
            model_name="test-model",
            source_node="node-1",
            target_node="nonexistent",
        )

        assert result.success is False
        assert "not found" in result.error_message

    @pytest.mark.asyncio
    async def test_create_replica_copy_failure(self, replica_manager, mock_cluster_manager):
        """测试创建副本时复制失败"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            status=NodeStatus.HEALTHY,
            loaded_models=["test-model"],
        ))
        mock_cluster_manager.add_loaded_model = AsyncMock()

        replica_manager._get_model_path = AsyncMock(return_value="/models/test")
        replica_manager._copy_model = AsyncMock(side_effect=Exception("Copy failed"))
        replica_manager._update_replica_index = AsyncMock()

        result = await replica_manager.create_replica(
            model_name="test-model",
            source_node="node-1",
            target_node="node-2",
        )

        assert result.success is False
        assert "Copy failed" in result.error_message


class TestSyncReplicaEdgeCases:
    """副本同步边界情况测试"""

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
    async def test_sync_replica_no_replica_index(self, replica_manager, mock_state_store):
        """测试同步时没有副本索引"""
        mock_state_store.load_replica_index = AsyncMock(return_value=None)

        result = await replica_manager.sync_replica("nonexistent-model", "node-2")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_sync_replica_no_primary_node(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试同步时没有主节点"""
        replica = ModelReplica(
            model_name="test-model",
            model_path="/models/test",
            primary_node=None,  # 没有主节点
            secondary_nodes=["node-2"],
            replica_count=1,
            sync_state="synced",
            last_sync_at=datetime.now() - timedelta(hours=1),
            checksum="old_checksum",
            version=1,
        )
        mock_state_store.load_replica_index = AsyncMock(return_value=replica)

        result = await replica_manager.sync_replica("test-model", "node-2")

        assert result.success is False
        assert "No primary node available" in result.error_message


class TestGetReplicaStatus:
    """获取副本状态测试"""

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
        policy = ReplicaPolicy()
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_get_replica_status_exists(self, replica_manager, mock_state_store):
        """测试获取存在的副本状态"""
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

        result = await replica_manager.get_replica_status("test-model")

        assert result is not None
        assert result.model_name == "test-model"
        assert result.primary_node == "node-1"

    @pytest.mark.asyncio
    async def test_get_replica_status_not_exists(self, replica_manager, mock_state_store):
        """测试获取不存在的副本状态"""
        mock_state_store.load_replica_index = AsyncMock(return_value=None)

        result = await replica_manager.get_replica_status("nonexistent")

        assert result is None


class TestVerifyReplicaIntegrity:
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
        policy = ReplicaPolicy()
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_verify_replica_integrity_success(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试验证副本完整性成功"""
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
            node_id="node-1",
            status=NodeStatus.HEALTHY,
            loaded_models=["test-model"],
        ))
        replica_manager._calculate_checksum = AsyncMock(return_value="abc123")

        result = await replica_manager.verify_replica_integrity("test-model", "node-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_replica_integrity_checksum_mismatch(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试验证副本完整性校验和不匹配"""
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
            node_id="node-1",
            status=NodeStatus.HEALTHY,
            loaded_models=["test-model"],
        ))
        replica_manager._calculate_checksum = AsyncMock(return_value="different_checksum")

        result = await replica_manager.verify_replica_integrity("test-model", "node-1")

        assert result is False


class TestElectNewPrimary:
    """选举新主节点测试"""

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
    async def test_elect_new_primary_no_replica(self, replica_manager, mock_state_store):
        """测试没有副本时选举"""
        mock_state_store.load_replica_index = AsyncMock(return_value=None)

        result = await replica_manager.elect_new_primary("nonexistent-model")

        assert result is None

    @pytest.mark.asyncio
    async def test_elect_new_primary_no_secondary(self, replica_manager, mock_state_store):
        """测试没有从节点时选举"""
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

        result = await replica_manager.elect_new_primary("test-model")

        assert result is None


class TestSelectReplicaForInference:
    """选择推理副本测试"""

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
        policy = ReplicaPolicy()
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_select_replica_no_replica_index(self, replica_manager, mock_state_store):
        """测试没有副本索引时选择"""
        mock_state_store.load_replica_index = AsyncMock(return_value=None)

        result = await replica_manager.select_replica_for_inference("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_select_replica_primary_healthy(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试选择健康的主节点"""
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
            node_id="node-1",
            status=NodeStatus.HEALTHY,
            model_health={},
        ))

        result = await replica_manager.select_replica_for_inference(
            "test-model",
            preferred_role=ReplicaRole.PRIMARY,
        )

        assert result == "node-1"

    @pytest.mark.asyncio
    async def test_select_replica_primary_unhealthy_fallback_secondary(self, replica_manager, mock_cluster_manager, mock_state_store):
        """测试主节点不健康时回退到从节点"""
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

        def get_node_side_effect(node_id):
            if node_id == "node-1":
                return MagicMock(
                    node_id="node-1",
                    status=NodeStatus.UNHEALTHY,
                    model_health={},
                )
            elif node_id == "node-2":
                return MagicMock(
                    node_id="node-2",
                    status=NodeStatus.HEALTHY,
                    model_health={"test-model": "healthy"},
                )
            return None

        mock_cluster_manager.get_node = AsyncMock(side_effect=get_node_side_effect)

        result = await replica_manager.select_replica_for_inference(
            "test-model",
            preferred_role=ReplicaRole.PRIMARY,
        )

        assert result == "node-2"


class TestGetAllReplicas:
    """获取所有副本测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.get_all_replica_indexes = AsyncMock(return_value=[])
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
    async def test_get_all_replicas_empty(self, replica_manager, mock_state_store):
        """测试获取所有副本为空"""
        mock_state_store.get_all_replica_indexes = AsyncMock(return_value=[])

        result = await replica_manager.get_all_replicas()

        assert result == []


class TestGetModelLocations:
    """获取模型位置测试"""

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
        policy = ReplicaPolicy()
        return ReplicaManager(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_get_model_locations_exists(self, replica_manager, mock_state_store):
        """测试获取存在的模型位置"""
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

        result = await replica_manager.get_model_locations("test-model")

        assert "node-1" in result
        assert "node-2" in result
        assert result["node-1"] == "primary"
        assert result["node-2"] == "secondary"

    @pytest.mark.asyncio
    async def test_get_model_locations_not_exists(self, replica_manager, mock_state_store):
        """测试获取不存在的模型位置"""
        mock_state_store.load_replica_index = AsyncMock(return_value=None)

        result = await replica_manager.get_model_locations("nonexistent")

        assert result == {}
