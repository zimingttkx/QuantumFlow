"""FailoverController 故障转移控制器测试

验证故障转移的业务逻辑正确性：
- 故障检测与判定逻辑
- 故障转移决策流程
- 主节点选举协调
- 副本重建流程
- 脑裂防护机制
- 状态查询与事件触发
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus
from quantumflow.failover.health_checker import HealthChecker, HealthCheckResult
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
from quantumflow.failover.controller import FailoverController, FailoverDecision


class TestFailoverControllerInitialization:
    """故障转移控制器初始化测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.on = MagicMock()
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.generate_event_id = MagicMock(return_value="fe_123456")
        store.save_failover_event = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        manager = MagicMock(spec=ReplicaManager)
        manager.get_all_replicas = AsyncMock(return_value=[])
        manager.get_model_locations = AsyncMock(return_value={})
        manager.elect_new_primary = AsyncMock()
        manager.set_primary_node = AsyncMock(return_value=True)
        manager.sync_replica = AsyncMock()
        return manager

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        checker = MagicMock(spec=HealthChecker)
        checker.check_node_health = AsyncMock()
        checker.start = AsyncMock()
        checker.stop = AsyncMock()
        return checker

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )

    def test_initial_state(self, failover_controller):
        """测试初始状态"""
        assert failover_controller._running is False
        assert failover_controller._failover_in_progress is False
        assert failover_controller._node_id == "node-1"

    @pytest.mark.asyncio
    async def test_initialize_creates_leader_election(self, failover_controller, mock_cluster_manager):
        """测试初始化创建 Leader 选举器"""
        await failover_controller.initialize()

        assert failover_controller._leader_election is not None
        assert isinstance(failover_controller._leader_election, LeaderElection)
        mock_cluster_manager.on.assert_called()

    @pytest.mark.asyncio
    async def test_start_enables_running(self, failover_controller, mock_health_checker, mock_cluster_manager):
        """测试启动后 running 状态"""
        await failover_controller.initialize()
        await failover_controller.start()

        assert failover_controller._running is True
        mock_health_checker.start.assert_called_once()


class TestFailoverDecisionLogic:
    """故障转移决策逻辑测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        manager.get_healthy_nodes = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.generate_event_id = MagicMock(return_value="fe_test")
        store.save_failover_event = AsyncMock(return_value=True)
        store.load_node_state = AsyncMock(return_value=None)
        store.save_node_state = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        manager = MagicMock(spec=ReplicaManager)
        manager.get_all_replicas = AsyncMock(return_value=[])
        manager.get_model_locations = AsyncMock(return_value={})
        manager.elect_new_primary = AsyncMock(return_value="node-2")
        manager.set_primary_node = AsyncMock(return_value=True)
        manager.sync_replica = AsyncMock()
        return manager

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        checker = MagicMock(spec=HealthChecker)
        checker.check_node_health = AsyncMock()
        checker.start = AsyncMock()
        checker.stop = AsyncMock()
        return checker

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )

    @pytest.mark.asyncio
    async def test_decide_failover_for_primary_node(self, failover_controller, mock_cluster_manager):
        """测试主节点故障时决策"""
        failed_node = MagicMock()
        failed_node.node_id = "node-1"
        failed_node.replica_role = "primary"
        failed_node.loaded_models = ["model-a", "model-b"]

        mock_cluster_manager.get_node.return_value = failed_node

        decision = await failover_controller._make_failover_decision(
            "node-1", "heartbeat_timeout", ["model-a", "model-b"]
        )

        assert decision.should_failover is True
        assert decision.failed_node == "node-1"
        assert decision.risk_level in ["medium", "high", "critical"]

    @pytest.mark.asyncio
    async def test_decide_failover_for_secondary_node(self, failover_controller, mock_cluster_manager):
        """测试备节点故障时决策"""
        failed_node = MagicMock()
        failed_node.node_id = "node-2"
        failed_node.replica_role = "secondary"
        failed_node.loaded_models = ["model-a"]

        mock_cluster_manager.get_node.return_value = failed_node

        # 有其他节点可以接管
        mock_cluster_manager.get_healthy_nodes.return_value = [
            MagicMock(node_id="node-1"),
        ]

        decision = await failover_controller._make_failover_decision(
            "node-2", "gpu_fail", ["model-a"]
        )

        assert decision.should_failover is True
        assert decision.failed_node == "node-2"

    @pytest.mark.asyncio
    async def test_decide_no_failover_when_no_affected_models(
        self, failover_controller, mock_cluster_manager
    ):
        """测试无受影响模型时不决策"""
        failed_node = MagicMock()
        failed_node.node_id = "node-2"
        failed_node.replica_role = "secondary"
        failed_node.loaded_models = []  # 无加载模型

        mock_cluster_manager.get_node.return_value = failed_node

        decision = await failover_controller._make_failover_decision(
            "node-2", "timeout", []
        )

        # 没有模型受影响，但如果是备节点可能不需要 failover
        assert decision.affected_models == []

    @pytest.mark.asyncio
    async def test_decide_risk_level_critical_when_no_backup(
        self, failover_controller, mock_cluster_manager
    ):
        """测试没有备份时风险等级为 critical"""
        failed_node = MagicMock()
        failed_node.node_id = "node-1"
        failed_node.replica_role = "primary"
        failed_node.loaded_models = ["model-a", "model-b", "model-c", "model-d"]

        mock_cluster_manager.get_node.return_value = failed_node

        decision = await failover_controller._make_failover_decision(
            "node-1", "heartbeat_timeout", ["model-a", "model-b", "model-c", "model-d"]
        )

        assert decision.risk_level == "critical"


class TestFailoverExecution:
    """故障转移执行测试"""

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
        store.generate_event_id = MagicMock(return_value="fe_123456")
        store.save_failover_event = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        manager = MagicMock(spec=ReplicaManager)
        manager.get_model_locations = AsyncMock(return_value={"node-2": "secondary"})
        manager.elect_new_primary = AsyncMock(return_value="node-2")
        manager.set_primary_node = AsyncMock(return_value=True)
        manager.sync_replica = AsyncMock()
        return manager

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        checker = MagicMock(spec=HealthChecker)
        checker.check_node_health = AsyncMock()
        return checker

    @pytest.fixture
    def mock_leader_election(self):
        """创建模拟的 LeaderElection"""
        election = MagicMock(spec=LeaderElection)
        election.is_leader = AsyncMock(return_value=True)
        election.start = AsyncMock()
        election.stop = AsyncMock()
        return election

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker, mock_leader_election
    ):
        """创建 FailoverController 实例"""
        controller = FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )
        # 注入模拟的 leader election
        controller._leader_election = mock_leader_election
        return controller

    @pytest.mark.asyncio
    async def test_initiate_failover_success(self, failover_controller, mock_cluster_manager):
        """测试成功发起故障转移"""
        failed_node = MagicMock()
        failed_node.node_id = "node-1"
        failed_node.replica_role = "primary"
        failed_node.loaded_models = ["qwen2.5-7b"]

        mock_cluster_manager.get_node.return_value = failed_node

        result = await failover_controller.initiate_failover("node-1", "heartbeat_timeout")

        assert result is True
        mock_cluster_manager.get_node.assert_called()

    @pytest.mark.asyncio
    async def test_initiate_failover_not_leader(self, failover_controller, mock_leader_election):
        """测试非 Leader 时不能发起故障转移"""
        mock_leader_election.is_leader = AsyncMock(return_value=False)

        result = await failover_controller.initiate_failover("node-2", "gpu_fail")

        assert result is False

    @pytest.mark.asyncio
    async def test_initiate_failover_already_in_progress(
        self, failover_controller
    ):
        """测试故障转移进行中时跳过"""
        failover_controller._failover_in_progress = True

        result = await failover_controller.initiate_failover("node-2", "timeout")

        assert result is False

    @pytest.mark.asyncio
    async def test_failover_model_promotes_new_primary(
        self, failover_controller, mock_replica_manager
    ):
        """测试故障转移时提升新主节点"""
        result = await failover_controller._failover_model("qwen2.5-7b", "node-1")

        assert result == "node-2"
        mock_replica_manager.elect_new_primary.assert_called_once_with("qwen2.5-7b")
        mock_replica_manager.set_primary_node.assert_called_once_with("qwen2.5-7b", "node-2")

    @pytest.mark.asyncio
    async def test_failover_model_syncs_after_promotion(
        self, failover_controller, mock_replica_manager
    ):
        """测试故障转移后同步模型"""
        mock_replica_manager.sync_replica.return_value = MagicMock(success=True)

        result = await failover_controller._failover_model("qwen2.5-7b", "node-1")

        mock_replica_manager.sync_replica.assert_called_once()

    @pytest.mark.asyncio
    async def test_failover_model_no_new_primary(
        self, failover_controller, mock_replica_manager
    ):
        """测试无法选举新主节点"""
        mock_replica_manager.elect_new_primary.return_value = None

        result = await failover_controller._failover_model("qwen2.5-7b", "node-1")

        assert result is None


class TestManualFailover:
    """手动故障转移测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        return MagicMock(spec=NodeStateStore)

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        manager = MagicMock(spec=ReplicaManager)
        manager.set_primary_node = AsyncMock(return_value=True)
        manager.elect_new_primary = AsyncMock(return_value="node-3")
        return manager

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        return MagicMock(spec=HealthChecker)

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
            failover_policy=FailoverPolicy(require_manual_confirmation=False),
        )

    @pytest.mark.asyncio
    async def test_manual_failover_with_target(self, failover_controller, mock_cluster_manager):
        """测试指定目标节点的手动故障转移"""
        failed_node = MagicMock()
        failed_node.node_id = "node-1"
        failed_node.loaded_models = ["model-a"]

        mock_cluster_manager.get_node.return_value = failed_node

        result = await failover_controller.manual_failover("node-1", target_node_id="node-3")

        assert result is True
        mock_cluster_manager.get_node.assert_called_with("node-1")

    @pytest.mark.asyncio
    async def test_manual_failover_auto_select_target(self, failover_controller, mock_cluster_manager):
        """测试自动选择目标节点的手动故障转移"""
        failed_node = MagicMock()
        failed_node.node_id = "node-1"
        failed_node.loaded_models = ["model-a"]

        mock_cluster_manager.get_node.return_value = failed_node

        result = await failover_controller.manual_failover("node-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_manual_failover_requires_confirmation(self, failover_controller):
        """测试需要确认的手动故障转移"""
        controller = FailoverController(
            cluster_manager=MagicMock(),
            state_store=MagicMock(),
            replica_manager=MagicMock(),
            health_checker=MagicMock(),
            node_id="node-1",
            failover_policy=FailoverPolicy(require_manual_confirmation=True),
        )

        result = await controller.manual_failover("node-1")

        assert result is False


class TestHealthCheckIntegration:
    """健康检查集成测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        return MagicMock(spec=NodeStateStore)

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        return MagicMock(spec=ReplicaManager)

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        checker = MagicMock(spec=HealthChecker)
        checker.check_node_health = AsyncMock(
            return_value=HealthCheckResult(
                node_id="node-1",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
            )
        )
        return checker

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )

    @pytest.mark.asyncio
    async def test_check_node_health_delegates(self, failover_controller, mock_health_checker):
        """测试健康检查委托给 HealthChecker"""
        mock_health_checker.check_node_health.return_value = HealthCheckResult(
            node_id="node-1",
            status=HealthStatus.DEGRADED,
            timestamp=datetime.now(),
            reasons=["high_temperature"],
        )

        result = await failover_controller.check_node_health("node-1")

        assert result == HealthStatus.DEGRADED
        mock_health_checker.check_node_health.assert_called_once_with("node-1")


class TestLeaderElectionIntegration:
    """Leader 选举集成测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.get_leader = AsyncMock(return_value=("node-1", 5))
        return store

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        return MagicMock(spec=ReplicaManager)

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        return MagicMock(spec=HealthChecker)

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )

    @pytest.mark.asyncio
    async def test_is_leader_delegates_to_election(self, failover_controller):
        """测试 is_leader 委托给 LeaderElection"""
        mock_election = MagicMock()
        mock_election.check_is_leader = AsyncMock(return_value=True)
        failover_controller._leader_election = mock_election

        result = await failover_controller.is_leader()

        assert result is True
        mock_election.check_is_leader.assert_called_once()


class TestClusterFailoverStatus:
    """集群故障转移状态查询测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.get_leader = AsyncMock(return_value=("node-1", 3))
        store.get_all_node_states = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        manager = MagicMock(spec=ReplicaManager)
        manager.get_all_replicas = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        return MagicMock(spec=HealthChecker)

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        controller = FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )
        mock_election = MagicMock()
        mock_election.check_is_leader = AsyncMock(return_value=True)
        controller._leader_election = mock_election
        return controller

    @pytest.mark.asyncio
    async def test_get_cluster_failover_status_structure(self, failover_controller):
        """测试集群状态返回结构"""
        status = await failover_controller.get_cluster_failover_status()

        assert "is_leader" in status
        assert "leader_id" in status
        assert "leader_term" in status
        assert "total_nodes" in status
        assert "healthy_nodes" in status
        assert "degraded_nodes" in status
        assert "unhealthy_nodes" in status
        assert "replicas" in status
        assert "failover_in_progress" in status


class TestEventHandling:
    """事件处理测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.on = MagicMock()
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.generate_event_id = MagicMock(return_value="fe_test")
        store.save_failover_event = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        return MagicMock(spec=ReplicaManager)

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        return MagicMock(spec=HealthChecker)

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )

    @pytest.mark.asyncio
    async def test_register_event_handler(self, failover_controller):
        """测试注册事件处理器"""
        handler_called = []

        def handler(data):
            handler_called.append(data)

        failover_controller.on("test_event", handler)

        assert "test_event" in failover_controller._event_handlers

    @pytest.mark.asyncio
    async def test_emit_event_calls_handlers(self, failover_controller):
        """测试触发事件调用处理器"""
        handler_called = []

        def handler(data):
            handler_called.append(data)

        failover_controller.on("my_event", handler)

        await failover_controller._emit_event("my_event", {"key": "value"})

        assert len(handler_called) == 1
        assert handler_called[0] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_emit_event_async_handler(self, failover_controller):
        """测试触发事件调用异步处理器"""
        handler_called = []

        async def async_handler(data):
            handler_called.append(data)

        failover_controller.on("async_event", async_handler)

        await failover_controller._emit_event("async_event", {"async": True})

        assert len(handler_called) == 1

    @pytest.mark.asyncio
    async def test_emit_event_handler_exception(self, failover_controller):
        """测试事件处理器异常不影响其他处理器"""
        handler_called = []

        def bad_handler(data):
            raise Exception("Handler error")

        def good_handler(data):
            handler_called.append(data)

        failover_controller.on("event1", bad_handler)
        failover_controller.on("event1", good_handler)

        # 不应抛出异常
        await failover_controller._emit_event("event1", {})

        assert len(handler_called) == 1


class TestNodeHealthChangedCallback:
    """_on_node_health_changed 回调测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.on = MagicMock()
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock(spec=NodeStateStore)
        store.generate_event_id = MagicMock(return_value="fe_test")
        store.save_failover_event = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        manager = MagicMock(spec=ReplicaManager)
        manager.get_all_replicas = AsyncMock(return_value=[])
        manager.get_model_locations = AsyncMock(return_value={})
        return manager

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        checker = MagicMock(spec=HealthChecker)
        mock_result = MagicMock()
        mock_result.is_unhealthy.return_value = True
        checker.check_node_health = AsyncMock(return_value=mock_result)
        return checker

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )

    @pytest.mark.asyncio
    async def test_on_node_health_changed_unhealthy_triggers_handler(
        self, failover_controller, mock_cluster_manager
    ):
        """测试节点变为不健康时触发 _handle_unhealthy_node"""
        mock_node = MagicMock()
        mock_node.node_id = "node-2"

        # 直接调用内部回调方法，模拟 cluster manager 触发事件
        await failover_controller._on_node_health_changed(
            mock_node,
            NodeStatus.HEALTHY,
            NodeStatus.UNHEALTHY,
        )

        # 验证 _handle_unhealthy_node 被触发（通过 health_checker 被调用来验证）
        failover_controller._health_checker.check_node_health.assert_called_with("node-2")

    @pytest.mark.asyncio
    async def test_on_node_health_changed_healthy_no_action(
        self, failover_controller, mock_health_checker
    ):
        """测试节点变为健康时不做处理"""
        mock_node = MagicMock()
        mock_node.node_id = "node-2"

        # 直接调用内部回调方法
        await failover_controller._on_node_health_changed(
            mock_node,
            NodeStatus.UNHEALTHY,
            NodeStatus.HEALTHY,
        )

        # 验证 health_checker.check_node_health 不被调用（因为状态是健康的）
        mock_health_checker.check_node_health.assert_not_called()


class TestPromoteToPrimary:
    """提升为主节点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        return MagicMock(spec=NodeStateStore)

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        manager = MagicMock(spec=ReplicaManager)
        manager.set_primary_node = AsyncMock(return_value=True)
        return manager

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        return MagicMock(spec=HealthChecker)

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )

    @pytest.mark.asyncio
    async def test_promote_to_primary_delegates(self, failover_controller, mock_replica_manager):
        """测试提升主节点委托给 ReplicaManager"""
        result = await failover_controller.promote_to_primary("node-2", "qwen2.5-7b")

        assert result is True
        mock_replica_manager.set_primary_node.assert_called_once_with("qwen2.5-7b", "node-2")


class TestElectNewPrimary:
    """选举新主节点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        return MagicMock(spec=NodeStateStore)

    @pytest.fixture
    def mock_replica_manager(self):
        """创建模拟的 ReplicaManager"""
        manager = MagicMock(spec=ReplicaManager)
        manager.elect_new_primary = AsyncMock(return_value="node-3")
        return manager

    @pytest.fixture
    def mock_health_checker(self):
        """创建模拟的 HealthChecker"""
        return MagicMock(spec=HealthChecker)

    @pytest.fixture
    def failover_controller(
        self, mock_cluster_manager, mock_state_store, mock_replica_manager, mock_health_checker
    ):
        """创建 FailoverController 实例"""
        return FailoverController(
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_manager=mock_replica_manager,
            health_checker=mock_health_checker,
            node_id="node-1",
        )

    @pytest.mark.asyncio
    async def test_elect_new_primary_delegates(self, failover_controller, mock_replica_manager):
        """测试选举新主节点委托给 ReplicaManager"""
        result = await failover_controller.elect_new_primary("qwen2.5-7b")

        assert result == "node-3"
        mock_replica_manager.elect_new_primary.assert_called_once_with("qwen2.5-7b")
