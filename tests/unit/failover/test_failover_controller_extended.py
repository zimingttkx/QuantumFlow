"""FailoverController 扩展测试 - 提高覆盖率

补充测试以覆盖未覆盖的代码分支：
- 控制器启动/停止边界
- 事件处理回调
- 节点故障处理路径
- 健康检查和手动故障转移
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus
from quantumflow.failover.controller import FailoverController, FailoverDecision
from quantumflow.failover.health_checker import HealthChecker, HealthCheckResult, GPUHealthResult
from quantumflow.failover.models import FailoverEvent, FailoverState, HealthStatus, ReplicaRole
from quantumflow.failover.policy import FailoverPolicy, ReplicaPolicy
from quantumflow.failover.replica_manager import ReplicaManager, SyncResult
from quantumflow.failover.state_store import NodeStateStore


class TestFailoverControllerStartStop:
    """控制器启动/停止测试"""

    @pytest.fixture
    def mock_dependencies(self):
        """创建模拟依赖"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            status=NodeStatus.HEALTHY,
            loaded_models=[],
        ))
        manager.on = MagicMock()

        store = MagicMock(spec=NodeStateStore)
        store.generate_event_id = MagicMock(return_value="fe_test123")
        store.save_failover_event = AsyncMock(return_value=True)
        store.get_leader = AsyncMock(return_value=(None, 0))
        store.get_all_node_states = AsyncMock(return_value=[])

        replica_mgr = MagicMock(spec=ReplicaManager)
        replica_mgr.get_all_replicas = AsyncMock(return_value=[])

        health_checker = MagicMock(spec=HealthChecker)
        health_checker.start = AsyncMock()
        health_checker.stop = AsyncMock()
        health_checker.check_node_health = AsyncMock(return_value=MagicMock(
            status=HealthStatus.HEALTHY,
            is_healthy=MagicMock(return_value=True),
            is_degraded=MagicMock(return_value=False),
            is_unhealthy=MagicMock(return_value=False),
            reasons=[],
        ))

        leader_election = MagicMock()
        leader_election.start = AsyncMock()
        leader_election.stop = AsyncMock()
        leader_election.check_is_leader = AsyncMock(return_value=False)
        leader_election.set_on_become_leader = MagicMock()
        leader_election.set_on_lose_leader = MagicMock()

        return manager, store, replica_mgr, health_checker, leader_election

    @pytest.fixture
    def failover_controller(self, mock_dependencies):
        """创建 FailoverController 实例"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        policy = FailoverPolicy()
        replica_policy = ReplicaPolicy()

        controller = FailoverController(
            cluster_manager=manager,
            state_store=store,
            replica_manager=replica_mgr,
            health_checker=health_checker,
            node_id="node-1",
            failover_policy=policy,
            replica_policy=replica_policy,
        )
        controller._leader_election = leader_election
        return controller

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, failover_controller):
        """测试启动时如果已在运行则跳过"""
        failover_controller._running = True

        await failover_controller.start()

        # 不应该再次启动
        failover_controller._health_checker.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_normal(self, failover_controller, mock_dependencies):
        """测试正常启动"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        await failover_controller.start()

        assert failover_controller._running is True
        leader_election.start.assert_called_once()
        health_checker.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_normal(self, failover_controller, mock_dependencies):
        """测试正常停止"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        failover_controller._running = True
        await failover_controller.stop()

        assert failover_controller._running is False
        health_checker.stop.assert_called_once()
        leader_election.stop.assert_called_once()


class TestFailoverControllerCallbacks:
    """控制器回调测试"""

    @pytest.fixture
    def mock_dependencies(self):
        """创建模拟依赖"""
        manager = MagicMock(spec=ClusterManager)
        store = MagicMock(spec=NodeStateStore)
        replica_mgr = MagicMock(spec=ReplicaManager)
        health_checker = MagicMock(spec=HealthChecker)
        leader_election = MagicMock()
        return manager, store, replica_mgr, health_checker, leader_election

    @pytest.fixture
    def failover_controller(self, mock_dependencies):
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        controller = FailoverController(
            cluster_manager=manager,
            state_store=store,
            replica_manager=replica_mgr,
            health_checker=health_checker,
            node_id="node-1",
        )
        controller._leader_election = leader_election
        return controller

    @pytest.mark.asyncio
    async def test_on_become_leader(self, failover_controller):
        """测试成为 Leader 回调"""
        event_received = []

        async def on_event(data):
            event_received.append(data)

        failover_controller.on("leader_changed", on_event)

        await failover_controller._on_become_leader("node-1", 5)

        assert len(event_received) == 1
        assert event_received[0]["is_leader"] is True
        assert event_received[0]["node_id"] == "node-1"

    @pytest.mark.asyncio
    async def test_on_lose_leader(self, failover_controller):
        """测试失去 Leader 回调"""
        event_received = []

        async def on_event(data):
            event_received.append(data)

        failover_controller.on("leader_changed", on_event)

        await failover_controller._on_lose_leader("node-1")

        assert len(event_received) == 1
        assert event_received[0]["is_leader"] is False


class TestFailoverControllerEventHandlers:
    """控制器事件处理测试"""

    @pytest.fixture
    def mock_dependencies(self):
        """创建模拟依赖"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()

        store = MagicMock(spec=NodeStateStore)
        replica_mgr = MagicMock(spec=ReplicaManager)

        health_checker = MagicMock(spec=HealthChecker)

        leader_election = MagicMock()
        leader_election.check_is_leader = AsyncMock(return_value=True)

        return manager, store, replica_mgr, health_checker, leader_election

    @pytest.fixture
    def failover_controller(self, mock_dependencies):
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        controller = FailoverController(
            cluster_manager=manager,
            state_store=store,
            replica_manager=replica_mgr,
            health_checker=health_checker,
            node_id="node-1",
        )
        controller._leader_election = leader_election
        return controller

    @pytest.mark.asyncio
    async def test_on_node_health_changed_unhealthy(self, failover_controller, mock_dependencies):
        """测试节点变为不健康时触发处理"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        unhealthy_node = MagicMock()
        unhealthy_node.node_id = "node-2"

        # 使用字符串使 str(new_status) == "unhealthy" 为真
        # 注意：实际代码检查的是 str() 而不是 .value
        old_status = "healthy"
        new_status = "unhealthy"

        health_checker.check_node_health = AsyncMock(return_value=MagicMock(
            status=HealthStatus.UNHEALTHY,
            is_unhealthy=MagicMock(return_value=True),
        ))

        failover_controller._handle_unhealthy_node = AsyncMock()

        await failover_controller._on_node_health_changed(unhealthy_node, old_status, new_status)

        # 应该触发不健康节点处理
        failover_controller._handle_unhealthy_node.assert_called_once_with("node-2")

    @pytest.mark.asyncio
    async def test_on_node_left(self, failover_controller, mock_dependencies):
        """测试节点离开时触发处理"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        left_node = MagicMock()
        left_node.node_id = "node-3"

        failover_controller.initiate_failover = AsyncMock(return_value=True)

        await failover_controller._on_node_left(left_node)

        failover_controller.initiate_failover.assert_called_once_with("node-3", "node_left")

    @pytest.mark.asyncio
    async def test_handle_unhealthy_node_auto_failover_disabled(self, failover_controller):
        """测试自动故障转移禁用时不处理"""
        failover_controller._failover_policy.auto_failover_enabled = False

        await failover_controller._handle_unhealthy_node("node-1")

        # 不应该执行健康检查
        failover_controller._health_checker.check_node_health.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_node_failure_already_in_progress(self, failover_controller):
        """测试故障转移进行中时跳过"""
        failover_controller._failover_in_progress = True

        result = await failover_controller._handle_node_failure("node-1", "test")

        assert result is None  # 没有返回（直接返回）

    @pytest.mark.asyncio
    async def test_handle_node_failure_not_leader(self, failover_controller, mock_dependencies):
        """测试不是 Leader 时跳过故障转移"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies
        leader_election.check_is_leader = AsyncMock(return_value=False)

        # 不应该触发 initiate_failover (通过 patch 追踪)
        with patch.object(failover_controller, 'initiate_failover', new_callable=AsyncMock) as mock_failover:
            await failover_controller._handle_node_failure("node-1", "test")
            # 不应该发起故障转移
            mock_failover.assert_not_called()


class TestFailoverControllerManualFailover:
    """手动故障转移测试"""

    @pytest.fixture
    def mock_dependencies(self):
        """创建模拟依赖"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()

        store = MagicMock(spec=NodeStateStore)
        replica_mgr = MagicMock(spec=ReplicaManager)
        replica_mgr.elect_new_primary = AsyncMock(return_value="node-2")
        replica_mgr.set_primary_node = AsyncMock(return_value=True)

        health_checker = MagicMock(spec=HealthChecker)

        leader_election = MagicMock()

        return manager, store, replica_mgr, health_checker, leader_election

    @pytest.fixture
    def failover_controller(self, mock_dependencies):
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        controller = FailoverController(
            cluster_manager=manager,
            state_store=store,
            replica_manager=replica_mgr,
            health_checker=health_checker,
            node_id="node-1",
        )
        controller._leader_election = leader_election
        return controller

    @pytest.mark.asyncio
    async def test_manual_failover_requires_confirmation(self, failover_controller):
        """测试需要手动确认时拒绝"""
        failover_controller._failover_policy.require_manual_confirmation = True

        result = await failover_controller.manual_failover("node-1", "node-2")

        assert result is False

    @pytest.mark.asyncio
    async def test_manual_failover_node_not_found(self, failover_controller, mock_dependencies):
        """测试手动故障转移节点不存在"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies
        manager.get_node = AsyncMock(return_value=None)

        result = await failover_controller.manual_failover("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_manual_failover_with_target(self, failover_controller, mock_dependencies):
        """测试带目标节点的手动故障转移"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        node = MagicMock()
        node.loaded_models = ["model-1"]
        manager.get_node = AsyncMock(return_value=node)

        result = await failover_controller.manual_failover("node-1", "node-2")

        assert result is True
        replica_mgr.set_primary_node.assert_called()

    @pytest.mark.asyncio
    async def test_manual_failover_auto_select(self, failover_controller, mock_dependencies):
        """测试自动选择目标节点的手动故障转移"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        node = MagicMock()
        node.loaded_models = ["model-1"]
        manager.get_node = AsyncMock(return_value=node)

        result = await failover_controller.manual_failover("node-1")

        assert result is True
        replica_mgr.elect_new_primary.assert_called()


class TestFailoverControllerQuery:
    """控制器查询测试"""

    @pytest.fixture
    def mock_dependencies(self):
        """创建模拟依赖"""
        manager = MagicMock(spec=ClusterManager)
        store = MagicMock(spec=NodeStateStore)
        replica_mgr = MagicMock(spec=ReplicaManager)
        health_checker = MagicMock(spec=HealthChecker)
        leader_election = MagicMock()
        return manager, store, replica_mgr, health_checker, leader_election

    @pytest.fixture
    def failover_controller(self, mock_dependencies):
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        controller = FailoverController(
            cluster_manager=manager,
            state_store=store,
            replica_manager=replica_mgr,
            health_checker=health_checker,
            node_id="node-1",
        )
        controller._leader_election = leader_election
        return controller

    @pytest.mark.asyncio
    async def test_check_node_health(self, failover_controller, mock_dependencies):
        """测试检查节点健康状态"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        health_checker.check_node_health = AsyncMock(return_value=MagicMock(
            status=HealthStatus.HEALTHY,
        ))

        result = await failover_controller.check_node_health("node-1")

        assert result == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_is_leader(self, failover_controller, mock_dependencies):
        """测试判断是否为 Leader"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies
        leader_election.check_is_leader = AsyncMock(return_value=True)

        result = await failover_controller.is_leader()

        assert result is True

    @pytest.mark.asyncio
    async def test_get_leader_election(self, failover_controller, mock_dependencies):
        """测试获取 Leader 选举器"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        result = failover_controller.get_leader_election()

        assert result is leader_election


class TestFailoverControllerClusterStatus:
    """集群状态测试"""

    @pytest.fixture
    def mock_dependencies(self):
        """创建模拟依赖"""
        manager = MagicMock(spec=ClusterManager)
        store = MagicMock(spec=NodeStateStore)
        replica_mgr = MagicMock(spec=ReplicaManager)
        health_checker = MagicMock(spec=HealthChecker)
        leader_election = MagicMock()
        return manager, store, replica_mgr, health_checker, leader_election

    @pytest.fixture
    def failover_controller(self, mock_dependencies):
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        controller = FailoverController(
            cluster_manager=manager,
            state_store=store,
            replica_manager=replica_mgr,
            health_checker=health_checker,
            node_id="node-1",
        )
        controller._leader_election = leader_election
        return controller

    @pytest.mark.asyncio
    async def test_get_cluster_failover_status(self, failover_controller, mock_dependencies):
        """测试获取集群故障转移状态"""
        manager, store, replica_mgr, health_checker, leader_election = mock_dependencies

        from quantumflow.failover.models import NodeFailoverState

        healthy_state = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.PRIMARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
        )
        degraded_state = NodeFailoverState(
            node_id="node-2",
            role=ReplicaRole.SECONDARY,
            state=FailoverState.DEGRADED,
            health=HealthStatus.DEGRADED,
            term=1,
            last_heartbeat=datetime.now(),
        )

        store.get_leader = AsyncMock(return_value=("node-1", 5))
        store.get_all_node_states = AsyncMock(return_value=[healthy_state, degraded_state])
        replica_mgr.get_all_replicas = AsyncMock(return_value=[])
        leader_election.check_is_leader = AsyncMock(return_value=True)

        result = await failover_controller.get_cluster_failover_status()

        assert result["leader_id"] == "node-1"
        assert result["leader_term"] == 5
        assert result["total_nodes"] == 2
        assert result["healthy_nodes"] == 1
        assert result["degraded_nodes"] == 1
        assert result["unhealthy_nodes"] == 0
        assert result["is_leader"] is True


class TestFailoverDecision:
    """故障转移决策测试"""

    def test_failover_decision_creation(self):
        """测试 FailoverDecision 创建"""
        decision = FailoverDecision(
            should_failover=True,
            reason="test_reason",
            failed_node="node-1",
            target_nodes=["node-2"],
            affected_models=["model-1"],
            risk_level="medium",
        )

        assert decision.should_failover is True
        assert decision.reason == "test_reason"
        assert decision.failed_node == "node-1"
        assert decision.target_nodes == ["node-2"]
        assert decision.affected_models == ["model-1"]
        assert decision.risk_level == "medium"
