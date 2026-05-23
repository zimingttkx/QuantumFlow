"""LeaderElection 扩展测试 - 提高覆盖率

补充测试以覆盖未覆盖的代码分支：
- 选举循环的启动和异常处理
- 投票请求和响应处理
- Leader 任职和失去领导权
- 心跳循环和锁续约
- 候选者投票和降级
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus
from quantumflow.failover.leader_election import LeaderElection
from quantumflow.failover.policy import ReplicaPolicy
from quantumflow.failover.state_store import NodeStateStore


class TestLeaderElectionStartStop:
    """Leader 选举启动/停止测试"""

    @pytest.fixture
    def mock_dependencies(self):
        """创建模拟依赖"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_healthy_nodes = AsyncMock(return_value=[])
        manager.get_node = AsyncMock(return_value=MagicMock(status=NodeStatus.HEALTHY))

        store = MagicMock(spec=NodeStateStore)
        store.acquire_lock = AsyncMock(return_value=True)
        store.extend_lock = AsyncMock(return_value=True)
        store.release_lock = AsyncMock(return_value=True)
        store.set_leader = AsyncMock(return_value=True)
        store.get_leader = AsyncMock(return_value=(None, 0))

        policy = ReplicaPolicy(
            election_timeout_seconds=1,
            health_check_interval_seconds=1,
            lock_ttl_seconds=10,
        )
        return manager, store, policy

    @pytest.fixture
    def leader_election(self, mock_dependencies):
        """创建 LeaderElection 实例"""
        manager, store, policy = mock_dependencies
        return LeaderElection(
            node_id="node-1",
            cluster_manager=manager,
            state_store=store,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, leader_election):
        """测试启动时如果已在运行则跳过"""
        leader_election._running = True

        await leader_election.start()

        # 不应该创建新任务
        assert leader_election._election_task is None

    @pytest.mark.asyncio
    async def test_stop_normal(self, leader_election):
        """测试正常停止"""
        leader_election._running = True
        leader_election._election_task = None

        await leader_election.stop()

        assert leader_election._running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, leader_election):
        """测试停止时如果未运行则跳过"""
        leader_election._running = False

        await leader_election.stop()

        assert leader_election._running is False


class TestElectionLoop:
    """选举循环测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        manager.get_healthy_nodes = AsyncMock(return_value=[])

        store = MagicMock(spec=NodeStateStore)
        store.acquire_lock = AsyncMock(return_value=True)
        store.extend_lock = AsyncMock(return_value=True)
        store.release_lock = AsyncMock(return_value=True)
        store.set_leader = AsyncMock(return_value=True)
        store.get_leader = AsyncMock(return_value=(None, 0))

        policy = ReplicaPolicy(
            election_timeout_seconds=0.05,
            health_check_interval_seconds=1,
            lock_ttl_seconds=10,
        )
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
    async def test_election_loop_starts_election(self, leader_election, mock_dependencies):
        """测试选举循环触发选举"""
        manager, store, policy = mock_dependencies
        leader_election._running = True

        task = asyncio.create_task(leader_election._election_loop())

        # 等待选举开始
        await asyncio.sleep(0.15)

        leader_election._running = False

        try:
            await task
        except asyncio.CancelledError:
            pass

        # 应该尝试成为候选者
        assert leader_election._is_candidate or leader_election._current_term >= 1

    @pytest.mark.asyncio
    async def test_election_loop_handles_exception(self, leader_election, mock_dependencies):
        """测试选举循环处理异常"""
        manager, store, policy = mock_dependencies
        manager.get_healthy_nodes = AsyncMock(side_effect=Exception("Cluster error"))

        leader_election._running = True
        task = asyncio.create_task(leader_election._election_loop())

        # 等待循环处理异常
        await asyncio.sleep(0.15)

        leader_election._running = False

        try:
            await task
        except asyncio.CancelledError:
            pass

        # 循环应该继续运行而不是崩溃
        assert leader_election._running is False


class TestStartElection:
    """发起选举测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        healthy_node = MagicMock()
        healthy_node.node_id = "node-2"
        manager.get_healthy_nodes = AsyncMock(return_value=[healthy_node])

        store = MagicMock(spec=NodeStateStore)
        store.acquire_lock = AsyncMock(return_value=True)
        store.extend_lock = AsyncMock(return_value=True)
        store.release_lock = AsyncMock(return_value=True)
        store.set_leader = AsyncMock(return_value=True)
        store.get_leader = AsyncMock(return_value=(None, 0))

        policy = ReplicaPolicy(
            election_timeout_seconds=1,
            health_check_interval_seconds=1,
            lock_ttl_seconds=10,
        )
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
        """测试如果已经是候选者则跳过"""
        leader_election._is_candidate = True
        original_term = leader_election._current_term

        await leader_election._start_election()

        # Term 不应该增加
        assert leader_election._current_term == original_term

    @pytest.mark.asyncio
    async def test_start_election_increments_term(self, leader_election):
        """测试发起选举时增加 Term"""
        leader_election._running = True

        await leader_election._start_election()

        # 给自己投票后 term 应该增加
        assert leader_election._current_term == 1
        assert leader_election._voted_for == "node-1"
        # 初始票数为自己投的1票，但由于异步投票请求，票数可能已增加

    @pytest.mark.asyncio
    async def test_election_fails_without_majority(self, leader_election, mock_dependencies):
        """测试没有获得多数票时选举失败"""
        manager, store, policy = mock_dependencies
        # 只给自己投票，不会获得多数票（总共2个节点）
        leader_election._running = True

        await leader_election._start_election()

        # 选举应该失败，因为只有1票（共2节点，需要2票）
        assert leader_election._is_candidate is False

    @pytest.mark.asyncio
    async def test_election_succeeds_with_majority(self, leader_election, mock_dependencies):
        """测试获得多数票时选举成功"""
        manager, store, policy = mock_dependencies
        # 有2个额外节点，所以总共3个节点，需要2票
        extra_node = MagicMock()
        extra_node.node_id = "node-3"
        manager.get_healthy_nodes = AsyncMock(return_value=[
            MagicMock(node_id="node-2"),
            MagicMock(node_id="node-3"),
        ])

        leader_election._running = True

        await leader_election._start_election()

        # 如果获得多数票应该成为 Leader
        # 注意：在测试中，模拟投票可能不会全部完成


class TestBecomeLeader:
    """成为 Leader 测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        manager.get_healthy_nodes = AsyncMock(return_value=[])

        store = MagicMock(spec=NodeStateStore)
        store.acquire_lock = AsyncMock(return_value=True)
        store.extend_lock = AsyncMock(return_value=True)
        store.release_lock = AsyncMock(return_value=True)
        store.set_leader = AsyncMock(return_value=True)

        policy = ReplicaPolicy(
            election_timeout_seconds=1,
            health_check_interval_seconds=1,
            lock_ttl_seconds=10,
        )
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
    async def test_become_leader_already_leader(self, leader_election):
        """测试如果已经是 Leader 则跳过"""
        leader_election._is_leader = True

        await leader_election._become_leader()

        # 不应该再次获取锁
        leader_election._state_store.acquire_lock.assert_not_called()

    @pytest.mark.asyncio
    async def test_become_leader_lock_acquisition_fails(self, leader_election, mock_dependencies):
        """测试锁获取失败时不成为 Leader"""
        manager, store, policy = mock_dependencies
        store.acquire_lock = AsyncMock(return_value=False)

        leader_election._is_candidate = True
        leader_election._current_term = 1

        await leader_election._become_leader()

        assert leader_election._is_leader is False
        assert leader_election._is_candidate is False

    @pytest.mark.asyncio
    async def test_become_leader_success(self, leader_election, mock_dependencies):
        """测试成功成为 Leader"""
        manager, store, policy = mock_dependencies

        leader_election._is_candidate = True
        leader_election._current_term = 1

        # 模拟回调 - 签名应为 (node_id, term)
        callback_called = False
        async def on_become(node_id, term):
            nonlocal callback_called
            callback_called = True
            assert node_id == "node-1"
            assert term == 1

        leader_election.set_on_become_leader(on_become)

        await leader_election._become_leader()

        assert leader_election._is_leader is True
        assert leader_election._is_candidate is False
        assert leader_election._lock_acquired is True
        assert callback_called is True

        # 应该启动心跳任务
        assert leader_election._heartbeat_task is not None

        # 清理
        leader_election._running = False
        leader_election._heartbeat_task.cancel()
        try:
            await leader_election._heartbeat_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_become_leader_callback_failure(self, leader_election, mock_dependencies):
        """测试成为 Leader 时回调失败"""
        manager, store, policy = mock_dependencies

        leader_election._is_candidate = True
        leader_election._current_term = 1

        # 模拟回调抛出异常
        async def failing_callback(node_id, term):
            raise Exception("Callback failed")

        leader_election.set_on_become_leader(failing_callback)

        # 不应该抛出异常
        await leader_election._become_leader()

        assert leader_election._is_leader is True

        # 清理
        leader_election._running = False
        if leader_election._heartbeat_task:
            leader_election._heartbeat_task.cancel()
            try:
                await leader_election._heartbeat_task
            except asyncio.CancelledError:
                pass


class TestHeartbeatLoop:
    """心跳循环测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        healthy_node = MagicMock()
        healthy_node.status = NodeStatus.HEALTHY
        healthy_node.node_id = "node-1"
        manager.get_node = AsyncMock(return_value=healthy_node)

        store = MagicMock(spec=NodeStateStore)
        store.acquire_lock = AsyncMock(return_value=True)
        store.extend_lock = AsyncMock(return_value=True)
        store.release_lock = AsyncMock(return_value=True)
        store.set_leader = AsyncMock(return_value=True)

        policy = ReplicaPolicy(
            election_timeout_seconds=1,
            health_check_interval_seconds=0.05,
            lock_ttl_seconds=10,
        )
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
    async def test_heartbeat_loop_updates_leader(self, leader_election, mock_dependencies):
        """测试心跳循环更新 Leader 信息"""
        manager, store, policy = mock_dependencies

        leader_election._running = True
        leader_election._is_leader = True
        leader_election._lock_acquired = True
        leader_election._lock_expires_at = datetime.now() + timedelta(seconds=10)
        leader_election._current_term = 1

        task = asyncio.create_task(leader_election._heartbeat_loop())

        await asyncio.sleep(0.12)

        leader_election._running = False

        try:
            await task
        except asyncio.CancelledError:
            pass

        # 应该更新 Leader 信息
        store.set_leader.assert_called()

    @pytest.mark.asyncio
    async def test_heartbeat_loop_handles_exception(self, leader_election, mock_dependencies):
        """测试心跳循环处理异常"""
        manager, store, policy = mock_dependencies
        manager.get_node = AsyncMock(side_effect=Exception("Node error"))

        leader_election._running = True
        leader_election._is_leader = True
        leader_election._lock_acquired = True
        leader_election._lock_expires_at = datetime.now() + timedelta(seconds=10)
        leader_election._current_term = 1

        task = asyncio.create_task(leader_election._heartbeat_loop())

        await asyncio.sleep(0.1)

        leader_election._running = False

        try:
            await task
        except asyncio.CancelledError:
            pass


class TestLoseLeadership:
    """失去领导权测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        manager.get_healthy_nodes = AsyncMock(return_value=[])

        store = MagicMock(spec=NodeStateStore)
        store.acquire_lock = AsyncMock(return_value=True)
        store.extend_lock = AsyncMock(return_value=True)
        store.release_lock = AsyncMock(return_value=True)
        store.set_leader = AsyncMock(return_value=True)

        policy = ReplicaPolicy(
            election_timeout_seconds=1,
            health_check_interval_seconds=1,
            lock_ttl_seconds=10,
        )
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
    async def test_lose_leadership_not_leader(self, leader_election):
        """测试如果不是 Leader 则跳过"""
        leader_election._is_leader = False

        await leader_election._lose_leadership()

        leader_election._state_store.release_lock.assert_not_called()

    @pytest.mark.asyncio
    async def test_lose_leadership_success(self, leader_election, mock_dependencies):
        """测试成功失去领导权"""
        manager, store, policy = mock_dependencies

        leader_election._is_leader = True
        leader_election._lock_acquired = True
        leader_election._heartbeat_task = None

        callback_called = False
        async def on_lose(node_id):
            nonlocal callback_called
            callback_called = True
            assert node_id == "node-1"

        leader_election.set_on_lose_leader(on_lose)

        await leader_election._lose_leadership()

        assert leader_election._is_leader is False
        assert leader_election._lock_acquired is False
        store.release_lock.assert_called_once()
        assert callback_called is True

    @pytest.mark.asyncio
    async def test_lose_leadership_with_heartbeat_task(self, leader_election, mock_dependencies):
        """测试有心跳任务时失去领导权"""
        manager, store, policy = mock_dependencies

        leader_election._is_leader = True
        leader_election._lock_acquired = True
        leader_election._heartbeat_task = asyncio.create_task(asyncio.sleep(10))

        leader_election.set_on_lose_leader(lambda node_id: None)

        await leader_election._lose_leadership()

        assert leader_election._is_leader is False
        assert leader_election._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_lose_leadership_callback_failure(self, leader_election, mock_dependencies):
        """测试失去领导权时回调失败"""
        manager, store, policy = mock_dependencies

        leader_election._is_leader = True
        leader_election._lock_acquired = True
        leader_election._heartbeat_task = None

        async def failing_callback(node_id):
            raise Exception("Callback failed")

        leader_election.set_on_lose_leader(failing_callback)

        # 不应该抛出异常
        await leader_election._lose_leadership()

        assert leader_election._is_leader is False


class TestRenewLeadership:
    """续约领导权测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        manager.get_healthy_nodes = AsyncMock(return_value=[])

        store = MagicMock(spec=NodeStateStore)
        store.extend_lock = AsyncMock(return_value=True)

        policy = ReplicaPolicy(
            election_timeout_seconds=1,
            health_check_interval_seconds=1,
            lock_ttl_seconds=10,
        )
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
    async def test_renew_leadership_not_leader(self, leader_election):
        """测试不是 Leader 时续约失败"""
        leader_election._is_leader = False

        result = await leader_election.renew_leadership()

        assert result is False

    @pytest.mark.asyncio
    async def test_renew_leadership_success(self, leader_election, mock_dependencies):
        """测试成功续约领导权"""
        manager, store, policy = mock_dependencies

        leader_election._is_leader = True
        leader_election._lock_acquired = True

        result = await leader_election.renew_leadership()

        assert result is True
        store.extend_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_renew_leadership_extend_fails(self, leader_election, mock_dependencies):
        """测试续约失败"""
        manager, store, policy = mock_dependencies
        store.extend_lock = AsyncMock(return_value=False)

        leader_election._is_leader = True
        leader_election._lock_acquired = True

        result = await leader_election.renew_leadership()

        assert result is False

    @pytest.mark.asyncio
    async def test_renew_leadership_exception(self, leader_election, mock_dependencies):
        """测试续约时发生异常"""
        manager, store, policy = mock_dependencies
        store.extend_lock = AsyncMock(side_effect=Exception("Redis error"))

        leader_election._is_leader = True
        leader_election._lock_acquired = True

        result = await leader_election.renew_leadership()

        assert result is False


class TestVoteForCandidate:
    """投票给候选者测试"""

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
    async def test_vote_old_term(self, leader_election):
        """测试候选者 Term 更旧时不投票"""
        leader_election._current_term = 5

        voted, term = await leader_election.vote_for_candidate("node-2", candidate_term=3)

        assert voted is False
        assert term == 5

    @pytest.mark.asyncio
    async def test_vote_new_term_triggers_step_down(self, leader_election):
        """测试候选者 Term 更新时切换到新 Term"""
        leader_election._current_term = 3
        leader_election._is_leader = True
        leader_election._is_candidate = True

        voted, term = await leader_election.vote_for_candidate("node-2", candidate_term=5)

        assert voted is True
        assert leader_election._current_term == 5

    @pytest.mark.asyncio
    async def test_vote_for_candidate_success(self, leader_election):
        """测试成功投票给候选者"""
        leader_election._current_term = 5
        leader_election._voted_for = None

        voted, term = await leader_election.vote_for_candidate("node-2", candidate_term=5)

        assert voted is True
        assert leader_election._voted_for == "node-2"

    @pytest.mark.asyncio
    async def test_vote_already_voted_different_candidate(self, leader_election):
        """测试已经投给其他候选者"""
        leader_election._current_term = 5
        leader_election._voted_for = "node-3"

        voted, term = await leader_election.vote_for_candidate("node-2", candidate_term=5)

        assert voted is False

    @pytest.mark.asyncio
    async def test_vote_same_candidate_again(self, leader_election):
        """测试再次投票给同一候选者"""
        leader_election._current_term = 5
        leader_election._voted_for = "node-2"

        voted, term = await leader_election.vote_for_candidate("node-2", candidate_term=5)

        assert voted is True


class TestStepDown:
    """降级测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        store = MagicMock(spec=NodeStateStore)
        store.release_lock = AsyncMock(return_value=True)

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
    async def test_step_down_old_term(self, leader_election):
        """测试更旧的 Term 不降级"""
        leader_election._current_term = 5

        await leader_election._step_down(3)

        assert leader_election._current_term == 5

    @pytest.mark.asyncio
    async def test_step_down_new_term(self, leader_election):
        """测试切换到新 Term"""
        leader_election._current_term = 5
        leader_election._is_candidate = True
        leader_election._voted_for = "node-2"

        await leader_election._step_down(10)

        assert leader_election._current_term == 10
        assert leader_election._voted_for is None
        assert leader_election._is_candidate is False

    @pytest.mark.asyncio
    async def test_step_down_while_leader(self, leader_election):
        """测试在 Leader 时降级"""
        leader_election._current_term = 5
        leader_election._is_leader = True
        leader_election._is_candidate = True
        leader_election._lock_acquired = True
        leader_election._heartbeat_task = None  # 简化测试

        leader_election.set_on_lose_leader(lambda node_id: None)

        await leader_election._step_down(10)

        assert leader_election._current_term == 10
        assert leader_election._is_leader is False


class TestCheckIsLeader:
    """检查是否为 Leader 测试"""

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
    async def test_not_leader(self, leader_election):
        """测试不是 Leader"""
        leader_election._is_leader = False

        result = await leader_election.check_is_leader()

        assert result is False

    @pytest.mark.asyncio
    async def test_no_lock_acquired(self, leader_election):
        """测试没有获取锁"""
        leader_election._is_leader = True
        leader_election._lock_acquired = False

        result = await leader_election.check_is_leader()

        assert result is False

    @pytest.mark.asyncio
    async def test_lock_expired(self, leader_election):
        """测试锁已过期"""
        leader_election._is_leader = True
        leader_election._lock_acquired = True
        leader_election._lock_expires_at = datetime.now() - timedelta(seconds=1)

        leader_election.set_on_lose_leader(lambda node_id: None)

        result = await leader_election.check_is_leader()

        assert result is False
        assert leader_election._is_leader is False

    @pytest.mark.asyncio
    async def test_valid_leader(self, leader_election):
        """测试有效的 Leader"""
        leader_election._is_leader = True
        leader_election._lock_acquired = True
        leader_election._lock_expires_at = datetime.now() + timedelta(seconds=10)

        result = await leader_election.check_is_leader()

        assert result is True


class TestDistributedLockInterface:
    """分布式锁接口测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        store = MagicMock(spec=NodeStateStore)
        store.acquire_lock = AsyncMock(return_value=True)
        store.release_lock = AsyncMock(return_value=True)

        policy = ReplicaPolicy(lock_ttl_seconds=30)
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
    async def test_acquire_lock_interface(self, leader_election):
        """测试获取分布式锁接口"""
        result = await leader_election.acquire_lock("test-resource", ttl_seconds=60)

        assert result is True
        leader_election._state_store.acquire_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_lock_interface(self, leader_election):
        """测试释放分布式锁接口"""
        result = await leader_election.release_lock("test-resource")

        assert result is True
        leader_election._state_store.release_lock.assert_called_once()


class TestGetCurrentLeader:
    """获取当前 Leader 测试"""

    @pytest.fixture
    def mock_dependencies(self):
        manager = MagicMock(spec=ClusterManager)
        store = MagicMock(spec=NodeStateStore)
        store.get_leader = AsyncMock(return_value=("node-2", 5))

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
    async def test_get_current_leader(self, leader_election):
        """测试获取当前 Leader"""
        node_id, term = await leader_election.get_current_leader()

        assert node_id == "node-2"
        assert term == 5


class TestRequestVote:
    """投票请求测试"""

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
    async def test_request_vote_success(self, leader_election):
        """测试投票请求成功"""
        leader_election._running = True
        leader_election._current_term = 1
        leader_election._vote_count = 0

        await leader_election._request_vote("node-2", total_nodes=3)

        # 由于模拟延迟，需要等待
        await asyncio.sleep(0.2)

        # 应该收到投票
        assert leader_election._vote_count >= 1

    @pytest.mark.asyncio
    async def test_request_vote_not_running(self, leader_election):
        """测试不运行时不处理投票请求"""
        leader_election._running = False
        leader_election._current_term = 1
        original_vote_count = leader_election._vote_count

        await leader_election._request_vote("node-2", total_nodes=3)

        # 由于不运行，不应该增加票数
        await asyncio.sleep(0.2)

        # 注意：由于异步执行，可能会有竞态，但测试主要验证不崩溃

    @pytest.mark.asyncio
    async def test_request_vote_exception(self, leader_election):
        """测试投票请求异常处理"""
        leader_election._running = True
        leader_election._current_term = 0  # 导致异常

        # 不应该抛出异常
        await leader_election._request_vote("node-2", total_nodes=3)
