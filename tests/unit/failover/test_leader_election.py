"""Leader Election 测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from quantumflow.failover.leader_election import LeaderElection
from quantumflow.failover.policy import ReplicaPolicy


class TestLeaderElection:
    """LeaderElection 测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock()
        manager.get_healthy_nodes = AsyncMock(return_value=[])
        manager.get_node = AsyncMock(return_value=None)
        return manager

    @pytest.fixture
    def mock_state_store(self):
        """创建模拟的 StateStore"""
        store = MagicMock()
        store.acquire_lock = AsyncMock(return_value=True)
        store.release_lock = AsyncMock(return_value=True)
        store.extend_lock = AsyncMock(return_value=True)
        store.set_leader = AsyncMock(return_value=True)
        store.get_leader = AsyncMock(return_value=("node-1", 1))
        return store

    @pytest.fixture
    def leader_election(self, mock_cluster_manager, mock_state_store):
        """创建 LeaderElection 实例"""
        policy = ReplicaPolicy(
            election_timeout_seconds=1,
            lock_ttl_seconds=30,
            health_check_interval_seconds=1,
        )
        return LeaderElection(
            node_id="node-1",
            cluster_manager=mock_cluster_manager,
            state_store=mock_state_store,
            replica_policy=policy,
        )

    def test_initial_state(self, leader_election):
        """测试初始状态"""
        assert leader_election.node_id == "node-1"
        assert leader_election.is_leader is False
        assert leader_election.current_term == 0
        assert leader_election.is_running is False

    @pytest.mark.asyncio
    async def test_start(self, leader_election):
        """测试启动"""
        await leader_election.start()
        assert leader_election.is_running is True

        await leader_election.stop()

    @pytest.mark.asyncio
    async def test_stop(self, leader_election):
        """测试停止"""
        await leader_election.start()
        await leader_election.stop()
        assert leader_election.is_running is False

    @pytest.mark.asyncio
    async def test_acquire_lock(self, leader_election, mock_state_store):
        """测试获取锁"""
        result = await leader_election.acquire_lock("test_resource")
        assert result is True
        mock_state_store.acquire_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_lock(self, leader_election, mock_state_store):
        """测试释放锁"""
        result = await leader_election.release_lock("test_resource")
        assert result is True
        mock_state_store.release_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_extend_lock(self, leader_election, mock_state_store):
        """测试延长锁"""
        result = await leader_election.extend_lock("test_resource")
        assert result is True
        mock_state_store.extend_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_vote_for_candidate(self, leader_election):
        """测试投票"""
        # 第一次投票
        voted, term = await leader_election.vote_for_candidate("node-2", 1)
        assert voted is True
        assert term == 1

        # 同一个候选者再次投票
        voted, term = await leader_election.vote_for_candidate("node-2", 1)
        assert voted is True

    @pytest.mark.asyncio
    async def test_vote_for_candidate_old_term(self, leader_election):
        """测试给旧 Term 的候选者投票"""
        leader_election._current_term = 5

        # 给旧 Term 投票
        voted, term = await leader_election.vote_for_candidate("node-2", 3)
        assert voted is False
        assert term == 5

    @pytest.mark.asyncio
    async def test_get_current_leader(self, leader_election, mock_state_store):
        """测试获取当前 Leader"""
        leader_id, term = await leader_election.get_current_leader()
        assert leader_id == "node-1"
        assert term == 1

    @pytest.mark.asyncio
    async def test_callbacks(self, leader_election):
        """测试回调"""
        become_called = []
        lose_called = []

        async def on_become(node_id, term):
            become_called.append((node_id, term))

        def on_lose(node_id):
            lose_called.append(node_id)

        leader_election.set_on_become_leader(on_become)
        leader_election.set_on_lose_leader(on_lose)

        # 模拟成为 Leader
        leader_election._is_leader = True
        await leader_election._on_become_leader("node-1", 1)
        assert len(become_called) == 1
        assert become_called[0] == ("node-1", 1)

        # 模拟失去 Leader - 调用 _lose_leadership 会触发回调
        leader_election._is_leader = True  # 需要是 leader 才能失去
        leader_election._lock_acquired = True  # 需要锁才能释放
        await leader_election._lose_leadership()
        assert len(lose_called) == 1
        assert lose_called[0] == "node-1"
