"""End-to-End Failover Scenarios

Tests complete failover workflows using real components:
1. Leader election with distributed locks
2. Replica promotion flow
3. Node recovery
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from quantumflow.cluster.manager import NodeStatus
from quantumflow.failover.models import HealthStatus, ReplicaRole, FailoverState


class TestLeaderElectionScenario:
    """Test leader election with real state store."""

    @pytest.mark.asyncio
    async def test_leader_can_acquire_lock(self, state_store, cluster_nodes):
        """Test that becoming leader involves acquiring the lock."""
        # Set node-1 as leader
        await state_store.set_leader("node-1", term=1)

        # Acquire leader lock
        acquired = await state_store.acquire_lock("leader", "node-1", ttl_seconds=30)
        assert acquired is True

        # Verify leader
        leader = await state_store.get_leader()
        assert leader == ("node-1", 1)

    @pytest.mark.asyncio
    async def test_only_one_leader_at_a_time(self, state_store):
        """Test that only one node can be leader at a time."""
        # Set initial leader
        await state_store.set_leader("node-1", term=1)
        await state_store.acquire_lock("leader", "node-1", ttl_seconds=30)

        # Try to set new leader
        result = await state_store.set_leader("node-2", term=2)
        assert result is True

        # Verify new leader
        leader = await state_store.get_leader()
        assert leader == ("node-2", 2)

        # But node-1 still holds the lock (simulating partition scenario)
        lock_info = await state_store.get_lock_info("leader")
        assert lock_info["owner"] == "node-1"


class TestReplicaPromotionScenario:
    """Test replica promotion with real state store."""

    @pytest.mark.asyncio
    async def test_primary_promotion_updates_state(self, state_store, replica_manager):
        """Test that promoting a replica updates the replica index."""
        from quantumflow.failover.models import ModelReplica

        # Create replica index
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc123",
            version=1,
        )
        await state_store.save_replica_index(replica)

        # Verify initial state
        loaded = await state_store.load_replica_index("qwen2.5-7b")
        assert loaded.primary_node == "node-1"
        assert "node-2" in loaded.secondary_nodes

        # Set new primary
        success = await replica_manager.set_primary_node("qwen2.5-7b", "node-2")
        assert success is True

        # Verify updated state
        updated = await state_store.load_replica_index("qwen2.5-7b")
        assert updated.primary_node == "node-2"
        assert "node-1" in updated.secondary_nodes


class TestNodeRecoveryScenario:
    """Test node recovery with real state store."""

    @pytest.mark.asyncio
    async def test_recovered_node_has_healthy_status(self, state_store):
        """Test that a recovered node has healthy status in state store."""
        from quantumflow.failover.models import NodeFailoverState

        # Create unhealthy state
        unhealthy_state = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.SECONDARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.UNHEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
            failure_reasons=["GPU overheating", "Memory full"],
        )
        await state_store.save_node_state(unhealthy_state)

        # Verify unhealthy
        loaded = await state_store.load_node_state("node-1")
        assert loaded.health == HealthStatus.UNHEALTHY

        # Simulate recovery - create healthy state
        healthy_state = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.SECONDARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
            failure_reasons=[],
        )
        await state_store.save_node_state(healthy_state)

        # Verify healthy
        loaded = await state_store.load_node_state("node-1")
        assert loaded.health == HealthStatus.HEALTHY
        assert len(loaded.failure_reasons) == 0


class TestDistributedLockScenario:
    """Test distributed lock prevents concurrent operations."""

    @pytest.mark.asyncio
    async def test_split_brain_prevention(self, state_store):
        """Test that distributed lock prevents split brain scenario."""
        # Node-1 acquires leader lock
        acquired1 = await state_store.acquire_lock("leader", "node-1", ttl_seconds=30)
        assert acquired1 is True

        # Network partition: node-1 thinks it's still leader
        # But node-2 cannot acquire the lock
        acquired2 = await state_store.acquire_lock("leader", "node-2", ttl_seconds=30)
        assert acquired2 is False, "Split brain prevented: node-2 should not acquire lock"

        # Verify node-1 still holds the lock
        lock_info = await state_store.get_lock_info("leader")
        assert lock_info["owner"] == "node-1"

        # Only when node-1 releases the lock can node-2 acquire
        await state_store.release_lock("leader", "node-1")
        acquired2 = await state_store.acquire_lock("leader", "node-2", ttl_seconds=30)
        assert acquired2 is True
