"""NodeStateStore Integration Tests with Real Redis

Tests NodeStateStore using fakeredis to verify real Redis interactions:
1. Lock acquisition and release
2. Leader election state
3. Failover event storage
4. Concurrent lock operations
"""

import asyncio
from datetime import datetime

import pytest

from quantumflow.failover.models import FailoverEvent, ReplicaRole, FailoverState, HealthStatus
from quantumflow.failover.state_store import NodeStateStore


class TestDistributedLocks:
    """Test distributed lock operations with real Redis."""

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, state_store):
        """Test successful lock acquisition."""
        acquired = await state_store.acquire_lock("resource-1", "node-1", ttl_seconds=30)
        assert acquired is True, "Failed to acquire lock"

    @pytest.mark.asyncio
    async def test_acquire_lock_already_held(self, state_store):
        """Test that lock acquisition fails when already held."""
        # First acquisition succeeds
        acquired1 = await state_store.acquire_lock("resource-1", "node-1", ttl_seconds=30)
        assert acquired1 is True

        # Second acquisition fails
        acquired2 = await state_store.acquire_lock("resource-1", "node-2", ttl_seconds=30)
        assert acquired2 is False, "Lock should not be acquired by another node"

    @pytest.mark.asyncio
    async def test_release_lock(self, state_store):
        """Test releasing a lock."""
        # Acquire lock
        await state_store.acquire_lock("resource-1", "node-1", ttl_seconds=30)

        # Release lock
        released = await state_store.release_lock("resource-1", "node-1")
        assert released is True

        # Verify lock is released (another node can acquire it)
        acquired = await state_store.acquire_lock("resource-1", "node-2", ttl_seconds=30)
        assert acquired is True

    @pytest.mark.asyncio
    async def test_release_lock_wrong_owner(self, state_store):
        """Test that releasing a lock with wrong owner fails."""
        # Acquire lock as node-1
        await state_store.acquire_lock("resource-1", "node-1", ttl_seconds=30)

        # Try to release as node-2 (wrong owner)
        released = await state_store.release_lock("resource-1", "node-2")
        assert released is False, "Should not release lock owned by another node"

    @pytest.mark.asyncio
    async def test_lock_expiration(self, state_store):
        """Test that lock expires after TTL."""
        # Acquire lock with very short TTL
        acquired = await state_store.acquire_lock("resource-1", "node-1", ttl_seconds=1)
        assert acquired is True

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Another node should be able to acquire
        acquired2 = await state_store.acquire_lock("resource-1", "node-2", ttl_seconds=30)
        assert acquired2 is True, "Lock should have expired"

    @pytest.mark.asyncio
    async def test_get_lock_info(self, state_store):
        """Test getting lock information."""
        # Acquire lock
        await state_store.acquire_lock("resource-1", "node-1", ttl_seconds=30)

        # Get lock info
        info = await state_store.get_lock_info("resource-1")
        assert info is not None
        assert info["owner"] == "node-1"
        assert "acquired_at" in info
        assert "expires_at" in info


class TestLeaderElection:
    """Test leader election state management."""

    @pytest.mark.asyncio
    async def test_set_and_get_leader(self, state_store):
        """Test setting and getting leader."""
        # Set leader
        result = await state_store.set_leader("node-1", term=1)
        assert result is True

        # Get leader
        leader = await state_store.get_leader()
        assert leader is not None
        leader_id, term = leader
        assert leader_id == "node-1"
        assert term == 1

    @pytest.mark.asyncio
    async def test_update_leader_term(self, state_store):
        """Test that leader term is updated."""
        # Set initial leader
        await state_store.set_leader("node-1", term=1)

        # Update to new term
        await state_store.set_leader("node-2", term=2)

        # Verify new leader
        leader = await state_store.get_leader()
        assert leader == ("node-2", 2)

    @pytest.mark.asyncio
    async def test_get_leader_no_leader(self, state_store):
        """Test getting leader when no leader exists."""
        leader = await state_store.get_leader()
        # Returns (None, 0) when no leader
        assert leader[0] is None


class TestFailoverEvents:
    """Test failover event persistence."""

    @pytest.mark.asyncio
    async def test_save_and_get_failover_events(self, state_store):
        """Test saving and retrieving failover events."""
        event = FailoverEvent(
            event_id="fe_001",
            event_type="node_fail",
            source_node="node-1",
            target_node="node-2",
            reason="health_check_failed",
            timestamp=datetime.now(),
            success=True,
            details={"model": "qwen2.5-7b"},
        )

        # Save event
        result = await state_store.save_failover_event(event)
        assert result is True

        # Load events
        events = await state_store.get_failover_events(limit=10)
        assert len(events) >= 1

        # Find our event
        found = any(e.event_id == "fe_001" for e in events)
        assert found is True

    @pytest.mark.asyncio
    async def test_get_failover_events_empty(self, state_store):
        """Test getting events when none exist."""
        events = await state_store.get_failover_events(limit=10)
        # May have events from previous tests, just verify it returns list
        assert isinstance(events, list)


class TestConcurrentOperations:
    """Test concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_lock_acquisition(self, state_store):
        """Test that only one node can acquire a lock under concurrency."""
        lock_acquired_by = []

        async def try_acquire(node_id: str):
            acquired = await state_store.acquire_lock("resource-1", node_id, ttl_seconds=30)
            if acquired:
                lock_acquired_by.append(node_id)
            return acquired

        # Try to acquire lock concurrently from multiple nodes
        results = await asyncio.gather(
            try_acquire("node-1"),
            try_acquire("node-2"),
            try_acquire("node-3"),
        )

        # Exactly one should succeed
        success_count = sum(1 for r in results if r)
        assert success_count == 1, f"Expected 1 lock acquisition, got {success_count}"
        assert len(lock_acquired_by) == 1

    @pytest.mark.asyncio
    async def test_concurrent_different_locks(self, state_store):
        """Test that concurrent acquisitions for different resources succeed."""
        async def try_acquire(resource: str, node_id: str):
            return await state_store.acquire_lock(resource, node_id, ttl_seconds=30)

        # Different resources should all succeed
        results = await asyncio.gather(
            try_acquire("resource-1", "node-1"),
            try_acquire("resource-2", "node-2"),
            try_acquire("resource-3", "node-3"),
        )

        # All should succeed (different resources)
        assert all(results), "All different resource locks should succeed"
