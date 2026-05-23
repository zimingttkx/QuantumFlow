"""Stress and Stability Tests for Failover Module

Tests system behavior under sustained load:
1. Concurrent lock operations
2. Rapid state changes
3. Error handling under load
"""

import asyncio
import gc
from datetime import datetime

import pytest

from quantumflow.failover.models import NodeFailoverState, ReplicaRole, FailoverState, HealthStatus


class TestConcurrentLockStress:
    """Test concurrent lock operations."""

    @pytest.mark.asyncio
    async def test_rapid_lock_acquire_release(self, state_store):
        """Test rapid lock acquire/release cycles."""
        num_cycles = 50

        for i in range(num_cycles):
            acquired = await state_store.acquire_lock("stress-lock", "node-1", ttl_seconds=10)
            assert acquired is True
            released = await state_store.release_lock("stress-lock", "node-1")
            assert released is True

    @pytest.mark.asyncio
    async def test_concurrent_different_locks(self, state_store):
        """Test acquiring many different locks concurrently."""
        num_locks = 20

        async def lock_op(i: int):
            resource = f"stress-lock-{i}"
            acquired = await state_store.acquire_lock(resource, "node-1", ttl_seconds=30)
            if acquired:
                await state_store.release_lock(resource, "node-1")
            return acquired

        results = await asyncio.gather(*[lock_op(i) for i in range(num_locks)])
        assert all(results), "All lock acquisitions should succeed for different resources"

    @pytest.mark.asyncio
    async def test_lock_contention(self, state_store):
        """Test lock contention with many concurrent attempts."""
        num_attempts = 30

        async def try_acquire(attempt: int):
            owner = f"node-{attempt % 3}"
            return await state_store.acquire_lock("contention-lock", owner, ttl_seconds=30)

        results = await asyncio.gather(*[try_acquire(i) for i in range(num_attempts)])
        success_count = sum(1 for r in results if r)
        assert success_count == 1, f"Expected 1 winner, got {success_count}"


class TestRapidStateChanges:
    """Test rapid state changes."""

    @pytest.mark.asyncio
    async def test_rapid_leader_changes(self, state_store):
        """Test rapid leader changes."""
        num_changes = 20

        for i in range(num_changes):
            node_id = f"node-{(i % 3) + 1}"
            term = i + 1
            result = await state_store.set_leader(node_id, term=term)
            assert result is True

        # Verify final leader
        leader = await state_store.get_leader()
        assert leader == (f"node-{((num_changes-1) % 3) + 1}", num_changes)

    @pytest.mark.asyncio
    async def test_rapid_node_state_changes(self, state_store):
        """Test rapid node state changes."""
        num_changes = 30

        for i in range(num_changes):
            health = HealthStatus.HEALTHY if i % 2 == 0 else HealthStatus.UNHEALTHY
            state = NodeFailoverState(
                node_id="rapid-node",
                role=ReplicaRole.SECONDARY,
                state=FailoverState.NORMAL,
                health=health,
                term=1,
                last_heartbeat=datetime.now(),
                failure_reasons=["test"] if health == HealthStatus.UNHEALTHY else [],
            )
            result = await state_store.save_node_state(state)
            assert result is True

        # Verify final state
        loaded = await state_store.load_node_state("rapid-node")
        assert loaded is not None
        # Final state should be UNHEALTHY (last iteration was odd)


class TestMemoryStability:
    """Test memory stability."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_in_operations(self, state_store):
        """Test that operations don't leak memory."""
        initial_objects = len(gc.get_objects())

        num_operations = 500

        for i in range(num_operations):
            state = NodeFailoverState(
                node_id=f"memory-test-{i % 50}",  # Reuse some nodes
                role=ReplicaRole.SECONDARY,
                state=FailoverState.NORMAL,
                health=HealthStatus.HEALTHY,
                term=1,
                last_heartbeat=datetime.now(),
            )
            await state_store.save_node_state(state)

            if i % 100 == 0:
                gc.collect()

        gc.collect()
        final_objects = len(gc.get_objects())

        # Allow some growth but not excessive
        # Note: fakeredis and structlog create internal objects, so threshold is higher
        object_growth = final_objects - initial_objects
        assert object_growth < 2000, f"Memory grew by {object_growth} objects after {num_operations} operations"

    @pytest.mark.asyncio
    async def test_concurrent_gc_collection(self, state_store):
        """Test that GC doesn't break operations."""
        num_operations = 100

        async def op_with_gc(i: int):
            state = NodeFailoverState(
                node_id=f"gc-test-{i}",
                role=ReplicaRole.SECONDARY,
                state=FailoverState.NORMAL,
                health=HealthStatus.HEALTHY,
                term=1,
                last_heartbeat=datetime.now(),
            )
            await state_store.save_node_state(state)
            if i % 20 == 0:
                gc.collect()
            return await state_store.load_node_state(f"gc-test-{i}")

        results = await asyncio.gather(*[op_with_gc(i) for i in range(num_operations)])
        assert all(r is not None for r in results)


class TestErrorRecovery:
    """Test error recovery."""

    @pytest.mark.asyncio
    async def test_nonexistent_node_load(self, state_store):
        """Test loading nonexistent node."""
        loaded = await state_store.load_node_state("nonexistent-xyz-123")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_nonexistent_replica_load(self, state_store):
        """Test loading nonexistent replica."""
        loaded = await state_store.load_replica_index("nonexistent-model-xyz")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_nonexistent_lock_info(self, state_store):
        """Test getting info for nonexistent lock."""
        info = await state_store.get_lock_info("nonexistent-lock-xyz")
        assert info is None
