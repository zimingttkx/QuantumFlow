"""Performance and Benchmark Tests for Failover Module

Tests performance characteristics:
1. Lock operation latency
2. State operations latency
3. Concurrent throughput
"""

import asyncio
import time
from datetime import datetime

import pytest

from quantumflow.failover.models import NodeFailoverState, ReplicaRole, FailoverState, HealthStatus


class TestLockPerformance:
    """Test lock operation performance."""

    @pytest.mark.asyncio
    async def test_lock_acquire_latency(self, state_store):
        """Measure lock acquisition latency."""
        latencies = []

        for _ in range(50):
            start = time.perf_counter()
            await state_store.acquire_lock("perf-lock", "node-1", ttl_seconds=30)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
            await state_store.release_lock("perf-lock", "node-1")

        latencies.sort()
        p50 = latencies[25]
        p95 = latencies[47]

        assert p50 < 10, f"p50 latency {p50:.2f}ms too high"
        assert p95 < 20, f"p95 latency {p95:.2f}ms too high"

    @pytest.mark.asyncio
    async def test_lock_release_latency(self, state_store):
        """Measure lock release latency."""
        await state_store.acquire_lock("perf-lock", "node-1", ttl_seconds=30)

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            await state_store.release_lock("perf-lock", "node-1")
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            # Re-acquire for next iteration
            await state_store.acquire_lock("perf-lock", "node-1", ttl_seconds=30)

        latencies.sort()
        p50 = latencies[25]

        assert p50 < 10, f"p50 latency {p50:.2f}ms too high"


class TestLeaderPerformance:
    """Test leader operation performance."""

    @pytest.mark.asyncio
    async def test_set_leader_latency(self, state_store):
        """Measure set leader latency."""
        latencies = []

        for i in range(100):
            start = time.perf_counter()
            await state_store.set_leader(f"node-{i % 3 + 1}", term=i)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p50 = latencies[50]

        assert p50 < 10, f"p50 latency {p50:.2f}ms too high"

    @pytest.mark.asyncio
    async def test_get_leader_latency(self, state_store):
        """Measure get leader latency."""
        await state_store.set_leader("node-1", term=1)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await state_store.get_leader()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p50 = latencies[50]

        assert p50 < 10, f"p50 latency {p50:.2f}ms too high"


class TestStateStorePerformance:
    """Test state store operation performance."""

    @pytest.mark.asyncio
    async def test_save_node_state_latency(self, state_store):
        """Measure node state save latency."""
        latencies = []

        for i in range(100):
            state = NodeFailoverState(
                node_id=f"perf-node-{i}",
                role=ReplicaRole.SECONDARY,
                state=FailoverState.NORMAL,
                health=HealthStatus.HEALTHY,
                term=1,
                last_heartbeat=datetime.now(),
            )
            start = time.perf_counter()
            await state_store.save_node_state(state)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p50 = latencies[50]

        assert p50 < 10, f"p50 latency {p50:.2f}ms too high"

    @pytest.mark.asyncio
    async def test_load_node_state_latency(self, state_store):
        """Measure node state load latency."""
        # Create state first
        state = NodeFailoverState(
            node_id="perf-node-load",
            role=ReplicaRole.SECONDARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
        )
        await state_store.save_node_state(state)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await state_store.load_node_state("perf-node-load")
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p50 = latencies[50]

        assert p50 < 10, f"p50 latency {p50:.2f}ms too high"


class TestThroughput:
    """Test throughput characteristics."""

    @pytest.mark.asyncio
    async def test_lock_ops_per_second(self, state_store):
        """Calculate lock operations per second."""
        duration = 1.0  # 1 second
        operation_count = 0

        start_time = time.perf_counter()
        end_time = start_time + duration

        while time.perf_counter() < end_time:
            await state_store.acquire_lock(f"throughput-lock-{operation_count}", "node-1", ttl_seconds=30)
            await state_store.release_lock(f"throughput-lock-{operation_count}", "node-1")
            operation_count += 1

        ops_per_second = operation_count / duration
        assert ops_per_second > 50, f"Throughput {ops_per_second:.0f} ops/s too low"

    @pytest.mark.asyncio
    async def test_concurrent_lock_throughput(self, state_store):
        """Test concurrent lock throughput."""
        num_locks = 20

        async def lock_op(i: int):
            resource = f"concurrent-lock-{i}"
            acquired = await state_store.acquire_lock(resource, "node-1", ttl_seconds=30)
            if acquired:
                await state_store.release_lock(resource, "node-1")
            return acquired

        start_time = time.perf_counter()
        results = await asyncio.gather(*[lock_op(i) for i in range(num_locks)])
        end_time = time.perf_counter()

        total_time = end_time - start_time
        ops_per_second = (num_locks * 2) / total_time  # acquire + release

        assert all(results), "All lock acquisitions should succeed"
        assert ops_per_second > 100, f"Throughput {ops_per_second:.0f} ops/s too low"
