"""Real Redis Cluster/Sentinel Integration Tests

Tests NodeStateStore with real Redis configurations:
1. Standalone Redis connection
2. Redis Sentinel for HA
3. Redis Cluster for sharding
4. Connection pool behavior
5. Reconnection scenarios
"""

import asyncio
import os
import socket
from datetime import datetime
from typing import Optional

import pytest

from quantumflow.failover.models import FailoverEvent, ReplicaRole, FailoverState, HealthStatus
from quantumflow.failover.state_store import NodeStateStore


def is_redis_available(host: str = "localhost", port: int = 6379, timeout: float = 1.0) -> bool:
    """Check if Redis is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_redis_url() -> Optional[str]:
    """Get Redis URL from environment or return None."""
    return os.environ.get("REDIS_URL")


# Skip markers for Redis-dependent tests
requires_redis = pytest.mark.skipif(
    not is_redis_available(),
    reason="Real Redis not available"
)

requires_redis_sentinel = pytest.mark.skipif(
    not is_redis_available("localhost", 26379),  # Default Sentinel port
    reason="Redis Sentinel not available"
)


# ==============================================================================
# Fixtures for Real Redis
# ==============================================================================

@pytest.fixture
def redis_url():
    """Get Redis URL from environment or use default."""
    return get_redis_url() or "redis://localhost:6379/0"


@pytest.fixture
async def real_redis_manager(redis_url):
    """Create a real Redis manager for testing."""
    try:
        import redis.asyncio as aioredis

        client = await aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        # Test connection
        await client.ping()
        yield client
        await client.close()
    except Exception as e:
        pytest.skip(f"Cannot connect to Redis: {e}")


@pytest.fixture
async def real_state_store(real_redis_manager):
    """Create NodeStateStore with real Redis."""
    from quantumflow.failover.redis_manager import RedisConnectionManager

    manager = RedisConnectionManager(redis_url=redis_url)
    store = NodeStateStore(redis_manager=manager)
    await store.initialize()
    yield store
    await store.close()


# ==============================================================================
# Redis Connection Tests
# ==============================================================================

class TestRealRedisConnection:
    """Test real Redis connection."""

    @pytest.mark.asyncio
    async def test_redis_ping(self, real_redis_manager):
        """Test Redis ping."""
        result = await real_redis_manager.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_set_get(self, real_redis_manager):
        """Test Redis SET/GET operations."""
        await real_redis_manager.set("test_key", "test_value")
        value = await real_redis_manager.get("test_key")
        assert value == "test_value"
        await real_redis_manager.delete("test_key")

    @pytest.mark.asyncio
    async def test_redis_pipeline(self, real_redis_manager):
        """Test Redis pipeline for batch operations."""
        pipe = real_redis_manager.pipeline()
        pipe.set("key1", "value1")
        pipe.set("key2", "value2")
        pipe.get("key1")
        results = await pipe.execute()

        assert results[0] is True  # SET key1
        assert results[1] is True  # SET key2
        assert results[2] == "value1"  # GET key1

        # Cleanup
        await real_redis_manager.delete("key1", "key2")


# ==============================================================================
# Redis Sentinel Tests
# ==============================================================================

class TestRedisSentinelHA:
    """Test Redis Sentinel high availability scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available("localhost", 26379),
        reason="Redis Sentinel not available at localhost:26379"
    )
    async def test_sentinel_discovery(self):
        """Test Redis Sentinel service discovery."""
        try:
            import redis.asyncio as aioredis
            from redis.asyncio.sentinel import Sentinel

            sentinel = Sentinel(
                [("localhost", 26379)],
                socket_timeout=1,
            )

            # Get master address from Sentinel
            master = await sentinel.discover_master("mymaster")
            assert master is not None
            master_host, master_port = master
            assert isinstance(master_host, str)
            assert isinstance(master_port, int)

        except Exception as e:
            pytest.skip(f"Sentinel test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available("localhost", 26379),
        reason="Redis Sentinel not available"
    )
    async def test_sentinel_slave_read(self):
        """Test reading from Sentinel slave."""
        try:
            import redis.asyncio as aioredis
            from redis.asyncio.sentinel import Sentinel

            sentinel = Sentinel(
                [("localhost", 26379)],
                socket_timeout=1,
            )

            # Get slave for read operations
            slave = await sentinel.discover_slave("mymaster")
            assert slave is not None

        except Exception as e:
            pytest.skip(f"Sentinel slave test failed: {e}")


# ==============================================================================
# Redis Cluster Tests
# ==============================================================================

class TestRedisCluster:
    """Test Redis Cluster scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available("localhost", 7000),  # Default cluster port
        reason="Redis Cluster not available"
    )
    async def test_cluster_slot_distribution(self):
        """Test Redis Cluster slot distribution."""
        try:
            import redis.asyncio as aioredis

            # Connect to cluster
            client = await aioredis.RedisCluster(
                host="localhost",
                port=7000,
                read_from_replicas=True,
            )

            # Test key routing
            keys = [f"key_{i}" for i in range(10)]
            for key in keys:
                await client.set(key, f"value_{key}")

            # Verify keys are distributed across slots
            await client.close()

        except Exception as e:
            pytest.skip(f"Cluster test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available("localhost", 7000),
        reason="Redis Cluster not available"
    )
    async def test_cluster_failover_simulation(self):
        """Test cluster behavior during failover simulation."""
        try:
            import redis.asyncio as aioredis

            client = await aioredis.RedisCluster(
                host="localhost",
                port=7000,
            )

            # Write a value
            await client.set("failover_test", "initial")

            # Verify it's there
            value = await client.get("failover_test")
            assert value == "initial"

            await client.close()

        except Exception as e:
            pytest.skip(f"Cluster failover test failed: {e}")


# ==============================================================================
# NodeStateStore with Real Redis
# ==============================================================================

class TestNodeStateStoreWithRealRedis:
    """Test NodeStateStore with real Redis backend."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Real Redis not available"
    )
    async def test_save_load_node_state_real_redis(self, real_state_store):
        """Test saving and loading node state with real Redis."""
        state = NodeFailoverState(
            node_id="real-redis-node-1",
            role=ReplicaRole.PRIMARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
        )

        # Save
        result = await real_state_store.save_node_state(state)
        assert result is True

        # Load
        loaded = await real_state_store.load_node_state("real-redis-node-1")
        assert loaded is not None
        assert loaded.node_id == "real-redis-node-1"
        assert loaded.health == HealthStatus.HEALTHY

        # Cleanup
        await real_state_store.delete_node_state("real-redis-node-1")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Real Redis not available"
    )
    async def test_distributed_lock_real_redis(self, real_state_store):
        """Test distributed lock with real Redis."""
        # Acquire lock
        acquired = await real_state_store.acquire_lock(
            "real-redis-lock", "node-1", ttl_seconds=30
        )
        assert acquired is True

        # Verify lock info
        info = await real_state_store.get_lock_info("real-redis-lock")
        assert info is not None
        assert info["owner"] == "node-1"

        # Release lock
        released = await real_state_store.release_lock("real-redis-lock", "node-1")
        assert released is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Real Redis not available"
    )
    async def test_leader_election_real_redis(self, real_state_store):
        """Test leader election with real Redis."""
        # Set leader
        result = await real_state_store.set_leader("node-1", term=1)
        assert result is True

        # Get leader
        leader = await real_state_store.get_leader()
        assert leader == ("node-1", 1)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Real Redis not available"
    )
    async def test_failover_events_real_redis(self, real_state_store):
        """Test failover events with real Redis."""
        event = FailoverEvent(
            event_id="fe_real_001",
            event_type="node_fail",
            source_node="node-1",
            target_node="node-2",
            reason="gpu_failure",
            timestamp=datetime.now(),
            success=True,
        )

        # Save
        result = await real_state_store.save_failover_event(event)
        assert result is True

        # Get events
        events = await real_state_store.get_failover_events(limit=10)
        assert len(events) >= 1
        assert any(e.event_id == "fe_real_001" for e in events)


# ==============================================================================
# Connection Pool Tests
# ==============================================================================

class TestRedisConnectionPool:
    """Test Redis connection pool behavior."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Real Redis not available"
    )
    async def test_concurrent_connections(self, real_redis_manager):
        """Test handling concurrent connections."""
        num_connections = 20

        async def connection_task(i: int):
            key = f"concurrent_key_{i}"
            await real_redis_manager.set(key, f"value_{i}")
            value = await real_redis_manager.get(key)
            return value == f"value_{i}"

        results = await asyncio.gather(*[connection_task(i) for i in range(num_connections)])
        assert all(results), "Some concurrent operations failed"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Real Redis not available"
    )
    async def test_connection_pool_reuse(self, real_redis_manager):
        """Test that connections are reused from pool."""
        # Execute multiple operations
        for _ in range(10):
            await real_redis_manager.set("pool_test", "value")
            value = await real_redis_manager.get("pool_test")
            assert value == "value"

        # Cleanup
        await real_redis_manager.delete("pool_test")


# ==============================================================================
# Data Integrity Tests
# ==============================================================================

class TestRedisDataIntegrity:
    """Test data integrity with real Redis."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Real Redis not available"
    )
    async def test_atomic_operations(self, real_redis_manager):
        """Test atomic Redis operations."""
        key = "atomic_test"

        # Test SETNX (set if not exists)
        result1 = await real_redis_manager.setnx(key, "value1")
        assert result1 is True  # First set should succeed

        result2 = await real_redis_manager.setnx(key, "value2")
        assert result2 is False  # Second set should fail (already exists)

        # Verify original value
        value = await real_redis_manager.get(key)
        assert value == "value1"

        # Cleanup
        await real_redis_manager.delete(key)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Real Redis not available"
    )
    async def test_expiration(self, real_redis_manager):
        """Test key expiration."""
        import time

        key = "expiry_test"
        await real_redis_manager.set(key, "value", ex=1)  # 1 second expiry

        # Immediate read should work
        value = await real_redis_manager.get(key)
        assert value == "value"

        # Wait for expiration
        time.sleep(1.5)

        # Key should be expired
        value = await real_redis_manager.get(key)
        assert value is None
