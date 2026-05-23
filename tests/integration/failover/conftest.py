"""Integration Test Fixtures for Failover Module

Provides comprehensive fixtures for integration testing:
- Real Redis (fakeredis) for state storage
- Mock cluster manager with configurable nodes
- Health checker with real thresholds
- Leader election with real distributed locks
- Replica manager with real state store
- Complete failover controller setup
"""

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
import fakeredis

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.failover.health_checker import HealthChecker
from quantumflow.failover.leader_election import LeaderElection
from quantumflow.failover.models import HealthStatus, ReplicaRole
from quantumflow.failover.policy import FailoverPolicy, HealthThresholds, ReplicaPolicy
from quantumflow.failover.replica_manager import ReplicaManager
from quantumflow.failover.state_store import NodeStateStore


# ==============================================================================
# Redis Fixtures
# ==============================================================================

@pytest.fixture
def redis_client():
    """Create a fake Redis client for testing."""
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture
def mock_redis_manager(redis_client):
    """Create a mock Redis manager that returns the fake Redis client."""
    manager = MagicMock()
    manager.get_client.return_value = redis_client
    return manager


# ==============================================================================
# Cluster Manager Fixtures
# ==============================================================================

@pytest.fixture
def cluster_nodes():
    """Create a dictionary of 3 cluster nodes for testing."""
    nodes = {}

    for i in range(1, 4):
        node_id = f"node-{i}"
        nodes[node_id] = Node(
            node_id=node_id,
            hostname=f"node-{i}.local",
            ip=f"192.168.1.{10+i}",
            port=8080,
            gpu_count=2,
            gpu_info=[
                GPUInfo(
                    gpu_id=0,
                    name="NVIDIA RTX 4090",
                    memory_total=24 * 1024,
                    memory_used=12 * 1024,
                    utilization=0.5,
                    temperature=70.0,
                ),
                GPUInfo(
                    gpu_id=1,
                    name="NVIDIA RTX 4090",
                    memory_total=24 * 1024,
                    memory_used=10 * 1024,
                    utilization=0.4,
                    temperature=65.0,
                ),
            ],
            status=NodeStatus.HEALTHY,
        )

    return nodes


@pytest.fixture
def cluster_manager(cluster_nodes):
    """Create a ClusterManager with configurable nodes."""
    manager = MagicMock(spec=ClusterManager)

    # Configure get_node
    async def mock_get_node(node_id: str):
        return cluster_nodes.get(node_id)

    # Configure get_healthy_nodes
    async def mock_get_healthy_nodes():
        return [n for n in cluster_nodes.values() if n.status == NodeStatus.HEALTHY]

    # Configure update_node_health
    async def mock_update_node_health(node_id: str, status: str, **kwargs):
        if node_id in cluster_nodes:
            old_status = cluster_nodes[node_id].status
            if status == "unhealthy":
                cluster_nodes[node_id].status = NodeStatus.UNHEALTHY
            elif status == "healthy":
                cluster_nodes[node_id].status = NodeStatus.HEALTHY
            elif status == "degraded":
                cluster_nodes[node_id].status = NodeStatus.DEGRADED
            return True
        return False

    # Configure update_model_health
    async def mock_update_model_health(node_id: str, model_name: str, health: str):
        if node_id in cluster_nodes:
            cluster_nodes[node_id].model_health[model_name] = health
            return True
        return False

    # Configure update_gpu_health
    async def mock_update_gpu_health(node_id: str, gpu_id: int, health: str, **kwargs):
        return True

    # Configure add_loaded_model
    async def mock_add_loaded_model(node_id: str, model_name: str):
        if node_id in cluster_nodes:
            if model_name not in cluster_nodes[node_id].loaded_models:
                cluster_nodes[node_id].loaded_models.append(model_name)
            return True
        return False

    # Configure remove_loaded_model
    async def mock_remove_loaded_model(node_id: str, model_name: str):
        if node_id in cluster_nodes:
            if model_name in cluster_nodes[node_id].loaded_models:
                cluster_nodes[node_id].loaded_models.remove(model_name)
            return True
        return False

    # Configure set_replica_role
    async def mock_set_replica_role(node_id: str, role: str):
        return True

    manager.get_node = mock_get_node
    manager.get_healthy_nodes = mock_get_healthy_nodes
    manager.update_node_health = mock_update_node_health
    manager.update_model_health = mock_update_model_health
    manager.update_gpu_health = mock_update_gpu_health
    manager.add_loaded_model = mock_add_loaded_model
    manager.remove_loaded_model = mock_remove_loaded_model
    manager.set_replica_role = mock_set_replica_role

    return manager


# ==============================================================================
# State Store Fixtures
# ==============================================================================

@pytest.fixture
async def state_store(mock_redis_manager):
    """Create a NodeStateStore with real Redis (fakeredis)."""
    store = NodeStateStore(redis_manager=mock_redis_manager)
    await store.initialize()
    return store


# ==============================================================================
# Policy Fixtures
# ==============================================================================

@pytest.fixture
def health_thresholds():
    """Create HealthThresholds for testing."""
    return HealthThresholds(
        gpu_temp_threshold=90.0,
        gpu_temp_warning=85.0,
        gpu_mem_threshold=0.95,
        gpu_util_threshold=0.99,
        heartbeat_timeout=30,
        failure_threshold=3,
        degraded_threshold=2,
    )


@pytest.fixture
def replica_policy():
    """Create ReplicaPolicy for testing."""
    return ReplicaPolicy(
        default_replica_count=2,
        min_replica_count=1,
        max_replica_count=5,
        sync_timeout_seconds=300,
        health_check_interval_seconds=1,  # Fast for testing
    )


@pytest.fixture
def failover_policy():
    """Create FailoverPolicy for testing."""
    return FailoverPolicy(
        auto_failover_enabled=True,
        failover_delay_seconds=0.0,
        require_manual_confirmation=False,
        multi_threshold_failover=True,
        gpu_isolation_check=True,
        auto_recovery_enabled=True,
        recovery_delay_seconds=1,  # Fast for testing
        notify_on_failover=True,
        notify_on_recovery=True,
    )


# ==============================================================================
# Component Fixtures
# ==============================================================================

@pytest.fixture
async def health_checker(cluster_manager, health_thresholds, replica_policy):
    """Create a HealthChecker with real thresholds."""
    checker = HealthChecker(
        cluster_manager=cluster_manager,
        health_thresholds=health_thresholds,
        replica_policy=replica_policy,
    )
    return checker


@pytest.fixture
async def leader_election(cluster_manager, state_store, replica_policy):
    """Create a LeaderElection with real state store."""
    election = LeaderElection(
        node_id="node-1",
        cluster_manager=cluster_manager,
        state_store=state_store,
        replica_policy=replica_policy,
    )
    return election


@pytest.fixture
async def replica_manager(cluster_manager, state_store, replica_policy):
    """Create a ReplicaManager with real state store."""
    manager = ReplicaManager(
        cluster_manager=cluster_manager,
        state_store=state_store,
        replica_policy=replica_policy,
    )
    return manager


@pytest.fixture
async def failover_controller(
    cluster_manager,
    state_store,
    replica_manager,
    health_checker,
    failover_policy,
):
    """Create a complete FailoverController with all real components."""
    from quantumflow.failover.controller import FailoverController

    controller = FailoverController(
        cluster_manager=cluster_manager,
        state_store=state_store,
        replica_manager=replica_manager,
        health_checker=health_checker,
        node_id="node-1",
        failover_policy=failover_policy,
    )
    return controller


# ==============================================================================
# Cluster Setup with Loaded Models
# ==============================================================================

@pytest.fixture
async def cluster_with_models(cluster_manager, cluster_nodes):
    """Set up a cluster with models loaded on nodes."""
    # node-1 loads primary model
    cluster_nodes["node-1"].loaded_models = ["qwen2.5-7b"]
    cluster_nodes["node-1"].model_health = {"qwen2.5-7b": "healthy"}

    # node-2 loads replica
    cluster_nodes["node-2"].loaded_models = ["qwen2.5-7b"]
    cluster_nodes["node-2"].model_health = {"qwen2.5-7b": "healthy"}

    return cluster_manager, cluster_nodes


# ==============================================================================
# Test Configuration
# ==============================================================================

@pytest.fixture(autouse=True)
async def reset_globals():
    """Reset any global state before each test."""
    # Clean up any existing tasks
    yield
    # Cancel any lingering tasks
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
