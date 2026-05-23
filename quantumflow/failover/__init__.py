"""QuantumFlow 容灾模块

提供企业级容灾能力：
- 故障检测与转移
- 模型副本管理
- Leader 选举与脑裂防护
"""

from quantumflow.failover.controller import FailoverController
from quantumflow.failover.health_checker import HealthChecker, GPUHealthResult, HealthCheckResult
from quantumflow.failover.leader_election import LeaderElection
from quantumflow.failover.models import (
    FailoverEvent,
    FailoverState,
    HealthStatus,
    ModelReplica,
    NodeFailoverState,
    ReplicaRole,
)
from quantumflow.failover.policy import FailoverPolicy, HealthThresholds, ReplicaPolicy
from quantumflow.failover.replica_manager import ReplicaManager
from quantumflow.failover.state_store import NodeStateStore

__all__ = [
    # Models
    "ReplicaRole",
    "FailoverState",
    "HealthStatus",
    "NodeFailoverState",
    "ModelReplica",
    "FailoverEvent",
    # State Store
    "NodeStateStore",
    # Core Classes
    "FailoverController",
    "HealthChecker",
    "LeaderElection",
    "ReplicaManager",
    # Health Check Results
    "GPUHealthResult",
    "HealthCheckResult",
    # Policy Classes
    "FailoverPolicy",
    "HealthThresholds",
    "ReplicaPolicy",
]
