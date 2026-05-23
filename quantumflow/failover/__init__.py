"""QuantumFlow 容灾模块

提供企业级容灾能力：
- 故障检测与转移
- 模型副本管理
- Leader 选举与脑裂防护
"""

from quantumflow.failover.models import (
    FailoverEvent,
    FailoverState,
    HealthStatus,
    ModelReplica,
    NodeFailoverState,
    ReplicaRole,
)
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
]
