"""容灾模块数据模型"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ReplicaRole(str, Enum):
    """副本角色"""

    PRIMARY = "primary"
    SECONDARY = "secondary"


class FailoverState(str, Enum):
    """故障转移状态"""

    NORMAL = "normal"
    DEGRADED = "degraded"
    FAILOVER = "failover"
    RECOVERING = "recovering"


class HealthStatus(str, Enum):
    """健康状态"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class NodeFailoverState:
    """节点故障转移状态"""

    node_id: str
    role: ReplicaRole
    state: FailoverState
    health: HealthStatus
    term: int
    last_heartbeat: datetime
    failure_reasons: list[str] = field(default_factory=list)
    gpu_status: dict[str, HealthStatus] = field(default_factory=dict)
    model_status: dict[str, HealthStatus] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "state": self.state.value,
            "health": self.health.value,
            "term": self.term,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "failure_reasons": self.failure_reasons,
            "gpu_status": {k: v.value for k, v in self.gpu_status.items()},
            "model_status": {k: v.value for k, v in self.model_status.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeFailoverState":
        """从字典创建"""
        return cls(
            node_id=data["node_id"],
            role=ReplicaRole(data["role"]),
            state=FailoverState(data["state"]),
            health=HealthStatus(data["health"]),
            term=data["term"],
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            failure_reasons=data.get("failure_reasons", []),
            gpu_status={
                k: HealthStatus(v) for k, v in data.get("gpu_status", {}).items()
            },
            model_status={
                k: HealthStatus(v) for k, v in data.get("model_status", {}).items()
            },
        )


@dataclass
class ModelReplica:
    """模型副本信息"""

    model_name: str
    model_path: str
    primary_node: str
    secondary_nodes: list[str]
    replica_count: int
    sync_state: str  # synced, syncing, degraded
    last_sync_at: datetime
    checksum: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "primary_node": self.primary_node,
            "secondary_nodes": self.secondary_nodes,
            "replica_count": self.replica_count,
            "sync_state": self.sync_state,
            "last_sync_at": self.last_sync_at.isoformat(),
            "checksum": self.checksum,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelReplica":
        """从字典创建"""
        return cls(
            model_name=data["model_name"],
            model_path=data["model_path"],
            primary_node=data["primary_node"],
            secondary_nodes=data.get("secondary_nodes", []),
            replica_count=data["replica_count"],
            sync_state=data["sync_state"],
            last_sync_at=datetime.fromisoformat(data["last_sync_at"]),
            checksum=data["checksum"],
            version=data["version"],
        )


@dataclass
class FailoverEvent:
    """故障转移事件"""

    event_id: str
    event_type: str  # node_fail, gpu_fail, model_fail, timeout, manual
    source_node: str
    target_node: str | None = None
    target_nodes: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    timestamp: datetime | None = None
    success: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "target_nodes": self.target_nodes,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailoverEvent":
        """从字典创建"""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            source_node=data["source_node"],
            target_node=data.get("target_node"),
            target_nodes=data.get("target_nodes", {}),
            reason=data["reason"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success=data["success"],
            details=data.get("details", {}),
        )
