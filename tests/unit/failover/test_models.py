"""容灾数据模型测试"""

import pytest
from datetime import datetime

from quantumflow.failover.models import (
    FailoverEvent,
    FailoverState,
    HealthStatus,
    ModelReplica,
    NodeFailoverState,
    ReplicaRole,
)


class TestReplicaRole:
    """ReplicaRole 枚举测试"""

    def test_primary_value(self):
        """测试 PRIMARY 值"""
        assert ReplicaRole.PRIMARY.value == "primary"

    def test_secondary_value(self):
        """测试 SECONDARY 值"""
        assert ReplicaRole.SECONDARY.value == "secondary"

    def test_is_string_enum(self):
        """测试是字符串枚举"""
        assert isinstance(ReplicaRole.PRIMARY, str)
        assert ReplicaRole.PRIMARY == "primary"


class TestFailoverState:
    """FailoverState 枚举测试"""

    def test_state_values(self):
        """测试状态值"""
        assert FailoverState.NORMAL.value == "normal"
        assert FailoverState.DEGRADED.value == "degraded"
        assert FailoverState.FAILOVER.value == "failover"
        assert FailoverState.RECOVERING.value == "recovering"

    def test_all_states_accounted(self):
        """测试所有状态都被定义"""
        states = [s.value for s in FailoverState]
        assert len(states) == 4


class TestHealthStatus:
    """HealthStatus 枚举测试"""

    def test_status_values(self):
        """测试状态值"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_is_string_enum(self):
        """测试是字符串枚举"""
        assert isinstance(HealthStatus.HEALTHY, str)
        assert HealthStatus.HEALTHY == "healthy"


class TestNodeFailoverState:
    """NodeFailoverState 数据类测试"""

    def test_creation(self):
        """测试创建"""
        state = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.PRIMARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
        )

        assert state.node_id == "node-1"
        assert state.role == ReplicaRole.PRIMARY
        assert state.state == FailoverState.NORMAL
        assert state.health == HealthStatus.HEALTHY
        assert state.term == 1
        assert state.failure_reasons == []
        assert state.gpu_status == {}
        assert state.model_status == {}

    def test_to_dict(self):
        """测试转换为字典"""
        now = datetime.now()
        state = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.PRIMARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=now,
            failure_reasons=["test_reason"],
            gpu_status={"0": HealthStatus.HEALTHY},
        )

        data = state.to_dict()

        assert data["node_id"] == "node-1"
        assert data["role"] == "primary"
        assert data["state"] == "normal"
        assert data["health"] == "healthy"
        assert data["term"] == 1
        assert data["failure_reasons"] == ["test_reason"]
        assert data["gpu_status"] == {"0": "healthy"}

    def test_from_dict(self):
        """测试从字典创建"""
        now = datetime.now()
        data = {
            "node_id": "node-1",
            "role": "primary",
            "state": "normal",
            "health": "healthy",
            "term": 1,
            "last_heartbeat": now.isoformat(),
            "failure_reasons": [],
            "gpu_status": {},
            "model_status": {},
        }

        state = NodeFailoverState.from_dict(data)

        assert state.node_id == "node-1"
        assert state.role == ReplicaRole.PRIMARY
        assert state.state == FailoverState.NORMAL
        assert state.health == HealthStatus.HEALTHY
        assert state.term == 1

    def test_round_trip(self):
        """测试往返转换"""
        now = datetime.now()
        original = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.SECONDARY,
            state=FailoverState.DEGRADED,
            health=HealthStatus.DEGRADED,
            term=5,
            last_heartbeat=now,
            failure_reasons=["gpu_overheat"],
            gpu_status={"0": HealthStatus.DEGRADED, "1": HealthStatus.HEALTHY},
        )

        # 转换为字典再转回
        data = original.to_dict()
        restored = NodeFailoverState.from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.role == original.role
        assert restored.state == original.state
        assert restored.health == original.health
        assert restored.term == original.term


class TestModelReplica:
    """ModelReplica 数据类测试"""

    def test_creation(self):
        """测试创建"""
        now = datetime.now()
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2", "node-3"],
            replica_count=3,
            sync_state="synced",
            last_sync_at=now,
            checksum="abc123",
            version=1,
        )

        assert replica.model_name == "qwen2.5-7b"
        assert replica.primary_node == "node-1"
        assert len(replica.secondary_nodes) == 2
        assert replica.replica_count == 3
        assert replica.sync_state == "synced"
        assert replica.checksum == "abc123"
        assert replica.version == 1

    def test_to_dict(self):
        """测试转换为字典"""
        now = datetime.now()
        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2"],
            replica_count=2,
            sync_state="synced",
            last_sync_at=now,
            checksum="abc123",
            version=1,
        )

        data = replica.to_dict()

        assert data["model_name"] == "qwen2.5-7b"
        assert data["primary_node"] == "node-1"
        assert data["secondary_nodes"] == ["node-2"]
        assert data["sync_state"] == "synced"

    def test_from_dict(self):
        """测试从字典创建"""
        now = datetime.now()
        data = {
            "model_name": "qwen2.5-7b",
            "model_path": "/models/qwen2.5-7b",
            "primary_node": "node-1",
            "secondary_nodes": ["node-2", "node-3"],
            "replica_count": 3,
            "sync_state": "syncing",
            "last_sync_at": now.isoformat(),
            "checksum": "def456",
            "version": 2,
        }

        replica = ModelReplica.from_dict(data)

        assert replica.model_name == "qwen2.5-7b"
        assert replica.primary_node == "node-1"
        assert len(replica.secondary_nodes) == 2
        assert replica.sync_state == "syncing"
        assert replica.version == 2


class TestFailoverEvent:
    """FailoverEvent 数据类测试"""

    def test_creation(self):
        """测试创建"""
        now = datetime.now()
        event = FailoverEvent(
            event_id="fe_123456",
            event_type="node_fail",
            source_node="node-1",
            target_node="node-2",
            reason="heartbeat_timeout",
            timestamp=now,
            success=True,
            details={"model": "qwen2.5-7b"},
        )

        assert event.event_id == "fe_123456"
        assert event.event_type == "node_fail"
        assert event.source_node == "node-1"
        assert event.target_node == "node-2"
        assert event.reason == "heartbeat_timeout"
        assert event.success is True
        assert event.details == {"model": "qwen2.5-7b"}

    def test_creation_without_target(self):
        """测试创建（无目标节点）"""
        now = datetime.now()
        event = FailoverEvent(
            event_id="fe_789",
            event_type="manual",
            source_node="node-1",
            target_node=None,
            reason="manual failover",
            timestamp=now,
            success=False,
        )

        assert event.target_node is None
        assert event.success is False

    def test_to_dict(self):
        """测试转换为字典"""
        now = datetime.now()
        event = FailoverEvent(
            event_id="fe_abc",
            event_type="gpu_fail",
            source_node="node-1",
            target_node="node-2",
            reason="temperature_exceeded",
            timestamp=now,
            success=True,
            details={"gpu_id": 0},
        )

        data = event.to_dict()

        assert data["event_id"] == "fe_abc"
        assert data["event_type"] == "gpu_fail"
        assert data["source_node"] == "node-1"
        assert data["success"] is True
        assert data["details"] == {"gpu_id": 0}

    def test_from_dict(self):
        """测试从字典创建"""
        now = datetime.now()
        data = {
            "event_id": "fe_xyz",
            "event_type": "model_fail",
            "source_node": "node-2",
            "target_node": "node-3",
            "reason": "load_failed",
            "timestamp": now.isoformat(),
            "success": False,
            "details": {},
        }

        event = FailoverEvent.from_dict(data)

        assert event.event_id == "fe_xyz"
        assert event.event_type == "model_fail"
        assert event.source_node == "node-2"
        assert event.target_node == "node-3"
        assert event.success is False
