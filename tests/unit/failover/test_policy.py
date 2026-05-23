"""Policy 配置测试"""

import pytest

from quantumflow.failover.policy import (
    FailoverPolicy,
    HealthThresholds,
    ReplicaPolicy,
)


class TestHealthThresholds:
    """HealthThresholds 测试"""

    def test_default_values(self):
        """测试默认值"""
        thresholds = HealthThresholds()

        assert thresholds.gpu_temp_threshold == 90.0
        assert thresholds.gpu_mem_threshold == 0.95
        assert thresholds.gpu_util_threshold == 0.99
        assert thresholds.heartbeat_timeout == 30
        assert thresholds.failure_threshold == 3

    def test_custom_values(self):
        """测试自定义值"""
        thresholds = HealthThresholds(
            gpu_temp_threshold=85.0,
            gpu_mem_threshold=0.90,
            heartbeat_timeout=60,
            failure_threshold=5,
        )

        assert thresholds.gpu_temp_threshold == 85.0
        assert thresholds.gpu_mem_threshold == 0.90
        assert thresholds.heartbeat_timeout == 60
        assert thresholds.failure_threshold == 5

    def test_to_dict(self):
        """测试转换为字典"""
        thresholds = HealthThresholds()
        data = thresholds.to_dict()

        assert isinstance(data, dict)
        assert "gpu_temp_threshold" in data
        assert "gpu_mem_threshold" in data
        assert "heartbeat_timeout" in data
        assert "failure_threshold" in data


class TestReplicaPolicy:
    """ReplicaPolicy 测试"""

    def test_default_values(self):
        """测试默认值"""
        policy = ReplicaPolicy()

        assert policy.default_replica_count == 2
        assert policy.min_replica_count == 1
        assert policy.max_replica_count == 5
        assert policy.sync_interval_seconds == 30
        assert policy.health_check_interval_seconds == 10
        assert policy.auto_promote_secondary is True

    def test_custom_values(self):
        """测试自定义值"""
        policy = ReplicaPolicy(
            default_replica_count=3,
            min_replica_count=2,
            max_replica_count=6,
            auto_promote_secondary=False,
        )

        assert policy.default_replica_count == 3
        assert policy.min_replica_count == 2
        assert policy.max_replica_count == 6
        assert policy.auto_promote_secondary is False

    def test_to_dict(self):
        """测试转换为字典"""
        policy = ReplicaPolicy()
        data = policy.to_dict()

        assert isinstance(data, dict)
        assert data["default_replica_count"] == 2
        assert data["min_replica_count"] == 1
        assert data["auto_promote_secondary"] is True


class TestFailoverPolicy:
    """FailoverPolicy 测试"""

    def test_default_values(self):
        """测试默认值"""
        policy = FailoverPolicy()

        assert policy.auto_failover_enabled is True
        assert policy.failover_delay_seconds == 0.0
        assert policy.require_manual_confirmation is False
        assert policy.auto_recovery_enabled is True
        assert policy.notify_on_failover is True

    def test_custom_values(self):
        """测试自定义值"""
        policy = FailoverPolicy(
            auto_failover_enabled=False,
            require_manual_confirmation=True,
            failover_delay_seconds=5.0,
        )

        assert policy.auto_failover_enabled is False
        assert policy.require_manual_confirmation is True
        assert policy.failover_delay_seconds == 5.0

    def test_to_dict(self):
        """测试转换为字典"""
        policy = FailoverPolicy()
        data = policy.to_dict()

        assert isinstance(data, dict)
        assert data["auto_failover_enabled"] is True
        assert data["require_manual_confirmation"] is False
        assert data["notify_on_failover"] is True
