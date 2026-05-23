"""FailoverController Integration Tests

Tests FailoverController with real component interactions:
1. Event handling
2. Health check integration
3. Status reporting
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from quantumflow.cluster.manager import NodeStatus


class TestEventHandling:
    """Test event emission and handling."""

    @pytest.mark.asyncio
    async def test_register_event_handler(self, failover_controller):
        """Test registering an event handler."""
        handler_called = []

        def handler(data):
            handler_called.append(data)

        failover_controller.on("test_event", handler)

        assert "test_event" in failover_controller._event_handlers
        assert len(failover_controller._event_handlers["test_event"]) == 1

    @pytest.mark.asyncio
    async def test_emit_event_calls_handlers(self, failover_controller):
        """Test that emitting an event calls registered handlers."""
        handler_called = []

        def handler(data):
            handler_called.append(data)

        failover_controller.on("my_event", handler)

        await failover_controller._emit_event("my_event", {"key": "value"})

        assert len(handler_called) == 1
        assert handler_called[0] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_emit_event_async_handler(self, failover_controller):
        """Test that emitting an event calls async handlers."""
        handler_called = []

        async def async_handler(data):
            handler_called.append(data)

        failover_controller.on("async_event", async_handler)

        await failover_controller._emit_event("async_event", {"async": True})

        assert len(handler_called) == 1

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, failover_controller):
        """Test multiple handlers for same event."""
        call_count = []

        def handler1(data):
            call_count.append(1)

        def handler2(data):
            call_count.append(2)

        failover_controller.on("multi_event", handler1)
        failover_controller.on("multi_event", handler2)

        await failover_controller._emit_event("multi_event", {})

        assert len(call_count) == 2


class TestClusterStatus:
    """Test cluster status reporting."""

    @pytest.mark.asyncio
    async def test_get_cluster_failover_status_structure(self, failover_controller):
        """Test cluster failover status returns correct structure."""
        status = await failover_controller.get_cluster_failover_status()

        # Verify structure
        assert isinstance(status, dict)
        assert "total_nodes" in status
        assert "healthy_nodes" in status
        assert "failover_in_progress" in status
        assert "replicas" in status
        assert "is_leader" in status

        # Verify values are of correct types
        assert isinstance(status["total_nodes"], int)
        assert isinstance(status["healthy_nodes"], int)
        assert isinstance(status["failover_in_progress"], bool)
        assert isinstance(status["replicas"], list)
        assert isinstance(status["is_leader"], bool)


class TestHealthCheckIntegration:
    """Test health check integration."""

    @pytest.mark.asyncio
    async def test_health_check_returns_result(self, health_checker, cluster_nodes):
        """Test that health check returns a proper result."""
        result = await health_checker.check_node_health("node-1")

        # Verify result structure
        assert result.node_id == "node-1"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
        assert isinstance(result.reasons, list)
        assert isinstance(result.checks_performed, dict)

    @pytest.mark.asyncio
    async def test_health_check_detects_issues(self, health_checker, cluster_nodes):
        """Test that health check detects GPU issues."""
        # Set GPU temperature above threshold
        cluster_nodes["node-1"].gpu_info[0].temperature = 95.0

        result = await health_checker.check_node_health("node-1")

        # Should detect unhealthy condition
        assert result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]


class TestStartStop:
    """Test controller start/stop."""

    @pytest.mark.asyncio
    async def test_start_enables_running(self, failover_controller):
        """Test that start enables running state."""
        assert failover_controller._running is False

        await failover_controller.start()

        assert failover_controller._running is True

        await failover_controller.stop()

    @pytest.mark.asyncio
    async def test_stop_disables_running(self, failover_controller):
        """Test that stop disables running state."""
        await failover_controller.start()
        assert failover_controller._running is True

        await failover_controller.stop()

        assert failover_controller._running is False


# Import for health_checker tests
from quantumflow.failover.models import HealthStatus
