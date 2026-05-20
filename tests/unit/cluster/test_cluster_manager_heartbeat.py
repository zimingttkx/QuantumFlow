"""ClusterManager heartbeat and edge case coverage tests.

Covers missing lines from manager.py:
- Line 150: task.cancel() in stop() with active heartbeat tasks
- Lines 298-299: get_node_resources() method
- Line 312: find_best_nodes() labels filtering
- Lines 363-383: _heartbeat_loop (sleep, node-gone, unhealthy, timeout, CancelledError, generic exception)
- Lines 387-391: _handle_node_timeout method
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantumflow.cluster.manager import ClusterManager, GPUInfo, Node
from quantumflow.core.constants import NodeStatus


# =============================================================================
# Helper factories
# =============================================================================


def _make_node_info(node_id="n1", gpu_count=1):
    return {
        "node_id": node_id,
        "hostname": f"worker-{node_id}",
        "ip": "10.0.0.1",
        "port": 8000,
        "gpu_count": gpu_count,
        "gpu_info": [
            {
                "gpu_id": i,
                "name": "NVIDIA A100",
                "memory_total": 80 * 1024**3,
                "memory_used": 10 * 1024**3,
                "utilization": 0.1,
                "temperature": 50.0,
            }
            for i in range(gpu_count)
        ],
        "labels": {},
        "version": "1.0.0",
        "cpu_count": 8,
        "memory_total": 64 * 1024**3,
        "memory_available": 32 * 1024**3,
        "loaded_models": [],
    }


# =============================================================================
# Line 150: stop() cancels heartbeat tasks
# =============================================================================


class TestStopCancelsHeartbeatTasks:
    """Tests that stop() cancels active heartbeat tasks."""

    @pytest.mark.asyncio
    async def test_stop_cancels_existing_heartbeat_tasks(self):
        """stop() calls task.cancel() on each registered heartbeat task (line 150)."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)
        await manager.register_node(_make_node_info("n1"))
        await manager.register_node(_make_node_info("n2"))

        assert "n1" in manager._heartbeat_tasks
        assert "n2" in manager._heartbeat_tasks
        assert len(manager._heartbeat_tasks) == 2

        # Verify tasks are not done before stop
        assert not manager._heartbeat_tasks["n1"].done()
        assert not manager._heartbeat_tasks["n2"].done()

        await manager.stop()

        # After stop, tasks should be cancelled
        for task in manager._heartbeat_tasks.values():
            assert task.cancelled() or task.done(), (
                f"Task should be cancelled after stop(), got {task}"
            )

    @pytest.mark.asyncio
    async def test_stop_with_no_heartbeat_tasks_does_not_raise(self):
        """stop() on a manager with no registered nodes does not raise."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)
        # No nodes registered, so _heartbeat_tasks is empty
        await manager.stop()
        assert manager._running is False


# =============================================================================
# Lines 298-299: get_node_resources()
# =============================================================================


class TestGetNodeResources:
    """Tests for get_node_resources (lines 298-299)."""

    @pytest.mark.asyncio
    async def test_get_node_resources_returns_all(self):
        """get_node_resources returns NodeResource list for all registered nodes."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)
        await manager.register_node(_make_node_info("n1", gpu_count=2))
        await manager.register_node(_make_node_info("n2", gpu_count=1))

        resources = await manager.get_node_resources()

        assert len(resources) == 2
        for r in resources:
            assert r.node_id in ("n1", "n2")
        # n1 should have 2 GPUs
        n1_res = next(r for r in resources if r.node_id == "n1")
        assert n1_res.gpu_count == 2

    @pytest.mark.asyncio
    async def test_get_node_resources_filter_by_status(self):
        """get_node_resources filters by node status."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)
        await manager.register_node(_make_node_info("n1"))
        await manager.register_node(_make_node_info("n2"))
        await manager.update_node_status("n2", NodeStatus.UNHEALTHY)

        healthy_resources = await manager.get_node_resources(status=NodeStatus.HEALTHY)
        unhealthy_resources = await manager.get_node_resources(status=NodeStatus.UNHEALTHY)

        assert len(healthy_resources) == 1
        assert healthy_resources[0].node_id == "n1"
        assert len(unhealthy_resources) == 1
        assert unhealthy_resources[0].node_id == "n2"

    @pytest.mark.asyncio
    async def test_get_node_resources_filter_by_labels(self):
        """get_node_resources filters by labels."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)
        info_a = _make_node_info("n-a")
        info_a["labels"] = {"zone": "us-east", "env": "prod"}
        info_b = _make_node_info("n-b")
        info_b["labels"] = {"zone": "us-west", "env": "staging"}
        await manager.register_node(info_a)
        await manager.register_node(info_b)

        resources = await manager.get_node_resources(labels={"zone": "us-east"})

        assert len(resources) == 1
        assert resources[0].node_id == "n-a"

    @pytest.mark.asyncio
    async def test_get_node_resources_empty(self):
        """get_node_resources returns empty list when no nodes match."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)

        resources = await manager.get_node_resources(status=NodeStatus.HEALTHY)
        assert resources == []


# =============================================================================
# Line 312: find_best_nodes() with labels filtering
# =============================================================================


class TestFindBestNodesWithLabels:
    """Tests for find_best_nodes labels filtering (line 312)."""

    @pytest.mark.asyncio
    async def test_find_best_nodes_with_labels_filter(self):
        """find_best_nodes applies labels filter to narrow candidates."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)

        # Node with matching labels and 4 available GPUs
        info_match = _make_node_info("n-match", gpu_count=4)
        info_match["labels"] = {"pool": "inference"}
        await manager.register_node(info_match)

        # Node without matching labels but with GPUs
        info_nomatch = _make_node_info("n-nomatch", gpu_count=4)
        info_nomatch["labels"] = {"pool": "training"}
        await manager.register_node(info_nomatch)

        best = await manager.find_best_nodes(required_gpus=1, labels={"pool": "inference"})

        assert len(best) == 1
        assert best[0].node_id == "n-match"

    @pytest.mark.asyncio
    async def test_find_best_nodes_labels_no_match_returns_empty(self):
        """find_best_nodes returns empty list when no node matches labels."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)

        info = _make_node_info("n1", gpu_count=4)
        info["labels"] = {"pool": "training"}
        await manager.register_node(info)

        best = await manager.find_best_nodes(required_gpus=1, labels={"pool": "inference"})

        assert best == []


# =============================================================================
# Lines 363-383: _heartbeat_loop
# =============================================================================


class TestHeartbeatLoop:
    """Tests for _heartbeat_loop covering lines 363-383."""

    @pytest.fixture
    def manager(self):
        return ClusterManager(heartbeat_interval=1, heartbeat_timeout=2)

    @pytest.mark.asyncio
    async def test_heartbeat_loop_breaks_when_not_running(self, manager):
        """Heartbeat loop exits immediately when _running is False."""
        manager._running = False

        await manager._heartbeat_loop("n1")
        # Should not hang - loop condition _running is False at start

    @pytest.mark.asyncio
    async def test_heartbeat_loop_breaks_when_node_removed(self, manager):
        """Heartbeat loop breaks when node is no longer in self.nodes (lines 366-368)."""
        manager._running = True
        manager.nodes["n1"] = MagicMock(spec=Node, status=NodeStatus.HEALTHY)

        # Mock sleep and then immediately remove the node
        sleep_count = [0]

        async def mock_sleep(duration):
            sleep_count[0] += 1
            if sleep_count[0] == 1:
                # After first sleep, remove the node
                manager.nodes.pop("n1", None)

        with patch("asyncio.sleep", mock_sleep):
            await manager._heartbeat_loop("n1")

        # Loop should have broken when node was not found
        assert "n1" not in manager.nodes

    @pytest.mark.asyncio
    async def test_heartbeat_loop_breaks_when_node_unhealthy(self, manager):
        """Heartbeat loop breaks when node status is no longer HEALTHY (lines 371-372)."""
        manager._running = True
        node = Node(
            node_id="n1", hostname="h", ip="1.1.1.1", port=8000,
            gpu_count=1, gpu_info=[], status=NodeStatus.HEALTHY,
        )
        manager.nodes["n1"] = node

        sleep_count = [0]

        async def mock_sleep(duration):
            sleep_count[0] += 1
            if sleep_count[0] == 1:
                # After first sleep, make the node unhealthy
                node.status = NodeStatus.UNHEALTHY

        with patch("asyncio.sleep", mock_sleep):
            await manager._heartbeat_loop("n1")

        # Loop should have broken when node was found unhealthy
        assert node.status == NodeStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_heartbeat_loop_detects_timeout_and_handles(self, manager):
        """Heartbeat loop detects timeout and calls _handle_node_timeout (lines 375-378, 387-391)."""
        manager._running = True
        # Set last_heartbeat to well in the past
        old_time = datetime.now() - timedelta(seconds=10)
        node = Node(
            node_id="n1", hostname="h", ip="1.1.1.1", port=8000,
            gpu_count=1, gpu_info=[], status=NodeStatus.HEALTHY,
            last_heartbeat=old_time,
        )
        manager.nodes["n1"] = node

        handle_called = [False]

        async def mock_handle_timeout(nid):
            handle_called[0] = True
            # Make node unhealthy so loop breaks on next iteration
            node.status = NodeStatus.OFFLINE

        manager._handle_node_timeout = mock_handle_timeout

        with patch("asyncio.sleep", AsyncMock()):
            await manager._heartbeat_loop("n1")

        assert handle_called[0] is True, "_handle_node_timeout should have been called"

    @pytest.mark.asyncio
    async def test_heartbeat_loop_cancelled_error_breaks(self, manager):
        """Heartbeat loop catches CancelledError and breaks (lines 380-381)."""
        manager._running = True
        node = Node(
            node_id="n1", hostname="h", ip="1.1.1.1", port=8000,
            gpu_count=1, gpu_info=[], status=NodeStatus.HEALTHY,
        )
        manager.nodes["n1"] = node

        async def mock_sleep_cancel(duration):
            raise asyncio.CancelledError()

        # Should not raise - CancelledError is caught
        with patch("asyncio.sleep", mock_sleep_cancel):
            await manager._heartbeat_loop("n1")

    @pytest.mark.asyncio
    async def test_heartbeat_loop_generic_exception_logs_and_continues(self, manager):
        """Heartbeat loop catches generic Exception, logs error, continues (lines 382-383)."""
        manager._running = True
        node = Node(
            node_id="n1", hostname="h", ip="1.1.1.1", port=8000,
            gpu_count=1, gpu_info=[], status=NodeStatus.HEALTHY,
        )
        manager.nodes["n1"] = node

        exception_raised = [False]

        async def mock_sleep_valueerror(duration):
            if not exception_raised[0]:
                exception_raised[0] = True
                raise ValueError("simulated sleep error")
            # After error, stop the loop
            manager._running = False

        with patch("asyncio.sleep", mock_sleep_valueerror):
            await manager._heartbeat_loop("n1")

        assert exception_raised[0] is True, "Generic exception should have been raised and handled"


# =============================================================================
# Lines 387-391: _handle_node_timeout
# =============================================================================


class TestHandleNodeTimeout:
    """Tests for _handle_node_timeout covering lines 387-391."""

    @pytest.mark.asyncio
    async def test_handle_node_timeout_sets_status_offline(self):
        """_handle_node_timeout sets node status to OFFLINE."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)
        await manager.register_node(_make_node_info("n1"))
        assert manager.nodes["n1"].status == NodeStatus.HEALTHY

        await manager._handle_node_timeout("n1")

        assert manager.nodes["n1"].status == NodeStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_handle_node_timeout_emits_health_changed_event(self):
        """_handle_node_timeout emits node_health_changed event with correct args."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)
        await manager.register_node(_make_node_info("n1"))

        handler = MagicMock()
        manager.on("node_health_changed", handler)

        await manager._handle_node_timeout("n1")

        handler.assert_called_once()
        call_args = handler.call_args
        # First positional arg: the node
        assert call_args.args[0].node_id == "n1"
        # Second positional arg: old status
        assert call_args.args[1] == NodeStatus.HEALTHY
        # Third positional arg: new status
        assert call_args.args[2] == NodeStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_handle_node_timeout_async_handler(self):
        """_handle_node_timeout correctly calls async event handlers."""
        manager = ClusterManager(heartbeat_interval=5, heartbeat_timeout=30)
        await manager.register_node(_make_node_info("n1"))

        async_handler = AsyncMock()
        manager.on("node_health_changed", async_handler)

        await manager._handle_node_timeout("n1")

        async_handler.assert_called_once()
        assert async_handler.call_args.args[0].node_id == "n1"
        assert async_handler.call_args.args[1] == NodeStatus.HEALTHY
        assert async_handler.call_args.args[2] == NodeStatus.OFFLINE
