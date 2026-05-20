"""Scheduler edge case and exception handling coverage tests.

Covers missing lines from scheduler.py:
- Lines 169-171: _scheduling_loop generic Exception handler
- Lines 185-186: _process_batch QueueEmpty handler
- Lines 393-396: unregister_node callback exception handler
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantumflow.scheduler.scheduler import Scheduler
from quantumflow.scheduler.strategy.base import (
    GPUResource,
    NodeResource,
    SchedulingRequest,
)


# =============================================================================
# Helper factory
# =============================================================================


def _make_node_resource(node_id, load=0.2):
    """Create a minimal healthy NodeResource."""
    return NodeResource(
        node_id=node_id,
        hostname=f"server-{node_id}",
        ip=f"10.0.0.{node_id}",
        status="healthy",
        gpu_count=1,
        gpus=[
            GPUResource(
                gpu_id=0,
                memory_total=24 * 1024**3,
                memory_used=8 * 1024**3,
                utilization=0.3,
                temperature=45.0,
                node_id=node_id,
            )
        ],
        cpu_count=4,
        memory_total=32 * 1024**3,
        memory_available=16 * 1024**3,
        disk_total=100 * 1024**3,
        disk_available=50 * 1024**3,
        load=load,
    )


# =============================================================================
# Lines 169-171: _scheduling_loop generic Exception handler
# =============================================================================


class TestSchedulingLoopExceptionHandler:
    """Tests that _scheduling_loop catches generic exceptions (lines 169-171)."""

    @pytest.mark.asyncio
    async def test_scheduling_loop_catches_exception_and_sleeps(self):
        """When _process_batch raises, _scheduling_loop catches it, logs, sleeps 1s."""
        scheduler = Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)
        scheduler._running = True

        # Add a node so _process_batch is called (loop has `if not available_nodes: continue`)
        from quantumflow.scheduler.strategy.base import NodeResource, GPUResource

        node = NodeResource(
            node_id="test-node",
            hostname="test-host",
            ip="10.0.0.1",
            status="healthy",
            gpu_count=1,
            gpus=[
                GPUResource(
                    gpu_id=0,
                    memory_total=24 * 1024**3,
                    memory_used=8 * 1024**3,
                    utilization=0.3,
                    temperature=45.0,
                    node_id="test-node",
                )
            ],
            cpu_count=4,
            memory_total=32 * 1024**3,
            memory_available=16 * 1024**3,
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
            load=0.2,
        )
        scheduler.available_nodes["test-node"] = node

        # Make _process_batch raise on first call
        raise_count = [0]

        async def mock_process_batch():
            raise_count[0] += 1
            if raise_count[0] == 1:
                raise RuntimeError("simulated batch error")
            else:
                # On second call, stop the loop
                scheduler._running = False

        scheduler._process_batch = mock_process_batch

        # Save reference to real asyncio.sleep before patching
        real_sleep = asyncio.sleep

        # Mock sleep to yield control properly
        async def mock_sleep(duration):
            await real_sleep(0)

        with patch("asyncio.sleep", mock_sleep):
            await scheduler._scheduling_loop()

        # Exception should have been raised and caught, and loop should have run at least twice
        assert raise_count[0] >= 2, (
            f"_process_batch should have been called at least twice, got {raise_count[0]}"
        )

    @pytest.mark.asyncio
    async def test_scheduling_loop_cancelled_error_breaks(self):
        """CancelledError in _scheduling_loop should break the loop."""
        scheduler = Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)
        scheduler._running = True

        async def mock_sleep_cancel(duration):
            raise asyncio.CancelledError()

        with patch("asyncio.sleep", mock_sleep_cancel):
            await scheduler._scheduling_loop()

        # No exception should propagate


# =============================================================================
# Lines 185-186: _process_batch QueueEmpty handler
# =============================================================================


class TestProcessBatchQueueEmpty:
    """Tests that _process_batch handles QueueEmpty (lines 185-186)."""

    @pytest.mark.asyncio
    async def test_process_batch_handles_queue_empty_during_get(self):
        """When get_nowait raises QueueEmpty, the batch loop breaks gracefully."""
        scheduler = Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)
        # qsize() reports items, but get_nowait raises QueueEmpty (race condition simulation)
        scheduler.pending_queue.qsize = MagicMock(return_value=5)

        # Queue has 0 actual items, get_nowait will raise QueueEmpty
        # Simulate: the queue reports size > 0 but is actually empty
        # Since PriorityQueue doesn't have items, get_nowait will raise QueueEmpty
        await scheduler._process_batch()
        # Should not raise - QueueEmpty is caught

    @pytest.mark.asyncio
    async def test_process_batch_breaks_on_first_queue_empty(self):
        """When the first get_nowait raises QueueEmpty, batch_size items are not consumed."""
        scheduler = Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)
        # Put one item, but make qsize report a large number
        request = SchedulingRequest(request_id="r1", model="m", priority=5)
        scheduler.pending_queue.put_nowait((5, request.created_at, 1, request))
        scheduler.pending_queue.qsize = MagicMock(return_value=10)

        # get_nowait will: first call returns the item, second call raises QueueEmpty
        get_count = [0]
        original_get = scheduler.pending_queue.get_nowait

        def tracked_get():
            get_count[0] += 1
            if get_count[0] <= 1:
                return original_get()
            raise asyncio.QueueEmpty()

        scheduler.pending_queue.get_nowait = tracked_get

        await scheduler._process_batch()

        # Should have tried to get items but stopped after QueueEmpty
        assert get_count[0] >= 1


# =============================================================================
# Lines 393-396: unregister_node callback exception handler
# =============================================================================


class TestUnregisterNodeCallbackException:
    """Tests that unregister_node handles callback exceptions (lines 393-396)."""

    @pytest.mark.asyncio
    async def test_unregister_node_callback_exception_does_not_block(self):
        """When a callback raises, other callbacks still execute and unregister completes."""
        scheduler = Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

        node = _make_node_resource("n1")
        scheduler.available_nodes["n1"] = node

        good_callback = AsyncMock()
        bad_callback = AsyncMock(side_effect=RuntimeError("callback broke"))

        scheduler.on_node_update(bad_callback)
        scheduler.on_node_update(good_callback)

        await scheduler.unregister_node("n1")

        assert "n1" not in scheduler.available_nodes, "Node should be removed despite callback error"
        bad_callback.assert_called_once()
        good_callback.assert_called_once()
        # Verify good_callback received the expected args
        assert good_callback.call_args.args[0] == "unregister"
        assert good_callback.call_args.args[1].node_id == "n1"

    @pytest.mark.asyncio
    async def test_unregister_node_with_only_bad_callback(self):
        """unregister_node completes when the only callback raises."""
        scheduler = Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

        node = _make_node_resource("n1")
        scheduler.available_nodes["n1"] = node

        bad_callback = AsyncMock(side_effect=RuntimeError("callback broke"))
        scheduler.on_node_update(bad_callback)

        await scheduler.unregister_node("n1")

        assert "n1" not in scheduler.available_nodes, "Node should be removed despite callback error"
        bad_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_node_callback_exception_does_not_block(self):
        """When a register callback raises, other callbacks still execute."""
        scheduler = Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

        node = _make_node_resource("n1")

        good_callback = AsyncMock()
        bad_callback = AsyncMock(side_effect=RuntimeError("register callback broke"))

        scheduler.on_node_update(bad_callback)
        scheduler.on_node_update(good_callback)

        await scheduler.register_node(node)

        assert "n1" in scheduler.available_nodes, "Node should be registered despite callback error"
        good_callback.assert_called_once()
        assert good_callback.call_args.args[0] == "register"

    @pytest.mark.asyncio
    async def test_unregister_node_callback_sync_exception(self):
        """Sync callback that raises during unregister is also handled."""
        scheduler = Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

        node = _make_node_resource("n1")
        scheduler.available_nodes["n1"] = node

        def bad_sync_callback(event, node_arg):
            raise RuntimeError("sync callback broke")

        scheduler.on_node_update(bad_sync_callback)

        await scheduler.unregister_node("n1")

        assert "n1" not in scheduler.available_nodes, (
            "Node should be removed despite sync callback error"
        )
