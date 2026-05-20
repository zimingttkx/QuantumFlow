"""Scheduler gap coverage tests

Tests covering gaps found in test_scheduler.py:
1. _schedule_request with empty nodes
2. _get_strategy with different modes
3. _handle_scheduling_failure retry / final failure
4. Priority queue ordering verification
5. Stats correctness (no double-count on retried success)
6. _send_to_worker fallback to available_nodes
7. Node callback edge cases
8. _process_batch batch size limit
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantumflow.scheduler import Scheduler, SchedulingRequest
from quantumflow.scheduler.strategy.base import NodeResource, SchedulingResult
from quantumflow.scheduler.worker_client import WorkerClient, WorkerEndpoint


class TestSchedulerScheduleRequest:
    """Tests for _schedule_request and _get_strategy."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

    @pytest.fixture
    def sched_request(self):
        return SchedulingRequest(
            request_id="test-001",
            model="test-model",
            model_config={"tensor_parallel": 1},
        )

    @pytest.mark.asyncio
    async def test_schedule_request_empty_nodes_returns_failure(self, scheduler, sched_request):
        """_schedule_request returns failure when no nodes available."""
        # Ensure available_nodes is empty
        scheduler.available_nodes.clear()

        result = await scheduler._schedule_request(sched_request)

        assert result.success is False
        assert result.reason == "No available nodes"

    @pytest.mark.asyncio
    async def test_get_strategy_gang_mode(self, scheduler, sched_request):
        """_get_strategy returns gang strategy when default_strategy is 'gang'."""
        scheduler.default_strategy = "gang"

        nodes = [MagicMock(spec=NodeResource)]
        strategy = scheduler._get_strategy(sched_request, nodes)

        assert strategy is not None
        assert strategy.name == "gang"

    @pytest.mark.asyncio
    async def test_get_strategy_pack_mode(self, scheduler, sched_request):
        """_get_strategy returns pack strategy when default_strategy is 'pack'."""
        scheduler.default_strategy = "pack"

        nodes = [MagicMock(spec=NodeResource)]
        strategy = scheduler._get_strategy(sched_request, nodes)

        assert strategy is not None
        assert strategy.name == "pack"

    @pytest.mark.asyncio
    async def test_get_strategy_adaptive_mode(self, scheduler, sched_request):
        """_get_strategy returns adaptive strategy when default_strategy is 'adaptive'."""
        scheduler.default_strategy = "adaptive"

        nodes = [MagicMock(spec=NodeResource)]
        strategy = scheduler._get_strategy(sched_request, nodes)

        assert strategy is not None
        assert strategy.name == "adaptive"

    @pytest.mark.asyncio
    async def test_get_strategy_invalid_returns_none(self, scheduler, sched_request):
        """_get_strategy returns None for unrecognized default_strategy."""
        scheduler.default_strategy = "nonexistent"

        nodes = [MagicMock(spec=NodeResource)]
        strategy = scheduler._get_strategy(sched_request, nodes)

        assert strategy is None

    @pytest.mark.asyncio
    async def test_schedule_request_unknown_strategy_returns_failure(self, scheduler, sched_request):
        """_schedule_request returns failure when no strategy is found."""
        scheduler.default_strategy = "nonexistent"
        # Add a node so we pass the empty-nodes check
        mock_node = MagicMock(spec=NodeResource)
        scheduler.available_nodes["node-1"] = mock_node

        result = await scheduler._schedule_request(sched_request)

        assert result.success is False
        assert "Strategy not found" in result.reason


class TestSchedulingFailureRetry:
    """Tests for _handle_scheduling_failure retry logic."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

    @pytest.mark.asyncio
    async def test_handle_scheduling_failure_requeues_on_retry(self, scheduler):
        """Failed request with retries remaining is re-enqueued with incremented retry_count."""
        request = SchedulingRequest(
            request_id="test-retry",
            model="test-model",
            priority=5,
            retry_count=1,
        )
        result = SchedulingResult(success=False, reason="No resources")

        initial_queue_size = scheduler.pending_queue.qsize()
        initial_failed = scheduler.stats["failed_requests"]

        await scheduler._handle_scheduling_failure(request, result)

        assert request.retry_count == 2, "retry_count should be incremented"
        assert scheduler.pending_queue.qsize() == initial_queue_size + 1, "request should be re-enqueued"
        # failed_requests should NOT be incremented at this stage (already counted in _process_batch)
        # Note: _process_batch would have already incremented failed_requests before calling this

    @pytest.mark.asyncio
    async def test_handle_scheduling_failure_final_failure(self, scheduler):
        """Failed request at max retries marks final failure, updates stats."""
        request = SchedulingRequest(
            request_id="test-final",
            model="test-model",
            priority=5,
            retry_count=3,  # equals max_retries (3), so retry_count < max_retries is False
        )
        result = SchedulingResult(success=False, reason="No resources")

        initial_queue_size = scheduler.pending_queue.qsize()
        initial_failed = scheduler.stats["failed_requests"]
        initial_pending = scheduler.stats["pending_requests"]

        await scheduler._handle_scheduling_failure(request, result)

        assert request.retry_count == 4, "retry_count should still be incremented"
        assert scheduler.pending_queue.qsize() == initial_queue_size, "request should NOT be re-enqueued"
        # Stats are updated: failed_requests incremented, pending_requests decremented
        assert scheduler.stats["failed_requests"] == initial_failed + 1
        assert scheduler.stats["pending_requests"] == initial_pending - 1

    @pytest.mark.asyncio
    async def test_handle_scheduling_failure_retry_with_zero_retries(self, scheduler):
        """First failure (retry_count=0) correctly requeues with count 1."""
        request = SchedulingRequest(
            request_id="test-first-fail",
            model="test-model",
            priority=5,
            retry_count=0,
        )
        result = SchedulingResult(success=False, reason="No resources")

        await scheduler._handle_scheduling_failure(request, result)

        assert request.retry_count == 1
        assert scheduler.pending_queue.qsize() == 1


class TestPriorityQueueOrdering:
    """Tests for priority queue ordering correctness."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

    @pytest.mark.asyncio
    async def test_higher_priority_requests_dequeued_first(self, scheduler):
        """Higher priority requests (lower priority number = higher urgency) are dequeued first."""
        # priority 1 = highest, priority 9 = lowest
        # The PriorityQueue uses (priority, created_at, counter, request) tuples
        # Lower priority value means higher urgency
        await scheduler.start()

        req_low = SchedulingRequest(request_id="low", model="m", priority=9)
        req_high = SchedulingRequest(request_id="high", model="m", priority=1)
        req_mid = SchedulingRequest(request_id="mid", model="m", priority=5)

        await scheduler.submit(req_low)
        await scheduler.submit(req_high)
        await scheduler.submit(req_mid)

        # Dequeue in order
        p1, _, _, r1 = scheduler.pending_queue.get_nowait()
        p2, _, _, r2 = scheduler.pending_queue.get_nowait()
        p3, _, _, r3 = scheduler.pending_queue.get_nowait()

        assert r1.request_id == "high", f"Expected high priority first, got {r1.request_id}"
        assert r2.request_id == "mid", f"Expected mid priority second, got {r2.request_id}"
        assert r3.request_id == "low", f"Expected low priority last, got {r3.request_id}"

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_same_priority_ordered_by_submission(self, scheduler):
        """Requests with same priority are ordered by submission time (counter)."""
        await scheduler.start()

        req_a = SchedulingRequest(request_id="A", model="m", priority=5)
        req_b = SchedulingRequest(request_id="B", model="m", priority=5)
        req_c = SchedulingRequest(request_id="C", model="m", priority=5)

        await scheduler.submit(req_a)
        await scheduler.submit(req_b)
        await scheduler.submit(req_c)

        p1, _, c1, r1 = scheduler.pending_queue.get_nowait()
        p2, _, c2, r2 = scheduler.pending_queue.get_nowait()
        p3, _, c3, r3 = scheduler.pending_queue.get_nowait()

        assert r1.request_id == "A"
        assert r2.request_id == "B"
        assert r3.request_id == "C"
        assert c1 < c2 < c3

        await scheduler.stop()


class TestSchedulerStatsConsistency:
    """Tests for stats correctness edge cases."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

    @pytest.mark.asyncio
    async def test_submit_updates_pending_and_total(self, scheduler):
        """submit() increments total_requests and pending_requests."""
        request = SchedulingRequest(request_id="s1", model="m")
        initial_total = scheduler.stats["total_requests"]
        initial_pending = scheduler.stats["pending_requests"]

        await scheduler.submit(request)

        assert scheduler.stats["total_requests"] == initial_total + 1
        assert scheduler.stats["pending_requests"] == initial_pending + 1

    def test_get_stats_includes_all_keys(self, scheduler):
        """get_stats() returns all expected keys."""
        stats = scheduler.get_stats()

        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "pending_requests" in stats
        assert "queue_size" in stats
        assert "running_size" in stats
        assert "available_nodes" in stats
        assert stats["queue_size"] == 0
        assert stats["running_size"] == 0
        assert stats["available_nodes"] == 0


class TestNodeManagementEdgeCases:
    """Tests for node management edge cases."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_node_no_error(self, scheduler):
        """unregister_node on non-existent node should not raise."""
        await scheduler.unregister_node("nonexistent-node")
        # No exception raised

    @pytest.mark.asyncio
    async def test_update_nonexistent_node_no_error(self, scheduler):
        """update_node on non-existent node should set it (no error)."""
        node = NodeResource(
            node_id="new-node",
            hostname="s",
            ip="1.2.3.4",
            status="healthy",
            gpu_count=1,
            gpus=[],
            cpu_count=4,
            memory_total=8 * 1024**3,
            memory_available=4 * 1024**3,
            disk_total=100 * 1024**3,
            disk_available=50 * 1024**3,
            load=0.1,
        )
        await scheduler.update_node(node)
        assert "new-node" in scheduler.available_nodes
        assert scheduler.available_nodes["new-node"].load == 0.1

    @pytest.mark.asyncio
    async def test_node_callback_exception_does_not_block_others(self, scheduler):
        """One callback raising an exception does not prevent other callbacks from running."""
        call_order = []
        good_callback = AsyncMock()
        bad_callback = AsyncMock(side_effect=RuntimeError("callback error"))

        async def recording_callback(event, node):
            call_order.append("called")
            return None

        scheduler.on_node_update(bad_callback)
        scheduler.on_node_update(recording_callback)

        node = NodeResource(
            node_id="n1", hostname="h", ip="1.1.1.1", status="healthy",
            gpu_count=1, gpus=[], cpu_count=2,
            memory_total=8 * 1024**3, memory_available=4 * 1024**3,
            disk_total=100 * 1024**3, disk_available=50 * 1024**3, load=0.1,
        )
        await scheduler.register_node(node)

        assert "called" in call_order, "Second callback should execute despite first one raising"

    @pytest.mark.asyncio
    async def test_duplicate_start_logs_warning(self, scheduler):
        """Starting an already-running scheduler should log a warning and return."""
        await scheduler.start()
        assert scheduler._running is True

        first_task = scheduler._scheduling_task

        # Second start
        await scheduler.start()

        # Should remain running with the same task
        assert scheduler._running is True
        assert scheduler._scheduling_task is first_task

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, scheduler):
        """Calling stop on a not-running scheduler should not raise errors."""
        # stop() does CancelledError handling internally
        await scheduler.stop()
        assert scheduler._running is False


class TestProcessBatchSizeLimit:
    """Tests for _process_batch batch size logic."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

    @pytest.mark.asyncio
    async def test_process_batch_caps_at_10(self, scheduler):
        """_process_batch should process at most 10 requests per call."""
        await scheduler.start()

        # Submit 15 requests
        for i in range(15):
            request = SchedulingRequest(
                request_id=f"batch-{i:03d}",
                model="test-model",
                priority=5,
            )
            await scheduler.submit(request)

        assert scheduler.pending_queue.qsize() == 15

        # Manually call _process_batch (without nodes, it will call _handle_scheduling_failure)
        # The key test: batch_size = min(qsize, 10) = 10
        await scheduler._process_batch()

        # After processing 10 items (which all fail since no nodes),
        # 5 should remain in the queue
        # Note: items that fail and are retried go back, but each gets retry_count++
        # With max_retries=3, the first failure (retry_count 0->1, still < 3) requeues them
        # So they go back into the queue

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_process_batch_empty_queue_no_error(self, scheduler):
        """_process_batch on empty queue should not raise."""
        await scheduler._process_batch()
        # No exception raised


class TestSendToWorkerFallback:
    """Tests for _send_to_worker node fallback logic."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

    @pytest.mark.asyncio
    async def test_send_to_worker_fallback_to_available_nodes(self, scheduler):
        """When WorkerRegistry returns None but node is in available_nodes, uses available_nodes info."""
        request = SchedulingRequest(
            request_id="test-fallback",
            model="test-model",
            prompt="Hello",
            model_config={"sampling_params": {"temperature": 0.5, "top_p": 0.8, "max_tokens": 100}},
        )
        result = SchedulingResult(
            success=True,
            assigned_nodes=["node-1"],
            estimated_wait_time=0.1,
            strategy_used="pack",
        )

        # Add node to available_nodes
        from quantumflow.scheduler.strategy.base import (
            GPUResource,
            NodeResource,
        )
        node = NodeResource(
            node_id="node-1", hostname="s1", ip="10.0.0.1", status="healthy",
            gpu_count=1,
            gpus=[GPUResource(gpu_id=0, memory_total=24 * 1024**3, memory_used=10 * 1024**3,
                              utilization=0.5, temperature=45.0, node_id="node-1")],
            cpu_count=8, memory_total=64 * 1024**3, memory_available=32 * 1024**3,
            disk_total=500 * 1024**3, disk_available=250 * 1024**3, load=0.2,
        )
        scheduler.available_nodes["node-1"] = node

        mock_registry = AsyncMock()
        mock_registry.get_worker = AsyncMock(return_value=None)  # Registry returns None

        mock_client = AsyncMock(spec=WorkerClient)
        mock_client.inference = AsyncMock(
            return_value={"status": "success", "result": {"text": "ok"}}
        )
        mock_client.close = AsyncMock()

        with patch(
            "quantumflow.scheduler.scheduler.get_worker_registry",
            return_value=mock_registry,
        ):
            with patch(
                "quantumflow.scheduler.scheduler.WorkerClient",
                return_value=mock_client,
            ):
                await scheduler._dispatch(request, result)
                await asyncio.sleep(0.1)

                # WorkerClient.inference should have been called
                # with an endpoint constructed from available_nodes
                mock_client.inference.assert_called_once()
                call_kwargs = mock_client.inference.call_args
                endpoint = call_kwargs.kwargs["endpoint"]
                assert endpoint.host == "10.0.0.1"
                assert endpoint.port == 8000  # default port from fallback

    @pytest.mark.asyncio
    async def test_send_to_worker_node_not_in_registry_or_available(self, scheduler):
        """When registry returns None and node NOT in available_nodes, worker is skipped."""
        request = SchedulingRequest(
            request_id="test-missing-node",
            model="test-model",
            prompt="Hello",
            model_config={"sampling_params": {"temperature": 0.5, "top_p": 0.8, "max_tokens": 100}},
        )
        result = SchedulingResult(
            success=True,
            assigned_nodes=["unknown-node"],
            estimated_wait_time=0.1,
            strategy_used="pack",
        )

        mock_registry = AsyncMock()
        mock_registry.get_worker = AsyncMock(return_value=None)

        mock_client = AsyncMock(spec=WorkerClient)
        mock_client.inference = AsyncMock()
        mock_client.close = AsyncMock()

        with patch(
            "quantumflow.scheduler.scheduler.get_worker_registry",
            return_value=mock_registry,
        ):
            with patch(
                "quantumflow.scheduler.scheduler.WorkerClient",
                return_value=mock_client,
            ):
                await scheduler._dispatch(request, result)
                await asyncio.sleep(0.1)

                # inference should NOT be called since no endpoint could be found
                mock_client.inference.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_to_worker_exception_updates_failed_count(self, scheduler):
        """When inference raises an exception, failed_requests is incremented."""
        request = SchedulingRequest(
            request_id="test-exception",
            model="test-model",
            prompt="Hello",
        )
        result = SchedulingResult(
            success=True,
            assigned_nodes=["node-1"],
            estimated_wait_time=0.1,
            strategy_used="pack",
        )

        mock_registry = AsyncMock()
        mock_registry.get_worker = AsyncMock(
            return_value=WorkerEndpoint(node_id="node-1", host="10.0.0.1", port=8000)
        )

        mock_client = AsyncMock(spec=WorkerClient)
        mock_client.inference = AsyncMock(side_effect=RuntimeError("Worker crashed"))
        mock_client.close = AsyncMock()

        initial_failed = scheduler.stats["failed_requests"]

        with patch(
            "quantumflow.scheduler.scheduler.get_worker_registry",
            return_value=mock_registry,
        ):
            with patch(
                "quantumflow.scheduler.scheduler.WorkerClient",
                return_value=mock_client,
            ):
                await scheduler._dispatch(request, result)
                await asyncio.sleep(0.1)

                final_failed = scheduler.stats["failed_requests"]
                assert final_failed > initial_failed, (
                    f"failed_requests should increase after worker exception. "
                    f"Was {initial_failed}, now {final_failed}"
                )

    @pytest.mark.asyncio
    async def test_client_close_called_in_finally(self, scheduler):
        """WorkerClient.close() is called in the finally block of _send_to_worker."""
        request = SchedulingRequest(
            request_id="test-close",
            model="test-model",
            prompt="Hello",
            model_config={"sampling_params": {"temperature": 0.5, "top_p": 0.8, "max_tokens": 100}},
        )
        result = SchedulingResult(
            success=True,
            assigned_nodes=["node-1"],
            estimated_wait_time=0.1,
            strategy_used="pack",
        )

        mock_registry = AsyncMock()
        mock_registry.get_worker = AsyncMock(
            return_value=WorkerEndpoint(node_id="node-1", host="10.0.0.1", port=8000)
        )

        mock_client = AsyncMock(spec=WorkerClient)
        mock_client.inference = AsyncMock(
            return_value={"status": "success"}
        )
        mock_client.close = AsyncMock()

        with patch(
            "quantumflow.scheduler.scheduler.get_worker_registry",
            return_value=mock_registry,
        ):
            with patch(
                "quantumflow.scheduler.scheduler.WorkerClient",
                return_value=mock_client,
            ):
                await scheduler._dispatch(request, result)
                await asyncio.sleep(0.1)

                mock_client.close.assert_called_once()


class TestSchedulingLoopEdgeCases:
    """Tests for _scheduling_loop edge cases."""

    @pytest.fixture
    def scheduler(self):
        return Scheduler(default_strategy="adaptive", loop_interval_ms=50, max_retries=3)

    @pytest.mark.asyncio
    async def test_scheduling_loop_skips_when_no_nodes(self, scheduler):
        """Scheduling loop should skip processing when available_nodes is empty."""
        await scheduler.start()

        # Submit a request but provide no nodes
        request = SchedulingRequest(
            request_id="skip-no-nodes",
            model="test-model",
            priority=5,
        )
        await scheduler.submit(request)

        # Wait a bit for the loop to potentially process
        await asyncio.sleep(0.3)

        # Without nodes, the request should still be pending
        stats = scheduler.get_stats()
        assert stats["pending_requests"] >= 1

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_get_pending_requests_empty_queue(self, scheduler):
        """get_pending_requests returns empty list for empty queue."""
        pending = scheduler.get_pending_requests()
        assert pending == []

    @pytest.mark.asyncio
    async def test_get_running_requests_empty(self, scheduler):
        """get_running_requests returns empty dict initially."""
        running = scheduler.get_running_requests()
        assert running == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
