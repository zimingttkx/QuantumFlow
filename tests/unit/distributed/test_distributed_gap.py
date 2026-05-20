"""DistributedScheduler gap coverage tests

Tests covering gaps found in test_distributed_comprehensive.py:
1. DistributedScheduler.submit with Redis enqueue (success path)
2. DistributedScheduler.submit with Redis failure -> memory fallback
3. DistributedScheduler._process_batch_from_redis request conversion
4. DistributedScheduler._handle_scheduling_failure_redis: retry and final failure
5. DistributedScheduler._dispatch: empty assigned_nodes, worker not found
6. DistributedScheduler._send_to_worker: success, error, exception paths
7. DistributedScheduler._handle_dispatch_failure: retry and final failure
8. DistributedScheduler.start: Redis unavailable fallback
9. DistributedScheduler.get_queue_stats: Redis and non-Redis cases
10. Global functions: get_scheduler, init_scheduler, close_scheduler
11. Metadata handling bug: getattr(request, "metadata", {}) on SchedulingRequest
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantumflow.scheduler.distributed import (
    DistributedScheduler,
    close_scheduler,
    get_scheduler,
    init_scheduler,
)
from quantumflow.scheduler.strategy.base import SchedulingRequest, SchedulingResult
from quantumflow.scheduler.worker_client import WorkerClient, WorkerEndpoint, WorkerRegistry
from quantumflow.storage.redis_queue import (
    QueuePriority,
    QueuedRequest,
    RedisQueue,
)


# ==================== Helpers ====================


def create_scheduling_request(request_id="test-1", model="test-model",
                              prompt="Hello", priority=5):
    return SchedulingRequest(
        request_id=request_id,
        model=model,
        prompt=prompt,
        priority=priority,
        created_at=datetime.now(),
    )


def create_queued_request(request_id="test-1", model_name="test-model",
                          prompt="Hello", priority=5):
    return QueuedRequest(
        request_id=request_id,
        model_name=model_name,
        prompt=prompt,
        priority=priority,
        created_at=datetime.now(),
    )


# ==================== submit: Redis enqueue ====================


class TestDistributedSubmitRedis:
    """Tests for DistributedScheduler.submit with Redis."""

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )

    @pytest.mark.asyncio
    async def test_submit_enqueues_to_redis(self, scheduler):
        """Successful submit adds request to Redis queue."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.enqueue = AsyncMock(return_value=True)
        scheduler._redis_queue = mock_queue
        scheduler._use_redis = True

        request = create_scheduling_request(request_id="redis-test")
        initial_total = scheduler.stats["total_requests"]
        initial_pending = scheduler.stats["pending_requests"]

        result_id = await scheduler.submit(request)

        assert result_id == "redis-test"
        assert scheduler.stats["total_requests"] == initial_total + 1
        assert scheduler.stats["pending_requests"] == initial_pending + 1

        mock_queue.enqueue.assert_called_once()
        # Verify the QueuedRequest was constructed correctly
        enqueued_req = mock_queue.enqueue.call_args[0][0]
        assert isinstance(enqueued_req, QueuedRequest)
        assert enqueued_req.request_id == "redis-test"
        assert enqueued_req.model_name == "test-model"
        assert enqueued_req.prompt == "Hello"
        assert enqueued_req.priority == 5

    @pytest.mark.asyncio
    async def test_submit_redis_failure_falls_back_to_memory(self, scheduler):
        """When Redis enqueue fails, falls back to memory PriorityQueue."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.enqueue = AsyncMock(return_value=False)
        scheduler._redis_queue = mock_queue
        scheduler._use_redis = True

        request = create_scheduling_request(request_id="fallback-test")
        initial_queue_size = scheduler.pending_queue.qsize()

        result_id = await scheduler.submit(request)

        assert result_id == "fallback-test"
        # After failure, _use_redis should be set to False
        assert scheduler._use_redis is False
        # Request should be in the memory queue
        assert scheduler.pending_queue.qsize() == initial_queue_size + 1


# ==================== process_batch_from_redis ====================


class TestProcessBatchFromRedis:
    """Tests for _process_batch_from_redis."""

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )

    @pytest.mark.asyncio
    async def test_converts_queued_request_to_scheduling_request(self, scheduler):
        """_process_batch_from_redis correctly converts QueuedRequest to SchedulingRequest."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.queue_size = AsyncMock(return_value=1)

        qr = create_queued_request(request_id="conv-test", model_name="conv-model",
                                   prompt="Convert me", priority=7)
        mock_queue.dequeue = AsyncMock(return_value=qr)

        scheduler._redis_queue = mock_queue

        # Mock _schedule_request to return failure so we don't need WorkerClient
        scheduler._schedule_request = AsyncMock(
            return_value=SchedulingResult(success=False, reason="No nodes")
        )

        await scheduler._process_batch_from_redis()

        # _schedule_request should have been called with a SchedulingRequest
        call_arg = scheduler._schedule_request.call_args[0][0]
        assert isinstance(call_arg, SchedulingRequest)
        assert call_arg.request_id == "conv-test"
        assert call_arg.model == "conv-model"
        assert call_arg.prompt == "Convert me"
        assert call_arg.priority == 7

    @pytest.mark.asyncio
    async def test_process_batch_from_redis_empty_queue(self, scheduler):
        """When Redis queue is empty, _process_batch_from_redis returns early."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.queue_size = AsyncMock(return_value=0)
        scheduler._redis_queue = mock_queue

        # Should not raise
        await scheduler._process_batch_from_redis()

        mock_queue.dequeue.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_batch_from_redis_no_queue_object(self, scheduler):
        """When _redis_queue is None, _process_batch_from_redis returns early."""
        scheduler._redis_queue = None

        # Should not raise
        await scheduler._process_batch_from_redis()


# ==================== _handle_scheduling_failure_redis ====================


class TestHandleSchedulingFailureRedis:
    """Tests for _handle_scheduling_failure_redis."""

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            max_retries=3,
            redis_url="redis://localhost:6379/0",
        )

    @pytest.mark.asyncio
    async def test_retry_path_requeues_request(self, scheduler):
        """When retry_count < max_retries, request is requeued via Redis."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.requeue = AsyncMock(return_value=True)
        scheduler._redis_queue = mock_queue

        request = create_scheduling_request(request_id="retry-test", priority=5)
        request.retry_count = 1
        result = SchedulingResult(success=False, reason="No resources")

        await scheduler._handle_scheduling_failure_redis(request, result)

        assert request.retry_count == 2
        mock_queue.requeue.assert_called_once()
        requeued = mock_queue.requeue.call_args[0][0]
        assert isinstance(requeued, QueuedRequest)
        assert requeued.request_id == "retry-test"
        assert requeued.metadata.get("retry_count") == 2

    @pytest.mark.asyncio
    async def test_final_failure_path_stores_error_result(self, scheduler):
        """When retry_count >= max_retries, stores error result and updates stats."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.store_result = AsyncMock(return_value=True)
        scheduler._redis_queue = mock_queue

        request = create_scheduling_request(request_id="final-fail")
        request.retry_count = 3  # equals max_retries
        result = SchedulingResult(success=False, reason="No resources")

        initial_failed = scheduler.stats["failed_requests"]
        initial_pending = scheduler.stats["pending_requests"]

        await scheduler._handle_scheduling_failure_redis(request, result)

        assert scheduler.stats["failed_requests"] == initial_failed + 1
        assert scheduler.stats["pending_requests"] == initial_pending - 1

        mock_queue.store_result.assert_called_once()
        stored = mock_queue.store_result.call_args[0]
        assert stored[0] == "final-fail"
        assert stored[1]["status"] == "error"
        assert stored[1]["reason"] == "No resources"


# ==================== Distributed _dispatch ====================


class TestDistributedDispatch:
    """Tests for DistributedScheduler._dispatch."""

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )

    @pytest.mark.asyncio
    async def test_dispatch_empty_assigned_nodes_logs_error(self, scheduler):
        """When result has empty assigned_nodes, calls _handle_dispatch_failure."""
        request = create_scheduling_request(request_id="no-nodes")
        result = SchedulingResult(success=True, assigned_nodes=[],
                                  estimated_wait_time=0.1, strategy_used="pack")

        scheduler._handle_dispatch_failure = AsyncMock()

        await scheduler._dispatch(request, result)

        scheduler._handle_dispatch_failure.assert_called_once_with(
            request, "No nodes assigned"
        )

    @pytest.mark.asyncio
    async def test_dispatch_worker_not_found_calls_failure(self, scheduler):
        """When WorkerRegistry returns None for the assigned node, calls failure handler."""
        request = create_scheduling_request(request_id="no-worker")
        result = SchedulingResult(success=True, assigned_nodes=["unknown-node"],
                                  estimated_wait_time=0.1, strategy_used="pack")

        mock_registry = AsyncMock(spec=WorkerRegistry)
        mock_registry.get_worker = AsyncMock(return_value=None)
        scheduler._worker_registry = mock_registry

        scheduler._handle_dispatch_failure = AsyncMock()

        await scheduler._dispatch(request, result)

        scheduler._handle_dispatch_failure.assert_called_once_with(
            request, "Worker unknown-node not found"
        )


# ==================== Distributed _send_to_worker ====================


class TestDistributedSendToWorker:
    """Tests for DistributedScheduler._send_to_worker."""

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )

    @pytest.mark.asyncio
    async def test_send_to_worker_success_stores_result(self, scheduler):
        """On success, stores result in Redis and cleans up running_requests."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.store_result = AsyncMock(return_value=True)
        scheduler._redis_queue = mock_queue

        mock_client = AsyncMock(spec=WorkerClient)
        mock_client.inference = AsyncMock(return_value={
            "status": "success",
            "results": ["generated text"],
            "latency_ms": 150,
        })
        scheduler._worker_client = mock_client

        request = create_scheduling_request(request_id="success-test")
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8000)
        sampling_params = {"temperature": 0.7}

        # Add to running_requests
        from quantumflow.scheduler.scheduler import QueueItem
        scheduler.running_requests["success-test"] = QueueItem(request=request)

        await scheduler._send_to_worker(request, endpoint, sampling_params)

        # Verify store_result was called with correct data
        mock_queue.store_result.assert_called_once()
        call_args = mock_queue.store_result.call_args[0]
        assert call_args[0] == "success-test"
        assert call_args[1]["status"] == "success"
        assert call_args[1]["result"] == ["generated text"]
        assert call_args[1]["latency_ms"] == 150

        # Verify running_requests was cleaned up
        assert "success-test" not in scheduler.running_requests

    @pytest.mark.asyncio
    async def test_send_to_worker_error_calls_dispatch_failure(self, scheduler):
        """When Worker returns error status, calls _handle_dispatch_failure."""
        mock_client = AsyncMock(spec=WorkerClient)
        mock_client.inference = AsyncMock(return_value={
            "status": "error",
            "error": "Model not loaded",
        })
        scheduler._worker_client = mock_client

        scheduler._handle_dispatch_failure = AsyncMock()

        request = create_scheduling_request(request_id="error-test")
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8000)
        sampling_params = {"temperature": 0.7}

        await scheduler._send_to_worker(request, endpoint, sampling_params)

        scheduler._handle_dispatch_failure.assert_called_once_with(
            request, "Model not loaded"
        )

    @pytest.mark.asyncio
    async def test_send_to_worker_exception_calls_dispatch_failure(self, scheduler):
        """When inference throws exception, calls _handle_dispatch_failure."""
        mock_client = AsyncMock(spec=WorkerClient)
        mock_client.inference = AsyncMock(side_effect=RuntimeError("Connection lost"))
        scheduler._worker_client = mock_client

        scheduler._handle_dispatch_failure = AsyncMock()

        request = create_scheduling_request(request_id="exception-test")
        endpoint = WorkerEndpoint(node_id="w1", host="localhost", port=8000)
        sampling_params = {"temperature": 0.7}

        await scheduler._send_to_worker(request, endpoint, sampling_params)

        scheduler._handle_dispatch_failure.assert_called_once_with(
            request, "Connection lost"
        )


# ==================== _handle_dispatch_failure ====================


class TestHandleDispatchFailure:
    """Tests for DistributedScheduler._handle_dispatch_failure."""

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            max_retries=3,
            redis_url="redis://localhost:6379/0",
        )

    @pytest.mark.asyncio
    async def test_retry_path_requeues(self, scheduler):
        """When retry_count < max_retries, requeues to Redis."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.requeue = AsyncMock(return_value=True)
        scheduler._redis_queue = mock_queue

        request = create_scheduling_request(request_id="dispatch-retry")
        # SchedulingRequest has no 'retry_count' attribute by default;
        # _handle_dispatch_failure uses getattr(request, "retry_count", 0)
        # so it defaults to 0

        await scheduler._handle_dispatch_failure(request, "Worker timeout")

        mock_queue.requeue.assert_called_once()
        requeued = mock_queue.requeue.call_args[0][0]
        assert requeued.request_id == "dispatch-retry"
        assert requeued.metadata.get("retry_count") == 1

    @pytest.mark.asyncio
    async def test_final_failure_stores_error_and_updates_stats(self, scheduler):
        """When retry_count >= max_retries, stores error result."""
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.store_result = AsyncMock(return_value=True)
        scheduler._redis_queue = mock_queue

        request = create_scheduling_request(request_id="final-dispatch-fail")
        # Simulate a request that has already been retried enough
        # Patch retry_count onto the request object
        request.retry_count = 3  # equals max_retries

        # Add to running_requests
        from quantumflow.scheduler.scheduler import QueueItem
        scheduler.running_requests["final-dispatch-fail"] = QueueItem(request=request)

        initial_failed = scheduler.stats["failed_requests"]

        await scheduler._handle_dispatch_failure(request, "Persistent error")

        assert scheduler.stats["failed_requests"] == initial_failed + 1
        assert "final-dispatch-fail" not in scheduler.running_requests

        mock_queue.store_result.assert_called_once()
        stored = mock_queue.store_result.call_args[0]
        assert stored[0] == "final-dispatch-fail"
        assert stored[1]["status"] == "error"
        assert stored[1]["reason"] == "Persistent error"


# ==================== start: Redis unavailable fallback ====================


class TestDistributedStart:
    """Tests for DistributedScheduler.start with Redis availability."""

    @pytest.mark.asyncio
    async def test_start_redis_not_available_falls_back(self):
        """When Redis is not available, scheduler starts with memory queue."""
        scheduler = DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )
        scheduler._running = False

        # Mock Redis manager as not connected
        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = False

        with patch(
            "quantumflow.scheduler.distributed.get_redis_manager",
            return_value=mock_redis_mgr,
        ):
            await scheduler.start()

        assert scheduler._running is True
        assert scheduler._use_redis is False
        assert scheduler._scheduling_task is not None

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_already_running_noop(self):
        """Starting an already-running scheduler is a no-op."""
        scheduler = DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )
        scheduler._running = True

        await scheduler.start()

        assert scheduler._running is True
        # _scheduling_task should not have been created
        assert scheduler._scheduling_task is None


# ==================== stop ====================


class TestDistributedStop:
    """Tests for DistributedScheduler.stop."""

    @pytest.mark.asyncio
    async def test_stop_disconnects_redis_queue(self):
        """stop() disconnects the Redis queue if it exists."""
        scheduler = DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )
        scheduler._running = True
        scheduler._worker_client = AsyncMock()
        scheduler._worker_client.close = AsyncMock()

        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.disconnect = AsyncMock()
        scheduler._redis_queue = mock_queue

        await scheduler.stop()

        mock_queue.disconnect.assert_called_once()
        assert scheduler._running is False


# ==================== get_queue_stats ====================


class TestGetQueueStats:
    """Tests for DistributedScheduler.get_queue_stats."""

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )

    def test_get_queue_stats_redis_enabled_flag(self, scheduler):
        """When Redis is enabled, queue_stats includes redis_enabled=True."""
        scheduler._use_redis = True
        mock_queue = AsyncMock(spec=RedisQueue)
        mock_queue.queue_size = AsyncMock(return_value=5)
        scheduler._redis_queue = mock_queue

        stats = scheduler.get_queue_stats()

        assert stats["redis_enabled"] is True
        assert stats["queue_size"] == 5

    def test_get_queue_stats_redis_disabled_flag(self, scheduler):
        """When Redis is disabled, queue_stats uses memory queue size."""
        scheduler._use_redis = False

        stats = scheduler.get_queue_stats()

        assert stats["redis_enabled"] is False
        assert stats["queue_size"] == scheduler.pending_queue.qsize()


# ==================== Global functions ====================


class TestGlobalFunctions:
    """Tests for module-level scheduler functions."""

    def test_get_scheduler_returns_singleton(self):
        """get_scheduler returns the same instance on repeated calls."""
        import quantumflow.scheduler.distributed as dist
        dist._scheduler = None  # Reset for test

        s1 = get_scheduler()
        s2 = get_scheduler()

        assert s1 is s2
        assert isinstance(s1, DistributedScheduler)

    @pytest.mark.asyncio
    async def test_init_scheduler_creates_and_starts(self):
        """init_scheduler creates a DistributedScheduler and starts it."""
        import quantumflow.scheduler.distributed as dist
        dist._scheduler = None

        # Prevent actual Redis connection
        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = False

        with patch(
            "quantumflow.scheduler.distributed.get_redis_manager",
            return_value=mock_redis_mgr,
        ):
            scheduler = await init_scheduler(
                default_strategy="gang",
                redis_url="redis://localhost:6379/1",
            )

        assert isinstance(scheduler, DistributedScheduler)
        # NOTE: init_scheduler accepts default_strategy but does NOT pass it
        # to DistributedScheduler() -- this is a bug in the production code.
        # The parameter is accepted but silently ignored.
        assert scheduler._running is True

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_close_scheduler_stops_and_clears(self):
        """close_scheduler stops the scheduler and sets global to None."""
        import quantumflow.scheduler.distributed as dist
        dist._scheduler = None

        mock_redis_mgr = AsyncMock()
        mock_redis_mgr.is_connected = False

        with patch(
            "quantumflow.scheduler.distributed.get_redis_manager",
            return_value=mock_redis_mgr,
        ):
            scheduler = await init_scheduler(redis_url="redis://localhost:6379/0")

        assert dist._scheduler is not None

        await close_scheduler()

        assert dist._scheduler is None


# ==================== Metadata bug test ====================


class TestMetadataBug:
    """Test capturing the metadata access bug in _dispatch.

    DistributedScheduler._dispatch line 304:
        metadata = getattr(request, "metadata", {}) or {}
    SchedulingRequest has no 'metadata' field (only 'tags' dict exists),
    so this will always return {}, meaning sampling_params always uses defaults.
    """

    @pytest.fixture
    def scheduler(self):
        return DistributedScheduler(
            default_strategy="adaptive",
            redis_url="redis://localhost:6379/0",
        )

    def test_scheduling_request_has_no_metadata_attr(self):
        """SchedulingRequest does NOT have a 'metadata' attribute.

        Documenting this because _dispatch uses getattr(request, "metadata", {})
        which always falls back to default dict, potentially ignoring intended
        sampling parameter overrides.
        """
        request = create_scheduling_request()
        assert not hasattr(request, "metadata"), (
            "SchedulingRequest has no 'metadata' attr; "
            "getattr(request, 'metadata', {}) in _dispatch always returns {}"
        )

    @pytest.mark.asyncio
    async def test_dispatch_uses_default_sampling_params_when_no_metadata(self, scheduler):
        """When SchedulingRequest has no metadata, sampling_params uses all defaults."""
        request = create_scheduling_request(request_id="sampling-test")
        result = SchedulingResult(success=True, assigned_nodes=["w1"],
                                  estimated_wait_time=0.1, strategy_used="pack")

        mock_registry = AsyncMock(spec=WorkerRegistry)
        mock_registry.get_worker = AsyncMock(
            return_value=WorkerEndpoint(node_id="w1", host="localhost", port=8000)
        )
        scheduler._worker_registry = mock_registry

        mock_client = AsyncMock(spec=WorkerClient)
        mock_client.inference = AsyncMock(return_value={"status": "success"})
        scheduler._worker_client = mock_client

        await scheduler._dispatch(request, result)

        # _dispatch creates an asyncio.Task for _send_to_worker, so wait for it
        await asyncio.sleep(0.1)

        # Check the sampling_params sent to the worker
        call_kwargs = scheduler._worker_client.inference.call_args
        assert call_kwargs is not None, "inference should have been called"
        sampling_params = call_kwargs.kwargs["sampling_params"]

        assert sampling_params["temperature"] == 0.7
        assert sampling_params["top_p"] == 0.9
        assert sampling_params["top_k"] == 50
        assert sampling_params["max_tokens"] == 512
        assert sampling_params["repetition_penalty"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
