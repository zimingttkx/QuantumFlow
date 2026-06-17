"""DistributedScheduler comprehensive coverage tests.

Covers all missing lines in distributed.py (currently ~15% coverage):
- Constructor and initialization
- start() with Redis connected / not connected
- stop() with Redis and Worker client cleanup
- submit() via Redis queue and fallback to in-memory queue
- _scheduling_loop with Redis and in-memory paths
- _process_batch_from_redis (dequeue, batch, success, failure)
- _handle_scheduling_failure_redis (retry/requeue, final failure)
- _dispatch (node assignment, worker resolution)
- _send_to_worker (success, failure, exception)
- _handle_dispatch_failure (retry, final failure)
- Worker management (register, unregister, count)
- get_queue_stats (with and without Redis)
- Global functions (get_scheduler, init_scheduler, close_scheduler)
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from quantumflow.scheduler.distributed import (
    DistributedScheduler,
    close_scheduler,
    get_scheduler,
    init_scheduler,
)
from quantumflow.scheduler.strategy.base import (
    GPUResource,
    NodeResource,
    SchedulingRequest,
    SchedulingResult,
)
from quantumflow.scheduler.worker_client import WorkerEndpoint
from quantumflow.storage.redis_queue import QueuedRequest


# =============================================================================
# Helpers
# =============================================================================


def _make_node(nid="n1", load=0.2):
    return NodeResource(
        node_id=nid, hostname=f"s-{nid}", ip=f"10.0.0.{nid}",
        status="healthy", gpu_count=1,
        gpus=[GPUResource(gpu_id=0, memory_total=24 * 1024**3,
                          memory_used=8 * 1024**3, utilization=0.3,
                          temperature=45.0, node_id=nid)],
        cpu_count=4, memory_total=32 * 1024**3, memory_available=16 * 1024**3,
        disk_total=100 * 1024**3, disk_available=50 * 1024**3, load=load,
    )


def _make_req(rid="r1", model="m", priority=5, max_tokens=512, prompt="hello",
              prompt_tokens=10):
    return SchedulingRequest(
        request_id=rid, model=model, prompt=prompt,
        priority=priority, max_tokens=max_tokens,
        model_config={"tensor_parallel": 1},
    )


def _make_qreq(rid="r1", model="m", priority=5):
    return QueuedRequest(
        request_id=rid, model_name=model, prompt="hello",
        priority=priority, created_at=datetime.now(),
    )


def _make_mock_redis_queue(**kwargs):
    """Return a mock RedisQueue with common async methods."""
    q = AsyncMock()
    q.connect = AsyncMock(return_value=True)
    q.disconnect = AsyncMock(return_value=None)
    q.enqueue = AsyncMock(return_value=True)
    q.dequeue = AsyncMock(return_value=None)
    q.requeue = AsyncMock(return_value=True)
    q.store_result = AsyncMock(return_value=True)
    q.queue_size = AsyncMock(return_value=0)
    return q


def _make_mock_redis_manager(connected=True):
    """Return a mock RedisConnectionManager."""
    mgr = MagicMock()
    type(mgr).is_connected = PropertyMock(return_value=connected)
    return mgr


def _make_mock_registry():
    """Return a mock WorkerRegistry."""
    reg = AsyncMock()
    reg.get_worker = AsyncMock(return_value=None)
    reg.register = AsyncMock(return_value=None)
    reg.unregister = AsyncMock(return_value=None)
    reg.get_worker_count = AsyncMock(return_value=0)
    reg.get_all_workers = AsyncMock(return_value=[])
    return reg


def _make_mock_worker_client(inference_return=None):
    """Return a mock WorkerClient."""
    wc = AsyncMock()
    if inference_return is None:
        inference_return = {"status": "success"}
    wc.inference = AsyncMock(return_value=inference_return)
    wc.close = AsyncMock(return_value=None)
    return wc


# =============================================================================
# Constructor
# =============================================================================


class TestDistributedSchedulerConstructor:
    """Tests for DistributedScheduler.__init__ (lines 49-63)."""

    def test_constructor_defaults(self):
        """Constructor initializes with correct default values."""
        ds = DistributedScheduler()
        assert ds.default_strategy == "adaptive"
        assert ds.loop_interval_ms == 100
        assert ds.max_retries == 3
        assert ds.redis_url == "redis://localhost:6379/0"
        assert ds._redis_queue is None
        assert ds._use_redis is True
        assert ds._worker_client is not None
        assert ds._worker_registry is not None
        assert ds._running is False

    def test_constructor_custom_params(self):
        """Constructor accepts and stores custom parameters."""
        ds = DistributedScheduler(
            default_strategy="pack",
            loop_interval_ms=200,
            max_retries=5,
            redis_url="redis://custom:6379/1",
            worker_timeout=60.0,
        )
        assert ds.default_strategy == "pack"
        assert ds.loop_interval_ms == 200
        assert ds.max_retries == 5
        assert ds.redis_url == "redis://custom:6379/1"
        assert ds._worker_client.timeout == 60.0


# =============================================================================
# start()
# =============================================================================


class TestDistributedSchedulerStart:
    """Tests for DistributedScheduler.start() (lines 70-87)."""

    @pytest.mark.asyncio
    async def test_start_with_redis_connected(self):
        """When Redis is connected, _redis_queue is created and connected (lines 77-79)."""
        ds = DistributedScheduler(redis_url="redis://test:6379/0")
        mock_redis_mgr = _make_mock_redis_manager(connected=True)

        with patch(
            "quantumflow.scheduler.distributed.get_redis_manager",
            AsyncMock(return_value=mock_redis_mgr),
        ):
            with patch(
                "quantumflow.scheduler.distributed.RedisQueue",
                _make_mock_redis_queue,
            ):
                await ds.start()

        assert ds._running is True
        assert ds._scheduling_task is not None
        assert ds._redis_queue is not None
        assert ds._use_redis is True

    @pytest.mark.asyncio
    async def test_start_with_redis_not_connected(self):
        """When Redis is not connected, _use_redis is set to False."""
        ds = DistributedScheduler()
        mock_redis_mgr = _make_mock_redis_manager(connected=False)

        with patch(
            "quantumflow.scheduler.distributed.get_redis_manager",
            AsyncMock(return_value=mock_redis_mgr),
        ):
            await ds.start()

        assert ds._running is True
        assert ds._use_redis is False

    @pytest.mark.asyncio
    async def test_start_already_running_returns(self):
        """When already running, start() logs warning and returns."""
        ds = DistributedScheduler()
        ds._running = True

        mock_redis_mgr = _make_mock_redis_manager(connected=True)
        with patch(
            "quantumflow.scheduler.distributed.get_redis_manager",
            AsyncMock(return_value=mock_redis_mgr),
        ):
            await ds.start()

        # Should still be running with no new task created
        assert ds._running is True


# =============================================================================
# stop()
# =============================================================================


class TestDistributedSchedulerStop:
    """Tests for DistributedScheduler.stop() (lines 91-106)."""

    @pytest.mark.asyncio
    async def test_stop_cancels_task_and_closes_resources(self):
        """stop() cancels scheduling task, closes WorkerClient and Redis queue."""
        ds = DistributedScheduler()
        ds._running = True
        # Create a dummy task
        ds._scheduling_task = asyncio.create_task(asyncio.sleep(10))

        mock_redis_queue = _make_mock_redis_queue()
        ds._redis_queue = mock_redis_queue

        await ds.stop()

        assert ds._running is False
        mock_redis_queue.disconnect.assert_called_once()
        # Worker client close is called

    @pytest.mark.asyncio
    async def test_stop_without_redis_queue(self):
        """stop() works when _redis_queue is None."""
        ds = DistributedScheduler()
        ds._running = True
        ds._scheduling_task = asyncio.create_task(asyncio.sleep(10))

        await ds.stop()

        assert ds._running is False

    @pytest.mark.asyncio
    async def test_stop_already_stopped(self):
        """stop() when not running does not raise."""
        ds = DistributedScheduler()
        ds._running = False
        ds._scheduling_task = None

        await ds.stop()
        assert ds._running is False


# =============================================================================
# submit()
# =============================================================================


class TestDistributedSchedulerSubmit:
    """Tests for DistributedScheduler.submit() (lines 121-158)."""

    @pytest.mark.asyncio
    async def test_submit_via_redis_queue(self):
        """submit() enqueues to Redis when connected."""
        ds = DistributedScheduler()
        ds._use_redis = True
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.enqueue = AsyncMock(return_value=True)

        request = _make_req("r1")
        initial_total = ds.stats["total_requests"]
        initial_pending = ds.stats["pending_requests"]

        rid = await ds.submit(request)

        assert rid == "r1"
        assert ds.stats["total_requests"] == initial_total + 1
        assert ds.stats["pending_requests"] == initial_pending + 1
        ds._redis_queue.enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_redis_enqueue_fails_falls_back_to_memory(self):
        """When Redis enqueue fails, submit falls back to memory queue (single failure).

        Bug fix (M-R2): 原本一次失败就 `self._use_redis = False` 永久禁用 Redis,
        改为: 失败计数器累加,达到阈值才临时禁用并启动指数退避。
        一次失败时: 请求走内存队列,但 Redis 仍然标记为可用(下次再试)。
        """
        ds = DistributedScheduler()
        ds._use_redis = True
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.enqueue = AsyncMock(return_value=False)

        request = _make_req("r1")
        initial_qsize = ds.pending_queue.qsize()

        rid = await ds.submit(request)

        assert rid == "r1"
        # Bug fix (M-R2): 一次失败不立即禁用 Redis
        assert ds._use_redis is True, (
            f"Expected _use_redis to remain True after 1 failure (M-R2 fix), "
            f"got {ds._use_redis}"
        )
        # 失败计数器应该累加
        assert ds._redis_consecutive_failures == 1, (
            f"Expected _redis_consecutive_failures=1, got {ds._redis_consecutive_failures}"
        )
        # 请求应该被放到内存队列
        assert ds.pending_queue.qsize() == initial_qsize + 1

    @pytest.mark.asyncio
    async def test_submit_redis_consecutive_failures_disables_with_backoff(self):
        """After threshold consecutive failures, Redis is temporarily disabled with backoff."""
        ds = DistributedScheduler()
        ds._use_redis = True
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.enqueue = AsyncMock(return_value=False)
        # 阈值改为 3 加速测试
        ds._redis_disable_threshold = 3

        # 3 次连续失败
        for i in range(3):
            await ds.submit(_make_req(f"r{i}"))

        # 应该临时禁用
        import time
        assert ds._redis_disabled_at is not None, (
            "After threshold failures, _redis_disabled_at should be set"
        )
        assert ds._redis_retry_after_ts > time.time(), (
            "After threshold failures, _redis_retry_after_ts should be in the future"
        )
        assert ds._redis_consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_submit_falls_back_to_memory_when_no_redis(self):
        """submit() uses memory queue when _use_redis is False."""
        ds = DistributedScheduler()
        ds._use_redis = False

        request = _make_req("r1")
        initial_qsize = ds.pending_queue.qsize()

        rid = await ds.submit(request)

        assert rid == "r1"
        assert ds.pending_queue.qsize() == initial_qsize + 1

    @pytest.mark.asyncio
    async def test_submit_increments_total_requests(self):
        """submit() increments total_requests regardless of path."""
        ds = DistributedScheduler()
        ds._use_redis = False

        initial = ds.stats["total_requests"]
        await ds.submit(_make_req("r1"))
        await ds.submit(_make_req("r2"))

        assert ds.stats["total_requests"] == initial + 2


# =============================================================================
# _scheduling_loop
# =============================================================================


class TestDistributedSchedulingLoop:
    """Tests for DistributedScheduler._scheduling_loop (lines 162-184)."""

    @pytest.mark.asyncio
    async def test_scheduling_loop_redis_path(self):
        """When Redis is available, loop calls _process_batch_from_redis."""
        ds = DistributedScheduler()
        ds._running = True
        ds._use_redis = True
        ds._redis_queue = _make_mock_redis_queue()
        ds.available_nodes["n1"] = _make_node("n1")

        called = [False]

        async def fake_proc_redis():
            called[0] = True
            ds._running = False  # Stop after one iteration

        ds._process_batch_from_redis = fake_proc_redis

        with patch("asyncio.sleep", AsyncMock()):
            await ds._scheduling_loop()

        assert called[0] is True

    @pytest.mark.asyncio
    async def test_scheduling_loop_memory_path(self):
        """When Redis is not available, loop calls _process_batch."""
        ds = DistributedScheduler()
        ds._running = True
        ds._use_redis = False
        ds.available_nodes["n1"] = _make_node("n1")

        called = [False]

        async def fake_proc():
            called[0] = True
            ds._running = False

        ds._process_batch = fake_proc

        with patch("asyncio.sleep", AsyncMock()):
            await ds._scheduling_loop()

        assert called[0] is True

    @pytest.mark.asyncio
    async def test_scheduling_loop_skips_when_no_nodes(self):
        """Scheduling loop skips processing when available_nodes is empty."""
        ds = DistributedScheduler()
        ds._running = True
        iteration_count = [0]

        async def tracked_sleep(d):
            iteration_count[0] += 1
            if iteration_count[0] >= 2:
                ds._running = False

        with patch("asyncio.sleep", tracked_sleep):
            await ds._scheduling_loop()

        assert iteration_count[0] >= 1

    @pytest.mark.asyncio
    async def test_scheduling_loop_exception_handler(self):
        """Scheduling loop catches generic exceptions and sleeps 1s (line 182)."""
        ds = DistributedScheduler()
        ds._running = True
        ds.available_nodes["n1"] = _make_node("n1")

        raise_count = [0]

        async def fake_proc_raise():
            raise_count[0] += 1
            if raise_count[0] == 1:
                raise RuntimeError("simulated loop error")
            ds._running = False

        ds._process_batch = fake_proc_raise

        with patch("asyncio.sleep", AsyncMock()):
            await ds._scheduling_loop()

        assert raise_count[0] >= 2

    @pytest.mark.asyncio
    async def test_scheduling_loop_cancelled_error(self):
        """Scheduling loop handles CancelledError (line 178-179)."""
        ds = DistributedScheduler()
        ds._running = True
        ds.available_nodes["n1"] = _make_node("n1")

        async def mock_sleep_cancel(d):
            raise asyncio.CancelledError()

        with patch("asyncio.sleep", mock_sleep_cancel):
            await ds._scheduling_loop()


# =============================================================================
# _process_batch_from_redis
# =============================================================================


class TestProcessBatchFromRedis:
    """Tests for _process_batch_from_redis (lines 188-224)."""

    @pytest.mark.asyncio
    async def test_batch_from_redis_returns_when_no_queue(self):
        """Returns early when _redis_queue is None."""
        ds = DistributedScheduler()
        ds._redis_queue = None

        await ds._process_batch_from_redis()
        # Should not raise

    @pytest.mark.asyncio
    async def test_batch_from_redis_returns_when_queue_empty(self):
        """Returns when queue size is 0."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.queue_size = AsyncMock(return_value=0)

        await ds._process_batch_from_redis()
        # Should not raise

    @pytest.mark.asyncio
    async def test_batch_from_redis_dequeue_returns_none(self):
        """When dequeue returns None, batch loop breaks (line 200)."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.queue_size = AsyncMock(return_value=5)
        ds._redis_queue.dequeue = AsyncMock(return_value=None)

        await ds._process_batch_from_redis()
        # Should not raise - break on None, then check if empty -> return

    @pytest.mark.asyncio
    async def test_batch_from_redis_returns_on_empty_batch(self):
        """When requests_batch is empty after dequeue, returns (line 213)."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.queue_size = AsyncMock(return_value=1)
        # dequeue returns None
        ds._redis_queue.dequeue = AsyncMock(return_value=None)

        await ds._process_batch_from_redis()
        # Should return without processing

    @pytest.mark.asyncio
    async def test_batch_from_redis_success_path(self):
        """On successful schedule, increments stats and dispatches (lines 220-221)."""
        ds = DistributedScheduler()
        ds.available_nodes["n1"] = _make_node("n1")
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.queue_size = AsyncMock(return_value=1)

        qreq = _make_qreq("r1", "test-model")
        ds._redis_queue.dequeue = AsyncMock(return_value=qreq)

        # Make _dispatch a mock to avoid real worker communication
        ds._dispatch = AsyncMock()
        # Make _scheduling_loop stop
        ds.stats["pending_requests"] = 1

        initial_success = ds.stats["successful_requests"]

        await ds._process_batch_from_redis()

        assert ds.stats["successful_requests"] == initial_success + 1
        ds._dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_from_redis_schedule_failure(self):
        """When scheduling fails, calls _handle_scheduling_failure_redis."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.queue_size = AsyncMock(return_value=1)

        qreq = _make_qreq("r1", "test-model")
        ds._redis_queue.dequeue = AsyncMock(return_value=qreq)

        ds.stats["pending_requests"] = 1
        initial_failed = ds.stats["failed_requests"]

        ds._handle_scheduling_failure_redis = AsyncMock()

        # No available nodes -> _schedule_request will fail
        await ds._process_batch_from_redis()

        assert ds.stats["failed_requests"] == initial_failed + 1
        ds._handle_scheduling_failure_redis.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_caps_at_10(self):
        """Batch size is capped at 10."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()
        ds._redis_queue.queue_size = AsyncMock(return_value=15)

        dequeue_count = [0]

        async def count_dequeue(timeout=0):
            dequeue_count[0] += 1
            return None

        ds._redis_queue.dequeue = count_dequeue

        await ds._process_batch_from_redis()

        # batch_size = min(15, 10) = 10, so dequeue called at most 10 times
        assert dequeue_count[0] <= 10


# =============================================================================
# _handle_scheduling_failure_redis
# =============================================================================


class TestHandleSchedulingFailureRedis:
    """Tests for _handle_scheduling_failure_redis (lines 230-263)."""

    @pytest.mark.asyncio
    async def test_failure_redis_retry_requeues(self):
        """When retry_count < max_retries, request is requeued."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()

        request = SchedulingRequest(
            request_id="r1", model="m", priority=5, retry_count=0,
        )
        result = SchedulingResult(success=False, reason="No resources")

        await ds._handle_scheduling_failure_redis(request, result)

        assert request.retry_count == 1
        ds._redis_queue.requeue.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_redis_final_failure(self):
        """When retries exhausted, stats updated and result stored."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()

        request = SchedulingRequest(
            request_id="r1", model="m", priority=5, retry_count=3,
        )
        result = SchedulingResult(success=False, reason="No resources")

        initial_failed = ds.stats["failed_requests"]
        # Set pending to 1 before call, code will decrement it
        ds.stats["pending_requests"] = 1

        await ds._handle_scheduling_failure_redis(request, result)

        assert ds.stats["failed_requests"] == initial_failed + 1
        # pending_requests was 1, decremented by 1 = 0
        assert ds.stats["pending_requests"] == 0
        ds._redis_queue.store_result.assert_called_once()


# =============================================================================
# _dispatch (distributed)
# =============================================================================


class TestDistributedDispatch:
    """Tests for DistributedScheduler._dispatch (lines 270-314)."""

    @pytest.mark.asyncio
    async def test_dispatch_no_nodes_assigned(self):
        """When no nodes assigned, calls _handle_dispatch_failure."""
        ds = DistributedScheduler()
        request = _make_req("r1")
        result = SchedulingResult(success=True, assigned_nodes=[])

        ds._handle_dispatch_failure = AsyncMock()

        await ds._dispatch(request, result)

        ds._handle_dispatch_failure.assert_called_once_with(request, "No nodes assigned")

    @pytest.mark.asyncio
    async def test_dispatch_worker_not_found(self):
        """When worker not found in registry, calls _handle_dispatch_failure."""
        ds = DistributedScheduler()
        request = _make_req("r1")
        result = SchedulingResult(success=True, assigned_nodes=["n1"])

        ds._worker_registry.get_worker = AsyncMock(return_value=None)
        ds._handle_dispatch_failure = AsyncMock()

        await ds._dispatch(request, result)

        ds._handle_dispatch_failure.assert_called_once()
        assert "Worker n1 not found" in ds._handle_dispatch_failure.call_args.args[1]

    @pytest.mark.asyncio
    async def test_dispatch_success_updates_pending_and_running(self):
        """Successful dispatch decrements pending_requests and adds to running."""
        ds = DistributedScheduler()
        request = _make_req("r1")
        result = SchedulingResult(success=True, assigned_nodes=["n1"])

        endpoint = WorkerEndpoint(node_id="n1", host="10.0.0.1", port=8000)
        ds._worker_registry.get_worker = AsyncMock(return_value=endpoint)

        ds.stats["pending_requests"] = 1
        ds.stats["successful_requests"] = 0

        await ds._dispatch(request, result)

        assert ds.stats["pending_requests"] == 0
        assert "r1" in ds.running_requests


# =============================================================================
# _send_to_worker (distributed)
# =============================================================================


class TestDistributedSendToWorker:
    """Tests for DistributedScheduler._send_to_worker (lines 323-377)."""

    @pytest.mark.asyncio
    async def test_send_to_worker_success(self):
        """Worker inference success stores result and cleans up running_requests."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()
        ds.running_requests["r1"] = MagicMock()

        request = _make_req("r1")
        endpoint = WorkerEndpoint(node_id="n1", host="10.0.0.1", port=8000)
        sampling_params = {"temperature": 0.7}

        mock_wc = _make_mock_worker_client(
            inference_return={"status": "success", "results": ["ok"], "latency_ms": 50}
        )
        ds._worker_client = mock_wc

        await ds._send_to_worker(request, endpoint, sampling_params)

        mock_wc.inference.assert_called_once()
        ds._redis_queue.store_result.assert_called_once()
        assert "r1" not in ds.running_requests

    @pytest.mark.asyncio
    async def test_send_to_worker_failure(self):
        """Worker inference failure calls _handle_dispatch_failure."""
        ds = DistributedScheduler()
        ds.running_requests["r1"] = MagicMock()
        ds._handle_dispatch_failure = AsyncMock()

        request = _make_req("r1")
        endpoint = WorkerEndpoint(node_id="n1", host="10.0.0.1", port=8000)
        sampling_params = {"temperature": 0.7}

        mock_wc = _make_mock_worker_client(
            inference_return={"status": "error", "error": "Model not loaded"}
        )
        ds._worker_client = mock_wc

        await ds._send_to_worker(request, endpoint, sampling_params)

        ds._handle_dispatch_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_worker_exception(self):
        """Inference exception calls _handle_dispatch_failure."""
        ds = DistributedScheduler()
        ds.running_requests["r1"] = MagicMock()
        ds._handle_dispatch_failure = AsyncMock()

        request = _make_req("r1")
        endpoint = WorkerEndpoint(node_id="n1", host="10.0.0.1", port=8000)
        sampling_params = {"temperature": 0.7}

        mock_wc = _make_mock_worker_client()
        mock_wc.inference = AsyncMock(side_effect=ConnectionError("Connection refused"))
        ds._worker_client = mock_wc

        await ds._send_to_worker(request, endpoint, sampling_params)

        ds._handle_dispatch_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_worker_success_without_redis_queue(self):
        """Success without Redis queue still cleans up running_requests."""
        ds = DistributedScheduler()
        ds._redis_queue = None
        ds.running_requests["r1"] = MagicMock()

        request = _make_req("r1")
        endpoint = WorkerEndpoint(node_id="n1", host="10.0.0.1", port=8000)
        sampling_params = {"temperature": 0.7}

        mock_wc = _make_mock_worker_client()
        ds._worker_client = mock_wc

        await ds._send_to_worker(request, endpoint, sampling_params)

        assert "r1" not in ds.running_requests


# =============================================================================
# _handle_dispatch_failure
# =============================================================================


class TestHandleDispatchFailure:
    """Tests for _handle_dispatch_failure (lines 381-417)."""

    @pytest.mark.asyncio
    async def test_dispatch_failure_retry_requeues(self):
        """When retry_count < max_retries, request is requeued via Redis."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()

        request = SchedulingRequest(request_id="r1", model="m", retry_count=1)

        await ds._handle_dispatch_failure(request, "Worker error")

        assert ds._redis_queue.requeue.called

    @pytest.mark.asyncio
    async def test_dispatch_failure_final_stores_result(self):
        """When retries exhausted, stores error result and updates stats."""
        ds = DistributedScheduler()
        ds._redis_queue = _make_mock_redis_queue()
        ds.running_requests["r1"] = MagicMock()

        request = SchedulingRequest(request_id="r1", model="m", retry_count=3)
        initial_failed = ds.stats["failed_requests"]

        await ds._handle_dispatch_failure(request, "Fatal error")

        ds._redis_queue.store_result.assert_called_once()
        assert "r1" not in ds.running_requests
        assert ds.stats["failed_requests"] == initial_failed + 1

    @pytest.mark.asyncio
    async def test_dispatch_failure_final_without_redis(self):
        """Final failure without Redis queue still cleans up."""
        ds = DistributedScheduler()
        ds._redis_queue = None
        ds.running_requests["r1"] = MagicMock()

        request = SchedulingRequest(request_id="r1", model="m", retry_count=3)

        await ds._handle_dispatch_failure(request, "Fatal")

        assert "r1" not in ds.running_requests

    @pytest.mark.asyncio
    async def test_dispatch_failure_retry_without_redis(self):
        """Retry without Redis queue does not requeue to Redis."""
        ds = DistributedScheduler()
        ds._redis_queue = None

        request = SchedulingRequest(request_id="r1", model="m", retry_count=1)

        await ds._handle_dispatch_failure(request, "Worker error")
        # Should not raise - just logs warning


# =============================================================================
# Worker management
# =============================================================================


class TestDistributedWorkerManagement:
    """Tests for register_worker, unregister_worker, get_worker_count (lines 438-470)."""

    @pytest.mark.asyncio
    async def test_register_worker(self):
        """register_worker creates WorkerEndpoint and registers it."""
        ds = DistributedScheduler()
        ds._worker_registry.register = AsyncMock()

        result = await ds.register_worker("n1", "10.0.0.1", 8080)

        assert result is True
        ds._worker_registry.register.assert_called_once()
        call_arg = ds._worker_registry.register.call_args.args[0]
        assert call_arg.node_id == "n1"
        assert call_arg.host == "10.0.0.1"
        assert call_arg.port == 8080
        assert call_arg.status == "healthy"

    @pytest.mark.asyncio
    async def test_unregister_worker(self):
        """unregister_worker unregisters worker from registry."""
        ds = DistributedScheduler()
        ds._worker_registry.unregister = AsyncMock()

        result = await ds.unregister_worker("n1")

        assert result is True
        ds._worker_registry.unregister.assert_called_once_with("n1")

    @pytest.mark.asyncio
    async def test_get_worker_count(self):
        """get_worker_count returns count from registry."""
        ds = DistributedScheduler()
        ds._worker_registry.get_worker_count = AsyncMock(return_value=5)

        count = await ds.get_worker_count()

        assert count == 5


# =============================================================================
# get_queue_stats
# =============================================================================


class TestGetQueueStats:
    """Tests for get_queue_stats."""

    def test_get_queue_stats_sync_without_redis(self):
        """Without Redis, sync stats include redis_enabled=False and use memory queue size."""
        ds = DistributedScheduler()
        ds._use_redis = False

        stats = ds.get_queue_stats()

        assert stats["redis_enabled"] is False
        assert stats["queue_size"] == 0
        assert "total_requests" in stats

    def test_get_queue_stats_sync_with_redis_uses_memory_fallback(self):
        """Sync path with Redis returns local memory queue size (no await)."""
        ds = DistributedScheduler()
        ds._use_redis = True
        ds._redis_queue = MagicMock()

        stats = ds.get_queue_stats()

        assert stats["redis_enabled"] is True
        assert "queue_size" in stats
        # 同步路径下 queue_size 来自 pending_queue 兜底（不是 Redis）
        assert stats["queue_size"] == 0

    def test_get_queue_stats_sync_with_redis_queue_none(self):
        """When _use_redis is True but _redis_queue is None, falls back to memory."""
        ds = DistributedScheduler()
        ds._use_redis = True
        ds._redis_queue = None

        stats = ds.get_queue_stats()

        assert stats["redis_enabled"] is False
        assert "queue_size" in stats

    @pytest.mark.asyncio
    async def test_get_queue_stats_async_with_redis(self):
        """Async path with Redis correctly awaits queue_size."""
        ds = DistributedScheduler()
        ds._use_redis = True
        ds._redis_queue = MagicMock()
        ds._redis_queue.queue_size = AsyncMock(return_value=10)

        stats = await ds.get_queue_stats_async()

        assert stats["redis_enabled"] is True
        assert stats["queue_size"] == 10
        assert "total_requests" in stats


# =============================================================================
# Global functions
# =============================================================================


class TestGlobalFunctions:
    """Tests for get_scheduler, init_scheduler, close_scheduler (lines 497-518)."""

    def teardown_method(self):
        """Reset global scheduler singleton after each test."""
        import quantumflow.scheduler.distributed as mod
        mod._scheduler = None

    @pytest.mark.asyncio
    async def test_get_scheduler_creates_singleton(self):
        """get_scheduler returns a DistributedScheduler instance."""
        import quantumflow.scheduler.distributed as mod
        mod._scheduler = None

        s1 = get_scheduler()
        s2 = get_scheduler()

        assert isinstance(s1, DistributedScheduler)
        assert s1 is s2  # Singleton

    @pytest.mark.asyncio
    async def test_init_scheduler(self):
        """init_scheduler creates and starts a scheduler."""
        import quantumflow.scheduler.distributed as mod
        mod._scheduler = None

        mock_redis_mgr = _make_mock_redis_manager(connected=True)
        with patch(
            "quantumflow.scheduler.distributed.get_redis_manager",
            AsyncMock(return_value=mock_redis_mgr),
        ):
            with patch(
                "quantumflow.scheduler.distributed.RedisQueue",
                _make_mock_redis_queue,
            ):
                s = await init_scheduler()

        assert isinstance(s, DistributedScheduler)
        assert s._running is True

    @pytest.mark.asyncio
    async def test_close_scheduler(self):
        """close_scheduler stops and clears the global scheduler."""
        import quantumflow.scheduler.distributed as mod

        mock_redis_mgr = _make_mock_redis_manager(connected=True)
        with patch(
            "quantumflow.scheduler.distributed.get_redis_manager",
            AsyncMock(return_value=mock_redis_mgr),
        ):
            with patch(
                "quantumflow.scheduler.distributed.RedisQueue",
                _make_mock_redis_queue,
            ):
                mod._scheduler = DistributedScheduler()
                mod._scheduler._running = True
                mod._scheduler._scheduling_task = asyncio.create_task(asyncio.sleep(10))

                await close_scheduler()

                assert mod._scheduler is None

    @pytest.mark.asyncio
    async def test_close_scheduler_when_none(self):
        """close_scheduler when _scheduler is None does not raise."""
        import quantumflow.scheduler.distributed as mod
        mod._scheduler = None

        await close_scheduler()
        # Should not raise

    def test_get_scheduler_returns_existing(self):
        """get_scheduler returns existing instance when already set."""
        import quantumflow.scheduler.distributed as mod
        existing = DistributedScheduler()
        mod._scheduler = existing

        result = get_scheduler()

        assert result is existing
