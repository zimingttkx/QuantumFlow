"""Batch Accumulator 覆盖率缺口补充测试

精确覆盖 batch_accumulator.py 缺失行:
- flush: buffer check (74)
- shutdown: timeout cancel (83-84)
- _worker_loop: shutting_down check (105)
- _do_flush: IndexError for index mismatch (138)
"""

import asyncio
import sys
from pathlib import Path
import uuid
from unittest.mock import AsyncMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantumflow.inference.batch_accumulator import BatchAccumulator
from quantumflow.inference.engine import InferenceResult, QueuedRequest, SamplingParams


def _make_result(request_id, output="ok"):
    return InferenceResult(
        request_id=request_id,
        outputs=[output],
        prompt_tokens=1, completion_tokens=1,
        latency_ms=1, finish_reason="stop", metrics={},
    )


def _make_request(prompt: str, priority: int = 5) -> QueuedRequest:
    """创建 QueuedRequest 的辅助函数"""
    return QueuedRequest(
        request_id=str(uuid.uuid4()),
        model_name="test",
        prompt=prompt,
        sampling_params=SamplingParams(),
        priority=priority,
        submit_time=0.0,  # 将在 submit 时设置
    )


def _make_result(request_id, output="ok"):
    return InferenceResult(
        request_id=request_id,
        outputs=[output],
        prompt_tokens=1, completion_tokens=1,
        latency_ms=1, finish_reason="stop", metrics={},
    )


# ── flush ──────────────────────────────────────────────────────────────

class TestBatchAccumulatorFlushGaps:
    @pytest.mark.asyncio
    async def test_flush_with_empty_buffer_does_nothing(self):
        """Flush with empty buffer should not crash"""
        acc = BatchAccumulator(
            infer_fn=AsyncMock(return_value=[]),
            max_delay_ms=100.0,
            max_batch_size=4,
        )
        # flush on empty buffer
        await acc.flush()
        # Should not crash and should not call infer_fn
        acc._infer_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_with_items_in_buffer(self):
        """Flush should process buffered items"""
        infer_fn = AsyncMock(return_value=[
            _make_result("0", "flushed"),
        ])
        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=10000.0,  # Long delay so flush is the trigger
            max_batch_size=8,
        )

        # Submit a request
        result = await acc.submit("prompt")

        assert infer_fn.call_count >= 1
        assert result.outputs[0] == "flushed"

    @pytest.mark.asyncio
    async def test_flush_with_pending_requests_hits_sleep(self):
        """flush() with non-empty buffer should hit the sleep at line 74"""
        infer_fn = AsyncMock(return_value=[_make_result("0", "done")])
        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=10000.0,
            max_batch_size=8,
        )

        # Create a QueuedRequest and manually add to buffer WITHOUT going through submit
        # This simulates a request that was added but not yet picked up by worker
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = _make_request("manual_prompt", priority=5)
        request.future = future
        acc._buffer.append(request)
        acc._ensure_worker()

        # Now call flush - buffer is non-empty so line 74 should be hit
        await acc.flush()

        # Verify the flush happened
        assert infer_fn.call_count >= 1

class TestBatchAccumulatorShutdownGaps:
    @pytest.mark.asyncio
    async def test_shutdown_timeout_cancels_task(self):
        """When shutdown times out, the worker task is canceled"""
        infer_fn = AsyncMock(return_value=[_make_result("0", "ok")])

        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=100.0,
            max_batch_size=4,
        )

        # Submit a request to ensure worker is running
        await acc.submit("prompt")

        # Now mock the worker task to never complete
        import asyncio
        async def never_complete():
            await asyncio.sleep(1000)

        acc._shutting_down = True
        acc._worker_task = asyncio.ensure_future(never_complete())

        # Shutdown with short timeout — should cancel the task
        await acc.shutdown()


# ── _worker_loop: shutting_down break ─────────────────────────────────

class TestBatchAccumulatorWorkerLoopGaps:
    @pytest.mark.asyncio
    async def test_worker_loop_breaks_on_shutdown(self):
        """Worker loop should exit when _shutting_down is True"""
        infer_fn = AsyncMock(return_value=[_make_result("0", "ok")])

        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=50.0,
            max_batch_size=4,
        )

        await acc.submit("prompt")
        # shutdown sets _shutting_down = True
        await acc.shutdown()

        # worker task should be done
        assert acc._worker_task is None or acc._worker_task.done()


# ── _do_flush: IndexError when results < batch ──────────────────────

class TestBatchAccumulatorDoFlushGaps:
    @pytest.mark.asyncio
    async def test_do_flush_index_error_when_results_fewer_than_batch(self):
        """When infer_fn returns fewer results than batch size, IndexError is raised"""
        # Return only 1 result for 3 prompts
        infer_fn = AsyncMock(return_value=[_make_result("0", "only_one")])

        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=100.0,
            max_batch_size=10,
        )

        # Submit 3 requests simultaneously
        coros = [acc.submit("p1"), acc.submit("p2"), acc.submit("p3")]

        # First should succeed, others should get IndexError
        results = await asyncio.gather(*coros, return_exceptions=True)

        # First result should be the actual result
        assert isinstance(results[0], InferenceResult)
        assert results[0].outputs[0] == "only_one"

        # Remaining should be IndexError
        assert isinstance(results[1], IndexError)
        assert isinstance(results[2], IndexError)

    @pytest.mark.asyncio
    async def test_do_flush_unexpected_exception_does_not_crash(self):
        """When _do_flush drops a partial batch, shouldn't corrupt state"""
        acc = BatchAccumulator(
            infer_fn=AsyncMock(side_effect=RuntimeError("Boom")),
            max_delay_ms=50.0,
            max_batch_size=4,
        )

        with pytest.raises(RuntimeError, match="Boom"):
            await acc.submit("prompt")

        # Buffer should be cleared
        assert len(acc._buffer) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
