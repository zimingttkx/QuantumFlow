"""BatchAccumulator 优先级调度测试

测试策略：
1. 优先级排序正确性
2. 优先级对处理顺序的影响
3. Anti-starvation 机制
4. 优先级统计准确性
"""

import asyncio
import sys
from pathlib import Path
import time
from unittest.mock import AsyncMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantumflow.inference.batch_accumulator import BatchAccumulator
from quantumflow.inference.engine import InferenceResult, QueuedRequest, SamplingParams


class TestPriorityOrdering:
    """优先级排序测试"""

    @pytest.fixture
    def accumulator(self):
        """创建启用了优先级的 BatchAccumulator"""
        infer_fn = AsyncMock(return_value=[])
        return BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=100.0,
            max_batch_size=16,
            enable_priority=True,
        )

    @pytest.mark.asyncio
    async def test_high_priority_processed_first(self, accumulator):
        """高优先级请求应该先于低优先级请求在批处理中被处理"""
        # 设置 mock 返回值，result_i 对应 batch 中第 i 个请求
        accumulator._infer_fn = AsyncMock(
            return_value=[
                InferenceResult(
                    request_id=f"batch_{i}",
                    outputs=[f"result_{i}"],
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=1,
                    finish_reason="stop",
                    metrics={},
                )
                for i in range(3)
            ]
        )

        # 提交 3 个请求：优先级从低到高提交
        # 但由于排序，处理顺序应该是：高 -> 中 -> 低
        # 结果返回顺序是 submit 顺序：低 -> 高 -> 中
        low_priority = accumulator.submit("low_prompt", priority=8)
        time.sleep(0.01)  # 确保时间戳不同
        high_priority = accumulator.submit("high_prompt", priority=2)
        time.sleep(0.01)
        medium_priority = accumulator.submit("medium_prompt", priority=5)

        results = await asyncio.gather(low_priority, high_priority, medium_priority)

        # results 是按 submit 顺序返回的：低, 高, 中
        # 所以 results[0] = low 的结果，results[1] = high 的结果，results[2] = medium 的结果

        # 验证：submit 顺序是 low(0) -> high(1) -> medium(2)
        # 但处理顺序是 high(0) -> medium(1) -> low(2)
        # infer_fn 返回 [result_0, result_1, result_2]
        # - result_0 给 batch 中位置 0 的请求 (high) -> high_future -> results[1]
        # - result_1 给 batch 中位置 1 的请求 (medium) -> medium_future -> results[2]
        # - result_2 给 batch 中位置 2 的请求 (low) -> low_future -> results[0]

        # 因此：results[0] = result_2, results[1] = result_0, results[2] = result_1
        assert results[0].outputs[0] == "result_2"  # low 先提交，得到最后处理的结果
        assert results[1].outputs[0] == "result_0"  # high 提交第二，得到第一处理的结果
        assert results[2].outputs[0] == "result_1"  # medium 最后提交，得到第二处理的结果

    @pytest.mark.asyncio
    async def test_same_priority_fifo_order(self, accumulator):
        """相同优先级的请求应该按 FIFO 顺序处理"""
        accumulator._infer_fn = AsyncMock(
            return_value=[
                InferenceResult(
                    request_id=f"batch_{i}",
                    outputs=[f"result_{i}"],
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=1,
                    finish_reason="stop",
                    metrics={},
                )
                for i in range(3)
            ]
        )

        # 提交 3 个相同优先级的请求
        r1 = accumulator.submit("first", priority=5)
        r2 = accumulator.submit("second", priority=5)
        r3 = accumulator.submit("third", priority=5)

        results = await asyncio.gather(r1, r2, r3)

        # FIFO 顺序
        assert results[0].outputs[0] == "result_0"  # 先提交先处理
        assert results[1].outputs[0] == "result_1"
        assert results[2].outputs[0] == "result_2"

    @pytest.mark.asyncio
    async def test_priority_range_0_to_10(self, accumulator):
        """测试优先级范围 0-10"""
        accumulator._infer_fn = AsyncMock(
            return_value=[
                InferenceResult(
                    request_id=f"batch_{i}",
                    outputs=[f"result_{i}"],
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=1,
                    finish_reason="stop",
                    metrics={},
                )
                for i in range(11)
            ]
        )

        # 提交 11 个请求，每个不同的优先级
        submitted_requests = []
        for i in range(11):
            coro = accumulator.submit(f"prompt_{i}", priority=i)
            submitted_requests.append(coro)

        # 等待所有结果
        results = await asyncio.gather(*submitted_requests)

        # 由于 infer_fn 按提交顺序返回结果，
        # 但处理按优先级顺序，所以我们验证：
        # 1. 高优先级的请求（priority=0）应该被最先处理
        # 2. 即便它后提交，也应该先返回
        # 3. 由于批处理，所有请求在同一个 batch 中按优先级排序处理
        # 结果返回顺序是 submit 顺序，但处理是优先级顺序

        # 验证返回的 outputs 顺序与 submit 顺序一致
        # (因为我们用 request_id 关联)
        for i, result in enumerate(results):
            assert result.outputs[0] == f"result_{i}"


class TestAntiStarvation:
    """Anti-starvation 机制测试"""

    @pytest.mark.asyncio
    async def test_low_priority_starvation_count(self):
        """低优先级请求被处理后，starvation 计数应该增加"""
        infer_fn = AsyncMock(
            return_value=[
                InferenceResult(
                    request_id="low_prio",
                    outputs=["low"],
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=1,
                    finish_reason="stop",
                    metrics={},
                )
            ]
        )

        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=50.0,
            max_batch_size=4,
            enable_priority=True,
            anti_starvation_threshold=5,
        )

        # 提交低优先级请求
        await acc.submit("low", priority=9)

        # 等待处理完成
        await asyncio.sleep(0.2)

        # 验证 starvation 计数增加
        assert acc.stats["low_priority_starved"] >= 1

    @pytest.mark.asyncio
    async def test_high_priority_processed_count(self):
        """高优先级请求被处理后，计数应该增加"""
        infer_fn = AsyncMock(
            return_value=[
                InferenceResult(
                    request_id="high_prio",
                    outputs=["high"],
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=1,
                    finish_reason="stop",
                    metrics={},
                )
            ]
        )

        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=50.0,
            max_batch_size=4,
            enable_priority=True,
        )

        # 提交高优先级请求
        await acc.submit("high", priority=2)

        # 等待处理完成
        await asyncio.sleep(0.2)

        # 验证高优先级计数增加
        assert acc.stats["high_priority_processed"] >= 1


class TestPriorityStatistics:
    """优先级统计测试"""

    @pytest.mark.asyncio
    async def test_stats_track_priority_correctly(self):
        """统计应该正确追踪高/低优先级请求"""
        infer_fn = AsyncMock(
            side_effect=lambda prompts: [
                InferenceResult(
                    request_id=f"batch_{i}",
                    outputs=[f"result_{i}"],
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=1,
                    finish_reason="stop",
                    metrics={},
                )
                for i in range(len(prompts))
            ]
        )

        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=50.0,
            max_batch_size=16,
            enable_priority=True,
        )

        # 提交混合优先级请求
        tasks = [
            acc.submit("p1", priority=2),  # 高
            acc.submit("p2", priority=5),  # 中
            acc.submit("p3", priority=8),  # 低
        ]
        await asyncio.gather(*tasks)

        # 等待批次处理
        await asyncio.sleep(0.2)

        stats = acc.stats
        assert stats["total_requests"] == 3
        assert stats["high_priority_processed"] >= 1
        assert stats["low_priority_starved"] >= 1


class TestQueuedRequestOrdering:
    """QueuedRequest 排序测试"""

    def test_queued_request_lt_same_priority(self):
        """相同优先级时，按 submit_time 排序"""
        earlier = QueuedRequest(
            request_id="1",
            model_name="test",
            prompt="test",
            sampling_params=SamplingParams(),
            priority=5,
            submit_time=1000.0,
        )
        later = QueuedRequest(
            request_id="2",
            model_name="test",
            prompt="test",
            sampling_params=SamplingParams(),
            priority=5,
            submit_time=1001.0,
        )

        # 相同优先级时，早提交的更小
        assert earlier < later

    def test_queued_request_lt_different_priority(self):
        """不同优先级时，优先级低的更小"""
        low_prio = QueuedRequest(
            request_id="1",
            model_name="test",
            prompt="test",
            sampling_params=SamplingParams(),
            priority=8,
            submit_time=1000.0,
        )
        high_prio = QueuedRequest(
            request_id="2",
            model_name="test",
            prompt="test",
            sampling_params=SamplingParams(),
            priority=2,
            submit_time=1000.0,
        )

        # 高优先级（数值小）更小
        assert high_prio < low_prio

    def test_queued_request_priority_takes_precedence(self):
        """优先级差异优先于时间差异"""
        # 高优先级但晚提交
        high_late = QueuedRequest(
            request_id="1",
            model_name="test",
            prompt="test",
            sampling_params=SamplingParams(),
            priority=2,
            submit_time=2000.0,
        )
        # 低优先级但早提交
        low_early = QueuedRequest(
            request_id="2",
            model_name="test",
            prompt="test",
            sampling_params=SamplingParams(),
            priority=8,
            submit_time=1000.0,
        )

        # 高优先级即使晚提交也优先
        assert high_late < low_early


class TestBatchAccumulatorWithPriorityDisabled:
    """禁用优先级时的测试"""

    @pytest.mark.asyncio
    async def test_priority_disabled_fifo(self):
        """禁用优先级时，按 FIFO 顺序处理"""
        infer_fn = AsyncMock(
            return_value=[
                InferenceResult(
                    request_id=f"batch_{i}",
                    outputs=[f"result_{i}"],
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=1,
                    finish_reason="stop",
                    metrics={},
                )
                for i in range(3)
            ]
        )

        # 禁用优先级
        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=50.0,
            max_batch_size=16,
            enable_priority=False,
        )

        # 按优先级从高到低提交
        r1 = acc.submit("first", priority=2)
        await asyncio.sleep(0.01)
        r2 = acc.submit("second", priority=5)
        await asyncio.sleep(0.01)
        r3 = acc.submit("third", priority=8)

        results = await asyncio.gather(r1, r2, r3)

        # 禁用优先级时，按 FIFO 处理
        assert results[0].outputs[0] == "result_0"
        assert results[1].outputs[0] == "result_1"
        assert results[2].outputs[0] == "result_2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
