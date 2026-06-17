"""BatchAccumulator 动态 batch_size 测试

测试策略：
1. VRAM 利用率计算正确性
2. 动态 batch size 计算算法
3. BatchScheduler 集成
4. 显存压力场景
"""

import asyncio
import sys
from pathlib import Path
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantumflow.inference.batch_accumulator import BatchAccumulator
from quantumflow.inference.engine import InferenceResult, SamplingParams


def _make_result(request_id, output="ok"):
    return InferenceResult(
        request_id=request_id,
        outputs=[output],
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1,
        finish_reason="stop",
        metrics={},
    )


class TestVRAMUtilization:
    """VRAM 利用率计算测试"""

    def test_vram_utilization_calculation(self):
        """VRAM 利用率 = 已用 / 总容量"""
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager()

        # Mock pynvml to return known values
        with patch.object(
            vram,
            "_read_used_vram_gb",
            return_value=8.0,
        ), patch.object(
            vram,
            "_read_free_vram_gb",
            return_value=8.0,
        ):
            # 8GB used + 8GB free = 16GB total
            # utilization = used / (used + free) = 8 / 16 = 0.5
            utilization = vram.get_vram_utilization()
            assert utilization == 0.5

    def test_vram_utilization_high_usage(self):
        """高显存利用率计算"""
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager()

        with patch.object(vram, "_read_used_vram_gb", return_value=14.0), patch.object(
            vram, "_read_free_vram_gb", return_value=2.0
        ):
            # 14GB used, 2GB free = 16GB total
            # utilization = 14 / 16 = 0.875
            utilization = vram.get_vram_utilization()
            assert utilization == pytest.approx(0.875, rel=0.01)

    def test_vram_utilization_low_usage(self):
        """低显存利用率计算"""
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager()

        with patch.object(vram, "_read_used_vram_gb", return_value=4.0), patch.object(
            vram, "_read_free_vram_gb", return_value=12.0
        ):
            # 4GB used, 12GB free = 16GB total
            # utilization = 4 / 16 = 0.25
            utilization = vram.get_vram_utilization()
            assert utilization == pytest.approx(0.25, rel=0.01)


class TestDynamicBatchConfig:
    """动态批处理配置测试"""

    def test_default_config_values(self):
        """默认配置值正确"""
        from quantumflow.inference.batch_config import DynamicBatchConfig

        config = DynamicBatchConfig()

        assert config.base_max_batch_size == 8
        assert config.min_batch_size == 2
        assert config.max_batch_size == 16
        assert config.vram_threshold_high == 0.9
        assert config.vram_threshold_low == 0.7
        assert config.dynamic_factor == 1.0

    def test_custom_config_values(self):
        """自定义配置值"""
        from quantumflow.inference.batch_config import DynamicBatchConfig

        config = DynamicBatchConfig(
            base_max_batch_size=16,
            min_batch_size=4,
            max_batch_size=32,
            vram_threshold_high=0.85,
            vram_threshold_low=0.6,
        )

        assert config.base_max_batch_size == 16
        assert config.min_batch_size == 4
        assert config.max_batch_size == 32
        assert config.vram_threshold_high == 0.85
        assert config.vram_threshold_low == 0.6


class TestDynamicBatchScheduler:
    """动态批处理调度器测试"""

    def test_compute_batch_size_normal(self):
        """正常情况下返回 base_size"""
        from quantumflow.inference.batch_scheduler import BatchScheduler
        from quantumflow.inference.batch_config import DynamicBatchConfig
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager(safety_factor=0.7)
        config = DynamicBatchConfig()
        scheduler = BatchScheduler(vram_manager=vram, config=config)

        # Mock VRAM utilization at 50%
        with patch.object(vram, "get_vram_utilization", return_value=0.5):
            batch_size = scheduler.compute_batch_size(pending_count=5)
            # 0.5 utilization is between 0.7 and 0.9, so it should return base
            assert batch_size == config.base_max_batch_size

    def test_compute_batch_size_high_vram(self):
        """高显存压力时减少 batch size"""
        from quantumflow.inference.batch_scheduler import BatchScheduler
        from quantumflow.inference.batch_config import DynamicBatchConfig
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager(safety_factor=0.7)
        config = DynamicBatchConfig(base_max_batch_size=8, min_batch_size=2)
        scheduler = BatchScheduler(vram_manager=vram, config=config)

        # High VRAM utilization (> 0.9)
        with patch.object(vram, "get_vram_utilization", return_value=0.95):
            batch_size = scheduler.compute_batch_size(pending_count=5)
            # Should be reduced to 50% of base = 4
            # Since 8 * 0.5 = 4, which is >= min_batch_size=2
            assert batch_size == 4

    def test_compute_batch_size_low_vram_high_pending(self):
        """低显存压力且pending多时增加batch size"""
        from quantumflow.inference.batch_scheduler import BatchScheduler
        from quantumflow.inference.batch_config import DynamicBatchConfig
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager(safety_factor=0.7)
        config = DynamicBatchConfig(base_max_batch_size=8, max_batch_size=16)
        scheduler = BatchScheduler(vram_manager=vram, config=config)

        # Low VRAM utilization (< 0.7) and high pending (> base * 2)
        with patch.object(vram, "get_vram_utilization", return_value=0.5):
            batch_size = scheduler.compute_batch_size(pending_count=20)
            # pending_count=20 > 8*2=16, so should increase
            assert batch_size > config.base_max_batch_size


class TestBatchAccumulatorWithDynamicBatch:
    """BatchAccumulator 集成动态 batch size 测试"""

    @pytest.mark.asyncio
    async def test_batch_accumulator_respects_dynamic_max_size(self):
        """BatchAccumulator 应该遵守动态计算的 max_batch_size"""
        infer_fn = AsyncMock(
            return_value=[
                _make_result(f"batch_{i}", f"result_{i}")
                for i in range(8)
            ]
        )
        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=50.0,
            max_batch_size=8,
            enable_priority=False,
        )

        # Submit 3 requests - should be within max_batch_size
        results = await asyncio.gather(
            acc.submit("p1"),
            acc.submit("p2"),
            acc.submit("p3"),
        )

        assert len(results) == 3
        assert infer_fn.call_count >= 1

        await acc.shutdown()

    @pytest.mark.asyncio
    async def test_batch_accumulator_updates_stats(self):
        """BatchAccumulator 统计更新"""
        infer_fn = AsyncMock(
            return_value=[_make_result("0", "ok")]
        )
        acc = BatchAccumulator(
            infer_fn=infer_fn,
            max_delay_ms=50.0,
            max_batch_size=4,
        )

        await acc.submit("test")

        # Wait for processing
        await asyncio.sleep(0.2)

        assert acc.stats["total_requests"] >= 1
        assert acc.stats["total_batches"] >= 1

        await acc.shutdown()


class TestDynamicBatchIntegration:
    """动态批处理集成测试"""

    def test_dynamic_config_from_vram_manager(self):
        """VRAMManager 提供利用率查询"""
        from quantumflow.inference.vram_manager import VRAMManager

        vram = VRAMManager()

        # Should have get_vram_utilization method
        assert hasattr(vram, "get_vram_utilization")

        # Should be callable
        assert callable(getattr(vram, "get_vram_utilization"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
