"""GPU Monitor 覆盖率缺口补充测试

精确覆盖 gpu_monitor.py 缺失行:
- nvmlInit succeeds but subsequent fails -> cleanup (70-76)
- _monitor_loop: CancelledError handler (144)
- _monitor_loop: full loop execution (116-148)
- _monitor_loop: await coroutine subscriber (127)
- _monitor_loop: every-10 logging (133-134)
- _read_via_torch: exception returns empty list (217-218)
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.inference.gpu_monitor import GPUMonitor, GPUSnapshot


class TestGPUMonitorInitEdgeCases:
    def test_nvml_init_succeeds_but_get_count_fails(self):
        """When nvmlInit succeeds but nvmlDeviceGetCount raises exception,
        pynvml_available becomes True then gets cleaned up correctly."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.side_effect = RuntimeError("GetCount failed")
        mock_pynvml.nvmlShutdown.side_effect = RuntimeError("Shutdown failed")
        mock_pynvml.NVML_TEMPERATURE_GPU = "GPU_TEMP"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            monitor = GPUMonitor()

        # The except block should catch the GetCount error and try cleanup
        # nvmlShutdown failure is caught by inner except (lines 74-75)
        assert monitor._pynvml_available is False
        assert monitor._gpu_count == 0


class TestMonitorLoopCancelledError:
    @pytest.mark.asyncio
    async def test_monitor_loop_exits_on_cancelled_error(self):
        """The _monitor_loop must break on asyncio.CancelledError"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor(interval_seconds=0.1)

        mock_snap = GPUSnapshot(
            index=0, name="GPU", total_vram_gb=8.0, used_vram_gb=2.0,
            free_vram_gb=6.0, utilization_pct=50.0, memory_util_pct=30.0,
            temperature_c=60.0,
        )

        with patch.object(monitor, "_read_gpu_state", return_value=[mock_snap]):
            monitor._running = True
            monitor._sample_count = 0

            # Start the loop and cancel after a short time
            task = asyncio.ensure_future(monitor._monitor_loop())
            await asyncio.sleep(0.2)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have collected at least 1 sample before cancellation
        assert monitor._sample_count >= 1

    @pytest.mark.asyncio
    async def test_monitor_loop_handles_error_gracefully(self):
        """When _read_gpu_state raises, the loop should continue"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor(interval_seconds=0.05)

        call_count = [0]
        def read_with_error():
            call_count[0] += 1
            # First call succeeds, subsequent calls fail
            if call_count[0] > 1:
                raise RuntimeError("GPU read error")
            return [GPUSnapshot(
                index=0, name="GPU", total_vram_gb=8.0, used_vram_gb=2.0,
                free_vram_gb=6.0, utilization_pct=50.0, memory_util_pct=30.0,
                temperature_c=60.0,
            )]

        with patch.object(monitor, "_read_gpu_state", side_effect=read_with_error):
            monitor._running = True
            monitor._sample_count = 0

            task = asyncio.ensure_future(monitor._monitor_loop())
            # Wait for the first successful iteration to complete
            await asyncio.sleep(0.08)
            monitor._running = False
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        # The first call should succeed, giving us at least 1 sample
        assert monitor._sample_count >= 1
        # Error was raised on subsequent calls but loop handled it
        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_monitor_loop_notifies_subscribers(self):
        """Subscriber callbacks should be called with snapshots"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor(interval_seconds=0.05)

        mock_snap = GPUSnapshot(
            index=0, name="GPU", total_vram_gb=8.0, used_vram_gb=2.0,
            free_vram_gb=6.0, utilization_pct=50.0, memory_util_pct=30.0,
            temperature_c=60.0,
        )

        callback_calls = []
        def callback(snapshots):
            callback_calls.append(snapshots)

        monitor.subscribe(callback)

        with patch.object(monitor, "_read_gpu_state", return_value=[mock_snap]):
            monitor._running = True
            monitor._sample_count = 0

            task = asyncio.ensure_future(monitor._monitor_loop())
            await asyncio.sleep(0.2)
            monitor._running = False
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(callback_calls) >= 1
        assert len(callback_calls[0]) == 1
        assert callback_calls[0][0].name == "GPU"

    @pytest.mark.asyncio
    async def test_monitor_loop_subscriber_error_handled(self):
        """Subscriber errors should not crash the loop"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor(interval_seconds=0.05)

        mock_snap = GPUSnapshot(
            index=0, name="GPU", total_vram_gb=8.0, used_vram_gb=2.0,
            free_vram_gb=6.0, utilization_pct=50.0, memory_util_pct=30.0,
            temperature_c=60.0,
        )

        def bad_callback(snapshots):
            raise RuntimeError("Callback error")

        monitor.subscribe(bad_callback)
        monitor.subscribe(lambda s: None)  # good callback

        with patch.object(monitor, "_read_gpu_state", return_value=[mock_snap]):
            monitor._running = True
            monitor._sample_count = 0

            task = asyncio.ensure_future(monitor._monitor_loop())
            await asyncio.sleep(0.15)
            monitor._running = False
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should not crash
        assert monitor._sample_count >= 1

    @pytest.mark.asyncio
    async def test_monitor_loop_awaits_async_subscriber_callback(self):
        """When subscriber returns a coroutine, it must be awaited (line 127)"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor(interval_seconds=0.01)

        mock_snap = GPUSnapshot(
            index=0, name="GPU", total_vram_gb=8.0, used_vram_gb=2.0,
            free_vram_gb=6.0, utilization_pct=50.0, memory_util_pct=30.0,
            temperature_c=60.0,
        )

        async def async_callback(snapshots):
            await asyncio.sleep(0)  # Simulate async work
            return "awaited"

        callback_result = []
        async def catching_callback(snapshots):
            try:
                result = async_callback(snapshots)
                if hasattr(result, "__await__"):
                    await result
                callback_result.append("success")
            except Exception:
                callback_result.append("error")

        monitor.subscribe(catching_callback)

        with patch.object(monitor, "_read_gpu_state", return_value=[mock_snap]):
            monitor._running = True
            monitor._sample_count = 0

            task = asyncio.ensure_future(monitor._monitor_loop())
            await asyncio.sleep(0.1)
            monitor._running = False
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(callback_result) >= 1
        assert callback_result[0] == "success"

    @pytest.mark.asyncio
    async def test_monitor_loop_logs_every_10_samples(self):
        """Every 10th sample should be logged (lines 133-134)"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor(interval_seconds=0.01)

        mock_snap = GPUSnapshot(
            index=0, name="GPU", total_vram_gb=8.0, used_vram_gb=2.0,
            free_vram_gb=6.0, utilization_pct=50.0, memory_util_pct=30.0,
            temperature_c=60.0,
        )

        with patch.object(monitor, "_read_gpu_state", return_value=[mock_snap]):
            with patch("quantumflow.inference.gpu_monitor.logger") as mock_logger:
                monitor._running = True
                monitor._sample_count = 0

                task = asyncio.ensure_future(monitor._monitor_loop())
                # Run for enough iterations to hit sample 10
                await asyncio.sleep(0.15)
                monitor._running = False
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Should have logged at least once (every 10 samples)
                assert mock_logger.info.called

    @pytest.mark.asyncio
    async def test_read_via_torch_exception_returns_empty_list(self):
        """When torch calls fail, _read_via_torch returns [] (lines 217-218)"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor(interval_seconds=0.1)

        with patch("torch.cuda.memory_allocated", side_effect=RuntimeError("CUDA error")):
            result = monitor._read_via_torch()
            assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
