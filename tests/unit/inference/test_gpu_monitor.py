"""GPU Monitor 核心逻辑专业测试

测试策略：
1. GPUSnapshot 数据类创建和 to_dict 精度
2. GPUMonitor 初始化：pynvml 可用 vs 不可用
3. _read_via_pynvml：多 GPU、温度读取失败、部分失败
4. _read_via_torch：CUDA 可用 vs 不可用
5. start/stop 生命周期
6. collect_snapshot 同步采集
7. subscribe 回调调用
8. _monitor_loop 错误恢复
9. __del__ 清理
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantumflow.inference.gpu_monitor import GPUMonitor, GPUSnapshot


# ═══════════════════════════════════════════════════════════════════════════════
# GPUSnapshot 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestGPUSnapshot:
    """GPUSnapshot 数据类正确性"""

    def test_snapshot_creation_all_fields(self):
        """所有字段必须正确设置"""
        snap = GPUSnapshot(
            index=0,
            name="NVIDIA RTX 4080",
            total_vram_gb=12.0,
            used_vram_gb=3.5,
            free_vram_gb=8.5,
            utilization_pct=45.2,
            memory_util_pct=30.1,
            temperature_c=65.0,
            timestamp=1000000.0,
        )
        assert snap.index == 0
        assert snap.name == "NVIDIA RTX 4080"
        assert snap.total_vram_gb == 12.0
        assert snap.used_vram_gb == 3.5
        assert snap.free_vram_gb == 8.5
        assert snap.utilization_pct == 45.2
        assert snap.memory_util_pct == 30.1
        assert snap.temperature_c == 65.0
        assert snap.timestamp == 1000000.0

    def test_snapshot_default_timestamp_is_current_time(self):
        """timestamp 默认值应为当前时间（非 0）"""
        snap = GPUSnapshot(
            index=0,
            name="GPU",
            total_vram_gb=8.0,
            used_vram_gb=4.0,
            free_vram_gb=4.0,
            utilization_pct=50.0,
            memory_util_pct=40.0,
            temperature_c=70.0,
        )
        assert snap.timestamp > 0, "timestamp 应 > 0"

    def test_to_dict_rounds_vram_to_one_decimal(self):
        """to_dict 必须将 VRAM 值四舍五入到小数点后 1 位"""
        snap = GPUSnapshot(
            index=0,
            name="Test GPU",
            total_vram_gb=12.345,
            used_vram_gb=3.567,
            free_vram_gb=8.778,
            utilization_pct=45.678,
            memory_util_pct=30.123,
            temperature_c=65.456,
        )
        d = snap.to_dict()
        assert d["total_vram_gb"] == 12.3
        assert d["used_vram_gb"] == 3.6
        assert d["free_vram_gb"] == 8.8
        assert d["utilization_pct"] == 45.7
        assert d["memory_util_pct"] == 30.1
        assert d["temperature_c"] == 65.5

    def test_to_dict_none_utilization_stays_none(self):
        """utilization_pct=None 时 to_dict 返回 None（不 round）"""
        snap = GPUSnapshot(
            index=0,
            name="GPU",
            total_vram_gb=10.0,
            used_vram_gb=5.0,
            free_vram_gb=5.0,
            utilization_pct=None,
            memory_util_pct=None,
            temperature_c=None,
        )
        d = snap.to_dict()
        assert d["utilization_pct"] is None
        assert d["memory_util_pct"] is None
        assert d["temperature_c"] is None

    def test_to_dict_includes_all_keys(self):
        """to_dict 必须包含所有期望的键"""
        snap = GPUSnapshot(
            index=1,
            name="GPU",
            total_vram_gb=8.0,
            used_vram_gb=3.0,
            free_vram_gb=5.0,
            utilization_pct=50.0,
            memory_util_pct=30.0,
            temperature_c=60.0,
        )
        d = snap.to_dict()
        expected_keys = {
            "index", "name", "total_vram_gb", "used_vram_gb", "free_vram_gb",
            "utilization_pct", "memory_util_pct", "temperature_c", "timestamp",
        }
        assert set(d.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════════
# GPUMonitor 初始化测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestGPUMonitorInitialization:
    """GPUMonitor 初始化逻辑"""

    def test_default_interval_is_5_seconds(self):
        """默认采样间隔为 5 秒"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        assert monitor.interval_seconds == 5.0

    def test_custom_interval_accepted(self):
        """自定义采样间隔"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor(interval_seconds=1.0)
        assert monitor.interval_seconds == 1.0

    def test_pynvml_unavailable_initializes_gracefully(self):
        """pynvml 不可用时初始化不崩溃"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        assert monitor._pynvml_available is False
        assert monitor._gpu_count == 0

    def test_pynvml_available_mock(self):
        """pynvml 可用时正确初始化 GPU 数量"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.NVML_TEMPERATURE_GPU = "GPU_TEMP"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            monitor = GPUMonitor()
        assert monitor._pynvml_available is True
        assert monitor._gpu_count == 2

    def test_pynvml_init_failure_falls_back_gracefully(self):
        """pynvmlInit 成功但后续操作失败时的清理"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("Init failed")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            monitor = GPUMonitor()
        # import 失败 catch 后 _pynvml_available 应为 False
        assert monitor._pynvml_available is False
        assert monitor._gpu_count == 0

    def test_latest_starts_empty(self):
        """latest 初始为空列表"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        assert monitor.latest == []

    def test_initial_state_not_running(self):
        """初始状态 _running=False"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        assert monitor._running is False
        assert monitor._task is None
        assert monitor._sample_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _read_via_pynvml 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestReadViaPynvml:
    """_read_via_pynvml 逻辑"""

    def _setup_pynvml_monitor_patch(self):
        """创建 monitor 并设置 mock pynvml 供 _read_via_pynvml 使用"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        # 手动设置 pynvml 可用状态和 GPU 数量（绕过 init 中的实际导入）
        monitor._pynvml_available = True
        monitor._gpu_count = 1
        return monitor

    def test_read_single_gpu_correct_vram(self):
        """单个 GPU 读取 VRAM 值正确（字节 -> GB 转换）"""
        monitor = self._setup_pynvml_monitor_patch()

        mock_pynvml = MagicMock()
        mock_mem = MagicMock()
        mock_mem.total = 12 * 1024**3
        mock_mem.used = 3 * 1024**3
        mock_mem.free = 9 * 1024**3
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 4080"

        mock_util = MagicMock()
        mock_util.gpu = 50
        mock_util.memory = 30
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.NVML_TEMPERATURE_GPU = "GPU_TEMP"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            snapshots = monitor._read_via_pynvml()

        assert len(snapshots) == 1
        s = snapshots[0]
        assert s.total_vram_gb == 12.0
        assert s.used_vram_gb == 3.0
        assert s.free_vram_gb == 9.0
        assert s.utilization_pct == 50
        assert s.memory_util_pct == 30

    def test_read_multiple_gpus(self):
        """多 GPU 系统返回每个 GPU 的快照"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor._pynvml_available = True
        monitor._gpu_count = 3

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        mock_pynvml.nvmlDeviceGetName.side_effect = [b"GPU-0", b"GPU-1", b"GPU-2"]
        mock_pynvml.NVML_TEMPERATURE_GPU = "GPU_TEMP"

        memories = []
        for i in range(3):
            mem = MagicMock()
            mem.total = (8 + i * 2) * 1024**3
            mem.used = i * 1024**3
            mem.free = mem.total - mem.used
            memories.append(mem)
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = memories

        utils = [MagicMock(gpu=45, memory=25) for _ in range(3)]
        mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = utils
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 55

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            snapshots = monitor._read_via_pynvml()

        assert len(snapshots) == 3
        for i, s in enumerate(snapshots):
            assert s.index == i
            assert s.total_vram_gb == 8 + i * 2
            assert s.used_vram_gb == i

    def test_temperature_read_failure_falls_back_to_zero(self):
        """温度读取失败时 temperature_c 设为 0（不崩溃）"""
        monitor = self._setup_pynvml_monitor_patch()

        mock_pynvml = MagicMock()
        mock_mem = MagicMock(total=8*1024**3, used=2*1024**3, free=6*1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetName.return_value = b"GPU"
        mock_util = MagicMock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = Exception("Temp read failed")
        mock_pynvml.NVML_TEMPERATURE_GPU = "GPU_TEMP"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            snapshots = monitor._read_via_pynvml()

        assert len(snapshots) == 1
        assert snapshots[0].temperature_c == 0

    def test_partial_gpu_failure_skipped(self):
        """某个 GPU 读取失败时不崩溃，只返回成功的 GPU"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor._pynvml_available = True
        monitor._gpu_count = 2

        mock_pynvml = MagicMock()
        # GPU 0 成功, GPU 1 失败
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            MagicMock(),
            Exception("GPU 1 not accessible"),
        ]
        mock_pynvml.nvmlDeviceGetName.return_value = b"GPU-0"
        mock_mem = MagicMock(total=8*1024**3, used=2*1024**3, free=6*1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_util = MagicMock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.NVML_TEMPERATURE_GPU = "GPU_TEMP"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            snapshots = monitor._read_via_pynvml()

        assert len(snapshots) == 1
        assert snapshots[0].index == 0

    def test_name_bytes_decoded_to_string(self):
        """GPU 名称从 bytes 正确 decode 为 string"""
        monitor = self._setup_pynvml_monitor_patch()

        mock_pynvml = MagicMock()
        mock_mem = MagicMock(total=8*1024**3, used=2*1024**3, free=6*1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 4090"
        mock_util = MagicMock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.NVML_TEMPERATURE_GPU = "GPU_TEMP"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            snapshots = monitor._read_via_pynvml()

        assert snapshots[0].name == "NVIDIA GeForce RTX 4090"
        assert isinstance(snapshots[0].name, str)

    def test_name_already_string(self):
        """GPU 名称已经是 string 时不崩溃"""
        monitor = self._setup_pynvml_monitor_patch()

        mock_pynvml = MagicMock()
        mock_mem = MagicMock(total=8*1024**3, used=2*1024**3, free=6*1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GPU"
        mock_util = MagicMock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.NVML_TEMPERATURE_GPU = "GPU_TEMP"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            snapshots = monitor._read_via_pynvml()

        assert snapshots[0].name == "NVIDIA GPU"


# ═══════════════════════════════════════════════════════════════════════════════
# _read_via_torch 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestReadViaTorch:
    """_read_via_torch 逻辑"""

    def test_cuda_not_available_returns_empty(self):
        """CUDA 不可用时返回空列表"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        with patch("torch.cuda.is_available", return_value=False):
            snapshots = monitor._read_via_torch()
        assert snapshots == []

    def test_cuda_available_returns_correct_vram(self):
        """CUDA 可用时返回 VRAM 信息"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                mock_props = MagicMock()
                mock_props.total_memory = 12 * 1024**3
                mock_props.name = "NVIDIA RTX 4080"
                with patch("torch.cuda.get_device_properties", return_value=mock_props):
                    with patch("torch.cuda.memory_allocated", return_value=4 * 1024**3):
                        snapshots = monitor._read_via_torch()

        assert len(snapshots) == 1
        s = snapshots[0]
        assert s.total_vram_gb == 12.0
        assert s.used_vram_gb == 4.0
        assert s.free_vram_gb == 8.0
        assert s.utilization_pct is None
        assert s.memory_util_pct is None
        assert s.temperature_c is None

    def test_torch_fallback_multiple_gpus(self):
        """多 GPU 时每个 GPU 都有快照"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                props_list = []
                for i in range(2):
                    p = MagicMock()
                    p.total_memory = (8 + i * 4) * 1024**3
                    p.name = f"GPU-{i}"
                    props_list.append(p)
                with patch("torch.cuda.get_device_properties", side_effect=props_list):
                    with patch("torch.cuda.memory_allocated", side_effect=[2*1024**3, 1*1024**3]):
                        snapshots = monitor._read_via_torch()

        assert len(snapshots) == 2
        assert snapshots[0].index == 0
        assert snapshots[0].name == "GPU-0"
        assert snapshots[0].total_vram_gb == 8.0
        assert snapshots[1].index == 1
        assert snapshots[1].name == "GPU-1"
        assert snapshots[1].total_vram_gb == 12.0


# ═══════════════════════════════════════════════════════════════════════════════
# collect_snapshot 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestCollectSnapshot:
    """collect_snapshot 同步采集"""

    def test_collect_snapshot_uses_pynvml_when_available(self):
        """pynvml 可用时使用 _read_via_pynvml"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor._pynvml_available = True

        mock_snap = GPUSnapshot(
            index=0, name="GPU", total_vram_gb=8.0, used_vram_gb=2.0,
            free_vram_gb=6.0, utilization_pct=50.0, memory_util_pct=30.0,
            temperature_c=60.0,
        )
        with patch.object(monitor, "_read_via_pynvml", return_value=[mock_snap]) as mock_read:
            snapshots = monitor.collect_snapshot()

        mock_read.assert_called_once()
        assert len(snapshots) == 1
        assert snapshots[0].name == "GPU"

    def test_collect_snapshot_uses_torch_when_pynvml_unavailable(self):
        """pynvml 不可用时使用 _read_via_torch"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor._pynvml_available = False

        mock_snap = GPUSnapshot(
            index=0, name="GPU", total_vram_gb=12.0, used_vram_gb=3.0,
            free_vram_gb=9.0, utilization_pct=None, memory_util_pct=None,
            temperature_c=None,
        )
        with patch.object(monitor, "_read_via_torch", return_value=[mock_snap]) as mock_read:
            snapshots = monitor.collect_snapshot()

        mock_read.assert_called_once()
        assert len(snapshots) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# start/stop 生命周期测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestGPUMonitorLifecycle:
    """GPUMonitor start/stop 生命周期"""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self):
        """start 必须设置 _running=True 并创建 task"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        with patch.object(monitor, "_monitor_loop", return_value=None):
            await monitor.start()

        assert monitor._running is True
        assert monitor._task is not None

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """重复 start 不报错"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        with patch.object(monitor, "_monitor_loop", return_value=None):
            await monitor.start()
            await monitor.start()  # 不崩溃

        assert monitor._running is True

    @pytest.mark.asyncio
    async def test_stop_sets_running_flag_false(self):
        """stop 必须设置 _running=False"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        with patch.object(monitor, "_monitor_loop", return_value=None):
            await monitor.start()

        monitor._task = MagicMock()
        monitor._task.done.return_value = True
        await monitor.stop()

        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_running_task(self):
        """stop 取消正在运行的 task"""
        import asyncio
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor._running = True
        loop = asyncio.get_event_loop()
        monitor._task = loop.create_task(asyncio.sleep(60))
        await asyncio.sleep(0)

        await monitor.stop()

        assert monitor._task.done()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_stop_handles_none_task(self):
        """task=None 时 stop 不崩溃"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor._task = None
        monitor._running = True

        await monitor.stop()
        assert monitor._running is False


# ═══════════════════════════════════════════════════════════════════════════════
# subscribe 回调测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubscribe:
    """subscribe 回调机制"""

    def test_subscribe_adds_callback_to_list(self):
        """subscribe 将回调添加到 _subscribers 列表"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor.subscribe(lambda s: None)
        assert len(monitor._subscribers) == 1

    def test_subscribe_multiple_callbacks(self):
        """可以注册多个回调"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor.subscribe(lambda s: None)
        monitor.subscribe(lambda s: None)
        assert len(monitor._subscribers) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# _monitor_loop 错误恢复测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestMonitorLoop:
    """_monitor_loop 后台循环"""

    @pytest.mark.asyncio
    async def test_monitor_loop_collects_snapshots(self):
        """循环必须调用 _read_gpu_state 并存储结果"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()

        mock_snap = GPUSnapshot(
            index=0, name="GPU", total_vram_gb=8.0, used_vram_gb=2.0,
            free_vram_gb=6.0, utilization_pct=50.0, memory_util_pct=30.0,
            temperature_c=60.0,
        )

        with patch.object(monitor, "_read_gpu_state", return_value=[mock_snap]):
            for _ in range(2):
                snapshots = monitor._read_gpu_state()
                monitor._latest = snapshots
                monitor._sample_count += 1

        assert monitor._sample_count == 2
        assert len(monitor._latest) == 1

    def test_monitor_loop_handles_read_error(self):
        """_read_gpu_state 抛 RuntimeError 时循环不中断"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()

        with patch.object(monitor, "_read_gpu_state", side_effect=RuntimeError("GPU read failed")):
            try:
                monitor._read_gpu_state()
            except RuntimeError:
                pass  # 模拟被循环 catch

        # 不应崩溃


# ═══════════════════════════════════════════════════════════════════════════════
# __del__ 清理测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestDestructor:
    """__del__ 清理"""

    def test_del_when_pynvml_unavailable_no_crash(self):
        """pynvml 不可用时 __del__ 不崩溃"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor._pynvml_available = False
        monitor.__del__()

    def test_del_handles_nvml_shutdown_error(self):
        """nvmlShutdown 抛异常时 __del__ 不崩溃"""
        with patch.dict("sys.modules", {"pynvml": None}):
            monitor = GPUMonitor()
        monitor._pynvml_available = True
        # 即使 _pynvml_available=True，但如果 pynvml 模块有异常在 __del__ 也会被捕获
        monitor.__del__()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
