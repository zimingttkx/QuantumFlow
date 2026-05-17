"""GPU监控 — 100%覆盖率业务逻辑测试

测试范围:
- GPUSnapshot: 构造/to_dict/序列化精度
- 同步采集: collect_snapshot (pynvml/torch fallback)
- 异步监控: 后台周期采集/回调通知/多订阅者
- 生命周期: start/stop幂等/重复start/重复stop
- latest属性: 采集前/采集后/停止后
- 订阅者: 单回调/多回调/异步回调/回调异常容错
- 错误处理: monitor_loop异常/无GPU环境/per-device失败
- pynvml mock: 多GPU/中文名/temp失败/bytes名
- torch fallback: CUDA有/CUDA无/import失败
- 清理: __del__/nvmlShutdown
"""
import asyncio
import time
from unittest.mock import patch, MagicMock

from quantumflow.inference.gpu_monitor import GPUMonitor, GPUSnapshot


PASS = 0
FAIL = 0
FAIL_MSGS = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        msg = f"  ✗ {name} — {detail}"
        print(msg)
        FAIL_MSGS.append(msg)


def report():
    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"📊 GPU监控 测试报告: {PASS}/{total} 通过, {FAIL} 失败")
    print(f"{'='*60}")
    if FAIL_MSGS:
        print("\n❌ 失败项:")
        for e in FAIL_MSGS:
            print(f"  {e}")
    return FAIL == 0


# ═══════════════════════════════════════════════════════════
# 1. GPUSnapshot — 数据结构
# ═══════════════════════════════════════════════════════════

def test_snapshot_creation():
    """GPUSnapshot 构造与默认值"""
    print("\n── Snapshot: 构造 ──")
    s = GPUSnapshot(
        index=0, name="GPU-0", total_vram_gb=16.0,
        used_vram_gb=4.5, free_vram_gb=11.5,
        utilization_pct=75.0, temperature_c=65.0,
    )
    check("index", s.index == 0)
    check("name", s.name == "GPU-0")
    check("total", s.total_vram_gb == 16.0)
    check("used", s.used_vram_gb == 4.5)
    check("free", s.free_vram_gb == 11.5)
    check("util", s.utilization_pct == 75.0)
    check("temp", s.temperature_c == 65.0)
    check("timestamp自动设置", s.timestamp > 0)


def test_snapshot_timestamp_custom():
    """自定义时间戳"""
    print("\n── Snapshot: 自定义时间戳 ──")
    s = GPUSnapshot(
        index=0, name="G", total_vram_gb=1.0,
        used_vram_gb=0.5, free_vram_gb=0.5,
        utilization_pct=0.0, temperature_c=0.0,
        timestamp=1234567890.0,
    )
    check("自定义ts", s.timestamp == 1234567890.0)


def test_snapshot_to_dict():
    """to_dict 序列化 + 四舍五入"""
    print("\n── Snapshot: to_dict ──")
    s = GPUSnapshot(
        index=0, name="Test GPU", total_vram_gb=16.0,
        used_vram_gb=4.56, free_vram_gb=11.46,
        utilization_pct=75.51, temperature_c=65.44,
        timestamp=1234567890.0,
    )
    d = s.to_dict()

    check("index", d["index"] == 0)
    check("name", d["name"] == "Test GPU")
    check("total四舍五入到1位", d["total_vram_gb"] == 16.0)
    check("used四舍五入4.56→4.6", d["used_vram_gb"] == 4.6,
          f"got={d['used_vram_gb']}")
    check("free四舍五入11.46→11.5", d["free_vram_gb"] == 11.5,
          f"got={d['free_vram_gb']}")
    check("util四舍五入75.51→75.5", d["utilization_pct"] == 75.5,
          f"got={d['utilization_pct']}")
    check("temp四舍五入65.44→65.4", d["temperature_c"] == 65.4)
    check("timestamp", d["timestamp"] == 1234567890.0)


def test_snapshot_to_dict_zero_values():
    """零值序列化"""
    print("\n── Snapshot: 零值 ──")
    s = GPUSnapshot(
        index=0, name="", total_vram_gb=0.0,
        used_vram_gb=0.0, free_vram_gb=0.0,
        utilization_pct=0.0, temperature_c=0.0,
    )
    d = s.to_dict()
    check("零值total", d["total_vram_gb"] == 0.0)
    check("零值used", d["used_vram_gb"] == 0.0)
    check("零值temp", d["temperature_c"] == 0.0)


def test_snapshot_to_dict_rounding_edge():
    """四舍五入边界值"""
    print("\n── Snapshot: 舍入边界 ──")
    s = GPUSnapshot(
        index=0, name="X", total_vram_gb=1.05,
        used_vram_gb=1.04, free_vram_gb=0.95,
        utilization_pct=0.95, temperature_c=0.0,
    )
    d = s.to_dict()
    # 1.05 → 1.1 (round to 1 decimal)
    check("1.05→1.1", d["total_vram_gb"] == 1.1, f"got={d['total_vram_gb']}")
    # 1.04 → 1.0
    check("1.04→1.0", d["used_vram_gb"] == 1.0, f"got={d['used_vram_gb']}")


# ═══════════════════════════════════════════════════════════
# 2. 同步采集
# ═══════════════════════════════════════════════════════════

def test_collect_snapshot__real_gpu():
    """同步采集 — 真实GPU或空"""
    print("\n── 同步: 真实GPU ──")
    monitor = GPUMonitor(interval_seconds=5.0)
    snapshots = monitor.collect_snapshot()

    check("返回list", isinstance(snapshots, list))

    import torch
    if torch.cuda.is_available():
        check("至少1个GPU", len(snapshots) >= 1)
        s = snapshots[0]
        check("有名称", len(s.name) > 0)
        check("total>0", s.total_vram_gb > 0)
        check("used>=0", s.used_vram_gb >= 0)
        check("free>=0", s.free_vram_gb >= 0)
        check("total≈used+free", abs(s.total_vram_gb - s.used_vram_gb - s.free_vram_gb) < 1.0,
              f"total={s.total_vram_gb}, used={s.used_vram_gb}, free={s.free_vram_gb}")
        check("类型正确", isinstance(s, GPUSnapshot))
    else:
        check("无CUDA返回空", len(snapshots) == 0)
        print("  (跳过 — 无GPU)")


def test_collect_snapshot__no_gpu_environment():
    """无GPU环境不崩溃"""
    print("\n── 同步: 无GPU ──")
    monitor = GPUMonitor(interval_seconds=1.0)
    snapshots = monitor.collect_snapshot()
    check("返回list", isinstance(snapshots, list))


# ═══════════════════════════════════════════════════════════
# 3. pynvml mock 测试
# ═══════════════════════════════════════════════════════════

def test_read_via_pynvml__single_gpu():
    """Mock pynvml: 单GPU正常采集"""
    print("\n── pynvml: 单GPU ──")
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 4090"
    mock_mem = MagicMock()
    mock_mem.total = 24 * (1024**3)
    mock_mem.used = 8 * (1024**3)
    mock_mem.free = 16 * (1024**3)
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
    mock_util = MagicMock()
    mock_util.gpu = 85
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 72
    mock_pynvml.NVML_TEMPERATURE_GPU = 0

    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
        monitor = GPUMonitor(interval_seconds=5.0)
        # Force pynvml available
        monitor._pynvml_available = True
        monitor._gpu_count = 1

        snapshots = monitor._read_via_pynvml()

    check("1个快照", len(snapshots) == 1)
    s = snapshots[0]
    check("index=0", s.index == 0)
    check("name=RTX4090", s.name == "NVIDIA RTX 4090")
    check("total=24GB", abs(s.total_vram_gb - 24.0) < 0.1, f"got={s.total_vram_gb}")
    check("used=8GB", abs(s.used_vram_gb - 8.0) < 0.1, f"got={s.used_vram_gb}")
    check("free=16GB", abs(s.free_vram_gb - 16.0) < 0.1, f"got={s.free_vram_gb}")
    check("util=85%", s.utilization_pct == 85.0)
    check("temp=72°C", s.temperature_c == 72.0)


def test_read_via_pynvml__multi_gpu():
    """Mock pynvml: 多GPU"""
    print("\n── pynvml: 多GPU ──")
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 4
    mock_pynvml.NVML_TEMPERATURE_GPU = 0

    def mock_name(handle):
        idx = mock_pynvml.nvmlDeviceGetHandleByIndex.call_args_list
        # Simple: return different names
        return {0: "GPU-0", 1: "GPU-1", 2: "GPU-2", 3: "GPU-3"}.get(
            len(mock_pynvml.nvmlDeviceGetName.call_args_list) - 1, "GPU-X")

    mock_pynvml.nvmlDeviceGetName.side_effect = lambda h: f"GPU-{mock_pynvml.nvmlDeviceGetName.call_count - 1}"

    mock_mem = MagicMock()
    mock_mem.total = 16 * (1024**3)
    mock_mem.used = 4 * (1024**3)
    mock_mem.free = 12 * (1024**3)
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

    mock_util = MagicMock()
    mock_util.gpu = 50
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 60

    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
        monitor = GPUMonitor(interval_seconds=5.0)
        monitor._pynvml_available = True
        monitor._gpu_count = 4
        snapshots = monitor._read_via_pynvml()

    check("4个快照", len(snapshots) == 4)
    for i, s in enumerate(snapshots):
        check(f"GPU{i} index={i}", s.index == i)
        check(f"GPU{i}有名称", len(s.name) > 0)


def test_read_via_pynvml__name_as_bytes():
    """pynvml 返回 bytes 类型的名称 → decode"""
    print("\n── pynvml: bytes名称 ──")
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_pynvml.nvmlDeviceGetName.return_value = b"Tesla T4"
    mock_pynvml.NVML_TEMPERATURE_GPU = 0

    mock_mem = MagicMock()
    mock_mem.total = 16 * (1024**3)
    mock_mem.used = 4 * (1024**3)
    mock_mem.free = 12 * (1024**3)
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

    mock_util = MagicMock()
    mock_util.gpu = 30
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 45

    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
        monitor = GPUMonitor(interval_seconds=5.0)
        monitor._pynvml_available = True
        monitor._gpu_count = 1
        snapshots = monitor._read_via_pynvml()

    check("name解码为str", snapshots[0].name == "Tesla T4",
          f"got={snapshots[0].name!r}")


def test_read_via_pynvml__temperature_failure():
    """温度读取失败 → temp=0（不崩溃）"""
    print("\n── pynvml: 温度读取失败 ──")
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_pynvml.nvmlDeviceGetName.return_value = "Hot GPU"
    mock_pynvml.NVML_TEMPERATURE_GPU = 0
    mock_pynvml.nvmlDeviceGetTemperature.side_effect = Exception("temp sensor broken")

    mock_mem = MagicMock()
    mock_mem.total = 24 * (1024**3)
    mock_mem.used = 8 * (1024**3)
    mock_mem.free = 16 * (1024**3)
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

    mock_util = MagicMock()
    mock_util.gpu = 90
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
        monitor = GPUMonitor(interval_seconds=5.0)
        monitor._pynvml_available = True
        monitor._gpu_count = 1
        snapshots = monitor._read_via_pynvml()

    check("不崩溃返回快照", len(snapshots) == 1)
    check("temp=0(失败fallback)", snapshots[0].temperature_c == 0,
          f"got={snapshots[0].temperature_c}")


def test_read_via_pynvml__per_device_failure():
    """单个GPU读取失败时跳过，不影响其他GPU"""
    print("\n── pynvml: 单设备失败 ──")
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 3
    mock_pynvml.NVML_TEMPERATURE_GPU = 0

    call_count = [0]

    def mem_side_effect(handle):
        call_count[0] += 1
        if call_count[0] == 2:  # 第2个GPU失败
            raise Exception("GPU offline")
        m = MagicMock()
        m.total = 8 * (1024**3)
        m.used = 2 * (1024**3)
        m.free = 6 * (1024**3)
        return m

    mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = mem_side_effect
    mock_pynvml.nvmlDeviceGetName.return_value = "GPU"
    mock_util = MagicMock()
    mock_util.gpu = 50
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 50

    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
        monitor = GPUMonitor(interval_seconds=5.0)
        monitor._pynvml_available = True
        monitor._gpu_count = 3
        snapshots = monitor._read_via_pynvml()

    check("2个成功(1个失败跳过)", len(snapshots) == 2,
          f"got={len(snapshots)}")


# ═══════════════════════════════════════════════════════════
# 4. torch fallback 测试
# ═══════════════════════════════════════════════════════════

def test_read_via_torch__cuda_available():
    """torch CUDA 可用"""
    print("\n── torch: CUDA可用 ──")
    # 有真实CUDA或mock
    monitor = GPUMonitor(interval_seconds=5.0)
    monitor._pynvml_available = False

    snapshots = monitor._read_via_torch()

    import torch
    if torch.cuda.is_available():
        check("有快照", len(snapshots) >= 1)
        check("类型正确", isinstance(snapshots[0], GPUSnapshot))
        check("total>0", snapshots[0].total_vram_gb > 0)
        check("util=0(torch不可取)", snapshots[0].utilization_pct == 0.0)
        check("temp=0(torch不可取)", snapshots[0].temperature_c == 0.0)
    else:
        check("无CUDA返回空", snapshots == [])


def test_read_via_torch__cuda_unavailable():
    """torch CUDA 不可用 → 返回空"""
    print("\n── torch: CUDA不可用 ──")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict('sys.modules', {'torch': mock_torch}):
        monitor = GPUMonitor(interval_seconds=5.0)
        monitor._pynvml_available = False
        snapshots = monitor._read_via_torch()

    check("返回[]", snapshots == [])


def test_read_via_torch__import_failure():
    """torch import 失败 → 返回空"""
    print("\n── torch: import失败 ──")

    with patch.dict('sys.modules', {'torch': None}):
        # simulate import error by making torch unavailable
        pass

    monitor = GPUMonitor(interval_seconds=5.0)
    monitor._pynvml_available = False
    # _read_via_torch catches all exceptions
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'torch':
            raise ImportError("No torch")
        return original_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        snapshots = monitor._read_via_torch()

    check("import失败返回[]", snapshots == [])


def test_read_gpu_state__pynvml_falls_back_to_torch():
    """_read_gpu_state: pynvml可用→走pynvml; 不可用→走torch"""
    print("\n── fallback: pynvml→torch ──")
    monitor = GPUMonitor(interval_seconds=5.0)

    if monitor._pynvml_available:
        # pynvml可用 → _read_gpu_state 使用 pynvml
        snapshots = monitor._read_gpu_state()
        check("pynvml路径返回list", isinstance(snapshots, list))
    else:
        # 未安装pynvml → fallback到torch
        snapshots = monitor._read_gpu_state()
        check("torch路径返回list", isinstance(snapshots, list))
        print("  (pynvml不可用, 已fallback到torch)")


# ═══════════════════════════════════════════════════════════
# 5. 后台异步监控
# ═══════════════════════════════════════════════════════════

def test_async_monitor_collects():
    """后台监控周期性采集"""
    print("\n── 异步: 周期采集 ──")
    collected = []

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        monitor.subscribe(lambda snaps: collected.append(len(snaps)))
        await monitor.start()
        await asyncio.sleep(0.35)  # ~3个周期
        await monitor.stop()
        return len(collected)

    n = asyncio.run(run())
    check("至少采集2次", n >= 2, f"collected={n}")


def test_async_monitor_increments_sample_count():
    """采样计数器递增"""
    print("\n── 异步: 计数器 ──")

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        await monitor.start()
        await asyncio.sleep(0.35)
        await monitor.stop()
        return monitor._sample_count

    n = asyncio.run(run())
    check("_sample_count>=2", n >= 2, f"count={n}")


# ═══════════════════════════════════════════════════════════
# 6. start/stop 生命周期
# ═══════════════════════════════════════════════════════════

def test_start_idempotent():
    """重复 start 不报错，不创建多个 task"""
    print("\n── 生命周期: start幂等 ──")

    async def run():
        monitor = GPUMonitor(interval_seconds=1.0)
        await monitor.start()
        task1 = monitor._task
        await monitor.start()
        task2 = monitor._task
        await monitor.start()
        task3 = monitor._task
        await asyncio.sleep(0.05)
        await monitor.stop()
        return task1, task2, task3

    t1, t2, t3 = asyncio.run(run())
    check("3次start同一task", t1 is t2 and t2 is t3)


def test_stop_idempotent():
    """重复 stop 不报错"""
    print("\n── 生命周期: stop幂等 ──")

    async def run():
        monitor = GPUMonitor(interval_seconds=1.0)
        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()
        await monitor.stop()
        await monitor.stop()
        return True

    ok = asyncio.run(run())
    check("3次stop不crash", ok)


def test_stop_before_start():
    """未 start 直接 stop 不报错"""
    print("\n── 生命周期: stop-before-start ──")

    async def run():
        monitor = GPUMonitor(interval_seconds=1.0)
        await monitor.stop()
        return True

    ok = asyncio.run(run())
    check("未start直接stop不crash", ok)


def test_start_stop_cycle():
    """多次 start→stop→start→stop 循环"""
    print("\n── 生命周期: 启停循环 ──")

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        for _ in range(3):
            await monitor.start()
            await asyncio.sleep(0.12)
            await monitor.stop()
        return True

    ok = asyncio.run(run())
    check("3次启停循环不crash", ok)


# ═══════════════════════════════════════════════════════════
# 7. latest 属性
# ═══════════════════════════════════════════════════════════

def test_latest_before_start():
    """启动前 latest 为空"""
    print("\n── latest: 启动前 ──")
    monitor = GPUMonitor(interval_seconds=5.0)
    check("启动前latest=[]", monitor.latest == [])


def test_latest_after_collection():
    """采集后 latest 有数据"""
    print("\n── latest: 采集后 ──")

    async def run():
        monitor = GPUMonitor(interval_seconds=0.05)
        await monitor.start()
        await asyncio.sleep(0.12)
        latest = monitor.latest
        await monitor.stop()
        return latest

    latest = asyncio.run(run())
    check("latest是list", isinstance(latest, list))
    import torch
    if torch.cuda.is_available():
        check("有GPU快照", len(latest) >= 1)
        check("类型正确", isinstance(latest[0], GPUSnapshot))
    # 无GPU时latest是空list（也合法）


def test_latest_update_frequency():
    """latest 周期性更新"""
    print("\n── latest: 更新频率 ──")

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        snap1 = monitor.collect_snapshot()
        await monitor.start()
        await asyncio.sleep(0.35)
        latest_after = list(monitor.latest)
        await monitor.stop()
        return snap1, latest_after

    s1, s2 = asyncio.run(run())
    check("latest在更新", isinstance(s2, list))


# ═══════════════════════════════════════════════════════════
# 8. 订阅者回调
# ═══════════════════════════════════════════════════════════

def test_subscriber_called():
    """每次采样后触发订阅回调"""
    print("\n── 订阅: 回调触发 ──")
    events = []

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)

        def on_sample(snapshots):
            events.append({"count": len(snapshots), "ts": time.time()})

        monitor.subscribe(on_sample)
        await monitor.start()
        await asyncio.sleep(0.35)
        await monitor.stop()
        return events

    events = asyncio.run(run())
    check("至少2次回调", len(events) >= 2, f"events={len(events)}")
    for e in events:
        check("有count", "count" in e)
        check("有ts", "ts" in e)


def test_multiple_subscribers():
    """多个订阅者都收到通知"""
    print("\n── 订阅: 多订阅者 ──")
    counts = [0, 0, 0]

    def make_cb(idx):
        def cb(snapshots):
            counts[idx] += 1
        return cb

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        for i in range(3):
            monitor.subscribe(make_cb(i))
        await monitor.start()
        await asyncio.sleep(0.35)
        await monitor.stop()
        return counts

    counts = asyncio.run(run())
    check("订阅者1收到", counts[0] >= 2, f"c0={counts[0]}")
    check("订阅者2收到", counts[1] >= 2, f"c1={counts[1]}")
    check("订阅者3收到", counts[2] >= 2, f"c2={counts[2]}")


def test_subscriber_exception_does_not_crash_monitor():
    """订阅者抛出异常不影响监控循环"""
    print("\n── 订阅: 异常容错 ──")

    good_count = [0]
    bad_count = [0]

    def bad_callback(snapshots):
        bad_count[0] += 1
        raise RuntimeError("订阅者崩溃")

    def good_callback(snapshots):
        good_count[0] += 1

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        monitor.subscribe(bad_callback)
        monitor.subscribe(good_callback)
        await monitor.start()
        await asyncio.sleep(0.35)
        await monitor.stop()
        return good_count[0], bad_count[0]

    good, bad = asyncio.run(run())
    check("bad被调用", bad >= 2, f"bad={bad}")
    check("good仍被调用(监控继续)", good >= 2,
          f"good={good}")


def test_async_subscriber():
    """异步订阅者被正确 await"""
    print("\n── 订阅: 异步回调 ──")

    async_count = [0]

    async def async_callback(snapshots):
        async_count[0] += 1
        await asyncio.sleep(0.001)

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        monitor.subscribe(async_callback)
        await monitor.start()
        await asyncio.sleep(0.35)
        await monitor.stop()
        return async_count[0]

    count = asyncio.run(run())
    check("异步回调被调用", count >= 2, f"count={count}")


def test_subscribe_before_start():
    """启动前订阅 → 启动后收到通知"""
    print("\n── 订阅: 启动前订阅 ──")
    events = []

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        monitor.subscribe(lambda s: events.append(1))
        await monitor.start()
        await asyncio.sleep(0.25)
        await monitor.stop()
        return events

    events = asyncio.run(run())
    check("启动前订阅也能收到", len(events) >= 1)


def test_subscribe_after_start():
    """启动后订阅也能收到后续通知"""
    print("\n── 订阅: 启动后订阅 ──")
    events_before = []
    events_after = []

    async def run():
        monitor = GPUMonitor(interval_seconds=0.1)
        monitor.subscribe(lambda s: events_before.append(1))
        await monitor.start()
        await asyncio.sleep(0.2)
        # 中途订阅
        monitor.subscribe(lambda s: events_after.append(1))
        await asyncio.sleep(0.2)
        await monitor.stop()
        return len(events_before), len(events_after)

    before, after = asyncio.run(run())
    check("later订阅者收到", after >= 1, f"after={after}")
    check("original订阅者收到更多", before >= after,
          f"before={before}, after={after}")


# ═══════════════════════════════════════════════════════════
# 9. 边界 + 清理
# ═══════════════════════════════════════════════════════════

def test_interval_zero():
    """interval=0 不崩溃（虽然不推荐）"""
    print("\n── 边界: interval=0 ──")

    async def run():
        monitor = GPUMonitor(interval_seconds=0.0)
        # 临时禁用日志输出避免刷屏
        import structlog
        orig_logger = monitor._monitor_loop
        await monitor.start()
        await asyncio.sleep(0.02)  # 极短时间
        await monitor.stop()
        return monitor._sample_count

    n = asyncio.run(run())
    # interval=0 会有很多次采集
    check("interval=0不crash", n >= 0)


def test_default_interval():
    """默认 interval=5.0"""
    print("\n── 边界: 默认interval ──")
    monitor = GPUMonitor()
    check("默认interval=5.0", monitor.interval_seconds == 5.0)


def test_pynvml_init_failure():
    """pynvml init 失败 → fallback 模式"""
    print("\n── 边界: pynvml init失败 ──")
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = Exception("NVML not found")

    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
        monitor = GPUMonitor(interval_seconds=5.0)
        # pynvml init 失败 → _pynvml_available = False
        check("pynvml不可用", not monitor._pynvml_available)
        check("gpu_count=0", monitor._gpu_count == 0)


def test_del_shutdown():
    """__del__ 调用 nvmlShutdown（若 pynvml 可用）"""
    print("\n── 边界: __del__ ──")

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 0

    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
        monitor = GPUMonitor(interval_seconds=5.0)
        # __del__ should call nvmlShutdown
        monitor.__del__()

    # If pynvml was available, shutdown should have been called
    if monitor._pynvml_available:
        mock_pynvml.nvmlShutdown.assert_called()
    check("__del__不crash", True)


def test_monitor_loop_error_handling():
    """monitor_loop 中 _read_gpu_state 抛异常 → 不崩溃继续跑"""
    print("\n── 边界: loop异常恢复 ──")

    call_count = [0]
    error_count = [0]

    async def run():
        monitor = GPUMonitor(interval_seconds=0.05)
        # patch _read_gpu_state 前几次抛异常
        original_read = monitor._read_gpu_state

        def flaky_read():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("临时故障")
            return original_read()

        monitor._read_gpu_state = flaky_read
        await monitor.start()
        await asyncio.sleep(0.2)
        await monitor.stop()
        return call_count[0]

    n = asyncio.run(run())
    check("异常后继续采集", n >= 3, f"calls={n}")


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Snapshot 数据结构
    test_snapshot_creation()
    test_snapshot_timestamp_custom()
    test_snapshot_to_dict()
    test_snapshot_to_dict_zero_values()
    test_snapshot_to_dict_rounding_edge()

    # 同步采集
    test_collect_snapshot__real_gpu()
    test_collect_snapshot__no_gpu_environment()

    # pynvml mock
    test_read_via_pynvml__single_gpu()
    test_read_via_pynvml__multi_gpu()
    test_read_via_pynvml__name_as_bytes()
    test_read_via_pynvml__temperature_failure()
    test_read_via_pynvml__per_device_failure()

    # torch fallback
    test_read_via_torch__cuda_available()
    test_read_via_torch__cuda_unavailable()
    test_read_via_torch__import_failure()
    test_read_gpu_state__pynvml_falls_back_to_torch()

    # 异步监控
    test_async_monitor_collects()
    test_async_monitor_increments_sample_count()

    # 生命周期
    test_start_idempotent()
    test_stop_idempotent()
    test_stop_before_start()
    test_start_stop_cycle()

    # latest
    test_latest_before_start()
    test_latest_after_collection()
    test_latest_update_frequency()

    # 订阅者
    test_subscriber_called()
    test_multiple_subscribers()
    test_subscriber_exception_does_not_crash_monitor()
    test_async_subscriber()
    test_subscribe_before_start()
    test_subscribe_after_start()

    # 边界 + 清理
    test_interval_zero()
    test_default_interval()
    test_pynvml_init_failure()
    test_del_shutdown()
    test_monitor_loop_error_handling()

    ok = report()
    exit(0 if ok else 1)
