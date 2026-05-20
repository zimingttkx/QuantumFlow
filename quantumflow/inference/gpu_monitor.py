"""GPU监控 — 后台周期性采集GPU指标，供API和前端使用"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="gpu_monitor")


@dataclass
class GPUSnapshot:
    """单次采样的GPU状态"""

    index: int
    name: str
    total_vram_gb: float
    used_vram_gb: float
    free_vram_gb: float
    utilization_pct: float | None  # GPU计算利用率（执行单元活跃度），None 表示不可用
    memory_util_pct: float | None  # GPU显存带宽利用率（memory控制器活跃度），None 表示不可用
    temperature_c: float | None  # GPU温度，None 表示不可用
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "total_vram_gb": round(self.total_vram_gb, 1),
            "used_vram_gb": round(self.used_vram_gb, 1),
            "free_vram_gb": round(self.free_vram_gb, 1),
            "utilization_pct": round(self.utilization_pct, 1) if self.utilization_pct is not None else None,
            "memory_util_pct": round(self.memory_util_pct, 1) if self.memory_util_pct is not None else None,
            "temperature_c": round(self.temperature_c, 1) if self.temperature_c is not None else None,
            "timestamp": self.timestamp,
        }


class GPUMonitor:
    """
    GPU监控器 — 后台周期性采集GPU指标。

    使用 pynvml 作为主数据源，torch.cuda 作为 fallback。
    """

    def __init__(self, interval_seconds: float = 5.0):
        self.interval_seconds = interval_seconds
        self._latest: list[GPUSnapshot] = []
        self._task: asyncio.Task | None = None
        self._running = False
        self._subscribers: list[Callable[[list[GPUSnapshot]], Any]] = []
        self._sample_count: int = 0

        # pynvml 初始化
        self._pynvml_available = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml_available = True
            self._gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            # nvmlInit 可能成功但后续操作失败，确保清理
            if self._pynvml_available:
                try:
                    import pynvml

                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                self._pynvml_available = False
            self._gpu_count = 0

    # ── public API ─────────────────────────────────────────

    @property
    def latest(self) -> list[GPUSnapshot]:
        """最新的GPU快照列表"""
        return self._latest

    def subscribe(self, callback: Callable[[list[GPUSnapshot]], Any]):
        """注册回调，每次采样后调用"""
        self._subscribers.append(callback)

    async def start(self):
        """启动后台监控"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("gpu_monitor_started", interval_seconds=self.interval_seconds)

    async def stop(self):
        """停止监控"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("gpu_monitor_stopped", samples_collected=self._sample_count)

    def collect_snapshot(self) -> list[GPUSnapshot]:
        """执行一次同步采集（用于测试或按需查询）"""
        return self._read_gpu_state()

    # ── internal ───────────────────────────────────────────

    async def _monitor_loop(self):
        while self._running:
            try:
                snapshots = self._read_gpu_state()
                self._latest = snapshots
                self._sample_count += 1

                # 通知订阅者
                for cb in self._subscribers:
                    try:
                        result = cb(snapshots)
                        if hasattr(result, "__await__"):
                            await result
                    except Exception:
                        pass

                # 结构化日志（每10次采样输出一次，避免刷屏）
                if self._sample_count % 10 == 0 and snapshots:
                    s = snapshots[0]
                    logger.info(
                        "gpu_sample",
                        sample=self._sample_count,
                        free_vram_gb=round(s.free_vram_gb, 1),
                        used_vram_gb=round(s.used_vram_gb, 1),
                        gpu_util_pct=round(s.utilization_pct, 1),
                        mem_util_pct=round(s.memory_util_pct, 1),
                        temp_c=round(s.temperature_c, 1),
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("gpu_monitor_error", exc_info=True)

            await asyncio.sleep(self.interval_seconds)

    def _read_gpu_state(self) -> list[GPUSnapshot]:
        """读取GPU状态（同步）"""
        if self._pynvml_available:
            return self._read_via_pynvml()
        return self._read_via_torch()

    def _read_via_pynvml(self) -> list[GPUSnapshot]:
        import pynvml

        snapshots = []
        for i in range(self._gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = 0
                snapshots.append(
                    GPUSnapshot(
                        index=i,
                        name=name,
                        total_vram_gb=mem.total / (1024**3),
                        used_vram_gb=mem.used / (1024**3),
                        free_vram_gb=mem.free / (1024**3),
                        utilization_pct=util.gpu,  # 计算利用率：CUDA核心活跃度
                        memory_util_pct=util.memory,  # 显存带宽利用率：HBM控制器活跃度
                        temperature_c=temp,
                    )
                )
            except Exception:
                pass
        return snapshots

    def _read_via_torch(self) -> list[GPUSnapshot]:
        try:
            import torch

            if not torch.cuda.is_available():
                return []
            snapshots = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                snapshots.append(
                    GPUSnapshot(
                        index=i,
                        name=props.name,
                        total_vram_gb=total,
                        used_vram_gb=allocated,
                        free_vram_gb=total - allocated,
                        # torch 不提供单独的计算/memory 利用率指标，返回 None 表示不可用
                        utilization_pct=None,
                        memory_util_pct=None,
                        # torch 不提供 GPU 温度，返回 None 表示不可用
                        temperature_c=None,
                    )
                )
            return snapshots
        except Exception:
            return []

    def __del__(self):
        if self._pynvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
