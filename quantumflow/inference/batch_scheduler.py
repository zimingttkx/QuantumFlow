"""Per-GPU 批处理调度器 — 根据 VRAM 状态动态调整 batch_size"""

import time
from typing import TYPE_CHECKING

import structlog

from quantumflow.inference.batch_config import DynamicBatchConfig

if TYPE_CHECKING:
    from quantumflow.inference.vram_manager import VRAMManager

logger = structlog.get_logger().bind(component="batch_scheduler")


class BatchScheduler:
    """
    Per-GPU 批处理调度器

    根据 VRAM 利用率动态计算最优 batch_size：
    - 高显存压力：减少 batch size 避免 OOM
    - 低显存压力 + 高 pending：增加 batch size 提高吞吐

    算法：
    ```
    if vram_utilization > threshold_high:
        batch_size = max(min_batch_size, base * 0.5)
    elif vram_utilization < threshold_low and pending > base * 2:
        batch_size = min(max_batch_size, base * 1.5)
    else:
        batch_size = base
    ```
    """

    def __init__(
        self,
        vram_manager: "VRAMManager",
        config: DynamicBatchConfig | None = None,
    ):
        """
        Args:
            vram_manager: VRAM 管理器
            config: 动态批处理配置
        """
        self._vram = vram_manager
        self._config = config or DynamicBatchConfig()
        self._current_batch_size = self._config.base_max_batch_size
        self._last_vram_check = 0.0
        self._last_vram_utilization = 0.0

    def compute_batch_size(self, pending_count: int) -> int:
        """
        根据 VRAM 利用率和待处理请求数计算 batch size

        Args:
            pending_count: 当前等待处理的请求数

        Returns:
            建议的 batch size
        """
        utilization = self._vram.get_vram_utilization()
        self._last_vram_utilization = utilization
        self._last_vram_check = time.time()

        base = self._config.base_max_batch_size
        high = self._config.vram_threshold_high
        low = self._config.vram_threshold_low

        if utilization > high:
            # 高显存压力：减少 batch size 到最小值
            new_size = max(self._config.min_batch_size, int(base * 0.5))
            self._current_batch_size = new_size
            logger.debug(
                "batch_size_reduced_high_vram",
                utilization=utilization,
                new_batch_size=new_size,
            )
        elif utilization < low and pending_count > base * 2:
            # 低显存压力 + 高 pending：可以增加 batch size
            new_size = min(self._config.max_batch_size, int(base * 1.5))
            self._current_batch_size = new_size
            logger.debug(
                "batch_size_increased_low_vram",
                utilization=utilization,
                pending_count=pending_count,
                new_batch_size=new_size,
            )
        else:
            # 正常情况：使用基础大小
            self._current_batch_size = base

        return self._current_batch_size

    def get_current_batch_size(self) -> int:
        """获取当前 batch size"""
        return self._current_batch_size

    def get_last_vram_utilization(self) -> float:
        """获取上次检查时的 VRAM 利用率"""
        return self._last_vram_utilization

    def get_config(self) -> DynamicBatchConfig:
        """获取配置"""
        return self._config
