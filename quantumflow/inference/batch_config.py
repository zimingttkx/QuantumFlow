"""动态批处理配置"""

from dataclasses import dataclass


@dataclass
class DynamicBatchConfig:
    """
    动态批处理配置

    根据 VRAM 利用率动态调整 batch_size：
    - 高显存压力 (> vram_threshold_high) 时减少 batch size
    - 低显存压力 (< vram_threshold_low) 且 pending 请求多时增加 batch size

    Attributes:
        base_max_batch_size: 基础最大批量大小
        min_batch_size: 最小批量大小（高显存压力时下限）
        max_batch_size: 最大批量大小（低显存压力时上限）
        vram_check_interval_ms: VRAM 检查间隔（毫秒）
        vram_threshold_high: 高显存阈值（超过此值减少 batch size）
        vram_threshold_low: 低显存阈值（低于此值且 pending 多时增加 batch size）
        dynamic_factor: 动态调整系数
    """

    base_max_batch_size: int = 8
    min_batch_size: int = 2
    max_batch_size: int = 16
    vram_check_interval_ms: int = 500
    vram_threshold_high: float = 0.9
    vram_threshold_low: float = 0.7
    dynamic_factor: float = 1.0
