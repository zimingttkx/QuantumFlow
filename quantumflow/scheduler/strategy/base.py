"""调度策略基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StrategyType(str, Enum):
    """策略类型"""

    GANG = "gang"
    PACK = "pack"
    ADAPTIVE = "adaptive"


@dataclass
class SchedulingRequest:
    """调度请求"""

    request_id: str
    model: str
    model_config: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    prompt_length: int = 0
    max_tokens: int = 2048
    priority: int = 5
    session_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    deadline: float | None = None
    retry_count: int = 0  # 追踪重试次数

    @property
    def model_size(self) -> int:
        """获取模型参数量"""
        return self.model_config.get("parameter_count", 0)

    @property
    def estimated_memory(self) -> int:
        """估算所需显存（字节）"""
        # 粗略估算：每10亿参数需要约2GB显存（fp16）
        return (self.model_size // 1_000_000_000) * 2 * 1024**3


@dataclass
class GPUResource:
    """GPU资源"""

    gpu_id: int
    memory_total: int
    memory_used: int
    utilization: float
    temperature: float
    node_id: str = ""

    @property
    def memory_available(self) -> int:
        """可用显存"""
        return self.memory_total - self.memory_used

    @property
    def memory_free_percent(self) -> float:
        """可用显存百分比"""
        if self.memory_total == 0:
            return 0.0
        return self.memory_available / self.memory_total


@dataclass
class NodeResource:
    """节点资源"""

    node_id: str
    hostname: str
    ip: str
    status: str
    gpu_count: int
    gpus: list[GPUResource]
    cpu_count: int
    memory_total: int
    memory_available: int
    disk_total: int
    disk_available: int
    load: float
    labels: dict[str, str] = field(default_factory=dict)
    loaded_models: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    last_heartbeat: datetime = field(default_factory=datetime.now)

    @property
    def available_gpus(self) -> list[GPUResource]:
        """获取可用GPU"""
        return [gpu for gpu in self.gpus if gpu.memory_free_percent > 0.1]

    @property
    def total_available_memory(self) -> int:
        """总可用显存"""
        return sum(gpu.memory_available for gpu in self.gpus)

    @property
    def is_healthy(self) -> bool:
        """节点是否健康"""
        return self.status == "healthy"

    def can_fit_model(self, model_config: dict[str, Any]) -> bool:
        """检查是否能容纳模型"""
        required_memory = model_config.get("estimated_memory", 0)
        required_gpus = model_config.get("tensor_parallel", 1)

        if len(self.available_gpus) < required_gpus:
            return False

        available_memory = sum(gpu.memory_available for gpu in self.available_gpus[:required_gpus])

        return available_memory >= required_memory


@dataclass
class SchedulingResult:
    """调度结果"""

    success: bool
    assigned_nodes: list[str] = field(default_factory=list)
    assigned_gpus: dict[str, list[int]] = field(default_factory=dict)
    estimated_wait_time: float = 0.0
    estimated_latency: float = 0.0
    strategy_used: str = ""
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SchedulingStrategy(ABC):
    """调度策略基类"""

    def __init__(self, strategy_type: StrategyType):
        self.strategy_type = strategy_type

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @abstractmethod
    def can_handle(self, request: SchedulingRequest, available_nodes: list[NodeResource]) -> bool:
        """判断是否可以使用此策略"""
        pass

    @abstractmethod
    def select_nodes(
        self, request: SchedulingRequest, available_nodes: list[NodeResource]
    ) -> SchedulingResult:
        """选择最优节点组合"""
        pass

    def estimate_wait_time(self, request: SchedulingRequest, nodes: list[NodeResource]) -> float:
        """估算等待时间（秒）"""
        if not nodes:
            return float("inf")

        # 简单估算：基于平均负载
        avg_load = sum(n.load for n in nodes) / len(nodes)
        base_time = request.max_tokens / 1000.0  # 粗略估算

        return base_time * (1 + avg_load)

    def estimate_latency(self, request: SchedulingRequest, nodes: list[NodeResource]) -> float:
        """估算推理延迟（秒）"""
        # 简单估算
        prompt_tokens = request.prompt_length // 4
        completion_tokens = request.max_tokens

        # 假设每token处理时间与模型大小和负载相关
        time_per_token = 0.01 * (1 + sum(n.load for n in nodes) / len(nodes))

        return (prompt_tokens + completion_tokens) * time_per_token

    def filter_healthy_nodes(self, nodes: list[NodeResource]) -> list[NodeResource]:
        """过滤健康节点"""
        return [n for n in nodes if n.is_healthy]
