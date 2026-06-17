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
    tenant_id: str = "default"  # 租户 ID
    session_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    deadline: float | None = None
    retry_count: int = 0  # 追踪重试次数

    # 新增字段：模型感知（用于精确路由）
    model_id: str = ""                              # 规范化 ID
    model_family: str = ""                          # qwen / llama / glm / deepseek
    backend_hint: str = ""                          # vllm / tgi / sglang / huggingface
    quantization: str | None = None                 # awq / gptq / gguf / bnb / None
    parameter_count: int = 0                        # 显式声明参数量
    min_memory_per_gpu_gb: int = 0                  # 推荐 per-GPU 显存
    max_model_len: int = 4096                       # 上下文长度
    recommended_tensor_parallel: int = 1            # 推荐的 TP
    recommended_pipeline_parallel: int = 1          # 推荐的 PP
    preferred_gpu_families: list[str] = field(default_factory=list)  # ["hopper","ampere"]

    @property
    def model_size(self) -> int:
        """获取模型参数量 — 优先用新字段，回退到老字段"""
        if self.parameter_count > 0:
            return self.parameter_count
        return self.model_config.get("parameter_count", 0)

    @property
    def estimated_memory(self) -> int:
        """估算所需总显存（字节）

        向后兼容公式（与现有单元测试对齐）：
            params_in_billions * 2 * 1024^3
        即"每 10 亿参数 ≈ 2 GiB"，量化感知由 estimated_memory_per_gpu_bytes 提供。

        Bug fix (C-B1): 0.3B-0.9B 模型原本被整数除法估成 0 字节，绕过所有
        显存检查，导致 1B 以下模型可能 OOM。现使用向上取整 + 最小 2GB 兜底，
        保证 0.3B 模型至少估算 2GB。
        """
        params = self.model_size
        if params == 0:
            return 0
        # 向上取整: (params + 999_999_999) // 1_000_000_000
        # 最小值 1 → 至少 2GB（避免 0 字节导致绕过所有 per-gpu 校验）
        billions = max(1, (params + 999_999_999) // 1_000_000_000)
        return billions * 2 * 1024**3

    @property
    def estimated_memory_per_gpu_bytes(self) -> int:
        """根据量化和 TP 估算 per-GPU 显存（字节）"""
        params = self.model_size
        if params == 0:
            return 0
        bytes_per_param = {
            None: 2.0, "fp16": 2.0, "bf16": 2.0, "fp32": 4.0,
            "int8": 1.0, "awq": 0.55, "gptq": 0.55,
            "gguf": 0.6, "bnb": 0.6,
        }.get(self.quantization, 2.0)
        # 权重 + 20% 框架开销
        raw = params * bytes_per_param * 1.2
        tp = max(self.recommended_tensor_parallel, 1)
        per_gpu_from_weights = raw / tp
        # KV cache 估算：每 1K context 约 0.1GB per GPU
        kv_overhead_gb = (self.max_model_len / 1024) * 0.1
        total_gb = (per_gpu_from_weights / 1024**3) + kv_overhead_gb
        return int(total_gb * 1024**3)


@dataclass
class GPUResource:
    """GPU资源"""

    gpu_id: int
    memory_total: int
    memory_used: int
    utilization: float
    temperature: float
    node_id: str = ""

    # 新增字段：异构 GPU 感知
    gpu_model: str = "unknown"          # "NVIDIA A100-SXM4-80GB" / "RTX 4090" / "Huawei Ascend 910B"
    compute_capability: tuple[int, int] | None = None  # (major, minor)
    memory_bandwidth_gb_s: float = 0.0  # HBM 带宽
    fp16_tflops: float = 0.0            # FP16 算力
    int8_tflops: float = 0.0            # INT8 算力

    # 新增字段：拓扑感知
    nvlink_domain_id: str = ""          # NVLink/RoCE 域 ID
    nvlink_peer_ids: list[int] = field(default_factory=list)
    pcie_topology: dict[str, str] = field(default_factory=dict)  # gpu_id -> numa/pcie 描述

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

    @property
    def model_family(self) -> str:
        """GPU 家族分类，用于调度亲和"""
        m = self.gpu_model.lower()
        if "h100" in m:
            return "hopper"
        if "a100" in m:
            return "ampere"
        if "h200" in m or "b100" in m or "b200" in m:
            return "blackwell"
        if "v100" in m:
            return "volta"
        if "4090" in m or "4080" in m or "3090" in m or "3080" in m:
            return "consumer"
        if "ascend" in m or "910" in m:
            return "ascend"
        if "mlu" in m or "370" in m:
            return "cambricon"
        if "dcu" in m:
            return "hygon"
        return "unknown"

    @property
    def estimated_relative_throughput(self) -> float:
        """相对吞吐（用于异构集群调度决策）"""
        if self.fp16_tflops > 0:
            return self.fp16_tflops
        # 回退：基于家族估算
        return {
            "hopper": 989.0,
            "ampere": 312.0,
            "blackwell": 1800.0,
            "volta": 125.0,
            "consumer": 165.0,
            "ascend": 320.0,
            "cambricon": 256.0,
            "hygon": 250.0,
            "unknown": 100.0,
        }.get(self.model_family, 100.0)

    def shares_nvlink_with(self, other: "GPUResource") -> bool:
        """判断两张 GPU 是否在同一 NVLink 域"""
        if not self.nvlink_domain_id or not other.nvlink_domain_id:
            return False
        if self.nvlink_domain_id != other.nvlink_domain_id:
            return False
        return other.gpu_id in self.nvlink_peer_ids and self.gpu_id in other.nvlink_peer_ids


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

    # 新增字段：模型加载状态
    model_loading: dict[str, datetime] = field(default_factory=dict)  # model -> 加载开始时间

    # 新增字段：异构 GPU
    primary_gpu_family: str = "unknown"  # 节点主流 GPU 家族

    # 新增字段：调度预留（防止超卖）
    reserved_gpus: dict[str, set[int]] = field(default_factory=dict)   # request_id -> {gpu_id}
    reserved_memory_bytes: int = 0
    # 记录每个 request_id 实际预留的字节数（避免平均分摊误差）
    reserved_memory_by_request: dict[str, int] = field(default_factory=dict)

    @property
    def available_gpus(self) -> list[GPUResource]:
        """获取可用 GPU（显存剩余 > 10% 且未被本节点完全预留）"""
        result = []
        for gpu in self.gpus:
            if gpu.memory_free_percent <= 0.1:
                continue
            # 排除已完全被预留的 GPU
            if self._gpu_fully_reserved(gpu):
                continue
            result.append(gpu)
        return result

    def _gpu_fully_reserved(self, gpu: GPUResource) -> bool:
        """判断单张 GPU 是否被本节点所有预留加起来耗尽了可用显存"""
        per_gpu_reserved = self._reserved_bytes_for_gpu(gpu.gpu_id)
        effective = max(0, gpu.memory_available - per_gpu_reserved)
        return effective < gpu.memory_total * 0.1

    def _reserved_bytes_for_gpu(self, gpu_id: int) -> int:
        """累计指定 GPU 上所有 request 的预留字节"""
        total = 0
        for rid, gpu_ids in self.reserved_gpus.items():
            if gpu_id in gpu_ids:
                total += self.reserved_memory_by_request.get(rid, 0)
        return total

    @property
    def total_available_memory(self) -> int:
        """总有效可用显存（排除预留部分）"""
        return sum(
            max(0, g.memory_available - self._reserved_bytes_for_gpu(g.gpu_id))
            for g in self.gpus
        )

    def effective_available_memory_per_gpu(self, gpu_id: int) -> int:
        """单 GPU 的有效可用显存（真实可用 - 已被本节点预留）"""
        gpu = next((g for g in self.gpus if g.gpu_id == gpu_id), None)
        if gpu is None:
            return 0
        return max(0, gpu.memory_available - self._reserved_bytes_for_gpu(gpu_id))

    def reserve_gpu(self, request_id: str, gpu_id: int, memory_bytes: int) -> None:
        """预留 GPU 和显存（防止超卖）

        多次调用同一 request_id 是累加的（不会重复计费）。
        """
        if request_id not in self.reserved_gpus:
            self.reserved_gpus[request_id] = set()
            self.reserved_memory_by_request[request_id] = 0
        self.reserved_gpus[request_id].add(gpu_id)
        self.reserved_memory_by_request[request_id] += memory_bytes
        self.reserved_memory_bytes += memory_bytes

    def release_reservation(self, request_id: str) -> int:
        """释放预留，返回释放的字节数"""
        if request_id not in self.reserved_gpus:
            return 0
        released = self.reserved_memory_by_request.pop(request_id, 0)
        self.reserved_memory_bytes = max(0, self.reserved_memory_bytes - released)
        del self.reserved_gpus[request_id]
        return released

    def can_serve_model(self, model_name: str) -> bool:
        """节点是否已加载该模型"""
        return model_name in self.loaded_models

    def can_fit_model(
        self,
        required_memory_per_gpu_bytes: int | dict[str, Any] | None = None,
        required_gpus: int = 1,
        model_config: dict[str, Any] | None = None,
    ) -> bool:
        """检查是否能容纳模型

        多种调用方式（全部向后兼容）：
        1. 旧 API（位置参数 dict）：   ``can_fit_model(model_config_dict)``
        2. 旧 API（关键字 model_config）：``can_fit_model(model_config={"estimated_memory": X, "tensor_parallel": N})``
        3. 新 API（显式参）：         ``can_fit_model(required_memory_per_gpu_bytes=20*1024**3, required_gpus=4)``
        """
        # 旧 API 兼容路径 1：位置参数直接传 dict
        if isinstance(required_memory_per_gpu_bytes, dict):
            model_config = required_memory_per_gpu_bytes
            required_memory_per_gpu_bytes = 0

        # 旧 API 兼容路径 2：keyword 传 model_config 或新 API 未指定 per-gpu 内存
        if (
            model_config is not None
            and (required_memory_per_gpu_bytes is None or required_memory_per_gpu_bytes == 0)
        ):
            total_memory = model_config.get("estimated_memory", 0)
            required_gpus = model_config.get("tensor_parallel", 1)
            # estimated_memory == 0 表示"未指定大小"，保守视为能装下（按 GPU 数量判断）
            if required_gpus == 0:
                return False
            available = self.available_gpus
            if len(available) < required_gpus:
                return False
            if total_memory == 0:
                return True
            # 总需求分摊到每张 GPU，再与每张 GPU 的有效可用比较
            per_gpu = total_memory // required_gpus
            for gpu in available[:required_gpus]:
                if self.effective_available_memory_per_gpu(gpu.gpu_id) < per_gpu:
                    return False
            return True

        # 新 API 路径
        assert isinstance(required_memory_per_gpu_bytes, (int, type(None)))
        required_memory_per_gpu_bytes = required_memory_per_gpu_bytes or 0
        if required_memory_per_gpu_bytes == 0 or required_gpus == 0:
            return False
        if len(self.available_gpus) < required_gpus:
            return False
        # Bug fix (H-C2): 原本按 throughput 排序选 top-N,只检查这 N 张。
        # 但 top-2 中第二张不够而第三张够时,会错误返回 False。
        # 修复: 先按 per-gpu 显存过滤掉装不下的,再按 throughput 排序选 N 张。
        fit_gpus = [
            g for g in self.available_gpus
            if self.effective_available_memory_per_gpu(g.gpu_id) >= required_memory_per_gpu_bytes
        ]
        if len(fit_gpus) < required_gpus:
            return False
        # 按算力从高到低选 GPU
        sorted_fit = sorted(
            fit_gpus,
            key=lambda g: g.estimated_relative_throughput,
            reverse=True,
        )
        for gpu in sorted_fit[:required_gpus]:
            if self.effective_available_memory_per_gpu(gpu.gpu_id) < required_memory_per_gpu_bytes:
                # 理论上不会发生(因为 fit_gpus 已经过滤过),防御性检查
                return False
        return True

    @property
    def is_healthy(self) -> bool:
        """节点是否健康"""
        return self.status == "healthy"


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
