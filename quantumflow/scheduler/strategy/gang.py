"""Gang调度策略 - 用于大模型与高优先级请求"""

from __future__ import annotations

import structlog

from quantumflow.scheduler.strategy.base import (
    NodeResource,
    SchedulingRequest,
    SchedulingResult,
    SchedulingStrategy,
    StrategyType,
)

logger = structlog.get_logger().bind(component="scheduler_gang")


class GangSchedulingStrategy(SchedulingStrategy):
    """
    Gang 调度策略

    特点：
    - 所有 GPU 同时分配，要么全部分配要么不分配
    - 适用于大模型（>30B 参数）或严格 QoS 请求
    - 感知 NVLink 拓扑：优先选同域 GPU
    - 感知 GPU 家族：满足 preferred_gpu_families
    - 感知量化与 per-GPU 显存需求
    """

    def __init__(self, config: dict | None = None):
        super().__init__(StrategyType.GANG)
        self.config = config or {}
        self.min_model_size = self.config.get("min_model_size", 30_000_000_000)  # 30B

    @property
    def name(self) -> str:
        return "gang"

    # ------------------------------------------------------------------ helpers

    def _required_tp(self, request: SchedulingRequest) -> int:
        """读取张量并行度（新字段优先，回退到老字段）"""
        return max(
            int(request.recommended_tensor_parallel)
            if request.recommended_tensor_parallel > 0
            else 1,
            int(request.model_config.get("tensor_parallel", 1)),
        )

    def _required_pp(self, request: SchedulingRequest) -> int:
        """读取流水线并行度"""
        return max(
            int(request.recommended_pipeline_parallel)
            if request.recommended_pipeline_parallel > 0
            else 1,
            int(request.model_config.get("pipeline_parallel", 1)),
        )

    def _per_gpu_memory_required(self, request: SchedulingRequest) -> int:
        """读取 per-GPU 显存需求（字节）"""
        # 优先用新字段估算
        if request.parameter_count > 0:
            return request.estimated_memory_per_gpu_bytes
        # 回退到老字段
        return int(request.model_config.get("estimated_memory", 0))

    def _gpu_matches_family(self, gpu, request: SchedulingRequest) -> bool:
        if not request.preferred_gpu_families:
            return True
        return gpu.model_family in request.preferred_gpu_families

    def _pick_within_nvlink_domain(
        self, node: NodeResource, need: int
    ) -> list | None:
        """挑选 GPU 支持 Gang 分配

        优先级：
        1. 同 NVLink 域内有 >= need 张 → 选该域内算力最高的 need 张
        2. 否则：把节点内所有可用 GPU 全部返回（调用方会跨节点聚合）
           只要节点至少 1 张可用 GPU 就返回，避免阻断跨节点 Gang
        3. 节点 0 张可用 → 返回 None
        """
        if not node.gpus:
            return None
        # 统计各 NVLink 域的可用 GPU
        domains: dict[str, list] = {}
        for gpu in node.gpus:
            if gpu.memory_free_percent <= 0.1:
                continue
            domain = gpu.nvlink_domain_id or f"__nondomain__{gpu.gpu_id}"
            domains.setdefault(domain, []).append(gpu)
        # 选 GPU 数 >= need 的最大域
        viable = [(d, gs) for d, gs in domains.items() if len(gs) >= need]
        if viable:
            viable.sort(
                key=lambda kv: sum(g.estimated_relative_throughput for g in kv[1]),
                reverse=True,
            )
            chosen_domain = viable[0][1]
            chosen_domain.sort(
                key=lambda g: g.estimated_relative_throughput, reverse=True
            )
            return chosen_domain[:need]
        # 无单域满足：聚合跨节点 — 返回本节点全部可用 GPU（至少 1 张）
        all_gpus = [g for g in node.gpus if g.memory_free_percent > 0.1]
        if not all_gpus:
            return None
        all_gpus.sort(key=lambda g: g.estimated_relative_throughput, reverse=True)
        # 返回最多 need 张，让调用方按需累计
        return all_gpus[:need]

    # ------------------------------------------------------------------ public

    def can_handle(self, request: SchedulingRequest, available_nodes: list[NodeResource]) -> bool:
        """检查是否可以使用 Gang 调度

        满足任一即返回 True：
        - 模型 > 30B
        - 高优先级（>=8）
        - 长输出（>4096）
        - TP >= 2
        - 用户指定了 preferred_gpu_families
        """
        required_gpus = self._required_tp(request)
        healthy_nodes = self.filter_healthy_nodes(available_nodes)
        total_available = sum(
            min(
                len(n.available_gpus),
                max(self._gpu_count_in_any_nvlink_domain(n), required_gpus),
            )
            for n in healthy_nodes
        )
        if total_available < required_gpus:
            return False
        if request.model_size >= self.min_model_size:
            return True
        if request.priority >= 8:
            return True
        if request.max_tokens >= 4096:
            return True
        if request.recommended_tensor_parallel >= 2:
            return True
        if request.preferred_gpu_families:
            return True
        return False

    def _gpu_count_in_any_nvlink_domain(self, node: NodeResource) -> int:
        if not node.gpus:
            return 0
        domains: dict[str, int] = {}
        for gpu in node.gpus:
            if gpu.memory_free_percent <= 0.1:
                continue
            d = gpu.nvlink_domain_id or f"__nondomain__{gpu.gpu_id}"
            domains[d] = domains.get(d, 0) + 1
        return max(domains.values()) if domains else 0

    def select_nodes(
        self, request: SchedulingRequest, available_nodes: list[NodeResource]
    ) -> SchedulingResult:
        """选择最优节点组合

        算法：
        1. 过滤健康节点
        2. 按 (NVLink 域内最大 GPU 数, 总可用 GPU, 1/(load+ε), 1/estimated_relative_throughput) 排序
        3. 在每个候选节点上按 NVLink 域分配 GPU
        4. 检查每张被选 GPU 的 per-GPU 显存是否够
        5. 累计到 required_tp 为止
        """
        required_gpus = self._required_tp(request)
        required_pipeline = self._required_pp(request)
        per_gpu_mem = self._per_gpu_memory_required(request)

        healthy_nodes = self.filter_healthy_nodes(available_nodes)
        if not healthy_nodes:
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason="No healthy nodes available",
            )

        logger.info(
            "gang_scheduling",
            request_id=request.request_id,
            required_gpus=required_gpus,
            available_nodes=len(healthy_nodes),
            per_gpu_mem_gb=per_gpu_mem / 1024**3,
            preferred_families=request.preferred_gpu_families,
        )

        # 排序：NVLink 域内最多 GPU 的节点优先；同域内高算力优先
        def node_score(n: NodeResource) -> tuple:
            domain_max = self._gpu_count_in_any_nvlink_domain(n)
            return (
                domain_max,  # 1. 域内 GPU 多优先
                len(n.available_gpus),  # 2. 总可用 GPU
                -n.load,  # 3. 负载小优先
            )

        sorted_nodes = sorted(healthy_nodes, key=node_score, reverse=True)

        selected_nodes: list[str] = []
        selected_node_objects: list[NodeResource] = []
        selected_gpus: dict[str, list[int]] = {}
        total_selected_gpus = 0

        for node in sorted_nodes:
            if total_selected_gpus >= required_gpus:
                break

            need_from_node = required_gpus - total_selected_gpus
            picks = self._pick_within_nvlink_domain(node, need_from_node)
            if not picks:
                continue

            # 进一步过滤：满足 GPU 家族 + 显存
            valid_picks = []
            for gpu in picks:
                if not self._gpu_matches_family(gpu, request):
                    continue
                if per_gpu_mem > 0 and node.effective_available_memory_per_gpu(gpu.gpu_id) < per_gpu_mem:
                    continue
                valid_picks.append(gpu)

            if not valid_picks:
                continue

            # 按算力降序取最多 need_from_node
            valid_picks.sort(key=lambda g: g.estimated_relative_throughput, reverse=True)
            take = valid_picks[:need_from_node]

            selected_nodes.append(node.node_id)
            selected_node_objects.append(node)
            selected_gpus[node.node_id] = [gpu.gpu_id for gpu in take]
            total_selected_gpus += len(take)

        if total_selected_gpus < required_gpus:
            available = sum(len(n.available_gpus) for n in healthy_nodes)
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason=(
                    f"Insufficient GPUs: need {required_gpus}, "
                    f"available {available} (after family+memory filter)"
                ),
            )

        wait_time = self.estimate_wait_time(request, selected_node_objects)
        latency = self.estimate_latency(request, selected_node_objects)

        logger.info(
            "gang_scheduling_success",
            request_id=request.request_id,
            selected_nodes=selected_nodes,
            selected_gpus=selected_gpus,
            estimated_wait_time=wait_time,
        )

        return SchedulingResult(
            success=True,
            assigned_nodes=selected_nodes,
            assigned_gpus=selected_gpus,
            estimated_wait_time=wait_time,
            estimated_latency=latency,
            strategy_used=self.name,
            metadata={
                "total_gpus": total_selected_gpus,
                "tensor_parallel": required_gpus,
                "pipeline_parallel": required_pipeline,
                "per_gpu_mem_gb": per_gpu_mem / 1024**3,
            },
        )
