"""Pack调度策略 - 用于小模型与共享场景"""

from __future__ import annotations

import structlog

from quantumflow.scheduler.strategy.base import (
    NodeResource,
    SchedulingRequest,
    SchedulingResult,
    SchedulingStrategy,
    StrategyType,
)

logger = structlog.get_logger().bind(component="scheduler_pack")


class PackSchedulingStrategy(SchedulingStrategy):
    """
    Pack 调度策略

    特点：
    - 允许多个小请求共享 GPU（同一节点内紧凑打包）
    - 适用于小模型（<30B 参数）
    - 最大化资源利用率
    - 按真实空闲显存 + 算力 + 负载综合评分
    """

    def __init__(self, config: dict | None = None):
        super().__init__(StrategyType.PACK)
        self.config = config or {}
        self.max_model_size = self.config.get("max_model_size", 30_000_000_000)  # 30B

    @property
    def name(self) -> str:
        return "pack"

    # ------------------------------------------------------------------ helpers

    def _required_tp(self, request: SchedulingRequest) -> int:
        return max(
            int(request.recommended_tensor_parallel)
            if request.recommended_tensor_parallel > 0
            else 1,
            int(request.model_config.get("tensor_parallel", 1)),
        )

    def _per_gpu_memory_required(self, request: SchedulingRequest) -> int:
        if request.parameter_count > 0:
            return request.estimated_memory_per_gpu_bytes
        return int(request.model_config.get("estimated_memory", 0))

    def _gpu_matches_family(self, gpu, request: SchedulingRequest) -> bool:
        if not request.preferred_gpu_families:
            return True
        return gpu.model_family in request.preferred_gpu_families

    def _node_real_load(self, node: NodeResource) -> float:
        """真实负载估算

        优先级：
        1. ``node.load`` （显式上报，包括 CPU/内存）— 通常最权威
        2. 计算值（基于 GPU 显存 + 利用率）— 作为兜底

        返回两者中较大者，避免在 GPU 短时低 util 时误判节点空闲。
        """
        if not node.gpus:
            return 1.0  # 满
        mem_used_ratio = 1.0 - (
            node.total_available_memory
            / max(1, sum(g.memory_total for g in node.gpus))
        )
        util_ratio = sum(g.utilization for g in node.gpus) / len(node.gpus)
        computed = max(0.0, min(1.0, 0.6 * mem_used_ratio + 0.4 * util_ratio))
        # 当 node.load 显式设置时（>0），优先用它；否则用计算值
        if node.load > 0:
            return float(node.load)
        return computed

    def _score_node(self, node: NodeResource, request: SchedulingRequest) -> float:
        """节点综合评分（越大越好）

        评分维度：
        - 有效可用显存（决定能不能装下）
        - 算力（异构 GPU 时高算力优先）
        - 真实负载（低负载优先）
        - 模型是否已加载（已加载 +50，避免重复加载）
        """
        if not node.available_gpus:
            return float("-inf")
        # GPU 家族匹配
        if request.preferred_gpu_families:
            matching = [g for g in node.available_gpus if self._gpu_matches_family(g, request)]
            if not matching:
                return float("-inf")
        else:
            matching = node.available_gpus

        # 算力均值
        avg_throughput = sum(g.estimated_relative_throughput for g in matching) / len(matching)
        # 真实可用显存
        total_free = sum(node.effective_available_memory_per_gpu(g.gpu_id) for g in matching)
        # 真实负载
        load = self._node_real_load(node)
        # 已加载模型奖励
        loaded_bonus = 50.0 if node.can_serve_model(request.model) else 0.0

        # 归一化
        throughput_score = min(avg_throughput / 1000.0, 5.0)  # 限制到 [0, 5]
        mem_score = min(total_free / (50 * 1024**3), 5.0)
        load_score = 5.0 * (1.0 - load)
        return throughput_score + mem_score + load_score + loaded_bonus

    # ------------------------------------------------------------------ public

    def can_handle(self, request: SchedulingRequest, available_nodes: list[NodeResource]) -> bool:
        """检查是否可以使用 Pack 调度

        满足任一即返回 True：
        - 小模型（<30B）
        - 模型信息未知（兜底）
        - 高 TP（>=2）被 Gang 接管时此处返回 False
        """
        if request.recommended_tensor_parallel >= 2:
            return False
        if request.preferred_gpu_families:
            # 用户明确指定家族，由 Gang 处理更稳
            return False
        healthy_nodes = self.filter_healthy_nodes(available_nodes)
        if not healthy_nodes:
            return False
        if request.model_size > 0:
            return request.model_size < self.max_model_size
        return True

    def select_nodes(
        self, request: SchedulingRequest, available_nodes: list[NodeResource]
    ) -> SchedulingResult:
        """选择最优节点

        算法：
        1. 过滤健康节点
        2. 计算每个节点的综合评分
        3. 取评分最高的节点
        4. 在该节点上选一张最匹配的 GPU（已加载模型 / 算力最高 / 显存够）
        5. 校验 per-GPU 显存
        """
        healthy_nodes = self.filter_healthy_nodes(available_nodes)
        if not healthy_nodes:
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason="No healthy nodes available",
            )

        per_gpu_mem = self._per_gpu_memory_required(request)

        # 节点打分
        scored: list[tuple[float, NodeResource]] = []
        for n in healthy_nodes:
            score = self._score_node(n, request)
            if score == float("-inf"):
                continue
            scored.append((score, n))
        if not scored:
            # 区分原因：所有节点都是 0 可用 GPU vs 家族不匹配
            any_has_gpu = any(n.available_gpus for n in healthy_nodes)
            if not any_has_gpu:
                return SchedulingResult(
                    success=False,
                    strategy_used=self.name,
                    reason="Selected node has no available GPUs (all full or filtered)",
                )
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason="No nodes match model family / availability constraints",
            )

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_node = scored[0]

        # 在该节点上选最佳 GPU
        candidates = best_node.available_gpus
        if request.preferred_gpu_families:
            candidates = [g for g in candidates if self._gpu_matches_family(g, request)]
        # 优先选已加载同模型的 GPU
        loaded = [g for g in candidates if request.model in best_node.loaded_models]
        if loaded:
            candidates = loaded
        # 按算力 + 显存排序
        candidates.sort(
            key=lambda g: (
                g.estimated_relative_throughput,
                best_node.effective_available_memory_per_gpu(g.gpu_id),
            ),
            reverse=True,
        )
        if not candidates:
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason=f"Node {best_node.node_id} has no matching GPUs",
            )

        selected_gpu = candidates[0]
        if per_gpu_mem > 0 and best_node.effective_available_memory_per_gpu(selected_gpu.gpu_id) < per_gpu_mem:
            # 节点上没有任何一张卡能装下模型
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason=(
                    f"Node {best_node.node_id} GPUs cannot fit per-gpu requirement "
                    f"({per_gpu_mem / 1024**3:.1f} GB)"
                ),
            )

        wait_time = self.estimate_wait_time(request, [best_node])
        latency = self.estimate_latency(request, [best_node])

        logger.info(
            "pack_scheduling_success",
            request_id=request.request_id,
            selected_node=best_node.node_id,
            selected_gpu=selected_gpu.gpu_id,
            score=best_score,
            real_load=self._node_real_load(best_node),
        )

        return SchedulingResult(
            success=True,
            assigned_nodes=[best_node.node_id],
            assigned_gpus={best_node.node_id: [selected_gpu.gpu_id]},
            estimated_wait_time=wait_time,
            estimated_latency=latency,
            strategy_used=self.name,
            metadata={
                "total_gpus": 1,
                "tensor_parallel": self._required_tp(request),
                "real_load": self._node_real_load(best_node),
                "load": best_node.load,
                "score": best_score,
            },
        )
