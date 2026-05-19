"""Gang调度策略 - 用于大模型"""

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
    Gang调度策略

    特点：
    - 所有GPU同时分配，要么全部分配要么不分配
    - 适用于大模型（>30B参数）
    - 确保GPU之间的高速通信
    """

    def __init__(self, config: dict = None):
        super().__init__(StrategyType.GANG)
        self.config = config or {}
        self.min_model_size = self.config.get("min_model_size", 30_000_000_000)  # 30B

    @property
    def name(self) -> str:
        return "gang"

    def can_handle(self, request: SchedulingRequest, available_nodes: list[NodeResource]) -> bool:
        """检查是否可以使用Gang调度"""
        # 检查是否有足够的GPU
        required_gpus = request.model_config.get("tensor_parallel", 1)

        healthy_nodes = self.filter_healthy_nodes(available_nodes)
        total_available_gpus = sum(len(n.available_gpus) for n in healthy_nodes)

        if total_available_gpus < required_gpus:
            return False

        # 大模型优先使用Gang调度
        model_size = request.model_size
        if model_size > 0:
            return model_size >= self.min_model_size

        # 根据参数量估算
        return request.max_tokens > 2048 or request.priority >= 8

    def select_nodes(
        self, request: SchedulingRequest, available_nodes: list[NodeResource]
    ) -> SchedulingResult:
        """选择最优节点组合"""
        required_gpus = request.model_config.get("tensor_parallel", 1)
        required_pipeline = request.model_config.get("pipeline_parallel", 1)

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
        )

        # 策略：贪心选择GPU数量最多的节点
        sorted_nodes = sorted(
            healthy_nodes,
            key=lambda n: (len(n.available_gpus), -n.load),
            reverse=True,
        )

        selected_nodes: list[str] = []
        selected_node_objects: list[NodeResource] = []
        selected_gpus: dict[str, list[int]] = {}
        total_selected_gpus = 0

        for node in sorted_nodes:
            if total_selected_gpus >= required_gpus:
                break

            # 计算从该节点需要分配的GPU数
            need_from_node = required_gpus - total_selected_gpus

            # 获取该节点的可用GPU
            node_available = node.available_gpus[:need_from_node]

            if not node_available:
                continue

            selected_nodes.append(node.node_id)
            selected_node_objects.append(node)
            selected_gpus[node.node_id] = [gpu.gpu_id for gpu in node_available]
            total_selected_gpus += len(node_available)

        # 检查是否分配成功
        if total_selected_gpus < required_gpus:
            available = sum(len(n.available_gpus) for n in healthy_nodes)
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason=f"Insufficient GPUs: need {required_gpus}, available {available}",
            )

        # 估算等待时间和延迟
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
            },
        )
