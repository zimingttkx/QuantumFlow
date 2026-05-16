"""Pack调度策略 - 用于小模型"""

from typing import List
import structlog

from quantumflow.scheduler.strategy.base import (
    SchedulingRequest,
    SchedulingResult,
    NodeResource,
    SchedulingStrategy,
    StrategyType,
)

logger = structlog.get_logger().bind(component="scheduler_pack")


class PackSchedulingStrategy(SchedulingStrategy):
    """
    Pack调度策略

    特点：
    - 允许多个小请求共享GPU
    - 适用于小模型（<30B参数）
    - 最大化资源利用率
    """

    def __init__(self, config: dict = None):
        super().__init__(StrategyType.PACK)
        self.config = config or {}
        self.max_model_size = self.config.get("max_model_size", 30_000_000_000)  # 30B

    @property
    def name(self) -> str:
        return "pack"

    def can_handle(
        self, request: SchedulingRequest, available_nodes: List[NodeResource]
    ) -> bool:
        """检查是否可以使用Pack调度"""
        healthy_nodes = self.filter_healthy_nodes(available_nodes)

        if not healthy_nodes:
            return False

        # 小模型使用Pack调度
        model_size = request.model_size
        if model_size > 0:
            return model_size < self.max_model_size

        # 默认策略：没有模型信息时使用Pack
        return True

    def select_nodes(
        self, request: SchedulingRequest, available_nodes: List[NodeResource]
    ) -> SchedulingResult:
        """选择最优节点"""
        healthy_nodes = self.filter_healthy_nodes(available_nodes)

        if not healthy_nodes:
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason="No healthy nodes available",
            )

        # 策略：选择负载最低的节点
        sorted_nodes = sorted(healthy_nodes, key=lambda n: n.load)

        best_node = sorted_nodes[0]

        # 检查是否有可用GPU
        if not best_node.available_gpus:
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason=f"Node {best_node.node_id} has no available GPUs",
            )

        # 只分配一个GPU
        selected_gpu = best_node.available_gpus[0]

        wait_time = self.estimate_wait_time(request, [best_node])
        latency = self.estimate_latency(request, [best_node])

        logger.info(
            "pack_scheduling_success",
            request_id=request.request_id,
            selected_node=best_node.node_id,
            selected_gpu=selected_gpu.gpu_id,
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
                "tensor_parallel": 1,
                "load": best_node.load,
            },
        )
