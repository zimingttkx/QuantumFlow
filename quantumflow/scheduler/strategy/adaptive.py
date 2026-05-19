"""自适应调度策略"""

from collections.abc import Callable

import structlog

from quantumflow.scheduler.strategy.base import (
    NodeResource,
    SchedulingRequest,
    SchedulingResult,
    SchedulingStrategy,
    StrategyType,
)

logger = structlog.get_logger().bind(component="scheduler_adaptive")


class AdaptiveSchedulingStrategy(SchedulingStrategy):
    """
    自适应调度策略

    特点：
    - 根据请求特征自动选择最优策略
    - 支持自定义规则
    - 灵活可扩展
    """

    def __init__(
        self,
        strategies: dict[str, SchedulingStrategy] = None,
        rules: list[dict] = None,
    ):
        super().__init__(StrategyType.ADAPTIVE)
        self.strategies = strategies or {}
        self.rules = rules or self._default_rules()

    @property
    def name(self) -> str:
        return "adaptive"

    def _default_rules(self) -> list[dict]:
        """默认调度规则"""
        return [
            # 规则1：大模型使用Gang调度
            {
                "condition": lambda r, n: r.model_size > 70_000_000_000,  # >70B
                "strategy": "gang",
                "priority": 10,
            },
            # 规则2：高优先级请求使用Gang调度
            {
                "condition": lambda r, n: r.priority >= 8,
                "strategy": "gang",
                "priority": 9,
            },
            # 规则3：长输出使用Gang调度
            {
                "condition": lambda r, n: r.max_tokens > 4096,
                "strategy": "gang",
                "priority": 7,
            },
            # 规则4：默认使用Pack调度
            {
                "condition": lambda r, n: True,
                "strategy": "pack",
                "priority": 1,
            },
        ]

    def _evaluate_condition(
        self, condition: Callable, request: SchedulingRequest, nodes: list[NodeResource]
    ) -> bool:
        """评估条件"""
        try:
            return condition(request, nodes)
        except Exception:
            return False

    def _select_best_strategy(
        self, request: SchedulingRequest, available_nodes: list[NodeResource]
    ) -> str:
        """选择最佳策略"""
        # 按优先级排序规则
        sorted_rules = sorted(self.rules, key=lambda r: r.get("priority", 0), reverse=True)

        for rule in sorted_rules:
            condition = rule.get("condition")
            if condition and self._evaluate_condition(condition, request, available_nodes):
                strategy_name = rule.get("strategy")
                if strategy_name in self.strategies:
                    return strategy_name

        # 默认使用Pack
        return "pack"

    def can_handle(self, request: SchedulingRequest, available_nodes: list[NodeResource]) -> bool:
        """总是可以使用自适应策略"""
        return True

    def select_nodes(
        self, request: SchedulingRequest, available_nodes: list[NodeResource]
    ) -> SchedulingResult:
        """使用自适应策略选择节点"""
        if not available_nodes:
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason="No available nodes",
            )

        # 选择最佳策略
        strategy_name = self._select_best_strategy(request, available_nodes)
        strategy = self.strategies.get(strategy_name)

        if not strategy:
            logger.warning(
                "strategy_not_found",
                strategy=strategy_name,
                request_id=request.request_id,
            )
            # 回退到Pack
            strategy = self.strategies.get("pack")
            if not strategy:
                return SchedulingResult(
                    success=False,
                    strategy_used=self.name,
                    reason="No available strategy",
                )
            strategy_name = "pack"

        logger.info(
            "adaptive_strategy_selected",
            request_id=request.request_id,
            strategy=strategy_name,
            model=request.model,
            priority=request.priority,
        )

        # 使用选中的策略
        result = strategy.select_nodes(request, available_nodes)

        # 更新策略名称
        return SchedulingResult(
            success=result.success,
            assigned_nodes=result.assigned_nodes,
            assigned_gpus=result.assigned_gpus,
            estimated_wait_time=result.estimated_wait_time,
            estimated_latency=result.estimated_latency,
            strategy_used=f"adaptive/{strategy_name}",
            reason=result.reason,
            metadata={**result.metadata, "base_strategy": strategy_name},
        )

    def add_strategy(self, name: str, strategy: SchedulingStrategy):
        """添加策略"""
        self.strategies[name] = strategy

    def add_rule(
        self,
        condition: Callable,
        strategy: str,
        priority: int = 5,
    ):
        """添加调度规则"""
        self.rules.append(
            {
                "condition": condition,
                "strategy": strategy,
                "priority": priority,
            }
        )
