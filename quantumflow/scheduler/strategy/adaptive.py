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
        """默认调度规则

        优先级从高到低：
        1. 超大模型（>70B）→ Gang（需要严格满足所有 TP 卡）
        2. 用户指定 preferred_gpu_families → Gang（拓扑敏感）
        3. 高优先级（>=8）→ Gang
        4. 长输出（>4096 tokens）→ Gang
        5. 默认 → Pack（紧凑）
        """
        return [
            # 规则1：超大模型使用 Gang 调度（要求严格满足所有 TP 卡）
            {
                "condition": lambda r, n: r.model_size > 70_000_000_000,  # >70B
                "strategy": "gang",
                "priority": 100,
            },
            # 规则2：用户指定 GPU 家族偏好 → Gang（拓扑敏感）
            {
                "condition": lambda r, n: bool(r.preferred_gpu_families),
                "strategy": "gang",
                "priority": 90,
            },
            # 规则3：高优先级请求（>=8）使用 Gang
            {
                "condition": lambda r, n: r.priority >= 8,
                "strategy": "gang",
                "priority": 80,
            },
            # 规则4：长输出（>4096 tokens）使用 Gang
            {
                "condition": lambda r, n: r.max_tokens > 4096,
                "strategy": "gang",
                "priority": 70,
            },
            # 规则5：单请求需要多张卡（TP>=2）→ Gang
            {
                "condition": lambda r, n: r.recommended_tensor_parallel >= 2,
                "strategy": "gang",
                "priority": 60,
            },
            # 规则6：默认使用 Pack 调度（紧凑打包）
            {
                "condition": lambda r, n: True,
                "strategy": "pack",
                "priority": 1,
            },
        ]

    def _evaluate_condition(
        self, condition: Callable, request: SchedulingRequest, nodes: list[NodeResource]
    ) -> bool:
        """评估条件

        Bug fix (H-C4): 原本静默吞掉所有异常，让用户自定义 rule 失效时无信号。
        现改为: 异常时 logger.exception 留痕，并 return False。
        """
        try:
            return condition(request, nodes)
        except Exception:
            # 必须留痕 — 规则失效在生产环境是难以察觉的隐性 bug
            # request 可能为 None(部分测试场景),用 getattr 防御
            request_id = getattr(request, "request_id", None) if request is not None else None
            logger.exception(
                "rule_evaluation_failed",
                request_id=request_id,
                condition_repr=repr(condition),
            )
            return False

    def _select_best_strategy(
        self, request: SchedulingRequest, available_nodes: list[NodeResource]
    ) -> str:
        """选择最佳策略

        Bug fix (C-C3 + M-O1): 原本"规则匹配但策略未注册"会静默降级到 Pack，
        导致 70B+ 大模型在没有 Gang 时被发到 Pack，可能 OOM。
        现改为: 跳到下一条 rule 之前 logger.warning; 如果高风险规则的所有策略都
        不可用，最终拒绝(返回 None)。
        """
        # 按优先级排序规则
        sorted_rules = sorted(self.rules, key=lambda r: r.get("priority", 0), reverse=True)

        matched_rule_priority: int | None = None

        for rule in sorted_rules:
            condition = rule.get("condition")
            if condition and self._evaluate_condition(condition, request, available_nodes):
                strategy_name = rule.get("strategy")
                if strategy_name in self.strategies:
                    return strategy_name
                # 规则匹配但策略未注册 — 不再静默跳过
                logger.warning(
                    "adaptive_rule_strategy_not_registered",
                    request_id=request.request_id,
                    rule_priority=rule.get("priority"),
                    rule_strategy=strategy_name,
                    available_strategies=list(self.strategies.keys()),
                )
                # 记录"已匹配到的最高规则优先级"，如果所有匹配都失败，用于拒绝
                if matched_rule_priority is None:
                    matched_rule_priority = rule.get("priority")

        # 如果有规则匹配但其策略都未注册,直接拒绝而不是降级到 pack
        if matched_rule_priority is not None:
            logger.error(
                "adaptive_no_strategy_for_matched_rule",
                request_id=request.request_id,
                matched_priority=matched_rule_priority,
                model=request.model,
                model_size=request.model_size,
            )
            return ""  # 空字符串表示"无可用策略"

        # 没有规则匹配 — 默认使用 Pack
        return "pack"

    def can_handle(self, request: SchedulingRequest, available_nodes: list[NodeResource]) -> bool:
        """检查自适应策略是否可处理

        Bug fix (M-C8): 原本永远返回 True,导致子策略拒绝时浪费 3 次重试。
        现增加: 没有 healthy 节点 / 没有可用 GPU 时直接返回 False。
        """
        if not available_nodes:
            return False
        healthy = self.filter_healthy_nodes(available_nodes)
        if not healthy:
            return False
        # 必须至少有一个节点有可用 GPU
        if not any(n.available_gpus for n in healthy):
            return False
        return True

    def select_nodes(
        self, request: SchedulingRequest, available_nodes: list[NodeResource]
    ) -> SchedulingResult:
        """使用自适应策略选择节点

        Bug fix (C-C3 后续): 当 _select_best_strategy 返回空字符串(无可用策略)
        时,直接返回失败而不是降级。
        """
        if not available_nodes:
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason="No available nodes",
            )

        # 选择最佳策略
        strategy_name = self._select_best_strategy(request, available_nodes)

        # 关键: 空字符串表示"有规则匹配但策略都未注册" — 拒绝而不是降级
        if not strategy_name:
            return SchedulingResult(
                success=False,
                strategy_used=self.name,
                reason=(
                    f"No strategy available for matched rule "
                    f"(model={request.model}, size={request.model_size})"
                ),
            )

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
