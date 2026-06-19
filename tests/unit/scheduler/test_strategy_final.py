"""Strategy final coverage tests.

Covers all remaining missing lines:
- adaptive.py line 94: _select_best_strategy final "pack" fallback
- adaptive.py lines 116-129: select_nodes strategy-not-found fallback logic
- base.py line 161: estimate_wait_time returns float("inf") on empty nodes
- gang.py line 100: continue when node has no available GPUs
- pack.py line 40: can_handle returns False when no healthy nodes
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from quantumflow.scheduler.strategy.adaptive import AdaptiveSchedulingStrategy
from quantumflow.scheduler.strategy.base import (
    GPUResource,
    NodeResource,
    SchedulingRequest,
    SchedulingResult,
    SchedulingStrategy,
    StrategyType,
)
from quantumflow.scheduler.strategy.gang import GangSchedulingStrategy
from quantumflow.scheduler.strategy.pack import PackSchedulingStrategy


# =============================================================================
# Helpers
# =============================================================================


def _make_gpu(gpu_id=0, total=24, used=8, node_id=""):
    return GPUResource(
        gpu_id=gpu_id, memory_total=total * 1024**3,
        memory_used=used * 1024**3, utilization=0.3,
        temperature=45.0, node_id=node_id,
    )


def _make_node(nid="n1", gpus=None, load=0.2, status="healthy"):
    if gpus is None:
        gpus = [_make_gpu(0, node_id=nid)]
    return NodeResource(
        node_id=nid, hostname=f"s-{nid}", ip=f"10.0.0.{nid}",
        status=status, gpu_count=len(gpus), gpus=gpus,
        cpu_count=4, memory_total=32 * 1024**3, memory_available=16 * 1024**3,
        disk_total=100 * 1024**3, disk_available=50 * 1024**3, load=load,
    )


# =============================================================================
# adaptive.py line 94: _select_best_strategy returns "pack" as final fallback
# =============================================================================


class TestAdaptiveSelectBestStrategyFallback:
    """Tests for _select_best_strategy fallback path (line 94)."""

    def test_select_best_strategy_fallback_when_no_matching_rule(self):
        """When no rule condition matches, returns 'pack' (line 94)."""
        # Remove the last "catch-all" default rule
        strategy = AdaptiveSchedulingStrategy(
            strategies={"pack": PackSchedulingStrategy()},
            rules=[
                {
                    "condition": lambda r, n: False,
                    "strategy": "nonexistent",
                    "priority": 10,
                },
            ],
        )

        request = SchedulingRequest(request_id="r1", model="m")
        nodes = [_make_node("n1")]

        result_name = strategy._select_best_strategy(request, nodes)

        assert result_name == "pack", (
            f"Expected 'pack' fallback, got '{result_name}'"
        )

    def test_select_best_strategy_fallback_no_rules(self):
        """When there are no rules at all, falls back to 'pack' (line 94)."""
        strategy = AdaptiveSchedulingStrategy(
            strategies={"pack": PackSchedulingStrategy()},
            rules=[],
        )

        request = SchedulingRequest(request_id="r1", model="m")
        nodes = [_make_node("n1")]

        result_name = strategy._select_best_strategy(request, nodes)

        assert result_name == "pack", (
            f"Expected 'pack' fallback, got '{result_name}'"
        )

    def test_select_best_strategy_rule_matches_but_strategy_not_registered(self):
        """Rule condition matches but strategy not in self.strategies, logs warning and returns "".

        Bug fix (C-C3): 原本静默降级到 "pack" 会让 70B+ 模型在没有 Gang 时被发到
        Pack,可能 OOM。修复后:_select_best_strategy 返回空字符串,select_nodes
        据此拒绝请求而不是降级。
        """
        strategy = AdaptiveSchedulingStrategy(
            strategies={"pack": PackSchedulingStrategy()},
            rules=[
                {
                    "condition": lambda r, n: True,
                    "strategy": "gang",  # gang is NOT in strategies
                    "priority": 100,
                },
            ],
        )

        request = SchedulingRequest(request_id="r1", model="m")
        nodes = [_make_node("n1")]

        result_name = strategy._select_best_strategy(request, nodes)

        # 关键: 不再静默降级到 "pack",而是返回空字符串表示"无可用策略"
        assert result_name == "", (
            f"Expected '' (拒绝) when rule's strategy not registered, got '{result_name!r}' — "
            f"this would have silently downgraded large-model requests to pack and caused OOM"
        )


# =============================================================================
# adaptive.py lines 116-129: select_nodes strategy fallback logic
# =============================================================================


class TestAdaptiveSelectNodesFallback:
    """Tests for select_nodes fallback when strategy is not in self.strategies."""

    def test_select_nodes_strategy_not_found_rejects(self):
        """When selected strategy is not in self.strategies, REJECTS instead of falling back.

        Bug fix (C-C3): 原本静默降级到 pack 会让 70B+ 模型在没有 Gang 时被发到
        Pack,可能 OOM。修复后:有规则匹配但策略未注册 → 拒绝请求。
        """
        gang = GangSchedulingStrategy()
        pack = PackSchedulingStrategy()

        strategy = AdaptiveSchedulingStrategy(
            strategies={"pack": pack, "gang": gang},
            rules=[
                {
                    # Will match, returning "custom", which is NOT in strategies
                    "condition": lambda r, n: True,
                    "strategy": "custom",
                    "priority": 999,
                },
            ],
        )

        request = SchedulingRequest(
            request_id="r1", model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )
        nodes = [_make_node("n1")]

        result = strategy.select_nodes(request, nodes)

        # 关键: 不再静默降级,直接失败
        assert result.success is False, (
            f"Expected failure when matched rule's strategy not registered, "
            f"got success with strategy_used={result.strategy_used!r}"
        )
        assert "No strategy available" in result.reason, (
            f"Expected reason to mention 'No strategy available', got {result.reason!r}"
        )

    def test_select_nodes_no_strategy_at_all(self):
        """When no strategy is available at all (not even pack), returns failure."""
        strategy = AdaptiveSchedulingStrategy(
            strategies={},  # Empty - no fallback available
            rules=[
                {
                    "condition": lambda r, n: True,
                    "strategy": "custom",
                    "priority": 999,
                },
            ],
        )

        request = SchedulingRequest(
            request_id="r1", model="test",
            model_config={"tensor_parallel": 1},
        )
        nodes = [_make_node("n1")]

        result = strategy.select_nodes(request, nodes)

        # _select_best_strategy returns "" (C-C3 fix), select_nodes returns failure
        assert result.success is False
        assert result.strategy_used == "adaptive"
        assert "No strategy available" in result.reason

    def test_select_nodes_strategy_missing_rejects(self):
        """When matched rule's strategy is not registered, REJECTS (no pack fallback).

        Bug fix (C-C3): 原本静默降级到 pack,会导致大模型 OOM。
        修复后:有规则匹配但策略未注册 → 拒绝请求。
        """
        pack = PackSchedulingStrategy()

        strategy = AdaptiveSchedulingStrategy(
            strategies={"pack": pack},
            rules=[
                {
                    "condition": lambda r, n: True,
                    "strategy": "nonexistent",  # 不在 strategies 中
                    "priority": 999,
                },
            ],
        )

        request = SchedulingRequest(
            request_id="r1", model="test",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )
        nodes = [_make_node("n1")]

        result = strategy.select_nodes(request, nodes)

        # 关键: 不再静默降级,直接失败
        assert result.success is False, (
            f"Expected failure when matched rule's strategy not registered, "
            f"got success with strategy_used={result.strategy_used!r}"
        )
        assert "No strategy available" in result.reason


# =============================================================================
# base.py line 161: estimate_wait_time with empty nodes
# =============================================================================


class TestEstimateWaitTimeEmptyNodes:
    """Tests for estimate_wait_time returning inf on empty nodes (line 161)."""

    def test_estimate_wait_time_empty_nodes_returns_inf(self):
        """estimate_wait_time returns float('inf') when nodes list is empty."""
        strategy = GangSchedulingStrategy()
        request = SchedulingRequest(request_id="r1", model="m", max_tokens=2048)

        wait_time = strategy.estimate_wait_time(request, [])

        assert wait_time == float("inf"), (
            f"Expected float('inf'), got {wait_time}"
        )

    def test_estimate_wait_time_empty_nodes_adaptive(self):
        """Adaptive strategy's base estimate_wait_time also returns inf on empty."""
        strategy = AdaptiveSchedulingStrategy(
            strategies={"pack": PackSchedulingStrategy()}
        )
        request = SchedulingRequest(request_id="r1", model="m", max_tokens=2048)

        wait_time = strategy.estimate_wait_time(request, [])

        assert wait_time == float("inf")

    def test_estimate_wait_time_empty_nodes_pack(self):
        """Pack strategy's base estimate_wait_time returns inf on empty nodes."""
        strategy = PackSchedulingStrategy()
        request = SchedulingRequest(request_id="r1", model="m", max_tokens=2048)

        wait_time = strategy.estimate_wait_time(request, [])

        assert wait_time == float("inf")

    def test_estimate_wait_time_with_nodes_returns_finite(self):
        """When nodes are provided, estimate_wait_time returns a finite number."""
        strategy = GangSchedulingStrategy()
        request = SchedulingRequest(request_id="r1", model="m", max_tokens=2048)
        nodes = [_make_node("n1", load=0.2)]

        wait_time = strategy.estimate_wait_time(request, nodes)

        assert isinstance(wait_time, float)
        assert wait_time > 0
        assert wait_time != float("inf")


# =============================================================================
# gang.py line 100: continue when node has no available GPUs
# =============================================================================


class TestGangContinueOnNoAvailableGpu:
    """Tests for gang select_nodes 'continue' when node has no available GPUs (line 100)."""

    def test_gang_continues_when_node_has_no_available_gpus(self):
        """A healthy node with GPUs that are all full should be skipped (continue)."""
        strategy = GangSchedulingStrategy()

        # Node 1: healthy, has 4 GPUs, but all are >90% full (not available)
        full_gpus = [
            GPUResource(
                gpu_id=i, memory_total=24 * 1024**3,
                memory_used=23 * 1024**3,  # ~96% used, memory_free_percent ~0.04 < 0.1
                utilization=0.96, temperature=70.0, node_id="n1",
            )
            for i in range(4)
        ]
        node_full = NodeResource(
            node_id="n1", hostname="s1", ip="10.0.0.1", status="healthy",
            gpu_count=4, gpus=full_gpus,
            cpu_count=8, memory_total=64 * 1024**3, memory_available=32 * 1024**3,
            disk_total=500 * 1024**3, disk_available=250 * 1024**3, load=0.1,
        )

        # Node 2: healthy, has 4 GPUs, all available
        avail_gpus = [_make_gpu(i, node_id="n2") for i in range(4)]
        node_avail = NodeResource(
            node_id="n2", hostname="s2", ip="10.0.0.2", status="healthy",
            gpu_count=4, gpus=avail_gpus,
            cpu_count=8, memory_total=64 * 1024**3, memory_available=32 * 1024**3,
            disk_total=500 * 1024**3, disk_available=250 * 1024**3, load=0.2,
        )

        request = SchedulingRequest(
            request_id="r1", model="big-model",
            model_config={
                "parameter_count": 50_000_000_000,
                "tensor_parallel": 4,
                "estimated_memory": 10 * 1024**3,  # per-GPU, fits in 14GB available
            },
        )

        result = strategy.select_nodes(request, [node_full, node_avail])

        # Should skip node_full (no available GPUs -> continue), use node_avail
        assert result.success is True
        assert result.assigned_nodes == ["n2"], (
            f"Expected only 'n2' (has GPUs), got {result.assigned_nodes}"
        )
        assert result.metadata["total_gpus"] == 4

    def test_gang_aggregation_skip_node_with_no_available_gpus(self):
        """When aggregating across nodes, full-GPU nodes are skipped (continue)."""
        strategy = GangSchedulingStrategy()

        # Node 1: all GPUs full
        full_gpus = [
            GPUResource(gpu_id=i, memory_total=24 * 1024**3,
                        memory_used=23 * 1024**3, utilization=0.96,
                        temperature=70.0, node_id="n1")
            for i in range(2)
        ]
        node_full = NodeResource(
            node_id="n1", hostname="s1", ip="10.0.0.1", status="healthy",
            gpu_count=2, gpus=full_gpus,
            cpu_count=4, memory_total=32 * 1024**3, memory_available=16 * 1024**3,
            disk_total=100 * 1024**3, disk_available=50 * 1024**3, load=0.1,
        )

        # Node 2: 2 available GPUs
        node_half = _make_node("n2", [_make_gpu(0, node_id="n2"), _make_gpu(1, node_id="n2")])

        # Node 3: 2 available GPUs
        node_avail = _make_node("n3", [_make_gpu(0, node_id="n3"), _make_gpu(1, node_id="n3")])

        request = SchedulingRequest(
            request_id="r1", model="big-model",
            model_config={
                "parameter_count": 50_000_000_000,
                "tensor_parallel": 3,
                "estimated_memory": 10 * 1024**3,  # per-GPU, fits in 14GB available
            },
        )

        result = strategy.select_nodes(request, [node_full, node_half, node_avail])

        # node_full should be skipped (continue on empty available_gpus)
        # node_half gives 2 GPUs, need 1 more -> node_avail gives 1
        assert result.success is True
        assert "n1" not in result.assigned_nodes, "n1 should be skipped (no available GPUs)"
        assert "n2" in result.assigned_nodes


# =============================================================================
# pack.py line 40: can_handle returns False when no healthy nodes
# =============================================================================


class TestPackCanHandleNoHealthyNodes:
    """Tests for pack can_handle returning False (line 40)."""

    def test_pack_can_handle_returns_false_when_all_unhealthy(self):
        """Pack can_handle returns False when filter_healthy_nodes returns empty."""
        strategy = PackSchedulingStrategy()

        unhealthy_nodes = [
            _make_node("n1", status="unhealthy"),
            _make_node("n2", status="draining"),
        ]

        request = SchedulingRequest(
            request_id="r1", model="m",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.can_handle(request, unhealthy_nodes)

        assert result is False, (
            f"Pack can_handle should return False when no healthy nodes, got {result}"
        )

    def test_pack_can_handle_returns_false_when_empty_nodes(self):
        """Pack can_handle returns False when node list is empty."""
        strategy = PackSchedulingStrategy()

        request = SchedulingRequest(
            request_id="r1", model="m",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.can_handle(request, [])

        assert result is False

    def test_pack_can_handle_returns_true_when_at_least_one_healthy(self):
        """Pack can_handle returns True when at least one node is healthy."""
        strategy = PackSchedulingStrategy()

        nodes = [
            _make_node("n1", status="unhealthy"),
            _make_node("n2", status="healthy"),
        ]

        request = SchedulingRequest(
            request_id="r1", model="m",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.can_handle(request, nodes)

        assert result is True
