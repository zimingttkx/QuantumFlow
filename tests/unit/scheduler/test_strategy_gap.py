"""Strategy gap coverage tests

Tests covering gaps found in test_strategy.py:
1. Gang select_nodes: cross-node GPU aggregation
2. Gang select_nodes: assigned_gpus dict verification
3. Gang select_nodes: result metadata verification
4. Gang can_handle: max_tokens / priority fallback when model_size=0
5. Gang select_nodes: mixed healthy/unhealthy nodes
6. Gang select_nodes: exact GPU count match
7. Pack select_nodes: precise lowest-load node verification
8. Pack select_nodes: no available GPUs on selected node
9. Pack can_handle: model_size=0 default
10. Adaptive: max_tokens > 4096 rule
11. Adaptive: rule priority ordering
12. Adaptive: strategy_used format and metadata
13. Adaptive: _evaluate_condition exception handling
14. Adaptive: empty nodes list
15. Adaptive: add_strategy + add_rule integration
16. NodeResource: can_fit_model with estimated_memory=0
17. SchedulingRequest: deadline, session_id, tags fields
"""

from datetime import datetime
from unittest.mock import Mock, patch

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


# ==================== Helper factories ====================


def make_gpu(gpu_id, memory_total=24, memory_used=10,
             utilization=0.5, temperature=45.0, node_id=""):
    return GPUResource(
        gpu_id=gpu_id,
        memory_total=memory_total * 1024**3,
        memory_used=memory_used * 1024**3,
        utilization=utilization,
        temperature=temperature,
        node_id=node_id,
    )


def make_node(node_id, gpus, load=0.3, status="healthy"):
    return NodeResource(
        node_id=node_id,
        hostname=f"server-{node_id}",
        ip=f"10.0.0.{node_id}",
        status=status,
        gpu_count=len(gpus),
        gpus=gpus,
        cpu_count=32,
        memory_total=128 * 1024**3,
        memory_available=64 * 1024**3,
        disk_total=2 * 1024**4,
        disk_available=1 * 1024**4,
        load=load,
    )


# ==================== Gang select_nodes: cross-node aggregation ====================


class TestGangCrossNodeAggregation:
    """Tests for Gang strategy's cross-node GPU aggregation."""

    @pytest.fixture
    def strategy(self):
        return GangSchedulingStrategy()

    def test_aggregates_gpus_across_nodes(self, strategy):
        """When no single node has enough GPUs, Gang aggregates across multiple nodes."""
        # 2 nodes, each with 2 available GPUs -- need 4 total
        nodes = [
            make_node("n1", [make_gpu(0, memory_total=24, memory_used=8, node_id="n1"),
                             make_gpu(1, memory_total=24, memory_used=8, node_id="n1")], load=0.3),
            make_node("n2", [make_gpu(0, memory_total=24, memory_used=8, node_id="n2"),
                             make_gpu(1, memory_total=24, memory_used=8, node_id="n2")], load=0.2),
        ]

        request = SchedulingRequest(
            request_id="cross-node",
            model="big-model",
            model_config={"parameter_count": 50_000_000_000, "tensor_parallel": 4},
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert result.strategy_used == "gang"
        assert len(result.assigned_nodes) == 2, "Should use GPUs from 2 nodes"
        assert set(result.assigned_nodes) == {"n1", "n2"}

    def test_cross_node_gpu_allocation_is_correct(self, strategy):
        """Verifies that assigned_gpus dict has correct GPU IDs per node."""
        # 3 nodes, 2 GPUs each -- need 5 GPUs
        nodes = [
            make_node("n1", [make_gpu(0, memory_total=24, memory_used=8, node_id="n1"),
                             make_gpu(1, memory_total=24, memory_used=8, node_id="n1")], load=0.3),
            make_node("n2", [make_gpu(0, memory_total=24, memory_used=8, node_id="n2"),
                             make_gpu(1, memory_total=24, memory_used=8, node_id="n2")], load=0.2),
            make_node("n3", [make_gpu(0, memory_total=24, memory_used=8, node_id="n3"),
                             make_gpu(1, memory_total=24, memory_used=8, node_id="n3")], load=0.1),
        ]

        request = SchedulingRequest(
            request_id="five-gpus",
            model="huge",
            model_config={"parameter_count": 100_000_000_000, "tensor_parallel": 5},
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert len(result.assigned_nodes) == 3
        # Gang sorts by key=(available_gpus, -load), reverse=True
        # All tied on gpu_count=2, then by -load:
        #   n3: (2, -0.1) > n2: (2, -0.2) > n1: (2, -0.3)
        # So order: n3, n2, n1 (highest -load first = lowest actual load first)
        # n3 gives 2, need 3 more. n2 gives 2, need 1 more. n1 gives 1.
        assigned = result.assigned_gpus
        assert len(assigned) == 3
        assert sum(len(gpus) for gpus in assigned.values()) == 5
        # n3 (first in sort) should get its 2 GPUs
        assert len(assigned["n3"]) == 2
        assert assigned["n3"] == [0, 1]
        # n2 (second in sort) should get its 2 GPUs
        assert len(assigned["n2"]) == 2
        # n1 (third in sort) should get the remaining 1 GPU
        assert len(assigned["n1"]) == 1

    def test_select_nodes_metadata_correct(self, strategy):
        """Verifies metadata fields in the scheduling result."""
        nodes = [
            make_node("n1", [make_gpu(i, node_id="n1") for i in range(4)], load=0.3),
        ]

        request = SchedulingRequest(
            request_id="meta-test",
            model="big-model",
            model_config={
                "parameter_count": 50_000_000_000,
                "tensor_parallel": 4,
                "pipeline_parallel": 2,
            },
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert result.metadata["total_gpus"] == 4
        assert result.metadata["tensor_parallel"] == 4
        assert result.metadata["pipeline_parallel"] == 2

    def test_result_has_estimated_wait_time_and_latency(self, strategy):
        """Success result must include estimated wait time and latency."""
        nodes = [
            make_node("n1", [make_gpu(i, node_id="n1") for i in range(4)], load=0.3),
        ]

        request = SchedulingRequest(
            request_id="timing-test",
            model="big-model",
            model_config={"parameter_count": 50_000_000_000, "tensor_parallel": 4},
            max_tokens=2048,
            prompt_length=100,
        )

        result = strategy.select_nodes(request, nodes)

        assert result.estimated_wait_time > 0
        assert result.estimated_latency > 0
        assert isinstance(result.estimated_wait_time, float)
        assert isinstance(result.estimated_latency, float)

    def test_mixed_healthy_unhealthy_nodes(self, strategy):
        """Gang should filter out unhealthy nodes and use only healthy ones."""
        nodes = [
            make_node("n1", [make_gpu(i, node_id="n1") for i in range(4)],
                      load=0.3, status="unhealthy"),
            make_node("n2", [make_gpu(i, node_id="n2") for i in range(4)],
                      load=0.2, status="healthy"),
        ]

        request = SchedulingRequest(
            request_id="mixed-health",
            model="big-model",
            model_config={"parameter_count": 50_000_000_000, "tensor_parallel": 4},
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        # n1 is unhealthy and should be filtered out; only n2 used
        assert result.assigned_nodes == ["n2"]

    def test_node_with_no_available_gpus_skipped_with_continue(self, strategy):
        """When a node has no available GPUs (all used), Gang skips it with continue (line 100)"""
        # Setup nodes so we MUST reach a node with 0 available GPUs:
        # - n3: 2 GPUs (sorted first due to higher GPU count)
        # - n1: 1 GPU (sorted second)
        # - n2: 0 GPUs (sorted last, we need to reach it)
        # We need 4 GPUs: get 2 from n3, 1 from n1, then reach n2 (0 GPUs) and continue
        full_gpu = GPUResource(
            gpu_id=0,
            memory_total=24 * 1024**3,
            memory_used=23 * 1024**3,  # <10% free => not available
            utilization=0.98,  # >95% utilized => not available
            temperature=80.0,
            node_id="n2",
        )
        n2 = NodeResource(
            node_id="n2", hostname="s2", ip="10.0.0.2", status="healthy",
            gpu_count=1, gpus=[full_gpu],
            cpu_count=8, memory_total=64 * 1024**3, memory_available=32 * 1024**3,
            disk_total=500 * 1024**3, disk_available=250 * 1024**3, load=0.3,
        )
        # n1 has 1 available GPU (lower load than n3)
        n1 = make_node("n1", [make_gpu(0, memory_total=24, memory_used=8, node_id="n1")], load=0.1)
        # n3 has 2 available GPUs but slightly higher load
        n3 = make_node("n3", [make_gpu(i, memory_total=24, memory_used=8, node_id="n3") for i in range(2)], load=0.2)
        nodes = [n1, n2, n3]

        request = SchedulingRequest(
            request_id="skip-no-gpu",
            model="big-model",
            model_config={"parameter_count": 50_000_000_000, "tensor_parallel": 4},
        )

        result = strategy.select_nodes(request, nodes)

        # Sorted: n3 (2 GPUs), n1 (1 GPU), n2 (0 GPUs)
        # We get 2 from n3, 1 from n1, still need 1 more
        # Then we reach n2 with 0 available -> continue at line 100
        # Total available = 3, needed = 4, so scheduling fails
        assert result.success is False
        assert "Insufficient GPUs" in result.reason
        # n2 should not be in assigned_nodes since it has no available GPUs
        assert "n2" not in result.assigned_nodes


# ==================== Gang can_handle: fallback conditions ====================


class TestGangCanHandleFallbacks:
    """Tests for Gang can_handle fallback logic when model_size is 0."""

    @pytest.fixture
    def strategy(self):
        return GangSchedulingStrategy()

    @pytest.fixture
    def nodes(self):
        return [
            make_node("n1", [make_gpu(i, node_id="n1") for i in range(4)]),
        ]

    def test_can_handle_by_max_tokens_fallback(self, strategy, nodes):
        """When model_size is 0, max_tokens > 2048 makes can_handle return True."""
        request = SchedulingRequest(
            request_id="token-fallback",
            model="unknown-model",
            model_config={"tensor_parallel": 1},
            max_tokens=4096,
        )
        # model_size = 0 (no parameter_count)
        assert strategy.can_handle(request, nodes) is True

    def test_can_handle_by_priority_fallback(self, strategy, nodes):
        """When model_size is 0, priority >= 8 makes can_handle return True."""
        request = SchedulingRequest(
            request_id="priority-fallback",
            model="unknown-model",
            model_config={"tensor_parallel": 1},
            priority=9,
        )
        # model_size = 0, max_tokens = 2048 (default), priority = 9
        # check: max_tokens > 2048? 2048 > 2048 is False
        # check: priority >= 8? 9 >= 8 is True
        assert strategy.can_handle(request, nodes) is True

    def test_can_handle_false_when_no_fallback_matches(self, strategy, nodes):
        """When model_size is 0, max_tokens <= 2048 and priority < 8, returns False."""
        request = SchedulingRequest(
            request_id="no-fallback",
            model="unknown-model",
            model_config={"tensor_parallel": 1},
            max_tokens=512,
            priority=5,
        )
        assert strategy.can_handle(request, nodes) is False


# ==================== Pack select_nodes: precise verification ====================


class TestPackSelectNodesPrecise:
    """Tests for Pack strategy's node selection with precise assertions."""

    @pytest.fixture
    def strategy(self):
        return PackSchedulingStrategy()

    def test_selects_absolute_lowest_load_node(self, strategy):
        """Pack must select the node with the lowest load value."""
        nodes = [
            make_node("n-high", [make_gpu(0, node_id="n-high")], load=0.9),
            make_node("n-mid", [make_gpu(0, node_id="n-mid")], load=0.5),
            make_node("n-low", [make_gpu(0, node_id="n-low")], load=0.05),
        ]

        request = SchedulingRequest(
            request_id="load-test",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert result.assigned_nodes == ["n-low"], (
            f"Should select lowest-load node (n-low, load=0.05), "
            f"got {result.assigned_nodes}"
        )

    def test_selects_any_node_when_all_equal_load(self, strategy):
        """When all nodes have equal load, any healthy node is acceptable."""
        nodes = [
            make_node("n1", [make_gpu(0, node_id="n1")], load=0.5),
            make_node("n2", [make_gpu(0, node_id="n2")], load=0.5),
        ]

        request = SchedulingRequest(
            request_id="equal-load",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert len(result.assigned_nodes) == 1
        assert result.assigned_nodes[0] in ["n1", "n2"]

    def test_no_available_gpus_on_lowest_load_node(self, strategy):
        """When the lowest-load node has no available GPUs, it returns failure."""
        # GPU with memory_used = 23 out of 24 => memory_free_percent = ~0.042 < 0.1
        # So it is NOT an available GPU
        full_gpu = GPUResource(
            gpu_id=0,
            memory_total=24 * 1024**3,
            memory_used=23 * 1024**3,  # >90% used, <10% free
            utilization=1.0,
            temperature=80.0,
            node_id="n1",
        )
        nodes = [
            NodeResource(
                node_id="n1", hostname="s1", ip="10.0.0.1", status="healthy",
                gpu_count=1, gpus=[full_gpu],
                cpu_count=8, memory_total=64 * 1024**3, memory_available=32 * 1024**3,
                disk_total=500 * 1024**3, disk_available=250 * 1024**3, load=0.1,
            ),
        ]

        request = SchedulingRequest(
            request_id="no-gpu",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is False
        assert "has no available gpus" in result.reason.lower()

    def test_all_nodes_unhealthy_returns_failure(self, strategy):
        """When all nodes are unhealthy, returns failure."""
        nodes = [
            make_node("n1", [make_gpu(0, node_id="n1")], load=0.3, status="unhealthy"),
            make_node("n2", [make_gpu(0, node_id="n2")], load=0.2, status="unhealthy"),
        ]

        request = SchedulingRequest(
            request_id="all-bad",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is False
        assert "No healthy nodes" in result.reason

    def test_pack_can_handle_zero_model_size(self, strategy):
        """Pack can_handle returns True when model_size is 0 (default behavior)."""
        nodes = [make_node("n1", [make_gpu(0, node_id="n1")])]

        request = SchedulingRequest(
            request_id="zero-size",
            model="unknown",
            model_config={"tensor_parallel": 1},
        )
        # model_size = 0

        assert strategy.can_handle(request, nodes) is True

    def test_pack_result_has_wait_time_and_latency(self, strategy):
        """Pack success result must include wait time and latency."""
        nodes = [make_node("n1", [make_gpu(0, node_id="n1")], load=0.3)]

        request = SchedulingRequest(
            request_id="timing",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
            max_tokens=512,
            prompt_length=50,
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert result.estimated_wait_time > 0
        assert result.estimated_latency > 0

    def test_pack_result_metadata_contains_expected_keys(self, strategy):
        """Pack result metadata contains total_gpus, tensor_parallel, and load."""
        nodes = [make_node("n1", [make_gpu(0, node_id="n1")], load=0.42)]

        request = SchedulingRequest(
            request_id="meta",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert result.metadata["total_gpus"] == 1
        assert result.metadata["tensor_parallel"] == 1
        assert result.metadata["load"] == 0.42


# ==================== Adaptive: rule ordering and format ====================


class TestAdaptiveRuleOrdering:
    """Tests for Adaptive strategy's rule evaluation and ordering."""

    @pytest.fixture
    def strategy(self):
        gang = GangSchedulingStrategy()
        pack = PackSchedulingStrategy()
        return AdaptiveSchedulingStrategy(strategies={"gang": gang, "pack": pack})

    @pytest.fixture
    def nodes(self):
        return [
            make_node("n1", [make_gpu(i, node_id="n1") for i in range(4)]),
        ]

    def test_max_tokens_rule_triggers_gang(self, strategy, nodes):
        """When max_tokens > 4096, adaptive should select gang."""
        request = SchedulingRequest(
            request_id="long-output",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 4},
            max_tokens=8192,
            priority=3,
        )
        # Rule1 (>70B): not met (1B)
        # Rule2 (priority>=8): not met (3)
        # Rule3 (max_tokens>4096): met (8192) -> gang
        # Rule4 (default): pack

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert "gang" in result.strategy_used

    def test_higher_priority_rule_overrides_lower(self, strategy, nodes):
        """Rule with higher priority overrides lower-priority matching rule."""
        # Create a request that matches BOTH rule2 (high priority) and rule3 (long output)
        # Rule2 has priority 9, Rule3 has priority 7 -> Rule2 should win
        request = SchedulingRequest(
            request_id="rule-priority",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 2},
            priority=9,    # Matches rule2 (priority >= 8) at pri 9
            max_tokens=8192,  # Also matches rule3 (max_tokens > 4096) at pri 7
        )
        # Rule2 (pri=9) > Rule3 (pri=7), so gang via rule2 wins

        result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert "gang" in result.strategy_used

    def test_strategy_used_format_contains_adaptive_prefix(self, strategy, nodes):
        """The strategy_used field uses 'adaptive/{strategy_name}' format."""
        request = SchedulingRequest(
            request_id="format-test",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
            priority=3,
        )

        result = strategy.select_nodes(request, nodes)

        assert result.strategy_used == "adaptive/pack", (
            f"Expected 'adaptive/pack', got '{result.strategy_used}'"
        )

    def test_metadata_includes_base_strategy(self, strategy, nodes):
        """Result metadata contains 'base_strategy' key."""
        request = SchedulingRequest(
            request_id="meta-base",
            model="small-model",
            model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
        )

        result = strategy.select_nodes(request, nodes)

        assert "base_strategy" in result.metadata
        assert result.metadata["base_strategy"] == "pack"

    def test_empty_nodes_returns_failure(self, strategy):
        """Adaptive with empty nodes returns failure immediately."""
        request = SchedulingRequest(
            request_id="no-nodes",
            model="test",
            model_config={"tensor_parallel": 1},
        )

        result = strategy.select_nodes(request, [])

        assert result.success is False
        assert result.reason == "No available nodes"
        assert result.strategy_used == "adaptive"

    def test_evaluate_condition_catches_exceptions(self, strategy):
        """_evaluate_condition returns False when the condition function raises."""
        def crashing_condition(req, nodes):
            raise ValueError("boom")

        result = strategy._evaluate_condition(crashing_condition, None, [])
        assert result is False

    def test_add_strategy_and_rule_integration(self, strategy, nodes):
        """Adding a custom strategy and rule makes it selectable."""
        custom_strategy = Mock(spec=SchedulingStrategy)
        custom_strategy.name = "custom"
        custom_strategy.strategy_type = StrategyType.PACK
        custom_strategy.select_nodes = Mock(return_value=SchedulingResult(
            success=True,
            assigned_nodes=["n1"],
            assigned_gpus={"n1": [0]},
            strategy_used="custom",
        ))

        strategy.add_strategy("custom", custom_strategy)
        strategy.add_rule(
            condition=lambda r, n: r.priority == 7,
            strategy="custom",
            priority=50,  # Higher than default rules
        )

        request = SchedulingRequest(
            request_id="custom-test",
            model="test",
            model_config={"parameter_count": 1_000_000_000},
            priority=7,
        )

        result = strategy.select_nodes(request, nodes)

        assert result.strategy_used == "adaptive/custom"
        custom_strategy.select_nodes.assert_called_once()

    def test_re_evaluate_condition_returns_false_on_exception(self, strategy):
        """_evaluate_condition should return False when condition raises, not propagate."""
        assert strategy._evaluate_condition(
            lambda r, n: (_ for _ in ()).throw(KeyError("bad")),
            SchedulingRequest(request_id="x", model="x"),
            [],
        ) is False

    def test_select_best_strategy_returns_nonexistent_strategy_falls_back_to_pack(self, strategy, nodes):
        """When _select_best_strategy returns a nonexistent strategy, fallback to pack (line 129)"""
        with patch.object(strategy, "_select_best_strategy", return_value="nonexistent_strategy"):
            request = SchedulingRequest(
                request_id="fallback-test",
                model="small-model",
                model_config={"parameter_count": 1_000_000_000, "tensor_parallel": 1},
            )
            result = strategy.select_nodes(request, nodes)

        assert result.success is True
        assert result.strategy_used == "adaptive/pack"


# ==================== NodeResource: can_fit_model edge ====================


class TestNodeResourceCanFitModel:
    """Tests for NodeResource.can_fit_model edge cases."""

    def test_can_fit_model_with_zero_estimated_memory(self, strategy=None):
        """When estimated_memory=0, the model should fit if enough GPUs are available."""
        node = NodeResource(
            node_id="n1", hostname="s1", ip="1.2.3.4", status="healthy",
            gpu_count=2,
            gpus=[
                GPUResource(gpu_id=0, memory_total=24 * 1024**3, memory_used=10 * 1024**3,
                            utilization=0.5, temperature=45.0),
                GPUResource(gpu_id=1, memory_total=24 * 1024**3, memory_used=10 * 1024**3,
                            utilization=0.5, temperature=45.0),
            ],
            cpu_count=8, memory_total=64 * 1024**3, memory_available=32 * 1024**3,
            disk_total=500 * 1024**3, disk_available=250 * 1024**3, load=0.1,
        )

        result = node.can_fit_model({"estimated_memory": 0, "tensor_parallel": 1})
        assert result is True

    def test_can_fit_model_with_exact_memory_match(self):
        """When available memory exactly equals required memory, the model should fit."""
        # GPU: 24 total - 10 used = 14 available per GPU
        # 2 GPUs = 28 available
        # Required: 28 * 1024**3
        node = NodeResource(
            node_id="n1", hostname="s1", ip="1.2.3.4", status="healthy",
            gpu_count=2,
            gpus=[
                GPUResource(gpu_id=0, memory_total=24 * 1024**3, memory_used=10 * 1024**3,
                            utilization=0.5, temperature=45.0),
                GPUResource(gpu_id=1, memory_total=24 * 1024**3, memory_used=10 * 1024**3,
                            utilization=0.5, temperature=45.0),
            ],
            cpu_count=8, memory_total=64 * 1024**3, memory_available=32 * 1024**3,
            disk_total=500 * 1024**3, disk_available=250 * 1024**3, load=0.1,
        )

        result = node.can_fit_model({"estimated_memory": 28 * 1024**3, "tensor_parallel": 2})
        assert result is True

    def test_can_fit_model_with_memory_just_insufficient(self):
        """When available memory is just below required, it should not fit."""
        node = NodeResource(
            node_id="n1", hostname="s1", ip="1.2.3.4", status="healthy",
            gpu_count=2,
            gpus=[
                GPUResource(gpu_id=0, memory_total=24 * 1024**3, memory_used=10 * 1024**3,
                            utilization=0.5, temperature=45.0),
                GPUResource(gpu_id=1, memory_total=24 * 1024**3, memory_used=10 * 1024**3,
                            utilization=0.5, temperature=45.0),
            ],
            cpu_count=8, memory_total=64 * 1024**3, memory_available=32 * 1024**3,
            disk_total=500 * 1024**3, disk_available=250 * 1024**3, load=0.1,
        )

        result = node.can_fit_model({"estimated_memory": 29 * 1024**3, "tensor_parallel": 2})
        assert result is False


# ==================== SchedulingRequest: additional fields ====================


class TestSchedulingRequestFields:
    """Tests for SchedulingRequest field defaults and properties."""

    def test_session_id_defaults_to_none(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert request.session_id is None

    def test_tags_defaults_to_empty_dict(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert request.tags == {}

    def test_deadline_defaults_to_none(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert request.deadline is None

    def test_retry_count_defaults_to_zero(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert request.retry_count == 0

    def test_created_at_is_datetime(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert isinstance(request.created_at, datetime)

    def test_model_config_defaults_to_empty_dict(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert request.model_config == {}

    def test_prompt_defaults_to_empty_string(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert request.prompt == ""

    def test_max_tokens_default(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert request.max_tokens == 2048

    def test_priority_default(self):
        request = SchedulingRequest(request_id="r1", model="m")
        assert request.priority == 5


# ==================== SchedulingResult: defaults ====================


class TestSchedulingResultDefaults:
    """Tests for SchedulingResult default values."""

    def test_default_assigned_nodes_is_empty(self):
        result = SchedulingResult(success=True)
        assert result.assigned_nodes == []

    def test_default_assigned_gpus_is_empty_dict(self):
        result = SchedulingResult(success=True)
        assert result.assigned_gpus == {}

    def test_default_metadata_is_empty_dict(self):
        result = SchedulingResult(success=True)
        assert result.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
