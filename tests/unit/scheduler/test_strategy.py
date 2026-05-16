"""调度策略单元测试"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from quantumflow.scheduler.strategy.base import (
    SchedulingStrategy,
    SchedulingRequest,
    SchedulingResult,
    NodeResource,
    GPUResource,
    StrategyType,
)
from quantumflow.scheduler.strategy.gang import GangSchedulingStrategy
from quantumflow.scheduler.strategy.pack import PackSchedulingStrategy
from quantumflow.scheduler.strategy.adaptive import AdaptiveSchedulingStrategy


class TestGPUResource:
    """GPU资源测试"""

    def test_memory_available(self):
        """测试可用显存计算"""
        gpu = GPUResource(
            gpu_id=0,
            memory_total=24 * 1024**3,
            memory_used=10 * 1024**3,
            utilization=0.5,
            temperature=45.0,
        )

        expected_available = 14 * 1024**3
        assert gpu.memory_available == expected_available

    def test_memory_free_percent(self):
        """测试可用显存百分比"""
        gpu = GPUResource(
            gpu_id=0,
            memory_total=24 * 1024**3,
            memory_used=12 * 1024**3,
            utilization=0.5,
            temperature=45.0,
        )

        assert gpu.memory_free_percent == pytest.approx(0.5, rel=0.01)

    def test_memory_free_percent_zero_total(self):
        """测试零总显存的情况"""
        gpu = GPUResource(
            gpu_id=0,
            memory_total=0,
            memory_used=0,
            utilization=0.0,
            temperature=0.0,
        )

        assert gpu.memory_free_percent == 0.0


class TestNodeResource:
    """节点资源测试"""

    @pytest.fixture
    def sample_gpus(self):
        """创建示例GPU列表"""
        return [
            GPUResource(
                gpu_id=i,
                memory_total=24 * 1024**3,
                memory_used=10 * 1024**3,
                utilization=0.5,
                temperature=45.0,
                node_id="node-1",
            )
            for i in range(4)
        ]

    def test_available_gpus(self, sample_gpus):
        """测试可用GPU筛选"""
        # 创建满载的GPU (内存使用超过95%)
        full_gpu = GPUResource(
            gpu_id=99,
            memory_total=24 * 1024**3,
            memory_used=23 * 1024**3,  # 95%+ 使用率
            utilization=1.0,
            temperature=80.0,
            node_id="node-1",
        )

        # 替换sample_gpus中的第一个GPU
        all_gpus = [full_gpu] + sample_gpus

        node = NodeResource(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.1",
            status="healthy",
            gpu_count=5,
            gpus=all_gpus,
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            load=0.3,
        )

        available = node.available_gpus
        assert len(available) == 4  # 5个GPU中1个满了
        assert all(gpu.memory_free_percent > 0.05 for gpu in available)

    def test_total_available_memory(self, sample_gpus):
        """测试总可用显存"""
        node = NodeResource(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.1",
            status="healthy",
            gpu_count=4,
            gpus=sample_gpus,
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            load=0.3,
        )

        expected = 4 * (14 * 1024**3)
        assert node.total_available_memory == expected

    def test_is_healthy(self):
        """测试健康状态检查"""
        healthy_node = NodeResource(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.1",
            status="healthy",
            gpu_count=4,
            gpus=[],
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            load=0.3,
        )

        unhealthy_node = NodeResource(
            node_id="node-2",
            hostname="server-2",
            ip="192.168.1.2",
            status="unhealthy",
            gpu_count=4,
            gpus=[],
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            load=0.3,
        )

        assert healthy_node.is_healthy is True
        assert unhealthy_node.is_healthy is False

    def test_can_fit_model_success(self):
        """测试模型适配成功"""
        node = NodeResource(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.1",
            status="healthy",
            gpu_count=4,
            gpus=[
                GPUResource(
                    gpu_id=i,
                    memory_total=24 * 1024**3,
                    memory_used=10 * 1024**3,
                    utilization=0.5,
                    temperature=45.0,
                )
                for i in range(4)
            ],
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            load=0.3,
        )

        model_config = {
            "estimated_memory": 20 * 1024**3,
            "tensor_parallel": 2,
        }

        assert node.can_fit_model(model_config) is True

    def test_can_fit_model_insufficient_gpus(self):
        """测试GPU不足"""
        node = NodeResource(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.1",
            status="healthy",
            gpu_count=2,
            gpus=[
                GPUResource(
                    gpu_id=i,
                    memory_total=24 * 1024**3,
                    memory_used=10 * 1024**3,
                    utilization=0.5,
                    temperature=45.0,
                )
                for i in range(2)
            ],
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            load=0.3,
        )

        model_config = {
            "estimated_memory": 20 * 1024**3,
            "tensor_parallel": 4,
        }

        assert node.can_fit_model(model_config) is False


class TestSchedulingRequest:
    """调度请求测试"""

    def test_model_size_property(self):
        """测试模型大小属性"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-7B",
            model_config={"parameter_count": 7_000_000_000},
            prompt="test",
        )

        assert request.model_size == 7_000_000_000

    def test_model_size_default(self):
        """测试默认模型大小"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-7B",
            prompt="test",
        )

        assert request.model_size == 0

    def test_estimated_memory(self):
        """测试显存估算"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-7B",
            model_config={"parameter_count": 7_000_000_000},
            prompt="test",
        )

        # 7B参数约需14GB显存
        expected_memory = (7_000_000_000 // 1_000_000_000) * 2 * 1024**3
        assert request.estimated_memory == expected_memory

    def test_priority_comparison(self):
        """测试优先级比较"""
        request1 = SchedulingRequest(
            request_id="req-001",
            model="test",
            priority=5,
        )

        request2 = SchedulingRequest(
            request_id="req-002",
            model="test",
            priority=8,
        )

        # 用于优先级队列排序
        assert request1.priority < request2.priority


class TestSchedulingResult:
    """调度结果测试"""

    def test_success_result(self):
        """测试成功结果"""
        result = SchedulingResult(
            success=True,
            assigned_nodes=["node-1", "node-2"],
            assigned_gpus={"node-1": [0, 1], "node-2": [0, 1]},
            estimated_wait_time=0.5,
            estimated_latency=1.0,
            strategy_used="gang",
        )

        assert result.success is True
        assert len(result.assigned_nodes) == 2
        assert result.estimated_wait_time == 0.5

    def test_failure_result(self):
        """测试失败结果"""
        result = SchedulingResult(
            success=False,
            reason="Insufficient GPUs",
        )

        assert result.success is False
        assert result.reason == "Insufficient GPUs"


class TestGangSchedulingStrategy:
    """Gang调度策略测试"""

    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        return GangSchedulingStrategy()

    @pytest.fixture
    def healthy_nodes(self):
        """创建健康节点列表"""
        return [
            NodeResource(
                node_id=f"node-{i}",
                hostname=f"server-{i}",
                ip=f"192.168.1.{100 + i}",
                status="healthy",
                gpu_count=4,
                gpus=[
                    GPUResource(
                        gpu_id=j,
                        memory_total=24 * 1024**3,
                        memory_used=10 * 1024**3,
                        utilization=0.5,
                        temperature=45.0,
                        node_id=f"node-{i}",
                    )
                    for j in range(4)
                ],
                cpu_count=32,
                memory_total=128 * 1024**3,
                memory_available=64 * 1024**3,
                disk_total=2 * 1024**4,
                disk_available=1 * 1024**4,
                load=0.3,
            )
            for i in range(3)
        ]

    def test_name(self, strategy):
        """测试策略名称"""
        assert strategy.name == "gang"

    def test_strategy_type(self, strategy):
        """测试策略类型"""
        assert strategy.strategy_type == StrategyType.GANG

    def test_can_handle_large_model(self, strategy, healthy_nodes):
        """测试处理大模型"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-72B",
            model_config={
                "parameter_count": 72_000_000_000,
                "tensor_parallel": 4,
            },
            priority=5,
        )

        assert strategy.can_handle(request, healthy_nodes) is True

    def test_can_handle_insufficient_gpus(self, strategy, healthy_nodes):
        """测试GPU不足情况"""
        request = SchedulingRequest(
            request_id="req-001",
            model="huge-model",
            model_config={
                "parameter_count": 200_000_000_000,
                "tensor_parallel": 20,
            },
        )

        assert strategy.can_handle(request, healthy_nodes) is False

    def test_can_handle_unhealthy_nodes(self, strategy):
        """测试不健康节点"""
        unhealthy_nodes = [
            NodeResource(
                node_id="node-1",
                hostname="server-1",
                ip="192.168.1.1",
                status="unhealthy",
                gpu_count=4,
                gpus=[],
                cpu_count=32,
                memory_total=128 * 1024**3,
                memory_available=64 * 1024**3,
                disk_total=2 * 1024**4,
                disk_available=1 * 1024**4,
                load=0.3,
            )
        ]

        request = SchedulingRequest(
            request_id="req-001",
            model="test",
            model_config={"parameter_count": 72_000_000_000},
        )

        assert strategy.can_handle(request, unhealthy_nodes) is False

    def test_select_nodes_success(self, strategy, healthy_nodes):
        """测试成功选择节点"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-72B",
            model_config={
                "parameter_count": 72_000_000_000,
                "tensor_parallel": 4,
            },
        )

        result = strategy.select_nodes(request, healthy_nodes)

        assert result.success is True
        assert len(result.assigned_nodes) >= 1
        assert result.strategy_used == "gang"

    def test_select_nodes_empty_nodes(self, strategy):
        """测试空节点列表"""
        request = SchedulingRequest(
            request_id="req-001",
            model="test",
            model_config={"tensor_parallel": 1},
        )

        result = strategy.select_nodes(request, [])

        assert result.success is False
        assert "No healthy nodes" in result.reason

    def test_select_nodes_insufficient_gpus(self, strategy, healthy_nodes):
        """测试GPU不足"""
        request = SchedulingRequest(
            request_id="req-001",
            model="huge-model",
            model_config={
                "parameter_count": 200_000_000_000,
                "tensor_parallel": 20,
            },
        )

        result = strategy.select_nodes(request, healthy_nodes)

        assert result.success is False
        assert "Insufficient GPUs" in result.reason

    def test_estimate_wait_time(self, strategy, healthy_nodes):
        """测试等待时间估算"""
        request = SchedulingRequest(
            request_id="req-001",
            model="test",
            max_tokens=2048,
        )

        wait_time = strategy.estimate_wait_time(request, healthy_nodes)

        assert wait_time > 0
        assert isinstance(wait_time, float)


class TestPackSchedulingStrategy:
    """Pack调度策略测试"""

    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        return PackSchedulingStrategy()

    @pytest.fixture
    def healthy_nodes(self):
        """创建健康节点列表"""
        return [
            NodeResource(
                node_id=f"node-{i}",
                hostname=f"server-{i}",
                ip=f"192.168.1.{100 + i}",
                status="healthy",
                gpu_count=4,
                gpus=[
                    GPUResource(
                        gpu_id=j,
                        memory_total=24 * 1024**3,
                        memory_used=10 * 1024**3,
                        utilization=0.5,
                        temperature=45.0,
                        node_id=f"node-{i}",
                    )
                    for j in range(4)
                ],
                cpu_count=32,
                memory_total=128 * 1024**3,
                memory_available=64 * 1024**3,
                disk_total=2 * 1024**4,
                disk_available=1 * 1024**4,
                load=0.3 if i == 0 else 0.1,
            )
            for i in range(3)
        ]

    def test_name(self, strategy):
        """测试策略名称"""
        assert strategy.name == "pack"

    def test_strategy_type(self, strategy):
        """测试策略类型"""
        assert strategy.strategy_type == StrategyType.PACK

    def test_can_handle_small_model(self, strategy, healthy_nodes):
        """测试处理小模型"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-7B",
            model_config={
                "parameter_count": 7_000_000_000,
                "tensor_parallel": 1,
            },
        )

        assert strategy.can_handle(request, healthy_nodes) is True

    def test_select_single_node(self, strategy, healthy_nodes):
        """测试选择单个节点"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-7B",
            model_config={
                "parameter_count": 7_000_000_000,
                "tensor_parallel": 1,
            },
        )

        result = strategy.select_nodes(request, healthy_nodes)

        assert result.success is True
        assert len(result.assigned_nodes) == 1
        assert result.metadata.get("total_gpus") == 1

    def test_select_load_balanced_node(self, strategy, healthy_nodes):
        """测试负载均衡选择"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-7B",
            model_config={
                "parameter_count": 7_000_000_000,
                "tensor_parallel": 1,
            },
        )

        result = strategy.select_nodes(request, healthy_nodes)

        # 应该选择负载最低的节点
        assert result.success is True
        # node-2 和 node-3 的负载是0.1，应该被选中
        assert result.assigned_nodes[0] in ["node-1", "node-2", "node-3"]


class TestAdaptiveSchedulingStrategy:
    """自适应调度策略测试"""

    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        gang = GangSchedulingStrategy()
        pack = PackSchedulingStrategy()
        return AdaptiveSchedulingStrategy(
            strategies={"gang": gang, "pack": pack}
        )

    @pytest.fixture
    def healthy_nodes(self):
        """创建健康节点列表"""
        return [
            NodeResource(
                node_id=f"node-{i}",
                hostname=f"server-{i}",
                ip=f"192.168.1.{100 + i}",
                status="healthy",
                gpu_count=4,
                gpus=[
                    GPUResource(
                        gpu_id=j,
                        memory_total=24 * 1024**3,
                        memory_used=10 * 1024**3,
                        utilization=0.5,
                        temperature=45.0,
                        node_id=f"node-{i}",
                    )
                    for j in range(4)
                ],
                cpu_count=32,
                memory_total=128 * 1024**3,
                memory_available=64 * 1024**3,
                disk_total=2 * 1024**4,
                disk_available=1 * 1024**4,
                load=0.3,
            )
            for i in range(3)
        ]

    def test_name(self, strategy):
        """测试策略名称"""
        assert strategy.name == "adaptive"

    def test_strategy_type(self, strategy):
        """测试策略类型"""
        assert strategy.strategy_type == StrategyType.ADAPTIVE

    def test_can_handle_always(self, strategy, healthy_nodes):
        """测试自适应策略总是可以处理"""
        request = SchedulingRequest(
            request_id="req-001",
            model="test",
        )

        assert strategy.can_handle(request, healthy_nodes) is True

    def test_select_gang_for_large_model(self, strategy, healthy_nodes):
        """测试大模型选择Gang策略"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-72B",
            model_config={
                "parameter_count": 72_000_000_000,
                "tensor_parallel": 4,
            },
        )

        result = strategy.select_nodes(request, healthy_nodes)

        assert result.success is True
        assert "gang" in result.strategy_used

    def test_select_pack_for_small_model(self, strategy, healthy_nodes):
        """测试小模型选择Pack策略"""
        request = SchedulingRequest(
            request_id="req-001",
            model="Qwen2.5-7B",
            model_config={
                "parameter_count": 7_000_000_000,
                "tensor_parallel": 1,
            },
        )

        result = strategy.select_nodes(request, healthy_nodes)

        assert result.success is True
        assert "pack" in result.strategy_used

    def test_select_gang_for_high_priority(self, strategy, healthy_nodes):
        """测试高优先级请求选择Gang策略"""
        request = SchedulingRequest(
            request_id="req-001",
            model="test",
            model_config={"parameter_count": 1_000_000_000},
            priority=9,
        )

        result = strategy.select_nodes(request, healthy_nodes)

        assert result.success is True
        assert "gang" in result.strategy_used

    def test_add_strategy(self, strategy):
        """测试添加新策略"""
        new_strategy = Mock(spec=SchedulingStrategy)
        new_strategy.name = "custom"
        new_strategy.strategy_type = StrategyType.PACK

        strategy.add_strategy("custom", new_strategy)

        assert "custom" in strategy.strategies

    def test_add_rule(self, strategy):
        """测试添加调度规则"""
        initial_rules_count = len(strategy.rules)

        strategy.add_rule(
            condition=lambda r, n: r.priority == 10,
            strategy="gang",
            priority=20,
        )

        assert len(strategy.rules) == initial_rules_count + 1
