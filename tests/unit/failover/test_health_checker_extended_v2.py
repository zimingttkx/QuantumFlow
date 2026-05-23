"""HealthChecker 扩展测试 V2 - 提高覆盖率

补充测试以覆盖未覆盖的代码分支：
- HealthCheckResult 状态判断方法
- GPUHealthResult 状态判断方法
- 配置方法
- 同步方法
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.failover.health_checker import HealthChecker, HealthCheckResult, GPUHealthResult
from quantumflow.failover.models import HealthStatus
from quantumflow.failover.policy import HealthThresholds, ReplicaPolicy


class TestHealthCheckResultMethods:
    """HealthCheckResult 方法测试"""

    def test_is_healthy_true(self):
        """测试 is_healthy 返回 True"""
        result = HealthCheckResult(
            node_id="node-1",
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            checks_performed={},
            details={},
            reasons=[],
        )
        assert result.is_healthy() is True
        assert result.is_degraded() is False
        assert result.is_unhealthy() is False

    def test_is_degraded_true(self):
        """测试 is_degraded 返回 True"""
        result = HealthCheckResult(
            node_id="node-1",
            status=HealthStatus.DEGRADED,
            timestamp=datetime.now(),
            checks_performed={},
            details={},
            reasons=["High temperature"],
        )
        assert result.is_healthy() is False
        assert result.is_degraded() is True
        assert result.is_unhealthy() is False

    def test_is_unhealthy_true(self):
        """测试 is_unhealthy 返回 True"""
        result = HealthCheckResult(
            node_id="node-1",
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.now(),
            checks_performed={},
            details={},
            reasons=["GPU failure", "Memory full"],
        )
        assert result.is_healthy() is False
        assert result.is_degraded() is False
        assert result.is_unhealthy() is True


class TestGPUHealthResultMethods:
    """GPUHealthResult 方法测试"""

    def test_gpu_health_result_healthy(self):
        """测试 GPU 健康结果为健康"""
        result = GPUHealthResult(
            gpu_id=0,
            status=HealthStatus.HEALTHY,
            temperature=60.0,
            memory_used_ratio=0.5,
            utilization=0.3,
        )
        assert result.is_healthy() is True
        assert result.is_degraded() is False
        assert result.is_unhealthy() is False

    def test_gpu_health_result_degraded(self):
        """测试 GPU 健康结果为降级"""
        result = GPUHealthResult(
            gpu_id=1,
            status=HealthStatus.DEGRADED,
            temperature=85.0,
            memory_used_ratio=0.90,
            utilization=0.95,
            reasons=["High temperature"],
        )
        assert result.is_healthy() is False
        assert result.is_degraded() is True
        assert result.is_unhealthy() is False

    def test_gpu_health_result_unhealthy(self):
        """测试 GPU 健康结果为不健康"""
        result = GPUHealthResult(
            gpu_id=2,
            status=HealthStatus.UNHEALTHY,
            temperature=95.0,
            memory_used_ratio=0.99,
            utilization=1.0,
            reasons=["Critical temperature", "Memory almost full"],
        )
        assert result.is_healthy() is False
        assert result.is_degraded() is False
        assert result.is_unhealthy() is True

    def test_gpu_health_result_no_reasons(self):
        """测试 GPU 健康结果无原因"""
        result = GPUHealthResult(
            gpu_id=0,
            status=HealthStatus.HEALTHY,
        )
        assert result.reasons == []


class TestHealthCheckerStartStop:
    """健康检测器启动/停止测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_healthy_nodes = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds()
        policy = ReplicaPolicy(health_check_interval_seconds=60)
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, health_checker):
        """测试启动时如果已在运行则跳过"""
        health_checker._running = True

        await health_checker.start()

        # 不应该创建新任务
        assert health_checker._health_check_task is None

    @pytest.mark.asyncio
    async def test_stop_normal(self, health_checker):
        """测试正常停止"""
        health_checker._running = True
        health_checker._health_check_task = None  # 没有实际任务

        await health_checker.stop()

        assert health_checker._running is False

    @pytest.mark.asyncio
    async def test_stop_with_task(self, health_checker):
        """测试停止时取消任务"""
        health_checker._running = True
        health_checker._health_check_task = asyncio.create_task(asyncio.sleep(10))

        await health_checker.stop()

        assert health_checker._running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, health_checker):
        """测试停止时如果未运行则跳过"""
        health_checker._running = False

        await health_checker.stop()

        # 不应该报错
        assert health_checker._running is False


class TestHealthCheckerConfigure:
    """健康检测器配置测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.mark.asyncio
    async def test_configure_thresholds_partial(self, mock_cluster_manager):
        """测试部分配置阈值"""
        health_checker = HealthChecker(
            cluster_manager=mock_cluster_manager,
        )

        # 只配置一个值
        health_checker.configure_thresholds(gpu_temp_threshold=85.0)

        assert health_checker._health_thresholds.gpu_temp_threshold == 85.0
        # 其他值应该保持默认值
        assert health_checker._health_thresholds.gpu_mem_threshold == 0.95

    @pytest.mark.asyncio
    async def test_configure_thresholds_multiple(self, mock_cluster_manager):
        """测试配置多个阈值"""
        health_checker = HealthChecker(
            cluster_manager=mock_cluster_manager,
        )

        health_checker.configure_thresholds(
            gpu_temp_threshold=80.0,
            gpu_mem_threshold=0.90,
            heartbeat_timeout=60,
            failure_threshold=5,
        )

        assert health_checker._health_thresholds.gpu_temp_threshold == 80.0
        assert health_checker._health_thresholds.gpu_mem_threshold == 0.90
        assert health_checker._health_thresholds.heartbeat_timeout == 60
        assert health_checker._health_thresholds.failure_threshold == 5

    @pytest.mark.asyncio
    async def test_get_thresholds(self, mock_cluster_manager):
        """测试获取当前阈值"""
        thresholds = HealthThresholds(gpu_temp_threshold=85.0)
        health_checker = HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
        )

        result = health_checker.get_thresholds()

        assert result.gpu_temp_threshold == 85.0
        assert isinstance(result, HealthThresholds)


class TestCheckNodeHealth:
    """检查节点健康测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        manager.get_healthy_nodes = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds()
        policy = ReplicaPolicy()
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_check_node_health_node_not_found(self, health_checker, mock_cluster_manager):
        """测试检查不存在的节点"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        result = await health_checker.check_node_health("nonexistent-node")

        # 节点不存在时状态为 UNKNOWN 或 UNHEALTHY
        assert result.status in (HealthStatus.UNKNOWN, HealthStatus.UNHEALTHY)
        assert len(result.reasons) > 0

    @pytest.mark.asyncio
    async def test_check_node_health_empty_reasons(self, health_checker, mock_cluster_manager):
        """测试检查节点健康无原因"""
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = []
        node.last_heartbeat = datetime.now()

        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_node_health("node-1")

        # 应该有原因列表（即使是空）
        assert isinstance(result.reasons, list)


class TestCheckAllGPUs:
    """检查所有 GPU 测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds()
        policy = ReplicaPolicy()
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_check_all_gpus_empty(self, health_checker, mock_cluster_manager):
        """测试检查空 GPU 列表"""
        node = MagicMock()
        node.gpu_info = []
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        results = await health_checker.check_all_gpus("node-1")

        assert results == {}

    @pytest.mark.asyncio
    async def test_check_all_gpus_multiple(self, health_checker, mock_cluster_manager):
        """测试检查多个 GPU"""
        gpu0 = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=70.0,
        )
        gpu1 = GPUInfo(
            gpu_id=1,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=75.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu0, gpu1]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_gpu_health = AsyncMock()

        results = await health_checker.check_all_gpus("node-1")

        assert len(results) == 2
        assert 0 in results
        assert 1 in results


class TestHandleUnhealthyDegradedNodes:
    """处理不健康/降级节点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.update_node_health = AsyncMock()
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds()
        policy = ReplicaPolicy()
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_handle_unhealthy_node(self, health_checker, mock_cluster_manager):
        """测试处理不健康节点"""
        result = MagicMock()
        result.reasons = ["GPU failure", "Memory full"]

        await health_checker._handle_unhealthy_node("node-1", result)

        mock_cluster_manager.update_node_health.assert_called_once()
        call_args = mock_cluster_manager.update_node_health.call_args
        assert call_args[0][0] == "node-1"
        assert call_args[0][1] == "unhealthy"

    @pytest.mark.asyncio
    async def test_handle_degraded_node(self, health_checker, mock_cluster_manager):
        """测试处理降级节点"""
        result = MagicMock()
        result.reasons = ["High temperature"]

        await health_checker._handle_degraded_node("node-1", result)

        mock_cluster_manager.update_node_health.assert_called_once()
        call_args = mock_cluster_manager.update_node_health.call_args
        assert call_args[0][1] == "degraded"


class TestHealthCheckerThresholds:
    """健康阈值测试"""

    def test_health_thresholds_default_values(self):
        """测试默认阈值"""
        thresholds = HealthThresholds()

        assert thresholds.gpu_temp_threshold == 90.0
        assert thresholds.gpu_mem_threshold == 0.95
        assert thresholds.gpu_util_threshold == 0.99
        assert thresholds.heartbeat_timeout == 30
        assert thresholds.failure_threshold == 3
        assert thresholds.degraded_threshold == 2

    def test_health_thresholds_boundary_values(self):
        """测试边界阈值"""
        thresholds = HealthThresholds(
            gpu_temp_threshold=90.0,
            gpu_mem_threshold=0.95,
            gpu_util_threshold=0.99,
            heartbeat_timeout=30,
            failure_threshold=3,
        )

        assert thresholds.gpu_temp_threshold == 90.0
        assert thresholds.gpu_mem_threshold == 0.95
        assert thresholds.failure_threshold == 3

    def test_health_thresholds_to_dict(self):
        """测试阈值转字典"""
        thresholds = HealthThresholds()

        data = thresholds.to_dict()

        # 验证所有字段都存在
        assert "gpu_temp_threshold" in data
        assert "gpu_mem_threshold" in data
        assert "gpu_util_threshold" in data
        assert "heartbeat_timeout" in data
        assert "failure_threshold" in data
        assert "degraded_threshold" in data
        assert "jitter_window" in data


# 需要导入 asyncio
import asyncio
