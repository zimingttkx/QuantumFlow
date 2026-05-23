"""HealthChecker 扩展测试 - 提高覆盖率

补充测试以覆盖未覆盖的代码分支：
- 异步健康检测循环
- 错误处理路径
- 阈值配置边界
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.failover.health_checker import HealthChecker, GPUHealthResult
from quantumflow.failover.models import HealthStatus
from quantumflow.failover.policy import HealthThresholds, ReplicaPolicy


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
    async def test_stop_when_not_running(self, health_checker):
        """测试停止时如果未运行则跳过"""
        health_checker._running = False

        await health_checker.stop()

        # 不应该报错
        assert health_checker._running is False


class TestHealthCheckerLoop:
    """健康检测循环测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds()
        policy = ReplicaPolicy(health_check_interval_seconds=1)
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_health_check_loop_gets_healthy_nodes(self, health_checker, mock_cluster_manager):
        """测试健康检测循环获取健康节点"""
        healthy_node = MagicMock()
        healthy_node.node_id = "node-1"

        mock_cluster_manager.get_healthy_nodes = AsyncMock(return_value=[healthy_node])
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(status=NodeStatus.HEALTHY))
        health_checker.check_node_health = AsyncMock(
            return_value=MagicMock(
                status=HealthStatus.HEALTHY,
                is_healthy=lambda: True,
                is_degraded=lambda: False,
                is_unhealthy=lambda: False,
                reasons=[],
            )
        )

        health_checker._running = True
        task = asyncio.create_task(health_checker._health_check_loop())

        # 等待循环执行一次
        await asyncio.sleep(0.1)
        health_checker._running = False

        try:
            await task
        except asyncio.CancelledError:
            pass

        mock_cluster_manager.get_healthy_nodes.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_loop_handles_exception(self, health_checker, mock_cluster_manager):
        """测试健康检测循环处理异常"""
        mock_cluster_manager.get_healthy_nodes = AsyncMock(
            side_effect=Exception("Cluster manager error")
        )

        health_checker._running = True
        task = asyncio.create_task(health_checker._health_check_loop())

        # 等待循环处理异常
        await asyncio.sleep(0.2)
        health_checker._running = False

        try:
            await task
        except asyncio.CancelledError:
            pass

        # 循环应该继续运行而不是崩溃
        assert health_checker._running is False


class TestGPUHealthResultMethods:
    """GPUHealthResult 方法测试"""

    def test_gpu_health_result_healthy(self):
        """测试 GPU 健康结果为健康"""
        result = GPUHealthResult(
            gpu_id=0,
            status=HealthStatus.HEALTHY,
        )

        assert result.is_healthy() is True
        assert result.is_degraded() is False
        assert result.is_unhealthy() is False

    def test_gpu_health_result_degraded(self):
        """测试 GPU 健康结果为降级"""
        result = GPUHealthResult(
            gpu_id=0,
            status=HealthStatus.DEGRADED,
        )

        assert result.is_healthy() is False
        assert result.is_degraded() is True
        assert result.is_unhealthy() is False

    def test_gpu_health_result_unhealthy(self):
        """测试 GPU 健康结果为不健康"""
        result = GPUHealthResult(
            gpu_id=0,
            status=HealthStatus.UNHEALTHY,
        )

        assert result.is_healthy() is False
        assert result.is_degraded() is False
        assert result.is_unhealthy() is True


class TestHealthThresholdsEdgeCases:
    """健康阈值边界测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    def test_thresholds_at_boundary(self, mock_cluster_manager):
        """测试阈值在边界值"""
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

    def test_thresholds_zero_values(self, mock_cluster_manager):
        """测试阈值为零"""
        thresholds = HealthThresholds(
            gpu_temp_threshold=0.0,
            gpu_mem_threshold=0.0,
            failure_threshold=0,
        )

        assert thresholds.gpu_temp_threshold == 0.0
        assert thresholds.gpu_mem_threshold == 0.0

    def test_thresholds_to_dict_complete(self, mock_cluster_manager):
        """测试阈值转字典完整性"""
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


class TestConfigureThresholdsEdgeCases:
    """配置阈值边界测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.mark.asyncio
    async def test_configure_partial_values(self, mock_cluster_manager):
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
    async def test_configure_multiple_values(self, mock_cluster_manager):
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


class TestHandleUnhealthyAndDegradedNodes:
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


class TestGetThresholds:
    """获取阈值测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

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


class TestCheckAllGPUsEdgeCases:
    """检查所有 GPU 边界测试"""

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
    async def test_check_all_gpus_empty_node(self, health_checker, mock_cluster_manager):
        """测试检查空节点的 GPU"""
        node = MagicMock()
        node.gpu_info = []
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        results = await health_checker.check_all_gpus("node-1")

        assert results == {}

    @pytest.mark.asyncio
    async def test_check_all_gpus_multiple_gpus(self, health_checker, mock_cluster_manager):
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


# 需要导入 asyncio
import asyncio
