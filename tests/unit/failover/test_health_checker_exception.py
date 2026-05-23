"""HealthChecker 异常路径和边界测试

覆盖未覆盖的异常处理分支：
- get_gpu_data 异常
- check_model_health 异常
- _do_model_health_check 异常
- check_communication_health 异常
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.failover.health_checker import HealthChecker, GPUHealthResult
from quantumflow.failover.models import HealthStatus
from quantumflow.failover.policy import HealthThresholds, ReplicaPolicy


class TestHealthCheckerExceptionPaths:
    """HealthChecker 异常路径测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        manager.update_model_health = AsyncMock()
        manager.update_gpu_health = AsyncMock()
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
    async def test_get_gpu_data_exception(self, health_checker, mock_cluster_manager):
        """测试获取 GPU 数据异常"""
        def get_node_side_effect(node_id):
            if "node-1" in node_id:
                raise Exception("Node error")
            return None

        mock_cluster_manager.get_node = AsyncMock(side_effect=get_node_side_effect)

        result = await health_checker._get_gpu_data("node-1", 0)

        assert result is None

    @pytest.mark.asyncio
    async def test_check_model_health_exception(self, health_checker, mock_cluster_manager):
        """测试检查模型健康异常"""
        mock_cluster_manager.get_node = AsyncMock(side_effect=Exception("Cluster error"))

        result = await health_checker.check_model_health("node-1", "test-model")

        # 异常时返回 UNHEALTHY（根据代码）
        assert result == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_model_health_model_not_loaded(self, health_checker, mock_cluster_manager):
        """测试检查模型健康但模型未加载"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            status=NodeStatus.HEALTHY,
            loaded_models=[],
        ))

        result = await health_checker.check_model_health("node-1", "test-model")

        assert result == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_do_model_health_check_exception(self, health_checker, mock_cluster_manager):
        """测试_do_model_health_check异常"""
        mock_cluster_manager.get_node = AsyncMock(side_effect=Exception("Node error"))

        result = await health_checker._do_model_health_check("node-1", "test-model")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_communication_health_exception(self, health_checker, mock_cluster_manager):
        """测试检查通信健康异常"""
        mock_cluster_manager.get_node = AsyncMock(side_effect=Exception("Cluster error"))

        result = await health_checker.check_communication_health("node-1")

        assert result == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_check_communication_health_no_peers(self, health_checker, mock_cluster_manager):
        """测试检查通信健康但没有对等节点"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            communication_health={},
        ))

        result = await health_checker.check_communication_health("node-1")

        assert result == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_communication_health_unhealthy(self, health_checker, mock_cluster_manager):
        """测试检查通信健康大部分不健康"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            communication_health={
                "node-2": "fail",
                "node-3": "timeout",
                "node-4": "healthy",  # 只有1个健康
            },
        ))

        result = await health_checker.check_communication_health("node-1")

        assert result == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_communication_health_degraded(self, health_checker, mock_cluster_manager):
        """测试检查通信健康降级"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            communication_health={
                "node-2": "degraded",
                "node-3": "healthy",
            },
        ))

        result = await health_checker.check_communication_health("node-1")

        assert result == HealthStatus.DEGRADED


class TestGPUHealthCheck:
    """GPU 健康检查测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
        manager.update_gpu_health = AsyncMock()
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
    async def test_check_gpu_health_healthy(self, health_checker, mock_cluster_manager):
        """测试检查 GPU 健康"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=70.0,
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            gpu_info=[gpu_info],
        ))

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_gpu_health_overheating(self, health_checker, mock_cluster_manager):
        """测试检查 GPU 过热"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=95.0,  # 超过阈值
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            gpu_info=[gpu_info],
        ))

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNHEALTHY
        assert any("temperature" in r for r in result.reasons)

    @pytest.mark.asyncio
    async def test_check_gpu_health_high_memory(self, health_checker, mock_cluster_manager):
        """测试检查 GPU 高内存使用"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,  # 96% 使用率
            utilization=0.5,
            temperature=70.0,
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            gpu_info=[gpu_info],
        ))

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_gpu_health_not_found(self, health_checker, mock_cluster_manager):
        """测试检查不存在的 GPU"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            gpu_info=[],
        ))

        result = await health_checker.check_gpu_health("node-1", 99)

        # GPU 不存在时返回 UNKNOWN
        assert result.status == HealthStatus.UNKNOWN


class TestHealthCheckerGetAllGPUHealth:
    """获取所有 GPU 健康状态测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
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
            memory_used=20 * 1024,
            utilization=0.8,
            temperature=80.0,
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            gpu_info=[gpu0, gpu1],
        ))

        results = await health_checker.check_all_gpus("node-1")

        assert len(results) == 2
        assert 0 in results
        assert 1 in results

    @pytest.mark.asyncio
    async def test_check_all_gpus_node_not_found(self, health_checker, mock_cluster_manager):
        """测试检查所有 GPU 但节点不存在"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        results = await health_checker.check_all_gpus("nonexistent")

        assert results == {}
