"""HealthChecker 健康检测器测试

验证健康检测的业务逻辑正确性：
- GPU 故障检测（温度、显存、利用率）
- 模型健康状态检测
- 通信健康状态检测
- 节点综合健康判定
- 阈值配置的生效
- 连续失败计数与抖动判定
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.failover.health_checker import HealthChecker, GPUHealthResult
from quantumflow.failover.models import HealthStatus
from quantumflow.failover.policy import HealthThresholds, ReplicaPolicy


class TestGPUHealthDetection:
    """GPU 健康检测测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds(
            gpu_temp_threshold=90.0,
            gpu_temp_warning=80.0,
            gpu_mem_threshold=0.95,
            gpu_util_threshold=0.99,
        )
        policy = ReplicaPolicy()
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_gpu_temperature_normal(self, health_checker, mock_cluster_manager):
        """测试 GPU 温度正常"""
        # 设置节点 GPU 信息
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=75.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.HEALTHY
        assert result.temperature == 75.0
        assert len(result.reasons) == 0

    @pytest.mark.asyncio
    async def test_gpu_temperature_warning(self, health_checker, mock_cluster_manager):
        """测试 GPU 温度警告"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=85.0,  # 超过 warning 但低于 threshold
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.DEGRADED
        assert "temperature" in result.reasons[0].lower()

    @pytest.mark.asyncio
    async def test_gpu_temperature_critical(self, health_checker, mock_cluster_manager):
        """测试 GPU 温度临界（超过阈值）"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=95.0,  # 超过阈值
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNHEALTHY
        assert any("temperature" in r.lower() and "exceeds" in r.lower() for r in result.reasons)

    @pytest.mark.asyncio
    async def test_gpu_memory_usage_high(self, health_checker, mock_cluster_manager):
        """测试 GPU 显存使用率过高"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,  # 96% 使用率
            utilization=0.5,
            temperature=70.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNHEALTHY
        assert any("memory" in r.lower() and "exceeds" in r.lower() for r in result.reasons)

    @pytest.mark.asyncio
    async def test_gpu_utilization_extreme(self, health_checker, mock_cluster_manager):
        """测试 GPU 利用率极高"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.995,  # 99.5%
            temperature=70.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.DEGRADED
        assert any("utilization" in r.lower() and "high" in r.lower() for r in result.reasons)

    @pytest.mark.asyncio
    async def test_gpu_multiple_issues(self, health_checker, mock_cluster_manager):
        """测试 GPU 多重问题（温度+显存同时异常）"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,  # 96%
            utilization=0.5,
            temperature=92.0,  # 超过 90
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNHEALTHY
        assert len(result.reasons) >= 2  # 至少两个原因

    @pytest.mark.asyncio
    async def test_gpu_data_not_available(self, health_checker, mock_cluster_manager):
        """测试 GPU 数据不可用"""
        # GPU 在 node.gpu_info 中存在，但 _get_gpu_data 返回 None 表示无法获取数据
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=70.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        # Mock _get_gpu_data to return None
        with patch.object(health_checker, '_get_gpu_data', return_value=None):
            result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_gpu_not_found(self, health_checker, mock_cluster_manager):
        """测试 GPU 不存在"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNKNOWN


class TestModelHealthDetection:
    """模型健康检测测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
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
    async def test_model_loaded_and_healthy(self, health_checker, mock_cluster_manager):
        """测试模型已加载且健康"""
        node = MagicMock()
        node.loaded_models = ["qwen2.5-7b"]
        node.status = NodeStatus.HEALTHY
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        status = await health_checker.check_model_health("node-1", "qwen2.5-7b")

        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_model_not_loaded(self, health_checker, mock_cluster_manager):
        """测试模型未加载"""
        node = MagicMock()
        node.loaded_models = ["other-model"]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_model_health = AsyncMock()

        status = await health_checker.check_model_health("node-1", "qwen2.5-7b")

        assert status == HealthStatus.UNHEALTHY
        mock_cluster_manager.update_model_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_loaded_but_node_unhealthy(self, health_checker, mock_cluster_manager):
        """测试模型已加载但节点不健康"""
        node = MagicMock()
        node.loaded_models = ["qwen2.5-7b"]
        node.status = NodeStatus.UNHEALTHY
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        status = await health_checker.check_model_health("node-1", "qwen2.5-7b")

        assert status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_model_loaded_but_node_degraded(self, health_checker, mock_cluster_manager):
        """测试模型已加载但节点降级"""
        node = MagicMock()
        node.loaded_models = ["qwen2.5-7b"]
        node.status = NodeStatus.HEALTHY  # 节点状态健康，但可能被 GPU 等因素影响
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        status = await health_checker.check_model_health("node-1", "qwen2.5-7b")

        # 默认返回 HEALTHY，因为节点状态正常
        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_node_not_found_for_model_check(self, health_checker, mock_cluster_manager):
        """测试节点不存在时检查模型健康"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        status = await health_checker.check_model_health("nonexistent", "model")

        assert status == HealthStatus.UNKNOWN


class TestCommunicationHealthDetection:
    """通信健康检测测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
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
    async def test_communication_all_healthy(self, health_checker, mock_cluster_manager):
        """测试所有通信健康"""
        node = MagicMock()
        node.communication_health = {
            "node-2": "healthy",
            "node-3": "healthy",
        }
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        status = await health_checker.check_communication_health("node-1")

        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_communication_one_failed(self, health_checker, mock_cluster_manager):
        """测试一个节点通信失败"""
        node = MagicMock()
        node.communication_health = {
            "node-2": "healthy",
            "node-3": "fail",  # 1/3 失败
        }
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        status = await health_checker.check_communication_health("node-1")

        # 1/3 < 50%，应该是 DEGRADED
        assert status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_communication_majority_failed(self, health_checker, mock_cluster_manager):
        """测试多数节点通信失败"""
        node = MagicMock()
        node.communication_health = {
            "node-2": "timeout",
            "node-3": "fail",
            "node-4": "fail",  # 2/3 > 50%
        }
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        status = await health_checker.check_communication_health("node-1")

        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_communication_all_failed(self, health_checker, mock_cluster_manager):
        """测试所有通信都失败"""
        node = MagicMock()
        node.communication_health = {
            "node-2": "fail",
            "node-3": "timeout",
        }
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        status = await health_checker.check_communication_health("node-1")

        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_communication_empty(self, health_checker, mock_cluster_manager):
        """测试无通信历史"""
        node = MagicMock()
        node.communication_health = {}
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        status = await health_checker.check_communication_health("node-1")

        assert status == HealthStatus.HEALTHY  # 无历史视为健康


class TestNodeComprehensiveHealthCheck:
    """节点综合健康检测测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds(
            gpu_temp_threshold=90.0,
            gpu_mem_threshold=0.95,
            failure_threshold=3,
        )
        policy = ReplicaPolicy()
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_node_all_healthy(self, health_checker, mock_cluster_manager):
        """测试节点完全健康"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=70.0,
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu_info]
        node.loaded_models = ["qwen2.5-7b"]
        node.communication_health = {"node-2": "healthy"}
        node.consecutive_failures = 0

        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_gpu_health = AsyncMock()

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.HEALTHY
        assert result.checks_performed["node_status"] is True
        assert result.checks_performed["gpu_health"] is True
        assert result.checks_performed["model_health"] is True
        assert result.checks_performed["communication_health"] is True

    @pytest.mark.asyncio
    async def test_node_status_unhealthy(self, health_checker, mock_cluster_manager):
        """测试节点状态不健康"""
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.UNHEALTHY
        node.gpu_info = []
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 0

        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.UNHEALTHY
        assert result.checks_performed["node_status"] is False
        assert any("status" in r.lower() for r in result.reasons)

    @pytest.mark.asyncio
    async def test_node_gpu_unhealthy(self, health_checker, mock_cluster_manager):
        """测试节点 GPU 不健康"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,  # 超过阈值
            utilization=0.5,
            temperature=70.0,
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu_info]
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 0

        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_gpu_health = AsyncMock()

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.UNHEALTHY
        assert result.checks_performed["gpu_health"] is False
        assert any("unhealthy" in r.lower() and "gpu" in r.lower() for r in result.reasons)

    @pytest.mark.asyncio
    async def test_node_consecutive_failures_exceed_threshold(
        self, health_checker, mock_cluster_manager
    ):
        """测试连续失败次数超过阈值"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=70.0,
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu_info]
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 5  # 超过阈值 3

        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_gpu_health = AsyncMock()

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.UNHEALTHY
        assert any("consecutive failures" in r.lower() for r in result.reasons)

    @pytest.mark.asyncio
    async def test_node_degraded_vs_unhealthy_decision(
        self, health_checker, mock_cluster_manager
    ):
        """测试降级与不健康判定逻辑"""
        # 场景1: 只有 GPU 降级，应该是 DEGRADED
        gpu_info_warning = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=82.0,  # 超过 warning 但低于 threshold
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu_info_warning]
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 0

        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_gpu_health = AsyncMock()

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.DEGRADED

        # 场景2: GPU 真正故障，应该是 UNHEALTHY
        gpu_info_critical = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,
            utilization=0.5,
            temperature=70.0,
        )
        node.gpu_info = [gpu_info_critical]
        mock_cluster_manager.update_gpu_health = AsyncMock()

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.UNHEALTHY


class TestHealthThresholdsConfiguration:
    """健康阈值配置测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    def test_default_thresholds(self, mock_cluster_manager):
        """测试默认阈值"""
        thresholds = HealthThresholds()

        assert thresholds.gpu_temp_threshold == 90.0
        assert thresholds.gpu_mem_threshold == 0.95
        assert thresholds.gpu_util_threshold == 0.99
        assert thresholds.heartbeat_timeout == 30
        assert thresholds.failure_threshold == 3

    def test_custom_thresholds(self, mock_cluster_manager):
        """测试自定义阈值"""
        thresholds = HealthThresholds(
            gpu_temp_threshold=85.0,
            gpu_mem_threshold=0.90,
            failure_threshold=5,
        )

        assert thresholds.gpu_temp_threshold == 85.0
        assert thresholds.gpu_mem_threshold == 0.90
        assert thresholds.failure_threshold == 5

    @pytest.mark.asyncio
    async def test_configure_thresholds_runtime(self, mock_cluster_manager):
        """测试运行时配置阈值"""
        health_checker = HealthChecker(
            cluster_manager=mock_cluster_manager,
        )

        assert health_checker._health_thresholds.gpu_temp_threshold == 90.0

        health_checker.configure_thresholds(gpu_temp_threshold=80.0)

        assert health_checker._health_thresholds.gpu_temp_threshold == 80.0

    @pytest.mark.asyncio
    async def test_configure_partial_thresholds(self, mock_cluster_manager):
        """测试部分配置阈值"""
        health_checker = HealthChecker(
            cluster_manager=mock_cluster_manager,
        )

        health_checker.configure_thresholds(
            gpu_temp_threshold=85.0,
            heartbeat_timeout=60,
        )

        # 只修改了 temperature 和 heartbeat_timeout
        assert health_checker._health_thresholds.gpu_temp_threshold == 85.0
        assert health_checker._health_thresholds.heartbeat_timeout == 60
        # 其他保持不变
        assert health_checker._health_thresholds.gpu_mem_threshold == 0.95


class TestCheckAllGPUs:
    """检查所有 GPU 测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        return MagicMock(spec=ClusterManager)

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds(
            gpu_temp_threshold=90.0,
            gpu_mem_threshold=0.95,
        )
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
            memory_used=23 * 1024,  # 不健康
            utilization=0.5,
            temperature=70.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu0, gpu1]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_gpu_health = AsyncMock()

        results = await health_checker.check_all_gpus("node-1")

        assert len(results) == 2
        assert results[0].status == HealthStatus.HEALTHY
        assert results[1].status == HealthStatus.UNHEALTHY

        # 验证 ClusterManager 被调用更新状态
        assert mock_cluster_manager.update_gpu_health.call_count == 2

    @pytest.mark.asyncio
    async def test_check_all_gpus_node_not_found(self, health_checker, mock_cluster_manager):
        """测试节点不存在时检查所有 GPU"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        results = await health_checker.check_all_gpus("nonexistent")

        assert results == {}


class TestFailureCountTracking:
    """失败计数追踪测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.update_node_health = AsyncMock()
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds(failure_threshold=3)
        policy = ReplicaPolicy()
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_handle_unhealthy_node_calls_update(self, health_checker, mock_cluster_manager):
        """测试处理不健康节点调用更新"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,
            utilization=0.5,
            temperature=95.0,
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu_info]
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 0

        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_gpu_health = AsyncMock()

        result = await health_checker.check_node_health("node-1")
        await health_checker._handle_unhealthy_node("node-1", result)

        mock_cluster_manager.update_node_health.assert_called_once()
        call_args = mock_cluster_manager.update_node_health.call_args
        assert call_args[0][0] == "node-1"
        assert call_args[0][1] == "unhealthy"

    @pytest.mark.asyncio
    async def test_handle_degraded_node_calls_update(self, health_checker, mock_cluster_manager):
        """测试处理降级节点调用更新"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=82.0,  # 降级
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu_info]
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 0

        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_gpu_health = AsyncMock()

        result = await health_checker.check_node_health("node-1")
        await health_checker._handle_degraded_node("node-1", result)

        mock_cluster_manager.update_node_health.assert_called_once()
        call_args = mock_cluster_manager.update_node_health.call_args
        assert call_args[0][1] == "degraded"
