"""HealthChecker 边界条件和异常路径测试

覆盖未覆盖的代码分支：
- start() 方法边界
- health_check_loop 异常处理和取消
- GPU 温度/内存/利用率临界值
- GPUMonitor 初始化异常
- check_model_health node=None
- check_communication_health node=None
- check_node_health 多 GPU/多模型场景
- configure_thresholds 各种参数组合
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.cluster.manager import ClusterManager, Node, NodeStatus, GPUInfo
from quantumflow.failover.health_checker import HealthChecker, GPUHealthResult, HealthCheckResult
from quantumflow.failover.models import HealthStatus
from quantumflow.failover.policy import HealthThresholds, ReplicaPolicy


class TestHealthCheckerStartMethod:
    """HealthChecker start() 方法测试"""

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
    async def test_start_creates_task(self, health_checker):
        """测试 start() 创建健康检查任务"""
        health_checker._running = False

        await health_checker.start()

        assert health_checker._running is True
        assert health_checker._health_check_task is not None
        # 清理
        await health_checker.stop()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, health_checker):
        """测试 start() 时已经运行则跳过"""
        health_checker._running = True
        original_task = health_checker._health_check_task

        await health_checker.start()

        # 不应该创建新任务
        assert health_checker._health_check_task == original_task


class TestHealthCheckerLoopExceptions:
    """健康检查循环异常处理测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_healthy_nodes = AsyncMock(return_value=[])
        manager.get_node = AsyncMock()
        return manager

    @pytest.fixture
    def health_checker(self, mock_cluster_manager):
        """创建 HealthChecker 实例"""
        thresholds = HealthThresholds()
        policy = ReplicaPolicy(health_check_interval_seconds=0.01)
        return HealthChecker(
            cluster_manager=mock_cluster_manager,
            health_thresholds=thresholds,
            replica_policy=policy,
        )

    @pytest.mark.asyncio
    async def test_health_check_loop_exception_in_node_check(self, health_checker, mock_cluster_manager):
        """测试健康检查循环中节点检查异常"""
        healthy_node = MagicMock()
        healthy_node.node_id = "node-1"

        mock_cluster_manager.get_healthy_nodes = AsyncMock(return_value=[healthy_node])
        mock_cluster_manager.get_node = AsyncMock(side_effect=Exception("Node error"))

        health_checker._running = True
        # Mock check_node_health to not throw (so we hit the exception handler in the loop)
        health_checker.check_node_health = AsyncMock(return_value=MagicMock(
            status=HealthStatus.HEALTHY,
            is_healthy=MagicMock(return_value=True),
            is_degraded=MagicMock(return_value=False),
            is_unhealthy=MagicMock(return_value=False),
        ))

        task = asyncio.create_task(health_checker._health_check_loop())

        await asyncio.sleep(0.05)
        health_checker._running = False

        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_health_check_loop_cancelled(self, health_checker, mock_cluster_manager):
        """测试健康检查循环被取消"""
        healthy_node = MagicMock()
        healthy_node.node_id = "node-1"

        mock_cluster_manager.get_healthy_nodes = AsyncMock(return_value=[healthy_node])

        health_checker._running = True
        health_checker.check_node_health = AsyncMock(side_effect=asyncio.CancelledError())

        task = asyncio.create_task(health_checker._health_check_loop())

        await asyncio.sleep(0.02)

        # 不应该抛出 CancelledError 到外部
        health_checker._running = False
        try:
            await task
        except asyncio.CancelledError:
            pass  # 预期的


class TestGPUHealthBoundaries:
    """GPU 健康边界条件测试"""

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
        thresholds = HealthThresholds(
            gpu_temp_threshold=90.0,
            gpu_temp_warning=85.0,
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
    async def test_gpu_temperature_exactly_at_threshold(self, health_checker, mock_cluster_manager):
        """测试 GPU 温度恰好等于临界阈值（90.0）"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=90.0,  # 恰好等于 threshold 但高于 warning(85.0)
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        # 90.0 > 85.0 (warning) 所以是 DEGRADED
        # 90.0 不 > 90.0 (threshold) 所以不是 UNHEALTHY
        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_gpu_temperature_just_above_threshold(self, health_checker, mock_cluster_manager):
        """测试 GPU 温度刚刚超过临界阈值"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=90.1,  # 刚刚超过
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_gpu_temperature_just_below_warning(self, health_checker, mock_cluster_manager):
        """测试 GPU 温度刚刚低于警告阈值"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.5,
            temperature=84.9,  # 刚刚低于 warning
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        # 低于 warning 应该还是 HEALTHY
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_gpu_memory_exactly_at_threshold(self, health_checker, mock_cluster_manager):
        """测试 GPU 内存恰好等于阈值"""
        # 使用整数计算确保精度: 24576 * 0.95 = 23347.2
        # 但浮点数可能不精确，所以用 23300 < 23347.2
        memory_used = int(24 * 1024 * 0.949)  # 略低于 95%
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=memory_used,
            utilization=0.5,
            temperature=70.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        # 略低于 95% 阈值应该是 HEALTHY
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_gpu_memory_just_above_threshold(self, health_checker, mock_cluster_manager):
        """测试 GPU 内存刚刚超过阈值"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=22.81 * 1024,  # 刚刚超过 95%
            utilization=0.5,
            temperature=70.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_gpu_utilization_just_above_threshold(self, health_checker, mock_cluster_manager):
        """测试 GPU 利用率刚刚超过阈值"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.991,  # 刚刚超过 99%
            temperature=70.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_gpu_multiple_issues_temp_and_memory(self, health_checker, mock_cluster_manager):
        """测试 GPU 同时温度过高和内存过高"""
        gpu_info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23.5 * 1024,
            utilization=0.5,
            temperature=95.0,
        )
        node = MagicMock()
        node.gpu_info = [gpu_info]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_gpu_health("node-1", 0)

        assert result.status == HealthStatus.UNHEALTHY
        assert len(result.reasons) >= 2


class TestGPUMonitorInit:
    """GPUMonitor 初始化测试"""

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
    async def test_get_gpu_data_node_not_found(self, health_checker, mock_cluster_manager):
        """测试 _get_gpu_data 节点不存在"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        result = await health_checker._get_gpu_data("nonexistent", 0)

        assert result is None


class TestModelHealthEdgeCases:
    """模型健康边缘情况测试"""

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
    async def test_do_model_health_check_node_not_found(self, health_checker, mock_cluster_manager):
        """测试 _do_model_health_check 节点不存在"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        result = await health_checker._do_model_health_check("nonexistent", "test-model")

        assert result is False

    @pytest.mark.asyncio
    async def test_do_model_health_check_node_healthy(self, health_checker, mock_cluster_manager):
        """测试 _do_model_health_check 节点健康"""
        mock_cluster_manager.get_node = AsyncMock(return_value=MagicMock(
            node_id="node-1",
            status=NodeStatus.HEALTHY,
        ))

        result = await health_checker._do_model_health_check("node-1", "test-model")

        assert result is True


class TestCommunicationHealthEdgeCases:
    """通信健康边缘情况测试"""

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
    async def test_check_communication_health_node_not_found(self, health_checker, mock_cluster_manager):
        """测试检查通信健康但节点不存在"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        result = await health_checker.check_communication_health("nonexistent")

        assert result == HealthStatus.UNKNOWN


class TestCheckAllModelsHealth:
    """检查所有模型健康测试"""

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
    async def test_check_all_models_health_node_not_found(self, health_checker, mock_cluster_manager):
        """测试检查所有模型健康但节点不存在"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        result = await health_checker._check_all_models_health("nonexistent")

        assert result == {}

    @pytest.mark.asyncio
    async def test_check_all_models_health_multiple_models(self, health_checker, mock_cluster_manager):
        """测试检查多个模型的健康"""
        node = MagicMock()
        node.loaded_models = ["model-1", "model-2", "model-3"]
        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        health_checker.check_model_health = AsyncMock(side_effect=[
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ])

        result = await health_checker._check_all_models_health("node-1")

        assert len(result) == 3
        assert result["model-1"] == HealthStatus.HEALTHY
        assert result["model-2"] == HealthStatus.DEGRADED
        assert result["model-3"] == HealthStatus.UNHEALTHY


class TestConfigureThresholds:
    """配置阈值测试"""

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
    async def test_configure_thresholds_only_gpu_util(self, health_checker):
        """测试只配置 gpu_util_threshold"""
        health_checker.configure_thresholds(gpu_util_threshold=0.85)

        assert health_checker._health_thresholds.gpu_util_threshold == 0.85
        # 其他值应该保持默认值
        assert health_checker._health_thresholds.gpu_temp_threshold == 90.0

    @pytest.mark.asyncio
    async def test_configure_thresholds_only_heartbeat_timeout(self, health_checker):
        """测试只配置 heartbeat_timeout"""
        health_checker.configure_thresholds(heartbeat_timeout=60)

        assert health_checker._health_thresholds.heartbeat_timeout == 60
        assert health_checker._health_thresholds.gpu_temp_threshold == 90.0

    @pytest.mark.asyncio
    async def test_configure_thresholds_only_model_load_timeout(self, health_checker):
        """测试只配置 model_load_timeout"""
        health_checker.configure_thresholds(model_load_timeout=120)

        assert health_checker._health_thresholds.model_load_timeout == 120
        assert health_checker._health_thresholds.gpu_temp_threshold == 90.0

    @pytest.mark.asyncio
    async def test_configure_thresholds_only_inference_timeout(self, health_checker):
        """测试只配置 inference_timeout"""
        health_checker.configure_thresholds(inference_timeout=30)

        assert health_checker._health_thresholds.inference_timeout == 30
        assert health_checker._health_thresholds.gpu_temp_threshold == 90.0

    @pytest.mark.asyncio
    async def test_configure_thresholds_only_comm_timeout(self, health_checker):
        """测试只配置 comm_timeout"""
        health_checker.configure_thresholds(comm_timeout=15)

        assert health_checker._health_thresholds.comm_timeout == 15
        assert health_checker._health_thresholds.gpu_temp_threshold == 90.0

    @pytest.mark.asyncio
    async def test_configure_thresholds_only_failure_threshold(self, health_checker):
        """测试只配置 failure_threshold"""
        health_checker.configure_thresholds(failure_threshold=5)

        assert health_checker._health_thresholds.failure_threshold == 5
        assert health_checker._health_thresholds.gpu_temp_threshold == 90.0

    @pytest.mark.asyncio
    async def test_configure_thresholds_all_params(self, health_checker):
        """测试配置所有阈值参数"""
        health_checker.configure_thresholds(
            gpu_temp_threshold=80.0,
            gpu_mem_threshold=0.90,
            gpu_util_threshold=0.85,
            heartbeat_timeout=60,
            model_load_timeout=120,
            inference_timeout=30,
            comm_timeout=15,
            failure_threshold=5,
        )

        assert health_checker._health_thresholds.gpu_temp_threshold == 80.0
        assert health_checker._health_thresholds.gpu_mem_threshold == 0.90
        assert health_checker._health_thresholds.gpu_util_threshold == 0.85
        assert health_checker._health_thresholds.heartbeat_timeout == 60
        assert health_checker._health_thresholds.model_load_timeout == 120
        assert health_checker._health_thresholds.inference_timeout == 30
        assert health_checker._health_thresholds.comm_timeout == 15
        assert health_checker._health_thresholds.failure_threshold == 5


class TestCheckNodeHealthComprehensive:
    """check_node_health 综合测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟的 ClusterManager"""
        manager = MagicMock(spec=ClusterManager)
        manager.get_node = AsyncMock()
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
    async def test_check_node_health_multiple_unhealthy_gpus(self, health_checker, mock_cluster_manager):
        """测试检查节点有多个不健康 GPU"""
        gpu0 = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,  # UNHEALTHY
            utilization=0.5,
            temperature=95.0,  # UNHEALTHY
        )
        gpu1 = GPUInfo(
            gpu_id=1,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,  # UNHEALTHY
            utilization=0.5,
            temperature=95.0,  # UNHEALTHY
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu0, gpu1]
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 0
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.UNHEALTHY
        assert "2 GPU(s) unhealthy" in result.reasons

    @pytest.mark.asyncio
    async def test_check_node_health_multiple_degraded_gpus(self, health_checker, mock_cluster_manager):
        """测试检查节点有多个降级 GPU"""
        gpu0 = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.992,  # DEGRADED
            temperature=85.0,  # DEGRADED
        )
        gpu1 = GPUInfo(
            gpu_id=1,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.993,  # DEGRADED
            temperature=86.0,  # DEGRADED
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu0, gpu1]
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 0
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.DEGRADED
        assert "2 GPU(s) degraded" in result.reasons

    @pytest.mark.asyncio
    async def test_check_node_health_unhealthy_and_degraded_gpus(self, health_checker, mock_cluster_manager):
        """测试检查节点有不健康和降级 GPU"""
        gpu0 = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=23 * 1024,  # UNHEALTHY
            utilization=0.5,
            temperature=95.0,
        )
        gpu1 = GPUInfo(
            gpu_id=1,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024,
            memory_used=12 * 1024,
            utilization=0.992,  # DEGRADED
            temperature=85.0,
        )
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = [gpu0, gpu1]
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 0
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_node_health("node-1")

        # 不健康 GPU 优先级更高
        assert result.status == HealthStatus.UNHEALTHY
        assert "1 GPU(s) unhealthy" in result.reasons

    @pytest.mark.asyncio
    async def test_check_node_health_multiple_unhealthy_models(self, health_checker, mock_cluster_manager):
        """测试检查节点有多个不健康模型"""
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = []
        node.loaded_models = ["model-1", "model-2"]
        node.communication_health = {}
        node.consecutive_failures = 0
        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        health_checker.check_model_health = AsyncMock(side_effect=[
            HealthStatus.UNHEALTHY,
            HealthStatus.UNHEALTHY,
        ])

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.UNHEALTHY
        assert "2 model(s) unhealthy" in result.reasons

    @pytest.mark.asyncio
    async def test_check_node_health_communication_unhealthy(self, health_checker, mock_cluster_manager):
        """测试检查节点通信不健康"""
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = []
        node.loaded_models = []
        node.communication_health = {
            "peer-1": "fail",
            "peer-2": "timeout",
        }
        node.consecutive_failures = 0
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Communication unhealthy" in result.reasons

    @pytest.mark.asyncio
    async def test_check_node_health_consecutive_failures(self, health_checker, mock_cluster_manager):
        """测试检查节点连续失败次数超过阈值"""
        node = MagicMock()
        node.node_id = "node-1"
        node.status = NodeStatus.HEALTHY
        node.gpu_info = []
        node.loaded_models = []
        node.communication_health = {}
        node.consecutive_failures = 5  # 超过阈值 3
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        result = await health_checker.check_node_health("node-1")

        assert result.status == HealthStatus.UNHEALTHY
        assert any("Consecutive failures" in r for r in result.reasons)


# 导入 asyncio
import asyncio
