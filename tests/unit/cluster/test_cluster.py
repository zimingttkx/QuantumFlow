"""集群管理器单元测试"""

import asyncio
from unittest.mock import Mock

import pytest

from quantumflow.cluster.manager import ClusterManager, GPUInfo, Node, NodeStatus
from quantumflow.scheduler.strategy.base import GPUResource, NodeResource


class TestNodeStatus:
    """节点状态枚举测试"""

    def test_node_status_values(self):
        """测试状态枚举值"""
        assert NodeStatus.JOINING.value == "joining"
        assert NodeStatus.HEALTHY.value == "healthy"
        assert NodeStatus.UNHEALTHY.value == "unhealthy"
        assert NodeStatus.DRAINING.value == "draining"
        assert NodeStatus.OFFLINE.value == "offline"


class TestGPUInfo:
    """GPU信息测试"""

    def test_gpu_info_creation(self):
        """测试GPU信息创建"""
        gpu = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024**3,
            memory_used=10 * 1024**3,
            utilization=0.5,
            temperature=45.0,
        )

        assert gpu.gpu_id == 0
        assert gpu.name == "NVIDIA RTX 4090"
        assert gpu.memory_total == 24 * 1024**3
        assert gpu.memory_used == 10 * 1024**3

    def test_to_resource(self):
        """测试转换为资源对象"""
        gpu = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024**3,
            memory_used=10 * 1024**3,
            utilization=0.5,
            temperature=45.0,
        )

        resource = gpu.to_resource("node-1")

        assert isinstance(resource, GPUResource)
        assert resource.gpu_id == 0
        assert resource.node_id == "node-1"
        assert resource.memory_total == 24 * 1024**3


class TestNode:
    """节点测试"""

    @pytest.fixture
    def sample_gpu_infos(self):
        """创建示例GPU信息列表"""
        return [
            GPUInfo(
                gpu_id=i,
                name="NVIDIA RTX 4090",
                memory_total=24 * 1024**3,
                memory_used=10 * 1024**3,
                utilization=0.5,
                temperature=45.0,
            )
            for i in range(4)
        ]

    def test_node_creation(self, sample_gpu_infos):
        """测试节点创建"""
        node = Node(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.101",
            port=8001,
            gpu_count=4,
            gpu_info=sample_gpu_infos,
            status=NodeStatus.HEALTHY,
            labels={"zone": "zone-a"},
        )

        assert node.node_id == "node-1"
        assert node.hostname == "server-1"
        assert node.gpu_count == 4
        assert node.status == NodeStatus.HEALTHY

    def test_available_gpus(self, sample_gpu_infos):
        """测试可用GPU"""
        # 标记一个GPU为满
        sample_gpu_infos[0].memory_used = sample_gpu_infos[0].memory_total

        node = Node(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.101",
            port=8001,
            gpu_count=4,
            gpu_info=sample_gpu_infos,
            status=NodeStatus.HEALTHY,
        )

        available = node.available_gpus
        assert len(available) == 3
        assert all(gpu.memory_used < gpu.memory_total * 0.95 for gpu in available)

    def test_is_healthy(self, sample_gpu_infos):
        """测试健康状态"""
        healthy_node = Node(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.101",
            port=8001,
            gpu_count=4,
            gpu_info=sample_gpu_infos,
            status=NodeStatus.HEALTHY,
        )

        unhealthy_node = Node(
            node_id="node-2",
            hostname="server-2",
            ip="192.168.1.102",
            port=8001,
            gpu_count=4,
            gpu_info=sample_gpu_infos,
            status=NodeStatus.UNHEALTHY,
        )

        assert healthy_node.is_healthy is True
        assert unhealthy_node.is_healthy is False

    def test_to_resource(self, sample_gpu_infos):
        """测试转换为资源对象"""
        node = Node(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.101",
            port=8001,
            gpu_count=4,
            gpu_info=sample_gpu_infos,
            status=NodeStatus.HEALTHY,
            labels={"zone": "zone-a"},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            loaded_models=["Qwen2.5-7B"],
        )

        resource = node.to_resource()

        assert isinstance(resource, NodeResource)
        assert resource.node_id == "node-1"
        assert resource.hostname == "server-1"
        assert resource.status == "healthy"
        assert len(resource.gpus) == 4
        assert len(resource.loaded_models) == 1


class TestClusterManager:
    """集群管理器测试"""

    @pytest.fixture
    def manager(self):
        """创建集群管理器实例"""
        return ClusterManager(
            heartbeat_interval=5,
            heartbeat_timeout=30,
        )

    @pytest.fixture
    def sample_node_info(self):
        """创建示例节点信息"""
        return {
            "node_id": "node-1",
            "hostname": "server-1",
            "ip": "192.168.1.101",
            "port": 8001,
            "gpu_count": 4,
            "gpu_info": [
                {
                    "gpu_id": i,
                    "name": "NVIDIA RTX 4090",
                    "memory_total": 24 * 1024**3,
                    "memory_used": 10 * 1024**3,
                    "utilization": 0.5,
                    "temperature": 45.0,
                }
                for i in range(4)
            ],
            "labels": {"zone": "zone-a", "gpu_type": "RTX4090"},
            "version": "1.0.0",
        }

    def test_manager_initialization(self, manager):
        """测试管理器初始化"""
        assert manager is not None
        assert manager.heartbeat_interval == 5
        assert manager.heartbeat_timeout == 30
        assert len(manager.nodes) == 0
        assert len(manager._event_handlers) == 4

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """测试启动和停止"""
        await manager.start()
        assert manager._running is True

        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_register_node(self, manager, sample_node_info):
        """测试注册节点"""
        node = await manager.register_node(sample_node_info)

        assert node is not None
        assert node.node_id == "node-1"
        assert node.status == NodeStatus.HEALTHY
        assert len(manager.nodes) == 1

    @pytest.mark.asyncio
    async def test_unregister_node(self, manager, sample_node_info):
        """测试注销节点"""
        await manager.register_node(sample_node_info)

        await manager.unregister_node("node-1")

        assert len(manager.nodes) == 0

    @pytest.mark.asyncio
    async def test_update_node_status(self, manager, sample_node_info):
        """测试更新节点状态"""
        await manager.register_node(sample_node_info)

        success = await manager.update_node_status("node-1", NodeStatus.DRAINING)

        assert success is True
        assert manager.nodes["node-1"].status == NodeStatus.DRAINING

    @pytest.mark.asyncio
    async def test_update_node_status_not_found(self, manager):
        """测试更新不存在的节点"""
        success = await manager.update_node_status("nonexistent", NodeStatus.DRAINING)

        assert success is False

    @pytest.mark.asyncio
    async def test_update_node_info(self, manager, sample_node_info):
        """测试更新节点信息"""
        await manager.register_node(sample_node_info)

        new_gpu_info = [
            {
                "gpu_id": i,
                "name": "NVIDIA RTX 4090",
                "memory_total": 24 * 1024**3,
                "memory_used": 12 * 1024**3,  # 更新使用量
                "utilization": 0.6,
                "temperature": 50.0,
            }
            for i in range(4)
        ]

        await manager.update_node_info("node-1", new_gpu_info, load=0.8)

        node = manager.nodes["node-1"]
        assert node.gpu_info[0].memory_used == 12 * 1024**3
        assert node.current_load == 0.8

    @pytest.mark.asyncio
    async def test_add_loaded_model(self, manager, sample_node_info):
        """测试添加已加载模型"""
        await manager.register_node(sample_node_info)

        await manager.add_loaded_model("node-1", "Qwen2.5-7B")

        assert "Qwen2.5-7B" in manager.nodes["node-1"].loaded_models

    @pytest.mark.asyncio
    async def test_remove_loaded_model(self, manager, sample_node_info):
        """测试移除已加载模型"""
        await manager.register_node(sample_node_info)
        await manager.add_loaded_model("node-1", "Qwen2.5-7B")

        await manager.remove_loaded_model("node-1", "Qwen2.5-7B")

        assert "Qwen2.5-7B" not in manager.nodes["node-1"].loaded_models

    @pytest.mark.asyncio
    async def test_get_node(self, manager, sample_node_info):
        """测试获取节点"""
        await manager.register_node(sample_node_info)

        node = await manager.get_node("node-1")

        assert node is not None
        assert node.node_id == "node-1"

    @pytest.mark.asyncio
    async def test_get_node_not_found(self, manager):
        """测试获取不存在的节点"""
        node = await manager.get_node("nonexistent")

        assert node is None

    @pytest.mark.asyncio
    async def test_get_node_resource(self, manager, sample_node_info):
        """测试获取节点资源"""
        await manager.register_node(sample_node_info)

        resource = await manager.get_node_resource("node-1")

        assert resource is not None
        assert isinstance(resource, NodeResource)
        assert resource.node_id == "node-1"

    @pytest.mark.asyncio
    async def test_get_nodes(self, manager):
        """测试获取节点列表"""
        for i in range(3):
            node_info = {
                "node_id": f"node-{i}",
                "hostname": f"server-{i}",
                "ip": f"192.168.1.{100 + i}",
                "port": 8001,
                "gpu_count": 4,
                "gpu_info": [
                    {
                        "gpu_id": j,
                        "name": "NVIDIA RTX 4090",
                        "memory_total": 24 * 1024**3,
                        "memory_used": 10 * 1024**3,
                        "utilization": 0.5,
                        "temperature": 45.0,
                    }
                    for j in range(4)
                ],
                "labels": {"zone": f"zone-{i % 2}"},
            }
            await manager.register_node(node_info)

        nodes = await manager.get_nodes()
        assert len(nodes) == 3

    @pytest.mark.asyncio
    async def test_get_nodes_by_status(self, manager):
        """测试按状态获取节点"""
        # 注册健康节点
        healthy_info = {
            "node_id": "healthy-node",
            "hostname": "healthy",
            "ip": "192.168.1.101",
            "port": 8001,
            "gpu_count": 4,
            "gpu_info": [],
        }
        await manager.register_node(healthy_info)

        # 注册不健康节点
        unhealthy_info = {
            "node_id": "unhealthy-node",
            "hostname": "unhealthy",
            "ip": "192.168.1.102",
            "port": 8001,
            "gpu_count": 4,
            "gpu_info": [],
        }
        await manager.register_node(unhealthy_info)
        await manager.update_node_status("unhealthy-node", NodeStatus.UNHEALTHY)

        healthy_nodes = await manager.get_nodes(status=NodeStatus.HEALTHY)
        unhealthy_nodes = await manager.get_nodes(status=NodeStatus.UNHEALTHY)

        assert len(healthy_nodes) == 1
        assert len(unhealthy_nodes) == 1

    @pytest.mark.asyncio
    async def test_get_nodes_by_labels(self, manager):
        """测试按标签获取节点"""
        for i in range(3):
            node_info = {
                "node_id": f"node-{i}",
                "hostname": f"server-{i}",
                "ip": f"192.168.1.{100 + i}",
                "port": 8001,
                "gpu_count": 4,
                "gpu_info": [],
                "labels": {"zone": "zone-a" if i < 2 else "zone-b"},
            }
            await manager.register_node(node_info)

        zone_a_nodes = await manager.get_nodes(labels={"zone": "zone-a"})
        assert len(zone_a_nodes) == 2

    @pytest.mark.asyncio
    async def test_get_healthy_nodes(self, manager):
        """测试获取健康节点"""
        for i in range(3):
            node_info = {
                "node_id": f"node-{i}",
                "hostname": f"server-{i}",
                "ip": f"192.168.1.{100 + i}",
                "port": 8001,
                "gpu_count": 4,
                "gpu_info": [],
            }
            await manager.register_node(node_info)

        # 将一个节点设为不健康
        await manager.update_node_status("node-0", NodeStatus.UNHEALTHY)

        healthy = await manager.get_healthy_nodes()
        assert len(healthy) == 2

    @pytest.mark.asyncio
    async def test_find_best_nodes(self, manager):
        """测试查找最优节点"""
        for i in range(3):
            node_info = {
                "node_id": f"node-{i}",
                "hostname": f"server-{i}",
                "ip": f"192.168.1.{100 + i}",
                "port": 8001,
                "gpu_count": 4,
                "gpu_info": [
                    {
                        "gpu_id": j,
                        "name": "NVIDIA RTX 4090",
                        "memory_total": 24 * 1024**3,
                        "memory_used": 10 * 1024**3,
                        "utilization": 0.5,
                        "temperature": 45.0,
                    }
                    for j in range(4)
                ],
            }
            await manager.register_node(node_info)

        # 查找需要8个GPU的情况
        best_nodes = await manager.find_best_nodes(required_gpus=8)

        assert len(best_nodes) >= 2
        assert sum(len(n.available_gpus) for n in best_nodes) >= 8

    @pytest.mark.asyncio
    async def test_get_cluster_stats(self, manager):
        """测试获取集群统计"""
        for i in range(3):
            node_info = {
                "node_id": f"node-{i}",
                "hostname": f"server-{i}",
                "ip": f"192.168.1.{100 + i}",
                "port": 8001,
                "gpu_count": 4,
                "gpu_info": [
                    {
                        "gpu_id": j,
                        "name": "NVIDIA RTX 4090",
                        "memory_total": 24 * 1024**3,
                        "memory_used": 10 * 1024**3,
                        "utilization": 0.5,
                        "temperature": 45.0,
                    }
                    for j in range(4)
                ],
                "loaded_models": ["Qwen2.5-7B"] if i == 0 else [],
            }
            await manager.register_node(node_info)

        stats = await manager.get_cluster_stats()

        assert stats["total_nodes"] == 3
        assert stats["healthy_nodes"] == 3
        assert stats["total_gpus"] == 12
        assert stats["total_models"] == 1

    @pytest.mark.asyncio
    async def test_event_handler(self, manager, sample_node_info):
        """测试事件处理器"""
        handler = Mock()

        manager.on("node_joined", handler)
        await manager.register_node(sample_node_info)

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_node_health_change_event(self, manager, sample_node_info):
        """测试节点健康状态变更事件"""
        handler = Mock()

        manager.on("node_health_changed", handler)
        await manager.register_node(sample_node_info)
        await manager.update_node_status("node-1", NodeStatus.DRAINING)

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_registration(self, manager):
        """测试并发注册"""

        async def register_node(node_id):
            node_info = {
                "node_id": node_id,
                "hostname": f"server-{node_id}",
                "ip": f"192.168.1.{100}",
                "port": 8001,
                "gpu_count": 4,
                "gpu_info": [],
            }
            return await manager.register_node(node_info)

        # 并发注册多个节点
        nodes = await asyncio.gather(*[register_node(f"node-{i}") for i in range(10)])

        assert len(nodes) == 10
        assert len(manager.nodes) == 10
