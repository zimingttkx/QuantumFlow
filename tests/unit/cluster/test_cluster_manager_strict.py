"""ClusterManager 严格业务逻辑测试

严格按照以下标准编写：
1. 校验业务逻辑正确性，不只是运行可用性
2. 全覆盖：正常用例、边界值、错误输入、异常场景
3. 强精准断言，明确预期值
4. 重点测试核心功能、易错细节、状态变更
"""

import asyncio
import sys
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.cluster.manager import ClusterManager, GPUInfo, Node, NodeStatus
from quantumflow.core.constants import NodeStatus as NodeStatusEnum


class TestNodeDataModel:
    """Node 数据模型测试"""

    def test_node_creation_with_all_fields(self):
        """[正常用例] 使用完整字段创建 Node"""
        gpu = GPUInfo(
            gpu_id=0,
            name="NVIDIA A100",
            memory_total=80 * 1024**3,
            memory_used=40 * 1024**3,
            utilization=0.5,
            temperature=65.0,
        )
        node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[gpu],
            status=NodeStatus.HEALTHY,
            labels={"zone": "us-east", "env": "prod"},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=["model-a", "model-b"],
        )

        assert node.node_id == "node-1"
        assert node.hostname == "worker-1"
        assert node.ip == "192.168.1.100"
        assert node.port == 8080
        assert node.gpu_count == 1
        assert len(node.gpu_info) == 1
        assert node.status == NodeStatus.HEALTHY
        assert node.labels == {"zone": "us-east", "env": "prod"}
        assert node.version == "1.0.0"
        assert node.cpu_count == 32
        assert node.memory_total == 128 * 1024**3
        assert node.memory_available == 64 * 1024**3
        assert node.disk_total == 1 * 1024**4
        assert node.disk_available == 512 * 1024**3
        assert node.current_load == 0.3
        assert node.loaded_models == ["model-a", "model-b"]
        assert isinstance(node.last_heartbeat, datetime)

    def test_node_creation_with_minimal_fields(self):
        """[边界用例] 使用最小字段创建 Node"""
        node = Node(
            node_id="node-min",
            hostname="minimal",
            ip="127.0.0.1",
            port=8000,
            gpu_count=0,
            gpu_info=[],
            status=NodeStatus.INITIALIZING,
        )

        assert node.node_id == "node-min"
        assert node.hostname == "minimal"
        assert node.labels == {}  # 默认空字典
        assert node.version == "1.0.0"  # 默认版本
        assert node.loaded_models == []  # 默认空列表
        assert node.last_heartbeat is not None

    def test_node_available_gpus_below_threshold(self):
        """[正常用例] available_gpus 返回 memory_used < 95% total 的 GPU"""
        gpu1 = GPUInfo(
            gpu_id=0,
            name="GPU 0",
            memory_total=100,
            memory_used=90,  # 90% < 95%, 属于可用
            utilization=0.9,
            temperature=70.0,
        )
        gpu2 = GPUInfo(
            gpu_id=1,
            name="GPU 1",
            memory_total=100,
            memory_used=96,  # 96% >= 95%, 属于不可用
            utilization=0.96,
            temperature=80.0,
        )
        gpu3 = GPUInfo(
            gpu_id=2,
            name="GPU 2",
            memory_total=100,
            memory_used=94.9,  # 94.9% < 95%, 属于可用
            utilization=0.949,
            temperature=60.0,
        )
        node = Node(
            node_id="test",
            hostname="h",
            ip="127.0.0.1",
            port=8000,
            gpu_count=3,
            gpu_info=[gpu1, gpu2, gpu3],
            status=NodeStatus.HEALTHY,
        )

        available = node.available_gpus
        assert len(available) == 2
        assert available[0].gpu_id == 0
        assert available[1].gpu_id == 2

    def test_node_available_gpus_exactly_at_threshold(self):
        """[边界用例] available_gpus 在 95% 阈值边界 (94.99% 可用, 95% 不可用)"""
        gpu_at_94_99 = GPUInfo(
            gpu_id=0,
            name="GPU 0",
            memory_total=10000,
            memory_used=9499,  # 94.99%
            utilization=0.9499,
            temperature=70.0,
        )
        gpu_at_95 = GPUInfo(
            gpu_id=1,
            name="GPU 1",
            memory_total=10000,
            memory_used=9500,  # exactly 95%
            utilization=0.95,
            temperature=75.0,
        )
        node = Node(
            node_id="test",
            hostname="h",
            ip="127.0.0.1",
            port=8000,
            gpu_count=2,
            gpu_info=[gpu_at_94_99, gpu_at_95],
            status=NodeStatus.HEALTHY,
        )

        available = node.available_gpus
        assert len(available) == 1
        assert available[0].gpu_id == 0  # 94.99% 可用

    def test_node_available_gpus_empty_list(self):
        """[边界用例] 无 GPU 时 available_gpus 返回空列表"""
        node = Node(
            node_id="no-gpu",
            hostname="nogpu",
            ip="127.0.0.1",
            port=8000,
            gpu_count=0,
            gpu_info=[],
            status=NodeStatus.HEALTHY,
        )

        assert node.available_gpus == []

    def test_node_is_healthy_property(self):
        """[正常用例] is_healthy 属性正确反映健康状态"""
        node = Node(
            node_id="test",
            hostname="h",
            ip="127.0.0.1",
            port=8000,
            gpu_count=0,
            gpu_info=[],
            status=NodeStatus.HEALTHY,
        )
        assert node.is_healthy is True

        node.status = NodeStatus.UNHEALTHY
        assert node.is_healthy is False

        node.status = NodeStatus.OFFLINE
        assert node.is_healthy is False

        node.status = NodeStatus.DRAINING
        assert node.is_healthy is False

    def test_gpu_info_to_resource(self):
        """[正常用例] GPUInfo.to_resource 正确转换"""
        gpu = GPUInfo(
            gpu_id=1,
            name="NVIDIA V100",
            memory_total=32 * 1024**3,
            memory_used=16 * 1024**3,
            utilization=0.5,
            temperature=60.0,
        )
        resource = gpu.to_resource("node-x")

        assert resource.gpu_id == 1
        assert resource.memory_total == 32 * 1024**3
        assert resource.memory_used == 16 * 1024**3
        assert resource.utilization == 0.5
        assert resource.temperature == 60.0
        assert resource.node_id == "node-x"


class TestClusterManagerRegistration:
    """ClusterManager 节点注册/注销逻辑测试"""

    @pytest.fixture
    def manager(self):
        """创建 ClusterManager 实例"""
        return ClusterManager(heartbeat_interval=1, heartbeat_timeout=5)

    @pytest.fixture
    def valid_node_info(self) -> dict[str, Any]:
        """标准节点信息"""
        return {
            "node_id": "node-1",
            "hostname": "worker-1",
            "ip": "192.168.1.100",
            "port": 8080,
            "gpu_count": 2,
            "gpu_info": [
                {
                    "gpu_id": 0,
                    "name": "NVIDIA A100",
                    "memory_total": 80 * 1024**3,
                    "memory_used": 40 * 1024**3,
                    "utilization": 0.5,
                    "temperature": 65.0,
                },
                {
                    "gpu_id": 1,
                    "name": "NVIDIA A100",
                    "memory_total": 80 * 1024**3,
                    "memory_used": 20 * 1024**3,
                    "utilization": 0.25,
                    "temperature": 55.0,
                },
            ],
            "status": "healthy",
            "labels": {"zone": "us-east"},
            "version": "1.0.0",
            "cpu_count": 32,
            "memory_total": 128 * 1024**3,
            "memory_available": 64 * 1024**3,
            "loaded_models": ["model-a"],
        }

    @pytest.mark.asyncio
    async def test_register_node_success(self, manager, valid_node_info):
        """[正常用例] 成功注册节点"""
        node = await manager.register_node(valid_node_info)

        # 验证节点基本属性
        assert node.node_id == "node-1"
        assert node.hostname == "worker-1"
        assert node.ip == "192.168.1.100"
        assert node.port == 8080
        assert node.gpu_count == 2
        assert node.status == NodeStatus.HEALTHY  # 注册时强制设为 HEALTHY

        # 验证 GPU 信息转换正确
        assert len(node.gpu_info) == 2
        assert node.gpu_info[0].gpu_id == 0
        assert node.gpu_info[0].memory_total == 80 * 1024**3
        assert node.gpu_info[0].memory_used == 40 * 1024**3
        assert node.gpu_info[1].gpu_id == 1

        # 验证额外属性
        assert node.cpu_count == 32
        assert node.memory_total == 128 * 1024**3
        assert node.loaded_models == ["model-a"]
        assert node.labels == {"zone": "us-east"}

    @pytest.mark.asyncio
    async def test_register_node_stores_in_nodes_dict(self, manager, valid_node_info):
        """[正常用例] 注册后节点存储在 nodes 字典中"""
        await manager.register_node(valid_node_info)

        assert "node-1" in manager.nodes
        assert manager.nodes["node-1"].node_id == "node-1"

    @pytest.mark.asyncio
    async def test_register_node_starts_heartbeat_monitor(self, manager, valid_node_info):
        """[正常用例] 注册后启动心跳监控"""
        await manager.register_node(valid_node_info)

        # 验证心跳监控任务已创建
        assert "node-1" in manager._heartbeat_tasks
        assert isinstance(manager._heartbeat_tasks["node-1"], asyncio.Task)

    @pytest.mark.asyncio
    async def test_register_multiple_nodes_independent(self, manager, valid_node_info):
        """[正常用例] 注册多个独立节点互不影响"""
        node_info_1 = valid_node_info.copy()
        node_info_2 = valid_node_info.copy()
        node_info_2["node_id"] = "node-2"
        node_info_2["ip"] = "192.168.1.101"

        await manager.register_node(node_info_1)
        await manager.register_node(node_info_2)

        assert len(manager.nodes) == 2
        assert "node-1" in manager.nodes
        assert "node-2" in manager.nodes
        assert manager.nodes["node-1"].ip == "192.168.1.100"
        assert manager.nodes["node-2"].ip == "192.168.1.101"

    @pytest.mark.asyncio
    async def test_register_node_minimal_info(self, manager):
        """[边界用例] 使用最小信息注册节点"""
        minimal_info = {
            "node_id": "node-min",
            "hostname": "minimal",
            "ip": "127.0.0.1",
            "port": 8000,
            "gpu_count": 0,
        }

        node = await manager.register_node(minimal_info)

        assert node.node_id == "node-min"
        assert node.gpu_count == 0
        assert node.gpu_info == []
        assert node.status == NodeStatus.HEALTHY
        assert node.labels == {}
        assert node.version == "1.0.0"
        assert node.loaded_models == []

    @pytest.mark.asyncio
    async def test_unregister_node_success(self, manager, valid_node_info):
        """[正常用例] 成功注销已注册节点"""
        await manager.register_node(valid_node_info)
        assert "node-1" in manager.nodes

        await manager.unregister_node("node-1")

        assert "node-1" not in manager.nodes

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_node_no_error(self, manager):
        """[错误处理] 注销不存在的节点不抛出异常"""
        # 不应抛出异常
        await manager.unregister_node("nonexistent")
        await manager.unregister_node("")
        await manager.unregister_node("node-999")

        assert len(manager.nodes) == 0

    @pytest.mark.asyncio
    async def test_unregister_node_cancels_heartbeat_task(self, manager, valid_node_info):
        """[正常用例] 注销节点时取消心跳监控任务"""
        await manager.register_node(valid_node_info)
        task = manager._heartbeat_tasks["node-1"]

        await manager.unregister_node("node-1")

        # 任务应该被取消
        assert task.cancelled() or "node-1" not in manager._heartbeat_tasks

    @pytest.mark.asyncio
    async def test_unregister_and_reregister_same_node(self, manager, valid_node_info):
        """[正常用例] 注销后可以重新注册同一节点 ID"""
        await manager.register_node(valid_node_info)
        await manager.unregister_node("node-1")

        # 重新注册
        node = await manager.register_node(valid_node_info)
        assert node.node_id == "node-1"
        assert "node-1" in manager.nodes


class TestClusterManagerStatusUpdate:
    """ClusterManager 状态更新逻辑测试"""

    @pytest.fixture
    def manager_with_node(self):
        """创建带节点的 ClusterManager"""
        manager = ClusterManager(heartbeat_interval=1, heartbeat_timeout=5)
        node_info = {
            "node_id": "node-1",
            "hostname": "worker-1",
            "ip": "192.168.1.100",
            "port": 8080,
            "gpu_count": 1,
            "gpu_info": [
                {
                    "gpu_id": 0,
                    "name": "NVIDIA A100",
                    "memory_total": 80 * 1024**3,
                    "memory_used": 40 * 1024**3,
                    "utilization": 0.5,
                    "temperature": 65.0,
                }
            ],
            "status": "healthy",
            "labels": {},
            "version": "1.0.0",
            "cpu_count": 32,
            "memory_total": 128 * 1024**3,
            "memory_available": 64 * 1024**3,
            "loaded_models": [],
        }
        return manager, node_info

    @pytest.mark.asyncio
    async def test_update_node_status_healthy_to_unhealthy(self, manager_with_node):
        """[正常用例] 更新节点状态 HEALTHY -> UNHEALTHY"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        result = await manager.update_node_status("node-1", NodeStatusEnum.UNHEALTHY)

        assert result is True
        assert manager.nodes["node-1"].status == NodeStatusEnum.UNHEALTHY

    @pytest.mark.asyncio
    async def test_update_node_status_healthy_to_draining(self, manager_with_node):
        """[正常用例] 更新节点状态 HEALTHY -> DRAINING"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        result = await manager.update_node_status("node-1", NodeStatusEnum.DRAINING)

        assert result is True
        assert manager.nodes["node-1"].status == NodeStatusEnum.DRAINING

    @pytest.mark.asyncio
    async def test_update_node_status_draining_to_healthy(self, manager_with_node):
        """[正常用例] 更新节点状态 DRAINING -> HEALTHY (恢复)"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)
        await manager.update_node_status("node-1", NodeStatusEnum.DRAINING)

        result = await manager.update_node_status("node-1", NodeStatusEnum.HEALTHY)

        assert result is True
        assert manager.nodes["node-1"].status == NodeStatusEnum.HEALTHY

    @pytest.mark.asyncio
    async def test_update_node_status_all_transitions(self, manager_with_node):
        """[多分支逻辑] 测试所有可能的状态转换"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        transitions = [
            (NodeStatusEnum.HEALTHY, NodeStatusEnum.UNHEALTHY),
            (NodeStatusEnum.UNHEALTHY, NodeStatusEnum.DRAINING),
            (NodeStatusEnum.DRAINING, NodeStatusEnum.OFFLINE),
            (NodeStatusEnum.OFFLINE, NodeStatusEnum.HEALTHY),
        ]

        for from_status, to_status in transitions:
            await manager.update_node_status("node-1", from_status)
            assert manager.nodes["node-1"].status == from_status

            result = await manager.update_node_status("node-1", to_status)
            assert result is True
            assert manager.nodes["node-1"].status == to_status

    @pytest.mark.asyncio
    async def test_update_node_status_nonexistent_node(self, manager_with_node):
        """[错误处理] 更新不存在节点的状态返回 False"""
        manager, _ = manager_with_node

        result = await manager.update_node_status("nonexistent", NodeStatusEnum.UNHEALTHY)

        assert result is False

    @pytest.mark.asyncio
    async def test_update_node_status_same_status_no_change(self, manager_with_node):
        """[正常用例] 更新为相同状态"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        result = await manager.update_node_status("node-1", NodeStatusEnum.HEALTHY)

        assert result is True
        assert manager.nodes["node-1"].status == NodeStatusEnum.HEALTHY


class TestClusterManagerInfoUpdate:
    """ClusterManager 信息更新逻辑测试"""

    @pytest.fixture
    def manager_with_node(self):
        """创建带节点的 ClusterManager"""
        manager = ClusterManager(heartbeat_interval=1, heartbeat_timeout=5)
        node_info = {
            "node_id": "node-1",
            "hostname": "worker-1",
            "ip": "192.168.1.100",
            "port": 8080,
            "gpu_count": 1,
            "gpu_info": [
                {
                    "gpu_id": 0,
                    "name": "NVIDIA A100",
                    "memory_total": 80 * 1024**3,
                    "memory_used": 40 * 1024**3,
                    "utilization": 0.5,
                    "temperature": 65.0,
                }
            ],
            "status": "healthy",
            "labels": {},
            "version": "1.0.0",
            "cpu_count": 32,
            "memory_total": 128 * 1024**3,
            "memory_available": 64 * 1024**3,
            "loaded_models": [],
        }
        return manager, node_info

    @pytest.mark.asyncio
    async def test_update_node_info_gpu_info(self, manager_with_node):
        """[正常用例] 更新 GPU 信息"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        new_gpu_info = [
            {
                "gpu_id": 0,
                "name": "NVIDIA A100",
                "memory_total": 80 * 1024**3,
                "memory_used": 60 * 1024**3,  # 增加使用量
                "utilization": 0.75,
                "temperature": 75.0,
            }
        ]

        await manager.update_node_info("node-1", gpu_info=new_gpu_info)

        updated_node = manager.nodes["node-1"]
        assert len(updated_node.gpu_info) == 1
        assert updated_node.gpu_info[0].memory_used == 60 * 1024**3
        assert updated_node.gpu_info[0].utilization == 0.75
        assert updated_node.gpu_info[0].temperature == 75.0

    @pytest.mark.asyncio
    async def test_update_node_info_load(self, manager_with_node):
        """[正常用例] 更新负载信息"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        original_load = manager.nodes["node-1"].current_load

        await manager.update_node_info("node-1", gpu_info=[], load=0.8)

        assert manager.nodes["node-1"].current_load == 0.8
        assert manager.nodes["node-1"].current_load != original_load

    @pytest.mark.asyncio
    async def test_update_node_info_updates_last_heartbeat(self, manager_with_node):
        """[正常用例] 更新信息时刷新 last_heartbeat"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        original_heartbeat = manager.nodes["node-1"].last_heartbeat

        # 等待一小段时间确保时间戳差异
        await asyncio.sleep(0.01)

        await manager.update_node_info("node-1", gpu_info=[], load=0.5)

        new_heartbeat = manager.nodes["node-1"].last_heartbeat
        assert new_heartbeat > original_heartbeat

    @pytest.mark.asyncio
    async def test_update_node_info_nonexistent_node(self, manager_with_node):
        """[错误处理] 更新不存在节点的 info 不抛出异常"""
        manager, _ = manager_with_node

        # 不应抛出异常
        await manager.update_node_info("nonexistent", gpu_info=[], load=0.5)


class TestClusterManagerModelManagement:
    """ClusterManager 模型加载管理测试"""

    @pytest.fixture
    def manager_with_node(self):
        """创建带节点的 ClusterManager"""
        manager = ClusterManager(heartbeat_interval=1, heartbeat_timeout=5)
        node_info = {
            "node_id": "node-1",
            "hostname": "worker-1",
            "ip": "192.168.1.100",
            "port": 8080,
            "gpu_count": 1,
            "gpu_info": [],
            "status": "healthy",
            "labels": {},
            "version": "1.0.0",
            "cpu_count": 32,
            "memory_total": 128 * 1024**3,
            "memory_available": 64 * 1024**3,
            "loaded_models": [],
        }
        return manager, node_info

    @pytest.mark.asyncio
    async def test_add_loaded_model_success(self, manager_with_node):
        """[正常用例] 成功添加已加载模型"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        await manager.add_loaded_model("node-1", "model-a")

        assert "model-a" in manager.nodes["node-1"].loaded_models

    @pytest.mark.asyncio
    async def test_add_loaded_model_duplicate_no_duplicate(self, manager_with_node):
        """[正常用例] 添加重复模型不会产生重复"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        await manager.add_loaded_model("node-1", "model-a")
        await manager.add_loaded_model("node-1", "model-a")
        await manager.add_loaded_model("node-1", "model-a")

        assert manager.nodes["node-1"].loaded_models.count("model-a") == 1

    @pytest.mark.asyncio
    async def test_add_multiple_models(self, manager_with_node):
        """[正常用例] 添加多个不同模型"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        await manager.add_loaded_model("node-1", "model-a")
        await manager.add_loaded_model("node-1", "model-b")
        await manager.add_loaded_model("node-1", "model-c")

        assert len(manager.nodes["node-1"].loaded_models) == 3
        assert "model-a" in manager.nodes["node-1"].loaded_models
        assert "model-b" in manager.nodes["node-1"].loaded_models
        assert "model-c" in manager.nodes["node-1"].loaded_models

    @pytest.mark.asyncio
    async def test_remove_loaded_model_success(self, manager_with_node):
        """[正常用例] 成功移除已加载模型"""
        manager, node_info = manager_with_node
        node_info["loaded_models"] = ["model-a", "model-b"]
        await manager.register_node(node_info)

        await manager.remove_loaded_model("node-1", "model-a")

        assert "model-a" not in manager.nodes["node-1"].loaded_models
        assert "model-b" in manager.nodes["node-1"].loaded_models

    @pytest.mark.asyncio
    async def test_remove_nonexistent_model_no_error(self, manager_with_node):
        """[错误处理] 移除不存在的模型不抛出异常"""
        manager, node_info = manager_with_node
        await manager.register_node(node_info)

        # 不应抛出异常
        await manager.remove_loaded_model("node-1", "nonexistent")

    @pytest.mark.asyncio
    async def test_add_model_to_nonexistent_node_no_error(self, manager_with_node):
        """[错误处理] 向不存在的节点添加模型不抛出异常"""
        manager, _ = manager_with_node

        # 不应抛出异常
        await manager.add_loaded_model("nonexistent", "model-a")


class TestClusterManagerQuery:
    """ClusterManager 查询接口测试"""

    @pytest.fixture
    def manager_with_nodes(self):
        """创建带多个节点的 ClusterManager"""
        manager = ClusterManager(heartbeat_interval=1, heartbeat_timeout=5)

        nodes_info = [
            {
                "node_id": "node-1",
                "hostname": "worker-1",
                "ip": "192.168.1.100",
                "port": 8080,
                "gpu_count": 2,
                "gpu_info": [],
                "status": "healthy",
                "labels": {"zone": "us-east", "type": "gpu"},
                "version": "1.0.0",
                "cpu_count": 32,
                "memory_total": 128 * 1024**3,
                "memory_available": 64 * 1024**3,
                "loaded_models": ["model-a"],
                "target_status": NodeStatusEnum.HEALTHY,
            },
            {
                "node_id": "node-2",
                "hostname": "worker-2",
                "ip": "192.168.1.101",
                "port": 8081,
                "gpu_count": 1,
                "gpu_info": [],
                "status": "unhealthy",
                "labels": {"zone": "us-east", "type": "cpu"},
                "version": "1.0.0",
                "cpu_count": 16,
                "memory_total": 64 * 1024**3,
                "memory_available": 32 * 1024**3,
                "loaded_models": [],
                "target_status": NodeStatusEnum.UNHEALTHY,
            },
            {
                "node_id": "node-3",
                "hostname": "worker-3",
                "ip": "192.168.1.102",
                "port": 8082,
                "gpu_count": 4,
                "gpu_info": [],
                "status": "healthy",
                "labels": {"zone": "us-west", "type": "gpu"},
                "version": "1.0.0",
                "cpu_count": 64,
                "memory_total": 256 * 1024**3,
                "memory_available": 128 * 1024**3,
                "loaded_models": ["model-b", "model-c"],
                "target_status": NodeStatusEnum.HEALTHY,
            },
        ]

        async def register_all():
            for info in nodes_info:
                target = info.pop("target_status")
                await manager.register_node(info)
                if target != NodeStatusEnum.HEALTHY:
                    await manager.update_node_status(info["node_id"], target)

        asyncio.run(register_all())

        return manager

    @pytest.mark.asyncio
    async def test_get_node_exists(self, manager_with_nodes):
        """[正常用例] 获取存在的节点"""
        manager = manager_with_nodes

        node = await manager.get_node("node-1")

        assert node is not None
        assert node.node_id == "node-1"
        assert node.hostname == "worker-1"

    @pytest.mark.asyncio
    async def test_get_node_nonexistent(self, manager_with_nodes):
        """[错误处理] 获取不存在的节点返回 None"""
        manager = manager_with_nodes

        node = await manager.get_node("nonexistent")

        assert node is None

    @pytest.mark.asyncio
    async def test_get_nodes_all(self, manager_with_nodes):
        """[正常用例] 获取所有节点"""
        manager = manager_with_nodes

        nodes = await manager.get_nodes()

        assert len(nodes) == 3

    @pytest.mark.asyncio
    async def test_get_nodes_filter_by_status(self, manager_with_nodes):
        """[正常用例] 按状态过滤节点"""
        manager = manager_with_nodes

        healthy_nodes = await manager.get_nodes(status=NodeStatusEnum.HEALTHY)
        unhealthy_nodes = await manager.get_nodes(status=NodeStatusEnum.UNHEALTHY)

        assert len(healthy_nodes) == 2
        assert len(unhealthy_nodes) == 1

        for node in healthy_nodes:
            assert node.status == NodeStatusEnum.HEALTHY

        for node in unhealthy_nodes:
            assert node.status == NodeStatusEnum.UNHEALTHY

    @pytest.mark.asyncio
    async def test_get_nodes_filter_by_labels(self, manager_with_nodes):
        """[正常用例] 按标签过滤节点"""
        manager = manager_with_nodes

        us_east_nodes = await manager.get_nodes(labels={"zone": "us-east"})
        gpu_nodes = await manager.get_nodes(labels={"type": "gpu"})

        assert len(us_east_nodes) == 2
        assert len(gpu_nodes) == 2

    @pytest.mark.asyncio
    async def test_get_nodes_filter_by_multiple_labels(self, manager_with_nodes):
        """[边界用例] 按多个标签过滤 (AND 逻辑)"""
        manager = manager_with_nodes

        nodes = await manager.get_nodes(labels={"zone": "us-east", "type": "gpu"})

        assert len(nodes) == 1
        assert nodes[0].node_id == "node-1"

    @pytest.mark.asyncio
    async def test_get_healthy_nodes(self, manager_with_nodes):
        """[正常用例] 获取所有健康节点"""
        manager = manager_with_nodes

        healthy = await manager.get_healthy_nodes()

        assert len(healthy) == 2
        for node in healthy:
            assert node.status == NodeStatusEnum.HEALTHY

    @pytest.mark.asyncio
    async def test_get_node_resource(self, manager_with_nodes):
        """[正常用例] 获取节点资源信息"""
        manager = manager_with_nodes

        resource = await manager.get_node_resource("node-1")

        assert resource is not None
        assert resource.node_id == "node-1"
        assert resource.hostname == "worker-1"
        assert resource.status == "healthy"
        assert resource.gpu_count == 2


class TestClusterManagerFindBestNodes:
    """ClusterManager 最优节点查找测试"""

    @pytest.fixture
    def manager_with_gpu_nodes(self):
        """创建带 GPU 信息的节点"""
        manager = ClusterManager(heartbeat_interval=1, heartbeat_timeout=5)

        # 创建三个节点，GPU 使用率不同
        async def setup():
            # 节点1: 4 GPU, 2 可用
            node1_info = {
                "node_id": "node-1",
                "hostname": "worker-1",
                "ip": "192.168.1.100",
                "port": 8080,
                "gpu_count": 4,
                "gpu_info": [
                    {
                        "gpu_id": i,
                        "name": "A100",
                        "memory_total": 80 * 1024**3,
                        "memory_used": 76 * 1024**3,  # 95% - 不可用
                        "utilization": 0.95,
                        "temperature": 70.0,
                    }
                    for i in range(4)
                ],
                "status": "healthy",
                "labels": {},
                "version": "1.0.0",
                "cpu_count": 32,
                "memory_total": 128 * 1024**3,
                "memory_available": 64 * 1024**3,
                "loaded_models": [],
            }

            # 节点2: 2 GPU, 2 可用 (轻载)
            node2_info = {
                "node_id": "node-2",
                "hostname": "worker-2",
                "ip": "192.168.1.101",
                "port": 8081,
                "gpu_count": 2,
                "gpu_info": [
                    {
                        "gpu_id": i,
                        "name": "A100",
                        "memory_total": 80 * 1024**3,
                        "memory_used": 10 * 1024**3,  # 轻载 - 可用
                        "utilization": 0.125,
                        "temperature": 50.0,
                    }
                    for i in range(2)
                ],
                "status": "healthy",
                "labels": {},
                "version": "1.0.0",
                "cpu_count": 32,
                "memory_total": 128 * 1024**3,
                "memory_available": 64 * 1024**3,
                "loaded_models": [],
            }

            # 节点3: 4 GPU, 0 可用 (满载)
            node3_info = {
                "node_id": "node-3",
                "hostname": "worker-3",
                "ip": "192.168.1.102",
                "port": 8082,
                "gpu_count": 4,
                "gpu_info": [
                    {
                        "gpu_id": i,
                        "name": "A100",
                        "memory_total": 80 * 1024**3,
                        "memory_used": 79 * 1024**3,  # 超过95% - 不可用
                        "utilization": 0.99,
                        "temperature": 85.0,
                    }
                    for i in range(4)
                ],
                "status": "healthy",
                "labels": {},
                "version": "1.0.0",
                "cpu_count": 64,
                "memory_total": 256 * 1024**3,
                "memory_available": 128 * 1024**3,
                "loaded_models": [],
            }

            await manager.register_node(node1_info)
            await manager.register_node(node2_info)
            await manager.register_node(node3_info)

        asyncio.run(setup())
        return manager

    @pytest.mark.asyncio
    async def test_find_best_nodes_single_node(self, manager_with_gpu_nodes):
        """[正常用例] 查找需要 1 个 GPU 的最优节点"""
        manager = manager_with_gpu_nodes

        best = await manager.find_best_nodes(required_gpus=1)

        assert len(best) >= 1
        # node-2 有 2 个可用 GPU，应该被优先选中
        assert best[0].node_id == "node-2"

    @pytest.mark.asyncio
    async def test_find_best_nodes_multiple_nodes(self, manager_with_gpu_nodes):
        """[正常用例] 查找需要多个 GPU 的最优节点组合"""
        manager = manager_with_gpu_nodes

        best = await manager.find_best_nodes(required_gpus=4)

        # node-2 有 2 个可用，node-1 有 0 个可用，node-3 有 0 个可用
        # 需要 4 个 GPU，但总共只有 2 个可用
        # 应该选择 node-2
        assert len(best) == 1
        assert best[0].node_id == "node-2"

    @pytest.mark.asyncio
    async def test_find_best_nodes_none_available(self, manager_with_gpu_nodes):
        """[边界用例] 没有可用 GPU 时返回空列表"""
        manager = manager_with_gpu_nodes

        # 先把所有节点设为不健康
        for node_id in manager.nodes:
            await manager.update_node_status(node_id, NodeStatusEnum.UNHEALTHY)

        best = await manager.find_best_nodes(required_gpus=1)

        assert best == []


class TestClusterManagerStats:
    """ClusterManager 统计信息测试"""

    @pytest.fixture
    def manager_with_mixed_nodes(self):
        """创建混合状态节点"""
        manager = ClusterManager(heartbeat_interval=1, heartbeat_timeout=5)

        async def setup():
            for i, status in enumerate(
                [NodeStatusEnum.HEALTHY, NodeStatusEnum.HEALTHY, NodeStatusEnum.UNHEALTHY]
            ):
                node_info = {
                    "node_id": f"node-{i}",
                    "hostname": f"worker-{i}",
                    "ip": f"192.168.1.{100+i}",
                    "port": 8080 + i,
                    "gpu_count": 2 if i < 2 else 1,
                    "gpu_info": [
                        {
                            "gpu_id": 0,
                            "name": "A100",
                            "memory_total": 80 * 1024**3,
                            "memory_used": 40 * 1024**3,
                            "utilization": 0.5,
                            "temperature": 60.0,
                        }
                    ],
                    "status": status.value,
                    "labels": {},
                    "version": "1.0.0",
                    "cpu_count": 32,
                    "memory_total": 128 * 1024**3,
                    "memory_available": 64 * 1024**3,
                    "loaded_models": [f"model-{i}"] if i == 0 else [],
                }
                await manager.register_node(node_info)
                if status != NodeStatusEnum.HEALTHY:
                    await manager.update_node_status(f"node-{i}", status)

        asyncio.run(setup())
        return manager

    @pytest.mark.asyncio
    async def test_get_cluster_stats(self, manager_with_mixed_nodes):
        """[正常用例] 获取集群统计信息"""
        manager = manager_with_mixed_nodes

        stats = await manager.get_cluster_stats()

        assert stats["total_nodes"] == 3
        assert stats["healthy_nodes"] == 2
        assert stats["unhealthy_nodes"] == 1
        assert stats["total_gpus"] == 5  # 2 + 2 + 1
        assert stats["total_models"] == 1  # 只有 model-0


class TestClusterManagerHeartbeatTimeout:
    """ClusterManager 心跳超时处理测试"""

    @pytest.fixture
    def manager_short_timeout(self):
        """创建短超时的 ClusterManager 用于测试"""
        return ClusterManager(heartbeat_interval=1, heartbeat_timeout=2)

    @pytest.mark.asyncio
    async def test_heartbeat_loop_stops_on_unhealthy(self, manager_short_timeout):
        """[异常场景] 心跳循环在节点不健康时停止"""
        node_info = {
            "node_id": "node-1",
            "hostname": "worker-1",
            "ip": "192.168.1.100",
            "port": 8080,
            "gpu_count": 1,
            "gpu_info": [],
            "status": "healthy",
            "labels": {},
            "version": "1.0.0",
            "cpu_count": 32,
            "memory_total": 128 * 1024**3,
            "memory_available": 64 * 1024**3,
            "loaded_models": [],
        }

        await manager_short_timeout.register_node(node_info)

        # 将节点设为不健康
        await manager_short_timeout.update_node_status("node-1", NodeStatusEnum.UNHEALTHY)

        # 等待心跳循环检测到状态变化并退出
        await asyncio.sleep(0.5)

        # 验证心跳任务已停止或即将停止
        task = manager_short_timeout._heartbeat_tasks.get("node-1")
        if task:
            assert task.done() or task.cancelled()


class TestClusterManagerEventSystem:
    """ClusterManager 事件系统测试"""

    @pytest.fixture
    def manager(self):
        """创建 ClusterManager"""
        return ClusterManager(heartbeat_interval=1, heartbeat_timeout=5)

    @pytest.mark.asyncio
    async def test_event_handler_registration(self, manager):
        """[正常用例] 事件处理器注册"""
        handler = MagicMock()

        manager.on("node_joined", handler)

        assert handler in manager._event_handlers["node_joined"]

    @pytest.mark.asyncio
    async def test_event_handler_sync_call(self, manager):
        """[正常用例] 同步事件处理器被调用"""
        handler = MagicMock()

        manager.on("node_joined", handler)

        await manager._emit_event("node_joined", "test_node")

        handler.assert_called_once_with("test_node")

    @pytest.mark.asyncio
    async def test_event_handler_async_call(self, manager):
        """[正常用例] 异步事件处理器被调用"""
        async_handler = AsyncMock()

        manager.on("node_joined", async_handler)

        await manager._emit_event("node_joined", "test_node")

        async_handler.assert_called_once_with("test_node")

    @pytest.mark.asyncio
    async def test_event_handler_error_does_not_crash(self, manager):
        """[异常场景] 事件处理器错误不导致系统崩溃"""
        bad_handler = MagicMock(side_effect=Exception("Handler error"))
        good_handler = MagicMock()

        manager.on("node_joined", bad_handler)
        manager.on("node_joined", good_handler)

        # 不应抛出异常
        await manager._emit_event("node_joined", "test_node")

        good_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_event_handlers_same_event(self, manager):
        """[正常用例] 同一事件可注册多个处理器"""
        handler1 = MagicMock()
        handler2 = MagicMock()

        manager.on("node_joined", handler1)
        manager.on("node_joined", handler2)

        await manager._emit_event("node_joined", "test_node")

        handler1.assert_called_once()
        handler2.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
