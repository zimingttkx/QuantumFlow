"""Cluster API Routes 严格业务逻辑测试

严格按照以下标准编写：
1. 校验业务逻辑正确性，不只是运行可用性
2. 全覆盖：正常用例、边界值、错误输入、异常场景
3. 强精准断言，明确预期值
4. 重点测试核心功能、易错细节、状态变更
"""

import sys
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "/home/dingziming/PycharmProjects/QuantumFlow")

from quantumflow.api.routes.cluster import _node_to_node_info, router
from quantumflow.cluster import ClusterManager, Node, NodeStatus, set_cluster_manager
from quantumflow.core.constants import NodeStatus as NodeStatusEnum


class TestNodeConversion:
    """Node 到 NodeInfo 转换逻辑测试"""

    def test_node_to_node_info_complete(self):
        """[正常用例] 完整 Node 转换为 NodeInfo"""
        from quantumflow.cluster.manager import GPUInfo

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
            version="1.1.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=["model-a", "model-b"],
        )

        node_info = _node_to_node_info(node)

        assert node_info.node_id == "node-1"
        assert node_info.hostname == "worker-1"
        assert node_info.ip == "192.168.1.100"
        assert node_info.port == 8080
        assert node_info.status == "healthy"
        assert node_info.gpu_count == 1
        assert node_info.cpu_count == 32
        assert node_info.memory_total == 128 * 1024**3
        assert node_info.memory_available == 64 * 1024**3
        assert node_info.disk_total == 1 * 1024**4
        assert node_info.disk_available == 512 * 1024**3
        assert node_info.current_load == 0.3
        assert node_info.labels == {"zone": "us-east", "env": "prod"}
        assert node_info.version == "1.1.0"
        assert node_info.loaded_models == ["model-a", "model-b"]

        # 验证 GPU 信息转换
        assert len(node_info.gpu_info) == 1
        assert node_info.gpu_info[0].gpu_id == 0
        assert node_info.gpu_info[0].name == "NVIDIA A100"
        assert node_info.gpu_info[0].memory_total == 80 * 1024**3
        assert node_info.gpu_info[0].memory_used == 40 * 1024**3
        # memory_free 应该 = memory_total - memory_used
        assert node_info.gpu_info[0].memory_free == 40 * 1024**3
        assert node_info.gpu_info[0].utilization == 0.5
        assert node_info.gpu_info[0].temperature == 65.0

    def test_node_to_node_info_empty_gpu_list(self):
        """[边界用例] 空 GPU 列表转换"""
        node = Node(
            node_id="node-no-gpu",
            hostname="nogpu",
            ip="127.0.0.1",
            port=8000,
            gpu_count=0,
            gpu_info=[],
            status=NodeStatus.HEALTHY,
        )

        node_info = _node_to_node_info(node)

        assert node_info.gpu_count == 0
        assert node_info.gpu_info == []  # 空列表返回空列表，不是 None

    def test_node_to_node_info_different_status(self):
        """[多分支逻辑] 不同状态的 Node 转换"""
        statuses = [
            (NodeStatus.HEALTHY, "healthy"),
            (NodeStatus.UNHEALTHY, "unhealthy"),
            (NodeStatus.DRAINING, "draining"),
            (NodeStatus.OFFLINE, "offline"),
            (NodeStatus.INITIALIZING, "initializing"),
        ]

        for node_status, expected_value in statuses:
            node = Node(
                node_id=f"node-{expected_value}",
                hostname="test",
                ip="127.0.0.1",
                port=8000,
                gpu_count=0,
                gpu_info=[],
                status=node_status,
            )
            node_info = _node_to_node_info(node)
            assert (
                node_info.status == expected_value
            ), f"Expected {expected_value}, got {node_info.status}"


class TestClusterStatusEndpoint:
    """GET /cluster/status 端点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟 ClusterManager"""
        manager = AsyncMock(spec=ClusterManager)
        set_cluster_manager(manager)
        yield manager
        # 清理
        set_cluster_manager(None)

    @pytest.fixture
    def client(self, mock_cluster_manager):
        """创建测试客户端"""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_status_empty_cluster(self, client, mock_cluster_manager):
        """[正常用例] 空集群状态"""
        mock_cluster_manager.get_nodes = AsyncMock(return_value=[])

        response = client.get("/cluster/status")

        assert response.status_code == 200
        data = response.json()
        assert data["total_nodes"] == 0
        assert data["healthy_nodes"] == 0
        assert data["unhealthy_nodes"] == 0
        assert data["draining_nodes"] == 0
        assert data["total_gpus"] == 0
        assert data["available_gpus"] == 0
        assert data["active_models"] == 0
        assert data["pending_jobs"] == 0
        assert data["running_jobs"] == 0
        assert "system_metrics" in data
        assert "uptime_seconds" in data

    def test_status_with_healthy_nodes(self, client, mock_cluster_manager):
        """[正常用例] 有健康节点的集群状态"""
        from quantumflow.cluster.manager import GPUInfo

        gpu = GPUInfo(
            gpu_id=0,
            name="A100",
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
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=["model-a"],
        )
        mock_cluster_manager.get_nodes = AsyncMock(return_value=[node])

        response = client.get("/cluster/status")

        assert response.status_code == 200
        data = response.json()
        assert data["total_nodes"] == 1
        assert data["healthy_nodes"] == 1
        assert data["unhealthy_nodes"] == 0
        assert data["total_gpus"] == 1
        assert data["active_models"] == 1

    def test_status_with_mixed_nodes(self, client, mock_cluster_manager):
        """[正常用例] 混合状态节点的集群状态"""
        from quantumflow.cluster.manager import GPUInfo

        gpu = GPUInfo(
            gpu_id=0,
            name="A100",
            memory_total=80 * 1024**3,
            memory_used=40 * 1024**3,
            utilization=0.5,
            temperature=65.0,
        )
        healthy_node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[gpu],
            status=NodeStatus.HEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        unhealthy_node = Node(
            node_id="node-2",
            hostname="worker-2",
            ip="192.168.1.101",
            port=8081,
            gpu_count=2,
            gpu_info=[gpu, gpu],
            status=NodeStatus.UNHEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=64,
            memory_total=256 * 1024**3,
            memory_available=128 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            current_load=0.8,
            loaded_models=["model-b"],
        )
        mock_cluster_manager.get_nodes = AsyncMock(return_value=[healthy_node, unhealthy_node])

        response = client.get("/cluster/status")

        assert response.status_code == 200
        data = response.json()
        assert data["total_nodes"] == 2
        assert data["healthy_nodes"] == 1
        assert data["unhealthy_nodes"] == 1
        assert data["total_gpus"] == 3


class TestListNodesEndpoint:
    """GET /cluster/nodes 端点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟 ClusterManager"""
        manager = AsyncMock(spec=ClusterManager)
        set_cluster_manager(manager)
        yield manager
        set_cluster_manager(None)

    @pytest.fixture
    def client(self, mock_cluster_manager):
        """创建测试客户端"""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_list_nodes_empty(self, client, mock_cluster_manager):
        """[正常用例] 列出空节点列表"""
        mock_cluster_manager.get_nodes = AsyncMock(return_value=[])

        response = client.get("/cluster/nodes")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_nodes_multiple(self, client, mock_cluster_manager):
        """[正常用例] 列出多个节点"""
        from quantumflow.cluster.manager import GPUInfo

        gpu = GPUInfo(
            gpu_id=0,
            name="A100",
            memory_total=80 * 1024**3,
            memory_used=40 * 1024**3,
            utilization=0.5,
            temperature=65.0,
        )
        nodes = [
            Node(
                node_id=f"node-{i}",
                hostname=f"worker-{i}",
                ip=f"192.168.1.{100+i}",
                port=8080 + i,
                gpu_count=1,
                gpu_info=[gpu],
                status=NodeStatus.HEALTHY,
                labels={},
                version="1.0.0",
                cpu_count=32,
                memory_total=128 * 1024**3,
                memory_available=64 * 1024**3,
                disk_total=1 * 1024**4,
                disk_available=512 * 1024**3,
                current_load=0.3,
                loaded_models=[],
            )
            for i in range(3)
        ]
        mock_cluster_manager.get_nodes = AsyncMock(return_value=nodes)

        response = client.get("/cluster/nodes")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["node_id"] == "node-0"
        assert data[1]["node_id"] == "node-1"
        assert data[2]["node_id"] == "node-2"

    def test_list_nodes_filter_by_status(self, client, mock_cluster_manager):
        """[正常用例] 按状态过滤节点"""
        from quantumflow.cluster.manager import GPUInfo

        gpu = GPUInfo(
            gpu_id=0,
            name="A100",
            memory_total=80 * 1024**3,
            memory_used=40 * 1024**3,
            utilization=0.5,
            temperature=65.0,
        )
        healthy_node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[gpu],
            status=NodeStatus.HEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        mock_cluster_manager.get_nodes = AsyncMock(return_value=[healthy_node])

        response = client.get("/cluster/nodes?status_filter=healthy")

        assert response.status_code == 200
        mock_cluster_manager.get_nodes.assert_called_once_with(status=NodeStatusEnum.HEALTHY)

    def test_list_nodes_filter_by_zone(self, client, mock_cluster_manager):
        """[正常用例] 按可用区标签过滤节点"""
        from quantumflow.cluster.manager import GPUInfo

        gpu = GPUInfo(
            gpu_id=0,
            name="A100",
            memory_total=80 * 1024**3,
            memory_used=40 * 1024**3,
            utilization=0.5,
            temperature=65.0,
        )
        us_east_node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[gpu],
            status=NodeStatus.HEALTHY,
            labels={"zone": "us-east"},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        us_west_node = Node(
            node_id="node-2",
            hostname="worker-2",
            ip="192.168.1.101",
            port=8081,
            gpu_count=1,
            gpu_info=[gpu],
            status=NodeStatus.HEALTHY,
            labels={"zone": "us-west"},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        mock_cluster_manager.get_nodes = AsyncMock(return_value=[us_east_node, us_west_node])

        response = client.get("/cluster/nodes?zone=us-east")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["node_id"] == "node-1"


class TestGetNodeEndpoint:
    """GET /cluster/nodes/{node_id} 端点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟 ClusterManager"""
        manager = AsyncMock(spec=ClusterManager)
        set_cluster_manager(manager)
        yield manager
        set_cluster_manager(None)

    @pytest.fixture
    def client(self, mock_cluster_manager):
        """创建测试客户端"""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_get_existing_node(self, client, mock_cluster_manager):
        """[正常用例] 获取存在的节点"""
        from quantumflow.cluster.manager import GPUInfo

        gpu = GPUInfo(
            gpu_id=0,
            name="A100",
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
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=["model-a"],
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        response = client.get("/cluster/nodes/node-1")

        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "node-1"
        assert data["hostname"] == "worker-1"
        assert data["ip"] == "192.168.1.100"
        assert data["port"] == 8080
        assert data["status"] == "healthy"
        assert data["gpu_count"] == 1
        assert data["loaded_models"] == ["model-a"]

    def test_get_nonexistent_node(self, client, mock_cluster_manager):
        """[错误处理] 获取不存在的节点返回 404"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        response = client.get("/cluster/nodes/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"]["code"] == "NODE_NOT_FOUND"
        assert "nonexistent" in data["detail"]["error"]["message"]


class TestHeartbeatEndpoint:
    """POST /cluster/heartbeat 端点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟 ClusterManager"""
        manager = AsyncMock(spec=ClusterManager)
        set_cluster_manager(manager)
        yield manager
        set_cluster_manager(None)

    @pytest.fixture
    def client(self, mock_cluster_manager):
        """创建测试客户端"""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_heartbeat_missing_node_id(self, client, mock_cluster_manager):
        """[错误处理] 缺少 node_id 返回 400"""
        response = client.post("/cluster/heartbeat", json={})

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"]["code"] == "INVALID_REQUEST"

    def test_heartbeat_register_new_node(self, client, mock_cluster_manager):
        """[正常用例] 注册新节点"""
        from quantumflow.cluster.manager import GPUInfo

        gpu = GPUInfo(
            gpu_id=0,
            name="A100",
            memory_total=80 * 1024**3,
            memory_used=40 * 1024**3,
            utilization=0.5,
            temperature=65.0,
        )
        registered_node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[gpu],
            status=NodeStatus.HEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=None)
        mock_cluster_manager.register_node = AsyncMock(return_value=registered_node)

        node_info = {
            "node_id": "node-1",
            "hostname": "worker-1",
            "ip": "192.168.1.100",
            "port": 8080,
            "gpu_count": 1,
            "gpu_info": [
                {
                    "gpu_id": 0,
                    "name": "A100",
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

        response = client.post("/cluster/heartbeat", json=node_info)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["node_id"] == "node-1"
        mock_cluster_manager.register_node.assert_called_once()

    def test_heartbeat_update_existing_node(self, client, mock_cluster_manager):
        """[正常用例] 更新已存在节点"""

        existing_node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[],
            status=NodeStatus.HEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=["model-a"],
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=existing_node)
        mock_cluster_manager.update_node_info = AsyncMock()
        mock_cluster_manager.add_loaded_model = AsyncMock()

        node_info = {
            "node_id": "node-1",
            "hostname": "worker-1",
            "ip": "192.168.1.100",
            "port": 8080,
            "gpu_count": 1,
            "gpu_info": [
                {
                    "gpu_id": 0,
                    "name": "A100",
                    "memory_total": 80 * 1024**3,
                    "memory_used": 50 * 1024**3,
                    "utilization": 0.6,
                    "temperature": 70.0,
                }
            ],
            "current_load": 0.5,
            "loaded_models": ["model-a", "model-b"],
        }

        response = client.post("/cluster/heartbeat", json=node_info)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["node_id"] == "node-1"

        # 验证更新被调用
        mock_cluster_manager.update_node_info.assert_called_once()
        # model-a 已存在，不应调用 add_loaded_model
        mock_cluster_manager.add_loaded_model.assert_called_once_with("node-1", "model-b")


class TestNodeActionEndpoint:
    """POST /cluster/nodes/{node_id}/action 端点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟 ClusterManager"""
        manager = AsyncMock(spec=ClusterManager)
        set_cluster_manager(manager)
        yield manager
        set_cluster_manager(None)

    @pytest.fixture
    def client(self, mock_cluster_manager):
        """创建测试客户端"""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_action_drain_existing_node(self, client, mock_cluster_manager):
        """[正常用例] drain 操作"""

        node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[],
            status=NodeStatus.HEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_node_status = AsyncMock(return_value=True)

        response = client.post("/cluster/nodes/node-1/action?action=drain")

        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "node-1"
        assert data["action"] == "drain"
        assert data["status"] == "completed"
        mock_cluster_manager.update_node_status.assert_called_once_with(
            "node-1", NodeStatusEnum.DRAINING
        )

    def test_action_uncordon_existing_node(self, client, mock_cluster_manager):
        """[正常用例] uncordon 操作"""

        node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[],
            status=NodeStatus.DRAINING,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.update_node_status = AsyncMock(return_value=True)

        response = client.post("/cluster/nodes/node-1/action?action=uncordon")

        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "node-1"
        assert data["action"] == "uncordon"
        mock_cluster_manager.update_node_status.assert_called_once_with(
            "node-1", NodeStatusEnum.HEALTHY
        )

    def test_action_restart_existing_node(self, client, mock_cluster_manager):
        """[正常用例] restart 操作（只返回成功）"""

        node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[],
            status=NodeStatus.HEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        response = client.post("/cluster/nodes/node-1/action?action=restart")

        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "node-1"
        assert data["action"] == "restart"
        # restart 不调用 update_node_status
        mock_cluster_manager.update_node_status.assert_not_called()

    def test_action_invalid_action(self, client, mock_cluster_manager):
        """[错误处理] 无效操作返回 400"""

        node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[],
            status=NodeStatus.HEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=node)

        response = client.post("/cluster/nodes/node-1/action?action=invalid")

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"]["code"] == "INVALID_ACTION"
        assert "invalid" in data["detail"]["error"]["message"]

    def test_action_nonexistent_node(self, client, mock_cluster_manager):
        """[错误处理] 不存在节点返回 404"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        response = client.post("/cluster/nodes/nonexistent/action?action=drain")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"]["code"] == "NODE_NOT_FOUND"


class TestUnregisterNodeEndpoint:
    """DELETE /cluster/nodes/{node_id} 端点测试"""

    @pytest.fixture
    def mock_cluster_manager(self):
        """创建模拟 ClusterManager"""
        manager = AsyncMock(spec=ClusterManager)
        set_cluster_manager(manager)
        yield manager
        set_cluster_manager(None)

    @pytest.fixture
    def client(self, mock_cluster_manager):
        """创建测试客户端"""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_unregister_existing_node(self, client, mock_cluster_manager):
        """[正常用例] 注销存在的节点"""

        node = Node(
            node_id="node-1",
            hostname="worker-1",
            ip="192.168.1.100",
            port=8080,
            gpu_count=1,
            gpu_info=[],
            status=NodeStatus.HEALTHY,
            labels={},
            version="1.0.0",
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=1 * 1024**4,
            disk_available=512 * 1024**3,
            current_load=0.3,
            loaded_models=[],
        )
        mock_cluster_manager.get_node = AsyncMock(return_value=node)
        mock_cluster_manager.unregister_node = AsyncMock()

        response = client.delete("/cluster/nodes/node-1")

        assert response.status_code == 200
        data = response.json()
        assert data["node_id"] == "node-1"
        assert data["status"] == "unregistered"
        mock_cluster_manager.unregister_node.assert_called_once_with("node-1")

    def test_unregister_nonexistent_node(self, client, mock_cluster_manager):
        """[错误处理] 注销不存在的节点返回 404"""
        mock_cluster_manager.get_node = AsyncMock(return_value=None)

        response = client.delete("/cluster/nodes/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"]["code"] == "NODE_NOT_FOUND"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
