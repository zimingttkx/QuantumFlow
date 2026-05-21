"""ClusterService 单元测试

严格测试集群管理服务的业务逻辑：
1. 请求验证
2. 节点注册/注销逻辑
3. 心跳处理
4. 节点列表过滤
"""

import uuid
from unittest.mock import MagicMock

import pytest

from quantumflow.grpc.generated import quantumflow_pb2
from quantumflow.grpc.services.cluster import ClusterServiceServicer


import grpc


class MockServicerContext:
    """Mock gRPC ServicerContext"""

    def __init__(self):
        self._aborted = False
        self._abort_code = None
        self._abort_message = None

    def abort(self, code, message):
        self._aborted = True
        self._abort_code = code
        self._abort_message = message
        raise Exception(f"abort: {code}, {message}")


class TestClusterServiceValidation:
    """ClusterService 请求验证测试"""

    @pytest.fixture
    def servicer(self):
        return ClusterServiceServicer()

    def test_rejects_empty_node_id_on_register(self, servicer):
        """拒绝空的 node_id（注册）"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="",
            host="192.168.1.100",
            port=8001,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.RegisterNode(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_host_on_register(self, servicer):
        """拒绝空的 host（注册）"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="",
            port=8001,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.RegisterNode(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_invalid_port_on_register(self, servicer):
        """拒绝无效端口（注册）"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=70000,  # 超出范围
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.RegisterNode(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_port_zero(self, servicer):
        """拒绝端口 0"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=0,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.RegisterNode(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_node_id_on_deregister(self, servicer):
        """拒绝空的 node_id（注销）"""
        request = quantumflow_pb2.DeregisterNodeRequest(
            node_id="",
            reason="Maintenance",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.DeregisterNode(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_node_id_on_heartbeat(self, servicer):
        """拒绝空的 node_id（心跳）"""
        request = quantumflow_pb2.HeartbeatRequest(
            node_id="",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Heartbeat(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT


class TestClusterServiceRegisterNode:
    """节点注册测试"""

    @pytest.fixture
    def servicer(self):
        mock_manager = MagicMock()
        mock_manager.register_node.return_value = "worker-001"
        return ClusterServiceServicer(cluster_manager=mock_manager)

    def test_register_success(self, servicer):
        """注册成功"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
        )
        context = MockServicerContext()

        response = servicer.RegisterNode(request, context)

        assert response.success is True
        assert response.assigned_id == "worker-001"

    def test_register_with_gpus(self, servicer):
        """带 GPU 信息注册"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
            gpus=[
                quantumflow_pb2.GPUInfo(
                    index=0,
                    name="NVIDIA RTX 4090",
                    memory=quantumflow_pb2.GPUMemory(
                        total_bytes=24 * 1024**3,
                    ),
                ),
            ],
        )
        context = MockServicerContext()

        response = servicer.RegisterNode(request, context)

        assert response.success is True

    def test_register_with_capabilities(self, servicer):
        """带 capabilities 注册"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
            capabilities={"gpu": "4x RTX 4090"},
        )
        context = MockServicerContext()

        response = servicer.RegisterNode(request, context)

        assert response.success is True

    def test_register_already_exists_error(self, servicer):
        """注册节点已存在错误"""
        from quantumflow.grpc.exceptions import AlreadyExistsError

        servicer.cluster_manager.register_node.side_effect = AlreadyExistsError(
            resource_type="Node",
            resource_id="worker-001",
        )

        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
        )
        context = MockServicerContext()

        response = servicer.RegisterNode(request, context)

        assert response.success is False
        assert "already exists" in response.message.lower()

    def test_register_generic_exception(self, servicer):
        """注册时通用异常"""
        servicer.cluster_manager.register_node.side_effect = RuntimeError("Database error")

        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
        )
        context = MockServicerContext()

        response = servicer.RegisterNode(request, context)

        assert response.success is False
        assert "Registration failed" in response.message


class TestClusterServiceDeregisterNode:
    """节点注销测试"""

    @pytest.fixture
    def servicer(self):
        mock_manager = MagicMock()
        return ClusterServiceServicer(cluster_manager=mock_manager)

    def test_deregister_success(self, servicer):
        """注销成功"""
        request = quantumflow_pb2.DeregisterNodeRequest(
            node_id="worker-001",
            reason="Maintenance",
        )
        context = MockServicerContext()

        response = servicer.DeregisterNode(request, context)

        assert response.success is True
        servicer.cluster_manager.deregister_node.assert_called_once()

    def test_deregister_empty_reason_allowed(self, servicer):
        """空的 reason 被允许"""
        request = quantumflow_pb2.DeregisterNodeRequest(
            node_id="worker-001",
            reason="",
        )
        context = MockServicerContext()

        response = servicer.DeregisterNode(request, context)

        assert response.success is True


class TestClusterServiceDeregisterNodeExceptions:
    """节点注销异常测试"""

    @pytest.fixture
    def servicer(self):
        mock_manager = MagicMock()
        return ClusterServiceServicer(cluster_manager=mock_manager)

    def test_deregister_node_not_found(self, servicer):
        """注销节点不存在"""
        from quantumflow.grpc.exceptions import NodeNotFoundError

        servicer.cluster_manager.deregister_node.side_effect = NodeNotFoundError("worker-001")

        request = quantumflow_pb2.DeregisterNodeRequest(
            node_id="worker-001",
            reason="Maintenance",
        )
        context = MockServicerContext()

        response = servicer.DeregisterNode(request, context)

        assert response.success is False

    def test_deregister_generic_exception(self, servicer):
        """注销时通用异常"""
        servicer.cluster_manager.deregister_node.side_effect = RuntimeError("Database error")

        request = quantumflow_pb2.DeregisterNodeRequest(
            node_id="worker-001",
            reason="Maintenance",
        )
        context = MockServicerContext()

        response = servicer.DeregisterNode(request, context)

        assert response.success is False
        assert "Deregistration failed" in response.message


class TestClusterServiceHeartbeat:
    """心跳测试"""

    @pytest.fixture
    def servicer(self):
        mock_manager = MagicMock()
        mock_manager.get_pending_tasks.return_value = ["task-1", "task-2"]
        return ClusterServiceServicer(cluster_manager=mock_manager)

    def test_heartbeat_success(self, servicer):
        """心跳成功"""
        request = quantumflow_pb2.HeartbeatRequest(
            node_id="worker-001",
        )
        context = MockServicerContext()

        response = servicer.Heartbeat(request, context)

        assert response.success is True
        assert response.server_time > 0
        assert len(response.pending_tasks) == 2

    def test_heartbeat_with_resources(self, servicer):
        """带资源信息的心跳"""
        request = quantumflow_pb2.HeartbeatRequest(
            node_id="worker-001",
            resources=quantumflow_pb2.NodeResources(
                node_id="worker-001",
                status=quantumflow_pb2.NODE_STATUS_BUSY,
            ),
        )
        context = MockServicerContext()

        response = servicer.Heartbeat(request, context)

        assert response.success is True


class TestClusterServiceHeartbeatExceptions:
    """心跳异常测试"""

    @pytest.fixture
    def servicer(self):
        mock_manager = MagicMock()
        mock_manager.get_pending_tasks.return_value = ["task-1"]
        return ClusterServiceServicer(cluster_manager=mock_manager)

    def test_heartbeat_node_not_found(self, servicer):
        """心跳节点不存在"""
        from quantumflow.grpc.exceptions import NodeNotFoundError

        servicer.cluster_manager.update_heartbeat.side_effect = NodeNotFoundError("worker-001")

        request = quantumflow_pb2.HeartbeatRequest(
            node_id="worker-001",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Heartbeat(request, context)

        assert context._abort_code == grpc.StatusCode.NOT_FOUND

    def test_heartbeat_generic_exception(self, servicer):
        """心跳时通用异常"""
        servicer.cluster_manager.update_heartbeat.side_effect = RuntimeError("Database error")

        request = quantumflow_pb2.HeartbeatRequest(
            node_id="worker-001",
        )
        context = MockServicerContext()

        response = servicer.Heartbeat(request, context)

        assert response.success is False


class TestClusterServiceListNodes:
    """节点列表测试"""

    @pytest.fixture
    def servicer(self):
        mock_manager = MagicMock()
        mock_manager.list_nodes.return_value = [
            {
                "node_id": "worker-001",
                "host": "192.168.1.100",
                "port": 8001,
                "status": quantumflow_pb2.NODE_STATUS_ACTIVE,
            },
            {
                "node_id": "worker-002",
                "host": "192.168.1.101",
                "port": 8001,
                "status": quantumflow_pb2.NODE_STATUS_IDLE,
            },
        ]
        return ClusterServiceServicer(cluster_manager=mock_manager)

    def test_list_all_nodes(self, servicer):
        """列出所有节点"""
        request = quantumflow_pb2.ListNodesRequest()
        context = MockServicerContext()

        response = servicer.ListNodes(request, context)

        assert len(response.nodes) == 2

    def test_list_filter_by_status(self, servicer):
        """按状态过滤"""
        request = quantumflow_pb2.ListNodesRequest(
            filter_status=quantumflow_pb2.NODE_STATUS_ACTIVE,
        )
        context = MockServicerContext()

        response = servicer.ListNodes(request, context)

        servicer.cluster_manager.list_nodes.assert_called_once()
        call_kwargs = servicer.cluster_manager.list_nodes.call_args[1]
        assert call_kwargs["status"] == quantumflow_pb2.NODE_STATUS_ACTIVE

    def test_list_filter_by_model(self, servicer):
        """按模型过滤"""
        request = quantumflow_pb2.ListNodesRequest(
            filter_model="llama-2-70b",
        )
        context = MockServicerContext()

        response = servicer.ListNodes(request, context)

        servicer.cluster_manager.list_nodes.assert_called_once()
        call_kwargs = servicer.cluster_manager.list_nodes.call_args[1]
        assert call_kwargs["model"] == "llama-2-70b"

    def test_list_generic_exception(self, servicer):
        """列表时通用异常"""
        servicer.cluster_manager.list_nodes.side_effect = RuntimeError("Database error")

        request = quantumflow_pb2.ListNodesRequest()
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.ListNodes(request, context)

        assert context._abort_code == grpc.StatusCode.INTERNAL


class TestClusterServiceSimulatedMode:
    """模拟模式测试（无 cluster_manager）"""

    @pytest.fixture
    def servicer(self):
        return ClusterServiceServicer()

    def test_register_returns_success_in_simulated_mode(self, servicer):
        """模拟模式下注册返回成功"""
        request = quantumflow_pb2.RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
        )
        context = MockServicerContext()

        response = servicer.RegisterNode(request, context)

        assert response.success is True
        assert response.assigned_id == "worker-001"

    def test_list_returns_simulated_nodes(self, servicer):
        """模拟模式下返回模拟节点"""
        request = quantumflow_pb2.ListNodesRequest(
            filter_status=quantumflow_pb2.NODE_STATUS_ACTIVE,
        )
        context = MockServicerContext()

        response = servicer.ListNodes(request, context)

        assert len(response.nodes) == 1
        assert response.nodes[0].node_id == "worker-001"
        assert response.nodes[0].status == quantumflow_pb2.NODE_STATUS_ACTIVE


class TestClusterServiceUpdateNodeResources:
    """UpdateNodeResources 测试"""

    @pytest.fixture
    def servicer(self):
        mock_manager = MagicMock()
        return ClusterServiceServicer(cluster_manager=mock_manager)

    def test_update_resources_stream(self, servicer):
        """流式更新资源"""
        resources_list = [
            quantumflow_pb2.NodeResources(
                node_id="worker-001",
                host="192.168.1.100",
                port=8001,
                status=quantumflow_pb2.NODE_STATUS_ACTIVE,
            ),
            quantumflow_pb2.NodeResources(
                node_id="worker-002",
                host="192.168.1.101",
                port=8001,
                status=quantumflow_pb2.NODE_STATUS_BUSY,
            ),
        ]

        request_iter = iter(resources_list)
        context = MockServicerContext()

        responses = list(servicer.UpdateNodeResources(request_iter, context))

        assert len(responses) == 2

    def test_update_resources_skips_empty_node_id(self, servicer):
        """跳过空 node_id 的资源"""
        resources_list = [
            quantumflow_pb2.NodeResources(
                node_id="",  # 空 node_id
                host="192.168.1.100",
                port=8001,
            ),
            quantumflow_pb2.NodeResources(
                node_id="worker-001",
                host="192.168.1.101",
                port=8001,
            ),
        ]

        request_iter = iter(resources_list)
        context = MockServicerContext()

        responses = list(servicer.UpdateNodeResources(request_iter, context))

        # 应该只有1个响应（跳过了空的）
        assert len(responses) == 1


class TestClusterServiceConvertMethods:
    """转换方法测试"""

    @pytest.fixture
    def servicer(self):
        return ClusterServiceServicer()

    def test_convert_to_node_resources_with_gpus(self, servicer):
        """_convert_to_node_resources 带 GPU 信息"""
        node = {
            "node_id": "worker-001",
            "host": "192.168.1.100",
            "port": 8001,
            "status": quantumflow_pb2.NODE_STATUS_ACTIVE,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA RTX 4090",
                    "utilization": 0.5,
                    "memory": {
                        "total": 24 * 1024**3,
                        "available": 12 * 1024**3,
                        "used": 12 * 1024**3,
                    },
                },
            ],
        }

        result = servicer._convert_to_node_resources(node)

        assert result.node_id == "worker-001"
        assert result.host == "192.168.1.100"
        assert result.port == 8001
        assert len(result.gpus) == 1
        assert result.gpus[0].index == 0
        assert result.gpus[0].name == "NVIDIA RTX 4090"

    def test_convert_to_node_resources_without_gpus(self, servicer):
        """_convert_to_node_resources 不带 GPU 信息"""
        node = {
            "node_id": "worker-001",
            "host": "192.168.1.100",
            "port": 8001,
            "status": quantumflow_pb2.NODE_STATUS_ACTIVE,
        }

        result = servicer._convert_to_node_resources(node)

        assert result.node_id == "worker-001"
        assert len(result.gpus) == 0

    def test_convert_to_node_resources_missing_fields(self, servicer):
        """_convert_to_node_resources 缺失字段"""
        node = {}

        result = servicer._convert_to_node_resources(node)

        assert result.node_id == ""
        assert result.host == ""
        assert result.port == 0
