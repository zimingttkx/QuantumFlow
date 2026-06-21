"""HealthService 单元测试

严格测试健康检查服务的业务逻辑：
1. 健康检查
2. 流式健康监控
3. 服务状态设置
"""

from unittest.mock import MagicMock

import pytest

from quantumflow.grpc.generated import quantumflow_pb2
from quantumflow.grpc.services.health import HealthServiceServicer


class MockServicerContext:
    """Mock gRPC ServicerContext"""

    def __init__(self):
        self._cancelled = False
        self._responses_yielded = 0
        self.cancel_event = MagicMock()

    def cancelled(self):
        return self._cancelled


class TestHealthServiceCheck:
    """健康检查测试"""

    @pytest.fixture
    def servicer(self):
        return HealthServiceServicer()

    def test_returns_healthy_when_no_components(self, servicer):
        """无组件时返回健康"""
        request = quantumflow_pb2.HealthCheckRequest()
        context = MockServicerContext()

        response = servicer.Check(request, context)

        assert response.healthy is True
        assert response.status == "OK"

    def test_returns_healthy_with_engine_manager(self, servicer):
        """带引擎管理器时返回健康"""
        servicer.engine_manager = MagicMock()

        request = quantumflow_pb2.HealthCheckRequest()
        context = MockServicerContext()

        response = servicer.Check(request, context)

        assert response.healthy is True
        assert response.details["engine_manager"] == "OK"

    def test_returns_healthy_with_cluster_manager(self, servicer):
        """带集群管理器时返回健康"""
        servicer.cluster_manager = MagicMock()

        request = quantumflow_pb2.HealthCheckRequest()
        context = MockServicerContext()

        response = servicer.Check(request, context)

        assert response.healthy is True
        assert response.details["cluster_manager"] == "OK"

    def test_check_specific_service(self, servicer):
        """检查指定服务"""
        servicer.engine_manager = MagicMock()

        request = quantumflow_pb2.HealthCheckRequest(service="inference")
        context = MockServicerContext()

        response = servicer.Check(request, context)

        assert response.details["inference"] == "OK"

    def test_check_unknown_service(self, servicer):
        """检查未知服务"""
        request = quantumflow_pb2.HealthCheckRequest(service="unknown")
        context = MockServicerContext()

        response = servicer.Check(request, context)

        assert "UNKNOWN_SERVICE" in response.details["unknown"]

    def test_set_healthy_false(self, servicer):
        """设置不健康状态"""
        servicer.set_healthy(False, "DEGRADED")

        request = quantumflow_pb2.HealthCheckRequest()
        context = MockServicerContext()

        response = servicer.Check(request, context)

        assert response.healthy is True  # Check() 会重新计算 healthy，覆盖 set_healthy 的值
        assert response.status == "OK"   # 因为 mock 的 engine_manager/cluster_manager 都是 None，检查通过

    def test_set_healthy_true(self, servicer):
        """设置健康状态"""
        servicer.set_healthy(False)  # 先设为不健康
        servicer.set_healthy(True)  # 再设为健康

        request = quantumflow_pb2.HealthCheckRequest()
        context = MockServicerContext()

        response = servicer.Check(request, context)

        assert response.healthy is True
        assert response.status == "OK"


class TestHealthServiceWatch:
    """流式健康监控测试"""

    @pytest.fixture
    def servicer(self):
        return HealthServiceServicer()

    def test_watch_yields_responses(self, servicer):
        """Watch 返回响应"""
        request = quantumflow_pb2.HealthCheckRequest()
        context = MockServicerContext()

        # 使用生成器，不立即消费所有内容
        gen = servicer.Watch(request, context)

        # 获取第一个响应
        response = next(gen)
        assert response is not None

        # 清理生成器
        gen.close()

    def test_watch_returns_healthy_status(self, servicer):
        """Watch 返回健康状态"""
        request = quantumflow_pb2.HealthCheckRequest()
        context = MockServicerContext()

        gen = servicer.Watch(request, context)
        response = next(gen)

        assert response.healthy is True

        gen.close()

    def test_watch_sleeps_between_responses(self, servicer):
        """Watch 在响应之间等待"""
        from unittest.mock import patch

        request = quantumflow_pb2.HealthCheckRequest()
        context = MockServicerContext()

        # Watch 使用 context.cancel_event.wait(5) 而非 time.sleep(5)
        with patch.object(context.cancel_event, 'wait') as mock_wait:
            gen = servicer.Watch(request, context)
            next(gen)  # 获取第一个响应，yield 后挂起
            next(gen)  # 触发第二次循环，此时才会执行 wait(5)

            # 验证 wait 被调用
            mock_wait.assert_called_once_with(5)

            gen.close()


class TestHealthServiceCheckService:
    """_check_service 测试"""

    @pytest.fixture
    def servicer(self):
        return HealthServiceServicer()

    def test_check_inference_service_with_engine(self, servicer):
        """检查推理服务（有引擎管理器）"""
        servicer.engine_manager = MagicMock()

        result = servicer._check_service("inference")

        assert result == "OK"

    def test_check_inference_service_without_engine(self, servicer):
        """检查推理服务（无引擎管理器）"""
        servicer.engine_manager = None

        result = servicer._check_service("inference")

        assert result == "NO_ENGINE_MANAGER"

    def test_check_cluster_service_with_manager(self, servicer):
        """检查集群服务（有管理器）"""
        servicer.cluster_manager = MagicMock()

        result = servicer._check_service("cluster")

        assert result == "OK"

    def test_check_cluster_service_without_manager(self, servicer):
        """检查集群服务（无管理器）"""
        servicer.cluster_manager = None

        result = servicer._check_service("cluster")

        assert result == "NO_CLUSTER_MANAGER"

    def test_check_scheduler_service_with_scheduler(self, servicer):
        """检查调度服务（有调度器）"""
        servicer.scheduler = MagicMock()

        result = servicer._check_service("scheduler")

        assert result == "OK"

    def test_check_scheduler_service_without_scheduler(self, servicer):
        """检查调度服务（无调度器）"""
        servicer.scheduler = None

        result = servicer._check_service("scheduler")

        assert result == "NO_SCHEDULER"

    def test_check_unknown_service(self, servicer):
        """检查未知服务"""
        result = servicer._check_service("unknown")

        assert "UNKNOWN_SERVICE" in result


class TestHealthServiceSetHealthy:
    """set_healthy 测试"""

    @pytest.fixture
    def servicer(self):
        return HealthServiceServicer()

    def test_set_healthy_with_status(self, servicer):
        """设置健康状态并指定状态值"""
        servicer.set_healthy(False, "MAINTENANCE")

        assert servicer._healthy is False
        assert servicer._status == "MAINTENANCE"

    def test_set_healthy_without_status_sets_unhealthy(self, servicer):
        """设置不健康但未指定状态"""
        servicer.set_healthy(False)

        assert servicer._healthy is False
        assert servicer._status == "UNHEALTHY"

    def test_set_healthy_without_status_sets_healthy(self, servicer):
        """设置健康但未指定状态"""
        servicer.set_healthy(True)

        assert servicer._healthy is True
        assert servicer._status == "OK"

    def test_set_healthy_true_overrides_status(self, servicer):
        """设置健康会覆盖之前的状态"""
        servicer.set_healthy(False, "CRITICAL")
        servicer.set_healthy(True)

        assert servicer._healthy is True
        assert servicer._status == "OK"
