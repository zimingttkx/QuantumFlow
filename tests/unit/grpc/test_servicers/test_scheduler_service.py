"""SchedulerService 单元测试

严格测试调度服务的业务逻辑：
1. 请求验证
2. 调度提交逻辑
3. 取消请求
4. 状态查询
"""

import uuid
from unittest.mock import MagicMock

import pytest

from quantumflow.grpc.generated import quantumflow_pb2
from quantumflow.grpc.services.scheduler import SchedulerServiceServicer


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


class TestSchedulerServiceValidation:
    """SchedulerService 请求验证测试"""

    @pytest.fixture
    def servicer(self):
        return SchedulerServiceServicer()

    def test_rejects_empty_request_id(self, servicer):
        """拒绝空的 request_id（提交）"""
        request = quantumflow_pb2.SchedulingRequest(
            request_id="",
            model_name="llama-2-7b",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.SubmitRequest(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_model_name(self, servicer):
        """拒绝空的 model_name（提交）"""
        request = quantumflow_pb2.SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="",
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.SubmitRequest(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_invalid_priority(self, servicer):
        """拒绝无效的优先级"""
        request = quantumflow_pb2.SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            priority=15,  # 超出 [0, 10]
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.SubmitRequest(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_negative_priority(self, servicer):
        """拒绝负数优先级"""
        request = quantumflow_pb2.SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            priority=-1,
        )
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.SubmitRequest(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_request_id_on_cancel(self, servicer):
        """拒绝空的 request_id（取消）"""
        request = quantumflow_pb2.CancelRequest(request_id="")
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.Cancel(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_request_id_on_get_status(self, servicer):
        """拒绝空的 request_id（获取状态）"""
        request = quantumflow_pb2.GetSchedulingStatusRequest(request_id="")
        context = MockServicerContext()

        with pytest.raises(Exception) as exc_info:
            servicer.GetStatus(request, context)

        assert context._abort_code == grpc.StatusCode.INVALID_ARGUMENT


class TestSchedulerServiceSubmit:
    """调度提交测试"""

    @pytest.fixture
    def servicer(self):
        mock_scheduler = MagicMock()
        return SchedulerServiceServicer(scheduler=mock_scheduler)

    def test_submit_success(self, servicer):
        """提交成功"""
        mock_response = quantumflow_pb2.SchedulingResponse(
            request_id="req-123",
            scheduled=True,
            assigned_node_id="worker-001",
            status=quantumflow_pb2.STATUS_PENDING,
        )
        servicer.scheduler.schedule.return_value = mock_response

        request = quantumflow_pb2.SchedulingRequest(
            request_id="req-123",
            model_name="llama-2-70b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
            tensor_parallel_size=4,
            gpu_memory_required_gb=80,
            priority=5,
        )
        context = MockServicerContext()

        response = servicer.SubmitRequest(request, context)

        assert response.scheduled is True
        assert response.assigned_node_id == "worker-001"

    def test_submit_passes_correct_parameters(self, servicer):
        """传递正确的参数"""
        request = quantumflow_pb2.SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-70b",
            backend=quantumflow_pb2.MODEL_BACKEND_VLLM,
            tensor_parallel_size=8,
            gpu_memory_required_gb=160,
            priority=10,
            mode=quantumflow_pb2.INFERENCE_MODE_GENERATE,
        )
        context = MockServicerContext()

        servicer.SubmitRequest(request, context)

        servicer.scheduler.schedule.assert_called_once()
        call_kwargs = servicer.scheduler.schedule.call_args[1]
        assert call_kwargs["model_name"] == "llama-2-70b"
        assert call_kwargs["tensor_parallel_size"] == 8
        assert call_kwargs["priority"] == 10

    def test_submit_returns_error_when_resources_unavailable(self, servicer):
        """资源不足时返回错误"""
        from quantumflow.grpc.exceptions import ResourceUnavailableError

        servicer.scheduler.schedule.side_effect = ResourceUnavailableError(
            resource="GPU memory",
            required=80,
            available=40,
        )

        request = quantumflow_pb2.SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-70b",
        )
        context = MockServicerContext()

        response = servicer.SubmitRequest(request, context)

        assert response.scheduled is False
        assert response.status == quantumflow_pb2.STATUS_ERROR

    def test_submit_returns_error_when_scheduling_error(self, servicer):
        """调度错误时返回错误"""
        from quantumflow.grpc.exceptions import SchedulingError

        servicer.scheduler.schedule.side_effect = SchedulingError(
            reason="No available nodes",
        )

        request = quantumflow_pb2.SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-70b",
        )
        context = MockServicerContext()

        response = servicer.SubmitRequest(request, context)

        assert response.scheduled is False
        assert response.status == quantumflow_pb2.STATUS_ERROR

    def test_submit_returns_error_when_generic_exception(self, servicer):
        """通用异常时返回错误"""
        servicer.scheduler.schedule.side_effect = RuntimeError("Unexpected error")

        request = quantumflow_pb2.SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-70b",
        )
        context = MockServicerContext()

        response = servicer.SubmitRequest(request, context)

        assert response.scheduled is False
        assert response.status == quantumflow_pb2.STATUS_ERROR
        assert "Scheduling failed" in response.error_message


class TestSchedulerServiceCancel:
    """取消请求测试"""

    @pytest.fixture
    def servicer(self):
        mock_scheduler = MagicMock()
        return SchedulerServiceServicer(scheduler=mock_scheduler)

    def test_cancel_success(self, servicer):
        """取消成功"""
        servicer.scheduler.cancel.return_value = True

        request = quantumflow_pb2.CancelRequest(request_id="req-123")
        context = MockServicerContext()

        response = servicer.Cancel(request, context)

        assert response.success is True

    def test_cancel_failure_when_not_found(self, servicer):
        """请求不存在时取消失败"""
        servicer.scheduler.cancel.return_value = False

        request = quantumflow_pb2.CancelRequest(request_id="nonexistent")
        context = MockServicerContext()

        response = servicer.Cancel(request, context)

        assert response.success is False

    def test_cancel_exception(self, servicer):
        """取消时发生异常"""
        servicer.scheduler.cancel.side_effect = RuntimeError("Unexpected error")

        request = quantumflow_pb2.CancelRequest(request_id="req-123")
        context = MockServicerContext()

        response = servicer.Cancel(request, context)

        assert response.success is False
        assert "Cancellation failed" in response.message


class TestSchedulerServiceGetStatus:
    """获取状态测试"""

    @pytest.fixture
    def servicer(self):
        mock_scheduler = MagicMock()
        return SchedulerServiceServicer(scheduler=mock_scheduler)

    def test_get_status_pending(self, servicer):
        """获取等待中状态"""
        mock_response = quantumflow_pb2.GetSchedulingStatusResponse(
            request_id="req-123",
            status=quantumflow_pb2.STATUS_PENDING,
        )
        servicer.scheduler.get_status.return_value = mock_response

        request = quantumflow_pb2.GetSchedulingStatusRequest(request_id="req-123")
        context = MockServicerContext()

        response = servicer.GetStatus(request, context)

        assert response.status == quantumflow_pb2.STATUS_PENDING

    def test_get_status_with_result(self, servicer):
        """获取完成状态带结果"""
        mock_response = quantumflow_pb2.GetSchedulingStatusResponse(
            request_id="req-123",
            status=quantumflow_pb2.STATUS_SUCCESS,
            result=quantumflow_pb2.InferenceResponse(
                text="Generated text",
                tokens_generated=10,
            ),
        )
        servicer.scheduler.get_status.return_value = mock_response

        request = quantumflow_pb2.GetSchedulingStatusRequest(request_id="req-123")
        context = MockServicerContext()

        response = servicer.GetStatus(request, context)

        assert response.status == quantumflow_pb2.STATUS_SUCCESS
        assert response.HasField("result")

    def test_get_status_error(self, servicer):
        """获取错误状态"""
        mock_response = quantumflow_pb2.GetSchedulingStatusResponse(
            request_id="req-123",
            status=quantumflow_pb2.STATUS_ERROR,
            error_message="GPU OOM",
        )
        servicer.scheduler.get_status.return_value = mock_response

        request = quantumflow_pb2.GetSchedulingStatusRequest(request_id="req-123")
        context = MockServicerContext()

        response = servicer.GetStatus(request, context)

        assert response.status == quantumflow_pb2.STATUS_ERROR
        assert "OOM" in response.error_message

    def test_get_status_exception(self, servicer):
        """获取状态时发生异常"""
        servicer.scheduler.get_status.side_effect = RuntimeError("Unexpected error")

        request = quantumflow_pb2.GetSchedulingStatusRequest(request_id="req-123")
        context = MockServicerContext()

        response = servicer.GetStatus(request, context)

        assert response.status == quantumflow_pb2.STATUS_ERROR
        assert "Failed to get status" in response.error_message


class TestSchedulerServiceSimulatedMode:
    """模拟模式测试"""

    @pytest.fixture
    def servicer(self):
        return SchedulerServiceServicer()

    def test_submit_returns_simulated_success(self, servicer):
        """模拟模式提交返回成功"""
        request = quantumflow_pb2.SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-70b",
        )
        context = MockServicerContext()

        response = servicer.SubmitRequest(request, context)

        assert response.scheduled is True
        assert response.assigned_node_id == "worker-001"

    def test_cancel_returns_success_in_simulated_mode(self, servicer):
        """模拟模式取消返回成功"""
        request = quantumflow_pb2.CancelRequest(request_id="req-123")
        context = MockServicerContext()

        response = servicer.Cancel(request, context)

        assert response.success is True

    def test_get_status_returns_pending_in_simulated_mode(self, servicer):
        """模拟模式获取状态返回 pending"""
        request = quantumflow_pb2.GetSchedulingStatusRequest(request_id="req-123")
        context = MockServicerContext()

        response = servicer.GetStatus(request, context)

        assert response.status == quantumflow_pb2.STATUS_PENDING
