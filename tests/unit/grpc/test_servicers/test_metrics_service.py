"""MetricsService 单元测试

严格测试指标服务的业务逻辑：
1. 指标获取
2. 流式指标
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from quantumflow.grpc.generated import quantumflow_pb2
from quantumflow.grpc.services.metrics import MetricsServiceServicer


class MockServicerContext:
    """Mock gRPC ServicerContext"""

    def __init__(self):
        self._cancelled = False

    def cancelled(self):
        return self._cancelled


class TestMetricsServiceGetMetrics:
    """GetMetrics 测试"""

    @pytest.fixture
    def servicer(self):
        return MetricsServiceServicer()

    def test_get_metrics_returns_all_when_no_filter(self, servicer):
        """无过滤条件时返回所有指标"""
        request = quantumflow_pb2.MetricsRequest()
        context = MockServicerContext()

        response = servicer.GetMetrics(request, context)

        assert len(response.metrics) == 4
        metric_names = [m.name for m in response.metrics]
        assert "requests_total" in metric_names
        assert "gpu_memory_usage" in metric_names
        assert "active_inferences" in metric_names
        assert "node_count" in metric_names

    def test_get_metrics_filters_by_name(self, servicer):
        """按名称过滤指标"""
        request = quantumflow_pb2.MetricsRequest(metric_names=["requests_total"])
        context = MockServicerContext()

        response = servicer.GetMetrics(request, context)

        assert len(response.metrics) == 1
        assert response.metrics[0].name == "requests_total"

    def test_get_metrics_contains_labels(self, servicer):
        """指标包含标签"""
        request = quantumflow_pb2.MetricsRequest()
        context = MockServicerContext()

        response = servicer.GetMetrics(request, context)

        for metric in response.metrics:
            if metric.name == "requests_total":
                assert metric.labels["method"] == "inference"

    def test_get_metrics_has_timestamp(self, servicer):
        """指标包含时间戳"""
        request = quantumflow_pb2.MetricsRequest()
        context = MockServicerContext()

        before = int(time.time())
        response = servicer.GetMetrics(request, context)
        after = int(time.time())

        for metric in response.metrics:
            assert before <= metric.timestamp <= after

    def test_get_metrics_multiple_filters(self, servicer):
        """多个指标过滤"""
        request = quantumflow_pb2.MetricsRequest(
            metric_names=["requests_total", "gpu_memory_usage"]
        )
        context = MockServicerContext()

        response = servicer.GetMetrics(request, context)

        assert len(response.metrics) == 2
        metric_names = [m.name for m in response.metrics]
        assert "requests_total" in metric_names
        assert "gpu_memory_usage" in metric_names

    def test_get_metrics_filter_nonexistent(self, servicer):
        """过滤不存在的指标"""
        request = quantumflow_pb2.MetricsRequest(metric_names=["nonexistent_metric"])
        context = MockServicerContext()

        response = servicer.GetMetrics(request, context)

        assert len(response.metrics) == 0


class TestMetricsServiceStreamMetrics:
    """StreamMetrics 测试"""

    @pytest.fixture
    def servicer(self):
        return MetricsServiceServicer()

    def test_stream_metrics_yields_responses(self, servicer):
        """StreamMetrics 返回响应"""
        request = quantumflow_pb2.MetricsRequest()
        context = MockServicerContext()

        gen = servicer.StreamMetrics(request, context)
        response = next(gen)

        assert len(response.metrics) > 0
        gen.close()

    def test_stream_metrics_sleeps_between_responses(self, servicer):
        """流式指标在响应之间等待"""
        request = quantumflow_pb2.MetricsRequest()
        context = MockServicerContext()

        with patch("quantumflow.grpc.services.metrics.time.sleep") as mock_sleep:
            gen = servicer.StreamMetrics(request, context)
            next(gen)
            next(gen)

            mock_sleep.assert_called_once_with(10)
            gen.close()
