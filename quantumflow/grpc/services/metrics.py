"""MetricsService gRPC 服务实现

提供指标功能：
- 获取指标 (GetMetrics)
- 流式指标 (StreamMetrics)
"""

import time
from typing import Iterator

import grpc

from quantumflow.grpc.generated import quantumflow_pb2, quantumflow_pb2_grpc
from quantumflow.grpc.services.base import BaseService


class MetricsServiceServicer(BaseService, quantumflow_pb2_grpc.MetricsServiceServicer):
    """指标服务实现

    提供 Prometheus 指标获取功能。
    """

    def __init__(self, engine_manager=None, cluster_manager=None):
        """
        Args:
            engine_manager: EngineManager 实例
            cluster_manager: ClusterManager 实例
        """
        super().__init__(engine_manager=engine_manager, cluster_manager=cluster_manager)

    def GetMetrics(
        self,
        request: quantumflow_pb2.MetricsRequest,
        context: grpc.ServicerContext,
    ) -> quantumflow_pb2.MetricsResponse:
        """获取指标

        Args:
            request: MetricsRequest 消息
            context: gRPC 上下文

        Returns:
            MetricsResponse 消息
        """
        metrics = []

        # 获取请求的指标名列表
        requested_metrics = list(request.metric_names) if request.metric_names else []

        # 收集指标
        current_time = int(time.time())

        # 模拟指标
        if not requested_metrics or "requests_total" in requested_metrics:
            metrics.append(quantumflow_pb2.MetricSample(
                name="requests_total",
                value=1000.0,
                timestamp=current_time,
                labels={"method": "inference"},
            ))

        if not requested_metrics or "gpu_memory_usage" in requested_metrics:
            metrics.append(quantumflow_pb2.MetricSample(
                name="gpu_memory_usage",
                value=0.75,
                timestamp=current_time,
                labels={"gpu": "0"},
            ))

        if not requested_metrics or "active_inferences" in requested_metrics:
            metrics.append(quantumflow_pb2.MetricSample(
                name="active_inferences",
                value=5.0,
                timestamp=current_time,
            ))

        if not requested_metrics or "node_count" in requested_metrics:
            metrics.append(quantumflow_pb2.MetricSample(
                name="node_count",
                value=3.0,
                timestamp=current_time,
                labels={"status": "active"},
            ))

        return quantumflow_pb2.MetricsResponse(metrics=metrics)

    def StreamMetrics(
        self,
        request: quantumflow_pb2.MetricsRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[quantumflow_pb2.MetricsResponse]:
        """流式指标

        Args:
            request: MetricsRequest 消息
            context: gRPC 上下文

        Yields:
            MetricsResponse 消息（定期）
        """
        while not context.cancelled():
            response = self.GetMetrics(request, context)
            yield response

            # 等待 10 秒再发送下一个
            time.sleep(10)
