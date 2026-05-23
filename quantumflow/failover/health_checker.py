"""健康检测器

提供细粒度的故障检测能力：
- GPU 故障检测（温度、显存、利用率）
- 模型故障检测（加载失败、推理超时）
- 通信故障检测（心跳超时、RPC 超时）
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from quantumflow.cluster.manager import ClusterManager
from quantumflow.failover.models import HealthStatus
from quantumflow.failover.policy import HealthThresholds, ReplicaPolicy

logger = structlog.get_logger().bind(component="health_checker")


@dataclass
class HealthCheckResult:
    """健康检测结果"""

    node_id: str
    status: HealthStatus
    timestamp: datetime
    checks_performed: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """是否健康"""
        return self.status == HealthStatus.HEALTHY

    def is_degraded(self) -> bool:
        """是否降级"""
        return self.status == HealthStatus.DEGRADED

    def is_unhealthy(self) -> bool:
        """是否不健康"""
        return self.status == HealthStatus.UNHEALTHY


@dataclass
class GPUHealthResult:
    """GPU 健康检测结果"""

    gpu_id: int
    status: HealthStatus
    temperature: float | None = None
    memory_used_ratio: float | None = None
    utilization: float | None = None
    reasons: list[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """是否健康"""
        return self.status == HealthStatus.HEALTHY

    def is_degraded(self) -> bool:
        """是否降级"""
        return self.status == HealthStatus.DEGRADED

    def is_unhealthy(self) -> bool:
        """是否不健康"""
        return self.status == HealthStatus.UNHEALTHY


class HealthChecker:
    """
    细粒度健康检测器

    支持多种检测类型：
    - GPU 故障检测
    - 模型故障检测
    - 通信故障检测
    - 节点综合健康检测
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        health_thresholds: HealthThresholds | None = None,
        replica_policy: ReplicaPolicy | None = None,
    ):
        self._cluster_manager = cluster_manager
        self._health_thresholds = health_thresholds or HealthThresholds()
        self._replica_policy = replica_policy or ReplicaPolicy()

        # GPU 监控客户端（延迟导入以避免循环依赖）
        self._gpu_monitor = None

        # 运行状态
        self._running = False
        self._health_check_task: asyncio.Task | None = None

        logger.info("health_checker_created", thresholds=self._health_thresholds.to_dict())

    async def start(self) -> None:
        """启动健康检测"""
        if self._running:
            logger.warning("health_checker_already_running")
            return

        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("health_checker_started")

    async def stop(self) -> None:
        """停止健康检测"""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("health_checker_stopped")

    async def _health_check_loop(self) -> None:
        """健康检测循环"""
        logger.info("health_check_loop_started")

        while self._running:
            try:
                await asyncio.sleep(self._replica_policy.health_check_interval_seconds)

                # 获取所有健康节点
                nodes = await self._cluster_manager.get_healthy_nodes()

                for node in nodes:
                    try:
                        result = await self.check_node_health(node.node_id)
                        if result.is_unhealthy():
                            await self._handle_unhealthy_node(node.node_id, result)
                        elif result.is_degraded():
                            await self._handle_degraded_node(node.node_id, result)
                    except Exception as e:
                        logger.error(
                            "health_check_failed",
                            node_id=node.node_id,
                            error=str(e),
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_loop_error", error=str(e))
                await asyncio.sleep(5)

        logger.info("health_check_loop_stopped")

    async def _handle_unhealthy_node(self, node_id: str, result: HealthCheckResult) -> None:
        """处理不健康节点"""
        logger.warning(
            "node_unhealthy",
            node_id=node_id,
            reasons=result.reasons,
        )
        # 触发 ClusterManager 更新节点状态
        await self._cluster_manager.update_node_health(
            node_id,
            "unhealthy",
            reason="; ".join(result.reasons),
        )

    async def _handle_degraded_node(self, node_id: str, result: HealthCheckResult) -> None:
        """处理降级节点"""
        logger.info(
            "node_degraded",
            node_id=node_id,
            reasons=result.reasons,
        )
        await self._cluster_manager.update_node_health(
            node_id,
            "degraded",
            reason="; ".join(result.reasons),
        )

    # ==================== GPU 健康检测 ====================

    async def check_gpu_health(
        self, node_id: str, gpu_id: int
    ) -> GPUHealthResult:
        """
        检查单 GPU 健康状态

        Args:
            node_id: 节点 ID
            gpu_id: GPU ID

        Returns:
            GPU 健康检测结果
        """
        result = GPUHealthResult(gpu_id=gpu_id, status=HealthStatus.HEALTHY)

        try:
            # 获取 GPU 监控数据
            gpu_data = await self._get_gpu_data(node_id, gpu_id)

            if gpu_data is None:
                result.status = HealthStatus.UNKNOWN
                result.reasons.append(f"GPU {gpu_id} data not available")
                return result

            # 检查温度
            if "temperature" in gpu_data:
                result.temperature = gpu_data["temperature"]
                if result.temperature > self._health_thresholds.gpu_temp_threshold:
                    result.status = HealthStatus.UNHEALTHY
                    result.reasons.append(
                        f"GPU temperature {result.temperature}C exceeds threshold "
                        f"{self._health_thresholds.gpu_temp_threshold}C"
                    )
                elif result.temperature > self._health_thresholds.gpu_temp_warning:
                    if result.status == HealthStatus.HEALTHY:
                        result.status = HealthStatus.DEGRADED
                    result.reasons.append(
                        f"GPU temperature {result.temperature}C exceeds warning "
                        f"{self._health_thresholds.gpu_temp_warning}C"
                    )

            # 检查显存
            if "memory_used" in gpu_data and "memory_total" in gpu_data:
                result.memory_used_ratio = gpu_data["memory_used"] / gpu_data["memory_total"]
                if result.memory_used_ratio > self._health_thresholds.gpu_mem_threshold:
                    result.status = HealthStatus.UNHEALTHY
                    result.reasons.append(
                        f"GPU memory usage {result.memory_used_ratio:.1%} exceeds threshold "
                        f"{self._health_thresholds.gpu_mem_threshold:.1%}"
                    )

            # 检查利用率
            if "utilization" in gpu_data:
                result.utilization = gpu_data["utilization"]
                if result.utilization > self._health_thresholds.gpu_util_threshold:
                    if result.status == HealthStatus.HEALTHY:
                        result.status = HealthStatus.DEGRADED
                    result.reasons.append(
                        f"GPU utilization {result.utilization:.1%} is very high"
                    )

        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.reasons.append(f"GPU health check error: {str(e)}")
            logger.error(
                "gpu_health_check_failed",
                node_id=node_id,
                gpu_id=gpu_id,
                error=str(e),
            )

        return result

    async def check_all_gpus(self, node_id: str) -> dict[int, GPUHealthResult]:
        """
        检查节点所有 GPU 健康状态

        Args:
            node_id: 节点 ID

        Returns:
            GPU ID -> GPU 健康检测结果
        """
        node = await self._cluster_manager.get_node(node_id)
        if not node:
            return {}

        results = {}
        for gpu_info in node.gpu_info:
            result = await self.check_gpu_health(node_id, gpu_info.gpu_id)
            results[gpu_info.gpu_id] = result

            # 更新 ClusterManager 中的 GPU 健康状态
            status_str = result.status.value
            await self._cluster_manager.update_gpu_health(
                node_id, gpu_info.gpu_id, status_str
            )

        return results

    async def _get_gpu_data(self, node_id: str, gpu_id: int) -> dict | None:
        """获取 GPU 监控数据"""
        try:
            # 延迟导入以避免循环依赖
            if self._gpu_monitor is None:
                from quantumflow.inference.gpu_monitor import GPUMonitor

                self._gpu_monitor = GPUMonitor()

            # 获取节点 GPU 信息
            node = await self._cluster_manager.get_node(node_id)
            if not node:
                return None

            # 从节点获取 GPU 数据
            for gpu_info in node.gpu_info:
                if gpu_info.gpu_id == gpu_id:
                    return {
                        "temperature": gpu_info.temperature,
                        "memory_used": gpu_info.memory_used,
                        "memory_total": gpu_info.memory_total,
                        "utilization": gpu_info.utilization,
                    }

            return None
        except Exception as e:
            logger.error(
                "get_gpu_data_failed",
                node_id=node_id,
                gpu_id=gpu_id,
                error=str(e),
            )
            return None

    # ==================== 模型健康检测 ====================

    async def check_model_health(
        self, node_id: str, model_name: str
    ) -> HealthStatus:
        """
        检查模型健康状态

        Args:
            node_id: 节点 ID
            model_name: 模型名称

        Returns:
            健康状态
        """
        try:
            node = await self._cluster_manager.get_node(node_id)
            if not node:
                return HealthStatus.UNKNOWN

            # 检查模型是否加载
            if model_name not in node.loaded_models:
                await self._cluster_manager.update_model_health(
                    node_id, model_name, HealthStatus.UNHEALTHY.value
                )
                return HealthStatus.UNHEALTHY

            # 尝试进行简单的推理健康检查
            is_healthy = await self._do_model_health_check(node_id, model_name)

            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED
            await self._cluster_manager.update_model_health(
                node_id, model_name, status.value
            )

            return status

        except Exception as e:
            logger.error(
                "model_health_check_failed",
                node_id=node_id,
                model_name=model_name,
                error=str(e),
            )
            await self._cluster_manager.update_model_health(
                node_id, model_name, HealthStatus.UNHEALTHY.value
            )
            return HealthStatus.UNHEALTHY

    async def _do_model_health_check(
        self, node_id: str, model_name: str, timeout: int = 30
    ) -> bool:
        """
        执行模型健康检查（发送测试推理请求）

        Args:
            node_id: 节点 ID
            model_name: 模型名称
            timeout: 超时时间（秒）

        Returns:
            是否健康
        """
        try:
            # 这里应该调用 WorkerClient 进行实际健康检查
            # 目前简化为检查节点是否可达
            node = await self._cluster_manager.get_node(node_id)
            if not node:
                return False

            # 如果节点状态为 HEALTHY，认为模型也是健康的
            return node.status.value == "healthy"

        except Exception as e:
            logger.error(
                "model_health_check_error",
                node_id=node_id,
                model_name=model_name,
                error=str(e),
            )
            return False

    # ==================== 通信健康检测 ====================

    async def check_communication_health(self, node_id: str) -> HealthStatus:
        """
        检查节点间通信健康状态

        Args:
            node_id: 节点 ID

        Returns:
            健康状态
        """
        try:
            node = await self._cluster_manager.get_node(node_id)
            if not node:
                return HealthStatus.UNKNOWN

            unhealthy_count = 0
            degraded_count = 0

            for peer_node_id, status in node.communication_health.items():
                if status == "fail" or status == "timeout":
                    unhealthy_count += 1
                elif status == "degraded":
                    degraded_count += 1

            total_peers = len(node.communication_health)
            if total_peers == 0:
                return HealthStatus.HEALTHY

            # 超过 50% 的通信不健康，标记为不健康
            if unhealthy_count > total_peers * 0.5:
                return HealthStatus.UNHEALTHY

            # 有通信问题但不严重，标记为降级
            if unhealthy_count > 0 or degraded_count > total_peers * 0.3:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            logger.error(
                "communication_health_check_failed",
                node_id=node_id,
                error=str(e),
            )
            return HealthStatus.UNKNOWN

    # ==================== 节点综合健康检测 ====================

    async def check_node_health(self, node_id: str) -> HealthCheckResult:
        """
        综合检查节点健康状态

        Args:
            node_id: 节点 ID

        Returns:
            节点健康检测结果
        """
        result = HealthCheckResult(
            node_id=node_id,
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
        )

        try:
            node = await self._cluster_manager.get_node(node_id)
            if not node:
                result.status = HealthStatus.UNKNOWN
                result.reasons.append("Node not found")
                return result

            # 1. 检查节点基础状态
            if node.status.value != "healthy":
                result.status = HealthStatus.UNHEALTHY
                result.reasons.append(f"Node status is {node.status.value}")
                result.checks_performed["node_status"] = False
            else:
                result.checks_performed["node_status"] = True

            # 2. 检查 GPU 健康
            gpu_results = await self.check_all_gpus(node_id)
            unhealthy_gpus = sum(
                1 for r in gpu_results.values() if r.is_unhealthy()
            )
            degraded_gpus = sum(
                1 for r in gpu_results.values() if r.is_degraded()
            )

            result.details["gpu_status"] = {
                gpu_id: r.status.value for gpu_id, r in gpu_results.items()
            }
            result.checks_performed["gpu_health"] = unhealthy_gpus == 0

            if unhealthy_gpus > 0:
                result.status = HealthStatus.UNHEALTHY
                result.reasons.append(f"{unhealthy_gpus} GPU(s) unhealthy")

            if degraded_gpus > 0 and result.status != HealthStatus.UNHEALTHY:
                result.status = HealthStatus.DEGRADED
                result.reasons.append(f"{degraded_gpus} GPU(s) degraded")

            # 3. 检查模型健康
            model_health = await self._check_all_models_health(node_id)
            unhealthy_models = sum(
                1 for s in model_health.values() if s == HealthStatus.UNHEALTHY
            )
            degraded_models = sum(
                1 for s in model_health.values() if s == HealthStatus.DEGRADED
            )

            result.details["model_health"] = model_health
            result.checks_performed["model_health"] = unhealthy_models == 0

            if unhealthy_models > 0:
                result.status = HealthStatus.UNHEALTHY
                result.reasons.append(f"{unhealthy_models} model(s) unhealthy")

            if degraded_models > 0 and result.status != HealthStatus.UNHEALTHY:
                result.status = HealthStatus.DEGRADED
                result.reasons.append(f"{degraded_models} model(s) degraded")

            # 4. 检查通信健康
            comm_health = await self.check_communication_health(node_id)
            result.details["communication_health"] = comm_health.value
            result.checks_performed["communication_health"] = (
                comm_health != HealthStatus.UNHEALTHY
            )

            if comm_health == HealthStatus.UNHEALTHY:
                result.status = HealthStatus.UNHEALTHY
                result.reasons.append("Communication unhealthy")

            # 5. 检查连续失败次数
            if node.consecutive_failures >= self._health_thresholds.failure_threshold:
                result.status = HealthStatus.UNHEALTHY
                result.reasons.append(
                    f"Consecutive failures {node.consecutive_failures} "
                    f"exceeds threshold {self._health_thresholds.failure_threshold}"
                )

        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.reasons.append(f"Health check error: {str(e)}")
            logger.error(
                "node_health_check_failed",
                node_id=node_id,
                error=str(e),
            )

        return result

    async def _check_all_models_health(self, node_id: str) -> dict[str, HealthStatus]:
        """检查节点上所有模型的健康状态"""
        node = await self._cluster_manager.get_node(node_id)
        if not node:
            return {}

        results = {}
        for model_name in node.loaded_models:
            results[model_name] = await self.check_model_health(node_id, model_name)

        return results

    # ==================== 阈值配置 ====================

    def configure_thresholds(
        self,
        gpu_temp_threshold: float | None = None,
        gpu_mem_threshold: float | None = None,
        gpu_util_threshold: float | None = None,
        heartbeat_timeout: int | None = None,
        model_load_timeout: int | None = None,
        inference_timeout: int | None = None,
        comm_timeout: int | None = None,
        failure_threshold: int | None = None,
    ) -> None:
        """
        配置健康检测阈值

        Args:
            gpu_temp_threshold: GPU 温度阈值
            gpu_mem_threshold: GPU 显存阈值
            gpu_util_threshold: GPU 利用率阈值
            heartbeat_timeout: 心跳超时
            model_load_timeout: 模型加载超时
            inference_timeout: 推理超时
            comm_timeout: 通信超时
            failure_threshold: 故障阈值
        """
        if gpu_temp_threshold is not None:
            self._health_thresholds.gpu_temp_threshold = gpu_temp_threshold
        if gpu_mem_threshold is not None:
            self._health_thresholds.gpu_mem_threshold = gpu_mem_threshold
        if gpu_util_threshold is not None:
            self._health_thresholds.gpu_util_threshold = gpu_util_threshold
        if heartbeat_timeout is not None:
            self._health_thresholds.heartbeat_timeout = heartbeat_timeout
        if model_load_timeout is not None:
            self._health_thresholds.model_load_timeout = model_load_timeout
        if inference_timeout is not None:
            self._health_thresholds.inference_timeout = inference_timeout
        if comm_timeout is not None:
            self._health_thresholds.comm_timeout = comm_timeout
        if failure_threshold is not None:
            self._health_thresholds.failure_threshold = failure_threshold

        logger.info(
            "health_thresholds_updated",
            thresholds=self._health_thresholds.to_dict(),
        )

    def get_thresholds(self) -> HealthThresholds:
        """获取当前阈值配置"""
        return self._health_thresholds
