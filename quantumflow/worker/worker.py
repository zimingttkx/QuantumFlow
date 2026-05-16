"""Worker节点实现"""

import asyncio
import platform
import socket
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

import structlog

from quantumflow.core.constants import NodeStatus
from quantumflow.inference.engine import InferenceEngine, ModelConfig, SamplingParams
from quantumflow.monitoring import metrics

logger = structlog.get_logger().bind(component="worker")


@dataclass
class WorkerConfig:
    """Worker配置"""

    node_id: str = field(default_factory=lambda: f"worker-{uuid.uuid4().hex[:8]}")
    host: str = "0.0.0.0"
    port: int = 8080
    heartbeat_interval: int = 5
    heartbeat_timeout: int = 30
    gpu_enabled: bool = True
    max_concurrent_requests: int = 100


class WorkerNode:
    """
    Worker节点

    负责：
    - 管理推理引擎生命周期
    - 收集节点资源信息
    - 与Controller保持心跳
    - 处理推理请求
    """

    def __init__(
        self,
        config: WorkerConfig,
        engine: Optional[InferenceEngine] = None,
    ):
        self.config = config
        self.engine = engine

        # 节点状态
        self.status = NodeStatus.INITIALIZING
        self.started_at = datetime.now()

        # 资源信息
        self._gpu_info: List[Dict[str, Any]] = []

        # 请求跟踪
        self.active_requests: Dict[str, float] = {}
        self.completed_requests: int = 0
        self.failed_requests: int = 0

        # 运行状态
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        # 控制器地址
        self.controller_url: Optional[str] = None

        logger.info(
            "worker_created",
            node_id=config.node_id,
            host=config.host,
            port=config.port,
        )

    @property
    def node_info(self) -> Dict[str, Any]:
        """获取节点信息"""
        return {
            "node_id": self.config.node_id,
            "hostname": socket.gethostname(),
            "ip": self._get_ip(),
            "port": self.config.port,
            "gpu_count": len(self._gpu_info),
            "gpu_info": self._gpu_info,
            "status": self.status.value,
            "labels": self._get_labels(),
            "version": "1.0.0",
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_total": psutil.disk_usage("/").total if platform.system() != "Windows" else psutil.disk_usage("C:\\").total,
            "disk_available": psutil.disk_usage("/").free if platform.system() != "Windows" else psutil.disk_usage("C:\\").free,
            "current_load": self._get_load(),
            "loaded_models": self.engine.loaded_model_names if self.engine else [],
            "active_requests": len(self.active_requests),
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
        }

    async def start(self, controller_url: Optional[str] = None):
        """启动Worker"""
        if self._running:
            return

        self.controller_url = controller_url
        self._running = True

        # 初始化引擎
        if self.engine and not self.engine.is_ready:
            await self.engine.initialize()

        # 收集GPU信息
        self._gpu_info = await self._collect_gpu_info()

        # 更新状态
        self.status = NodeStatus.HEALTHY

        # 启动心跳
        if controller_url:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("worker_started", node_id=self.config.node_id)

    async def stop(self):
        """停止Worker"""
        if not self._running:
            return

        self._running = False
        self.status = NodeStatus.OFFLINE

        # 停止心跳
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # 关闭引擎
        if self.engine:
            for model_name in self.engine.loaded_model_names:
                await self.engine.unload_model(model_name)

        logger.info("worker_stopped", node_id=self.config.node_id)

    async def load_model(self, config: ModelConfig) -> bool:
        """加载模型"""
        if not self.engine:
            logger.error("no_engine_available")
            return False

        try:
            success = await self.engine.load_model(config)
            if success:
                metrics.MODEL_LOADED.labels(
                    node_id=self.config.node_id,
                    model=config.model_name,
                ).set(1)

                logger.info(
                    "model_loaded",
                    node_id=self.config.node_id,
                    model=config.model_name,
                )
            return success
        except Exception as e:
            logger.error(
                "model_load_failed",
                node_id=self.config.node_id,
                model=config.model_name,
                error=str(e),
            )
            return False

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if not self.engine:
            return False

        try:
            success = await self.engine.unload_model(model_name)
            if success:
                metrics.MODEL_LOADED.labels(
                    node_id=self.config.node_id,
                    model=model_name,
                ).set(0)

                logger.info(
                    "model_unloaded",
                    node_id=self.config.node_id,
                    model=model_name,
                )
            return success
        except Exception as e:
            logger.error(
                "model_unload_failed",
                node_id=self.config.node_id,
                model=model_name,
                error=str(e),
            )
            return False

    async def inference(
        self,
        request_id: str,
        model_name: str,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> Dict[str, Any]:
        """
        执行推理

        Args:
            request_id: 请求ID
            model_name: 模型名称
            prompts: 提示列表
            sampling_params: 采样参数

        Returns:
            推理结果
        """
        start_time = time.time()

        # 跟踪请求
        self.active_requests[request_id] = start_time
        metrics.ACTIVE_INFERENCES.labels(
            node_id=self.config.node_id,
            model=model_name,
        ).inc()

        try:
            if not self.engine:
                raise RuntimeError("No inference engine available")

            # 检查模型是否已加载
            if not await self.engine.is_model_loaded(model_name):
                raise RuntimeError(f"Model {model_name} not loaded")

            # 执行推理
            results = await self.engine.generate(
                model_name=model_name,
                prompts=prompts,
                sampling_params=sampling_params,
            )

            # 计算延迟
            latency_ms = (time.time() - start_time) * 1000

            # 更新指标
            self.completed_requests += 1
            metrics.REQUEST_COUNT.labels(
                node_id=self.config.node_id,
                model=model_name,
                status="success",
            ).inc()
            metrics.REQUEST_LATENCY.labels(
                node_id=self.config.node_id,
                model=model_name,
            ).observe(latency_ms / 1000)

            return {
                "request_id": request_id,
                "status": "success",
                "results": [
                    {
                        "output": r.outputs,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "finish_reason": r.finish_reason,
                    }
                    for r in results
                ],
                "latency_ms": latency_ms,
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.failed_requests += 1

            metrics.REQUEST_COUNT.labels(
                node_id=self.config.node_id,
                model=model_name,
                status="error",
            ).inc()

            logger.error(
                "inference_failed",
                node_id=self.config.node_id,
                request_id=request_id,
                error=str(e),
            )

            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "latency_ms": latency_ms,
            }

        finally:
            # 移除跟踪
            self.active_requests.pop(request_id, None)
            metrics.ACTIVE_INFERENCES.labels(
                node_id=self.config.node_id,
                model=model_name,
            ).dec()

    async def get_stats(self, model_name: str) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.engine:
            return {}

        stats = await self.engine.get_stats(model_name)
        stats["active_requests"] = len(self.active_requests)
        stats["completed_requests"] = self.completed_requests
        stats["failed_requests"] = self.failed_requests

        return stats

    # ==================== 内部方法 ====================

    def _get_ip(self) -> str:
        """获取本机IP"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _get_load(self) -> float:
        """获取负载"""
        return psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0

    def _get_labels(self) -> Dict[str, str]:
        """获取节点标签"""
        return {
            "platform": platform.system().lower(),
            "arch": platform.machine().lower(),
            "gpu_enabled": str(self.config.gpu_enabled).lower(),
        }

    async def _collect_gpu_info(self) -> List[Dict[str, Any]]:
        """收集GPU信息"""
        gpu_info = []

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)

                    gpu_info.append({
                        "gpu_id": i,
                        "name": props.name,
                        "memory_total": props.total_memory,
                        "memory_used": memory_allocated,
                        "utilization": (memory_allocated / props.total_memory) * 100
                        if props.total_memory > 0
                        else 0,
                        "temperature": 0,  # 需要nvidia-ml-py
                    })

                    # 更新监控指标
                    metrics.GPU_MEMORY.labels(
                        node_id=self.config.node_id,
                        gpu_id=str(i),
                    ).set(memory_allocated)
                    metrics.GPU_UTILIZATION.labels(
                        node_id=self.config.node_id,
                        gpu_id=str(i),
                    ).set((memory_allocated / props.total_memory) * 100
                          if props.total_memory > 0 else 0)

        except ImportError:
            logger.warning("torch_not_available_gpu_info_unavailable")

        return gpu_info

    async def _heartbeat_loop(self):
        """心跳循环"""
        import httpx

        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                if not self.controller_url:
                    continue

                # 更新GPU信息
                self._gpu_info = await self._collect_gpu_info()

                # 发送心跳
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        f"{self.controller_url}/api/v1/cluster/heartbeat",
                        json=self.node_info,
                    )

                # 更新指标
                metrics.NODE_COUNT.labels(
                    node_id=self.config.node_id,
                    status=self.status.value,
                ).set(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "heartbeat_failed",
                    node_id=self.config.node_id,
                    error=str(e),
                )
