"""Worker HTTP 客户端

提供与 Worker 节点通信的 HTTP 客户端，用于分布式调度。
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

logger = structlog.get_logger().bind(component="worker_client")


@dataclass
class WorkerEndpoint:
    """Worker节点端点信息"""

    node_id: str
    host: str
    port: int
    status: str = "healthy"

    # 新增字段：与模型路由配合
    supported_models: list[str] = field(default_factory=list)  # 已加载/支持的模型名
    backend: str = ""                                         # 主后端 (vllm/tgi/...)
    gpu_count: int = 0
    gpu_family: str = "unknown"                               # 主流 GPU 家族
    failure_count: int = 0                                    # 连续失败次数（用于熔断）
    last_success_at: float = 0.0                              # 上次成功时间
    last_failure_at: float = 0.0                              # 上次失败时间
    last_failure_reason: str = ""

    @property
    def url(self) -> str:
        """获取Worker的完整URL"""
        return f"http://{self.host}:{self.port}"

    def record_success(self) -> None:
        self.failure_count = 0
        self.last_success_at = time.time()
        self.status = "healthy"

    def record_failure(self, reason: str = "") -> bool:
        """记录一次失败；返回是否需要熔断（连续失败 >= 3）"""
        self.failure_count += 1
        self.last_failure_at = time.time()
        self.last_failure_reason = reason
        if self.failure_count >= 3:
            self.status = "unhealthy"
            return True
        return False

    def can_serve(self, model_name: str) -> bool:
        """判断 worker 能否服务该模型（粗筛：模型名 + 状态）"""
        if self.status != "healthy":
            return False
        if not self.supported_models:
            # 没有上报过模型列表，假设能服务（兼容老 worker）
            return True
        return model_name in self.supported_models


class WorkerClient:
    """
    Worker HTTP 客户端

    负责：
    - 与 Worker 节点建立 HTTP 通信
    - 发送推理请求到指定 Worker
    - 处理 Worker 响应和错误
    """

    def __init__(self, timeout: float = 30.0, max_retries: int = 1, retry_backoff: float = 0.5):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self, endpoint: WorkerEndpoint) -> bool:
        """健康检查 — 调用 /status 并更新 endpoint 状态

        Returns:
            worker 是否健康
        """
        try:
            client = await self._get_client()
            url = f"{endpoint.url}/status"
            response = await client.get(url)
            if response.status_code == 200:
                body = response.json()
                # 用 worker 上报的状态更新本地缓存
                if isinstance(body, dict):
                    if "loaded_models" in body and isinstance(body["loaded_models"], list):
                        endpoint.supported_models = body["loaded_models"]
                    if "backend" in body:
                        endpoint.backend = body.get("backend", endpoint.backend)
                    if "gpu_count" in body:
                        endpoint.gpu_count = int(body.get("gpu_count", 0))
                endpoint.record_success()
                return True
            endpoint.record_failure(f"http {response.status_code}")
            return False
        except Exception as e:
            endpoint.record_failure(str(e))
            return False

    async def cancel(
        self,
        endpoint: WorkerEndpoint,
        request_id: str,
    ) -> bool:
        """取消正在执行的推理请求

        Returns:
            是否成功发送取消信号（不代表推理已停止）
        """
        try:
            client = await self._get_client()
            url = f"{endpoint.url}/cancel/{request_id}"
            response = await client.post(url)
            return response.status_code == 200
        except Exception as e:
            logger.warning(
                "worker_cancel_failed",
                worker_url=endpoint.url,
                request_id=request_id,
                error=str(e),
            )
            return False

    async def inference(
        self,
        endpoint: WorkerEndpoint,
        request_id: str,
        model_name: str,
        prompt: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        向 Worker 发送推理请求

        Args:
            endpoint: Worker 端点信息
            request_id: 请求ID
            model_name: 模型名称
            prompt: 提示词
            sampling_params: 采样参数

        Returns:
            推理结果字典

        失败语义：所有重试都失败后，endpoint.failure_count 递增。
        端点级熔断由调用方基于 can_serve() 决定是否跳过。
        """
        client = await self._get_client()

        payload = {
            "request_id": request_id,
            "model_name": model_name,
            "prompts": [prompt],
            "sampling_params": sampling_params
            or {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_tokens": 512,
            },
        }

        url = f"{endpoint.url}/inference"
        last_error = ""

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    "sending_inference_to_worker",
                    request_id=request_id,
                    worker_url=url,
                    model=model_name,
                    attempt=attempt,
                )

                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    # 即使 HTTP 200，body 也可能包含 status=error
                    if isinstance(result, dict) and result.get("status") == "error":
                        last_error = result.get("error", "unknown")
                        endpoint.record_failure(f"worker_error: {last_error[:200]}")
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_backoff * (attempt + 1))
                            continue
                        return result
                    endpoint.record_success()
                    logger.info(
                        "inference_response_received",
                        request_id=request_id,
                        worker_url=url,
                        status=result.get("status"),
                    )
                    return result

                # 非 2xx
                error_text = response.text
                last_error = f"HTTP {response.status_code}: {error_text[:200]}"
                endpoint.record_failure(last_error)
                logger.error(
                    "worker_inference_failed",
                    request_id=request_id,
                    worker_url=url,
                    status=response.status_code,
                    error=error_text,
                    attempt=attempt,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff * (attempt + 1))
                    continue
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": f"Worker returned status {response.status_code}: {error_text}",
                }

            except httpx.TimeoutException:
                last_error = f"timeout after {self.timeout}s"
                endpoint.record_failure(last_error)
                logger.error(
                    "worker_inference_timeout",
                    request_id=request_id,
                    worker_url=url,
                    timeout=self.timeout,
                    attempt=attempt,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff * (attempt + 1))
                    continue
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": last_error,
                }

            except Exception as e:
                last_error = str(e)
                endpoint.record_failure(last_error)
                logger.error(
                    "worker_inference_error",
                    request_id=request_id,
                    worker_url=url,
                    error=last_error,
                    attempt=attempt,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff * (attempt + 1))
                    continue
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": last_error,
                }

        # 不可达
        return {
            "request_id": request_id,
            "status": "error",
            "error": last_error or "all retries exhausted",
        }

    async def load_model(
        self,
        endpoint: WorkerEndpoint,
        model_name: str,
        model_path: str | None = None,
        tensor_parallel: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> dict[str, Any]:
        """
        请求 Worker 加载模型

        Args:
            endpoint: Worker 端点信息
            model_name: 模型名称
            model_path: 模型路径
            tensor_parallel: 张量并行数
            gpu_memory_utilization: GPU 内存利用率

        Returns:
            操作结果
        """
        client = await self._get_client()

        payload = {
            "model_name": model_name,
            "model_path": model_path,
            "tensor_parallel": tensor_parallel,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

        url = f"{endpoint.url}/load_model"

        try:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "code": response.status_code}

        except Exception as e:
            logger.error(
                "worker_load_model_error",
                worker_url=url,
                model=model_name,
                error=str(e),
            )
            return {"status": "error", "error": str(e)}

    async def get_status(self, endpoint: WorkerEndpoint) -> dict[str, Any]:
        """
        获取 Worker 状态

        Args:
            endpoint: Worker 端点信息

        Returns:
            Worker 状态信息
        """
        client = await self._get_client()
        url = f"{endpoint.url}/status"

        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "code": response.status_code}

        except Exception as e:
            logger.error(
                "worker_get_status_error",
                worker_url=url,
                error=str(e),
            )
            return {"status": "error", "error": str(e)}


class WorkerRegistry:
    """
    Worker 注册表

    管理所有可用的 Worker 节点，提供节点发现和健康检查。
    """

    def __init__(self):
        self._workers: dict[str, WorkerEndpoint] = {}
        self._lock = asyncio.Lock()

    async def register(self, endpoint: WorkerEndpoint):
        """注册 Worker 节点"""
        async with self._lock:
            self._workers[endpoint.node_id] = endpoint
            logger.info(
                "worker_registered",
                node_id=endpoint.node_id,
                url=endpoint.url,
            )

    async def unregister(self, node_id: str):
        """注销 Worker 节点"""
        async with self._lock:
            if node_id in self._workers:
                del self._workers[node_id]
                logger.info("worker_unregistered", node_id=node_id)

    async def get_worker(self, node_id: str) -> WorkerEndpoint | None:
        """获取指定节点"""
        async with self._lock:
            return self._workers.get(node_id)

    async def get_all_workers(self) -> list[WorkerEndpoint]:
        """获取所有 Worker"""
        async with self._lock:
            return list(self._workers.values())

    async def get_worker_count(self) -> int:
        """获取 Worker 数量"""
        async with self._lock:
            return len(self._workers)

    async def update_worker_status(self, node_id: str, status: str):
        """更新 Worker 状态"""
        async with self._lock:
            if node_id in self._workers:
                self._workers[node_id].status = status


# 全局 Worker 注册表
_registry: WorkerRegistry | None = None


def get_worker_registry() -> WorkerRegistry:
    """获取全局 Worker 注册表"""
    global _registry
    if _registry is None:
        _registry = WorkerRegistry()
    return _registry
