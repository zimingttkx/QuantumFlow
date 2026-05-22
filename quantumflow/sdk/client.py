"""SDK 客户端实现"""
import httpx
from typing import Any

from quantumflow.sdk.exceptions import APIError, RateLimitError, TimeoutError


class SyncQuantumFlowClient:
    """QuantumFlow 同步客户端"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        tenant_id: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        if api_key:
            self._client.headers["X-API-Key"] = api_key
        if tenant_id:
            self._client.headers["X-Tenant-ID"] = tenant_id

    def _post(self, path: str, **kwargs) -> dict[str, Any]:
        """发送 POST 请求"""
        try:
            response = self._client.post(path, **kwargs)
            if response.status_code == 429:
                raise RateLimitError()
            if response.status_code >= 400:
                raise APIError(response.status_code, response.text)
            return response.json()
        except httpx.TimeoutException:
            raise TimeoutError(f"Request to {path} timed out")

    def _get(self, path: str, **kwargs) -> dict[str, Any]:
        """发送 GET 请求"""
        try:
            response = self._client.get(path, **kwargs)
            if response.status_code == 429:
                raise RateLimitError()
            if response.status_code >= 400:
                raise APIError(response.status_code, response.text)
            return response.json()
        except httpx.TimeoutException:
            raise TimeoutError(f"Request to {path} timed out")

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs
    ):
        """文本生成"""
        from quantumflow.sdk.models import InferenceRequest, SamplingParams, InferenceResponse

        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, **kwargs)
        request = InferenceRequest(
            model=model,
            prompt=prompt,
            sampling_params=sampling,
            stream=stream
        )

        data = self._post("/api/v1/inference/generate", json=request.to_dict())
        return InferenceResponse(
            request_id=data["request_id"],
            model=data["model"],
            generated_text=data["generated_text"],
            finish_reason=data["finish_reason"],
            latency_ms=data["latency_ms"],
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
            total_tokens=data["usage"]["total_tokens"],
        )

    def list_models(self) -> list[dict[str, Any]]:
        """获取模型列表"""
        return self._get("/api/v1/models")

    def health_check(self) -> dict[str, Any]:
        """健康检查"""
        return self._get("/api/v1/health")

    def close(self):
        """关闭客户端"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncQuantumFlowClient:
    """QuantumFlow 异步客户端"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        tenant_id: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout = timeout

    async def _arequest(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        """发送异步 HTTP 请求"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers,
        ) as client:
            try:
                response = await client.request(method, path, **kwargs)
                if response.status_code == 429:
                    raise RateLimitError()
                if response.status_code >= 400:
                    raise APIError(response.status_code, response.text)
                return response.json()
            except httpx.TimeoutException:
                raise TimeoutError(f"Request to {path} timed out")

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs
    ):
        """异步文本生成"""
        from quantumflow.sdk.models import InferenceRequest, SamplingParams, InferenceResponse

        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, **kwargs)
        request = InferenceRequest(
            model=model,
            prompt=prompt,
            sampling_params=sampling,
            stream=stream
        )

        data = await self._arequest("POST", "/api/v1/inference/generate", json=request.to_dict())
        return InferenceResponse(
            request_id=data["request_id"],
            model=data["model"],
            generated_text=data["generated_text"],
            finish_reason=data["finish_reason"],
            latency_ms=data["latency_ms"],
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
            total_tokens=data["usage"]["total_tokens"],
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """异步获取模型列表"""
        return await self._arequest("GET", "/api/v1/models")

    async def health_check(self) -> dict[str, Any]:
        """异步健康检查"""
        return await self._arequest("GET", "/api/v1/health")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


QuantumFlowSDK = SyncQuantumFlowClient