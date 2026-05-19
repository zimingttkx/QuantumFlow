"""SGLang推理后端"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from quantumflow.core.constants import InferenceBackendType
from quantumflow.inference.engine import (
    InferenceEngine,
    InferenceResult,
    ModelConfig,
    SamplingParams,
)

logger = structlog.get_logger().bind(component="sglang_backend")


class SGLangEngine(InferenceEngine):
    """
    SGLang推理引擎实现

    SGLang 是一个基于 RadixAttention 的高性能推理框架，
    支持树搜索、结构化输出、思维链等高级特性。
    """

    def __init__(
        self,
        config: dict[str, Any] = None,
        base_url: str = "http://localhost:30000",
    ):
        super().__init__(InferenceBackendType.SGLANG)
        self.config = config or {}
        self.base_url = base_url.rstrip("/")
        self._client: Any | None = None
        self._timeout = 300

    async def initialize(self) -> bool:
        """初始化SGLang客户端"""
        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._timeout),
            )

            # 健康检查
            response = await self._client.get("/health")
            if response.status_code == 200:
                self._is_initialized = True
                logger.info("sglang_initialized", base_url=self.base_url)
                return True

            logger.error("sglang_health_check_failed")
            return False

        except ImportError:
            logger.error("httpx_not_installed")
            return False
        except Exception as e:
            logger.error("sglang_init_error", error=str(e))
            return False

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()

    async def load_model(self, config: ModelConfig) -> bool:
        """
        加载模型

        注意：SGLang服务器需要在外部启动，此方法用于跟踪模型状态
        """
        if not self._client:
            logger.error("sglang_client_not_initialized")
            return False

        try:
            # 检查服务器状态
            response = await self._client.get("/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                for model in models:
                    if config.model_name.lower() in model.get("id", "").lower():
                        self._loaded_models[config.model_name] = config
                        logger.info("model_tracked", model=config.model_name)
                        return True

            # 如果服务器已运行，假设模型已加载
            self._loaded_models[config.model_name] = config
            logger.info("model_tracked", model=config.model_name)
            return True

        except Exception as e:
            logger.error(
                "model_load_error",
                model=config.model_name,
                error=str(e),
            )
            return False

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            logger.info("model_untracked", model=model_name)
            return True
        return False

    async def generate(
        self,
        model_name: str,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[InferenceResult]:
        """同步生成"""
        if not self._client:
            logger.error("sglang_client_not_initialized")
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=["[SGLang错误: 客户端未初始化]"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0,
                    finish_reason="error",
                    metrics={},
                )
                for i in range(len(prompts))
            ]

        try:
            time.time()

            async def _single_generate(prompt: str, index: int) -> InferenceResult:
                """发送单个推理请求"""
                req_start = time.time()
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": sampling_params.max_tokens,
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                }
                if sampling_params.stop:
                    payload["stop"] = sampling_params.stop

                try:
                    response = await self._client.post("/v1/completions", json=payload)
                    if response.status_code != 200:
                        logger.error(
                            "sglang_generate_failed",
                            status_code=response.status_code,
                            detail=response.text,
                        )
                        return InferenceResult(
                            request_id=f"{model_name}_{index}",
                            outputs=[f"[SGLang错误: HTTP {response.status_code}]"],
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency_ms=(time.time() - req_start) * 1000,
                            finish_reason="error",
                            metrics={},
                        )

                    result = response.json()
                    choices = result.get("choices", [])
                    if not choices:
                        return InferenceResult(
                            request_id=f"{model_name}_{index}",
                            outputs=["[SGLang错误: 空响应]"],
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency_ms=(time.time() - req_start) * 1000,
                            finish_reason="error",
                            metrics={},
                        )

                    choice = choices[0]
                    return InferenceResult(
                        request_id=f"{model_name}_{index}",
                        outputs=[choice.get("text", "")],
                        prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                        latency_ms=(time.time() - req_start) * 1000,
                        finish_reason=choice.get("finish_reason", "stop"),
                        metrics={},
                    )
                except asyncio.TimeoutError:
                    return InferenceResult(
                        request_id=f"{model_name}_{index}",
                        outputs=["[SGLang超时]"],
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=(time.time() - req_start) * 1000,
                        finish_reason="error",
                        metrics={},
                    )

            # 并行发送所有请求
            tasks = [_single_generate(prompts[i], i) for i in range(len(prompts))]
            results = await asyncio.gather(*tasks)
            return list(results)

        except asyncio.TimeoutError:
            logger.error("sglang_request_timeout", model=model_name)
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=["[SGLang超时]"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0,
                    finish_reason="error",
                    metrics={},
                )
                for i in range(len(prompts))
            ]
        except Exception as e:
            logger.error(
                "generate_error",
                model=model_name,
                error=str(e),
            )
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=[f"[SGLang错误: {str(e)}]"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0,
                    finish_reason="error",
                    metrics={},
                )
                for i in range(len(prompts))
            ]

    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """流式生成"""
        if not self._client:
            logger.error("sglang_client_not_initialized")
            return  # async generator 提前结束，async for 会正常结束

        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "stream": True,
            }

            if sampling_params.stop:
                payload["stop"] = sampling_params.stop

            async with self._client.stream(
                "POST",
                "/v1/completions",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    logger.error(
                        "sglang_stream_failed",
                        status_code=response.status_code,
                    )
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break

                        import json

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            choice = choices[0]
                            # /v1/completions SSE: text 在 choice.text
                            # /v1/chat/completions SSE: text 在 choice.delta.text
                            text = choice.get("text", "") or choice.get("delta", {}).get("text", "")
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue

        except asyncio.TimeoutError:
            logger.error("sglang_stream_timeout", model=model_name)
        except Exception as e:
            logger.error(
                "stream_generate_error",
                model=model_name,
                error=str(e),
            )

    async def get_stats(self, model_name: str) -> dict[str, float]:
        """获取引擎统计"""
        if not self._client:
            return {}

        try:
            response = await self._client.get("/v1/models")
            if response.status_code == 200:
                return {"healthy": 1.0}

            return {}

        except Exception:
            return {}

    # ==================== SGLang特有功能 ====================

    async def chat(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        sampling_params: SamplingParams,
    ) -> InferenceResult:
        """
        SGLang特有的Chat接口

        Args:
            model_name: 模型名称
            messages: 消息列表 [{"role": "user", "content": "..."}]
            sampling_params: 采样参数

        Returns:
            推理结果
        """
        if not self._client:
            logger.error("sglang_client_not_initialized")
            return InferenceResult(
                request_id=f"{model_name}_chat",
                outputs=["[SGLang错误: 客户端未初始化]"],
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                finish_reason="error",
                metrics={},
            )

        try:
            start_time = time.time()

            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
            }

            if sampling_params.stop:
                payload["stop"] = sampling_params.stop

            response = await self._client.post("/v1/chat/completions", json=payload)

            if response.status_code != 200:
                logger.error("sglang_chat_failed", status_code=response.status_code)
                return InferenceResult(
                    request_id=f"{model_name}_chat",
                    outputs=[f"[SGLang错误: HTTP {response.status_code}]"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    finish_reason="error",
                    metrics={},
                )

            result = response.json()
            latency_ms = (time.time() - start_time) * 1000

            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})

            return InferenceResult(
                request_id=f"{model_name}_chat",
                outputs=[message.get("content", "")],
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                latency_ms=latency_ms,
                finish_reason=choice.get("finish_reason", "stop"),
                metrics={},
            )

        except Exception as e:
            logger.error("chat_error", model=model_name, error=str(e))
            return InferenceResult(
                request_id=f"{model_name}_chat",
                outputs=[f"[SGLang错误: {str(e)}]"],
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                finish_reason="error",
                metrics={},
            )
