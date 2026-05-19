"""HuggingFace TGI推理后端"""

from typing import List, Dict, Optional, Any, AsyncIterator
import time
import asyncio
import structlog

from quantumflow.inference.engine import (
    InferenceEngine,
    ModelConfig,
    SamplingParams,
    InferenceResult,
)
from quantumflow.core.constants import InferenceBackendType

logger = structlog.get_logger().bind(component="tgi_backend")


class TGIEngine(InferenceEngine):
    """
    HuggingFace Text Generation Inference (TGI) 引擎实现

    TGI 是 HuggingFace 官方的高性能推理服务器，
    支持 FlashAttention、连续批处理、PagedAttention 等优化。
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        base_url: str = "http://localhost:8080",
    ):
        super().__init__(InferenceBackendType.TGI)
        self.config = config or {}
        self.base_url = base_url.rstrip("/")
        self._client: Optional[Any] = None
        self._timeout = 300  # 超时时间（秒）

    async def initialize(self) -> bool:
        """初始化TGI客户端"""
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
                logger.info("tgi_initialized", base_url=self.base_url)
                return True

            logger.error("tgi_health_check_failed")
            return False

        except ImportError:
            logger.error("httpx_not_installed")
            return False
        except Exception as e:
            logger.error("tgi_init_error", error=str(e))
            return False

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()

    async def load_model(self, config: ModelConfig) -> bool:
        """
        加载模型

        注意：TGI服务器需要在外部启动，此方法用于跟踪模型状态
        """
        if not self._client:
            logger.error("tgi_client_not_initialized")
            return False

        try:
            # 检查模型是否可用
            response = await self._client.get("/info")
            if response.status_code != 200:
                logger.error("tgi_info_request_failed")
                return False

            info = response.json()
            available_models = info.get("model_ids", [])

            # 如果TGI已加载特定模型
            if available_models:
                for model_id in available_models:
                    if config.model_name.lower() in model_id.lower():
                        self._loaded_models[config.model_name] = config
                        logger.info("model_tracked", model=config.model_name)
                        return True

            # 假设TGI运行在模型模式下
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
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[InferenceResult]:
        """同步生成"""
        if not self._client:
            logger.error("tgi_client_not_initialized")
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=[f"[TGI错误: 客户端未初始化]"],
                    prompt_tokens=0, completion_tokens=0, latency_ms=0,
                    finish_reason="error", metrics={},
                )
                for i in range(len(prompts))
            ]

        if not prompts:
            return []

        try:
            start_time = time.time()

            # 构建请求
            payload = {
                "inputs": prompts if len(prompts) > 1 else prompts[0],
                "parameters": {
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "max_new_tokens": sampling_params.max_tokens,
                    "repetition_penalty": sampling_params.repetition_penalty,
                },
            }

            if sampling_params.top_k > 0:
                payload["parameters"]["top_k"] = sampling_params.top_k

            if sampling_params.stop:
                payload["parameters"]["stop"] = sampling_params.stop

            # 单请求 vs 批量请求
            if len(prompts) == 1:
                response = await self._client.post(
                    "/generate",
                    json=payload,
                )
            else:
                response = await self._client.post(
                    "/generate_batch",
                    json=payload,
                )

            if response.status_code != 200:
                logger.error(
                    "tgi_generate_failed",
                    status_code=response.status_code,
                )
                return []

            result = response.json()
            latency_ms = (time.time() - start_time) * 1000

            # 解析响应
            results = []
            if isinstance(result.get("generated_text"), list):
                generated_texts = result["generated_text"]
                prompt_tokens_val = result.get("prompt_tokens", 0)
                generated_tokens_val = result.get("generated_tokens", 0)

                for i, text in enumerate(generated_texts):
                    if isinstance(prompt_tokens_val, list) and i < len(prompt_tokens_val):
                        pt = prompt_tokens_val[i]
                    elif isinstance(prompt_tokens_val, (int, float)) and len(generated_texts) > 0:
                        pt = int(prompt_tokens_val) // len(generated_texts)
                    else:
                        pt = len(prompts[i]) // 4 if i < len(prompts) else 0

                    if isinstance(generated_tokens_val, list) and i < len(generated_tokens_val):
                        ct = generated_tokens_val[i]
                    elif isinstance(generated_tokens_val, (int, float)) and len(generated_texts) > 0:
                        ct = int(generated_tokens_val) // len(generated_texts)
                    else:
                        ct = len(text.split())

                    results.append(
                        InferenceResult(
                            request_id=f"{model_name}_{i}",
                            outputs=[text],
                            prompt_tokens=pt,
                            completion_tokens=ct,
                            latency_ms=latency_ms,
                            finish_reason="stop",
                            metrics={},
                        )
                    )
            else:
                results.append(
                    InferenceResult(
                        request_id=f"{model_name}_0",
                        outputs=[result.get("generated_text", "")],
                        prompt_tokens=result.get("prompt_tokens", 0),
                        completion_tokens=result.get("generated_tokens", 0),
                        latency_ms=latency_ms,
                        finish_reason="stop",
                        metrics={},
                    )
                )

            return results

        except asyncio.TimeoutError:
            logger.error("tgi_request_timeout", model=model_name)
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=[f"[TGI超时]"],
                    prompt_tokens=0, completion_tokens=0, latency_ms=0,
                    finish_reason="error", metrics={},
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
                    outputs=[f"[TGI错误: {str(e)}]"],
                    prompt_tokens=0, completion_tokens=0, latency_ms=0,
                    finish_reason="error", metrics={},
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
            logger.error("tgi_client_not_initialized")
            return

        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "max_new_tokens": sampling_params.max_tokens,
                    "repetition_penalty": sampling_params.repetition_penalty,
                },
            }

            if sampling_params.top_k > 0:
                payload["parameters"]["top_k"] = sampling_params.top_k
            if sampling_params.stop:
                payload["parameters"]["stop"] = sampling_params.stop

            async with self._client.stream(
                "POST",
                "/generate_stream",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    logger.error(
                        "tgi_stream_failed",
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
                            token = chunk.get("token", {})
                            text = token.get("text", "")
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue

        except asyncio.TimeoutError:
            logger.error("tgi_stream_timeout", model=model_name)
        except Exception as e:
            logger.error(
                "stream_generate_error",
                model=model_name,
                error=str(e),
            )

    async def get_stats(self, model_name: str) -> Dict[str, float]:
        """获取引擎统计"""
        if not self._client:
            return {}

        try:
            response = await self._client.get("/info")
            if response.status_code == 200:
                return {"healthy": 1.0}

            return {}

            return {}

        except Exception:
            return {}
