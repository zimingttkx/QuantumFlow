"""HuggingFace TGI推理后端"""

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

logger = structlog.get_logger().bind(component="tgi_backend")


class TGIEngine(InferenceEngine):
    """
    HuggingFace Text Generation Inference (TGI) 引擎实现

    TGI 是 HuggingFace 官方的高性能推理服务器，
    支持 FlashAttention、连续批处理、PagedAttention 等优化。
    """

    def __init__(
        self,
        config: dict[str, Any] = None,
        base_url: str = "http://localhost:8080",
    ):
        super().__init__(InferenceBackendType.TGI)
        self.config = config or {}
        self.base_url = base_url.rstrip("/")
        self._client: Any | None = None
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
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[InferenceResult]:
        """同步生成"""
        if not self._client:
            logger.error("tgi_client_not_initialized")
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=["[TGI错误: 客户端未初始化]"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0,
                    finish_reason="error",
                    metrics={},
                )
                for i in range(len(prompts))
            ]

        if not prompts:
            return []

        try:
            start_time = time.time()

            # 构建请求
            # 重要：top_k 和 truncate 是完全独立的参数！
            # top_k 控制采样阶段的 token 候选数量
            # truncate 控制输入序列的最大长度
            # 两者不能混淆！
            payload: dict[str, Any] = {
                "inputs": prompts if len(prompts) > 1 else prompts[0],
                "parameters": {
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "max_new_tokens": sampling_params.max_tokens,
                    "repetition_penalty": sampling_params.repetition_penalty,
                    # 明确不传递 truncate，即使 top_k > 0
                    # truncate 由 ModelConfig 单独控制，不应与 top_k 混淆
                },
            }

            # top_k: 仅控制采样，不影响输入处理
            if sampling_params.top_k > 0:
                payload["parameters"]["top_k"] = sampling_params.top_k

            # stop 序列
            if sampling_params.stop:
                payload["parameters"]["stop"] = sampling_params.stop

            # details: 请求返回 token 级详情（用于调试和精确统计）
            if hasattr(sampling_params, "details") and sampling_params.details:
                payload["parameters"]["details"] = True
                payload["parameters"]["decoder_input_details"] = True

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
                    body=response.text[:500],
                )
                # 返回 N 条对齐的错误结果（不返回 []，避免与 caller 索引错位）
                return [
                    InferenceResult(
                        request_id=f"{model_name}_{i}",
                        outputs=[f"[TGI错误: HTTP {response.status_code}]"],
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=(time.time() - start_time) * 1000,
                        finish_reason="error",
                        metrics={},
                    )
                    for i in range(len(prompts))
                ]

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
                    elif (
                        isinstance(generated_tokens_val, (int, float)) and len(generated_texts) > 0
                    ):
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
                text = result.get("generated_text", "")
                # Only use the fallback when the key is not present in the response
                if "generated_tokens" in result:
                    generated_tokens_val = result["generated_tokens"]
                    if isinstance(generated_tokens_val, (int, float)):
                        ct = int(generated_tokens_val)
                    else:
                        ct = len(text.split())
                else:
                    ct = len(text.split())
                results.append(
                    InferenceResult(
                        request_id=f"{model_name}_0",
                        outputs=[text],
                        prompt_tokens=result.get("prompt_tokens", 0),
                        completion_tokens=ct,
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
                    outputs=["[TGI超时]"],
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
                    outputs=[f"[TGI错误: {str(e)}]"],
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
            logger.error("tgi_client_not_initialized")
            return

        try:
            payload: dict[str, Any] = {
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
            if hasattr(sampling_params, "details") and sampling_params.details:
                payload["parameters"]["details"] = True

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
                    # 跳过空行
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if not data:
                            continue

                        if data == "[DONE]":
                            break

                        import json

                        try:
                            chunk = json.loads(data)
                            # 尝试从 token.text 获取
                            token = chunk.get("token", {})
                            text = token.get("text", "")
                            # 备用：从 generated_text 获取（TGI 某些版本格式）
                            if not text:
                                text = chunk.get("generated_text", "")
                            # 备用：从 content 获取
                            if not text:
                                choices = chunk.get("choices", [])
                                if choices:
                                    text = choices[0].get("text", "")

                            if text:
                                yield text
                        except json.JSONDecodeError:
                            # 忽略无法解析的行，继续处理后续数据
                            logger.debug("tgi_stream_json_decode_error", raw_data=data[:100])
                            continue
                        except Exception as e:
                            # 记录其他解析错误但继续处理
                            logger.debug("tgi_stream_parse_error", error=str(e), raw_data=data[:100])
                            continue

        except asyncio.TimeoutError:
            logger.error("tgi_stream_timeout", model=model_name)
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
            response = await self._client.get("/info")
            if response.status_code == 200:
                return {"healthy": 1.0}

            return {"healthy": 0.0}

        except asyncio.TimeoutError:
            logger.warning("tgi_stats_timeout")
            return {"healthy": 0.0, "timeout": 1.0}
        except Exception:
            return {"healthy": 0.0}
