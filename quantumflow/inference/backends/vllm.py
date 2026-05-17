"""vLLM推理后端"""

import asyncio
from typing import List, Dict, Optional, Any, AsyncIterator
import time
import structlog

from quantumflow.inference.engine import (
    InferenceEngine,
    ModelConfig,
    SamplingParams,
    InferenceResult,
)
from quantumflow.core.constants import InferenceBackendType

logger = structlog.get_logger().bind(component="vllm_backend")


def _build_vllm_llm(config: ModelConfig):
    """同步构造 vLLM LLM 实例（供 run_in_executor 调用）"""
    import os
    from vllm import LLM

    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    return LLM(
        model=config.model_path,
        tensor_parallel_size=config.tensor_parallel,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        dtype=config.dtype,
        quantization=config.quantization,
        trust_remote_code=config.trust_remote_code,
        enforce_eager=getattr(config, 'enforce_eager', False),
    )


def _run_vllm_generate(llm, prompts, vllm_params):
    """同步执行 vLLM generate（供 run_in_executor 调用）"""
    return llm.generate(prompts, vllm_params)


class VLLMEngine(InferenceEngine):
    """vLLM推理引擎实现"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(InferenceBackendType.VLLM)
        self.config = config or {}
        self._llm_instances: Dict[str, Any] = {}
        self._async_engines: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        try:
            import vllm
            logger.info("vllm_initializing", version=vllm.__version__)
            self._is_initialized = True
            logger.info("vllm_initialized")
            return True
        except ImportError:
            logger.error("vllm_not_installed")
            return False
        except Exception as e:
            logger.error("vllm_init_error", error=str(e))
            return False

    async def load_model(self, config: ModelConfig) -> bool:
        if not self._is_initialized:
            logger.error("vllm_not_initialized")
            return False

        try:
            logger.info(
                "loading_model",
                model=config.model_name,
                path=config.model_path,
                tensor_parallel=config.tensor_parallel,
            )

            loop = asyncio.get_running_loop()
            llm = await loop.run_in_executor(None, _build_vllm_llm, config)

            self._llm_instances[config.model_name] = llm
            self._loaded_models[config.model_name] = config

            logger.info(
                "model_loaded",
                model=config.model_name,
                tensor_parallel=config.tensor_parallel,
            )
            return True

        except Exception as e:
            logger.error("model_load_error", model=config.model_name, error=str(e))
            return False

    async def unload_model(self, model_name: str) -> bool:
        if model_name in self._llm_instances:
            del self._llm_instances[model_name]
            del self._loaded_models[model_name]

            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info("model_unloaded", model=model_name)
            return True
        return False

    async def generate(
        self,
        model_name: str,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[InferenceResult]:
        if model_name not in self._llm_instances:
            logger.error("model_not_loaded", model=model_name)
            return []

        try:
            from vllm import SamplingParams as VLLMSamplingParams

            llm = self._llm_instances[model_name]
            start_time = time.time()

            vllm_params = VLLMSamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                max_tokens=sampling_params.max_tokens,
                repetition_penalty=sampling_params.repetition_penalty,
                stop=sampling_params.stop,
            )

            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(
                None, _run_vllm_generate, llm, prompts, vllm_params
            )

            latency_ms = (time.time() - start_time) * 1000

            results = []
            for i, output in enumerate(outputs):
                output_text = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason or "stop"
                results.append(
                    InferenceResult(
                        request_id=f"{model_name}_{i}",
                        outputs=[output_text],
                        prompt_tokens=len(output.prompt_token_ids),
                        completion_tokens=len(output.outputs[0].token_ids),
                        latency_ms=latency_ms,
                        finish_reason=finish_reason,
                        metrics={},
                    )
                )

            return results

        except Exception as e:
            logger.error("generate_error", model=model_name, error=str(e))
            return []

    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """流式生成 — 在 executor 中完成推理，逐词 yield 以模拟 token 级流式输出"""
        if model_name not in self._llm_instances:
            logger.error("model_not_loaded", model=model_name)
            return

        try:
            from vllm import SamplingParams as VLLMSamplingParams

            llm = self._llm_instances[model_name]

            vllm_params = VLLMSamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                max_tokens=sampling_params.max_tokens,
                repetition_penalty=sampling_params.repetition_penalty,
                stop=sampling_params.stop,
            )

            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(
                None, _run_vllm_generate, llm, [prompt], vllm_params
            )

            # 逐词 yield，模拟流式效果
            for output in outputs:
                for output_item in output.outputs:
                    text = output_item.text
                    if text:
                        # 按空白字符拆分，逐个 yield 单词+空格
                        parts = text.split(' ')
                        for i, part in enumerate(parts):
                            chunk = part + (' ' if i < len(parts) - 1 else '')
                            if chunk:
                                yield chunk
                                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error("stream_generate_error", model=model_name, error=str(e))

    async def get_stats(self, model_name: str) -> Dict[str, float]:
        if model_name not in self._llm_instances:
            return {}

        try:
            llm = self._llm_instances[model_name]
            stats = llm.get_stats()
            return {
                "num_requests": stats.get("num_requests", 0),
                "num_running": stats.get("num_running", 0),
                "num_waiting": stats.get("num_waiting", 0),
                "gpu_memory_usage": stats.get("gpu_memory_usage", 0),
            }
        except Exception:
            return {}
