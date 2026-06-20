"""TensorRT-LLM inference backend"""

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

logger = structlog.get_logger().bind(component="tensorrt_llm_backend")


class TensorRTLLMEngine(InferenceEngine):
    """TensorRT-LLM inference engine implementation"""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(InferenceBackendType.TRT_LLM)
        self.config = config or {}
        self._engines: dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Initialize TensorRT-LLM engine"""
        try:
            import tensorrt_llm

            logger.info("tensorrt_llm_initializing", version=tensorrt_llm.__version__)
            self._is_initialized = True
            logger.info("tensorrt_llm_initialized")
            return True
        except ImportError:
            logger.error("tensorrt_llm_not_installed")
            return False
        except Exception as e:
            logger.error("tensorrt_llm_init_error", error=str(e))
            return False

    async def load_model(self, config: ModelConfig) -> bool:
        """Load TensorRT-LLM model"""
        if not self._is_initialized:
            logger.error("tensorrt_llm_not_initialized")
            return False

        try:
            logger.info(
                "loading_model",
                model=config.model_name,
                path=config.model_path,
                tensor_parallel=config.tensor_parallel,
            )

            loop = asyncio.get_running_loop()
            engine = await loop.run_in_executor(None, self._build_engine, config)

            self._engines[config.model_name] = engine
            self._loaded_models[config.model_name] = config

            logger.info("model_loaded", model=config.model_name)
            return True

        except Exception as e:
            logger.error("model_load_error", model=config.model_name, error=str(e))
            return False

    def _build_engine(self, config: ModelConfig):
        """Build TensorRT-LLM engine"""
        import os
        from tensorrt_llm import LLM

        # Set environment variable
        os.environ.setdefault("TRT_LLM_ENGINE_DIR", config.model_path)

        # Create LLM instance
        engine = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel,
            dtype=config.dtype,
        )
        return engine

    async def unload_model(self, model_name: str) -> bool:
        """Unload model"""
        if model_name in self._engines:
            del self._engines[model_name]
            del self._loaded_models[model_name]

            # 清理 GPU 显存
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
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[InferenceResult]:
        """同步生成"""
        if model_name not in self._engines:
            logger.error("model_not_loaded", model=model_name)
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=[f"[TensorRT-LLM错误: 模型 '{model_name}' 未加载]"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0,
                    finish_reason="error",
                    metrics={},
                )
                for i in range(len(prompts))
            ]

        try:
            from tensorrt_llm import SamplingParams as TRTLSamplingParams

            engine = self._engines[model_name]
            start_time = time.time()

            trtl_params = TRTLSamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                max_tokens=sampling_params.max_tokens,
                repetition_penalty=sampling_params.repetition_penalty,
                stop=sampling_params.stop,
            )

            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(
                None, engine.generate, prompts, trtl_params
            )

            latency_ms = (time.time() - start_time) * 1000

            results = []
            for i, output in enumerate(outputs):
                output_text = output.outputs[0].text if hasattr(output.outputs[0], 'text') else str(output.outputs[0])
                finish_reason = getattr(output.outputs[0], 'finish_reason', 'stop') or 'stop'
                prompt_tokens = getattr(output, 'prompt_tokens', 0)
                completion_tokens = len(getattr(output.outputs[0], 'token_ids', [])) if hasattr(output.outputs[0], 'token_ids') else 0

                results.append(
                    InferenceResult(
                        request_id=f"{model_name}_{i}",
                        outputs=[output_text],
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        finish_reason=finish_reason,
                        metrics={},
                    )
                )

            return results

        except Exception as e:
            logger.error("generate_error", model=model_name, error=str(e))
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=[f"[TensorRT-LLM错误: {str(e)}]"],
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
        if model_name not in self._engines:
            logger.error("model_not_loaded", model=model_name)
            return

        try:
            from tensorrt_llm import SamplingParams as TRTLSamplingParams

            engine = self._engines[model_name]

            trtl_params = TRTLSamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                max_tokens=sampling_params.max_tokens,
                repetition_penalty=sampling_params.repetition_penalty,
                stop=sampling_params.stop,
            )

            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(
                None, engine.generate, [prompt], trtl_params
            )

            for output in outputs:
                for output_item in output.outputs:
                    text = getattr(output_item, 'text', '') or str(output_item)
                    if text:
                        for char in text:
                            if char:
                                yield char
                                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error("stream_generate_error", model=model_name, error=str(e))

    async def get_stats(self, model_name: str) -> dict[str, float]:
        """获取引擎统计"""
        if model_name not in self._engines:
            return {}

        try:
            import torch

            stats: dict[str, float] = {}
            if torch.cuda.is_available():
                stats["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)
                stats["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)

                try:
                    import pynvml

                    pynvml.nvmlInit()
                    try:
                        gpu_count = pynvml.nvmlDeviceGetCount()
                        total_util = 0.0
                        total_mem_util = 0.0
                        for i in range(gpu_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            total_util += util.gpu
                            total_mem_util += util.memory
                        if gpu_count > 0:
                            stats["gpu_utilization"] = (total_util / gpu_count) / 100.0
                            stats["gpu_memory_utilization"] = (total_mem_util / gpu_count) / 100.0
                    finally:
                        pynvml.nvmlShutdown()
                except Exception:
                    pass

            return stats
        except Exception:
            return {}
