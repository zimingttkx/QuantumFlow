"""vLLM推理后端"""

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
        enforce_eager=getattr(config, "enforce_eager", False),
        # vLLM 0.21+ 的 Chunked Prefill 是内置默认行为，无需显式开关
    )


def _run_vllm_generate(llm, prompts, vllm_params):
    """同步执行 vLLM generate（供 run_in_executor 调用）"""
    return llm.generate(prompts, vllm_params)


class VLLMEngine(InferenceEngine):
    """vLLM推理引擎实现"""

    def __init__(self, config: dict[str, Any] = None):
        super().__init__(InferenceBackendType.VLLM)
        self.config = config or {}
        self._llm_instances: dict[str, Any] = {}
        self._async_engines: dict[str, Any] = {}

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
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[InferenceResult]:
        if model_name not in self._llm_instances:
            logger.error("model_not_loaded", model=model_name)
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=["[vLLM错误: 模型未加载]"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0,
                    finish_reason="error",
                    metrics={},
                )
                for i in range(len(prompts))
            ]

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
                prompt_tokens = (
                    len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
                )
                completion_tokens = (
                    len(output.outputs[0].token_ids)
                    if output.outputs[0].token_ids is not None
                    else 0
                )
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
                    outputs=[f"[vLLM错误: {str(e)}]"],
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
        """流式生成

        优先使用 vLLM AsyncLLMEngine（真实逐 token 流式）；
        若不可用则用 LLM.generate 同步获取完整结果，再按 token 拆分近似流式。
        """
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

            # 检测是否是 AsyncLLMEngine
            engine_class = type(llm).__name__
            if engine_class == "AsyncLLMEngine":
                # 真实流式
                async for output in self._real_stream(llm, prompt, vllm_params):
                    yield output
            else:
                # 同步 LLM：先获取完整结果，再按 token 拆分近似流式
                loop = asyncio.get_running_loop()
                outputs = await loop.run_in_executor(
                    None, _run_vllm_generate, llm, [prompt], vllm_params
                )
                for output in outputs:
                    for output_item in output.outputs:
                        token_ids = output_item.token_ids
                        text = output_item.text
                        if token_ids is not None and len(token_ids) > 0:
                            # 按 token 切分回退：用空白分块（无法 100% 还原 token 边界）
                            parts = text.split(" ") if text else [""]
                            for i, part in enumerate(parts):
                                chunk = part + (" " if i < len(parts) - 1 else "")
                                if chunk:
                                    yield chunk
                                    await asyncio.sleep(0.01)
                        elif text:
                            yield text
        except Exception as e:
            logger.error("stream_generate_error", model=model_name, error=str(e))

    async def _real_stream(self, async_engine, prompt: str, vllm_params):
        """使用 AsyncLLMEngine 进行真实逐 token 流式"""
        from vllm.utils import random_uuid

        request_id = f"stream-{random_uuid()}"
        # vLLM 0.4+ 使用 generate() 返回 async iterator
        try:
            results = async_engine.generate(prompt, vllm_params, request_id=request_id)
            async for result in results:
                if hasattr(result, "outputs") and result.outputs:
                    output_item = result.outputs[0]
                    new_text = getattr(output_item, "text", "")
                    if new_text:
                        yield new_text
                    # finish_reason 出现后停止
                    if getattr(output_item, "finish_reason", None):
                        return
        except Exception as e:
            logger.error("real_stream_error", error=str(e))
            raise

    async def get_stats(self, model_name: str) -> dict[str, float]:
        if model_name not in self._llm_instances:
            return {}

        try:
            import torch

            stats: dict[str, float] = {}
            if torch.cuda.is_available():
                # 当前进程看到的 torch 显存
                stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                stats["gpu_count_visible"] = float(torch.cuda.device_count())

            # 多 GPU 统计：遍历所有可见 GPU，不再硬编码 0
            try:
                from quantumflow.inference.gpu_monitor import _get_nvml_manager
                mgr = _get_nvml_manager()
                gpu_count = int(stats.get("gpu_count_visible", 1)) or 1
                utils = []
                mems = []
                temps = []
                for i in range(gpu_count):
                    try:
                        u = mgr.get_utilization(i)
                        m = mgr.get_memory_info(i)
                        t = mgr.get_temperature(i)
                        if u is not None:
                            utils.append(u)
                        if m is not None:
                            mems.append(m.get("utilization", 0.0))
                        if t is not None:
                            temps.append(t)
                    except Exception:
                        continue
                if utils:
                    stats["gpu_utilization_avg"] = sum(utils) / len(utils) / 100.0
                if mems:
                    stats["gpu_memory_utilization_avg"] = sum(mems) / len(mems) / 100.0
                if temps:
                    stats["gpu_temperature_avg"] = sum(temps) / len(temps)
            except Exception:
                # GPU monitor 不可用时不报错（vLLM 仍在工作）
                pass

            return stats
        except Exception:
            return {}
