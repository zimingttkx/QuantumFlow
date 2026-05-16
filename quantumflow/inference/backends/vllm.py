"""vLLM推理后端"""

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


class VLLMEngine(InferenceEngine):
    """vLLM推理引擎实现"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(InferenceBackendType.VLLM)
        self.config = config or {}
        self._llm_instances: Dict[str, Any] = {}  # vLLM LLM实例
        self._async_engines: Dict[str, Any] = {}  # 异步引擎

    async def initialize(self) -> bool:
        """初始化vLLM引擎"""
        try:
            # 导入vLLM
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
        """加载模型"""
        if not self._is_initialized:
            logger.error("vllm_not_initialized")
            return False

        try:
            from vllm import LLM, SamplingParams as VLLMSamplingParams

            logger.info(
                "loading_model",
                model=config.model_name,
                path=config.model_path,
                tensor_parallel=config.tensor_parallel,
            )

            # 创建LLM实例
            llm = LLM(
                model=config.model_path,
                tensor_parallel_size=config.tensor_parallel,
                pipeline_parallel_size=config.pipeline_parallel,
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_model_len=config.max_model_len,
                dtype=config.dtype,
                quantization=config.quantization,
                trust_remote_code=config.trust_remote_code,
                block_size=config.block_size,
                max_num_batched_tokens=config.max_num_batched_tokens,
                max_num_seqs=config.max_num_seqs,
                enforce_eager=config.enforce_eager,
            )

            self._llm_instances[config.model_name] = llm
            self._loaded_models[config.model_name] = config

            logger.info(
                "model_loaded",
                model=config.model_name,
                tensor_parallel=config.tensor_parallel,
            )

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
        if model_name in self._llm_instances:
            del self._llm_instances[model_name]
            del self._loaded_models[model_name]

            # 清理GPU内存
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
        """同步生成"""
        if model_name not in self._llm_instances:
            logger.error("model_not_loaded", model=model_name)
            return []

        try:
            from vllm import SamplingParams as VLLMSamplingParams

            llm = self._llm_instances[model_name]
            start_time = time.time()

            # 构建vLLM采样参数
            vllm_params = VLLMSamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                max_tokens=sampling_params.max_tokens,
                repetition_penalty=sampling_params.repetition_penalty,
                stop=sampling_params.stop,
            )

            # 执行推理
            outputs = llm.generate(prompts, vllm_params)

            latency_ms = (time.time() - start_time) * 1000

            # 转换结果
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
            logger.error(
                "generate_error",
                model=model_name,
                error=str(e),
            )
            return []

    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """流式生成"""
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
                stop=sampling_params.stop,
            )

            # 使用异步生成器
            results = llm.generate([prompt], vllm_params)

            for output in results:
                for output_item in output.outputs:
                    yield output_item.text

        except Exception as e:
            logger.error(
                "stream_generate_error",
                model=model_name,
                error=str(e),
            )

    async def get_stats(self, model_name: str) -> Dict[str, float]:
        """获取引擎统计"""
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
