"""HuggingFace推理后端"""

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

logger = structlog.get_logger().bind(component="hf_backend")


class HuggingFaceEngine(InferenceEngine):
    """HuggingFace推理引擎实现"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(InferenceBackendType.HUGGINGFACE)
        self.config = config or {}
        self._models: Dict[str, Any] = {}  # Transformers模型实例
        self._tokenizers: Dict[str, Any] = {}  # Tokenizer实例

    async def initialize(self) -> bool:
        """初始化HuggingFace引擎"""
        try:
            import torch
            import transformers

            logger.info(
                "huggingface_initializing",
                torch_version=torch.__version__,
                transformers_version=transformers.__version__,
            )

            # 检查CUDA是否可用
            if torch.cuda.is_available():
                logger.info("cuda_available", device_count=torch.cuda.device_count())
            else:
                logger.warning("cuda_not_available")

            self._is_initialized = True
            logger.info("huggingface_initialized")
            return True

        except ImportError as e:
            logger.error("huggingface_import_error", error=str(e))
            return False
        except Exception as e:
            logger.error("huggingface_init_error", error=str(e))
            return False

    async def load_model(self, config: ModelConfig) -> bool:
        """加载模型"""
        if not self._is_initialized:
            logger.error("huggingface_not_initialized")
            return False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(
                "loading_model",
                model=config.model_name,
                path=config.model_path,
                dtype=config.dtype,
            )

            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code,
            )
            self._tokenizers[config.model_name] = tokenizer

            # 确定device和dtype
            device = "cuda" if config.dtype != "cpu" else "cpu"
            torch_dtype = self._get_torch_dtype(config.dtype)

            model_kwargs = {}
            try:
                from transformers import AutoConfig
                model_config = AutoConfig.from_pretrained(
                    config.model_path,
                    trust_remote_code=config.trust_remote_code,
                )
                if hasattr(model_config, 'rope_scaling') and isinstance(model_config.rope_scaling, dict):
                    if 'type' not in model_config.rope_scaling and 'rope_type' in model_config.rope_scaling:
                        model_config.rope_scaling['type'] = model_config.rope_scaling['rope_type']
                        logger.info("fixed_rope_scaling", model=config.model_name)
                model_kwargs['config'] = model_config
            except Exception:
                pass

            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=config.trust_remote_code,
                low_cpu_mem_usage=True,
                **model_kwargs,
            )

            if device == "cpu":
                model = model.to(device)

            self._models[config.model_name] = model
            self._loaded_models[config.model_name] = config

            logger.info(
                "model_loaded",
                model=config.model_name,
                device=str(model.device) if hasattr(model, 'device') else "unknown",
            )

            return True

        except Exception as e:
            logger.error(
                "model_load_error",
                model=config.model_name,
                error=str(e),
            )
            return False

    def _get_torch_dtype(self, dtype_str: str):
        """将字符串dtype转换为torch dtype"""
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.float16,  # HuggingFace默认用float16
        }
        return dtype_map.get(dtype_str, torch.float16)

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name in self._models:
            del self._models[model_name]
            del self._tokenizers[model_name]
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
        if model_name not in self._models:
            logger.error("model_not_loaded", model=model_name)
            return []

        try:
            import torch
            from transformers import GenerationConfig

            model = self._models[model_name]
            tokenizer = self._tokenizers[model_name]

            results = []
            start_time = time.time()

            for i, prompt in enumerate(prompts):
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt")
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # 生成
                generation_config = GenerationConfig(
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    top_k=sampling_params.top_k,
                    max_new_tokens=sampling_params.max_tokens,
                    repetition_penalty=sampling_params.repetition_penalty,
                    do_sample=sampling_params.temperature > 0,
                )

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config,
                        stop_strings=sampling_params.stop,
                    )

                # 解码 - 只解码新生成的token
                prompt_tokens = inputs["input_ids"].shape[1]
                completion_tokens = outputs.shape[1] - prompt_tokens
                new_tokens = outputs[0][prompt_tokens:]
                output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                results.append(
                    InferenceResult(
                        request_id=f"{model_name}_{i}",
                        outputs=[output_text],
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=(time.time() - start_time) * 1000,
                        finish_reason="stop",
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
        if model_name not in self._models:
            logger.error("model_not_loaded", model=model_name)
            return

        try:
            import torch
            from transformers import TextIteratorStreamer
            from threading import Thread

            model = self._models[model_name]
            tokenizer = self._tokenizers[model_name]

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 创建流式器
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # 生成配置
            generation_config = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
                "max_new_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "repetition_penalty": sampling_params.repetition_penalty,
                "streamer": streamer,
                "do_sample": sampling_params.temperature > 0,
            }

            # 在后台线程生成
            thread = Thread(target=model.generate, kwargs=generation_config)
            thread.start()

            # 流式输出
            for text in streamer:
                yield text

            thread.join()

        except Exception as e:
            logger.error(
                "stream_generate_error",
                model=model_name,
                error=str(e),
            )

    async def get_stats(self, model_name: str) -> Dict[str, float]:
        """获取引擎统计"""
        if model_name not in self._models:
            return {}

        try:
            import torch

            model = self._models[model_name]
            stats = {
                "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            }
            return stats

        except Exception:
            return {}
