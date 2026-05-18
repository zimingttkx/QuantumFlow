"""HuggingFace推理后端"""

import asyncio
from typing import List, Dict, Optional, Any, AsyncIterator
import time
import structlog
import torch

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

            # use trust_remote_code=False for built‑in modeling code
            # (more reliable than cached custom code, handles rope_scaling correctly)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )

            if device == "cpu":
                model = model.to(device)

            # torch.compile 优化 — 减少 Python 开销和 kernel launch 次数
            # 对 7B+ 大模型收益显著（编译开销高，小模型不值得）
            if device == "cuda" and getattr(config, 'torch_compile', False):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("torch_compile_applied", model=config.model_name)
                except Exception as compile_err:
                    logger.warning("torch_compile_skipped", model=config.model_name, reason=str(compile_err))

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
        """
        批量生成 — BatchAccumulator 在上层 50ms 窗口内合并并发请求，
        此处接收多个 prompt 一次性批量处理。
        """
        if model_name not in self._models:
            logger.error("model_not_loaded", model=model_name)
            return []

        model = self._models[model_name]
        tokenizer = self._tokenizers[model_name]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        start_time = time.time()

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

        gen_kwargs = {
            "max_new_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature if sampling_params.temperature > 0 else 1.0,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            "repetition_penalty": sampling_params.repetition_penalty,
            "do_sample": sampling_params.temperature > 0,
            "use_cache": False,
        }
        if sampling_params.stop:
            gen_kwargs["stop_strings"] = sampling_params.stop

        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            results: List[InferenceResult] = []
            for i in range(len(prompts)):
                prompt_len = int(prompt_lens[i])
                pad_offset = (outputs[i] == tokenizer.pad_token_id).sum().item() if tokenizer.pad_token_id else 0
                start_idx = pad_offset + prompt_len
                new_tokens = outputs[i][start_idx:]
                output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                results.append(
                    InferenceResult(
                        request_id=f"{model_name}_{i}",
                        outputs=[output_text],
                        prompt_tokens=prompt_len,
                        completion_tokens=max(len(new_tokens), 0),
                        latency_ms=(time.time() - start_time) * 1000,
                        finish_reason="stop" if len(new_tokens) < sampling_params.max_tokens else "length",
                        metrics={},
                    )
                )
        except Exception as e:
            logger.error("generate_error", model=model_name, error=str(e))

        return results


    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """流式生成 — 使用 asyncio 队列桥接后台生成线程，不阻塞事件循环"""
        if model_name not in self._models:
            logger.error("model_not_loaded", model=model_name)
            return

        try:
            from transformers import TextIteratorStreamer
            from threading import Thread

            model = self._models[model_name]
            tokenizer = self._tokenizers[model_name]

            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
                "max_new_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature if sampling_params.temperature > 0 else 1.0,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "repetition_penalty": sampling_params.repetition_penalty,
                "streamer": streamer,
                "do_sample": sampling_params.temperature > 0,
                "use_cache": False,
            }

            # 后台线程跑 model.generate，通过 streamer 传递 token
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # 通过 async queue 桥接同步 streamer → async generator
            import queue
            q: queue.Queue = queue.Queue()

            def _enqueue():
                try:
                    for text in streamer:
                        q.put(('token', text))
                    q.put(('done', None))
                except Exception as exc:
                    q.put(('error', exc))

            import concurrent.futures
            loop = asyncio.get_running_loop()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            fut = loop.run_in_executor(executor, _enqueue)

            while True:
                try:
                    kind, value = await loop.run_in_executor(None, q.get, True, 0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if kind == 'token':
                    yield value
                elif kind == 'error':
                    logger.error("stream_generate_error", model=model_name, error=str(value))
                    break
                elif kind == 'done':
                    break

            await fut
            thread.join()
            executor.shutdown(wait=False)

        except Exception as e:
            logger.error("stream_generate_error", model=model_name, error=str(e))

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
