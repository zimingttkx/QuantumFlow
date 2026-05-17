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
        批量生成 — 支持 Chunked Prefill 长提示词分块处理。

        当 enable_chunked_prefill=True 且 prompt 长度超过 prefill_chunk_size 时：
        - 将长 prompt 分成多个 chunk
        - 逐块前向传播，累积 KV Cache 状态
        - 避免单次处理超长 prompt 时的显存爆炸
        """
        if model_name not in self._models:
            logger.error("model_not_loaded", model=model_name)
            return []

        model = self._models[model_name]
        tokenizer = self._tokenizers[model_name]
        config = self._loaded_models.get(model_name)

        chunk_size = getattr(config, 'prefill_chunk_size', 512) if config else 512
        enable_chunked = getattr(config, 'enable_chunked_prefill', True) if config else True

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        start_time = time.time()

        # ── 分块处理长 prompt ───────────────────────────────
        # 逐条处理（chunked prefill 需要 kv cache 累积，无法批量）
        results: List[InferenceResult] = []

        if enable_chunked and len(prompts) == 1:
            # 单请求时启用分块预填充（多请求走批量路径）
            result = await self._chunked_generate(
                model, tokenizer, prompts[0], sampling_params, chunk_size, start_time
            )
            if result:
                results.append(result)
            return results

        # ── 标准批量路径（短 prompt 或多请求）────────────────
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

    async def _chunked_generate(
        self,
        model,
        tokenizer,
        prompt: str,
        sampling_params: SamplingParams,
        chunk_size: int,
        start_time: float,
    ) -> Optional[InferenceResult]:
        """
        Chunked Prefill: 分块处理超长 prompt，逐块累积 KV Cache。

        步骤：
        1. Tokenize prompt
        2. 如果 token 数 > chunk_size，分块前向传播，累积 past_key_values
        3. 最后一块的隐藏状态作为 generation 的输入
        4. 调用 model.generate 完成生成
        """
        try:
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=chunk_size * 4)["input_ids"]
            input_ids = input_ids.to(model.device)
            prompt_len = input_ids.shape[1]

            if prompt_len <= chunk_size:
                # 短 prompt，直接 generate
                return await self._direct_generate(
                    model, tokenizer, input_ids, prompt, sampling_params, prompt_len, start_time
                )

            # 长 prompt：分块 prefill，累积 KV Cache
            logger.info("chunked_prefill_start", prompt_len=prompt_len, chunk_size=chunk_size)

            all_hidden_states = None
            past_key_values = None
            num_chunks = (prompt_len + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, prompt_len)
                chunk_ids = input_ids[:, start:end]

                with torch.no_grad():
                    chunk_out = model(
                        chunk_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        attention_mask=None,
                    )

                past_key_values = chunk_out.past_key_values
                all_hidden_states = chunk_out.last_hidden_state

                logger.debug(
                    "chunk_processed",
                    chunk=chunk_idx + 1,
                    total=num_chunks,
                    kv_cache_entries=len(past_key_values) if past_key_values else 0,
                )

            # 从最后一块的隐状态获取 logits，生成第一个新 token
            # 使用 last_hidden_state 的最后一个 token 做 AR 生成
            last_token_logits = all_hidden_states[:, -1, :]
            next_token_id = last_token_logits.argmax(dim=-1, keepdim=True)

            # 合并 input_ids + 第一个生成的 token
            gen_input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # 更新 attention_mask
            attention_mask = torch.ones_like(gen_input_ids)

            # 调用 generate（传入已累积的 past_key_values）
            gen_kwargs = {
                "max_new_tokens": max(1, sampling_params.max_tokens - 1),
                "temperature": sampling_params.temperature if sampling_params.temperature > 0 else 1.0,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "repetition_penalty": sampling_params.repetition_penalty,
                "do_sample": sampling_params.temperature > 0,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
            if sampling_params.stop:
                gen_kwargs["stop_strings"] = sampling_params.stop

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)

            # 解码输出（跳过原始 prompt tokens）
            output_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

            logger.info("chunked_prefill_done", prompt_len=prompt_len, output_len=len(outputs[0]) - prompt_len)

            return InferenceResult(
                request_id=f"{getattr(self, '_model_name', 'unknown')}_chunked",
                outputs=[output_text],
                prompt_tokens=prompt_len,
                completion_tokens=len(outputs[0]) - prompt_len,
                latency_ms=(time.time() - start_time) * 1000,
                finish_reason="stop" if (len(outputs[0]) - prompt_len) < sampling_params.max_tokens else "length",
                metrics={"chunked_prefill": True, "num_chunks": num_chunks},
            )

        except Exception as e:
            logger.error("chunked_generate_error", error=str(e))
            return None

    async def _direct_generate(
        self,
        model,
        tokenizer,
        input_ids,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_len: int,
        start_time: float,
    ) -> Optional[InferenceResult]:
        """直接 generate（短 prompt 路径）"""
        try:
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

            with torch.no_grad():
                outputs = model.generate(input_ids, **gen_kwargs)

            output_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
            return InferenceResult(
                request_id=f"{getattr(self, '_model_name', 'unknown')}_direct",
                outputs=[output_text],
                prompt_tokens=prompt_len,
                completion_tokens=len(outputs[0]) - prompt_len,
                latency_ms=(time.time() - start_time) * 1000,
                finish_reason="stop" if (len(outputs[0]) - prompt_len) < sampling_params.max_tokens else "length",
                metrics={},
            )
        except Exception as e:
            logger.error("direct_generate_error", error=str(e))
            return None


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
