"""HuggingFace推理后端"""

import asyncio
import concurrent.futures
import queue
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
import torch

from quantumflow.core.constants import InferenceBackendType
from quantumflow.inference.engine import (
    InferenceEngine,
    InferenceResult,
    ModelConfig,
    SamplingParams,
)

logger = structlog.get_logger().bind(component="hf_backend")

# Chunked Prefill 阈值：超过此长度自动使用分块预填充
CHUNKED_PREFILL_THRESHOLD_TOKENS = 512


class HuggingFaceEngine(InferenceEngine):
    """HuggingFace推理引擎实现"""

    def __init__(self, config: dict[str, Any] = None):
        super().__init__(InferenceBackendType.HUGGINGFACE)
        self.config = config or {}
        self._models: dict[str, Any] = {}  # Transformers模型实例
        self._tokenizers: dict[str, Any] = {}  # Tokenizer实例

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

            # use trust_remote_code=config.trust_remote_code
            # 注意：某些模型需要 remote code (如 Qwen/ChatGLM 等国产模型)
            # 但 cached custom code 可能与本地版本不一致，需要根据实际情况选择
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=config.trust_remote_code,
                low_cpu_mem_usage=True,
            )

            if device == "cpu":
                model = model.to(device)

            # torch.compile 优化 — 减少 Python 开销和 kernel launch 次数
            # 对 7B+ 大模型收益显著（编译开销高，小模型不值得）
            if device == "cuda" and getattr(config, "torch_compile", False):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("torch_compile_applied", model=config.model_name)
                except Exception as compile_err:
                    logger.warning(
                        "torch_compile_skipped", model=config.model_name, reason=str(compile_err)
                    )

            self._models[config.model_name] = model
            self._loaded_models[config.model_name] = config

            logger.info(
                "model_loaded",
                model=config.model_name,
                device=str(model.device) if hasattr(model, "device") else "unknown",
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

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """
        从 logits 中采样下一个 token。

        Args:
            logits: [vocab_size] - 最后一个位置的 logits
            temperature: 温度参数（0 表示 greedy）
            top_p: Nucleus sampling 的阈值
            top_k: Top-k 采样的 k 值
            repetition_penalty: 重复惩罚

        Returns:
            选中的 token id [1]
        """
        # 应用 repetition_penalty（每个 token 只惩罚一次）
        if repetition_penalty != 1.0:
            prev_tokens = getattr(self, "_generated_tokens", {})
            for tok_id in set(prev_tokens.values()):  # 使用 set 去重，避免同一 token 被多次惩罚
                logits[tok_id] /= repetition_penalty

        # Temperature 为 0 使用 greedy
        if temperature == 0:
            return logits.argmax().unsqueeze(0)

        # 保存 greedy token 作为 fallback（当所有 token 被过滤时使用）
        greedy_token = logits.argmax().unsqueeze(0)

        # 应用 temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            topk_values, _ = torch.topk(logits, top_k)
            threshold = topk_values[-1]
            indices_to_remove = logits < threshold
            logits[indices_to_remove] = float("-inf")

        # Top-p (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumsum <= top_p
            if not mask.all():
                first_exceed_idx = (~mask).nonzero()[0].item()
                mask[first_exceed_idx + 1 :] = False
            indices_to_remove = sorted_indices[~mask]
            logits[indices_to_remove] = float("-inf")

        # 采样 — 如果所有 token 被过滤则回退到 greedy
        probs = torch.softmax(logits, dim=-1)
        if torch.isnan(probs).any():
            return greedy_token
        return torch.multinomial(probs, num_samples=1)

    def _build_attention_mask(
        self, seq_len: int, past_len: int, device: torch.device
    ) -> torch.Tensor:
        """构建 4D causal attention mask

        返回形状 [1, 1, total_len, total_len] 的下三角 mask，
        其中 total_len = past_len + seq_len。

        注意：在常规 generate() 流程中我们不直接传 mask，
        HF 模型会用 default causal mask。但 chunked_prefill 路径里
        未来如果需要手动 forward，可以复用这个函数构造 mask。
        """
        import torch  # 局部导入以避免循环依赖

        total_len = past_len + seq_len
        # [total_len, total_len] 下三角（不含对角线以上）
        mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
        # 扩展到 4D
        return mask.view(1, 1, total_len, total_len)

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

    async def _chunked_generate_impl(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> tuple[str, int, int, float]:
        """
        Chunked Prefill 核心实现 — 使用手动 forward pass 进行自回归生成。

        工作流程：
        1. Prefill 阶段：将输入 token 分块处理，累积 past_key_values（KV cache）
        2. Decode 阶段：使用累积的 past_key_values 逐 token 自回归生成

        关键区别于旧实现：
        - 旧实现（bug）：Prefill 累积 past_key_values 后传给 model.generate()，
          导致 generate() 重新处理完整序列，手动累积的 KV cache 被忽略
        - 新实现：Prefill 和 Decode 都使用 model.forward() 手动控制，
          past_key_values 在整个生成过程中正确传递和使用

        Args:
            model_name: 模型名称
            prompt: 输入文本
            sampling_params: 采样参数

        Returns:
            (generated_text, prompt_tokens, completion_tokens, latency_ms)
        """
        start_time = time.time()

        model = self._models[model_name]
        tokenizer = self._tokenizers[model_name]
        model_config = self._loaded_models[model_name]

        device = next(model.parameters()).device
        chunk_size = getattr(model_config, "prefill_chunk_size", 512)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        prompt_lens = input_ids.shape[1]

        if prompt_lens == 0:
            return "", 0, 0, (time.time() - start_time) * 1000

        # 追踪已生成的 token（用于 repetition penalty）
        self._generated_tokens = {}

        # ── Phase 1: Prefill（分块处理输入，累积 KV cache）──────────────
        past_key_values = None
        num_chunks = (prompt_lens + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * chunk_size
            end_pos = min(start_pos + chunk_size, prompt_lens)
            chunk_ids = input_ids[:, start_pos:end_pos]

            with torch.no_grad():
                outputs = model(
                    chunk_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            logger.debug(
                "chunked_prefill_chunk",
                model=model_name,
                chunk_idx=chunk_idx + 1,
                num_chunks=num_chunks,
                start_pos=start_pos,
                end_pos=end_pos,
            )

        # ── Phase 2: Decode（自回归生成，使用累积的 KV cache）──────────
        # 从最后一个 prefill chunk 的 logits 采样第一个 token
        # (prefill 已经处理了全部输入，logits[:,-1,:] 即下一个 token 的分布)
        generated_ids = []

        if sampling_params.max_tokens <= 0:
            latency_ms = (time.time() - start_time) * 1000
            return "", prompt_lens, 0, latency_ms

        first_logits = outputs.logits[:, -1, :].squeeze(0)
        next_token_id = self._sample_token(
            first_logits,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            repetition_penalty=1.0,
        ).item()

        if next_token_id == tokenizer.eos_token_id:
            latency_ms = (time.time() - start_time) * 1000
            return "", prompt_lens, 0, latency_ms

        generated_ids.append(next_token_id)
        self._generated_tokens[0] = next_token_id

        for step in range(1, sampling_params.max_tokens):
            cur_token = torch.tensor([[generated_ids[-1]]], device=device)

            with torch.no_grad():
                outputs = model(
                    cur_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :].squeeze(0)

            if sampling_params.repetition_penalty != 1.0:
                for tok_id in set(generated_ids):
                    logits[tok_id] /= sampling_params.repetition_penalty

            next_token_id = self._sample_token(
                logits,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                repetition_penalty=1.0,
            ).item()

            if next_token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)
            self._generated_tokens[step] = next_token_id

        # Decode 生成结果
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        latency_ms = (time.time() - start_time) * 1000

        return generated_text, prompt_lens, len(generated_ids), latency_ms

    async def generate(
        self,
        model_name: str,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[InferenceResult]:
        """
        批量生成 — BatchAccumulator 在上层 50ms 窗口内合并并发请求，
        此处接收多个 prompt 一次性批量处理。

        策略：
        - 短 prompt（<= CHUNKED_PREFILL_THRESHOLD_TOKENS）：使用 model.generate()（HuggingFace 优化路径）
        - 长 prompt（> CHUNKED_PREFILL_THRESHOLD_TOKENS）且启用 Chunked Prefill：
          使用手动 forward 路径，避免 model.generate() 的重复处理开销
        """
        if model_name not in self._models:
            logger.error("model_not_loaded", model=model_name)
            return [
                InferenceResult(
                    request_id=f"{model_name}_{i}",
                    outputs=[f"[模型未加载: {model_name}]"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0,
                    finish_reason="error",
                    metrics={"error": "model_not_loaded"},
                )
                for i in range(len(prompts))
            ]

        model = self._models[model_name]
        tokenizer = self._tokenizers[model_name]
        model_config = self._loaded_models.get(model_name)
        enable_chunked = (
            getattr(model_config, "enable_chunked_prefill", False) if model_config else False
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        start_time = time.time()
        results: list[InferenceResult] = []

        # ── 决策：哪些 prompt 使用 Chunked Prefill ───────────────────────
        # Tokenize 统计各 prompt 长度
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,  # 不自动添加 EOS/BOS，准确统计
        )
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

        use_chunked = [
            enable_chunked and length > CHUNKED_PREFILL_THRESHOLD_TOKENS for length in prompt_lens
        ]

        # ── 短 prompt：使用 model.generate() 批量处理 ──────────────────
        short_indices = [i for i, uc in enumerate(use_chunked) if not uc]
        if short_indices:
            short_prompts = [prompts[i] for i in short_indices]
            short_lens = [prompt_lens[i] for i in short_indices]

            short_inputs = tokenizer(
                short_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            if hasattr(model, "device"):
                short_inputs = {k: v.to(model.device) for k, v in short_inputs.items()}

            gen_kwargs = {
                "max_new_tokens": sampling_params.max_tokens,
                "temperature": (
                    sampling_params.temperature if sampling_params.temperature > 0 else 1.0
                ),
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "repetition_penalty": sampling_params.repetition_penalty,
                "do_sample": sampling_params.temperature > 0,
                "use_cache": True,
            }
            if sampling_params.stop:
                gen_kwargs["stop_strings"] = sampling_params.stop

            try:
                with torch.no_grad():
                    outputs = model.generate(**short_inputs, **gen_kwargs)

                for idx, i in enumerate(short_indices):
                    prompt_len = int(short_lens[idx])
                    output_ids = outputs[idx]
                    # model.generate 返回完整序列 (prompt + generated)，跳过 prompt 部分
                    new_tokens = output_ids[prompt_len:]
                    new_tokens = [t for t in new_tokens.tolist() if t != tokenizer.pad_token_id]
                    gen_len = len(new_tokens)
                    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                    results.append(
                        InferenceResult(
                            request_id=f"{model_name}_{i}",
                            outputs=[output_text],
                            prompt_tokens=prompt_len,
                            completion_tokens=gen_len,
                            latency_ms=(time.time() - start_time) * 1000,
                            finish_reason=(
                                "stop" if gen_len < sampling_params.max_tokens else "length"
                            ),
                            metrics={"path": "generate"},
                        )
                    )
            except Exception as e:
                logger.error("generate_error", model=model_name, error=str(e))
                for idx, i in enumerate(short_indices):
                    results.append(
                        InferenceResult(
                            request_id=f"{model_name}_{i}",
                            outputs=[f"[生成错误: {str(e)}]"],
                            prompt_tokens=int(short_lens[idx]),
                            completion_tokens=0,
                            latency_ms=(time.time() - start_time) * 1000,
                            finish_reason="error",
                            metrics={"path": "generate", "error": str(e)},
                        )
                    )

        # ── 长 prompt：使用 Chunked Prefill（手动 forward）──────────────
        long_indices = [i for i, uc in enumerate(use_chunked) if uc]
        if long_indices:
            for i in long_indices:
                try:
                    gen_text, prompt_len, completion_len, chunked_latency = (
                        await self._chunked_generate_impl(model_name, prompts[i], sampling_params)
                    )
                    results.append(
                        InferenceResult(
                            request_id=f"{model_name}_{i}",
                            outputs=[gen_text],
                            prompt_tokens=prompt_len,
                            completion_tokens=completion_len,
                            latency_ms=chunked_latency,
                            finish_reason=(
                                "stop" if completion_len < sampling_params.max_tokens else "length"
                            ),
                            metrics={"path": "chunked_prefill"},
                        )
                    )
                except Exception as e:
                    logger.error(
                        "chunked_generate_error", model=model_name, prompt_idx=i, error=str(e)
                    )
                    results.append(
                        InferenceResult(
                            request_id=f"{model_name}_{i}",
                            outputs=[f"[分块预填充错误: {str(e)}]"],
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency_ms=(time.time() - start_time) * 1000,
                            finish_reason="error",
                            metrics={"path": "chunked_prefill", "error": str(e)},
                        )
                    )

        # 按原始顺序排序结果
        results.sort(key=lambda r: int(r.request_id.split("_")[-1]))

        return results

    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """
        流式生成 — 支持 Chunked Prefill 和标准生成两种路径。

        策略：
        - 短 prompt：使用 TextIteratorStreamer（model.generate() 的标准流式输出）
        - 长 prompt 且启用 Chunked Prefill：使用手动 forward，在 decode 阶段逐 token yield
        """
        if model_name not in self._models:
            logger.error("model_not_loaded", model=model_name)
            return  # async generator 提前结束，async for 会正常结束

        model = self._models[model_name]
        tokenizer = self._tokenizers[model_name]
        model_config = self._loaded_models.get(model_name)
        enable_chunked = (
            getattr(model_config, "enable_chunked_prefill", False) if model_config else False
        )

        # 检查是否使用 Chunked Prefill
        inputs_check = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = inputs_check["input_ids"].shape[1]
        use_chunked = enable_chunked and prompt_len > CHUNKED_PREFILL_THRESHOLD_TOKENS

        if use_chunked:
            # Chunked Prefill 流式：手动 forward，逐 token yield
            async for token_text in self._chunked_generate_stream_impl(
                model_name, prompt, sampling_params
            ):
                yield token_text
        else:
            # 标准流式：使用 TextIteratorStreamer
            async for text in self._stream_with_streamer(
                model_name, prompt, sampling_params, tokenizer, model, inputs_check
            ):
                yield text

    async def _chunked_generate_stream_impl(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """
        Chunked Prefill 流式实现 — 逐 token yield。

        工作流程：
        1. Prefill 阶段：分块处理输入，累积 past_key_values（无输出）
        2. Decode 阶段：逐 token 生成，每生成一个就 yield
        """
        model = self._models[model_name]
        tokenizer = self._tokenizers[model_name]
        model_config = self._loaded_models[model_name]

        device = next(model.parameters()).device
        chunk_size = getattr(model_config, "prefill_chunk_size", 512)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        prompt_lens = input_ids.shape[1]

        # ── Phase 1: Prefill（不输出，累积 KV cache）──────────────
        past_key_values = None
        num_chunks = (prompt_lens + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * chunk_size
            end_pos = min(start_pos + chunk_size, prompt_lens)
            chunk_ids = input_ids[:, start_pos:end_pos]

            with torch.no_grad():
                outputs = model(
                    chunk_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            logger.debug(
                "chunked_prefill_stream_chunk",
                model=model_name,
                chunk_idx=chunk_idx + 1,
                num_chunks=num_chunks,
            )

        # ── Phase 2: Decode（逐 token yield）──────────
        # 从最后一个 prefill chunk 的 logits 采样第一个 token
        first_logits = outputs.logits[:, -1, :].squeeze(0)
        next_token_id = self._sample_token(
            first_logits,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            repetition_penalty=1.0,
        ).item()

        generated_ids = []
        if next_token_id == tokenizer.eos_token_id:
            return

        generated_ids.append(next_token_id)
        cur_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if cur_text:
            yield cur_text
        prev_text_len = len(cur_text)

        for _step in range(1, sampling_params.max_tokens):
            cur_token = torch.tensor([[generated_ids[-1]]], device=device)

            with torch.no_grad():
                outputs = model(
                    cur_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].squeeze(0)

            if sampling_params.repetition_penalty != 1.0:
                for tok_id in set(generated_ids):
                    logits[tok_id] /= sampling_params.repetition_penalty

            next_token_id = self._sample_token(
                logits,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                repetition_penalty=1.0,
            ).item()

            if next_token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)

            cur_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_text = cur_text[prev_text_len:]
            if new_text:
                yield new_text
                prev_text_len = len(cur_text)

            await asyncio.sleep(0)

    async def _stream_with_streamer(
        self,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
        tokenizer,
        model,
        inputs,
    ) -> AsyncIterator[str]:
        """标准流式生成（使用 TextIteratorStreamer）"""
        from transformers import TextIteratorStreamer

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
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
            "use_cache": True,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        q: queue.Queue = queue.Queue()

        def _enqueue():
            try:
                for text in streamer:
                    q.put(("token", text))
                q.put(("done", None))
            except Exception as exc:
                q.put(("error", exc))

        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = loop.run_in_executor(executor, _enqueue)

        while True:
            try:
                kind, value = await loop.run_in_executor(None, q.get, True, 0.1)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if kind == "token":
                yield value
            elif kind == "error":
                logger.error("stream_generate_error", model=model_name, error=str(value))
                break
            elif kind == "done":
                break

        await fut
        thread.join()
        executor.shutdown(wait=False)

    async def get_stats(self, model_name: str) -> dict[str, float]:
        """获取引擎统计"""
        if model_name not in self._models:
            return {}

        try:
            import torch

            stats: dict[str, float] = {}
            if torch.cuda.is_available():
                stats["memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)
                stats["memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)

                # 尝试获取 GPU 利用率
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats["gpu_utilization"] = util.gpu / 100.0
                    stats["gpu_memory_utilization"] = util.memory / 100.0
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

            return stats

        except Exception:
            return {}
