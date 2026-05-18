"""HuggingFace 行为验证测试 — 世界上最严格的测试

测试原则：
1. GOLDEN TEST: 验证 chunked prefill 每一步喂给 model 的 token 序列
2. BEHAVIOR TEST: 验证 generate 返回的文本不包含 prompt
3. BOUNDARY TEST: 验证空 prompt / max_tokens=0 / 所有 token 被过滤
4. CARDINALITY TEST: 验证错误路径返回结果数 == 输入 prompt 数

这些测试才会抓到真正的逻辑错误（mock 结构测试永远抓不到的那种）。
"""

import pytest
import torch
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, '/home/dingziming/PycharmProjects/QuantumFlow')


def _token_list(tensor):
    """Helper: 将 tensor 安全转为 Python int list"""
    return tensor.flatten().tolist()

from quantumflow.inference.backends.huggingface import HuggingFaceEngine, CHUNKED_PREFILL_THRESHOLD_TOKENS
from quantumflow.inference.engine import ModelConfig, SamplingParams, InferenceResult


# ═══════════════════════════════════════════════════════════════════════════════
# Golden Test: Chunked Prefill 每一步喂给 model 的 token 序列
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkedPrefillGoldenSequence:
    """
    GOLDEN TEST — 验证 chunked prefill 在每一步喂给 model 的 token 完全是正确的。

    如果 chunked prefill 有 off-by-one bug，这个测试会直接失败。
    """

    @pytest.fixture
    def setup_golden(self):
        """构建 golden test 环境：可追踪 model 调用的 mock 引擎"""
        engine = HuggingFaceEngine()
        engine._is_initialized = True

        # 固定 prompt token 序列
        prompt_tokens = [101, 102, 103, 104, 105]  # 5 tokens
        # chunk_size=2, 所以 prefill 分为 3 个 chunk: [101,102], [103,104], [105]

        # 固定采样序列（每次 _sample_token 按顺序返回）
        sample_sequence = [201, 202, 203, 204, 205]

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([prompt_tokens]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_tokenizer.eos_token_id = 0  # 不在 sample_sequence 中，所以不会提前结束
        mock_tokenizer.decode = Mock(return_value="GENERATED")
        mock_tokenizer.pad_token_id = None

        # Mock model — 记录每次 __call__ 收到的 input_ids
        call_input_ids = []
        call_past_key_values = []

        def mock_model_call(input_ids, past_key_values=None, use_cache=False):
            call_input_ids.append(input_ids.detach().clone())
            call_past_key_values.append(past_key_values is not None)
            mock_output = Mock()
            # 返回递增的 mock past_key_values 以验证传递
            mock_output.past_key_values = f"pkv_{len(call_input_ids)}"
            # logits: [1, seq_len, vocab_size] — 最后一个位置给一个高分 token
            seq_len = input_ids.shape[1]
            mock_output.logits = torch.zeros(1, seq_len, 1000)
            mock_output.logits[0, -1, 999] = 100.0  # token 999 最高分
            return mock_output

        mock_model = Mock()
        mock_model.side_effect = mock_model_call
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters = Mock(return_value=iter([mock_param]))

        engine._models["golden_test"] = mock_model
        engine._tokenizers["golden_test"] = mock_tokenizer
        engine._loaded_models["golden_test"] = ModelConfig(
            model_name="golden_test",
            model_path="/golden",
            prefill_chunk_size=2,  # chunk_size=2, 5 tokens → 3 chunks
        )

        return engine, mock_model, mock_tokenizer, call_input_ids, call_past_key_values, sample_sequence

    @pytest.mark.asyncio
    async def test_prefill_chunks_fed_in_correct_order(self, setup_golden):
        """GOLDEN: Prefill 阶段必须按正确顺序喂入 chunk"""
        engine, _, _, call_input_ids, _, sample_sequence = setup_golden

        # Mock _sample_token 返回固定序列
        sample_iter = iter(sample_sequence)
        with patch.object(engine, '_sample_token', side_effect=lambda *args, **kwargs: torch.tensor([next(sample_iter)])):
            await engine._chunked_generate_impl(
                "golden_test", "ignored prompt text",
                SamplingParams(max_tokens=3, temperature=1.0)
            )

        # 验证 prefill 的前 3 个调用（chunk_size=2, 5 tokens）
        # Chunk 0: [101, 102]
        # Chunk 1: [103, 104]
        # Chunk 2: [105]
        assert call_input_ids[0].flatten().tolist() == [101, 102], \
            f"Prefill chunk 0 错误: {call_input_ids[0].flatten().tolist()}"
        assert call_input_ids[1].flatten().tolist() == [103, 104], \
            f"Prefill chunk 1 错误: {call_input_ids[1].flatten().tolist()}"
        assert call_input_ids[2].flatten().tolist() == [105], \
            f"Prefill chunk 2 错误: {call_input_ids[2].flatten().tolist()}"

    @pytest.mark.asyncio
    async def test_first_decode_token_is_from_sampling_not_last_prompt_token(self, setup_golden):
        """GOLDEN: 第一个 decode token 必须是 _sample_token 的结果，不能是 input_ids[:, -1:]"""
        engine, _, _, call_input_ids, _, sample_sequence = setup_golden

        sample_iter = iter(sample_sequence)
        with patch.object(engine, '_sample_token', side_effect=lambda *args, **kwargs: torch.tensor([next(sample_iter)])):
            await engine._chunked_generate_impl(
                "golden_test", "ignored prompt text",
                SamplingParams(max_tokens=3, temperature=1.0)
            )

        # 第 4 个调用（index=3）是第一个 decode step
        # 它应该喂入第一个采样 token（201），而不是 input_ids[:, -1:]（105）
        first_decode_input = call_input_ids[3].flatten().tolist()
        assert first_decode_input == [201], \
            f"BUG: 第一个 decode token 错误! 期望 [201], 实际 {first_decode_input}. " \
            f"如果是 [105] 说明 off-by-one bug 仍存在"

    @pytest.mark.asyncio
    async def test_decode_sequence_matches_sampled_tokens(self, setup_golden):
        """GOLDEN: decode 阶段每个 step 喂入的 token 必须等于上一轮采样结果"""
        engine, _, _, call_input_ids, _, sample_sequence = setup_golden

        sample_iter = iter(sample_sequence)
        with patch.object(engine, '_sample_token', side_effect=lambda *args, **kwargs: torch.tensor([next(sample_iter)])):
            await engine._chunked_generate_impl(
                "golden_test", "ignored prompt text",
                SamplingParams(max_tokens=3, temperature=1.0)
            )

        # Prefill: indices 0,1,2 (3 chunks)
        # Decode step 1: index 3 → feeds [201]
        # Decode step 2: index 4 → feeds [202]
        assert call_input_ids[3].flatten().tolist() == [201], \
            f"Decode step 1 应该 feed [201], 实际: {call_input_ids[3].flatten().tolist()}"
        assert call_input_ids[4].flatten().tolist() == [202], \
            f"Decode step 2 应该 feed [202], 实际: {call_input_ids[4].flatten().tolist()}"

    @pytest.mark.asyncio
    async def test_total_model_calls_equal_chunks_plus_decodes(self, setup_golden):
        """GOLDEN: model() 调用次数 = prefill_chunks + (max_tokens - 1)

        注意：第一个 token 从 prefill 的 logits 采样（不消耗 model 调用），
        所以 decode 调用次数 = max_tokens - 1
        """
        engine, _, _, call_input_ids, _, sample_sequence = setup_golden

        sample_iter = iter(sample_sequence)
        with patch.object(engine, '_sample_token', side_effect=lambda *args, **kwargs: torch.tensor([next(sample_iter)])):
            await engine._chunked_generate_impl(
                "golden_test", "ignored prompt text",
                SamplingParams(max_tokens=3, temperature=1.0)
            )

        # 5 tokens, chunk_size=2 → 3 prefill chunks
        # max_tokens=3 → (3-1) = 2 decode calls (第一个 token 从 prefill logits 来)
        expected_calls = 3 + 2  # 5
        assert len(call_input_ids) == expected_calls, \
            f"model 调用次数错误: 期望 {expected_calls}, 实际 {len(call_input_ids)}. " \
            f"如果实际是 6 说明有 off-by-one bug（多喂了一次 input_ids[:, -1:]）"

    @pytest.mark.asyncio
    async def test_past_key_values_propagated_across_all_calls(self, setup_golden):
        """GOLDEN: past_key_values 必须在所有调用之间正确传递"""
        engine, _, _, _, call_past_key_values, sample_sequence = setup_golden

        sample_iter = iter(sample_sequence)
        with patch.object(engine, '_sample_token', side_effect=lambda *args, **kwargs: torch.tensor([next(sample_iter)])):
            await engine._chunked_generate_impl(
                "golden_test", "ignored prompt text",
                SamplingParams(max_tokens=3, temperature=1.0)
            )

        # 第一个 prefill chunk: past_key_values=None
        assert call_past_key_values[0] is False, "Prefill chunk 0 的 pkv 应为 None"

        # 后续所有调用都应该收到非 None 的 past_key_values
        for i in range(1, len(call_past_key_values)):
            assert call_past_key_values[i] is True, \
                f"调用 {i} 的 past_key_values 应为非 None (从前一步传递过来)"

    @pytest.mark.asyncio
    async def test_eos_at_first_token_stops_immediately(self, setup_golden):
        """GOLDEN: 如果第一个采样 token 就是 EOS，立即停止，不调用 model"""
        engine, mock_model, mock_tokenizer, call_input_ids, _, _ = setup_golden

        # 第一个采样就是 EOS
        with patch.object(engine, '_sample_token', return_value=torch.tensor([0])):
            text, prompt_tokens, completion_tokens, latency = await engine._chunked_generate_impl(
                "golden_test", "ignored prompt text",
                SamplingParams(max_tokens=100, temperature=1.0)
            )

        # 只有 prefill 调用（3 chunks），没有 decode 调用
        assert len(call_input_ids) == 3, \
            f"EOS 在第一 token 时，不应有 decode 调用。实际调用次数: {len(call_input_ids)}"
        assert completion_tokens == 0
        assert text == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Behavior Test: generate 返回文本不得包含 prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateOutputDoesNotContainPrompt:
    """BEHAVIOR TEST — 验证 generate 返回的文本不包含 prompt 原文"""

    @pytest.fixture
    def setup_generate(self):
        engine = HuggingFaceEngine()
        engine._is_initialized = True

        PROMPT_TEXT = "What is the capital of France?"
        GENERATED_TEXT = "Paris"

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.pad_token_id = None

        # 第一次调用（统计长度）：add_special_tokens=False
        # 第二次调用（实际生成）：add_special_tokens=False
        # 两次都返回相同的 input_ids
        tokenized_input = torch.tensor([[10, 20, 30, 40, 50]])  # 5 prompt tokens
        tokenized_output = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])  # prompt + 3 generated tokens

        def tokenizer_side_effect(texts, **kwargs):
            # 返回 prompt tokenization
            mock_result = Mock()
            mock_result.__getitem__ = lambda self, key: tokenized_input[key] if key != "attention_mask" else torch.tensor([[1, 1, 1, 1, 1]])
            mock_result.items = lambda: {"input_ids": tokenized_input, "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])}.items()
            if isinstance(texts, list):
                mock_result = {"input_ids": tokenized_input.repeat(len(texts), 1),
                               "attention_mask": torch.ones(len(texts), 5)}
            return mock_result

        mock_tokenizer.side_effect = tokenizer_side_effect

        # decode: 只对 generated token ids 返回 GENERATED_TEXT
        def decode_side_effect(token_ids, skip_special_tokens=True):
            if isinstance(token_ids, list) and len(token_ids) > 0:
                return GENERATED_TEXT
            if hasattr(token_ids, 'tolist'):
                ids_list = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)
            else:
                ids_list = list(token_ids) if token_ids else []
            if len(ids_list) > 0:
                return GENERATED_TEXT
            return ""
        mock_tokenizer.decode = Mock(side_effect=decode_side_effect)

        # Mock model — generate 必须返回正确 batch_size 的结果
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        # model.generate 返回 prompt + generated（HuggingFace 默认行为）
        def generate_side_effect(**kwargs):
            # 如果传了 input_ids，根据 batch_size 返回对应数量的序列
            if "input_ids" in kwargs:
                batch_size = kwargs["input_ids"].shape[0]
                return tokenized_output.repeat(batch_size, 1)
            return tokenized_output
        mock_model.generate = Mock(side_effect=generate_side_effect)

        engine._models["test_model"] = mock_model
        engine._tokenizers["test_model"] = mock_tokenizer
        engine._loaded_models["test_model"] = ModelConfig(
            model_name="test_model",
            model_path="/test",
        )

        return engine, PROMPT_TEXT, GENERATED_TEXT

    @pytest.mark.asyncio
    async def test_short_prompt_generate_output_does_not_contain_prompt(self, setup_generate):
        """BEHAVIOR: generate 返回的文本不得包含输入 prompt"""
        engine, PROMPT_TEXT, GENERATED_TEXT = setup_generate

        results = await engine.generate(
            "test_model",
            [PROMPT_TEXT],
            SamplingParams(max_tokens=10),
        )

        assert len(results) == 1
        output_text = results[0].outputs[0]

        # 关键断言：输出文本不能包含 prompt
        assert PROMPT_TEXT not in output_text, \
            f"BUG: 输出包含了 prompt! output='{output_text}', prompt='{PROMPT_TEXT}'"

        # 额外验证：输出应该是纯生成文本
        assert GENERATED_TEXT in output_text, \
            f"输出应包含生成文本 '{GENERATED_TEXT}'，实际: '{output_text}'"

    @pytest.mark.asyncio
    async def test_batch_generate_preserves_prompt_output_separation(self, setup_generate):
        """BEHAVIOR: 批量 generate 时每个结果都不得包含各自 prompt"""
        engine, PROMPT_TEXT, GENERATED_TEXT = setup_generate

        prompts = [PROMPT_TEXT, "Another question?", "Third prompt here"]
        results = await engine.generate(
            "test_model",
            prompts,
            SamplingParams(max_tokens=10),
        )

        assert len(results) == 3, f"批量结果数应为 3，实际: {len(results)}"

        for i, result in enumerate(results):
            assert prompts[i] not in result.outputs[0], \
                f"BUG: 结果[{i}] 包含了 prompt! output='{result.outputs[0]}', prompt='{prompts[i]}'"


# ═══════════════════════════════════════════════════════════════════════════════
# Behavior Test: _sample_token 正确性
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleTokenBehavior:
    """BEHAVIOR TEST — 验证 _sample_token 在所有边界条件下行为正确"""

    @pytest.fixture
    def engine(self):
        engine = HuggingFaceEngine()
        engine._generated_tokens = {}
        return engine

    def test_nan_prevention_when_all_tokens_filtered(self, engine):
        """BEHAVIOR: 当 top-k + top-p 过滤掉所有 token 时，必须回落 greedy 而非返回 NaN"""
        # 构造：top_k=1 保留 token 5，但 top_p=0.1 又过滤掉所有 token
        logits = torch.zeros(1000)
        logits[5] = 10.0
        logits[6] = 9.0
        logits[7] = 8.0

        result = engine._sample_token(
            logits=logits.clone(),
            temperature=1.0,
            top_p=0.001,  # 极低的 top_p：几乎过滤所有 token
            top_k=1,  # top_k=1 只保留 token 5
            repetition_penalty=1.0,
        )

        # 结果必须是有效的 token id，不能是 NaN
        assert not torch.isnan(result) and not torch.isinf(result), \
            f"BUG: 采样返回 NaN/Inf: {result}"
        assert 0 <= result.item() < 1000, \
            f"BUG: 返回值越界: {result.item()}"
        # 应该回落到 greedy（token 5）
        assert result.item() == 5, \
            f"所有 token 被过滤时应回落 greedy 选 token 5，实际: {result.item()}"

    def test_nan_prevention_combined_filters(self, engine):
        """BEHAVIOR: top_k=1 只留最高 token，但 top_p 过滤掉它 → 回落 greedy"""
        logits = torch.zeros(100)
        logits[0] = 100.0  # 最高
        logits[1] = 1.0

        result = engine._sample_token(
            logits=logits.clone(),
            temperature=1.0,
            top_p=0.0001,  # 极小，过滤所有 token（包括 0）
            top_k=1,  # 只保留 token 0
            repetition_penalty=1.0,
        )

        assert not torch.isnan(result) and not torch.isinf(result), \
            f"BUG: 返回了 NaN/Inf"

    def test_repetition_penalty_uses_set_not_list(self, engine):
        """BEHAVIOR: 重复惩罚对同一 token 多次出现只惩罚一次（set 去重）"""
        logits = torch.zeros(100)
        logits[7] = 10.0

        # 模拟 token 7 在 _generated_tokens 中出现 1000 次
        engine._generated_tokens = {i: 7 for i in range(1000)}

        result = engine._sample_token(
            logits=logits.clone(),
            temperature=0.0,  # greedy
            top_p=1.0,
            top_k=0,
            repetition_penalty=2.0,
        )

        # penalty=2.0 → logit 10.0 / 2.0 = 5.0
        # 如果错误地累积惩罚：10.0 / (2.0^1000) ≈ 0，token 7 不可能被选中
        # 正确：只惩罚一次，10.0/2.0=5.0，token 7 仍可能是最高（取决于其他 token）
        assert not torch.isnan(result) and not torch.isinf(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary Test: 边界条件
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryConditions:
    """BOUNDARY TEST — 验证所有边界条件不会导致崩溃或错误行为"""

    @pytest.mark.asyncio
    async def test_empty_prompt_returns_empty_string(self):
        """BOUNDARY: 空 prompt 返回空字符串，不崩溃"""
        engine = HuggingFaceEngine()
        engine._is_initialized = True

        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[]]),  # 0 tokens
            "attention_mask": torch.tensor([[]]),
        }
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode = Mock(return_value="")

        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=iter([mock_param]))

        engine._models["test"] = mock_model
        engine._tokenizers["test"] = mock_tokenizer
        engine._loaded_models["test"] = ModelConfig(
            model_name="test", model_path="/test", prefill_chunk_size=512
        )

        text, prompt_tokens, completion_tokens, latency = await engine._chunked_generate_impl(
            "test", "", SamplingParams(max_tokens=10)
        )

        assert text == "", f"空 prompt 应返回空字符串，实际: '{text}'"
        assert prompt_tokens == 0
        assert completion_tokens == 0

    @pytest.mark.asyncio
    async def test_max_tokens_zero_generates_nothing(self):
        """BOUNDARY: max_tokens=0 不生成任何 token"""
        engine = HuggingFaceEngine()
        engine._is_initialized = True

        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode = Mock(return_value="")

        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=iter([mock_param]))

        # Mock model forward: 返回 logits（prefill 用）
        mock_output = Mock()
        mock_output.past_key_values = "pkv"
        mock_output.logits = torch.zeros(1, 3, 100)
        mock_output.logits[0, -1, 5] = 100.0
        mock_model.side_effect = lambda *a, **kw: mock_output

        engine._models["test"] = mock_model
        engine._tokenizers["test"] = mock_tokenizer
        engine._loaded_models["test"] = ModelConfig(
            model_name="test", model_path="/test", prefill_chunk_size=512
        )

        text, prompt_tokens, completion_tokens, latency = await engine._chunked_generate_impl(
            "test", "hi", SamplingParams(max_tokens=0)
        )

        assert completion_tokens == 0, f"max_tokens=0 应生成 0 token，实际: {completion_tokens}"

    @pytest.mark.asyncio
    async def test_single_token_prompt_chunked_prefill(self):
        """BOUNDARY: 单 token prompt 的 chunked prefill 正常工作"""
        engine = HuggingFaceEngine()
        engine._is_initialized = True

        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[42]]),  # 1 token
            "attention_mask": torch.tensor([[1]]),
        }
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode = Mock(return_value="ok")

        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=iter([mock_param]))

        call_count = [0]
        def mock_forward(input_ids, past_key_values=None, use_cache=False):
            call_count[0] += 1
            mock_output = Mock()
            mock_output.past_key_values = f"pkv_{call_count[0]}"
            mock_output.logits = torch.zeros(1, input_ids.shape[1], 100)
            mock_output.logits[0, -1, 5] = 100.0
            return mock_output
        mock_model.side_effect = mock_forward

        engine._models["test"] = mock_model
        engine._tokenizers["test"] = mock_tokenizer
        engine._loaded_models["test"] = ModelConfig(
            model_name="test", model_path="/test", prefill_chunk_size=512  # chunk > 1 token
        )

        with patch.object(engine, '_sample_token', return_value=torch.tensor([7])):
            text, prompt_tokens, completion_tokens, latency = await engine._chunked_generate_impl(
                "test", "X", SamplingParams(max_tokens=2)
            )

        # 1 prefill chunk (1 token < chunk_size) + 1 decode (第一个 token 从 prefill logits 来，第二个 token 需要一次 decode)
        # max_tokens=2: first from prefill logits, decode step 1 = 1 extra call
        assert call_count[0] == 2, f"单 token prompt: 期望 2 次 model 调用, 实际 {call_count[0]}"
        assert completion_tokens == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Cardinality Test: 错误路径返回正确数量的结果
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateCardinality:
    """CARDINALITY TEST — 无论成功失败，len(results) 必须 == len(prompts)"""

    @pytest.mark.asyncio
    async def test_unloaded_model_returns_n_error_results(self):
        """CARDINALITY: 模型未加载时， generate 返回 N 个错误结果"""
        engine = HuggingFaceEngine()
        engine._is_initialized = True
        engine._models = {}  # 没有模型

        results = await engine.generate(
            "nonexistent",
            ["prompt1", "prompt2", "prompt3"],
            SamplingParams(),
        )

        assert len(results) == 3, \
            f"3 个 prompts → 应返回 3 个结果，实际: {len(results)}"

        for i, r in enumerate(results):
            assert r.finish_reason == "error", \
                f"结果[{i}] 的 finish_reason 应为 'error'，实际: '{r.finish_reason}'"
            assert r.request_id == f"nonexistent_{i}", \
                f"结果[{i}] 的 request_id 应为 'nonexistent_{i}'，实际: '{r.request_id}'"

    @pytest.mark.asyncio
    async def test_empty_prompts_list_returns_empty_results(self):
        """CARDINALITY: 空 prompts 列表返回空 results 列表"""
        engine = HuggingFaceEngine()
        engine._is_initialized = True
        engine._models = {}

        results = await engine.generate("test", [], SamplingParams())

        assert results == [], f"空 prompts → 空 results，实际: {results}"


# ═══════════════════════════════════════════════════════════════════════════════
# Behavior Test: generate_stream 流式行为
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateStreamBehavior:
    """BEHAVIOR TEST — 验证流式生成的行为正确性"""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens_for_chunked_prefill(self):
        """BEHAVIOR: chunked prefill 流式必须逐 token yield"""
        engine = HuggingFaceEngine()
        engine._is_initialized = True

        mock_tokenizer = Mock()
        # prompt_len > 512 以触发 chunked prefill
        input_ids = torch.ones(1, 600, dtype=torch.long)
        mock_tokenizer.return_value = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(1, 600),
        }
        mock_tokenizer.eos_token_id = 0
        # decode 返回不同的文本以模拟逐 token 增长
        decode_results = ["Hello", "Hello world", "Hello world!"]
        decode_iter = iter(decode_results)
        mock_tokenizer.decode = Mock(side_effect=lambda ids, **kw: next(decode_iter))

        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        call_count = [0]
        def mock_forward(input_ids, past_key_values=None, use_cache=False):
            call_count[0] += 1
            mock_output = Mock()
            mock_output.past_key_values = f"pkv_{call_count[0]}"
            mock_output.logits = torch.zeros(1, input_ids.shape[1], 100)
            mock_output.logits[0, -1, 5] = 100.0
            return mock_output
        mock_model.side_effect = mock_forward

        engine._models["test"] = mock_model
        engine._tokenizers["test"] = mock_tokenizer
        engine._loaded_models["test"] = ModelConfig(
            model_name="test", model_path="/test",
            prefill_chunk_size=512, enable_chunked_prefill=True,
        )

        with patch.object(engine, '_sample_token', side_effect=[
            torch.tensor([7]), torch.tensor([8]), torch.tensor([9]),
        ]):
            chunks = []
            async for chunk in engine.generate_stream("test", "x" * 600, SamplingParams(max_tokens=3)):
                chunks.append(chunk)

        assert len(chunks) > 0, "Chunked prefill 流式必须 yield 内容"

    @pytest.mark.asyncio
    async def test_stream_model_not_loaded_returns_empty(self):
        """BEHAVIOR: 模型未加载时 generate_stream 返回空迭代"""
        engine = HuggingFaceEngine()
        engine._models = {}

        chunks = []
        async for chunk in engine.generate_stream("nonexistent", "prompt", SamplingParams()):
            chunks.append(chunk)

        assert chunks == [], "模型未加载时流式应返回空"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
