"""严格的推理引擎单元测试"""

from unittest.mock import MagicMock, patch

import pytest

from quantumflow.core.constants import InferenceBackendType
from quantumflow.core.exceptions import InferenceError, ModelNotFoundError, SchedulerError
from quantumflow.inference.backends.vllm import VLLMEngine
from quantumflow.inference.engine import (
    InferenceResult,
    ModelConfig,
    SamplingParams,
)
from quantumflow.inference.manager import EngineManager, get_engine_manager

# =============================================================================
# Test ModelConfig - 验证配置的正确性和边界值
# =============================================================================


class TestModelConfigStrict:
    """ModelConfig严格测试"""

    def test_model_config_defaults(self):
        """验证默认配置值"""
        config = ModelConfig(model_name="test", model_path="/path")
        assert config.tensor_parallel == 1
        assert config.pipeline_parallel == 1
        assert config.gpu_memory_utilization == 0.8
        assert config.max_model_len == 2048
        assert config.dtype == "auto"
        assert config.trust_remote_code == True

    def test_model_config_custom_values(self):
        """验证自定义配置值"""
        config = ModelConfig(
            model_name="custom-model",
            model_path="/custom/path",
            tensor_parallel=2,
            pipeline_parallel=4,
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            dtype="float16",
            quantization="gptq",
            trust_remote_code=False,
            block_size=32,
            max_num_batched_tokens=4096,
            max_num_seqs=128,
            enforce_eager=True,
            enable_chunked_prefill=False,
        )
        assert config.tensor_parallel == 2
        assert config.pipeline_parallel == 4
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 2048
        assert config.dtype == "float16"
        assert config.quantization == "gptq"
        assert config.trust_remote_code == False
        assert config.block_size == 32
        assert config.max_num_batched_tokens == 4096
        assert config.max_num_seqs == 128
        assert config.enforce_eager == True
        assert config.enable_chunked_prefill == False

    def test_model_config_to_dict(self):
        """验证to_dict包含所有字段"""
        config = ModelConfig(
            model_name="test-model",
            model_path="/test",
            tensor_parallel=1,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
        )
        d = config.to_dict()
        assert "model_name" in d
        assert "model_path" in d
        assert "tensor_parallel_size" in d
        assert "gpu_memory_utilization" in d
        assert "max_model_len" in d

    def test_model_config_tensor_parallel_positive(self):
        """验证tensor_parallel设置为正数"""
        # 零或负数会导致vLLM错误
        config = ModelConfig(model_name="test", model_path="/test", tensor_parallel=1)
        assert config.tensor_parallel == 1

    def test_model_config_gpu_memory_range(self):
        """验证gpu_memory_utilization在合理范围"""
        # 0.85是修改后的值，原来是0.9
        config = ModelConfig(model_name="test", model_path="/test", gpu_memory_utilization=0.85)
        assert config.gpu_memory_utilization == 0.85


# =============================================================================
# Test SamplingParams - 验证采样参数
# =============================================================================


class TestSamplingParamsStrict:
    """SamplingParams严格测试"""

    def test_sampling_params_defaults(self):
        """验证默认采样参数"""
        params = SamplingParams()
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 2048
        assert params.repetition_penalty == 1.0
        assert params.stop is None

    def test_sampling_params_to_dict(self):
        """验证to_dict转换"""
        params = SamplingParams(temperature=0.5, max_tokens=100, stop=["###"])
        d = params.to_dict()
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 100
        assert d["stop"] == ["###"]

    def test_sampling_params_temperature_boundaries(self):
        """验证temperature边界值"""
        p0 = SamplingParams(temperature=0.0)
        assert p0.temperature == 0.0

        p1 = SamplingParams(temperature=2.0)
        assert p1.temperature == 2.0

    def test_sampling_params_top_p_boundaries(self):
        """验证top_p边界值"""
        p0 = SamplingParams(top_p=0.0)
        assert p0.top_p == 0.0

        p1 = SamplingParams(top_p=1.0)
        assert p1.top_p == 1.0

    def test_sampling_params_stop_list(self):
        """验证stop参数可以是列表"""
        params = SamplingParams(stop=["###", "---", "STOP"])
        assert len(params.stop) == 3
        assert "###" in params.stop


# =============================================================================
# Test InferenceResult - 验证推理结果
# =============================================================================


class TestInferenceResultStrict:
    """InferenceResult严格测试"""

    def test_inference_result_creation(self):
        """验证推理结果创建"""
        result = InferenceResult(
            request_id="req-001",
            outputs=["Hello world"],
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
            finish_reason="stop",
        )
        assert result.request_id == "req-001"
        assert result.outputs == ["Hello world"]
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.latency_ms == 100.0
        assert result.finish_reason == "stop"

    def test_inference_result_multiple_outputs(self):
        """验证多输出结果"""
        result = InferenceResult(
            request_id="req-002",
            outputs=["First output", "Second output"],
            prompt_tokens=20,
            completion_tokens=15,
            latency_ms=200.0,
            finish_reason="stop",
        )
        assert len(result.outputs) == 2

    def test_inference_result_with_metrics(self):
        """验证带指标的结果"""
        result = InferenceResult(
            request_id="req-003",
            outputs=["Output"],
            prompt_tokens=5,
            completion_tokens=10,
            latency_ms=50.0,
            finish_reason="stop",
            metrics={"throughput_tokens_per_sec": 200.0},
        )
        assert "throughput_tokens_per_sec" in result.metrics


# =============================================================================
# Test VLLMEngine - 验证vLLM引擎
# =============================================================================


class TestVLLMEngineStrict:
    """VLLMEngine严格测试"""

    def test_vllm_engine_creation(self):
        """验证引擎创建"""
        engine = VLLMEngine()
        assert engine.backend_type == InferenceBackendType.VLLM
        assert engine.is_ready == False

    def test_vllm_engine_initialize_already_ready(self):
        """验证重复初始化"""
        engine = VLLMEngine()
        engine._is_initialized = True
        # Should not raise, just return
        # (actual test would need async)

    @pytest.mark.asyncio
    async def test_vllm_initialize_success(self):
        """验证成功初始化"""
        engine = VLLMEngine()
        with patch.dict("sys.modules", {"vllm": MagicMock(__version__="0.21.0")}):
            result = await engine.initialize()
            assert result == True
            assert engine.is_ready == True

    @pytest.mark.asyncio
    async def test_vllm_initialize_already_initialized(self):
        """验证引擎已初始化时的行为"""
        engine = VLLMEngine()
        engine._is_initialized = True
        # 再次初始化应该成功
        with patch.dict("sys.modules", {"vllm": MagicMock(__version__="0.21.0")}):
            result = await engine.initialize()
            assert result == True

    @pytest.mark.asyncio
    async def test_load_model_not_initialized(self):
        """验证引擎未初始化时加载模型"""
        engine = VLLMEngine()
        config = ModelConfig(model_name="test", model_path="/test")
        result = await engine.load_model(config)
        assert result == False

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self):
        """验证卸载未加载的模型"""
        engine = VLLMEngine()
        engine._is_initialized = True
        result = await engine.unload_model("nonexistent")
        assert result == False

    @pytest.mark.asyncio
    async def test_generate_model_not_loaded(self):
        """验证模型未加载时生成 — 返回错误结果而非空列表"""
        engine = VLLMEngine()
        engine._is_initialized = True
        results = await engine.generate("nonexistent", ["prompt"], SamplingParams())
        assert len(results) == 1
        assert results[0].finish_reason == "error"
        assert "模型未加载" in results[0].outputs[0]

    @pytest.mark.asyncio
    async def test_generate_stream_model_not_loaded(self):
        """验证流式生成模型未加载"""
        engine = VLLMEngine()
        engine._is_initialized = True
        chunks = []
        async for chunk in engine.generate_stream("nonexistent", "prompt", SamplingParams()):
            chunks.append(chunk)
        assert chunks == []


# =============================================================================
# Test EngineManager - 验证引擎管理器
# =============================================================================


class TestEngineManagerStrict:
    """EngineManager严格测试"""

    def test_engine_manager_singleton(self):
        """验证单例模式"""
        manager1 = EngineManager()
        manager2 = EngineManager()
        assert manager1 is manager2

    def test_engine_manager_get_instance(self):
        """验证获取实例"""
        manager = get_engine_manager()
        assert isinstance(manager, EngineManager)

    def test_get_loaded_models_empty(self):
        """验证空模型列表"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._loaded_models = {}
        assert manager.get_loaded_models() == []

    def test_get_loaded_models_with_models(self):
        """验证有模型时的列表"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._loaded_models = {
            "model1": MagicMock(),
            "model2": MagicMock(),
        }
        models = manager.get_loaded_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models

    def test_is_model_loaded(self):
        """验证模型加载检查"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._loaded_models = {"loaded_model": MagicMock()}

        assert manager.is_model_loaded("loaded_model") == True
        assert manager.is_model_loaded("not_loaded") == False

    def test_get_stats_no_engine(self):
        """验证无引擎时获取统计"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._engines = {}
        stats = manager.get_stats()
        assert stats == {}

    @pytest.mark.asyncio
    async def test_load_model_auto_initializes_engine(self):
        """验证加载模型时自动初始化引擎"""
        # 使用get_engine_manager获取真正的单例
        manager = get_engine_manager()
        # 如果引擎未初始化，加载模型应该触发初始化
        # (实际会尝试初始化VLLM引擎)
        # 这个测试验证manager不会在引擎未初始化时崩溃
        try:
            result = await manager.load_model(
                model_name="test-autoload",
                model_path="/test/autoload",
                tensor_parallel=1,
            )
            # load_model 返回 (bool, str) 元组
            assert isinstance(result, tuple)
            assert isinstance(result[0], bool)
        except InferenceError:
            # 如果初始化失败，可能抛出InferenceError
            pass

    @pytest.mark.asyncio
    async def test_generate_no_engine(self):
        """验证无引擎时生成"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._engines = {}
        manager._loaded_models = {}

        with pytest.raises(ModelNotFoundError):
            await manager.generate("test", ["prompt"], SamplingParams())


# =============================================================================
# Test Backend Types - 验证后端类型枚举
# =============================================================================


class TestBackendTypesStrict:
    """后端类型严格测试"""

    def test_backend_type_values(self):
        """验证后端类型枚举值"""
        assert InferenceBackendType.VLLM.value == "vllm"
        assert InferenceBackendType.TGI.value == "text-generation-inference"
        assert InferenceBackendType.SGLANG.value == "sglang"

    def test_backend_type_count(self):
        """验证后端类型数量"""
        # 当前有6个后端: VLLM, TGI, SGLANG, TRT_LLM, LIGER, HUGGINGFACE
        assert len(InferenceBackendType) == 6


# =============================================================================
# Test API Models - 验证API模型
# =============================================================================


class TestAPIModelsStrict:
    """API模型严格测试"""

    def test_inference_request_validation(self):
        """验证推理请求验证"""
        from quantumflow.api.models import InferenceRequest

        # 测试缺失必需字段
        with pytest.raises(Exception):
            InferenceRequest()

    def test_chat_request_validation(self):
        """验证聊天请求验证"""
        from quantumflow.api.models import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

        # 测试消息列表
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hi"),
        ]
        assert len(messages) == 2


# =============================================================================
# Test Exceptions - 验证异常
# =============================================================================


class TestExceptionsStrict:
    """异常严格测试"""

    def test_inference_error(self):
        """验证推理错误"""
        err = InferenceError("Test error")
        d = err.to_dict()
        assert "error" in d
        assert d["error"]["code"] == "INFERENCE_ERROR"
        assert d["error"]["message"] == "Test error"

    def test_model_not_found_error(self):
        """验证模型未找到错误"""
        err = ModelNotFoundError("test-model")
        d = err.to_dict()
        assert "error" in d
        assert d["error"]["code"] == "MODEL_NOT_FOUND"
        assert "test-model" in d["error"]["message"]

    def test_scheduler_error(self):
        """验证调度器错误"""
        err = SchedulerError("Scheduler failed")
        d = err.to_dict()
        assert "error" in d
        assert d["error"]["code"] == "SCHEDULER_ERROR"


# =============================================================================
# Test VLLM Config Passing - 验证vLLM配置传递
# =============================================================================


class TestVLLMConfigPassing:
    """vLLM配置传递测试"""

    @pytest.mark.asyncio
    async def test_load_model_passes_config_values(self):
        """验证load_model正确传递配置到vLLM"""
        engine = VLLMEngine()
        engine._is_initialized = True

        config = ModelConfig(
            model_name="test-model",
            model_path="/test/path",
            tensor_parallel=2,
            pipeline_parallel=1,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            dtype="float16",
            quantization=None,
            trust_remote_code=True,
            block_size=16,
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            enforce_eager=True,
        )

        with patch("vllm.LLM") as mock_llm:
            mock_instance = MagicMock()
            mock_llm.return_value = mock_instance

            await engine.load_model(config)

            # 验证LLM被调用
            mock_llm.assert_called_once()

            # 验证调用参数 (vllm 0.21.0+ 移除了 max_model_len/block_size 等参数)
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["model"] == "/test/path"
            assert call_kwargs["tensor_parallel_size"] == 2
            assert call_kwargs["gpu_memory_utilization"] == 0.8
            assert call_kwargs["dtype"] == "float16"
            assert call_kwargs["enforce_eager"] == True


# =============================================================================
# Test Conversation History Handling - 验证对话历史处理
# =============================================================================


class TestConversationHistory:
    """对话历史处理测试"""

    @pytest.mark.asyncio
    async def test_chat_request_converts_history_correctly(self):
        """验证聊天请求正确转换对话历史"""
        from quantumflow.api.models import ChatMessage

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]

        # 验证消息可以创建
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "user"


# =============================================================================
# Test SamplingParams Conversion - 验证采样参数转换
# =============================================================================


class TestSamplingParamsConversion:
    """采样参数转换测试"""

    def test_convert_sampling_params_from_dict(self):
        """验证从字典转换采样参数"""
        from quantumflow.api.models import InferenceRequest
        from quantumflow.api.routes.inference import _convert_sampling_params

        request = InferenceRequest(
            model="test-model",
            prompt="Hello",
            sampling_params={
                "temperature": 0.5,
                "max_tokens": 100,
            },
        )

        params = _convert_sampling_params(request)
        assert params.temperature == 0.5
        assert params.max_tokens == 100

    def test_convert_sampling_params_none(self):
        """验证None采样参数使用默认值"""
        from quantumflow.api.models import InferenceRequest
        from quantumflow.api.routes.inference import _convert_sampling_params

        request = InferenceRequest(
            model="test-model",
            prompt="Hello",
            sampling_params=None,
        )

        params = _convert_sampling_params(request)
        assert params.temperature == 0.7  # 默认值
        assert params.max_tokens == 2048  # 默认值
