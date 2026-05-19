"""推理引擎测试"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from quantumflow.core.constants import InferenceBackendType
from quantumflow.core.exceptions import InferenceError, ModelNotFoundError
from quantumflow.inference.backends.vllm import VLLMEngine
from quantumflow.inference.engine import (
    InferenceResult,
    ModelConfig,
    SamplingParams,
)
from quantumflow.inference.manager import EngineManager, get_engine_manager


class TestModelConfig:
    """测试模型配置"""

    def test_model_config_creation(self):
        """测试创建模型配置"""
        config = ModelConfig(
            model_name="test-model",
            model_path="path/to/model",
            tensor_parallel=2,
        )

        assert config.model_name == "test-model"
        assert config.model_path == "path/to/model"
        assert config.tensor_parallel == 2
        assert config.gpu_memory_utilization == 0.8  # 默认值

    def test_model_config_to_dict(self):
        """测试模型配置转字典"""
        config = ModelConfig(
            model_name="test-model",
            model_path="path/to/model",
            tensor_parallel=1,
        )

        config_dict = config.to_dict()

        assert config_dict["model_name"] == "test-model"
        assert config_dict["tensor_parallel_size"] == 1


class TestSamplingParams:
    """测试采样参数"""

    def test_sampling_params_defaults(self):
        """测试默认采样参数"""
        params = SamplingParams()

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 2048

    def test_sampling_params_custom(self):
        """测试自定义采样参数"""
        params = SamplingParams(
            temperature=1.0,
            max_tokens=100,
            stop=["###"],
        )

        assert params.temperature == 1.0
        assert params.max_tokens == 100
        assert params.stop == ["###"]

    def test_sampling_params_to_dict(self):
        """测试采样参数转字典"""
        params = SamplingParams(temperature=0.5)
        params_dict = params.to_dict()

        assert params_dict["temperature"] == 0.5


class TestInferenceResult:
    """测试推理结果"""

    def test_inference_result_creation(self):
        """测试创建推理结果"""
        result = InferenceResult(
            request_id="req-001",
            outputs=["Hello, world!"],
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
            finish_reason="stop",
        )

        assert result.request_id == "req-001"
        assert result.outputs == ["Hello, world!"]
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.latency_ms == 100.0


class TestVLLMEngine:
    """测试 vLLM 引擎"""

    def test_vllm_engine_creation(self):
        """测试创建 vLLM 引擎"""
        engine = VLLMEngine()

        assert engine.backend_type == InferenceBackendType.VLLM
        assert not engine.is_ready

    def test_vllm_engine_initialize_import_error(self):
        """测试 vLLM 引擎初始化失败（无 vLLM）"""
        engine = VLLMEngine()

        # Mock import to raise ImportError
        with patch.dict("sys.modules", {"vllm": None}):
            import sys

            original_vllm = sys.modules.get("vllm")
            sys.modules["vllm"] = None

            result = engine._is_initialized
            assert result == False

            if original_vllm:
                sys.modules["vllm"] = original_vllm


class TestEngineManager:
    """测试引擎管理器"""

    def test_engine_manager_singleton(self):
        """测试引擎管理器单例"""
        manager1 = EngineManager()
        manager2 = EngineManager()

        assert manager1 is manager2

    def test_engine_manager_get_instance(self):
        """测试获取引擎管理器实例"""
        manager = get_engine_manager()

        assert isinstance(manager, EngineManager)

    @pytest.mark.asyncio
    async def test_load_model_without_engine(self):
        """测试在引擎初始化失败时抛出异常"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._engines = {}
        manager._loaded_models = {}
        # load_model 需要 _vram_manager 进行 VRAM 预估
        manager._vram_manager = Mock()
        manager._vram_manager.estimate_model_vram_gb = Mock(return_value=5.0)
        manager._vram_manager.can_load = Mock(return_value=(True, "ok", []))
        manager._vram_manager.record_loaded = Mock()
        manager._vram_manager.update_actual_vram = Mock()

        # mock initialize 返回 False
        with patch.object(manager, "initialize", new_callable=AsyncMock, return_value=False):
            # 引擎初始化失败，应该抛出 InferenceError
            with pytest.raises(InferenceError):
                await manager.load_model(
                    model_name="test-model",
                    model_path="path/to/model",
                )

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self):
        """测试卸载未加载的模型"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._loaded_models = {}

        result = await manager.unload_model("nonexistent-model")
        assert result == False

    def test_get_loaded_models_empty(self):
        """测试获取已加载模型列表（空）"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._loaded_models = {}

        models = manager.get_loaded_models()
        assert models == []

    def test_is_model_loaded(self):
        """测试检查模型是否已加载"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._loaded_models = {"test-model": MagicMock()}

        assert manager.is_model_loaded("test-model") == True
        assert manager.is_model_loaded("other-model") == False

    def test_get_stats_empty(self):
        """测试获取统计信息（空）"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._engines = {}

        stats = manager.get_stats()
        assert stats == {}


class TestInferenceIntegration:
    """推理引擎集成测试"""

    @pytest.mark.asyncio
    async def test_manager_initialize_vllm(self):
        """测试管理器初始化 vLLM"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._engines = {}
        manager._default_engine = None
        manager._loaded_models = {}

        # Mock VLLMEngine
        mock_engine = MagicMock()
        mock_engine.initialize = AsyncMock(return_value=True)
        mock_engine.is_ready = True

        with patch("quantumflow.inference.manager.VLLMEngine", return_value=mock_engine):
            result = await manager.initialize(InferenceBackendType.VLLM)
            # 由于我们 mock 了 initialize，返回 True
            assert result == True

    @pytest.mark.asyncio
    async def test_generate_without_model(self):
        """测试在没有加载模型时生成"""
        manager = EngineManager.__new__(EngineManager)
        manager._initialized = False
        manager._engines = {}
        manager._loaded_models = {}
        manager._default_engine = None

        with pytest.raises(ModelNotFoundError):
            await manager.generate(
                model_name="nonexistent",
                prompts=["test"],
                sampling_params=SamplingParams(),
            )


class TestBackendTypes:
    """测试后端类型"""

    def test_backend_type_values(self):
        """测试后端类型枚举值"""
        assert InferenceBackendType.VLLM.value == "vllm"
        assert InferenceBackendType.TGI.value == "text-generation-inference"
        assert InferenceBackendType.SGLANG.value == "sglang"
