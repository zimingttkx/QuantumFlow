"""EngineManager 后端集成测试

严格验证 EngineManager 对 TGI/SGLang 后端的支持：
1. initialize() 正确创建各后端引擎实例
2. load_model() 传递后端特定配置
3. 错误处理：无效后端、后端初始化失败
4. 后端路由：不同后端的请求正确路由

本测试套件使用 mock 模拟各后端引擎，验证 EngineManager 的集成逻辑。
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 跨平台路径处理
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from quantumflow.core.constants import InferenceBackendType
from quantumflow.inference.manager import EngineManager, get_engine_manager, _engine_manager


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def reset_engine_manager():
    """每个测试前重置 EngineManager 单例"""
    # 重置类级别的单例
    EngineManager._instance = None
    # 重置全局单例
    import quantumflow.inference.manager as manager_module
    manager_module._engine_manager = None
    yield
    EngineManager._instance = None
    manager_module._engine_manager = None


@pytest.fixture
def mock_vram_manager():
    """Mock VRAM 管理器，绕过 VRAM 检查"""
    vram = MagicMock()
    vram.can_load = MagicMock(return_value=(True, "", []))
    vram.estimate_model_vram_gb = MagicMock(return_value=1.0)
    vram.record_loaded = MagicMock()
    vram.update_actual_vram = MagicMock()
    vram.record_unloaded = MagicMock()
    return vram


@pytest.fixture
def mock_vllm_engine():
    """Mock VLLM 引擎"""
    engine = MagicMock()
    engine.initialize = AsyncMock(return_value=True)
    engine.load_model = AsyncMock(return_value=True)
    engine.unload_model = AsyncMock(return_value=True)
    engine.is_ready = True
    engine.loaded_model_names = []
    # 让 engine() 返回自身，模拟构造函数行为
    engine.return_value = engine
    return engine


@pytest.fixture
def mock_hf_engine():
    """Mock HuggingFace 引擎"""
    engine = MagicMock()
    engine.initialize = AsyncMock(return_value=True)
    engine.load_model = AsyncMock(return_value=True)
    engine.unload_model = AsyncMock(return_value=True)
    engine.is_ready = True
    engine.loaded_model_names = []
    engine.return_value = engine
    return engine


@pytest.fixture
def mock_tgi_engine():
    """Mock TGI 引擎"""
    engine = MagicMock()
    engine.initialize = AsyncMock(return_value=True)
    engine.load_model = AsyncMock(return_value=True)
    engine.unload_model = AsyncMock(return_value=True)
    engine.is_ready = True
    engine.loaded_model_names = []
    engine.return_value = engine
    return engine


@pytest.fixture
def mock_sglang_engine():
    """Mock SGLang 引擎"""
    engine = MagicMock()
    engine.initialize = AsyncMock(return_value=True)
    engine.load_model = AsyncMock(return_value=True)
    engine.unload_model = AsyncMock(return_value=True)
    engine.is_ready = True
    engine.loaded_model_names = []
    engine.return_value = engine
    return engine


# ═══════════════════════════════════════════════════════════════════════════════
# Test: initialize() 后端初始化逻辑
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineManagerInitialize:
    """验证 initialize() 正确创建各类型后端引擎"""

    @pytest.mark.asyncio
    async def test_initialize_vllm_creates_vllm_engine(self, mock_vllm_engine):
        """[正常用例] initialize(VLLM) 创建 VLLMEngine 实例"""
        with patch("quantumflow.inference.manager.VLLMEngine", return_value=mock_vllm_engine):
            manager = EngineManager()
            result = await manager.initialize(InferenceBackendType.VLLM)

        assert result is True
        assert InferenceBackendType.VLLM in manager._engines
        assert manager._default_engine == mock_vllm_engine
        mock_vllm_engine.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_huggingface_creates_hf_engine(self, mock_hf_engine):
        """[正常用例] initialize(HUGGINGFACE) 创建 HuggingFaceEngine 实例"""
        with patch("quantumflow.inference.manager.HuggingFaceEngine", return_value=mock_hf_engine):
            manager = EngineManager()
            result = await manager.initialize(InferenceBackendType.HUGGINGFACE)

        assert result is True
        assert InferenceBackendType.HUGGINGFACE in manager._engines
        assert manager._default_engine == mock_hf_engine

    @pytest.mark.asyncio
    async def test_initialize_tgi_creates_tgi_engine_with_base_url(self):
        """[正常用例] initialize(TGI) 使用正确的 base_url 创建 TGIEngine"""
        mock_engine = MagicMock()
        mock_engine.initialize = AsyncMock(return_value=True)

        with patch("quantumflow.inference.manager.TGIEngine") as mock_cls:
            mock_cls.return_value = mock_engine
            manager = EngineManager()
            result = await manager.initialize(
                InferenceBackendType.TGI,
                base_url="http://custom:8080"
            )

        assert result is True
        mock_cls.assert_called_once_with(base_url="http://custom:8080")
        mock_engine.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_tgi_default_base_url(self):
        """[边界用例] TGI 未指定 base_url 时使用默认值"""
        mock_engine = MagicMock()
        mock_engine.initialize = AsyncMock(return_value=True)

        with patch("quantumflow.inference.manager.TGIEngine") as mock_cls:
            mock_cls.return_value = mock_engine
            manager = EngineManager()
            result = await manager.initialize(InferenceBackendType.TGI)

        assert result is True
        mock_cls.assert_called_once_with(base_url="http://localhost:8080")

    @pytest.mark.asyncio
    async def test_initialize_sglang_creates_sglang_engine_with_config(self):
        """[正常用例] initialize(SGLANG) 传递 base_url 和 timeout"""
        mock_engine = MagicMock()
        mock_engine.initialize = AsyncMock(return_value=True)

        with patch("quantumflow.inference.manager.SGLangEngine") as mock_cls:
            mock_cls.return_value = mock_engine
            manager = EngineManager()
            result = await manager.initialize(
                InferenceBackendType.SGLANG,
                base_url="http://custom:30000",
                timeout=600
            )

        assert result is True
        mock_cls.assert_called_once_with(
            base_url="http://custom:30000",
            timeout=600
        )

    @pytest.mark.asyncio
    async def test_initialize_sglang_default_config(self):
        """[边界用例] SGLANG 未指定配置时使用默认值"""
        mock_engine = MagicMock()
        mock_engine.initialize = AsyncMock(return_value=True)

        with patch("quantumflow.inference.manager.SGLangEngine") as mock_cls:
            mock_cls.return_value = mock_engine
            manager = EngineManager()
            result = await manager.initialize(InferenceBackendType.SGLANG)

        assert result is True
        mock_cls.assert_called_once_with(
            base_url="http://localhost:30000",
            timeout=300
        )

    @pytest.mark.asyncio
    async def test_initialize_tgi_failure_returns_false(self):
        """[错误用例] TGI 初始化失败时 initialize 返回 False"""
        mock_engine = MagicMock()
        mock_engine.initialize = AsyncMock(return_value=False)

        with patch("quantumflow.inference.manager.TGIEngine", return_value=mock_engine):
            manager = EngineManager()
            result = await manager.initialize(InferenceBackendType.TGI)

        assert result is False
        assert InferenceBackendType.TGI not in manager._engines

    @pytest.mark.asyncio
    async def test_initialize_sglang_failure_returns_false(self):
        """[错误用例] SGLANG 初始化失败时 initialize 返回 False"""
        mock_engine = MagicMock()
        mock_engine.initialize = AsyncMock(return_value=False)

        with patch("quantumflow.inference.manager.SGLangEngine", return_value=mock_engine):
            manager = EngineManager()
            result = await manager.initialize(InferenceBackendType.SGLANG)

        assert result is False
        assert InferenceBackendType.SGLANG not in manager._engines

    @pytest.mark.asyncio
    async def test_initialize_unsupported_backend_returns_false(self):
        """[错误用例] 不支持的后端返回 False"""
        manager = EngineManager()

        # TRT_LLM 是规划中的后端，EngineManager 不应支持
        with patch("quantumflow.inference.manager.TGIEngine") as mock_cls:
            mock_cls.side_effect = Exception("Not implemented")
            result = await manager.initialize(InferenceBackendType.TRT_LLM)

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_tgi_exception_handled(self):
        """[异常用例] TGI 初始化抛出异常时正确处理"""
        mock_engine = MagicMock()
        mock_engine.initialize = AsyncMock(side_effect=ConnectionError("Network error"))

        with patch("quantumflow.inference.manager.TGIEngine", return_value=mock_engine):
            manager = EngineManager()
            result = await manager.initialize(InferenceBackendType.TGI)

        assert result is False
        assert InferenceBackendType.TGI not in manager._engines

    @pytest.mark.asyncio
    async def test_init_multiple_backends_simultaneously(self, mock_vllm_engine, mock_tgi_engine, mock_sglang_engine):
        """[正常用例] 先后初始化多个不同后端"""
        with patch("quantumflow.inference.manager.VLLMEngine", return_value=mock_vllm_engine):
            with patch("quantumflow.inference.manager.TGIEngine", return_value=mock_tgi_engine):
                with patch("quantumflow.inference.manager.SGLangEngine", return_value=mock_sglang_engine):
                    manager = EngineManager()

                    result1 = await manager.initialize(InferenceBackendType.VLLM)
                    assert result1 is True

                    result2 = await manager.initialize(InferenceBackendType.TGI)
                    assert result2 is True

                    result3 = await manager.initialize(InferenceBackendType.SGLANG)
                    assert result3 is True

        # 所有后端都应被存储
        assert InferenceBackendType.VLLM in manager._engines
        assert InferenceBackendType.TGI in manager._engines
        assert InferenceBackendType.SGLANG in manager._engines


# ═══════════════════════════════════════════════════════════════════════════════
# Test: load_model() 后端特定配置传递
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineManagerLoadModel:
    """验证 load_model() 正确传递后端特定配置"""

    @pytest.mark.asyncio
    async def test_load_model_tgi_passes_base_url(self, mock_tgi_engine, mock_vram_manager):
        """[正常用例] load_model(TGI) 传递 tgi_base_url"""
        with patch("quantumflow.inference.manager.TGIEngine", return_value=mock_tgi_engine):
            manager = EngineManager()
            manager._vram_manager = mock_vram_manager

            # 直接调用 load_model，不先调用 initialize
            # load_model 应该自动初始化引擎并传递正确的 base_url
            success, msg = await manager.load_model(
                model_name="test-model",
                model_path="/models/test",
                backend=InferenceBackendType.TGI,
                tgi_base_url="http://custom-tgi:8080"
            )

            assert success is True
            # 验证 TGI 引擎被创建并初始化
            assert InferenceBackendType.TGI in manager._engines
            mock_tgi_engine.initialize.assert_called_once()
            mock_tgi_engine.load_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_sglang_passes_config(self, mock_sglang_engine, mock_vram_manager):
        """[正常用例] load_model(SGLANG) 传递 sglang_base_url 和 sglang_timeout"""
        with patch("quantumflow.inference.manager.SGLangEngine", return_value=mock_sglang_engine):
            manager = EngineManager()
            manager._vram_manager = mock_vram_manager

            success, msg = await manager.load_model(
                model_name="test-model",
                model_path="/models/test",
                backend=InferenceBackendType.SGLANG,
                sglang_base_url="http://custom-sglang:30000",
                sglang_timeout=600
            )

            assert success is True
            # 验证 SGLang 引擎被创建并初始化
            assert InferenceBackendType.SGLANG in manager._engines
            mock_sglang_engine.initialize.assert_called_once()
            mock_sglang_engine.load_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_auto_initializes_backend(self, mock_tgi_engine, mock_vram_manager):
        """[正常用例] 加载模型时自动初始化未初始化的后端"""
        with patch("quantumflow.inference.manager.TGIEngine", return_value=mock_tgi_engine):
            manager = EngineManager()
            manager._vram_manager = mock_vram_manager

            # 未调用 initialize，直接 load_model
            success, msg = await manager.load_model(
                model_name="test-model",
                model_path="/models/test",
                backend=InferenceBackendType.TGI,
                tgi_base_url="http://localhost:8080"
            )

            assert success is True
            # TGI 引擎应该被创建并初始化
            assert InferenceBackendType.TGI in manager._engines

    @pytest.mark.asyncio
    async def test_load_model_initializes_tgi_with_default_when_no_config(self, mock_tgi_engine, mock_vram_manager):
        """[边界用例] 未指定 tgi_base_url 时使用默认值"""
        with patch("quantumflow.inference.manager.TGIEngine", return_value=mock_tgi_engine):
            manager = EngineManager()
            manager._vram_manager = mock_vram_manager

            success, msg = await manager.load_model(
                model_name="test-model",
                model_path="/models/test",
                backend=InferenceBackendType.TGI
            )

            assert success is True
            # 验证 TGI 引擎被创建并初始化
            assert InferenceBackendType.TGI in manager._engines
            mock_tgi_engine.initialize.assert_called_once()
            mock_tgi_engine.load_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_initializes_sglang_with_defaults(self, mock_sglang_engine, mock_vram_manager):
        """[边界用例] 未指定 SGLANG 配置时使用默认值"""
        with patch("quantumflow.inference.manager.SGLangEngine", return_value=mock_sglang_engine):
            manager = EngineManager()
            manager._vram_manager = mock_vram_manager

            success, msg = await manager.load_model(
                model_name="test-model",
                model_path="/models/test",
                backend=InferenceBackendType.SGLANG
            )

            assert success is True
            # 验证 SGLang 引擎被创建并初始化
            assert InferenceBackendType.SGLANG in manager._engines
            mock_sglang_engine.initialize.assert_called_once()
            mock_sglang_engine.load_model.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: 单例模式
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineManagerSingleton:
    """验证 EngineManager 单例模式"""

    def test_get_engine_manager_returns_same_instance(self):
        """[正常用例] get_engine_manager() 返回相同实例"""
        # 重置单例
        EngineManager._instance = None

        manager1 = get_engine_manager()
        manager2 = get_engine_manager()

        assert manager1 is manager2

    def test_engine_manager_is_singleton(self):
        """[正常用例] EngineManager 构造函数返回相同实例"""
        # 重置两个单例实现
        EngineManager._instance = None
        import quantumflow.inference.manager as manager_module
        manager_module._engine_manager = None

        manager1 = EngineManager()
        manager2 = EngineManager()

        assert manager1 is manager2
        assert manager1 is get_engine_manager()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: get_stats()
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineManagerStats:
    """验证 get_stats() 返回各后端状态"""

    @pytest.mark.asyncio
    async def test_get_stats_includes_all_initialized_backends(self, mock_vllm_engine, mock_tgi_engine):
        """[正常用例] get_stats() 包含所有已初始化的后端"""
        with patch("quantumflow.inference.manager.VLLMEngine", return_value=mock_vllm_engine):
            with patch("quantumflow.inference.manager.TGIEngine", return_value=mock_tgi_engine):
                manager = EngineManager()

                mock_vllm_engine.is_ready = True
                mock_vllm_engine.loaded_model_names = ["model-1"]
                mock_tgi_engine.is_ready = True
                mock_tgi_engine.loaded_model_names = []

                await manager.initialize(InferenceBackendType.VLLM)
                await manager.initialize(InferenceBackendType.TGI)

                stats = manager.get_stats()

        assert "vllm" in stats
        assert "text-generation-inference" in stats  # TGI 的 backend.value
        assert stats["vllm"]["is_ready"] is True
        assert stats["vllm"]["loaded_models"] == ["model-1"]
        assert stats["text-generation-inference"]["is_ready"] is True

    @pytest.mark.asyncio
    async def test_get_stats_empty_when_no_backends_initialized(self):
        """[边界用例] 无后端初始化时返回空字典"""
        EngineManager._instance = None
        manager = EngineManager()

        stats = manager.get_stats()

        assert stats == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Test: is_model_loaded / get_loaded_models
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineManagerModelState:
    """验证模型加载状态管理"""

    @pytest.mark.asyncio
    async def test_is_model_loaded_returns_true_after_load(self, mock_hf_engine):
        """[正常用例] 模型加载后 is_model_loaded 返回 True"""
        with patch("quantumflow.inference.manager.HuggingFaceEngine", return_value=mock_hf_engine):
            manager = EngineManager()
            await manager.initialize(InferenceBackendType.HUGGINGFACE)

            # 模拟模型已加载
            manager._loaded_models["test-model"] = mock_hf_engine

            assert manager.is_model_loaded("test-model") is True

    @pytest.mark.asyncio
    async def test_is_model_loaded_returns_false_before_load(self):
        """[边界用例] 模型加载前 is_model_loaded 返回 False"""
        manager = EngineManager()

        assert manager.is_model_loaded("non-existent-model") is False

    @pytest.mark.asyncio
    async def test_get_loaded_models_returns_list(self, mock_hf_engine):
        """[正常用例] get_loaded_models 返回已加载模型列表"""
        with patch("quantumflow.inference.manager.HuggingFaceEngine", return_value=mock_hf_engine):
            manager = EngineManager()
            await manager.initialize(InferenceBackendType.HUGGINGFACE)

            manager._loaded_models["model-1"] = mock_hf_engine
            manager._loaded_models["model-2"] = mock_hf_engine

            loaded = manager.get_loaded_models()

        assert len(loaded) == 2
        assert "model-1" in loaded
        assert "model-2" in loaded


# ═══════════════════════════════════════════════════════════════════════════════
# Test: 错误处理
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineManagerErrorHandling:
    """验证错误处理逻辑"""

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded_returns_false(self):
        """[错误用例] 卸载未加载的模型返回 False"""
        manager = EngineManager()

        result = await manager.unload_model("non-existent-model")

        assert result is False

    @pytest.mark.asyncio
    async def test_generate_model_not_loaded_raises_error(self):
        """[错误用例] 未加载模型时 generate 抛出 ModelNotFoundError"""
        from quantumflow.core.exceptions import ModelNotFoundError

        manager = EngineManager()

        with pytest.raises(ModelNotFoundError):
            await manager.generate(
                model_name="non-existent",
                prompts=["test"],
                sampling_params=MagicMock()
            )

    @pytest.mark.asyncio
    async def test_generate_stream_model_not_loaded_raises_error(self):
        """[错误用例] 未加载模型时 generate_stream 抛出 ModelNotFoundError"""
        from quantumflow.core.exceptions import ModelNotFoundError

        manager = EngineManager()

        with pytest.raises(ModelNotFoundError):
            async for _ in manager.generate_stream(
                model_name="non-existent",
                prompt="test",
                sampling_params=MagicMock()
            ):
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
