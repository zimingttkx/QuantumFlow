"""Model Registry测试"""

import pytest
from quantumflow.models import ModelRegistry, ModelInfo, ModelStatus, get_registry


class TestModelInfo:
    """ModelInfo测试"""

    def test_create_model_info(self):
        """测试创建模型信息"""
        info = ModelInfo(
            name="TestModel",
            path="/models/test",
            backend="vllm",
            parameter_count=7_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=14,
            max_memory_gb=18,
        )
        assert info.name == "TestModel"
        assert info.parameter_count == 7_000_000_000
        assert info.status == ModelStatus.AVAILABLE


class TestModelRegistry:
    """ModelRegistry测试"""

    @pytest.fixture
    def registry(self):
        """创建新的注册表"""
        return ModelRegistry()

    def test_registry_initialization(self, registry):
        """测试注册表初始化"""
        assert registry is not None
        assert len(registry.list_models()) > 0

    def test_builtin_models_loaded(self, registry):
        """测试内置模型已加载"""
        models = registry.list_models()
        model_names = [m.name for m in models]
        assert "Qwen2.5-7B-Instruct" in model_names
        assert "Qwen2.5-72B-Instruct" in model_names
        assert "LLaMA-3-8B-Instruct" in model_names

    def test_register_model(self, registry):
        """测试注册新模型"""
        info = ModelInfo(
            name="CustomModel",
            path="/models/custom",
            backend="vllm",
            parameter_count=3_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=6,
            max_memory_gb=8,
        )
        result = registry.register_model(info)
        assert result is True
        retrieved = registry.get_model("CustomModel")
        assert retrieved is not None

    def test_get_model(self, registry):
        """测试获取模型"""
        model = registry.get_model("Qwen2.5-7B-Instruct")
        assert model is not None
        assert model.parameter_count == 7_000_000_000

    def test_suggest_tensor_parallel(self, registry):
        """测试推荐tensor parallel大小"""
        tp_small = registry.suggest_tensor_parallel("Qwen2.5-7B-Instruct")
        assert tp_small == 1
        tp_large = registry.suggest_tensor_parallel("Qwen2.5-72B-Instruct")
        assert tp_large == 4

    def test_estimate_memory(self, registry):
        """测试估算显存需求"""
        memory = registry.estimate_memory("Qwen2.5-7B-Instruct")
        assert memory > 0
        assert memory <= 18


class TestGlobalRegistry:
    """全局注册表测试"""

    def test_get_registry(self):
        """测试获取全局注册表"""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
