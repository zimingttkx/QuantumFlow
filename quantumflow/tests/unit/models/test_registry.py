"""Model Registry测试"""

import pytest
from datetime import datetime

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
        assert info.path == "/models/test"
        assert info.backend == "vllm"
        assert info.parameter_count == 7_000_000_000
        assert info.recommended_tensor_parallel == 1
        assert info.status == ModelStatus.AVAILABLE

    def test_model_info_with_backends(self):
        """测试带后端支持的模型"""
        info = ModelInfo(
            name="MultiBackendModel",
            path="/models/multi",
            backend="vllm",
            parameter_count=14_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=28,
            max_memory_gb=32,
            supported_backends=["vllm", "tgi", "sglang"],
        )

        assert len(info.supported_backends) == 3
        assert "vllm" in info.supported_backends
        assert "tgi" in info.supported_backends
        assert "sglang" in info.supported_backends

    def test_model_info_with_metadata(self):
        """测试带元数据的模型"""
        info = ModelInfo(
            name="MetadataModel",
            path="/models/meta",
            backend="vllm",
            parameter_count=7_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=14,
            max_memory_gb=18,
            metadata={"author": "Test", "version": "1.0"},
        )

        assert info.metadata["author"] == "Test"
        assert info.metadata["version"] == "1.0"


class TestModelStatus:
    """ModelStatus枚举测试"""

    def test_status_values(self):
        """测试状态值"""
        assert ModelStatus.AVAILABLE.value == "available"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.LOADED.value == "loaded"
        assert ModelStatus.UNLOADING.value == "unloading"
        assert ModelStatus.ERROR.value == "error"


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

        expected_models = [
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-72B-Instruct",
            "LLaMA-3-8B-Instruct",
            "LLaMA-3-70B-Instruct",
            "GLM-4-9B",
        ]

        model_names = [m.name for m in models]
        for expected in expected_models:
            assert expected in model_names

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
        assert retrieved.name == "CustomModel"
        assert retrieved.parameter_count == 3_000_000_000

    def test_register_duplicate_model(self, registry):
        """测试注册重复模型"""
        info = ModelInfo(
            name="DuplicateModel",
            path="/models/dup",
            backend="vllm",
            parameter_count=1_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=2,
            max_memory_gb=4,
        )

        result1 = registry.register_model(info)
        assert result1 is True

        result2 = registry.register_model(info)
        assert result2 is False

    def test_unregister_model(self, registry):
        """测试注销模型"""
        info = ModelInfo(
            name="UnregisterTest",
            path="/models/unreg",
            backend="vllm",
            parameter_count=1_000_000_000,
            recommended_tensor_parallel=1,
            min_memory_gb=2,
            max_memory_gb=4,
        )

        registry.register_model(info)
        assert registry.get_model("UnregisterTest") is not None

        result = registry.unregister_model("UnregisterTest")
        assert result is True
        assert registry.get_model("UnregisterTest") is None

    def test_unregister_nonexistent(self, registry):
        """测试注销不存在的模型"""
        result = registry.unregister_model("NonExistent")
        assert result is False

    def test_get_model(self, registry):
        """测试获取模型"""
        model = registry.get_model("Qwen2.5-7B-Instruct")

        assert model is not None
        assert model.name == "Qwen2.5-7B-Instruct"
        assert model.parameter_count == 7_000_000_000

    def test_get_nonexistent_model(self, registry):
        """测试获取不存在的模型"""
        model = registry.get_model("NonExistentModel")
        assert model is None

    def test_list_models(self, registry):
        """测试列出模型"""
        models = registry.list_models()
        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models)

    def test_list_models_by_status(self, registry):
        """测试按状态过滤模型"""
        models = registry.list_models(status=ModelStatus.AVAILABLE)
        assert all(m.status == ModelStatus.AVAILABLE for m in models)

    def test_list_models_by_backend(self, registry):
        """测试按后端过滤模型"""
        models = registry.list_models(backend="vllm")
        assert all("vllm" in m.supported_backends for m in models)

    def test_update_model_status(self, registry):
        """测试更新模型状态"""
        model = registry.get_model("Qwen2.5-7B-Instruct")
        assert model.status == ModelStatus.AVAILABLE

        result = registry.update_model_status("Qwen2.5-7B-Instruct", ModelStatus.LOADING)
        assert result is True

        updated = registry.get_model("Qwen2.5-7B-Instruct")
        assert updated.status == ModelStatus.LOADING

    def test_update_nonexistent_model_status(self, registry):
        """测试更新不存在的模型状态"""
        result = registry.update_model_status("NonExistent", ModelStatus.LOADING)
        assert result is False

    def test_get_models_by_backend(self, registry):
        """测试获取支持特定后端的模型"""
        models = registry.get_models_by_backend("vllm")
        assert len(models) > 0
        assert all("vllm" in m.supported_backends for m in models)

    def test_suggest_tensor_parallel(self, registry):
        """测试推荐tensor parallel大小"""
        # 小模型
        tp_small = registry.suggest_tensor_parallel("Qwen2.5-7B-Instruct")
        assert tp_small == 1

        # 大模型
        tp_large = registry.suggest_tensor_parallel("Qwen2.5-72B-Instruct")
        assert tp_large == 4

        # 未知模型
        tp_unknown = registry.suggest_tensor_parallel("UnknownModel")
        assert tp_unknown == 1  # 默认值

    def test_estimate_memory(self, registry):
        """测试估算显存需求"""
        memory = registry.estimate_memory("Qwen2.5-7B-Instruct", tensor_parallel=1)
        assert memory > 0
        assert memory <= 18  # 不超过最大需求

    def test_estimate_memory_with_parallel(self, registry):
        """测试带并行的显存估算"""
        memory_single = registry.estimate_memory("Qwen2.5-72B-Instruct", tensor_parallel=1)
        memory_multi = registry.estimate_memory("Qwen2.5-72B-Instruct", tensor_parallel=4)

        # TP4 应该需要更少的单卡显存
        assert memory_multi <= memory_single


class TestGlobalRegistry:
    """全局注册表测试"""

    def test_get_registry(self):
        """测试获取全局注册表"""
        registry1 = get_registry()
        registry2 = get_registry()

        # 应该是同一个实例
        assert registry1 is registry2

    def test_global_registry_has_builtins(self):
        """测试全局注册表包含内置模型"""
        registry = get_registry()
        models = registry.list_models()

        assert len(models) > 0
        assert any(m.name == "Qwen2.5-7B-Instruct" for m in models)
