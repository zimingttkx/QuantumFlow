"""Model Registry测试"""

import pytest

from quantumflow.models import ModelInfo, ModelRegistry, ModelStatus, get_registry


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


class TestModelInfoDefaults:
    """ModelInfo 默认值和字段测试"""

    def test_default_supported_backends_is_empty_list(self):
        info = ModelInfo(
            name="M", path="/p", backend="vllm",
            parameter_count=1, recommended_tensor_parallel=1,
            min_memory_gb=1, max_memory_gb=2,
        )
        assert info.supported_backends == []

    def test_default_status_is_available(self):
        info = ModelInfo(
            name="M", path="/p", backend="vllm",
            parameter_count=1, recommended_tensor_parallel=1,
            min_memory_gb=1, max_memory_gb=2,
        )
        assert info.status == ModelStatus.AVAILABLE

    def test_default_metadata_is_empty_dict(self):
        info = ModelInfo(
            name="M", path="/p", backend="vllm",
            parameter_count=1, recommended_tensor_parallel=1,
            min_memory_gb=1, max_memory_gb=2,
        )
        assert info.metadata == {}

    def test_custom_supported_backends(self):
        info = ModelInfo(
            name="M", path="/p", backend="vllm",
            parameter_count=1, recommended_tensor_parallel=1,
            min_memory_gb=1, max_memory_gb=2,
            supported_backends=["vllm", "tgi"],
        )
        assert info.supported_backends == ["vllm", "tgi"]


class TestModelStatusEnum:
    """ModelStatus 枚举值测试"""

    def test_all_status_values(self):
        assert ModelStatus.AVAILABLE.value == "available"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.LOADED.value == "loaded"
        assert ModelStatus.UNLOADING.value == "unloading"
        assert ModelStatus.ERROR.value == "error"

    def test_status_count(self):
        statuses = list(ModelStatus)
        assert len(statuses) == 5


class TestModelRegistryGaps:
    """补充未覆盖的 ModelRegistry 方法测试"""

    @pytest.fixture
    def registry(self):
        return ModelRegistry()

    # ---- unregister_model ----

    def test_unregister_existing_model(self, registry):
        """[核心功能] 注销存在的模型返回 True"""
        result = registry.unregister_model("Qwen2.5-7B-Instruct")
        assert result is True
        assert registry.get_model("Qwen2.5-7B-Instruct") is None

    def test_unregister_nonexistent_model(self, registry):
        """[核心功能] 注销不存在的模型返回 False"""
        result = registry.unregister_model("NoSuchModel")
        assert result is False

    # ---- register_model duplicate ----

    def test_register_duplicate_model_returns_false(self, registry):
        """[核心功能] 注册重名模型返回 False"""
        info = ModelInfo(
            name="Qwen2.5-7B-Instruct",  # already exists as built-in
            path="/other/path",
            backend="tgi",
            parameter_count=7_000_000_000,
            recommended_tensor_parallel=2,
            min_memory_gb=14,
            max_memory_gb=18,
        )
        result = registry.register_model(info)
        assert result is False

    # ---- get_model nonexistent ----

    def test_get_model_nonexistent_returns_none(self, registry):
        """[核心功能] 获取不存在的模型返回 None"""
        model = registry.get_model("NoSuchModel")
        assert model is None

    # ---- list_models with filters ----

    def test_list_models_with_status_filter(self, registry):
        """[核心功能] 按状态过滤模型列表"""
        all_models = registry.list_models()
        available = registry.list_models(status=ModelStatus.AVAILABLE)
        loading = registry.list_models(status=ModelStatus.LOADING)
        assert len(available) == len(all_models)  # all built-in models are AVAILABLE
        assert len(loading) == 0

    def test_list_models_with_backend_filter(self, registry):
        """[核心功能] 按后端过滤模型列表"""
        vllm_models = registry.list_models(backend="vllm")
        tgi_models = registry.list_models(backend="tgi")
        # vLLM models should be all built-in models
        assert len(vllm_models) > 0
        # TGI should be a subset
        assert len(tgi_models) <= len(vllm_models)

    def test_list_models_combined_filters(self, registry):
        """[核心功能] 合并状态和后端过滤"""
        result = registry.list_models(status=ModelStatus.AVAILABLE, backend="vllm")
        assert len(result) > 0
        for m in result:
            assert m.status == ModelStatus.AVAILABLE
            assert "vllm" in m.supported_backends

    # ---- update_model_status ----

    def test_update_model_status_existing(self, registry):
        """[核心功能] 更新已存在模型的状态"""
        result = registry.update_model_status("Qwen2.5-7B-Instruct", ModelStatus.LOADED)
        assert result is True
        model = registry.get_model("Qwen2.5-7B-Instruct")
        assert model.status == ModelStatus.LOADED

    def test_update_model_status_nonexistent(self, registry):
        """[核心功能] 更新不存在模型的状态返回 False"""
        result = registry.update_model_status("NoSuchModel", ModelStatus.ERROR)
        assert result is False

    def test_update_model_status_cycle(self, registry):
        """[核心功能] 完整状态变更周期: LOADING -> LOADED -> UNLOADING -> AVAILABLE"""
        assert registry.update_model_status("Qwen2.5-7B-Instruct", ModelStatus.LOADING) is True
        assert registry.get_model("Qwen2.5-7B-Instruct").status == ModelStatus.LOADING

        assert registry.update_model_status("Qwen2.5-7B-Instruct", ModelStatus.LOADED) is True
        assert registry.get_model("Qwen2.5-7B-Instruct").status == ModelStatus.LOADED

        assert registry.update_model_status("Qwen2.5-7B-Instruct", ModelStatus.UNLOADING) is True
        assert registry.get_model("Qwen2.5-7B-Instruct").status == ModelStatus.UNLOADING

        assert registry.update_model_status("Qwen2.5-7B-Instruct", ModelStatus.AVAILABLE) is True
        assert registry.get_model("Qwen2.5-7B-Instruct").status == ModelStatus.AVAILABLE

    # ---- get_models_by_backend ----

    def test_get_models_by_backend_returns_correct_models(self, registry):
        """[核心功能] 按后端获取模型列表"""
        vllm_models = registry.get_models_by_backend("vllm")
        assert len(vllm_models) > 0
        for m in vllm_models:
            assert "vllm" in m.supported_backends

    def test_get_models_by_unsupported_backend_returns_empty(self, registry):
        """[核心功能] 不支持的后端返回空列表"""
        result = registry.get_models_by_backend("nonexistent_backend")
        assert result == []

    # ---- suggest_tensor_parallel edge cases ----

    def test_suggest_tensor_parallel_nonexistent_model(self, registry):
        """[核心功能] 不存在的模型返回默认值 1"""
        tp = registry.suggest_tensor_parallel("NoSuchModel")
        assert tp == 1

    # ---- estimate_memory edge cases ----

    def test_estimate_memory_with_tensor_parallel(self, registry):
        """[核心功能] tensor_parallel>1 时显存估算被除以 tp"""
        # Qwen2.5-72B: min_memory_gb=140, tp=4 => 140//4=35
        memory = registry.estimate_memory("Qwen2.5-72B-Instruct", tensor_parallel=4)
        assert memory == 35

    def test_estimate_memory_nonexistent_model(self, registry):
        """[核心功能] 不存在的模型返回默认值 16"""
        memory = registry.estimate_memory("NoSuchModel")
        assert memory == 16

    def test_estimate_memory_with_tensor_parallel_large(self, registry):
        """[核心功能] 大 tensor_parallel 值正确计算"""
        # min_memory_gb=14, tp=14 => 14//14=1
        memory = registry.estimate_memory("Qwen2.5-7B-Instruct", tensor_parallel=14)
        assert memory == 1

    # ---- 验证所有内置模型字段完整性 ----

    def test_all_builtin_models_have_required_fields(self, registry):
        """[核心功能] 所有内置模型必须包含完整的元数据字段"""
        all_models = registry.list_models()
        for model in all_models:
            assert model.name, f"{model.name}: name 不能为空"
            assert model.path, f"{model.name}: path 不能为空"
            assert model.backend, f"{model.name}: backend 不能为空"
            assert model.parameter_count > 0, f"{model.name}: parameter_count 必须 > 0"
            assert model.recommended_tensor_parallel >= 1, f"{model.name}: recommended_tensor_parallel >= 1"
            assert model.min_memory_gb > 0, f"{model.name}: min_memory_gb 必须 > 0"
            assert model.max_memory_gb >= model.min_memory_gb, f"{model.name}: max >= min"
            assert len(model.supported_backends) >= 1, f"{model.name}: 至少一个支持的后端"


class TestGlobalRegistry:
    """全局注册表测试"""

    def test_get_registry(self):
        """测试获取全局注册表"""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
