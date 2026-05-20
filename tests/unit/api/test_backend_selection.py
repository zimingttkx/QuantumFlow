"""API 路由后端选择测试

严格验证 API 路由正确处理 TGI/SGLang 后端选择：
1. LoadModelRequest 正确接受 backend 参数
2. 后端参数映射正确（"tgi" -> "text-generation-inference"）
3. 后端特定配置字段正确传递
4. API 响应包含正确的后端信息

本测试套件验证 API 层与 EngineManager 的集成。
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

from quantumflow.api.models.responses import LoadModelRequest
from quantumflow.core.constants import InferenceBackendType


# ═══════════════════════════════════════════════════════════════════════════════
# Test: LoadModelRequest 模型验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadModelRequestBackend:
    """验证 LoadModelRequest 的后端参数处理"""

    def test_load_model_request_accepts_vllm_backend(self):
        """[正常用例] LoadModelRequest 接受 vllm 后端"""
        request = LoadModelRequest(
            model="test-model",
            backend="vllm"
        )
        assert request.backend == "vllm"
        assert request.tgi_base_url is None
        assert request.sglang_base_url is None

    def test_load_model_request_accepts_tgi_backend(self):
        """[正常用例] LoadModelRequest 接受 tgi 后端"""
        request = LoadModelRequest(
            model="test-model",
            backend="tgi",
            tgi_base_url="http://localhost:8080"
        )
        assert request.backend == "tgi"
        assert request.tgi_base_url == "http://localhost:8080"

    def test_load_model_request_accepts_sglang_backend(self):
        """[正常用例] LoadModelRequest 接受 sglang 后端"""
        request = LoadModelRequest(
            model="test-model",
            backend="sglang",
            sglang_base_url="http://localhost:30000",
            sglang_timeout=600
        )
        assert request.backend == "sglang"
        assert request.sglang_base_url == "http://localhost:30000"
        assert request.sglang_timeout == 600

    def test_load_model_request_tgi_base_url_is_optional(self):
        """[边界用例] tgi_base_url 为可选字段"""
        request = LoadModelRequest(
            model="test-model",
            backend="tgi"
        )
        assert request.tgi_base_url is None

    def test_load_model_request_sglang_config_optional(self):
        """[边界用例] sglang 配置字段为可选"""
        request = LoadModelRequest(
            model="test-model",
            backend="sglang"
        )
        assert request.sglang_base_url is None
        assert request.sglang_timeout is None

    def test_load_model_request_default_backend_is_huggingface(self):
        """[边界用例] 默认后端是 huggingface"""
        request = LoadModelRequest(model="test-model")
        assert request.backend == "huggingface"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: InferenceBackendType 枚举值验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferenceBackendTypeEnum:
    """验证 InferenceBackendType 枚举值"""

    def test_tgi_enum_value_is_text_generation_inference(self):
        """[正常用例] TGI 枚举值是 "text-generation-inference"

        注意：这与 API 文档中的 "tgi" 不一致！
        需要在 API 层做字符串映射。
        """
        assert InferenceBackendType.TGI.value == "text-generation-inference"

    def test_sglang_enum_value_is_sglang(self):
        """[正常用例] SGLANG 枚举值是 "sglang" """
        assert InferenceBackendType.SGLANG.value == "sglang"

    def test_vllm_enum_value_is_vllm(self):
        """[正常用例] VLLM 枚举值是 "vllm" """
        assert InferenceBackendType.VLLM.value == "vllm"

    def test_huggingface_enum_value_is_huggingface(self):
        """[正常用例] HUGGINGFACE 枚举值是 "huggingface" """
        assert InferenceBackendType.HUGGINGFACE.value == "huggingface"

    def test_backend_string_to_enum_mapping_tgi(self):
        """[映射验证] "tgi" 字符串无法直接映射到 InferenceBackendType.TGI

        这是一个已知问题：API 文档说 backend="tgi"，
        但实际枚举值是 "text-generation-inference"。
        """
        # 直接用字符串创建会失败
        with pytest.raises(ValueError):
            InferenceBackendType("tgi")

        # 必须用完整值
        backend = InferenceBackendType("text-generation-inference")
        assert backend == InferenceBackendType.TGI

    def test_backend_string_to_enum_mapping_sglang(self):
        """[映射验证] "sglang" 可以直接映射"""
        backend = InferenceBackendType("sglang")
        assert backend == InferenceBackendType.SGLANG


# ═══════════════════════════════════════════════════════════════════════════════
# Test: 后端字符串映射函数
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackendStringMapping:
    """验证后端字符串到枚举的映射逻辑"""

    def test_tgi_string_mapping_via_resolve_function(self):
        """[正常用例] API 层通过 _resolve_backend 正确映射 "tgi" """
        from quantumflow.api.routes.model_management import _resolve_backend

        backend = _resolve_backend("tgi")
        assert backend == InferenceBackendType.TGI

    def test_tgi_full_value_also_works(self):
        """[正常用例] "text-generation-inference" 也能映射到 TGI """
        from quantumflow.api.routes.model_management import _resolve_backend

        backend = _resolve_backend("text-generation-inference")
        assert backend == InferenceBackendType.TGI

    def test_sglang_string_mapping(self):
        """[正常用例] "sglang" 正确映射 """
        from quantumflow.api.routes.model_management import _resolve_backend

        backend = _resolve_backend("sglang")
        assert backend == InferenceBackendType.SGLANG

    def test_vllm_string_mapping(self):
        """[正常用例] "vllm" 正确映射 """
        from quantumflow.api.routes.model_management import _resolve_backend

        backend = _resolve_backend("vllm")
        assert backend == InferenceBackendType.VLLM

    def test_huggingface_string_mapping(self):
        """[正常用例] "huggingface" 正确映射 """
        from quantumflow.api.routes.model_management import _resolve_backend

        backend = _resolve_backend("huggingface")
        assert backend == InferenceBackendType.HUGGINGFACE

    def test_none_returns_huggingface_default(self):
        """[边界用例] None 返回默认的 HuggingFace """
        from quantumflow.api.routes.model_management import _resolve_backend

        backend = _resolve_backend(None)
        assert backend == InferenceBackendType.HUGGINGFACE

    def test_invalid_backend_raises_error(self):
        """[错误用例] 无效的后端字符串抛出 ValueError """
        from quantumflow.api.routes.model_management import _resolve_backend

        with pytest.raises(ValueError) as exc_info:
            _resolve_backend("invalid-backend")

        assert "invalid-backend" in str(exc_info.value)

    def test_all_supported_backends_mapped(self):
        """[覆盖验证] 所有文档中提到的后端都有映射 """
        from quantumflow.api.routes.model_management import _resolve_backend

        supported = ["vllm", "huggingface", "tgi", "text-generation-inference", "sglang"]
        for backend_str in supported:
            backend = _resolve_backend(backend_str)
            assert backend is not None
            assert isinstance(backend, InferenceBackendType)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: 模型加载 API 逻辑
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelManagementRoute:
    """验证模型管理路由的后端处理逻辑"""

    @pytest.mark.asyncio
    async def test_load_model_passes_backend_specific_config(self):
        """[正常用例] 加载模型时传递后端特定配置"""
        from quantumflow.api.routes.model_management import load_model

        mock_manager = MagicMock()
        mock_manager.load_model = AsyncMock(return_value=(True, "Loaded"))

        with patch("quantumflow.api.routes.model_management.get_engine_manager", return_value=mock_manager):
            # 使用 MODEL_PATH_MAPPING 中的已知模型名，跳过 HF 验证
            request = LoadModelRequest(
                model="Qwen2.5-0.5B",
                backend="tgi",
                tgi_base_url="http://custom:8080"
            )

            response = await load_model(request)

            # 验证 backend 被正确解析
            call_kwargs = mock_manager.load_model.call_args[1]
            assert call_kwargs.get("backend") == InferenceBackendType.TGI
            assert call_kwargs.get("tgi_base_url") == "http://custom:8080"

    @pytest.mark.asyncio
    async def test_load_model_sglang_config_passed(self):
        """[正常用例] SGLang 配置正确传递"""
        from quantumflow.api.routes.model_management import load_model

        mock_manager = MagicMock()
        mock_manager.load_model = AsyncMock(return_value=(True, "Loaded"))

        with patch("quantumflow.api.routes.model_management.get_engine_manager", return_value=mock_manager):
            # 使用 MODEL_PATH_MAPPING 中的已知模型名
            request = LoadModelRequest(
                model="Qwen2.5-0.5B",
                backend="sglang",
                sglang_base_url="http://custom:30000",
                sglang_timeout=600
            )

            response = await load_model(request)

            # 验证配置被传递
            call_kwargs = mock_manager.load_model.call_args[1]
            assert call_kwargs.get("sglang_base_url") == "http://custom:30000"
            assert call_kwargs.get("sglang_timeout") == 600

    @pytest.mark.asyncio
    async def test_load_model_invalid_backend_raises_error(self):
        """[错误用例] 无效的后端抛出错误"""
        from quantumflow.api.routes.model_management import load_model

        mock_manager = MagicMock()
        mock_manager.load_model = AsyncMock(return_value=(True, "Loaded"))

        with patch("quantumflow.api.routes.model_management.get_engine_manager", return_value=mock_manager):
            # 使用已知模型名，这样可以通过模型验证
            request = LoadModelRequest(
                model="Qwen2.5-0.5B",
                backend="invalid-backend"
            )

            # 应该抛出 HTTPException
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await load_model(request)

            # ValueError 被通用异常处理器捕获，返回 500
            assert exc_info.value.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
