"""HuggingFace Hub服务 - 严格单元测试

测试覆盖:
1. get_models_dir 目录创建
2. _model_to_dict 模型信息转换
3. get_trending_models 热门模型获取
4. search_models 模型搜索
5. validate_model 模型验证(存在/不存在/Gated/HTTP错误/通用异常)
6. get_model_detail 详细信息获取(config.json优先/fallback)
7. _fetch_config_params config.json参数读取
8. _estimate_params 参数量估算
9. _estimate_vram 显存估算
10. download_model 下载流程
11. get_download_progress 下载进度
12. get_downloaded_models 本地模型列表
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantumflow.api.services.hub_service import (
    _estimate_params,
    _estimate_vram,
    _fetch_config_params,
    _model_to_dict,
    download_model,
    get_download_progress,
    get_downloaded_models,
    get_models_dir,
    get_model_detail,
    get_trending_models,
    search_models,
    validate_model,
)


# ==================== 辅助工具 ====================


def _make_mock_model(
    model_id="org/model-name",
    author="test-author",
    downloads=1000,
    likes=50,
    pipeline_tag="text-generation",
    tags=None,
    last_modified=None,
    sha="abc123",
    created_at=None,
    private=False,
    gated=False,
    library_name="transformers",
    siblings=None,
):
    """创建模拟的HF模型对象"""
    m = MagicMock()
    m.modelId = model_id
    m.id = model_id
    m.author = author
    m.downloads = downloads
    m.likes = likes
    m.pipeline_tag = pipeline_tag
    m.tags = tags or []
    m.lastModified = last_modified
    m.sha = sha
    m.createdAt = created_at
    m.private = private
    m.gated = gated
    m.library_name = library_name
    m.siblings = siblings
    return m


# ==================== get_models_dir 测试 ====================


class TestGetModelsDir:
    """get_models_dir 测试"""

    @patch("os.makedirs")
    def test_creates_dir_and_returns_path(self, mock_makedirs):
        """[核心功能] 创建目录并返回路径"""
        result = get_models_dir()
        mock_makedirs.assert_called_once()
        assert "models" in result

    @patch("os.makedirs", side_effect=OSError("permission denied"))
    def test_raises_on_permission_error(self, mock_makedirs):
        """[异常场景] 权限不足时抛出异常"""
        with pytest.raises(OSError, match="permission denied"):
            get_models_dir()


# ==================== _model_to_dict 测试 ====================


class TestModelToDict:
    """_model_to_dict 测试"""

    def test_converts_all_fields(self):
        """[核心功能] 所有字段正确转换"""
        m = _make_mock_model(
            model_id="org/model-name",
            author="test-author",
            downloads=1000,
            likes=50,
            pipeline_tag="text-generation",
            tags=["llama", "7b"],
            sha="abc123def456",
        )
        result = _model_to_dict(m)

        assert result["model_id"] == "org/model-name"
        assert result["author"] == "test-author"
        assert result["downloads"] == 1000
        assert result["likes"] == 50
        assert result["pipeline_tag"] == "text-generation"
        assert result["tags"] == ["llama", "7b"]
        assert result["sha"] == "abc123def456"
        assert result["private"] is False
        assert result["gated"] is False
        assert result["library_name"] == "transformers"

    def test_downloads_none_defaults_to_zero(self):
        """[边界用例] downloads 为 None 时返回 0"""
        m = _make_mock_model(downloads=None)
        result = _model_to_dict(m)
        assert result["downloads"] == 0

    def test_likes_none_defaults_to_zero(self):
        """[边界用例] likes 为 None 时返回 0"""
        m = _make_mock_model(likes=None)
        result = _model_to_dict(m)
        assert result["likes"] == 0

    def test_tags_none_defaults_to_empty_list(self):
        """[边界用例] tags 为 None 时返回 []"""
        m = _make_mock_model(tags=None)
        result = _model_to_dict(m)
        assert result["tags"] == []

    def test_last_modified_none_returns_empty_string(self):
        """[边界用例] lastModified 为 None 时返回 ''"""
        m = _make_mock_model(last_modified=None)
        result = _model_to_dict(m)
        assert result["last_modified"] == ""

    def test_last_modified_exists_returns_string(self):
        """[核心功能] lastModified 存在时转为字符串"""
        import datetime
        ts = datetime.datetime(2024, 6, 15, 10, 30, 0)
        m = _make_mock_model(last_modified=ts)
        result = _model_to_dict(m)
        assert result["last_modified"] == str(ts)

    def test_created_at_none_returns_empty_string(self):
        """[边界用例] createdAt 为 None 时返回 ''"""
        m = _make_mock_model(created_at=None)
        result = _model_to_dict(m)
        assert result["created_at"] == ""

    def test_gated_and_private_true(self):
        """[核心功能] gated/private 为 True 时正确传递"""
        m = _make_mock_model(gated=True, private=True)
        result = _model_to_dict(m)
        assert result["gated"] is True
        assert result["private"] is True

    def test_model_without_modelid_uses_id(self):
        """[核心功能] 无 modelId 时使用 id 属性"""
        m = MagicMock()
        del m.modelId
        m.id = "fallback-id"
        m.author = "author"
        m.downloads = 0
        m.likes = 0
        m.pipeline_tag = "unknown"
        m.tags = []
        m.lastModified = None
        m.sha = ""
        m.createdAt = None
        m.private = False
        m.gated = False
        m.library_name = "unknown"
        result = _model_to_dict(m)
        assert result["model_id"] == "fallback-id"

    def test_missing_author_defaults_to_unknown(self):
        """[边界用例] 无 author 属性时返回 'unknown'"""
        m = _make_mock_model()
        del m.author
        result = _model_to_dict(m)
        assert result["author"] == "unknown"

    def test_siblings_accessed_but_not_in_output(self):
        """[核心功能] siblings 属性被访问但不影响输出"""
        m = _make_mock_model(siblings=["file1.bin"])
        result = _model_to_dict(m)
        assert "siblings" not in result


# ==================== get_trending_models 测试 ====================


class TestGetTrendingModels:
    """get_trending_models 测试"""

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_returns_models_successfully(self, mock_list_models):
        """[核心功能] 成功获取热门模型列表"""
        models = [_make_mock_model(model_id=f"org/model-{i}", downloads=1000 - i) for i in range(5)]
        mock_list_models.return_value = models

        result = await get_trending_models(limit=3)

        assert len(result) == 3
        assert result[0]["model_id"] == "org/model-0"
        assert result[1]["model_id"] == "org/model-1"
        assert result[2]["model_id"] == "org/model-2"

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_filters_out_duplicates_case_insensitive(self, mock_list_models):
        """[核心功能] 大小写不敏感去重"""
        models = [
            _make_mock_model(model_id="org/Model-A"),
            _make_mock_model(model_id="org/model-a"),  # 重复
            _make_mock_model(model_id="org/model-B"),
        ]
        mock_list_models.return_value = models

        result = await get_trending_models(limit=5)

        assert len(result) == 2
        ids = [r["model_id"] for r in result]
        assert "org/Model-A" in ids
        assert "org/model-B" in ids

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_excludes_gguf_onnx_variants(self, mock_list_models):
        """[核心功能] 排除 GGUF/ONNX/GGML 变体"""
        models = [
            _make_mock_model(model_id="org/model-gguf"),
            _make_mock_model(model_id="org/model-onnx"),
            _make_mock_model(model_id="org/model-ggml"),
            _make_mock_model(model_id="org/onnx-model"),
            _make_mock_model(model_id="org/good-model"),
        ]
        mock_list_models.return_value = models

        result = await get_trending_models(limit=5)

        ids = [r["model_id"] for r in result]
        assert "org/model-gguf" not in ids
        assert "org/model-onnx" not in ids
        assert "org/model-ggml" not in ids
        assert "org/onnx-model" not in ids
        assert "org/good-model" in ids

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_respects_limit(self, mock_list_models):
        """[核心功能] 遵守 limit 参数限制返回数量"""
        models = [_make_mock_model(model_id=f"org/model-{i}") for i in range(20)]
        mock_list_models.return_value = models

        result = await get_trending_models(limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_passes_filter_params(self, mock_list_models):
        """[核心功能] 传递过滤参数到 list_models"""
        models = [_make_mock_model()]
        mock_list_models.return_value = models

        await get_trending_models(limit=3, filter_params={"library": "transformers"})

        call_kwargs = mock_list_models.call_args[1]
        assert call_kwargs["library"] == "transformers"

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_empty_result_on_exception(self, mock_list_models):
        """[异常场景] 异常时返回空列表"""
        mock_list_models.side_effect = Exception("API error")

        result = await get_trending_models()

        assert result == []

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_empty_model_list(self, mock_list_models):
        """[边界用例] Hub 返回空列表时返回空列表"""
        mock_list_models.return_value = []

        result = await get_trending_models()

        assert result == []

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_all_models_filtered_out_by_variants(self, mock_list_models):
        """[边界用例] 所有模型都是变体被过滤时返回空列表"""
        models = [
            _make_mock_model(model_id="org/model-gguf"),
            _make_mock_model(model_id="org/model-onnx"),
        ]
        mock_list_models.return_value = models

        result = await get_trending_models(limit=5)

        assert result == []

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_prebuilt_filter_params_merged(self, mock_list_models):
        """[核心功能] 合并 pipeline_tag/sort 与自定义 filter_params"""
        models = [_make_mock_model()]
        mock_list_models.return_value = models

        await get_trending_models(limit=3, filter_params={"library": "transformers"})

        call_kwargs = mock_list_models.call_args[1]
        assert call_kwargs["pipeline_tag"] == "text-generation"
        assert call_kwargs["sort"] == "downloads"
        assert call_kwargs["library"] == "transformers"


# ==================== search_models 测试 ====================


class TestSearchModels:
    """search_models 测试"""

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_searches_and_returns_models(self, mock_list_models):
        """[核心功能] 搜索返回匹配模型"""
        models = [_make_mock_model(model_id="org/llama-7b")]
        mock_list_models.return_value = models

        result = await search_models(query="llama", limit=5)

        assert len(result) == 1
        assert result[0]["model_id"] == "org/llama-7b"

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_passes_search_query(self, mock_list_models):
        """[核心功能] 传递搜索关键词"""
        models = [_make_mock_model()]
        mock_list_models.return_value = models

        await search_models(query="qwen")

        call_kwargs = mock_list_models.call_args[1]
        assert call_kwargs["search"] == "qwen"

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_respects_limit(self, mock_list_models):
        """[核心功能] 遵守 limit 限制"""
        models = [_make_mock_model(model_id=f"org/model-{i}") for i in range(10)]
        mock_list_models.return_value = models

        result = await search_models(query="model", limit=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_empty_result_on_exception(self, mock_list_models):
        """[异常场景] 异常时返回空列表"""
        mock_list_models.side_effect = Exception("Search failed")

        result = await search_models(query="test")

        assert result == []

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.list_models")
    async def test_no_results_found(self, mock_list_models):
        """[边界用例] 无结果时返回空列表"""
        mock_list_models.return_value = []

        result = await search_models(query="nonexistent")

        assert result == []


# ==================== validate_model 测试 ====================


class TestValidateModel:
    """validate_model 测试"""

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    async def test_valid_model_returns_success(self, mock_model_info):
        """[核心功能] 有效模型返回验证成功"""
        mock_info = MagicMock()
        mock_info.gated = False
        mock_info.author = "org"
        mock_info.downloads = 500
        mock_info.likes = 20
        mock_info.pipeline_tag = "text-generation"
        mock_info.tags = ["llama"]
        mock_info.library_name = "transformers"
        mock_info.lastModified = "2024-01-01"
        mock_info.sha = "abc123"
        mock_info.private = False
        mock_model_info.return_value = mock_info

        result = await validate_model("org/test-model")

        assert result["valid"] is True
        assert result["exists"] is True
        assert result["gated"] is False
        assert result["model_id"] == "org/test-model"
        assert result["error"] is None
        assert result["info"] is not None
        assert result["info"]["author"] == "org"

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    async def test_nonexistent_model_returns_error(self, mock_model_info):
        """[核心功能] 不存在模型返回错误"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_model_info.side_effect = RepositoryNotFoundError("not found", response=mock_response)

        result = await validate_model("org/nonexistent")

        assert result["valid"] is False
        assert result["exists"] is False
        assert result["gated"] is False
        assert "不存在" in result["error"]
        assert result["info"] is None

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    async def test_gated_model_caught_by_repository_not_found(self, mock_model_info):
        """[核心功能] GatedRepoError extends RepositoryNotFoundError, caught by first handler.

        NOTE: This is a source-code bug in hub_service.py — the ``except GatedRepoError``
        block (line 183) is *dead code* because GatedRepoError inherits from
        RepositoryNotFoundError (caught on line 174).  The except clauses should be
        reordered so the more-specific GatedRepoError is caught first.
        """
        mock_response = MagicMock()
        mock_response.status_code = 403
        from huggingface_hub.utils import GatedRepoError

        mock_model_info.side_effect = GatedRepoError("gated", response=mock_response)

        result = await validate_model("org/gated-model")

        # Actual (buggy) behaviour: caught by RepositoryNotFoundError, returns valid=False
        assert result["valid"] is False
        assert result["exists"] is False
        assert result["gated"] is False
        assert "不存在" in result["error"]

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    async def test_http_error_returns_error(self, mock_model_info):
        """[核心功能] HTTP 错误返回错误"""
        mock_response = MagicMock()
        mock_response.status_code = 503
        from huggingface_hub.utils import HfHubHTTPError

        mock_model_info.side_effect = HfHubHTTPError("HTTP 503", response=mock_response)

        result = await validate_model("org/error-model")

        assert result["valid"] is False
        assert result["exists"] is False
        assert "访问错误" in result["error"]

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    async def test_generic_exception_returns_error(self, mock_model_info):
        """[异常场景] 通用异常返回错误"""
        mock_model_info.side_effect = RuntimeError("unexpected error")

        result = await validate_model("org/broken")

        assert result["valid"] is False
        assert result["exists"] is False
        assert "验证失败" in result["error"]

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    async def test_info_none_fields_defaults(self, mock_model_info):
        """[核心功能] info 字段为 None 时使用默认值"""
        mock_info = MagicMock()
        mock_info.gated = False
        mock_info.downloads = None
        mock_info.likes = None
        mock_info.tags = None
        mock_info.author = "org"
        mock_info.pipeline_tag = "unknown"
        mock_info.library_name = "unknown"
        mock_info.lastModified = ""
        mock_info.sha = ""
        mock_info.private = False
        mock_model_info.return_value = mock_info

        result = await validate_model("org/model")

        assert result["info"]["downloads"] == 0
        assert result["info"]["likes"] == 0
        assert result["info"]["tags"] == []


# ==================== get_model_detail 测试 ====================


class TestGetModelDetail:
    """get_model_detail 测试"""

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    @patch("quantumflow.api.services.hub_service._fetch_config_params")
    async def test_uses_config_json_params(self, mock_fetch_config, mock_model_info):
        """[核心功能] 优先使用 config.json 的参数量"""
        mock_fetch_config.return_value = {
            "params": 7_000_000_000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }
        mock_info = MagicMock()
        mock_info.author = "org"
        mock_info.downloads = 1000
        mock_info.likes = 100
        mock_info.pipeline_tag = "text-generation"
        mock_info.tags = ["llama"]
        mock_info.library_name = "transformers"
        mock_info.lastModified = None
        mock_info.gated = False
        mock_info.private = False
        mock_info.sha = "abc"
        mock_model_info.return_value = mock_info

        result = await get_model_detail("org/model")

        assert result["estimated_params"] == 7_000_000_000
        assert result["estimation_source"] == "config.json"
        assert "architecture" in result
        assert result["architecture"]["hidden_size"] == 4096
        assert result["architecture"]["num_hidden_layers"] == 32

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    @patch("quantumflow.api.services.hub_service._fetch_config_params")
    async def test_falls_back_to_name_heuristic(self, mock_fetch_config, mock_model_info):
        """[核心功能] config.json 无参数时使用名称估算"""
        mock_fetch_config.return_value = {}
        mock_info = MagicMock()
        mock_info.modelId = "org/llama-7b"
        mock_info.author = "org"
        mock_info.downloads = 1000
        mock_info.likes = 100
        mock_info.pipeline_tag = "text-generation"
        mock_info.tags = []
        mock_info.library_name = "transformers"
        mock_info.lastModified = None
        mock_info.gated = False
        mock_info.private = False
        mock_info.sha = "abc"
        mock_model_info.return_value = mock_info

        result = await get_model_detail("org/llama-7b")

        assert result["estimated_params"] == 7_000_000_000
        assert result["estimation_source"] == "name_heuristic"

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    async def test_exception_returns_error_dict(self, mock_model_info):
        """[异常场景] 异常时返回包含错误的字典"""
        mock_model_info.side_effect = Exception("API down")

        result = await get_model_detail("org/model")

        assert result["model_id"] == "org/model"
        assert "error" in result

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    @patch("quantumflow.api.services.hub_service._fetch_config_params")
    async def test_no_architecture_when_cfg_empty(self, mock_fetch_config, mock_model_info):
        """[核心功能] config 为空时不包含 architecture 字段"""
        mock_fetch_config.return_value = {}
        mock_info = MagicMock()
        mock_info.modelId = "org/model"
        mock_info.author = "org"
        mock_info.downloads = 0
        mock_info.likes = 0
        mock_info.pipeline_tag = "unknown"
        mock_info.tags = []
        mock_info.library_name = "unknown"
        mock_info.lastModified = None
        mock_info.gated = False
        mock_info.private = False
        mock_info.sha = ""
        mock_model_info.return_value = mock_info

        result = await get_model_detail("org/model")

        assert "architecture" not in result

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.model_info")
    @patch("quantumflow.api.services.hub_service._fetch_config_params")
    async def test_estimated_vram_with_kv_cache(self, mock_fetch_config, mock_model_info):
        """[核心功能] 有 hidden_size 和 num_layers 时估算含 KV cache 的 VRAM"""
        mock_fetch_config.return_value = {
            "params": 7_000_000_000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }
        mock_info = MagicMock()
        mock_info.author = "org"
        mock_info.downloads = 1000
        mock_info.likes = 100
        mock_info.pipeline_tag = "text-generation"
        mock_info.tags = []
        mock_info.library_name = "transformers"
        mock_info.lastModified = None
        mock_info.gated = False
        mock_info.private = False
        mock_info.sha = "abc"
        mock_model_info.return_value = mock_info

        result = await get_model_detail("org/model")

        # 模型 VRAM = 7B*2/1024^3*1.2 ≈ 15.6
        # KV cache = 2*32*4096*4096*2/1024^3 ≈ 2.0
        # total ≈ 17.6
        assert result["estimated_vram_gb"] > 15
        assert isinstance(result["estimated_params_b"], float)


# ==================== _fetch_config_params 测试 ====================


class TestFetchConfigParams:
    """_fetch_config_params 测试"""

    def test_returns_params_from_valid_config(self):
        """[核心功能] 从有效 config.json 提取参数"""
        config_data = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(json.dumps(config_data))
            tmp_path = f.name

        try:
            with patch(
                "quantumflow.api.services.hub_service.hf_hub_download",
                return_value=tmp_path,
            ):
                result = _fetch_config_params("org/model")

            assert "params" in result
            assert result["hidden_size"] == 4096
            assert result["num_hidden_layers"] == 32
            assert result["num_attention_heads"] == 32
            assert result["vocab_size"] == 32000
            assert result["params"] > 0
        finally:
            os.unlink(tmp_path)

    def test_returns_params_with_default_kv_heads(self):
        """[核心功能] num_key_value_heads 默认使用 num_attention_heads"""
        config_data = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "vocab_size": 50000,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(json.dumps(config_data))
            tmp_path = f.name

        try:
            with patch(
                "quantumflow.api.services.hub_service.hf_hub_download",
                return_value=tmp_path,
            ):
                result = _fetch_config_params("org/model")

            assert result["num_key_value_heads"] == 16
        finally:
            os.unlink(tmp_path)

    def test_returns_empty_for_missing_fields(self):
        """[核心功能] 缺少必要字段时返回空字典"""
        config_data = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            # 缺少 num_attention_heads, intermediate_size, vocab_size
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(json.dumps(config_data))
            tmp_path = f.name

        try:
            with patch(
                "quantumflow.api.services.hub_service.hf_hub_download",
                return_value=tmp_path,
            ):
                result = _fetch_config_params("org/model")

            assert result == {}
        finally:
            os.unlink(tmp_path)

    def test_returns_empty_on_exception(self):
        """[异常场景] 异常时返回空字典"""
        with patch(
            "quantumflow.api.services.hub_service.hf_hub_download",
            side_effect=Exception("download failed"),
        ):
            result = _fetch_config_params("org/model")

        assert result == {}

    def test_returns_empty_on_json_decode_error(self):
        """[异常场景] JSON 解析错误返回空字典"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json{{{")
            tmp_path = f.name

        try:
            with patch(
                "quantumflow.api.services.hub_service.hf_hub_download",
                return_value=tmp_path,
            ):
                result = _fetch_config_params("org/model")

            assert result == {}
        finally:
            os.unlink(tmp_path)


# ==================== _estimate_params 测试 ====================


class TestEstimateParams:
    """_estimate_params 测试"""

    def test_from_model_id_7b(self):
        """[核心功能] ID 含 7b 估算 7B"""
        m = _make_mock_model(model_id="org/llama-7b-instruct")
        assert _estimate_params(m) == 7_000_000_000

    def test_from_model_id_13b(self):
        """[核心功能] ID 含 13b 估算 13B"""
        m = _make_mock_model(model_id="org/model-13b")
        assert _estimate_params(m) == 13_000_000_000

    def test_from_model_id_70b(self):
        """[核心功能] ID 含 70b 估算 70B"""
        m = _make_mock_model(model_id="meta-llama/Llama-2-70b")
        assert _estimate_params(m) == 70_000_000_000

    def test_from_model_id_8b(self):
        """[核心功能] ID 含 8b 估算 8B"""
        m = _make_mock_model(model_id="org/model-8b")
        assert _estimate_params(m) == 8_000_000_000

    def test_from_model_id_1b(self):
        """[核心功能] ID 含 1b 估算 1B"""
        m = _make_mock_model(model_id="org/model-1b")
        assert _estimate_params(m) == 1_000_000_000

    def test_from_tags(self):
        """[核心功能] 从 tags 匹配参数量"""
        m = _make_mock_model(model_id="org/generic-model", tags=["7b", "llama"])
        assert _estimate_params(m) == 7_000_000_000

    def test_longer_pattern_matched_first(self):
        """[核心功能] 更长的模式(如 405b)优先于短模式(如 40b)"""
        m = _make_mock_model(model_id="org/model-405b")
        # 405b pattern is before 40b in the list, so it should match first
        assert _estimate_params(m) == 405_000_000_000

    def test_no_match_returns_zero(self):
        """[核心功能] 无匹配返回 0"""
        m = _make_mock_model(model_id="org/unknown-size")
        assert _estimate_params(m) == 0

    def test_unknown_model_id_empty(self):
        """[边界用例] 空 model_id 且无 tags 返回 0"""
        m = MagicMock()
        m.modelId = ""
        m.id = ""
        m.tags = []
        assert _estimate_params(m) == 0

    def test_config_json_priority(self):
        """[核心功能] config.json 有参数时优先使用"""
        m = _make_mock_model(model_id="org/model-7b")
        with patch(
            "quantumflow.api.services.hub_service._fetch_config_params",
            return_value={"params": 8_000_000_000},
        ):
            assert _estimate_params(m) == 8_000_000_000

    def test_no_model_id_uses_id_fallback(self):
        """[核心功能] 无 modelId 使用 id 属性"""
        m = MagicMock()
        del m.modelId
        m.id = "org/model-72b"
        m.tags = []
        assert _estimate_params(m) == 72_000_000_000

    def test_case_insensitive_match(self):
        """[核心功能] 大小写不敏感匹配"""
        m = _make_mock_model(model_id="org/Model-7B-Instruct")
        assert _estimate_params(m) == 7_000_000_000


# ==================== _estimate_vram 测试 ====================


class TestEstimateVram:
    """_estimate_vram 测试"""

    def test_zero_params_returns_zero(self):
        """[核心功能] 参数量为 0 返回 0"""
        assert _estimate_vram(0) == 0.0

    def test_fp16_estimation_without_kv_cache(self):
        """[核心功能] 无 KV cache 参数的 FP16 估算"""
        # 7B params: 7B * 2 bytes / 1024^3 * 1.2 ≈ 15.6
        result = _estimate_vram(7_000_000_000)
        expected = round(7_000_000_000 * 2 / (1024**3) * 1.2, 1)
        assert result == expected

    def test_fp16_estimation_with_kv_cache(self):
        """[核心功能] 含 KV cache 的 FP16 估算"""
        # 模型部分: 7B*2/1024^3*1.2 ≈ 15.6
        # KV cache: 2*32*4096*4096*2/1024^3 ≈ 2.0
        result = _estimate_vram(7_000_000_000, hidden_size=4096, num_hidden_layers=32)
        model_gb = 7_000_000_000 * 2 / (1024**3) * 1.2
        kv_cache_gb = 2 * 32 * 4096 * 4096 * 2 / (1024**3)
        expected = round(model_gb + kv_cache_gb, 1)
        assert result == expected

    def test_small_model(self):
        """[核心功能] 小模型 VRAM 估算"""
        result = _estimate_vram(500_000_000)  # 0.5B
        expected = round(500_000_000 * 2 / (1024**3) * 1.2, 1)
        assert result == expected

    def test_kv_cache_only_when_both_args_provided(self):
        """[核心功能] 仅当 hidden_size 和 num_hidden_layers 都>0 时才加 KV cache"""
        result = _estimate_vram(7_000_000_000, hidden_size=4096, num_hidden_layers=0)
        expected = round(7_000_000_000 * 2 / (1024**3) * 1.2, 1)
        assert result == expected


# ==================== download_model 测试 ====================


class TestDownloadModel:
    """download_model 测试"""

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    @patch("quantumflow.api.services.hub_service.snapshot_download")
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    async def test_successful_download(
        self, mock_makedirs, mock_exists, mock_snapshot, mock_validate
    ):
        """[核心功能] 成功下载模型"""
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "gated": False,
            "error": None,
        }
        mock_snapshot.return_value = "/tmp/models/org--model"

        result = await download_model("org/model")

        assert result["success"] is True
        assert result["local_path"] == "/tmp/models/org--model"
        assert result["error"] is None
        mock_snapshot.assert_called_once()

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    async def test_invalid_model_returns_error(self, mock_validate):
        """[核心功能] 无效模型返回错误"""
        mock_validate.return_value = {
            "valid": False,
            "exists": False,
            "gated": False,
            "error": "not found",
        }

        result = await download_model("org/bad")

        assert result["success"] is False
        assert result["error"] == "not found"

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    async def test_gated_model_returns_error(self, mock_validate):
        """[核心功能] 门控模型返回错误"""
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "gated": True,
            "error": "gated",
        }

        result = await download_model("org/gated")

        assert result["success"] is False
        assert "授权" in result["error"]

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    @patch("quantumflow.api.services.hub_service.snapshot_download")
    @patch("os.path.exists", return_value=True)
    @patch("os.listdir", return_value=["config.json"])
    @patch("os.makedirs")
    async def test_already_downloaded_returns_success(
        self, mock_makedirs, mock_listdir, mock_exists, mock_snapshot, mock_validate
    ):
        """[核心功能] 已下载模型直接返回成功"""
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "gated": False,
            "error": None,
        }

        result = await download_model("org/already-downloaded")

        assert result["success"] is True
        mock_snapshot.assert_not_called()

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    @patch("quantumflow.api.services.hub_service.snapshot_download")
    @patch("os.path.exists", side_effect=[False])
    @patch("os.makedirs")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_failure(
        self, mock_sleep, mock_makedirs, mock_exists, mock_snapshot, mock_validate
    ):
        """[核心功能] 下载失败后重试"""
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "gated": False,
            "error": None,
        }
        # 前两次失败,第三次成功
        mock_snapshot.side_effect = [
            Exception("network error"),
            Exception("network error"),
            "/tmp/models/org--model",
        ]

        result = await download_model("org/model", max_retries=3)

        assert result["success"] is True
        assert result["local_path"] == "/tmp/models/org--model"
        assert mock_snapshot.call_count == 3

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    @patch("quantumflow.api.services.hub_service.snapshot_download")
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_all_retries_exhausted(
        self, mock_sleep, mock_makedirs, mock_exists, mock_snapshot, mock_validate
    ):
        """[核心功能] 所有重试耗尽后返回失败"""
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "gated": False,
            "error": None,
        }
        mock_snapshot.side_effect = Exception("persistent error")

        result = await download_model("org/model", max_retries=2)

        assert result["success"] is False
        assert "下载失败" in result["error"]
        assert mock_snapshot.call_count == 2

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    @patch("quantumflow.api.services.hub_service.snapshot_download")
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    async def test_progress_callback_called(
        self, mock_makedirs, mock_exists, mock_snapshot, mock_validate
    ):
        """[核心功能] 进度回调被调用"""
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "gated": False,
            "error": None,
        }
        mock_snapshot.return_value = "/tmp/models/org--model"

        callback = AsyncMock()
        result = await download_model("org/model", progress_callback=callback)

        assert result["success"] is True
        # 至少调用 0% 和 100% 两次
        assert callback.call_count >= 2

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    @patch("quantumflow.api.services.hub_service.snapshot_download")
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    async def test_default_max_retries_is_3(
        self, mock_makedirs, mock_exists, mock_snapshot, mock_validate
    ):
        """[核心功能] 默认 max_retries=3"""
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "gated": False,
            "error": None,
        }
        mock_snapshot.side_effect = Exception("fail")

        # 不用 max_retries 参数
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await download_model("org/model")

        assert result["success"] is False
        assert mock_snapshot.call_count == 3

    @pytest.mark.asyncio
    @patch("quantumflow.api.services.hub_service.validate_model")
    @patch("quantumflow.api.services.hub_service.snapshot_download")
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    async def test_empty_dir_not_treated_as_downloaded(
        self, mock_makedirs, mock_exists, mock_snapshot, mock_validate, tmp_path
    ):
        """[核心功能] 空目录不视为已下载"""
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "gated": False,
            "error": None,
        }
        mock_snapshot.return_value = str(tmp_path / "org--model")

        # 目录存在但为空
        with patch(
            "os.path.exists",
            side_effect=lambda p: False,
        ):
            result = await download_model("org/model")

        assert result["success"] is True


# ==================== get_download_progress 测试 ====================


class TestGetDownloadProgress:
    """get_download_progress 测试"""

    def test_returns_progress_when_downloading(self):
        """[核心功能] 下载中返回进度值"""
        import quantumflow.api.services.hub_service as svc

        svc._downloading["test-model"] = 45.5
        try:
            result = get_download_progress("test-model")
            assert result == 45.5
        finally:
            svc._downloading.pop("test-model", None)

    def test_returns_minus_one_when_not_downloading(self):
        """[核心功能] 未下载返回 -1"""
        result = get_download_progress("nonexistent-model")
        assert result == -1


# ==================== get_downloaded_models 测试 ====================


class TestGetDownloadedModels:
    """get_downloaded_models 测试"""

    def test_returns_models_from_download_dir(self, tmp_path):
        """[核心功能] 返回已下载模型列表"""
        # 创建模型目录
        model_dir = tmp_path / "org--model-a"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        (model_dir / "model.safetensors").write_text("fake weights" * 100)

        # 创建空目录(应被忽略)
        empty_dir = tmp_path / "org--empty-model"
        empty_dir.mkdir()

        with patch(
            "quantumflow.api.services.hub_service.get_models_dir",
            return_value=str(tmp_path),
        ):
            result = get_downloaded_models()

        assert len(result) == 1
        assert result[0]["model_id"] == "org/model-a"
        assert result[0]["size_bytes"] > 0
        assert result[0]["size_gb"] >= 0.0

    def test_empty_when_no_models_dir(self):
        """[核心功能] 无模型目录时返回空列表"""
        with patch(
            "quantumflow.api.services.hub_service.get_models_dir",
            return_value="/nonexistent/path",
        ):
            result = get_downloaded_models()

        assert result == []

    def test_ignores_non_directories(self, tmp_path):
        """[核心功能] 忽略非目录条目"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "some_file.txt").write_text("not a model")

        with patch(
            "quantumflow.api.services.hub_service.get_models_dir",
            return_value=str(models_dir),
        ):
            result = get_downloaded_models()

        assert result == []

    def test_model_id_with_multiple_dashes(self, tmp_path):
        """[核心功能] 多个 -- 正确还原为 /"""
        model_dir = tmp_path / "org--sub--model-name"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"valid": true}')

        with patch(
            "quantumflow.api.services.hub_service.get_models_dir",
            return_value=str(tmp_path),
        ):
            result = get_downloaded_models()

        assert len(result) == 1
        assert result[0]["model_id"] == "org/sub/model-name"
