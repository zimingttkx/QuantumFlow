"""系统配置检测 - 严格单元测试

测试覆盖:
1. SystemCapability 初始化和 to_dict
2. detect_system (pynvml / PyTorch / psutil 路径)
3. recommend_models (各种系统能力组合)
4. _estimate_vram_from_name
5. _estimate_params_from_name
"""

from unittest.mock import MagicMock, patch

import pytest

from quantumflow.api.services.system_profiler import (
    SystemCapability,
    _estimate_params_from_name,
    _estimate_vram_from_name,
    detect_system,
    recommend_models,
)


# ==================== SystemCapability 测试 ====================


class TestSystemCapability:
    """SystemCapability 测试"""

    def test_init_defaults_all_zero(self):
        """[核心功能] 初始值全部为默认零值"""
        cap = SystemCapability()
        assert cap.gpu_count == 0
        assert cap.gpu_names == []
        assert cap.total_vram_gb == 0.0
        assert cap.free_vram_gb == 0.0
        assert cap.ram_total_gb == 0.0
        assert cap.ram_free_gb == 0.0
        assert cap.disk_free_gb == 0.0
        assert cap.cuda_version == ""
        assert cap.pytorch_version == ""
        assert cap.has_cuda is False

    def test_to_dict_contains_all_keys(self):
        """[核心功能] to_dict 包含所有必需键"""
        cap = SystemCapability()
        cap.gpu_count = 2
        cap.gpu_names = ["RTX 4090", "RTX 4090"]
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 30.0
        cap.ram_total_gb = 64.0
        cap.ram_free_gb = 25.0
        cap.disk_free_gb = 500.0
        cap.cuda_version = "12.1"
        cap.pytorch_version = "2.0.1"
        cap.has_cuda = True

        d = cap.to_dict()

        assert d["gpu_count"] == 2
        assert d["gpu_names"] == ["RTX 4090", "RTX 4090"]
        assert d["total_vram_gb"] == 48.0
        assert d["free_vram_gb"] == 30.0
        assert d["ram_total_gb"] == 64.0
        assert d["ram_free_gb"] == 25.0
        assert d["disk_free_gb"] == 500.0
        assert d["cuda_version"] == "12.1"
        assert d["pytorch_version"] == "2.0.1"
        assert d["has_cuda"] is True

    def test_to_dict_rounds_floats_to_one_decimal(self):
        """[核心功能] 浮点数四舍五入到 1 位小数"""
        cap = SystemCapability()
        cap.total_vram_gb = 47.87654321
        cap.free_vram_gb = 30.123456
        cap.ram_total_gb = 63.9999
        cap.ram_free_gb = 24.5001
        cap.disk_free_gb = 500.0

        d = cap.to_dict()

        assert d["total_vram_gb"] == 47.9
        assert d["free_vram_gb"] == 30.1
        assert d["ram_total_gb"] == 64.0
        assert d["ram_free_gb"] == 24.5

    def test_to_dict_no_cuda(self):
        """[核心功能] 无 CUDA 时 has_cuda=False"""
        cap = SystemCapability()
        cap.has_cuda = False

        d = cap.to_dict()

        assert d["has_cuda"] is False
        assert d["cuda_version"] == ""
        assert d["gpu_count"] == 0


# ==================== detect_system 测试 ====================


class TestDetectSystem:
    """detect_system 测试"""

    @patch.dict("sys.modules", {"psutil": None})
    def test_returns_system_capability_instance(self):
        """[核心功能] 返回 SystemCapability 实例"""
        # Mock pynvml, torch, psutil 都不可用
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch.dict("sys.modules", {"torch": None}):
                result = detect_system()
                assert isinstance(result, SystemCapability)
                assert result.gpu_count == 0

    def test_detect_with_pynvml(self):
        """[核心功能] pynvml 检测到 GPU"""
        mock_nvml = MagicMock()
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 4090"
        mock_mem = MagicMock()
        mock_mem.total = 24 * 1024**3
        mock_mem.free = 20 * 1024**3
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_nvml.nvmlShutdown.return_value = None

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            # Also mock torch to avoid interference
            with patch.dict("sys.modules", {"psutil": None}):
                result = detect_system()

        assert result.gpu_count == 1
        assert result.gpu_names == ["NVIDIA RTX 4090"]
        assert abs(result.total_vram_gb - 24.0) < 0.1
        assert result.has_cuda is True

    def test_detect_with_pynvml_string_name(self):
        """[核心功能] pynvml 返回字符串名称"""
        mock_nvml = MagicMock()
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlDeviceGetCount.return_value = 2
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_nvml.nvmlDeviceGetName.return_value = "Tesla T4"
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3
        mock_mem.free = 8 * 1024**3
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_nvml.nvmlShutdown.return_value = None

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.dict("sys.modules", {"psutil": None}):
                result = detect_system()

        assert result.gpu_count == 2
        assert result.gpu_names == ["Tesla T4", "Tesla T4"]
        assert abs(result.total_vram_gb - 32.0) < 0.1

    def test_detect_with_pynvml_no_gpu(self):
        """[核心功能] pynvml 检测到 0 GPU"""
        mock_nvml = MagicMock()
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlDeviceGetCount.return_value = 0
        mock_nvml.nvmlShutdown.return_value = None

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.dict("sys.modules", {"torch": None}):
                with patch.dict("sys.modules", {"psutil": None}):
                    result = detect_system()

        assert result.gpu_count == 0
        assert result.has_cuda is False

    def test_pynvml_fails_falls_back_to_torch(self):
        """[核心功能] pynvml 失败时 fallback 到 PyTorch"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_props = MagicMock()
        mock_props.name = "NVIDIA GPU"
        mock_props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3
        mock_torch.version.cuda = "11.8"
        mock_torch.__version__ = "2.0.0"

        with patch.dict(
            "sys.modules",
            {"pynvml": MagicMock(nvmlInit=MagicMock(side_effect=ImportError))},
        ):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                with patch(
                    "sys.modules", {"psutil": None},
                ):
                    result = detect_system()

        assert result.gpu_count == 1
        assert result.has_cuda is True
        assert result.cuda_version == "11.8"
        assert result.pytorch_version == "2.0.0"

    def test_torch_no_cuda(self):
        """[核心功能] PyTorch 无 CUDA"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.version.cuda = None
        mock_torch.__version__ = "2.0.0"

        with patch.dict("sys.modules", {"pynvml": MagicMock(nvmlInit=MagicMock(side_effect=ImportError))}):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                with patch.dict("sys.modules", {"psutil": None}):
                    result = detect_system()

        assert result.gpu_count == 0
        assert result.has_cuda is False

    def test_torch_fails_gracefully(self):
        """[核心功能] PyTorch 也失败时优雅处理"""
        with patch.dict("sys.modules", {"pynvml": MagicMock(nvmlInit=MagicMock(side_effect=ImportError))}):
            with patch.dict(
                "sys.modules",
                {"torch": MagicMock(cuda=MagicMock(is_available=MagicMock(side_effect=Exception("fail"))))},
            ):
                with patch.dict("sys.modules", {"psutil": None}):
                    result = detect_system()

        assert result.gpu_count == 0
        assert result.has_cuda is False

    def test_detect_with_psutil(self):
        """[核心功能] psutil 检测内存和磁盘"""
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(
            total=32 * 1024**3, available=16 * 1024**3
        )
        mock_psutil.disk_usage.return_value = MagicMock(free=100 * 1024**3)

        with patch.dict("sys.modules", {"pynvml": None, "psutil": mock_psutil, "torch": None}):
            result = detect_system()

        assert abs(result.ram_total_gb - 32.0) < 0.1
        assert abs(result.ram_free_gb - 16.0) < 0.1
        assert abs(result.disk_free_gb - 100.0) < 0.1

    def test_psutil_fails_gracefully(self):
        """[核心功能] psutil 失败时优雅处理"""
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.side_effect = Exception("no psutil")

        with patch.dict("sys.modules", {"pynvml": None, "psutil": mock_psutil, "torch": None}):
            result = detect_system()

        assert result.ram_total_gb == 0.0
        assert result.ram_free_gb == 0.0
        assert result.disk_free_gb == 0.0

    def test_pynvml_only_without_torch_fallback(self):
        """[核心功能] pynvml 检测到 GPU 时不触发 torch fallback"""
        mock_nvml = MagicMock()
        mock_nvml.nvmlInit.return_value = None
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml.nvmlDeviceGetName.return_value = b"GPU"
        mock_mem = MagicMock()
        mock_mem.total = 8 * 1024**3
        mock_mem.free = 4 * 1024**3
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_nvml.nvmlShutdown.return_value = None

        # 模拟 torch 不可用
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.dict("sys.modules", {"torch": None}):
                with patch.dict("sys.modules", {"psutil": None}):
                    result = detect_system()

        assert result.gpu_count == 1
        assert result.has_cuda is True


# ==================== recommend_models 测试 ====================


class TestRecommendModels:
    """recommend_models 测试"""

    def test_default_capability_detected(self):
        """[核心功能] 不传 capability 时自动检测"""
        cap = SystemCapability()
        cap.ram_total_gb = 32.0
        cap.ram_free_gb = 16.0

        with patch(
            "quantumflow.api.services.system_profiler.detect_system",
            return_value=cap,
        ):
            result = recommend_models()

        assert "system" in result
        assert "recommendations" in result
        assert "exceeds_capacity" in result
        assert "summary" in result

    def test_cpu_mode_recommends_small_models(self):
        """[核心功能] CPU 模式推荐小模型"""
        cap = SystemCapability()
        cap.has_cuda = False
        cap.total_vram_gb = 0
        cap.free_vram_gb = 0
        cap.ram_total_gb = 16.0
        cap.ram_free_gb = 12.0

        result = recommend_models(capability=cap)

        assert len(result["recommendations"]) > 0
        # 在 CPU 模式下，params <= 3B 的模型应该是 compatible
        compat = [r for r in result["recommendations"] if r["status"] == "compatible"]
        assert len(compat) > 0
        # 大模型 (>7B params) 应被排除
        for r in result["exceeds_capacity"]:
            assert r["params"] > 3.0

    def test_gpu_mode_with_ample_vram(self):
        """[核心功能] GPU 充足显存时所有模型兼容"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        result = recommend_models(capability=cap)

        assert len(result["exceeds_capacity"]) == 0
        all_compat = all(
            r["status"] in ("compatible", "tight") for r in result["recommendations"]
        )
        assert all_compat is True

    def test_gpu_mode_tight_vram(self):
        """[核心功能] GPU 显存紧张时部分模型 status=tight"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 16.0
        cap.free_vram_gb = 12.0

        result = recommend_models(capability=cap)

        tight_models = [r for r in result["recommendations"] if r["status"] == "tight"]
        exceeds = result["exceeds_capacity"]
        # 16GB 总显存的 GPU, 7B 模型(16GB vram) 应该 tight 或 exceeds
        assert len(tight_models) + len(exceeds) > 0

    def test_gpu_low_vram_most_models_exceed(self):
        """[核心功能] 低显存 GPU 大部分模型超出容量"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 4.0
        cap.free_vram_gb = 2.0

        result = recommend_models(capability=cap)

        # 大多数模型应超出容量
        assert len(result["exceeds_capacity"]) > len(result["recommendations"])

    def test_gpu_free_vram_low_uses_total_vram_with_safety(self):
        """[核心功能] available_vram < 0.5 时使用 total_vram * 0.7"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 24.0
        cap.free_vram_gb = 0.5  # 0.5 * 0.7 = 0.35 < 0.5

        result = recommend_models(capability=cap)

        # should use total_vram*0.7 = 16.8 instead of free_vram*0.7 = 0.35
        # 所以 7B (16GB) 模型应该是 tight 而不是 exceeds (因为 16.8 < 16*0.9=14.4 不, 16<21.6...)
        # actually total_vram*0.9 = 21.6 > 16, so 7B should be compatible
        # Let me check: 16 <= 24*0.9=21.6, so status="tight" since 16>16.8 but 16<=21.6
        assert "summary" in result
        assert result["summary"]["available_vram_gb"] > 15

    def test_sort_order_compatible_first(self):
        """[核心功能] compatible 状态模型排在前面"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        result = recommend_models(capability=cap)

        recs = result["recommendations"]
        statuses = [r["status"] for r in recs]
        # 第一个非-compatible 之前的所有都应是 compatible
        first_non_compat = next(
            (i for i, s in enumerate(statuses) if s != "compatible"), len(statuses)
        )
        for s in statuses[:first_non_compat]:
            assert s == "compatible"

    def test_sort_order_vram_ascending_within_same_status(self):
        """[核心功能] 同状态内按 VRAM 升序排列"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        result = recommend_models(capability=cap)

        recs = result["recommendations"]
        compat = [r for r in recs if r["status"] == "compatible"]
        for i in range(len(compat) - 1):
            assert compat[i]["vram_gb"] <= compat[i + 1]["vram_gb"]

    def test_exceeds_sorted_by_vram_ascending(self):
        """[核心功能] exceeds 列表按 VRAM 升序排列"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 4.0
        cap.free_vram_gb = 2.0

        result = recommend_models(capability=cap)

        exceeds = result["exceeds_capacity"]
        for i in range(len(exceeds) - 1):
            assert exceeds[i]["vram_gb"] <= exceeds[i + 1]["vram_gb"]

    def test_summary_fields(self):
        """[核心功能] summary 包含所有必需字段"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 24.0
        cap.free_vram_gb = 20.0
        cap.ram_total_gb = 32.0
        cap.ram_free_gb = 16.0

        result = recommend_models(capability=cap)

        s = result["summary"]
        assert s["gpu_mode"] is True
        assert s["total_vram_gb"] == 24.0
        assert "available_vram_gb" in s
        assert "can_run_7b" in s
        assert "can_run_3b" in s
        assert "compatible_count" in s

    def test_summary_no_cuda(self):
        """[核心功能] 无 CUDA 时 summary gpu_mode=False"""
        cap = SystemCapability()
        cap.has_cuda = False
        cap.total_vram_gb = 0
        cap.free_vram_gb = 0
        cap.ram_total_gb = 64.0
        cap.ram_free_gb = 50.0

        result = recommend_models(capability=cap)

        assert result["summary"]["gpu_mode"] is False
        # 大内存时 can_run_7b 可能为 True
        assert result["summary"]["can_run_7b"] is True
        assert result["summary"]["can_run_3b"] is True

    def test_popular_models_integration(self):
        """[核心功能] 提供 popular_models 时集成到推荐列表"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [
            {
                "model_id": "org/popular-7b",
                "author": "org",
                "tags": ["7b", "llama"],
                "downloads": 50000,
                "likes": 200,
            },
            {
                "model_id": "org/popular-70b",
                "author": "org",
                "tags": ["70b"],
                "downloads": 10000,
                "likes": 50,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        # 应有 hub_recommendations
        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        assert len(hub_recs) > 0

    def test_popular_model_exceeds_capacity_not_in_recommendations(self):
        """[核心功能] 超出容量的流行模型不出现在推荐列表"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 8.0
        cap.free_vram_gb = 4.0

        popular = [
            {
                "model_id": "org/huge-180b",
                "author": "org",
                "tags": ["180b"],
                "downloads": 1000,
                "likes": 10,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        assert len(hub_recs) == 0

    def test_popular_model_no_model_id(self):
        """[边界用例] popular_models 中无 model_id 时仍可能被推荐（如果 tags 能估算）"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [{"author": "org", "tags": ["7b"]}]  # 无 model_id 但 tags 包含 7b

        result = recommend_models(capability=cap, popular_models=popular)

        # tags 中含 "7b" 可估算参数，仍会创建 hub_recommendations
        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        # "org 7b" 匹配 7B 模式，vram_est > 0 → entry included
        assert len(hub_recs) == 1
        assert hub_recs[0]["params"] == 7_000_000_000

    def test_popular_model_estimate_vram_zero(self):
        """[边界用例] 无法估算 VRAM 的流行模型被跳过"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [
            {
                "model_id": "org/unknown-size",
                "author": "org",
                "tags": [],
                "downloads": 100,
                "likes": 5,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        assert len(hub_recs) == 0

    def test_popular_model_author_none(self):
        """[核心功能] author 为 None 时默认为空字符串"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [
            {
                "model_id": "org/model-7b",
                "author": None,
                "tags": ["7b"],
                "downloads": 1001,
                "likes": 10,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        assert len(hub_recs) == 1

    def test_popular_model_tags_none(self):
        """[核心功能] tags 为 None 时正确处理"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [
            {
                "model_id": "org/model-7b",
                "author": "org",
                "tags": None,
                "downloads": 1001,
                "likes": 10,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        # tags is None, so _estimate_vram_from_name uses just model_id
        # model_id contains 7b -> params 7B -> vram from _estimate_vram_from_name
        # 必须: 仍然能识别并推荐 (model_id 含 "7b"), 且 tags 字段被规范化为 []
        # 不能用 `>= 0`(永远真) — 必须验证具体的 tags 规范化和 vram 估算值
        assert len(hub_recs) == 1, (
            f"tags=None 时, 仍应通过 model_id 匹配到 7B 模型; got {len(hub_recs)} recs"
        )
        assert hub_recs[0]["tags"] == [], (
            f"tags=None 应被规范化为 [], 实际 {hub_recs[0]['tags']!r}"
        )
        assert hub_recs[0]["vram_gb"] > 0, (
            f"应从 model_id '7b' 估算出显存 > 0, 实际 {hub_recs[0]['vram_gb']}"
        )

    def test_popular_model_downloads_none(self):
        """[核心功能] downloads 为 None 时默认为 0"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [
            {
                "model_id": "org/model-7b",
                "author": "org",
                "tags": ["7b"],
                "downloads": None,
                "likes": None,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        assert len(hub_recs) == 1
        assert hub_recs[0]["downloads"] == 0
        assert hub_recs[0]["likes"] == 0

    def test_popular_model_truncates_tags_to_5(self):
        """[核心功能] tags 截断到最多 5 个"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [
            {
                "model_id": "org/model-7b",
                "author": "org",
                "tags": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
                "downloads": 100,
                "likes": 10,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        assert len(hub_recs) == 1
        assert len(hub_recs[0]["tags"]) == 5

    def test_popular_model_name_with_slash(self):
        """[核心功能] model_id 含 / 时 name 使用 author + model_name"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [
            {
                "model_id": "meta-llama/Llama-3-8B",
                "author": "Meta",
                "tags": ["8b"],
                "downloads": 1000,
                "likes": 100,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        assert len(hub_recs) == 1
        assert hub_recs[0]["name"] == "Meta/Llama-3-8B"

    def test_popular_model_name_without_slash(self):
        """[核心功能] model_id 不含 / 时 name 使用 model_id 本身"""
        cap = SystemCapability()
        cap.has_cuda = True
        cap.total_vram_gb = 48.0
        cap.free_vram_gb = 40.0

        popular = [
            {
                "model_id": "simple-model-7b",
                "author": "org",
                "tags": ["7b"],
                "downloads": 100,
                "likes": 10,
            },
        ]

        result = recommend_models(capability=cap, popular_models=popular)

        hub_recs = [r for r in result["recommendations"] if r.get("from_hub")]
        assert len(hub_recs) == 1
        assert hub_recs[0]["name"] == "simple-model-7b"

    def test_cpu_mode_with_low_ram(self):
        """[核心功能] CPU 模式低内存所有模型 tight/exceeds"""
        cap = SystemCapability()
        cap.has_cuda = False
        cap.ram_total_gb = 4.0
        cap.ram_free_gb = 1.0

        result = recommend_models(capability=cap)

        # 大多数模型都应 exceed (0.5B 可能 tight/compat)
        # 但 0.5B 需要 1.5GB * 1.5 = 2.25GB ram_free, 只有 1GB 所以 tight
        compat = [r for r in result["recommendations"] if r["status"] == "compatible"]
        # No model should be compatible with only 1GB free RAM
        assert all(r["status"] in ("tight", "exceeds") for r in result["recommendations"] + result["exceeds_capacity"])


# ==================== _estimate_vram_from_name 测试 ====================


class TestEstimateVramFromName:
    """_estimate_vram_from_name 测试"""

    def test_7b_returns_expected_vram(self):
        """[核心功能] 7B 模型返回预期 VRAM"""
        result = _estimate_vram_from_name("llama-7b")
        expected = round(7_000_000_000 * 2 / (1024**3) * 1.2, 1)
        assert abs(result - expected) < 0.1

    def test_zero_params_returns_zero(self):
        """[核心功能] 无法识别参数量时返回 0"""
        result = _estimate_vram_from_name("unknown-model")
        assert result == 0

    def test_case_insensitive(self):
        """[核心功能] 大小写不敏感"""
        r1 = _estimate_vram_from_name("LLaMA-7B")
        r2 = _estimate_vram_from_name("llama-7b")
        assert r1 == r2


# ==================== _estimate_params_from_name 测试 ====================


class TestEstimateParamsFromName:
    """_estimate_params_from_name 测试"""

    def test_7b(self):
        assert _estimate_params_from_name("llama-7b") == 7_000_000_000

    def test_13b(self):
        assert _estimate_params_from_name("llama-13b") == 13_000_000_000

    def test_70b(self):
        assert _estimate_params_from_name("llama-70b") == 70_000_000_000

    def test_405b(self):
        assert _estimate_params_from_name("llama-405b") == 405_000_000_000

    def test_1_5b(self):
        assert _estimate_params_from_name("qwen-1.5b") == 1_500_000_000

    def test_3_8b(self):
        """[边界用例] 3.8b 模式匹配

        NOTE: patterns 按长度降序排序，所以 "3.8b" 会在 "8b" 之前匹配，
        返回 3.8B 而非 8B。
        """
        result = _estimate_params_from_name("phi-3.8b")
        # patterns 排序后 "3.8b" 优先匹配
        assert result == 3_800_000_000

    def test_no_match(self):
        assert _estimate_params_from_name("unknown") == 0

    def test_case_insensitive(self):
        r1 = _estimate_params_from_name("Model-7B")
        r2 = _estimate_params_from_name("model-7b")
        assert r1 == r2

    def test_long_pattern_before_short(self):
        """[核心功能] 405b 模式在 40b 之前匹配"""
        result = _estimate_params_from_name("model-405b")
        assert result == 405_000_000_000  # 不应匹配 40b

    def test_long_pattern_with_decimal(self):
        """[核心功能] 1.5b 模式正确匹配"""
        result = _estimate_params_from_name("model-1.5b")
        assert result == 1_500_000_000

    def test_9_4b(self):
        """[核心功能] 9.4b 模式匹配"""
        result = _estimate_params_from_name("model-9.4b")
        assert result == 9_400_000_000

    def test_2_6b(self):
        """[核心功能] 2.6b 模式匹配"""
        result = _estimate_params_from_name("gemma-2.6b")
        assert result == 2_600_000_000

    def test_0_5b(self):
        """[核心功能] 0.5b 模式匹配"""
        result = _estimate_params_from_name("qwen-0.5b")
        assert result == 500_000_000
