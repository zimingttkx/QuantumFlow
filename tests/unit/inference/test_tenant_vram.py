"""租户 VRAM 隔离测试"""
import pytest
from unittest.mock import MagicMock, patch
from quantumflow.inference.vram_manager import VRAMManager


@pytest.fixture
def vram_manager():
    """创建 VRAM 管理器"""
    vm = VRAMManager()
    vm._initialized = True
    return vm


def test_default_tenant_allocation(vram_manager):
    """测试默认租户分配不受配额限制"""
    with patch.object(vram_manager, "_allocate_blocks", return_value=True):
        result = vram_manager.allocate("model1", 4 * 1024**3, tenant_id="default")
        assert result is True


def test_tenant_quota_enforcement(vram_manager):
    """测试租户配额强制执行"""
    vram_manager._tenant_quota_enabled = True
    with patch.object(vram_manager, "_get_tenant_quota", return_value=MagicMock(gpu_memory_mb=4096)):
        with patch.object(vram_manager, "_allocate_blocks", return_value=True):
            result = vram_manager.allocate("model1", 5 * 1024**3, tenant_id="tenant-1")
            assert result is False


def test_tenant_quota_within_limit(vram_manager):
    """测试租户配额内分配"""
    vram_manager._tenant_quota_enabled = True
    with patch.object(vram_manager, "_get_tenant_quota", return_value=MagicMock(gpu_memory_mb=8192)):
        with patch.object(vram_manager, "_allocate_blocks", return_value=True):
            result = vram_manager.allocate("model1", 4 * 1024**3, tenant_id="tenant-1")
            assert result is True


def test_tenant_release_updates_usage(vram_manager):
    """测试释放显存更新租户使用量"""
    vram_manager._tenant_allocations["tenant-1"] = 4 * 1024**3
    vram_manager._model_allocations["model1"] = 4 * 1024**3
    with patch.object(vram_manager, "_release_blocks", return_value=True):
        result = vram_manager.release("model1", tenant_id="tenant-1")
        assert result is True
        assert vram_manager._tenant_allocations.get("tenant-1", 0) == 0


def test_get_tenant_usage(vram_manager):
    """测试获取租户显存使用"""
    vram_manager._tenant_allocations["tenant-1"] = 2 * 1024**3
    vram_manager._tenant_quota_enabled = True
    with patch.object(vram_manager, "_get_tenant_quota", return_value=MagicMock(gpu_memory_mb=4096)):
        usage = vram_manager.get_tenant_usage("tenant-1")
        assert usage["tenant_id"] == "tenant-1"
        assert usage["allocated_mb"] == 2048
        assert usage["quota_mb"] == 4096
        assert usage["utilization"] == 0.5
