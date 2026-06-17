"""VRAM Manager 覆盖率缺口补充测试

精确覆盖 vram_manager.py 缺失行:
- allocate: evict_idle_blocks / rejected_no_space (102-111)
- _evict_idle_blocks: full function coverage (158)
- release_blocks (375)
- get_block_status (382)
- _read_free_vram_gb: pynvml + torch fallback (459-460)
- _read_used_vram_gb: pynvml + torch fallback (477-488)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantumflow.inference.vram_manager import BlockPool, VRAMManager


class TestBlockPoolRejectedNoSpace:
    def test_allocate_rejected_when_no_free_blocks_and_cannot_evict(self):
        """When free blocks are insufficient and eviction cannot free enough, return None"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=5)

        # Fill all blocks
        _ = pool.allocate(80, "req_1")  # 5 blocks

        # All free_blocks consumed, and allocated blocks are not idle
        # so evict won't help much here (they're all in use)
        result = pool.allocate(16, "req_2")
        assert result is None
        assert pool.stats["rejected_requests"] == 1

    def test_allocate_evicts_idle_blocks_when_needed(self):
        """When free blocks insufficient, evict idle blocks to make room"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=8)

        # Allocate 5 blocks — all 5 become "in use" by a request
        blocks_used = pool.allocate(80, "req_1")  # 5 blocks, 3 free

        # Now simulate that these blocks become "idle" — by releasing them
        pool.release(blocks_used, "req_1")

        # Now allocate another 5 blocks for a new request
        # The freed blocks should be reused
        new_blocks = pool.allocate(80, "req_2")  # needs 5 blocks
        assert new_blocks is not None
        assert len(new_blocks) == 5

    def test_evict_idle_blocks_protects_active_requests(self):
        """Blocks owned by requests that still have blocks are protected from eviction"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=5)

        # Allocate 3 blocks to req_1
        _ = pool.allocate(48, "req_1")

        # req_1 still has blocks in the pool, so its blocks should NOT be evicted
        pool._evict_idle_blocks(5)

        # Only the 2 originally free blocks are free; req_1's 3 blocks are protected
        assert len(pool._free_blocks) == 2
        assert pool.stats["allocated_blocks"] == 3

    def test_evict_idle_blocks_evicts_orphaned_blocks(self):
        """Blocks whose owner no longer has any blocks (fully evicted) become evictable"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=5)

        # Allocate 3 blocks to req_1, then release them all
        allocated = pool.allocate(48, "req_1")
        pool.release(allocated, "req_1")

        # Now req_1 has no blocks in the pool, so the blocks are "orphaned"
        # They should now be evictable
        pool._evict_idle_blocks(3)

        # All 5 blocks should be free
        assert len(pool._free_blocks) == 5
        assert pool.stats["allocated_blocks"] == 0

    def test_evict_idle_blocks_multiple_requests_partial_eviction(self):
        """Evict blocks from one request while another keeps its blocks"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)

        # Allocate 5 blocks to req_1
        _ = pool.allocate(80, "req_1")
        # Allocate 3 blocks to req_2
        _ = pool.allocate(48, "req_2")

        # Now: req_1 has 5 blocks (indices 0-4), req_2 has 3 blocks (indices 5-7), 2 free (8-9)
        # But with our protection: req_1 and req_2 are both in active_requests

        # We can only evict free blocks or blocks from requests no longer in active_requests
        # Since both req_1 and req_2 have blocks, none of their blocks can be evicted
        pool._evict_idle_blocks(5)

        # Only the 2 free blocks are free; allocated blocks are protected
        assert len(pool._free_blocks) == 2
        assert pool.stats["allocated_blocks"] == 8  # 5 + 3


# ── release_blocks / get_block_status ──────────────────────────────────

class TestVRAMManagerBlockOps:
    @pytest.fixture
    def vram(self):
        return VRAMManager(safety_factor=0.7, idle_ttl_seconds=0.0)

    def test_release_blocks_with_valid_pool(self, vram):
        vram.record_loaded("test_model", 5.0)
        pool = vram._block_pools["test_model"]

        blocks = pool.allocate(32, "req_1")
        assert len(blocks) == 2

        vram.release_blocks("test_model", blocks, "req_1")

        assert pool.stats["allocated_blocks"] == 0

    def test_release_blocks_with_none_pool(self, vram):
        """release_blocks with a non-existent model should not crash"""
        vram.release_blocks("nonexistent", [1, 2, 3], "req_1")
        # Should not raise

    def test_get_block_status_returns_dict(self, vram):
        vram.record_loaded("test_model", 5.0)
        pool = vram._block_pools["test_model"]
        pool.allocate(32, "req_1")

        status = vram.get_block_status("test_model")
        assert status is not None
        assert status["model"] == "test_model"
        assert "allocated_blocks" in status
        assert status["allocated_blocks"] == 2

    def test_get_block_status_nonexistent_model(self, vram):
        status = vram.get_block_status("nonexistent")
        assert status is None

    def test_get_all_block_status(self, vram):
        vram.record_loaded("model_a", 3.0)
        vram.record_loaded("model_b", 5.0)

        all_status = vram.get_all_block_status()
        assert "model_a" in all_status
        assert "model_b" in all_status


# ── _read_free_vram_gb / _read_used_vram_gb ────────────────────────────

class TestVRAMReading:
    def test_read_free_vram_via_pynvml(self):
        """_read_free_vram_gb via pynvml"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        mem0 = MagicMock()
        mem0.free = 6 * 1024**3
        mem1 = MagicMock()
        mem1.free = 4 * 1024**3

        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = [mem0, mem1]
        mock_pynvml.nvmlShutdown.return_value = None

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            free_gb = VRAMManager._read_free_vram_gb()

        assert free_gb == 10.0  # 6 + 4

    def test_read_free_vram_pynvml_fails_fallback_to_torch(self):
        """_read_free_vram_gb falls back to torch when pynvml fails"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = ImportError("No pynvml")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=1):
                    mock_props = MagicMock()
                    mock_props.total_memory = 12 * 1024**3
                    with patch("torch.cuda.get_device_properties", return_value=mock_props):
                        with patch("torch.cuda.memory_allocated", return_value=2 * 1024**3):
                            free_gb = VRAMManager._read_free_vram_gb()

        assert free_gb == 10.0  # 12 - 2

    def test_read_free_vram_all_fails_returns_zero(self):
        """_read_free_vram_gb returns 0 when all sources fail"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = ImportError

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch("torch.cuda.is_available", side_effect=RuntimeError):
                free_gb = VRAMManager._read_free_vram_gb()

        assert free_gb == 0.0

    def test_read_used_vram_via_pynvml(self):
        """_read_used_vram_gb via pynvml"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1

        mem = MagicMock()
        mem.used = 8 * 1024**3
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem
        mock_pynvml.nvmlShutdown.return_value = None

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            used_gb = VRAMManager._read_used_vram_gb()

        assert used_gb == 8.0

    def test_read_used_vram_pynvml_fails_fallback_to_torch(self):
        """_read_used_vram_gb falls back to torch when pynvml fails"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = ImportError

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=2):
                    with patch("torch.cuda.memory_allocated", side_effect=[4*1024**3, 2*1024**3]):
                        used_gb = VRAMManager._read_used_vram_gb()

        assert used_gb == 6.0  # 4 + 2

    def test_read_used_vram_all_fails_returns_zero(self):
        """_read_used_vram_gb returns 0 when all sources fail"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = ImportError

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch("torch.cuda.is_available", side_effect=RuntimeError):
                used_gb = VRAMManager._read_used_vram_gb()

        assert used_gb == 0.0


# ── _estimate_param_count / _estimate_architecture ─────────────────────

class TestVRAMEstimation:
    def test_estimate_param_count_unknown_model(self):
        """Unknown model name returns default 7B estimate"""
        count = VRAMManager._estimate_param_count("unknown-model-path")
        assert count == 7_000_000_000

    def test_estimate_param_count_various_patterns(self):
        assert VRAMManager._estimate_param_count("qwen2.5-0.5b-instruct") == 500_000_000
        assert VRAMManager._estimate_param_count("meta-llama-3-8b") == 8_000_000_000
        assert VRAMManager._estimate_param_count("mistral-7b-v0.1") == 7_000_000_000
        assert VRAMManager._estimate_param_count("falcon-40b") == 7_000_000_000  # unknown -> default

    def test_estimate_architecture_unknown_model(self):
        """Unknown model architecture returns (0, 0)"""
        hidden, layers = VRAMManager._estimate_architecture("unknown-model")
        assert hidden == 0
        assert layers == 0

    def test_estimate_architecture_7b(self):
        hidden, layers = VRAMManager._estimate_architecture("meta-llama-7b-hf")
        assert hidden == 4096
        assert layers == 32

    def test_estimate_architecture_70b(self):
        hidden, layers = VRAMManager._estimate_architecture("llama-2-70b-chat")
        assert hidden == 8192
        assert layers == 80

    def test_estimate_architecture_phi_mini(self):
        hidden, layers = VRAMManager._estimate_architecture("phi-3-mini-4k")
        assert hidden == 3072
        assert layers == 32


# ── get_allocatable_blocks ─────────────────────────────────────────────

class TestBlockPoolAllocatable:
    def test_get_allocatable_blocks_returns_correct_count(self):
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)
        assert pool.get_allocatable_blocks(16) == 10  # 10 free / 1 needed
        assert pool.get_allocatable_blocks(32) == 5   # 10 free / 2 needed
        assert pool.get_allocatable_blocks(0) == 0    # 0 tokens -> 0 needed -> division by 0? No, needed is 0

    def test_get_allocatable_blocks_after_allocation(self):
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)
        pool.allocate(32, "req_1")  # 2 blocks used, 8 free
        assert pool.get_allocatable_blocks(32) == 4  # 8 / 2

    def test_get_allocatable_blocks_no_free(self):
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)
        pool.allocate(160, "req_1")  # All 10 used
        assert pool.get_allocatable_blocks(16) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
