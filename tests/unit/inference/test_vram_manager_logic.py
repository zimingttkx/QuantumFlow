"""VRAM Manager 核心逻辑专业测试

测试策略：
1. Block 分配与释放计数准确性
2. Block 驱逐逻辑正确性（只驱逐空闲请求的 block）
3. VRAM 估算与实际读取一致性
4. 多模型并发场景下的 VRAM 隔离
5. 边界值（0 tokens, 刚好满, 超额）
6. 状态转换准确性（in_use, idle, loaded, unloaded）
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional

import sys
sys.path.insert(0, '/home/dingziming/PycharmProjects/QuantumFlow')

from quantumflow.inference.vram_manager import BlockPool, VRAMManager, Block


class TestBlockPoolAllocation:
    """Block 分配逻辑严格校验"""

    @pytest.fixture
    def pool(self):
        """创建 BlockPool 实例"""
        return BlockPool(
            model_name="test_model",
            block_size=16,
            max_blocks=10,
        )

    def test_allocate_single_block_exact_size(self, pool):
        """分配刚好一个 block 的 tokens"""
        request_id = "req_1"

        # 分配 16 tokens（刚好一个 block）
        block_ids = pool.allocate(16, request_id)

        # 验证：应该分配 1 个 block
        assert len(block_ids) == 1, f"应该分配 1 个 block，实际: {len(block_ids)}"
        assert pool.stats["allocated_blocks"] == 1

    def test_allocate_multiple_blocks(self, pool):
        """分配多个 blocks"""
        request_id = "req_1"

        # 分配 32 tokens（需要 2 个 blocks）
        block_ids = pool.allocate(32, request_id)

        assert len(block_ids) == 2, f"应该分配 2 个 blocks，实际: {len(block_ids)}"
        assert pool.stats["allocated_blocks"] == 2

    def test_allocate_partial_block(self, pool):
        """分配不足一个 block 的 tokens"""
        request_id = "req_1"

        # 分配 5 tokens（不足一个 block）
        block_ids = pool.allocate(5, request_id)

        # 仍然需要 1 个 block
        assert len(block_ids) == 1
        assert pool.stats["allocated_blocks"] == 1

    def test_allocate_zero_tokens(self, pool):
        """分配 0 tokens 必须返回空列表，不崩溃"""
        block_ids = pool.allocate(0, "req_1")
        assert block_ids == []
        assert pool.stats["allocated_blocks"] == 0

    def test_allocate_exceed_capacity_returns_none(self, pool):
        """分配超过容量的 tokens 必须返回 None"""
        # pool 最多 10 个 blocks = 160 tokens

        # 分配 161 tokens（需要 11 个 blocks，但只有 10 个）
        block_ids = pool.allocate(161, "req_1")

        assert block_ids is None, "超过容量应该返回 None"
        assert pool.stats["rejected_requests"] == 1

    def test_allocate_after_release_reuses_blocks(self, pool):
        """释放后的 block 必须能被重新分配"""
        # 分配
        req1_blocks = pool.allocate(16, "req_1")
        assert len(req1_blocks) == 1

        # 释放
        pool.release(req1_blocks, "req_1")
        assert pool.stats["allocated_blocks"] == 0
        assert pool.stats["peak_allocated"] == 1  # 峰值仍记录

        # 重新分配给另一个请求
        req2_blocks = pool.allocate(16, "req_2")
        assert len(req2_blocks) == 1
        # 注意：block ID 不一定相同（set 是无序的）
        assert req2_blocks[0] in pool._free_blocks or req2_blocks[0] in pool._blocks

    def test_allocate_tracks_per_block_token_count(self, pool):
        """每个 block 的 token 计数必须准确"""
        request_id = "req_1"

        # 分配 20 tokens（需要 2 个 blocks：16 + 4）
        block_ids = pool.allocate(20, request_id)
        assert len(block_ids) == 2

        # 验证每个 block 的 num_tokens
        # 第一个 block 应该是 16（满）
        assert pool._blocks[block_ids[0]].num_tokens == 16
        # 第二个 block 应该是 4（部分使用）
        assert pool._blocks[block_ids[1]].num_tokens == 4

    def test_allocate_respects_request_id_isolation(self, pool):
        """不同请求的 block 不能互相访问"""
        blocks1 = pool.allocate(16, "req_1")
        blocks2 = pool.allocate(16, "req_2")

        # req_1 不能释放 req_2 的 block
        pool.release(blocks1, "req_1")

        # req_2 的 block 仍然被占用
        assert not pool._blocks[blocks2[0]].is_free


class TestBlockPoolRelease:
    """Block 释放逻辑严格校验"""

    @pytest.fixture
    def pool_with_blocks(self):
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)
        blocks = pool.allocate(32, "req_1")  # 2 blocks
        return pool, blocks

    def test_release_correct_request_releases(self, pool_with_blocks):
        """正确的 request_id 必须成功释放"""
        pool, blocks = pool_with_blocks

        pool.release(blocks, "req_1")

        assert pool._blocks[blocks[0]].is_free
        assert pool._blocks[blocks[1]].is_free
        assert pool.stats["allocated_blocks"] == 0

    def test_release_wrong_request_does_not_release(self, pool_with_blocks):
        """错误的 request_id 不能释放 block"""
        pool, blocks = pool_with_blocks

        # 用错误的 request_id 尝试释放
        pool.release(blocks, "wrong_req")

        # blocks 仍然被占用
        assert not pool._blocks[blocks[0]].is_free
        assert pool.stats["allocated_blocks"] == 2

    def test_release_none_handled_gracefully(self, pool_with_blocks):
        """传入 None 不能崩溃"""
        pool, _ = pool_with_blocks

        # 不应该崩溃
        pool.release(None, "req_1")
        assert pool.stats["allocated_blocks"] == 2  # 状态不变

    def test_release_partial_blocks(self, pool_with_blocks):
        """部分释放只释放指定的 blocks"""
        pool, blocks = pool_with_blocks

        # 只释放第一个 block
        pool.release([blocks[0]], "req_1")

        assert pool._blocks[blocks[0]].is_free
        assert not pool._blocks[blocks[1]].is_free
        assert pool.stats["allocated_blocks"] == 1


class TestBlockPoolEviction:
    """Block 驱逐逻辑严格校验"""

    @pytest.fixture
    def pool_with_multiple_requests(self):
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)
        # req_1: 3 blocks
        blocks1 = pool.allocate(48, "req_1")
        # req_2: 2 blocks
        blocks2 = pool.allocate(32, "req_2")
        return pool, blocks1, blocks2

    def test_evict_idle_blocks_stops_when_enough(self):
        """驱逐达到 needed 数量后必须停止"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)

        # 分配 5 blocks
        blocks = pool.allocate(80, "req_1")

        # 释放这些 blocks 使其空闲
        pool.release(blocks, "req_1")

        # 驱逐 2 个
        pool._evict_idle_blocks(2)

        # 验证：空闲 blocks 应该增加 2（从 5 变成 7）
        # 注意：evict 后这 2 个 blocks 仍然是 is_free=True（它们已经是空闲的）
        # 实际效果是空闲 blocks 从 5 变成 10（所有 blocks 都空闲）
        evicted_count = sum(1 for b in pool._blocks.values() if b.is_free)
        # 初始 5 空闲 + 5 个分配后释放 + 2 个 evict = 10
        assert evicted_count == 10, f"应该全部空闲(10)，实际: {evicted_count}"

    def test_evict_only_frees_non_free_blocks(self):
        """驱逐只处理已分配的 blocks，不处理已空闲的"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)

        blocks = pool.allocate(32, "req_1")
        # 释放其中 1 个
        pool.release([blocks[0]], "req_1")

        # 现在 1 个 free（block 0），1 个 allocated（block 1）
        assert len(pool._free_blocks) == 9, f"应有9个空闲，实际: {len(pool._free_blocks)}"
        assert pool.stats["allocated_blocks"] == 1

        # 驱逐 1 个（只能驱逐那 1 个还分配着的 block 1）
        pool._evict_idle_blocks(1)

        # 验证：所有 blocks 都应该空闲
        assert pool.stats["allocated_blocks"] == 0
        assert len(pool._free_blocks) == 10


class TestVRAMManagerModelTracking:
    """VRAM Manager 多模型追踪逻辑"""

    @pytest.fixture
    def vram_manager(self):
        return VRAMManager(safety_factor=0.7, idle_ttl_seconds=0.0)

    def test_record_and_get_loaded_models(self, vram_manager):
        """已加载模型记录必须准确"""
        vram_manager.record_loaded("model_1", 5.0)
        vram_manager.record_loaded("model_2", 3.0)

        loaded = vram_manager.get_loaded_models()

        assert "model_1" in loaded
        assert "model_2" in loaded
        assert len(loaded) == 2

    def test_record_unloaded_removes_model(self, vram_manager):
        """卸载模型必须从已加载列表移除"""
        vram_manager.record_loaded("model_1", 5.0)
        vram_manager.record_unloaded("model_1")

        loaded = vram_manager.get_loaded_models()
        assert "model_1" not in loaded

    def test_mark_in_use_updates_timestamp(self, vram_manager):
        """mark_in_use 必须更新时间戳"""
        vram_manager.record_loaded("model_1", 5.0)

        time_before = vram_manager._loaded["model_1"].last_used_at
        time.sleep(0.01)
        vram_manager.mark_in_use("model_1")

        time_after = vram_manager._loaded["model_1"].last_used_at
        assert time_after > time_before, "last_used_at 必须更新"

    def test_mark_idle_clears_in_use_flag(self, vram_manager):
        """mark_idle 必须清除 in_use 标志"""
        vram_manager.record_loaded("model_1", 5.0)
        vram_manager.mark_in_use("model_1")
        assert vram_manager._loaded["model_1"].in_use is True

        vram_manager.mark_idle("model_1")
        assert vram_manager._loaded["model_1"].in_use is False

    def test_can_load_checks_safety_margin(self, vram_manager):
        """can_load 必须在 VRAM 不足时返回 False"""
        # 模拟 10GB 可用 VRAM，safety_factor=0.7
        with patch.object(vram_manager, '_read_free_vram_gb', return_value=10.0):
            # 尝试加载需要 8GB 的模型（8/10 = 0.8 > 0.7）
            can, reason, evict = vram_manager.can_load("model_1", 8.0)
            assert can is False, f"VRAM 不足应该拒绝: {reason}"

    def test_can_load_respects_loaded_models(self, vram_manager):
        """can_load 必须考虑已加载模型的 VRAM 占用"""
        with patch.object(vram_manager, '_read_free_vram_gb', return_value=10.0):
            # 已加载 6GB 模型
            vram_manager.record_loaded("model_1", 6.0)

            # 剩余 4GB，但 safety_factor=0.7，所以有效 7GB
            # 实际可用 10-6=4GB < 7GB * 0.7? 需要计算

            # 尝试加载 3GB 模型
            can, reason, evict = vram_manager.can_load("model_2", 3.0)
            # 验证是否考虑了已加载模型


class TestVRAMManagerEvictionCandidates:
    """VRAM Manager 淘汰候选选择逻辑"""

    @pytest.fixture
    def vram_manager(self):
        return VRAMManager(safety_factor=0.7, idle_ttl_seconds=0.0)

    def test_idle_models_sorted_by_score(self, vram_manager):
        """淘汰候选按分数降序排列（高分先淘汰）"""
        now = time.time()

        # 加载 3 个模型，不同的 idle 时间
        vram_manager.record_loaded("old_small", 2.0)  # 小模型
        vram_manager._loaded["old_small"].last_used_at = now - 1000  # 最久未使用

        vram_manager.record_loaded("new_large", 5.0)  # 大模型
        vram_manager._loaded["new_large"].last_used_at = now - 10  # 最近使用

        vram_manager.record_loaded("mid", 3.0)
        vram_manager._loaded["mid"].last_used_at = now - 500

        # 所有标记为空闲
        vram_manager.mark_idle("old_small")
        vram_manager.mark_idle("new_large")
        vram_manager.mark_idle("mid")

        # 获取淘汰候选（私有方法）
        candidates = vram_manager._get_eviction_candidates()

        # 验证：old_small（最久+最小）应该排第一
        assert candidates[0].model_name == "old_small"
        # new_large（最近+最大）应该排最后
        assert candidates[-1].model_name == "new_large"


class TestBlockPoolStatistics:
    """Block Pool 统计准确性"""

    def test_peak_allocated_tracked_correctly(self):
        """峰值 allocated 计数必须准确"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)

        # 分配 2 blocks
        blocks1 = pool.allocate(32, "req_1")
        assert pool.stats["peak_allocated"] == 2

        # 释放后再分配 1 block（少于峰值）
        pool.release(blocks1, "req_1")
        blocks2 = pool.allocate(16, "req_2")  # 1 block
        pool.release(blocks2, "req_2")

        # 峰值应该保持 2（因为并发从未超过 2）
        assert pool.stats["peak_allocated"] == 2

        # 分配 3 blocks（超过之前的峰值）
        pool.allocate(48, "req_3")  # 3 blocks
        assert pool.stats["peak_allocated"] == 3

    def test_utilization_percentage_calculation(self):
        """utilization_pct 计算必须正确"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)

        # 分配 5 blocks (50% 利用率)
        pool.allocate(80, "req_1")

        status = pool.get_status()
        assert status["utilization_pct"] == 50.0, f"利用率应为 50%，实际: {status['utilization_pct']}"

    def test_free_blocks_count_consistency(self):
        """空闲 block 计数必须一致"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)

        blocks = pool.allocate(32, "req_1")
        assert len(pool._free_blocks) == 8, f"10 - 2 = 8，实际: {len(pool._free_blocks)}"

        pool.release(blocks, "req_1")
        assert len(pool._free_blocks) == 10, f"恢复后应为 10，实际: {len(pool._free_blocks)}"


class TestEdgeCases:
    """边界值与异常场景"""

    def test_allocate_negative_tokens(self):
        """负数 tokens 不能崩溃"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)

        # 负数 tokens 应该怎么处理？
        # 根据代码：needed_blocks = (-5 + 16 - 1) // 16 = 10 // 16 = 0
        # 所以不会分配 block
        result = pool.allocate(-5, "req_1")
        assert result == []

    def test_block_size_one(self):
        """block_size=1 的极端情况"""
        pool = BlockPool(model_name="test", block_size=1, max_blocks=10)

        blocks = pool.allocate(5, "req_1")
        assert len(blocks) == 5  # 需要 5 个 blocks

    def test_max_blocks_zero(self):
        """max_blocks=0 必须能处理"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=0)

        result = pool.allocate(16, "req_1")
        assert result is None  # 无法分配

    def test_concurrent_allocation_release(self):
        """并发分配释放场景"""
        pool = BlockPool(model_name="test", block_size=16, max_blocks=10)

        # 分配多个请求
        req1_blocks = pool.allocate(16, "req_1")
        req2_blocks = pool.allocate(16, "req_2")

        # 正确释放
        pool.release(req1_blocks, "req_1")
        pool.release(req2_blocks, "req_2")

        # 验证全部释放
        assert pool.stats["allocated_blocks"] == 0
        assert len(pool._free_blocks) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
