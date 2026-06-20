"""VRAM感知模型加载管理 — 单卡GPU利用率优化核心"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger().bind(component="vram_manager")

# 安全边界：仅使用可用VRAM的70%，预留推理峰值/碎片空间
VRAM_SAFETY_FACTOR = 0.7

# Block 大小配置（参考 vLLM PagedAttention）
DEFAULT_BLOCK_SIZE = 16  # 每个 block 多少 tokens
DEFAULT_MAX_BLOCKS_PER_MODEL = 256  # 每个模型最大 block 数


@dataclass
class LoadedModelInfo:
    """已加载模型的VRAM追踪信息"""

    model_name: str
    estimated_vram_gb: float
    actual_vram_gb: float = 0.0
    last_used_at: float = field(default_factory=time.time)
    in_use: bool = False  # 正在推理中，受保护不可淘汰


@dataclass
class Block:
    """VRAM 显存块 — 类比 PagedAttention 的物理页"""

    block_id: int
    num_tokens: int  # 该 block 承载的 token 数
    owner_request_id: str = ""  # 当前占用该 block 的请求
    is_free: bool = True


class BlockPool:
    """
    细粒度 VRAM 显存块管理器。

    参考 vLLM PagedAttention 的 block 分配策略：
    - 每个模型预分配固定数量的 block（物理页）
    - 每个请求按需占用 blocks（logical pages）
    - 推理完成后释放 blocks

    这样可以：
    1. 追踪每个请求的 KV cache 显存占用
    2. 拒绝超出 block 配额的超大请求
    3. 动态调整并发上限
    """

    def __init__(
        self,
        model_name: str,
        max_blocks: int = DEFAULT_MAX_BLOCKS_PER_MODEL,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        self.model_name = model_name
        self.block_size = block_size
        self.max_blocks = max_blocks

        # 初始化 block 池
        self._blocks: dict[int, Block] = {
            i: Block(block_id=i, num_tokens=block_size) for i in range(max_blocks)
        }
        self._free_blocks: set[int] = set(range(max_blocks))

        # 统计
        self.stats = {
            "total_blocks": max_blocks,
            "allocated_blocks": 0,
            "peak_allocated": 0,
            "total_requests": 0,
            "rejected_requests": 0,
        }

    def allocate(self, num_tokens: int, request_id: str) -> list[int] | None:
        """
        为请求分配足够的 blocks 承载 num_tokens。

        Returns:
            分配的 block_ids 列表，失败返回 None
        """
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size

        if needed_blocks > self.max_blocks:
            logger.warning(
                "block_alloc_rejected_too_large",
                model=self.model_name,
                needed_blocks=needed_blocks,
                max_blocks=self.max_blocks,
            )
            self.stats["rejected_requests"] += 1
            return None

        if len(self._free_blocks) < needed_blocks:
            # 尝试驱逐空闲请求的 blocks（保留 in_use 的）
            self._evict_idle_blocks(needed_blocks)
            if len(self._free_blocks) < needed_blocks:
                logger.warning(
                    "block_alloc_rejected_no_space",
                    model=self.model_name,
                    needed_blocks=needed_blocks,
                    free_blocks=len(self._free_blocks),
                )
                self.stats["rejected_requests"] += 1
                return None

        # 分配
        allocated = []
        for _ in range(needed_blocks):
            block_id = self._free_blocks.pop()
            self._blocks[block_id].is_free = False
            self._blocks[block_id].owner_request_id = request_id
            self._blocks[block_id].num_tokens = min(self.block_size, num_tokens)
            num_tokens -= self._blocks[block_id].num_tokens
            allocated.append(block_id)

        self.stats["allocated_blocks"] += len(allocated)
        self.stats["total_requests"] += 1
        self.stats["peak_allocated"] = max(
            self.stats["peak_allocated"], self.stats["allocated_blocks"]
        )

        logger.debug(
            "blocks_allocated",
            model=self.model_name,
            request=request_id,
            num_blocks=len(allocated),
            total_allocated=self.stats["allocated_blocks"],
        )
        return allocated

    def release(self, block_ids: list[int], request_id: str):
        """释放指定 blocks"""
        if block_ids is None:
            return
        for block_id in block_ids:
            if block_id in self._blocks and not self._blocks[block_id].is_free:
                if self._blocks[block_id].owner_request_id == request_id:
                    self._blocks[block_id].is_free = True
                    self._blocks[block_id].owner_request_id = ""
                    self._free_blocks.add(block_id)
                    self.stats["allocated_blocks"] -= 1

    def _evict_idle_blocks(self, needed: int):
        """驱逐当前未使用的请求的 blocks"""
        # 找出有活跃请求的 block（同一 request 还有其他 in-use blocks）
        # 只有完全空闲的 block 才能被驱逐
        active_requests: set = set()
        for block in self._blocks.values():
            if not block.is_free and block.owner_request_id:
                active_requests.add(block.owner_request_id)

        evicted = 0
        for block_id, block in self._blocks.items():
            if block.is_free:
                continue
            # 跳过活跃请求的 blocks（同一 request 还有其他 blocks 在用）
            if block.owner_request_id and block.owner_request_id in active_requests:
                continue
            if evicted >= needed:
                break
            block.is_free = True
            block.owner_request_id = ""
            self._free_blocks.add(block_id)
            self.stats["allocated_blocks"] -= 1
            evicted += 1

    def get_status(self) -> dict:
        """获取 block 池状态"""
        return {
            "model": self.model_name,
            "total_blocks": self.max_blocks,
            "free_blocks": len(self._free_blocks),
            "allocated_blocks": self.stats["allocated_blocks"],
            "peak_allocated": self.stats["peak_allocated"],
            "utilization_pct": round(self.stats["peak_allocated"] / self.max_blocks * 100, 1),
            "rejected_requests": self.stats["rejected_requests"],
        }

    def get_allocatable_blocks(self, num_tokens: int) -> int:
        """给定 token 数，返回还能分配多少个请求"""
        needed = (num_tokens + self.block_size - 1) // self.block_size
        if needed == 0:
            return 0
        return max(0, len(self._free_blocks) // needed)


class VRAMManager:
    """
    VRAM感知的模型加载管理器

    职责：
    - 检测当前GPU可用VRAM
    - 估算模型VRAM需求
    - 判断加载可行性（直接加载/需淘汰/拒绝）
    - 追踪已加载模型的VRAM占用
    - LRU淘汰决策
    - 空闲超时自动卸载
    - Block 级别细粒度显存管理（PagedAttention 风格）
    """

    def __init__(
        self,
        safety_factor: float = VRAM_SAFETY_FACTOR,
        idle_ttl_seconds: float = 0.0,  # 0=禁用空闲卸载
    ):
        self.safety_factor = safety_factor
        self.idle_ttl_seconds = idle_ttl_seconds
        self._loaded: dict[str, LoadedModelInfo] = {}
        self._block_pools: dict[str, BlockPool] = {}  # model_name -> BlockPool
        self._tenant_allocations: dict[str, int] = {}
        self._tenant_quota_enabled: bool = True
        self._model_allocations: dict[str, int] = {}  # model_name -> size_bytes
        self._model_total_allocated: dict[str, int] = {}  # model_name -> total bytes allocated
        self._model_reserved_bytes: dict[str, int] = {}  # model_name -> reserved GPU bytes

    # ── public API ─────────────────────────────────────────────

    def get_available_vram_gb(self) -> float:
        """获取当前可用VRAM (GB)"""
        return self._read_free_vram_gb()

    def get_vram_utilization(self) -> float:
        """
        获取当前 VRAM 利用率。

        Returns:
            0.0 到 1.0 之间的利用率值
            - 0.0 = 空闲
            - 1.0 = 完全占满
        """
        used = self._read_used_vram_gb()
        free = self._read_free_vram_gb()
        total = used + free
        if total <= 0:
            return 0.0
        return round(used / total, 3)

    def estimate_model_vram_gb(self, model_path: str, max_model_len: int = 2048) -> float:
        """估算模型需要的VRAM (GB)，含模型权重+KV cache"""
        param_count = self._estimate_param_count(model_path)
        if param_count == 0:
            return 0.0

        # FP16 weights: 2 bytes per param, ~20% overhead
        model_gb = param_count * 2 / (1024**3) * 1.2

        # KV cache: 2 * num_layers * hidden_size * max_seq_len * 2 bytes (FP16)
        hidden_size, num_layers = self._estimate_architecture(model_path)
        if hidden_size > 0 and num_layers > 0:
            kv_cache_gb = 2 * num_layers * hidden_size * max_model_len * 2 / (1024**3)
        else:
            kv_cache_gb = model_gb * 0.3  # 粗略估算

        return round(model_gb + kv_cache_gb, 1)

    def can_load(
        self,
        model_name: str,
        required_vram_gb: float,
    ) -> tuple[bool, str, list[str]]:
        """
        判断是否可以加载模型。

        Returns:
            (can_load, reason, models_to_evict)
            - can_load: 是否可以加载
            - reason: 判断原因
            - models_to_evict: 需淘汰的模型名列表（仅当 can_load=True 且需淘汰时非空）
        """
        available = self.get_available_vram_gb()
        usable = available * self.safety_factor

        if required_vram_gb <= 0:
            return False, f"无法估算模型 {model_name} 的VRAM需求", []

        if required_vram_gb <= usable:
            return True, f"VRAM充足 (可用{usable:.1f}GB, 需要{required_vram_gb:.1f}GB)", []

        # 不够 — 计算需要淘汰多少空间
        shortage = required_vram_gb - usable
        # 考虑淘汰已加载模型（加上当前他们占用的空间）
        evictable = self._get_eviction_candidates()
        freed = 0.0
        to_evict = []
        for info in evictable:
            freed += info.estimated_vram_gb
            to_evict.append(info.model_name)
            if freed >= shortage:
                break

        if freed >= shortage:
            return (
                True,
                f"需淘汰 {len(to_evict)} 个模型释放 {freed:.1f}GB 以加载 {model_name} ({required_vram_gb:.1f}GB)",
                to_evict,
            )

        total_freeable = freed + usable
        return (
            False,
            (
                f"VRAM不足: 需要{required_vram_gb:.1f}GB, 可用{usable:.1f}GB, "
                f"即使淘汰所有模型也仅释放{total_freeable:.1f}GB"
            ),
            [],
        )

    def record_loaded(self, model_name: str, estimated_vram_gb: float):
        """记录模型已加载，并初始化 BlockPool"""
        self._loaded[model_name] = LoadedModelInfo(
            model_name=model_name,
            estimated_vram_gb=estimated_vram_gb,
        )
        # 初始化 Block Pool（参考 vLLM PagedAttention 策略）
        # max_blocks 根据模型大小和 VRAM 计算
        max_blocks = self._estimate_max_blocks(model_name, estimated_vram_gb)
        self._block_pools[model_name] = BlockPool(
            model_name=model_name,
            max_blocks=max_blocks,
            block_size=DEFAULT_BLOCK_SIZE,
        )
        logger.info(
            "vram_model_tracked",
            model=model_name,
            estimated_vram_gb=estimated_vram_gb,
            total_tracked=len(self._loaded),
            block_pool=f"{max_blocks} blocks",
        )

    def record_unloaded(self, model_name: str):
        """记录模型已卸载"""
        self._loaded.pop(model_name, None)
        self._block_pools.pop(model_name, None)

    def mark_in_use(self, model_name: str):
        """标记模型正在推理（受保护，不可淘汰）"""
        info = self._loaded.get(model_name)
        if info:
            info.in_use = True
            info.last_used_at = time.time()

    def mark_idle(self, model_name: str):
        """标记模型推理完成"""
        info = self._loaded.get(model_name)
        if info:
            info.in_use = False

    def update_actual_vram(self, model_name: str):
        """推理完成后，从GPU读取实际VRAM占用并更新记录。

        注意：这里记录的是当前 GPU 总已用 VRAM（包含所有已加载模型），
        而非单个模型的增量 VRAM。如需精确的模型级增量，应在加载前后
        分别采样并计算差值。
        """
        info = self._loaded.get(model_name)
        if info is None:
            return
        # 读取当前 GPU 总已用 VRAM
        info.actual_vram_gb = self._read_used_vram_gb()
        logger.info(
            "vram_actual_updated",
            model=model_name,
            actual_vram_gb=round(info.actual_vram_gb, 1),
        )

    def get_idle_models_to_evict(self) -> list[str]:
        """返回空闲超时、应被淘汰的模型名列表"""
        if self.idle_ttl_seconds <= 0:
            return []
        now = time.time()
        candidates = []
        for name, info in self._loaded.items():
            if not info.in_use and (now - info.last_used_at) > self.idle_ttl_seconds:
                candidates.append(name)
        return candidates

    def get_loaded_models(self) -> list[str]:
        return list(self._loaded.keys())

    # ── Block 级别细粒度 VRAM 管理 ──────────────────────────────

    def allocate_blocks(
        self, model_name: str, num_tokens: int, request_id: str
    ) -> list[int] | None:
        """
        为请求分配 KV Cache blocks。

        参考 vLLM PagedAttention：
        - 每个 block 固定 16 tokens
        - 超出 max_blocks 的请求被拒绝
        - 用于动态控制并发数和显存占用

        Returns:
            分配的 block_ids 列表，失败返回 None
        """
        pool = self._block_pools.get(model_name)
        if not pool:
            return None
        return pool.allocate(num_tokens, request_id)

    def release_blocks(self, model_name: str, block_ids: list[int], request_id: str):
        """释放指定 blocks"""
        pool = self._block_pools.get(model_name)
        if pool:
            pool.release(block_ids, request_id)

    def get_block_status(self, model_name: str) -> dict | None:
        """获取模型的 block 池状态"""
        pool = self._block_pools.get(model_name)
        if not pool:
            return None
        return pool.get_status()

    def get_all_block_status(self) -> dict:
        """获取所有模型的 block 池状态"""
        return {name: pool.get_status() for name, pool in self._block_pools.items()}

    # ── tenant-aware allocation ──────────────────────────────────

    def allocate(self, model_name: str, size_bytes: int, tenant_id: str = "default") -> bool:
        """租户感知的显存分配"""
        if tenant_id != "default" and self._tenant_quota_enabled:
            quota = self._get_tenant_quota(tenant_id)
            if quota.gpu_memory_mb > 0:
                current_usage = self._tenant_allocations.get(tenant_id, 0)
                requested_total = current_usage + size_bytes
                max_bytes = quota.gpu_memory_mb * 1024 * 1024
                if requested_total > max_bytes:
                    return False
        success = self._allocate_blocks(model_name, size_bytes)
        if success:
            self._tenant_allocations[tenant_id] = self._tenant_allocations.get(tenant_id, 0) + size_bytes
            self._model_allocations[model_name] = size_bytes
        return success

    def release(self, model_name: str, tenant_id: str = "default") -> bool:
        """释放模型显存并更新租户使用量"""
        size = self._model_allocations.get(model_name, 0)
        if self._release_blocks(model_name):
            if tenant_id in self._tenant_allocations:
                self._tenant_allocations[tenant_id] = max(0, self._tenant_allocations[tenant_id] - size)
            return True
        return False

    def _allocate_blocks(self, model_name: str, size_bytes: int) -> bool:
        """内部块分配逻辑 — 追踪模型级别的显存分配"""
        current = self._model_reserved_bytes.get(model_name, 0)
        self._model_reserved_bytes[model_name] = current + size_bytes
        logger.debug(
            "vram_blocks_allocated",
            model=model_name,
            size_bytes=size_bytes,
            total_reserved=self._model_reserved_bytes[model_name],
        )
        return True

    def _release_blocks(self, model_name: str) -> bool:
        """内部块释放逻辑 — 释放模型级别的显存分配"""
        size = self._model_allocations.get(model_name, 0)
        current = self._model_reserved_bytes.get(model_name, 0)
        self._model_reserved_bytes[model_name] = max(0, current - size)
        logger.debug(
            "vram_blocks_released",
            model=model_name,
            size_bytes=size,
            total_reserved=self._model_reserved_bytes[model_name],
        )
        return True

    def _get_tenant_quota(self, tenant_id: str):
        """获取租户显存配额（从 Redis 加载，key 与 tenants.py 对齐）"""
        from quantumflow.api.models.tenant import QuotaConfig
        from quantumflow.core.constants import DEFAULT_TENANT_QUOTA, TENANT_PREFIX

        redis = None
        try:
            from quantumflow.storage import get_redis_manager_sync
            redis = get_redis_manager_sync().get_client()
        except Exception:
            pass
        if redis:
            hashed_key = redis.get(f"qf:tenant:id:{tenant_id}")
            if hashed_key:
                if isinstance(hashed_key, bytes):
                    hashed_key = hashed_key.decode()
                data = redis.hgetall(f"{TENANT_PREFIX}{hashed_key}")
                if data:
                    return QuotaConfig(
                        requests_per_minute=int(data.get(b"quota_requests_per_minute", DEFAULT_TENANT_QUOTA["requests_per_minute"])),
                        requests_per_day=int(data.get(b"quota_requests_per_day", DEFAULT_TENANT_QUOTA["requests_per_day"])),
                        max_tokens_per_request=int(data.get(b"quota_max_tokens", DEFAULT_TENANT_QUOTA["max_tokens_per_request"])),
                        gpu_memory_mb=int(data.get(b"quota_gpu_memory", DEFAULT_TENANT_QUOTA["gpu_memory_mb"])),
                        concurrent_requests=int(data.get(b"quota_concurrent", DEFAULT_TENANT_QUOTA["concurrent_requests"])),
                    )
        return QuotaConfig(**DEFAULT_TENANT_QUOTA)

    def get_tenant_usage(self, tenant_id: str) -> dict:
        """获取租户显存使用情况"""
        allocated = self._tenant_allocations.get(tenant_id, 0)
        quota = self._get_tenant_quota(tenant_id)
        return {
            "tenant_id": tenant_id,
            "allocated_bytes": allocated,
            "allocated_mb": allocated / (1024 * 1024),
            "quota_mb": quota.gpu_memory_mb,
            "utilization": allocated / (quota.gpu_memory_mb * 1024 * 1024) if quota.gpu_memory_mb > 0 else 0.0,
        }

    # ── internal ───────────────────────────────────────────────

    def _estimate_max_blocks(self, model_name: str, estimated_vram_gb: float) -> int:
        """
        根据模型大小估算最大 block 数。

        每个 block = 16 tokens × 16 layers × 2 (K+V) × hidden_size × 2 bytes
        ≈ 0.1 MB per block (for 1.5B model)

        估算：VRAM 总量 × safety_factor × 80% 用于 KV cache / 单 block 大小
        """
        # 粗略估算：1GB VRAM ≈ 10000 blocks
        # 安全范围内用于 KV cache 块的数量
        usable_gb = estimated_vram_gb * 0.5  # 一半给 KV cache
        base_blocks = int(usable_gb * 10000)
        # 限制在合理范围
        return max(16, min(base_blocks, 2048))

    def _get_eviction_candidates(self) -> list[LoadedModelInfo]:
        """淘汰候选列表：综合评分排序（低分优先淘汰）。

        评分 = last_used_at（归一化）+ 模型大小权重。
        最近使用且体积小的模型得分高，优先保留。
        """
        candidates = [m for m in self._loaded.values() if not m.in_use]
        if not candidates:
            return []

        now = time.time()
        # 归一化：最久未用的得0，最近使用的得1
        max_age = max(now - m.last_used_at for m in candidates) or 1.0
        # 归一化大小：最大模型得1
        max_vram = max(m.estimated_vram_gb for m in candidates) or 1.0

        def score(m: LoadedModelInfo) -> float:
            age_norm = (now - m.last_used_at) / max_age  # 0=最近, 1=最久
            size_norm = m.estimated_vram_gb / max_vram  # 0=最小, 1=最大
            # 得分=年龄(70%权重) + 大小(30%权重) — 高年龄+大体积=高分=优先淘汰
            return age_norm * 0.7 + size_norm * 0.3

        candidates.sort(key=score, reverse=True)  # 高分优先淘汰
        return candidates

    @staticmethod
    def _read_free_vram_gb() -> float:
        """读取可用VRAM (GB)"""
        # pynvml first
        try:
            import pynvml

            pynvml.nvmlInit()
            total_free = 0.0
            for i in range(pynvml.nvmlDeviceGetCount()):
                mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
                total_free += mem.free / (1024**3)
            pynvml.nvmlShutdown()
            if total_free > 0:
                return total_free
        except Exception:
            pass

        # PyTorch fallback
        try:
            import torch

            if torch.cuda.is_available():
                total_free = 0.0
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_free += (props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
                return total_free
        except Exception:
            pass

        return 0.0

    @staticmethod
    def _read_used_vram_gb() -> float:
        """读取已用VRAM (GB)"""
        try:
            import pynvml

            pynvml.nvmlInit()
            total_used = 0.0
            for i in range(pynvml.nvmlDeviceGetCount()):
                mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
                total_used += mem.used / (1024**3)
            pynvml.nvmlShutdown()
            return total_used
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                return sum(
                    torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
                ) / (1024**3)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _estimate_param_count(model_path: str) -> int:
        """从模型名称估算参数量"""
        patterns = [
            ("0.5b", 500_000_000),
            ("1.5b", 1_500_000_000),
            ("2.6b", 2_600_000_000),
            ("3.8b", 3_800_000_000),
            ("14b", 14_000_000_000),
            ("13b", 13_000_000_000),
            ("11b", 11_000_000_000),
            ("9b", 9_000_000_000),
            ("8b", 8_000_000_000),
            ("7b", 7_000_000_000),
            ("72b", 72_000_000_000),
            ("70b", 70_000_000_000),
            ("34b", 34_000_000_000),
            ("20b", 20_000_000_000),
            ("3b", 3_000_000_000),
            ("2b", 2_000_000_000),
            ("1b", 1_000_000_000),
        ]
        path_lower = model_path.lower()
        for pattern, count in patterns:
            if pattern in path_lower:
                return count
        # 无法识别时返回保守估计（约 7B），避免 VRAM 估算返回 0 从而拒绝加载
        return 7_000_000_000

    @staticmethod
    def _estimate_architecture(model_path: str) -> tuple[int, int]:
        """
        从模型名称估算 (hidden_size, num_layers)。
        基于常见模型架构的典型值。
        """
        path_lower = model_path.lower()
        # LLaMA/Qwen 体系 — 长模式优先避免 "3b" 误匹配 "13b"
        if "0.5b" in path_lower:
            return (896, 24)
        if "1.5b" in path_lower:
            return (1536, 28)
        if "14b" in path_lower:
            return (5120, 40)
        if "13b" in path_lower:
            return (5120, 40)
        if "8b" in path_lower:
            return (4096, 32)
        if "7b" in path_lower:
            return (4096, 32)
        if "3b" in path_lower:
            return (2560, 32)
        if "70b" in path_lower or "72b" in path_lower:
            return (8192, 80)
        # Phi-3-mini
        if "phi" in path_lower and "mini" in path_lower:
            return (3072, 32)
        # 默认小模型
        return (0, 0)
