"""调度可视化 API — 暴露VRAM/批处理/GPU/调度器内部状态"""

import time

import structlog
from fastapi import APIRouter

logger = structlog.get_logger().bind(component="api_scheduler")

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])


def _build_scheduler_status() -> dict:
    """聚合所有调度相关状态"""
    from quantumflow.inference import get_engine_manager

    mgr = get_engine_manager()
    vram = mgr._vram_manager  # noqa: SLF001 — intentional internal access for observability

    now = time.time()

    # ── VRAM 状态 ──
    available_vram_gb = vram.get_available_vram_gb()
    loaded_models_detail = []
    for name, info in vram._loaded.items():  # noqa: SLF001
        loaded_models_detail.append(
            {
                "model_name": name,
                "estimated_vram_gb": info.estimated_vram_gb,
                "actual_vram_gb": round(info.actual_vram_gb, 1),
                "last_used_at": info.last_used_at,
                "idle_seconds": round(now - info.last_used_at, 1),
                "in_use": info.in_use,
            }
        )

    # ── 淘汰候选（按优先级排序） ──
    eviction_candidates = []
    for info in vram._get_eviction_candidates():  # noqa: SLF001
        eviction_candidates.append(
            {
                "model_name": info.model_name,
                "estimated_vram_gb": info.estimated_vram_gb,
                "idle_seconds": round(now - info.last_used_at, 1),
                "in_use": info.in_use,
            }
        )

    # ── 空闲超时淘汰 ──
    idle_to_evict = vram.get_idle_models_to_evict()

    # ── GPU 状态 ──
    gpu_status = mgr.get_gpu_status()
    if not gpu_status:
        gpu_status = mgr.get_gpu_snapshot()

    # ── 批处理统计 ──
    batch_stats = mgr.get_batch_stats()

    # ── 调度器状态（如果有的话） ──
    scheduler_stats = None
    try:
        # 尝试获取全局 scheduler 实例（如果存在）
        import gc

        from quantumflow.scheduler.scheduler import Scheduler

        for obj in gc.get_objects():
            if isinstance(obj, Scheduler):
                sched = obj
                scheduler_stats = sched.get_stats()
                pending = sched.get_pending_requests()
                running = sched.get_running_requests()
                scheduler_stats["pending_requests_detail"] = [
                    {
                        "request_id": r.request_id,
                        "model": r.model,
                        "priority": r.priority,
                    }
                    for r in pending
                ]
                scheduler_stats["running_requests_detail"] = [
                    {
                        "request_id": rid,
                        "model": item.request.model,
                        "scheduled_at": (
                            item.scheduled_at.isoformat() if item.scheduled_at else None
                        ),
                    }
                    for rid, item in running.items()
                ]
                break
    except Exception:
        pass

    # ── Block 细粒度 VRAM 状态 ──
    block_status = vram.get_all_block_status()

    return {
        "vram": {
            "available_vram_gb": round(available_vram_gb, 1),
            "safety_factor": vram.safety_factor,
            "usable_vram_gb": round(available_vram_gb * vram.safety_factor, 1),
            "idle_ttl_seconds": vram.idle_ttl_seconds,
            "loaded_models": loaded_models_detail,
            "loaded_count": len(loaded_models_detail),
        },
        "eviction": {
            "candidates": eviction_candidates,
            "idle_to_evict": idle_to_evict,
        },
        "gpu": gpu_status,
        "batch": batch_stats,
        "blocks": block_status,
        "scheduler": scheduler_stats,
    }


@router.get(
    "/status",
    summary="获取调度状态",
    description="获取VRAM、批处理、GPU和调度器的完整内部状态，用于调度可视化",
)
async def get_scheduler_status() -> dict:
    return _build_scheduler_status()
