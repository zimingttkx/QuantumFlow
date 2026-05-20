"""健康检查路由"""

import time
from datetime import datetime

from fastapi import APIRouter, status

from quantumflow.api.models import HealthResponse
from quantumflow.version import __version__

router = APIRouter(prefix="/health", tags=["System"])

# 应用启动时间（用于计算 uptime）
_app_start_time: float | None = None


def set_app_start_time(t: float | None = None):
    """设置应用启动时间（供测试注入）"""
    global _app_start_time
    _app_start_time = t


def get_app_start_time() -> float:
    """获取应用启动时间"""
    global _app_start_time
    if _app_start_time is not None:
        return _app_start_time
    return time.time()


@router.get(
    "",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查服务健康状态",
)
async def health_check() -> HealthResponse:
    """健康检查接口"""
    checks = {}
    overall_status = "healthy"

    # API 健康检查（自身）
    checks["api"] = "healthy"

    # Redis 连接检查
    try:
        from quantumflow.storage import get_redis_manager

        redis_mgr = await get_redis_manager()
        redis_health = await redis_mgr.health_check()
        if redis_health.get("connected"):
            checks["redis"] = "healthy"
        else:
            checks["redis"] = "unhealthy"
            overall_status = "degraded"
    except Exception:
        checks["redis"] = "unhealthy"
        overall_status = "degraded"

    # 集群状态检查
    try:
        from quantumflow.cluster.manager import ClusterManager

        cluster_mgr = ClusterManager()
        cluster_stats = await cluster_mgr.get_cluster_stats()
        unhealthy = cluster_stats.get("unhealthy_nodes", 0)
        if unhealthy > 0:
            checks["cluster"] = "degraded"
            if overall_status == "healthy":
                overall_status = "degraded"
        else:
            checks["cluster"] = "healthy"
    except Exception:
        checks["cluster"] = "unknown"
        if overall_status == "healthy":
            overall_status = "degraded"

    # 计算 uptime
    uptime_seconds = int(time.time() - get_app_start_time())

    return HealthResponse(
        status=overall_status,
        version=__version__,
        uptime_seconds=uptime_seconds,
        checks=checks,
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="就绪检查",
    description="检查服务是否就绪接收请求",
)
async def readiness_check() -> dict:
    """就绪检查 - 检查所有依赖是否就绪"""
    # 检查 Redis
    try:
        from quantumflow.storage import get_redis_manager

        redis_mgr = await get_redis_manager()
        redis_health = await redis_mgr.health_check()
        if not redis_health.get("connected"):
            return {"ready": False, "reason": "Redis not connected"}
    except Exception:
        return {"ready": False, "reason": "Redis unavailable"}

    # 检查集群（允许空集群，但管理器必须正常）
    try:
        from quantumflow.cluster.manager import ClusterManager

        _ = ClusterManager()
    except Exception:
        return {"ready": False, "reason": "Cluster manager unavailable"}

    return {"ready": True}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="存活检查",
    description="检查服务是否存活",
)
async def liveness_check() -> dict:
    """存活检查"""
    return {"alive": True}
