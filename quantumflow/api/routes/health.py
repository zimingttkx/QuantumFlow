"""健康检查路由"""

from fastapi import APIRouter, status

from quantumflow.api.models import HealthResponse
from quantumflow.version import __version__

router = APIRouter(prefix="/health", tags=["System"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查服务健康状态",
)
async def health_check() -> HealthResponse:
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=0,  # TODO: 从应用上下文获取
        checks={
            "api": "healthy",
            "redis": "healthy",  # TODO: 检查Redis连接
            "cluster": "healthy",  # TODO: 检查集群状态
        },
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="就绪检查",
    description="检查服务是否就绪接收请求",
)
async def readiness_check() -> dict:
    """就绪检查"""
    # TODO: 检查所有依赖是否就绪
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
