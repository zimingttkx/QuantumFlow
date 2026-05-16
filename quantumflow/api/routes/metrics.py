"""监控路由"""

from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter(prefix="/metrics", tags=["Monitoring"])


@router.get(
    "",
    summary="获取指标",
    description="获取Prometheus格式的指标",
    responses={
        200: {
            "content": {"text/plain": {}},
            "description": "Prometheus指标",
        }
    },
)
async def get_metrics() -> Response:
    """获取Prometheus指标"""
    metrics = generate_latest()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)
