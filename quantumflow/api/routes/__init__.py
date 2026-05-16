"""API路由"""

from fastapi import APIRouter

from quantumflow.api.routes import health, inference, models, cluster, metrics

router = APIRouter()

# 注册子路由
router.include_router(health.router)
router.include_router(inference.router)
router.include_router(models.router)
router.include_router(cluster.router)
router.include_router(metrics.router)
