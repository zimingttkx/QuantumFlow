"""API路由"""

from fastapi import APIRouter

from quantumflow.api.routes import health, inference, models, cluster, metrics, model_management, hub

router = APIRouter()

# 注册子路由
router.include_router(health.router)
router.include_router(inference.router)
router.include_router(model_management.router)
router.include_router(models.router)
router.include_router(cluster.router)
router.include_router(metrics.router)
router.include_router(hub.router)
