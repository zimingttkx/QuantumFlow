"""API路由"""

from fastapi import APIRouter

from quantumflow.api.routes import (
    cluster,
    health,
    hub,
    inference,
    metrics,
    model_management,
    models,
    scheduler,
)

router = APIRouter()

# 注册子路由
router.include_router(health.router)
router.include_router(inference.router)
router.include_router(model_management.router)
router.include_router(models.router)
router.include_router(cluster.router)
router.include_router(metrics.router)
router.include_router(hub.router)
router.include_router(scheduler.router)
