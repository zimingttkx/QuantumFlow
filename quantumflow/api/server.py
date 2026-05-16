"""FastAPI应用"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from quantumflow.api.routes import router
from quantumflow.api.models import ErrorResponse
from quantumflow.core.exceptions import QuantumFlowError
from quantumflow.version import __version__
from quantumflow.utils.config import get_config
from quantumflow.utils.logging import setup_logging

logger = structlog.get_logger().bind(component="api_server")


def create_app() -> FastAPI:
    """创建FastAPI应用"""

    # 加载配置
    config = get_config()

    # 配置日志
    setup_logging(
        log_level=config.app.log_level,
        log_format="console" if config.app.environment == "development" else "json",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """应用生命周期管理"""
        logger.info("app_starting", version=__version__)
        yield
        logger.info("app_shutdown")

    # 创建应用
    app = FastAPI(
        title="QuantumFlow API",
        description="分布式大模型推理平台 API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 添加中间件
    if config.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 异常处理
    @app.exception_handler(QuantumFlowError)
    async def quantumflow_exception_handler(
        request: Request, exc: QuantumFlowError
    ) -> JSONResponse:
        """QuantumFlow异常处理"""
        logger.error(
            "quantumflow_error",
            path=request.url.path,
            error_code=exc.code,
            message=exc.message,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=exc.to_dict(),
        )

    # 注册路由
    app.include_router(router, prefix="/api/v1")

    # Prometheus监控
    Instrumentator().instrument(app).expose(app, endpoint="/api/v1/metrics")

    # 根路径
    @app.get("/", include_in_schema=False)
    async def root():
        """根路径"""
        return {
            "name": "QuantumFlow",
            "version": __version__,
            "docs": "/docs",
        }

    logger.info("app_created", version=__version__)

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = get_config()

    uvicorn.run(
        "quantumflow.api.server:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.app.environment == "development",
        log_level=config.app.log_level.lower(),
    )
