"""HuggingFace Hub路由 - 模型发现、搜索、下载"""

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from quantumflow.api.services.hub_service import (
    download_model,
    get_download_progress,
    get_downloaded_models,
    get_model_detail,
    get_trending_models,
    search_models,
    validate_model,
)
from quantumflow.api.services.system_profiler import (
    detect_system,
    recommend_models,
)

logger = structlog.get_logger().bind(component="api_hub")

router = APIRouter(prefix="/hub", tags=["Model Hub"])

# ---- Pydantic schemas ----


class HubModelInfo(BaseModel):
    model_id: str = Field(..., description="HuggingFace模型ID")
    author: str = Field(default="unknown")
    downloads: int = Field(default=0)
    likes: int = Field(default=0)
    pipeline_tag: str = Field(default="unknown")
    tags: list = Field(default_factory=list)
    last_modified: str = Field(default="")
    gated: bool = Field(default=False)
    private: bool = Field(default=False)
    library_name: str = Field(default="unknown")

    class Config:
        extra = "allow"


class ValidateResponse(BaseModel):
    valid: bool
    model_id: str
    exists: bool
    gated: bool = False
    error: str | None = None
    info: dict | None = None


class DownloadRequest(BaseModel):
    model_id: str = Field(..., description="HuggingFace模型ID，如 Qwen/Qwen2.5-1.5B-Instruct")


class DownloadResponse(BaseModel):
    success: bool
    model_id: str
    local_path: str = ""
    error: str | None = None


class ProgressResponse(BaseModel):
    model_id: str
    progress: float = Field(default=0, description="下载进度 0-100，-1表示不在下载中")


class DownloadedModel(BaseModel):
    model_id: str
    local_path: str
    size_bytes: int
    size_gb: float


class RecommendationResponse(BaseModel):
    system: dict
    recommendations: list
    exceeds_capacity: list
    summary: dict


# ---- Routes ----


@router.get(
    "/trending",
    summary="获取热门模型",
    description="从HuggingFace获取下载量最高的文本生成模型列表",
)
async def trending_models(
    limit: int = Query(default=30, ge=1, le=100, description="返回数量"),
):
    """获取HF热门模型"""
    models = await get_trending_models(limit=limit)
    return {"models": models, "total": len(models)}


@router.get(
    "/search",
    summary="搜索模型",
    description="在HuggingFace上搜索模型",
)
async def search_hub_models(
    q: str = Query(..., description="搜索关键词"),
    limit: int = Query(default=20, ge=1, le=50, description="返回数量"),
):
    """搜索模型"""
    if not q or len(q.strip()) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="搜索关键词不能为空",
        )
    models = await search_models(query=q.strip(), limit=limit)
    return {"query": q, "models": models, "total": len(models)}


@router.get(
    "/validate",
    response_model=ValidateResponse,
    summary="验证模型是否存在",
    description="检查模型ID是否在HuggingFace上存在，不会下载模型文件",
)
async def validate_hub_model(
    model_id: str = Query(..., description="HuggingFace模型ID"),
):
    """验证模型是否存在"""
    result = await validate_model(model_id)
    return result


@router.get(
    "/detail",
    summary="获取模型详细信息",
    description="获取模型在HF上的详细信息，包括估算的参数量和显存需求",
)
async def model_detail(
    model_id: str = Query(..., description="HuggingFace模型ID"),
):
    """获取模型详情"""
    detail = await get_model_detail(model_id)
    if "error" in detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail["error"],
        )
    return detail


@router.post(
    "/download",
    response_model=DownloadResponse,
    summary="下载模型",
    description="从HuggingFace下载模型到本地缓存",
)
async def download_hub_model(request: DownloadRequest):
    """下载模型"""
    logger.info("download_request", model_id=request.model_id)

    result = await download_model(model_id=request.model_id)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    return DownloadResponse(
        success=True,
        model_id=request.model_id,
        local_path=result["local_path"],
    )


@router.get(
    "/download/progress",
    response_model=ProgressResponse,
    summary="查询下载进度",
    description="查询模型下载进度",
)
async def download_progress(
    model_id: str = Query(..., description="HuggingFace模型ID"),
):
    """查询下载进度"""
    progress = get_download_progress(model_id)
    return ProgressResponse(model_id=model_id, progress=progress)


@router.get(
    "/downloaded",
    summary="已下载模型列表",
    description="列出本地已下载的模型",
)
async def downloaded_models():
    """列出已下载模型"""
    models = get_downloaded_models()
    return {"models": models, "total": len(models)}


@router.get(
    "/recommendations",
    response_model=RecommendationResponse,
    summary="模型推荐",
    description="基于当前系统配置，推荐合适的模型",
)
async def get_recommendations():
    """系统检测 + 模型推荐"""
    cap = detect_system()
    trending = await get_trending_models(limit=10)
    result = recommend_models(capability=cap, popular_models=trending)
    return result
