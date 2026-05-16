"""模型管理路由 - 处理模型的加载和卸载"""

from typing import Dict
from fastapi import APIRouter, HTTPException, status
import structlog

from quantumflow.api.models import (
    LoadModelRequest,
    LoadModelResponse,
    UnloadModelResponse,
    ModelStatusResponse,
)
from quantumflow.inference import get_engine_manager
from quantumflow.core.constants import InferenceBackendType

logger = structlog.get_logger().bind(component="api_model_management")

router = APIRouter(prefix="/models", tags=["Model Management"])

# 模型路径映射
MODEL_PATH_MAPPING: Dict[str, str] = {
    "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}


@router.post(
    "/load",
    response_model=LoadModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="加载模型",
    description="将模型加载到推理引擎",
)
async def load_model(request: LoadModelRequest) -> LoadModelResponse:
    """加载模型到推理引擎"""
    logger.info(
        "load_model_request",
        model=request.model,
        backend=request.backend,
    )

    engine_manager = get_engine_manager()

    # 获取模型路径
    model_path = request.model_path or MODEL_PATH_MAPPING.get(request.model, request.model)

    try:
        success = await engine_manager.load_model(
            model_name=request.model,
            model_path=model_path,
            backend=InferenceBackendType(request.backend) if request.backend else InferenceBackendType.VLLM,
            tensor_parallel=request.tensor_parallel or 1,
            gpu_memory_utilization=request.gpu_memory_utilization or 0.9,
            max_model_len=request.max_model_len or 8192,
            dtype=request.dtype or "auto",
            quantization=request.quantization,
        )

        if success:
            logger.info("model_load_success", model=request.model)
            return LoadModelResponse(
                model=request.model,
                status="loaded",
                message=f"Model {request.model} loaded successfully",
            )
        else:
            logger.error("model_load_failed", model=request.model)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model {request.model}",
            )

    except Exception as e:
        logger.error("model_load_error", model=request.model, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/unload",
    response_model=UnloadModelResponse,
    summary="卸载模型",
    description="从推理引擎卸载模型",
)
async def unload_model(request: LoadModelRequest) -> UnloadModelResponse:
    """卸载模型"""
    logger.info("unload_model_request", model=request.model)

    engine_manager = get_engine_manager()

    try:
        success = await engine_manager.unload_model(request.model)

        if success:
            logger.info("model_unload_success", model=request.model)
            return UnloadModelResponse(
                model=request.model,
                status="unloaded",
                message=f"Model {request.model} unloaded successfully",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model} not found or not loaded",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("model_unload_error", model=request.model, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="获取模型状态",
    description="获取所有已加载模型的状态",
)
async def get_model_status() -> ModelStatusResponse:
    """获取已加载模型状态"""
    engine_manager = get_engine_manager()
    loaded_models = engine_manager.get_loaded_models()

    return ModelStatusResponse(
        loaded_models=loaded_models,
        total=len(loaded_models),
    )


@router.get(
    "/list",
    response_model=Dict,
    summary="可用模型列表",
    description="获取支持的模型列表",
)
async def list_available_models() -> Dict:
    """获取可用模型列表"""
    return {
        "available_models": list(MODEL_PATH_MAPPING.keys()),
        "mappings": MODEL_PATH_MAPPING,
    }
