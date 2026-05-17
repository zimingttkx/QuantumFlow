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
from quantumflow.api.services.hub_service import validate_model, get_downloaded_models

logger = structlog.get_logger().bind(component="api_model_management")

router = APIRouter(prefix="/models", tags=["Model Management"])

# 模型路径映射 - HuggingFace可以直接使用Hub ID
MODEL_PATH_MAPPING: Dict[str, str] = {
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Phi-3-mini-4k": "microsoft/Phi-3-mini-4k-instruct",
}


def _resolve_model_path(model_name: str, model_path: str = None) -> str:
    """解析模型路径"""
    if model_path:
        return model_path
    # 检查已知映射
    if model_name in MODEL_PATH_MAPPING:
        return MODEL_PATH_MAPPING[model_name]
    # 检查本地下载的模型
    downloaded = {m["model_id"]: m for m in get_downloaded_models()}
    if model_name in downloaded:
        return downloaded[model_name]["local_path"]
    # 直接返回用户输入（可能已经是合法HF ID）
    return model_name


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

    # 解析模型路径
    model_path = _resolve_model_path(request.model, request.model_path)

    # 检查模型名是否可能是有效的HF模型
    is_known = request.model in MODEL_PATH_MAPPING
    is_hf_id = "/" in model_path
    if not is_known and not is_hf_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_MODEL_NAME",
                    "message": (
                        f"模型 '{request.model}' 不在已知模型列表中，也不是有效的HuggingFace模型ID。"
                        f"请使用 'org/model_name' 格式指定HuggingFace模型，或先通过模型中心搜索和下载。"
                    ),
                }
            },
        )

    # 验证模型是否存在（在HF上），避免陷入等待死循环
    # 已知的内置映射模型跳过HF验证
    if not is_known and is_hf_id:
        try:
            validation = await validate_model(model_path)
            if not validation["valid"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": {
                            "code": "MODEL_NOT_FOUND",
                            "message": validation["error"] or f"模型 '{model_path}' 在HuggingFace上不存在",
                        }
                    },
                )
            if validation["gated"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": {
                            "code": "MODEL_GATED",
                            "message": f"模型 '{model_path}' 需要授权，请先在HuggingFace上申请权限并登录",
                        }
                    },
                )
        except HTTPException:
            raise
        except Exception:
            pass  # HF验证不可用时跳过，直接尝试加载

    engine_manager = get_engine_manager()

    try:
        success, message = await engine_manager.load_model(
            model_name=request.model,
            model_path=model_path,
            backend=InferenceBackendType(request.backend) if request.backend else InferenceBackendType.HUGGINGFACE,
            tensor_parallel=request.tensor_parallel or 1,
            gpu_memory_utilization=request.gpu_memory_utilization or 0.8,
            max_model_len=request.max_model_len or 2048,
            dtype=request.dtype or "auto",
            quantization=request.quantization,
        )

        if success:
            logger.info("model_load_success", model=request.model)
            return LoadModelResponse(
                model=request.model,
                status="loaded",
                message=message,
            )
        else:
            logger.error("model_load_failed", model=request.model, reason=message)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message or f"Failed to load model {request.model}",
            )

    except HTTPException:
        raise
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
    description="获取支持的模型列表（含已下载的模型）",
)
async def list_available_models() -> Dict:
    """获取可用模型列表"""
    downloaded = get_downloaded_models()
    return {
        "available_models": list(MODEL_PATH_MAPPING.keys()),
        "mappings": MODEL_PATH_MAPPING,
        "downloaded_models": [m["model_id"] for m in downloaded],
        "downloaded_count": len(downloaded),
    }
