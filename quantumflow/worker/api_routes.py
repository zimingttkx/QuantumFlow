"""Worker API 路由 - 接收 Controller 的指令"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from quantumflow.inference.engine import ModelConfig, SamplingParams
from quantumflow.worker.worker import (
    InferenceRequest,
    InferenceResponse,
    LoadModelRequest,
    UnloadModelRequest,
    WorkerNode,
)


def create_worker_router(worker: WorkerNode) -> APIRouter:
    """创建 Worker API 路由"""
    router = APIRouter()

    # ==================== 模型管理 ====================

    @router.post("/load_model")
    async def load_model(req: LoadModelRequest) -> dict[str, Any]:
        """
        加载模型

        Controller 通过此接口指示 Worker 加载模型
        """
        try:
            config = ModelConfig(
                model_name=req.model_name,
                model_path=req.model_path or req.model_name,
                tensor_parallel=req.tensor_parallel,
                gpu_memory_utilization=req.gpu_memory_utilization,
                enable_chunked_prefill=req.enable_chunked_prefill,
                prefill_chunk_size=req.prefill_chunk_size,
            )

            success = await worker.load_model(config)

            if success:
                return {
                    "status": "success",
                    "model": req.model_name,
                    "message": f"Model {req.model_name} loaded successfully",
                }
            else:
                return {
                    "status": "error",
                    "model": req.model_name,
                    "message": f"Failed to load model {req.model_name}",
                }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/unload_model")
    async def unload_model(req: UnloadModelRequest) -> dict[str, Any]:
        """
        卸载模型

        Controller 通过此接口指示 Worker 卸载模型
        """
        try:
            success = await worker.unload_model(req.model_name)

            if success:
                return {
                    "status": "success",
                    "model": req.model_name,
                    "message": f"Model {req.model_name} unloaded successfully",
                }
            else:
                return {
                    "status": "error",
                    "model": req.model_name,
                    "message": f"Failed to unload model {req.model_name}",
                }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ==================== 推理请求 ====================

    @router.post("/inference", response_model=InferenceResponse)
    async def inference(req: InferenceRequest) -> InferenceResponse:
        """
        执行推理

        Controller 通过此接口指示 Worker 执行推理
        """
        try:
            # 构建采样参数
            if req.sampling_params:
                sampling_params = SamplingParams(
                    temperature=req.sampling_params.get("temperature", 0.7),
                    top_p=req.sampling_params.get("top_p", 0.9),
                    top_k=req.sampling_params.get("top_k", 50),
                    max_tokens=req.sampling_params.get("max_tokens", 2048),
                    repetition_penalty=req.sampling_params.get("repetition_penalty", 1.0),
                    stop=req.sampling_params.get("stop"),
                )
            else:
                sampling_params = SamplingParams()

            # 执行推理
            result = await worker.inference(
                request_id=req.request_id,
                model_name=req.model_name,
                prompts=req.prompts,
                sampling_params=sampling_params,
            )

            return InferenceResponse(
                request_id=result["request_id"],
                status=result["status"],
                results=result.get("results"),
                error=result.get("error"),
                latency_ms=result["latency_ms"],
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ==================== 状态查询 ====================

    @router.get("/status")
    async def get_status() -> dict[str, Any]:
        """
        获取 Worker 状态

        返回 Worker 的当前状态信息
        """
        return {
            "node_id": worker.config.node_id,
            "status": worker.status.value,
            "hostname": worker.node_info.get("hostname"),
            "ip": worker.node_info.get("ip"),
            "port": worker.config.port,
            "gpu_count": len(worker.node_info.get("gpu_info", [])),
            "gpu_info": worker.node_info.get("gpu_info", []),
            "loaded_models": worker.engine.loaded_model_names if worker.engine else [],
            "active_requests": len(worker.active_requests),
            "completed_requests": worker.completed_requests,
            "failed_requests": worker.failed_requests,
            "started_at": worker.started_at.isoformat() if worker.started_at else None,
        }

    @router.get("/stats")
    async def get_stats(model_name: Optional[str] = Query(None)) -> dict[str, Any]:
        """
        获取模型统计信息

        Args:
            model_name: 模型名称（可选，不指定则返回所有模型的统计）
        """
        try:
            if model_name:
                stats = await worker.get_stats(model_name)
                return {
                    "model": model_name,
                    "stats": stats,
                }
            else:
                # Return aggregate stats for all loaded models
                return {
                    "stats": {
                        "active_requests": len(worker.active_requests),
                        "completed_requests": worker.completed_requests,
                        "failed_requests": worker.failed_requests,
                    },
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/node_info")
    async def get_node_info() -> dict[str, Any]:
        """
        获取节点完整信息

        返回 Worker 节点的完整信息，包含 GPU、内存等资源
        """
        return worker.node_info

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        """
        列出已加载模型
        """
        return {
            "models": worker.engine.loaded_model_names if worker.engine else [],
        }

    return router
