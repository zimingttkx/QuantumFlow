"""推理路由"""

from typing import AsyncIterator
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import asyncio
import time
import structlog

from quantumflow.api.models import (
    InferenceRequest,
    InferenceResponse,
    StreamResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    ChatRequest,
)
from quantumflow.core.exceptions import (
    InferenceError,
    ModelNotFoundError,
    SchedulerError,
)
from quantumflow.inference import get_engine_manager, SamplingParams

logger = structlog.get_logger().bind(component="api_inference")

router = APIRouter(prefix="/inference", tags=["Inference"])

# 模拟请求ID生成
_request_counter = 0


async def _ensure_model_loaded(model_name: str, start_time: float) -> tuple[bool, str]:
    """确保模型已加载。不在推理路径中自动加载模型（会阻塞事件循环），只返回状态。"""
    engine_manager = get_engine_manager()

    if engine_manager.is_model_loaded(model_name):
        return True, ""

    # 模型未加载 — 不自动加载，返回明确错误让用户先通过 /models/load 加载
    return False, (
        f"模型 '{model_name}' 尚未加载。请先通过 'python -m quantumflow.cli load {model_name}' "
        f"或在前端模型管理页面加载该模型。"
    )


def _generate_request_id() -> str:
    """生成请求ID"""
    global _request_counter
    _request_counter += 1
    return f"req_{_request_counter:08d}"


def _convert_sampling_params(request: InferenceRequest) -> SamplingParams:
    """转换采样参数"""
    if request.sampling_params is None:
        return SamplingParams()
    if isinstance(request.sampling_params, dict):
        return SamplingParams(**request.sampling_params)
    return request.sampling_params


@router.post(
    "/generate",
    response_model=InferenceResponse,
    summary="文本生成",
    description="使用指定模型生成文本",
)
async def generate(request: InferenceRequest) -> InferenceResponse:
    """文本生成接口"""
    start_time = time.time()

    # 生成请求ID
    request_id = request.request_id or _generate_request_id()

    logger.info(
        "generate_request",
        request_id=request_id,
        model=request.model,
        prompt_length=len(request.prompt),
        stream=request.stream,
    )

    try:
        engine_manager = get_engine_manager()

        # 确保模型已加载（含HF验证，避免死循环等待）
        if not engine_manager.is_model_loaded(request.model):
            ok, err = await _ensure_model_loaded(request.model, start_time)
            if not ok:
                return InferenceResponse(
                    request_id=request_id,
                    model=request.model,
                    prompt=request.prompt,
                    generated_text=f"[模型加载失败] {err}\n\n用户输入: {request.prompt}",
                    finish_reason="error",
                    latency_ms=(time.time() - start_time) * 1000,
                    usage={
                        "prompt_tokens": len(request.prompt) // 4,
                        "completion_tokens": 0,
                        "total_tokens": len(request.prompt) // 4,
                    },
                )

        # 执行推理
        sampling_params = _convert_sampling_params(request)
        results = await engine_manager.generate(
            model_name=request.model,
            prompts=[request.prompt],
            sampling_params=sampling_params,
        )

        latency_ms = (time.time() - start_time) * 1000

        if results and len(results) > 0:
            result = results[0]
            return InferenceResponse(
                request_id=request_id,
                model=request.model,
                prompt=request.prompt,
                generated_text=result.outputs[0] if result.outputs else "",
                finish_reason=result.finish_reason,
                latency_ms=result.latency_ms,
                usage={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            )

        # 如果没有结果，返回错误
        raise InferenceError("Generation produced no results")

    except ModelNotFoundError as e:
        logger.error("model_not_found", request_id=request_id, model=request.model)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.to_dict(),
        )
    except SchedulerError as e:
        logger.error("scheduler_error", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.to_dict(),
        )
    except Exception as e:
        logger.error("generate_error", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "INFERENCE_ERROR", "message": str(e)}},
        )


@router.post(
    "/generate/stream",
    summary="流式文本生成",
    description="使用指定模型流式生成文本",
)
async def generate_stream(request: InferenceRequest) -> StreamingResponse:
    """流式文本生成接口"""

    request_id = request.request_id or _generate_request_id()

    logger.info(
        "generate_stream_request",
        request_id=request_id,
        model=request.model,
    )

    async def event_generator() -> AsyncIterator[str]:
        """生成SSE事件流"""
        engine_manager = get_engine_manager()

        # 确保模型已加载
        if not engine_manager.is_model_loaded(request.model):
            ok, err = await _ensure_model_loaded(request.model, time.time())
            if not ok:
                error_response = StreamResponse(
                    request_id=request_id,
                    delta=f"模型加载失败: {err}",
                    is_final=True,
                    finish_reason="error",
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

        sampling_params = _convert_sampling_params(request)

        try:
            full_text = ""
            async for text_chunk in engine_manager.generate_stream(
                request.model, request.prompt, sampling_params
            ):
                full_text += text_chunk
                response = StreamResponse(
                    request_id=request_id,
                    delta=text_chunk,
                    is_final=False,
                )
                yield f"data: {response.model_dump_json()}\n\n"

            # 发送最终结果
            final_response = StreamResponse(
                request_id=request_id,
                delta="",
                is_final=True,
                finish_reason="stop",
                usage={
                    "prompt_tokens": len(request.prompt) // 4,
                    "completion_tokens": len(full_text) // 4,
                    "total_tokens": len(request.prompt) // 4 + len(full_text) // 4,
                },
            )
            yield f"data: {final_response.model_dump_json()}\n\n"

        except Exception as e:
            logger.error("stream_generate_error", error=str(e))
            error_response = StreamResponse(
                request_id=request_id,
                delta=f"生成错误: {str(e)}",
                is_final=True,
                finish_reason="error",
            )
            yield f"data: {error_response.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


@router.post(
    "/batch",
    response_model=BatchInferenceResponse,
    summary="批量推理",
    description="对多个提示词批量生成文本",
)
async def batch_generate(request: BatchInferenceRequest) -> BatchInferenceResponse:
    """批量推理接口"""
    batch_id = f"batch_{int(time.time() * 1000)}"

    logger.info(
        "batch_generate_request",
        batch_id=batch_id,
        model=request.model,
        prompt_count=len(request.prompts),
    )

    engine_manager = get_engine_manager()

    # 确保模型已加载
    if not engine_manager.is_model_loaded(request.model):
        ok, err = await _ensure_model_loaded(request.model, time.time())
        if not ok:
            mock_results = []
            for i, prompt in enumerate(request.prompts):
                mock_results.append(
                    InferenceResponse(
                        request_id=f"{batch_id}_{i}",
                        model=request.model,
                        prompt=prompt,
                        generated_text=f"[模型加载失败] {err}\n\n用户输入: {prompt}",
                        finish_reason="error",
                        latency_ms=0,
                        usage={
                            "prompt_tokens": len(prompt) // 4,
                            "completion_tokens": 0,
                            "total_tokens": len(prompt) // 4,
                        },
                    )
                )
        return BatchInferenceResponse(
            batch_id=batch_id,
            model=request.model,
            total=len(request.prompts),
            completed=len(mock_results),
            failed=0,
            results=mock_results,
            total_latency_ms=0,
            avg_latency_ms=0,
        )

    # 从请求中提取采样参数
    req_sampling = request.sampling_params
    sampling_params = SamplingParams(
        temperature=req_sampling.temperature if req_sampling else 0.7,
        top_p=req_sampling.top_p if req_sampling else 0.9,
        top_k=req_sampling.top_k if req_sampling else 50,
        max_tokens=req_sampling.max_tokens if req_sampling else 500,
        repetition_penalty=req_sampling.repetition_penalty if req_sampling else 1.0,
        stop=req_sampling.stop if req_sampling else None,
    )

    results = []
    total_latency = 0

    try:
        # 调用批量推理
        inference_results = await engine_manager.generate(
            model_name=request.model,
            prompts=request.prompts,
            sampling_params=sampling_params,
        )

        for i, result in enumerate(inference_results):
            total_latency += result.latency_ms
            results.append(
                InferenceResponse(
                    request_id=f"{batch_id}_{i}",
                    model=request.model,
                    prompt=request.prompts[i],
                    generated_text=result.outputs[0] if result.outputs else "",
                    finish_reason=result.finish_reason,
                    latency_ms=result.latency_ms,
                    usage={
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.prompt_tokens + result.completion_tokens,
                    },
                )
            )

    except Exception as e:
        logger.error("batch_generate_error", error=str(e))
        # 如果出错，返回错误结果
        for i, prompt in enumerate(request.prompts):
            results.append(
                InferenceResponse(
                    request_id=f"{batch_id}_{i}",
                    model=request.model,
                    prompt=prompt,
                    generated_text=f"[错误: {str(e)}]",
                    finish_reason="error",
                    latency_ms=0,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                )
            )

    return BatchInferenceResponse(
        batch_id=batch_id,
        model=request.model,
        total=len(request.prompts),
        completed=len(results),
        failed=len(request.prompts) - len(results),
        results=results,
        total_latency_ms=total_latency,
        avg_latency_ms=total_latency / len(results) if results else 0,
    )


@router.post(
    "/chat",
    response_model=InferenceResponse,
    summary="对话",
    description="使用指定模型进行对话",
)
async def chat(request: ChatRequest) -> InferenceResponse:
    """对话接口"""
    # 使用ChatML格式构建prompt
    prompt_parts = []
    for msg in request.messages:
        role = msg.role.lower()
        if role not in ("user", "assistant", "system"):
            role = "user"
        prompt_parts.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")

    prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"

    # 调用生成接口
    return await generate(
        InferenceRequest(
            model=request.model,
            prompt=prompt,
            sampling_params=request.sampling_params,
            stream=False,
            session_id=request.session_id,
            priority=request.priority,
        )
    )
