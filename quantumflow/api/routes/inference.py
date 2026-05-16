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

logger = structlog.get_logger().bind(component="api_inference")

router = APIRouter(prefix="/inference", tags=["Inference"])

# 模拟请求ID生成
_request_counter = 0


def _generate_request_id() -> str:
    """生成请求ID"""
    global _request_counter
    _request_counter += 1
    return f"req_{_request_counter:08d}"


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
        # TODO: 调用调度器进行推理
        # 这里暂时返回模拟数据
        await asyncio.sleep(0.1)  # 模拟延迟

        latency_ms = (time.time() - start_time) * 1000

        return InferenceResponse(
            request_id=request_id,
            model=request.model,
            prompt=request.prompt,
            generated_text="这是模拟生成的回复。在实际实现中，这里会调用QuantumFlow的推理引擎来生成文本。",
            finish_reason="stop",
            latency_ms=latency_ms,
            usage={
                "prompt_tokens": len(request.prompt) // 4,  # 粗略估算
                "completion_tokens": 100,
                "total_tokens": len(request.prompt) // 4 + 100,
            },
        )

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
        # TODO: 调用调度器进行流式推理
        # 这里暂时发送模拟数据

        sample_text = "这是模拟的流式回复。每个片段都会单独发送。"
        words = sample_text.split()

        for i, word in enumerate(words):
            # 模拟逐词生成
            await asyncio.sleep(0.1)

            is_final = i == len(words) - 1

            response = StreamResponse(
                request_id=request_id,
                delta=word + " ",
                is_final=is_final,
            )

            if is_final:
                response.usage = {
                    "prompt_tokens": 50,
                    "completion_tokens": len(words),
                    "total_tokens": 50 + len(words),
                }
                response.finish_reason = "stop"

            # SSE格式
            yield f"data: {response.model_dump_json()}\n\n"

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

    # TODO: 调用调度器进行批量推理
    results = []

    for i, prompt in enumerate(request.prompts):
        await asyncio.sleep(0.05)  # 模拟延迟

        results.append(
            InferenceResponse(
                request_id=f"{batch_id}_{i}",
                model=request.model,
                prompt=prompt,
                generated_text=f"批量回复 {i + 1}",
                finish_reason="stop",
                latency_ms=50,
                usage={
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": 50,
                    "total_tokens": len(prompt) // 4 + 50,
                },
            )
        )

    total_latency = sum(r.latency_ms for r in results)

    return BatchInferenceResponse(
        batch_id=batch_id,
        model=request.model,
        total=len(request.prompts),
        completed=len(results),
        failed=0,
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
    # 将对话历史转换为单个提示词
    prompt_parts = []
    for msg in request.messages:
        role = msg.role.upper()
        prompt_parts.append(f"{role}: {msg.content}")

    prompt = "\n\n".join(prompt_parts)
    prompt += "\n\nASSISTANT:"

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
