"""模型管理路由"""

import time

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from quantumflow.api.models import (
    BenchmarkRequest,
    BenchmarkResponse,
    DeployRequest,
    DeployResponse,
    ModelInfo,
    UndeployRequest,
    UndeployResponse,
)
from quantumflow.inference import get_engine_manager
from quantumflow.models.registry import ModelRegistry, ModelStatus as RegistryModelStatus
from quantumflow.core.constants import InferenceBackendType

logger = structlog.get_logger().bind(component="api_models")

router = APIRouter(prefix="/models", tags=["Models"])

# 后端字符串到枚举的映射
BACKEND_STRING_TO_ENUM: dict[str, InferenceBackendType] = {
    "vllm": InferenceBackendType.VLLM,
    "huggingface": InferenceBackendType.HUGGINGFACE,
    "text-generation-inference": InferenceBackendType.TGI,
    "tgi": InferenceBackendType.TGI,
    "sglang": InferenceBackendType.SGLANG,
}

# 模型注册表
_registry = ModelRegistry()


def _model_info_from_registry(name: str, registry_info: "ModelInfo") -> ModelInfo:
    """将 ModelRegistry 的 ModelInfo 转换为 API 的 ModelInfo"""
    # 判断模型是否已加载
    engine_manager = get_engine_manager()
    is_loaded = engine_manager.is_model_loaded(name)

    return ModelInfo(
        model_id=name.lower().replace("/", "-"),
        name=name,
        architecture=registry_info.metadata.get("architecture", "Unknown"),
        parameter_count=registry_info.parameter_count,
        dtype="bfloat16",
        status="ready" if is_loaded else "available",
        replicas=1,
        tensor_parallel=registry_info.recommended_tensor_parallel,
        max_model_length=8192,
        backend=registry_info.backend,
        loaded_on_nodes=[],
    )


@router.get(
    "",
    response_model=list[ModelInfo],
    summary="列出模型",
    description="列出所有可用模型",
)
async def list_models(
    status_filter: str | None = Query(None, description="状态过滤"),
    backend: str | None = Query(None, description="后端过滤"),
) -> list[ModelInfo]:
    """列出所有可用模型"""
    # 从模型注册表获取所有可用模型
    all_models = _registry.list_models()

    models = []
    for model_info in all_models:
        api_model = _model_info_from_registry(model_info.name, model_info)

        if backend and api_model.backend != backend:
            continue
        if status_filter and api_model.status != status_filter:
            continue

        models.append(api_model)

    return models


@router.get(
    "/{model_name}",
    response_model=ModelInfo,
    summary="获取模型信息",
    description="获取指定模型的详细信息",
)
async def get_model(model_name: str) -> ModelInfo:
    """获取模型信息"""
    registry_info = _registry.get_model(model_name)

    if registry_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "MODEL_NOT_FOUND",
                    "message": f"Model not found: {model_name}",
                }
            },
        )

    return _model_info_from_registry(model_name, registry_info)


@router.post(
    "/deploy",
    response_model=DeployResponse,
    status_code=status.HTTP_201_CREATED,
    summary="部署模型",
    description="部署指定的模型到集群",
)
async def deploy_model(request: DeployRequest) -> DeployResponse:
    """部署模型"""
    model_id = f"{request.model.lower().replace('/', '-')}_{int(time.time())}"

    logger.info(
        "deploy_model_request",
        model_id=model_id,
        model=request.model,
        tensor_parallel=request.tensor_parallel,
        replicas=request.replicas,
    )

    # 解析后端
    backend = BACKEND_STRING_TO_ENUM.get(request.backend, InferenceBackendType.HUGGINGFACE)

    # 调用 EngineManager 部署模型
    engine_manager = get_engine_manager()
    success, message = await engine_manager.load_model(
        model_name=request.model,
        model_path=request.model,
        backend=backend,
        tensor_parallel=request.tensor_parallel or 1,
        gpu_memory_utilization=0.8,
        max_model_len=request.max_model_length or 8192,
        dtype=request.dtype or "auto",
        quantization=None,
    )

    if success:
        return DeployResponse(
            model_id=model_id,
            status="loading",
            replicas=request.replicas,
            message=f"Model {request.model} deployment started: {message}",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "DEPLOY_FAILED",
                    "message": f"Model {request.model} deployment failed: {message}",
                }
            },
        )


@router.post(
    "/undeploy",
    response_model=UndeployResponse,
    summary="卸载模型",
    description="从集群卸载指定的模型",
)
async def undeploy_model(request: UndeployRequest) -> UndeployResponse:
    """卸载模型"""
    logger.info("undeploy_model_request", model=request.model, force=request.force)

    # 调用 EngineManager 卸载模型
    engine_manager = get_engine_manager()
    success = await engine_manager.unload_model(request.model)

    if success:
        return UndeployResponse(
            model_id=request.model,
            status="unloaded",
            message=f"Model {request.model} unloaded successfully",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "MODEL_NOT_FOUND",
                    "message": f"Model not found: {request.model}",
                }
            },
        )


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="运行基准测试",
    description="对指定模型运行基准测试",
)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """运行基准测试"""
    benchmark_id = f"bench_{int(time.time() * 1000)}"

    logger.info(
        "benchmark_request",
        benchmark_id=benchmark_id,
        model=request.model,
        test_set=request.test_set,
    )

    # 验证模型是否已加载
    engine_manager = get_engine_manager()
    loaded_models = engine_manager.get_loaded_models()
    if request.model not in loaded_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "MODEL_NOT_LOADED",
                    "message": f"Model {request.model} is not loaded. Please load it first.",
                }
            },
        )

    # 启动基准测试后台任务
    import asyncio
    asyncio.create_task(_run_benchmark_task(benchmark_id, request.model, request.test_set, request.num_samples))

    return BenchmarkResponse(
        benchmark_id=benchmark_id,
        model=request.model,
        test_set=request.test_set,
        status="running",
        total_samples=request.num_samples,
        completed_samples=0,
    )


async def _run_benchmark_task(benchmark_id: str, model: str, test_set: str, num_samples: int):
    """后台执行基准测试任务"""
    import time

    from quantumflow.inference.engine import InferenceResult, SamplingParams

    engine_manager = get_engine_manager()

    # 根据 test_set 选择默认提示词
    default_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "In a distant galaxy, the stars twinkled brightly against the dark canvas of space.",
        "The art of programming lies in the ability to express complex ideas in simple terms.",
        "Climate change poses significant challenges to global ecosystems and human societies.",
    ]

    # 生成测试提示词（循环使用直到达到 num_samples）
    prompts = (default_prompts * ((num_samples // len(default_prompts)) + 1))[:num_samples]

    # 默认采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=256,
        repetition_penalty=1.0,
    )

    # 执行基准测试
    total_tokens = 0
    total_latency_ms = 0.0
    successful_samples = 0
    start_time = time.time()

    try:
        results: list[InferenceResult] = await engine_manager.generate(
            model_name=model,
            prompts=prompts,
            sampling_params=sampling_params,
        )

        # 计算统计信息
        for result in results:
            total_latency_ms += result.latency_ms
            total_tokens += result.prompt_tokens + result.completion_tokens
            successful_samples += 1

        elapsed_time = time.time() - start_time

        # 计算指标
        avg_latency_ms = total_latency_ms / successful_samples if successful_samples > 0 else 0
        tokens_per_second = (total_tokens / elapsed_time) if elapsed_time > 0 else 0

        logger.info(
            "benchmark_completed",
            benchmark_id=benchmark_id,
            model=model,
            samples=successful_samples,
            total_samples=num_samples,
            avg_latency_ms=round(avg_latency_ms, 2),
            tokens_per_second=round(tokens_per_second, 2),
            elapsed_seconds=round(elapsed_time, 2),
        )

    except Exception as e:
        logger.error(
            "benchmark_failed",
            benchmark_id=benchmark_id,
            model=model,
            error=str(e),
        )
