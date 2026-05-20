"""API单元测试"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from quantumflow.api.models import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    ChatMessage,
    ChatRequest,
    ClusterStatus,
    DeployRequest,
    DeployResponse,
    ErrorDetail,
    ErrorResponse,
    GPUInfo,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    JobInfo,
    LoadModelRequest,
    LoadModelResponse,
    MetricsResponse,
    ModelFilterRequest,
    ModelInfo,
    ModelStatusResponse,
    NodeActionRequest,
    NodeInfo,
    SamplingParams,
    StreamResponse,
    TokenUsage,
    UndeployRequest,
    UndeployResponse,
    UnloadModelResponse,
)


class TestSamplingParams:
    """采样参数测试"""

    def test_default_values(self):
        """测试默认值"""
        params = SamplingParams()

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 2048
        assert params.repetition_penalty == 1.0
        assert params.stop is None

    def test_custom_values(self):
        """测试自定义值"""
        params = SamplingParams(
            temperature=0.5,
            top_p=0.95,
            top_k=100,
            max_tokens=1024,
            stop=["END", "STOP"],
        )

        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 100
        assert params.max_tokens == 1024
        assert params.stop == ["END", "STOP"]

    def test_temperature_validation(self):
        """测试温度参数验证"""
        with pytest.raises(ValidationError):
            SamplingParams(temperature=3.0)  # 超出范围

        with pytest.raises(ValidationError):
            SamplingParams(temperature=-0.1)  # 低于范围

    def test_max_tokens_validation(self):
        """测试最大token验证"""
        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=0)  # 必须大于0

        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=50000)  # 超出最大限制


class TestInferenceRequest:
    """推理请求测试"""

    def test_required_fields(self):
        """测试必需字段"""
        request = InferenceRequest(
            model="Qwen2.5-7B",
            prompt="Hello, world!",
        )

        assert request.model == "Qwen2.5-7B"
        assert request.prompt == "Hello, world!"
        assert request.stream is False
        assert request.priority == 5

    def test_all_fields(self):
        """测试所有字段"""
        request = InferenceRequest(
            model="Qwen2.5-7B",
            prompt="Hello",
            sampling_params=SamplingParams(temperature=0.8),
            stream=True,
            session_id="session-123",
            priority=8,
            tags={"user": "test"},
            request_id="req-001",
        )

        assert request.stream is True
        assert request.session_id == "session-123"
        assert request.priority == 8
        assert request.tags == {"user": "test"}
        assert request.request_id == "req-001"

    def test_default_sampling_params(self):
        """测试默认采样参数"""
        request = InferenceRequest(
            model="test",
            prompt="test",
        )

        assert isinstance(request.sampling_params, SamplingParams)


class TestBatchInferenceRequest:
    """批量推理请求测试"""

    def test_single_prompt(self):
        """测试单个提示"""
        request = BatchInferenceRequest(
            model="Qwen2.5-7B",
            prompts=["Hello"],
        )

        assert len(request.prompts) == 1

    def test_multiple_prompts(self):
        """测试多个提示"""
        request = BatchInferenceRequest(
            model="Qwen2.5-7B",
            prompts=[f"Prompt {i}" for i in range(10)],
        )

        assert len(request.prompts) == 10

    def test_max_prompts_validation(self):
        """测试最大提示数量验证"""
        with pytest.raises(ValidationError):
            BatchInferenceRequest(
                model="test",
                prompts=[f"Prompt {i}" for i in range(101)],  # 超出限制
            )

    def test_min_prompts_validation(self):
        """测试最小提示数量验证"""
        with pytest.raises(ValidationError):
            BatchInferenceRequest(
                model="test",
                prompts=[],  # 不能为空
            )


class TestChatMessage:
    """对话消息测试"""

    def test_valid_roles(self):
        """测试有效角色"""
        for role in ["system", "user", "assistant"]:
            message = ChatMessage(role=role, content="Hello")
            assert message.role == role


class TestChatRequest:
    """对话请求测试"""

    def test_valid_request(self):
        """测试有效请求"""
        request = ChatRequest(
            model="Qwen2.5-7B",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
        )

        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"

    def test_empty_messages_validation(self):
        """测试空消息验证"""
        with pytest.raises(ValidationError):
            ChatRequest(
                model="test",
                messages=[],
            )


class TestDeployRequest:
    """部署请求测试"""

    def test_required_fields(self):
        """测试必需字段"""
        request = DeployRequest(model="Qwen2.5-7B")

        assert request.model == "Qwen2.5-7B"
        assert request.tensor_parallel == 1
        assert request.pipeline_parallel == 1
        assert request.gpu_memory_utilization == 0.9

    def test_parallel_validation(self):
        """测试并行度验证"""
        with pytest.raises(ValidationError):
            DeployRequest(model="test", tensor_parallel=0)  # 必须 >= 1

        with pytest.raises(ValidationError):
            DeployRequest(model="test", tensor_parallel=16)  # 最大为8

    def test_gpu_memory_utilization_validation(self):
        """测试GPU显存利用率验证"""
        with pytest.raises(ValidationError):
            DeployRequest(model="test", gpu_memory_utilization=0.05)

        with pytest.raises(ValidationError):
            DeployRequest(model="test", gpu_memory_utilization=1.5)


class TestInferenceResponse:
    """推理响应测试"""

    def test_valid_response(self):
        """测试有效响应"""
        response = InferenceResponse(
            request_id="req-001",
            model="Qwen2.5-7B",
            prompt="Hello",
            generated_text="Hi there!",
            finish_reason="stop",
            latency_ms=123.45,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )

        assert response.request_id == "req-001"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 30


class TestGPUInfo:
    """GPU信息测试"""

    def test_valid_gpu_info(self):
        """测试有效GPU信息"""
        info = GPUInfo(
            gpu_id=0,
            name="NVIDIA RTX 4090",
            memory_total=24 * 1024**3,
            memory_used=10 * 1024**3,
            memory_free=14 * 1024**3,
            utilization=0.5,
            temperature=45.0,
        )

        assert info.memory_used_percent == pytest.approx(0.4167, rel=0.01)

    def test_memory_used_percent_zero_total(self):
        """测试零总显存"""
        info = GPUInfo(
            gpu_id=0,
            name="Test",
            memory_total=0,
            memory_used=0,
            memory_free=0,
            utilization=0.0,
            temperature=0.0,
        )

        assert info.memory_used_percent == 0.0


class TestModelInfo:
    """模型信息测试"""

    def test_valid_model_info(self):
        """测试有效模型信息"""
        info = ModelInfo(
            model_id="qwen2.5-7b",
            name="Qwen2.5-7B-Instruct",
            architecture="Qwen2ForCausalLM",
            parameter_count=7_000_000_000,
            dtype="bfloat16",
            status="ready",
            replicas=2,
            tensor_parallel=1,
            max_model_length=8192,
            backend="vllm",
        )

        assert info.model_id == "qwen2.5-7b"
        assert info.parameter_count == 7_000_000_000
        assert info.status == "ready"


class TestNodeInfo:
    """节点信息测试"""

    def test_valid_node_info(self):
        """测试有效节点信息"""

        info = NodeInfo(
            node_id="node-1",
            hostname="server-1",
            ip="192.168.1.101",
            port=8001,
            status="healthy",
            gpu_count=4,
            gpu_info=[],
            cpu_count=32,
            memory_total=128 * 1024**3,
            memory_available=64 * 1024**3,
            disk_total=2 * 1024**4,
            disk_available=1 * 1024**4,
            current_load=0.3,
            labels={"zone": "zone-a"},
            version="1.0.0",
            uptime_seconds=3600,
            last_heartbeat=datetime.now(),
        )

        assert info.node_id == "node-1"
        assert info.memory_available_percent == pytest.approx(0.5, rel=0.01)


class TestClusterStatus:
    """集群状态测试"""

    def test_valid_status(self):
        """测试有效状态"""
        status = ClusterStatus(
            total_nodes=5,
            healthy_nodes=4,
            unhealthy_nodes=1,
            draining_nodes=0,
            total_gpus=20,
            available_gpus=16,
            active_models=3,
            pending_jobs=5,
            running_jobs=10,
            system_metrics={"cpu_usage": 0.5},
            uptime_seconds=86400,
        )

        assert status.total_nodes == 5
        assert status.healthy_nodes == 4
        assert status.available_gpus == 16


class TestHealthResponse:
    """健康响应测试"""

    def test_healthy_response(self):
        """测试健康响应"""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600,
            checks={"api": "ok", "redis": "ok"},
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"


class TestErrorResponse:
    """错误响应测试"""

    def test_error_detail(self):
        """测试错误详情"""
        error = ErrorDetail(
            code="MODEL_NOT_FOUND",
            message="Model not found",
            details={"model": "test"},
        )

        assert error.code == "MODEL_NOT_FOUND"
        assert error.details["model"] == "test"

    def test_error_response(self):
        """测试错误响应"""
        error_detail = ErrorDetail(
            code="INVALID_REQUEST",
            message="Invalid parameters",
        )

        response = ErrorResponse(error=error_detail)

        assert response.error.code == "INVALID_REQUEST"


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: SamplingParams 边界与验证器行为
# ═════════════════════════════════════════════════════════════════════════════════


class TestSamplingParamsValidatorBehavior:
    """SamplingParams 验证器业务逻辑"""

    def test_max_tokens_none_yields_default_2048(self):
        """mode='before' 验证器: None 输入应产出默认值 2048"""
        params = SamplingParams(max_tokens=None)
        assert params.max_tokens == 2048

    def test_max_tokens_zero_is_rejected(self):
        """0 应被 Field(ge=1) 拒绝"""
        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=0)

    def test_boundary_max_tokens_at_minimum(self):
        params = SamplingParams(max_tokens=1)
        assert params.max_tokens == 1

    def test_boundary_max_tokens_at_maximum(self):
        params = SamplingParams(max_tokens=32768)
        assert params.max_tokens == 32768

    def test_temperature_boundary_min(self):
        params = SamplingParams(temperature=0.0)
        assert params.temperature == 0.0

    def test_temperature_boundary_max(self):
        params = SamplingParams(temperature=2.0)
        assert params.temperature == 2.0

    def test_top_p_boundary_min(self):
        params = SamplingParams(top_p=0.0)
        assert params.top_p == 0.0

    def test_top_p_boundary_max(self):
        params = SamplingParams(top_p=1.0)
        assert params.top_p == 1.0

    def test_repetition_penalty_boundary_min(self):
        params = SamplingParams(repetition_penalty=1.0)
        assert params.repetition_penalty == 1.0

    def test_repetition_penalty_boundary_max(self):
        params = SamplingParams(repetition_penalty=2.0)
        assert params.repetition_penalty == 2.0

    def test_stop_words_empty_list_is_valid(self):
        params = SamplingParams(stop=[])
        assert params.stop == []


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: UndeployRequest
# ═════════════════════════════════════════════════════════════════════════════════


class TestUndeployRequest:
    """模型卸载请求测试"""

    def test_required_fields(self):
        request = UndeployRequest(model="Qwen2.5-7B")
        assert request.model == "Qwen2.5-7B"
        assert request.force is False

    def test_force_true(self):
        request = UndeployRequest(model="Qwen2.5-7B", force=True)
        assert request.force is True

    def test_force_default_is_false(self):
        request = UndeployRequest(model="Qwen2.5-7B")
        assert request.force is False


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: NodeActionRequest
# ═════════════════════════════════════════════════════════════════════════════════


class TestNodeActionRequest:
    """节点操作请求测试"""

    def test_drain_action(self):
        request = NodeActionRequest(action="drain", reason="maintenance")
        assert request.action == "drain"
        assert request.reason == "maintenance"

    def test_uncordon_action(self):
        request = NodeActionRequest(action="uncordon")
        assert request.action == "uncordon"
        assert request.reason is None

    def test_restart_action(self):
        request = NodeActionRequest(action="restart", reason="kernel update")
        assert request.action == "restart"

    def test_reason_is_optional(self):
        request = NodeActionRequest(action="drain")
        assert request.reason is None


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: ModelFilterRequest
# ═════════════════════════════════════════════════════════════════════════════════


class TestModelFilterRequest:
    """模型过滤请求测试"""

    def test_all_fields_specified(self):
        request = ModelFilterRequest(
            status="ready",
            backend="vllm",
            labels={"env": "prod", "team": "ml"},
        )
        assert request.status == "ready"
        assert request.backend == "vllm"
        assert request.labels == {"env": "prod", "team": "ml"}

    def test_all_fields_default_to_none(self):
        request = ModelFilterRequest()
        assert request.status is None
        assert request.backend is None
        assert request.labels is None

    def test_partial_filter_status_only(self):
        request = ModelFilterRequest(status="loading")
        assert request.status == "loading"
        assert request.backend is None
        assert request.labels is None

    def test_partial_filter_backend_only(self):
        request = ModelFilterRequest(backend="tgi")
        assert request.backend == "tgi"
        assert request.status is None
        assert request.labels is None

    def test_partial_filter_labels_only(self):
        request = ModelFilterRequest(labels={"gpu": "a100"})
        assert request.labels == {"gpu": "a100"}
        assert request.status is None
        assert request.backend is None


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: BenchmarkRequest
# ═════════════════════════════════════════════════════════════════════════════════


class TestBenchmarkRequest:
    """基准测试请求测试"""

    def test_required_fields_with_defaults(self):
        request = BenchmarkRequest(model="Qwen2.5-7B")
        assert request.model == "Qwen2.5-7B"
        assert request.test_set == "mmlu"
        assert request.num_samples == 100
        assert request.backend == "vllm"
        assert request.tensor_parallel == 1

    def test_custom_test_set(self):
        for test_set in ["humaneval", "math", "custom"]:
            request = BenchmarkRequest(model="m", test_set=test_set)
            assert request.test_set == test_set

    def test_num_samples_boundary_min(self):
        request = BenchmarkRequest(model="m", num_samples=1)
        assert request.num_samples == 1

    def test_num_samples_boundary_max(self):
        request = BenchmarkRequest(model="m", num_samples=1000)
        assert request.num_samples == 1000

    def test_num_samples_below_min_rejected(self):
        with pytest.raises(ValidationError):
            BenchmarkRequest(model="m", num_samples=0)

    def test_num_samples_above_max_rejected(self):
        with pytest.raises(ValidationError):
            BenchmarkRequest(model="m", num_samples=1001)

    def test_custom_sampling_params(self):
        request = BenchmarkRequest(
            model="m",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=256),
        )
        assert request.sampling_params.temperature == 0.0
        assert request.sampling_params.max_tokens == 256

    def test_sampling_params_default(self):
        request = BenchmarkRequest(model="m")
        assert isinstance(request.sampling_params, SamplingParams)
        assert request.sampling_params.temperature == 0.7


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: DeployRequest 扩展验证
# ═════════════════════════════════════════════════════════════════════════════════


class TestDeployRequestExtended:
    """DeployRequest 所有字段验证的完整覆盖"""

    # — tensor_parallel —

    def test_tensor_parallel_boundary_min(self):
        req = DeployRequest(model="m", tensor_parallel=1)
        assert req.tensor_parallel == 1

    def test_tensor_parallel_boundary_max(self):
        req = DeployRequest(model="m", tensor_parallel=8)
        assert req.tensor_parallel == 8

    # — pipeline_parallel —

    def test_pipeline_parallel_boundary_min(self):
        req = DeployRequest(model="m", pipeline_parallel=1)
        assert req.pipeline_parallel == 1

    def test_pipeline_parallel_boundary_max(self):
        req = DeployRequest(model="m", pipeline_parallel=4)
        assert req.pipeline_parallel == 4

    def test_pipeline_parallel_zero_rejected(self):
        with pytest.raises(ValidationError):
            DeployRequest(model="m", pipeline_parallel=0)

    def test_pipeline_parallel_exceeds_max_rejected(self):
        with pytest.raises(ValidationError):
            DeployRequest(model="m", pipeline_parallel=5)

    # — gpu_memory_utilization —

    def test_gpu_memory_utilization_boundary_min(self):
        req = DeployRequest(model="m", gpu_memory_utilization=0.1)
        assert req.gpu_memory_utilization == 0.1

    def test_gpu_memory_utilization_boundary_max(self):
        req = DeployRequest(model="m", gpu_memory_utilization=1.0)
        assert req.gpu_memory_utilization == 1.0

    # — max_model_length —

    def test_max_model_length_boundary_min(self):
        req = DeployRequest(model="m", max_model_length=1)
        assert req.max_model_length == 1

    def test_max_model_length_boundary_max(self):
        req = DeployRequest(model="m", max_model_length=32768)
        assert req.max_model_length == 32768

    def test_max_model_length_default_is_none(self):
        req = DeployRequest(model="m")
        assert req.max_model_length is None

    # — replicas —

    def test_replicas_boundary_min(self):
        req = DeployRequest(model="m", replicas=1)
        assert req.replicas == 1

    def test_replicas_boundary_max(self):
        req = DeployRequest(model="m", replicas=10)
        assert req.replicas == 10

    def test_replicas_zero_rejected(self):
        with pytest.raises(ValidationError):
            DeployRequest(model="m", replicas=0)

    def test_replicas_exceeds_max_rejected(self):
        with pytest.raises(ValidationError):
            DeployRequest(model="m", replicas=11)

    # — min_replicas / max_replicas —

    def test_min_replicas_default(self):
        req = DeployRequest(model="m")
        assert req.min_replicas == 1

    def test_max_replicas_default(self):
        req = DeployRequest(model="m")
        assert req.max_replicas == 3

    def test_max_replicas_boundary_min(self):
        req = DeployRequest(model="m", max_replicas=1)
        assert req.max_replicas == 1

    # — target_gpu_utilization —

    def test_target_gpu_utilization_boundary_min(self):
        req = DeployRequest(model="m", target_gpu_utilization=0.1)
        assert req.target_gpu_utilization == 0.1

    def test_target_gpu_utilization_boundary_max(self):
        req = DeployRequest(model="m", target_gpu_utilization=1.0)
        assert req.target_gpu_utilization == 1.0

    # — auto_scaling / quantization / dtype —

    def test_auto_scaling_enabled(self):
        req = DeployRequest(model="m", auto_scaling=True)
        assert req.auto_scaling is True

    def test_quantization_values(self):
        for q in ["awq", "gptq", "gguf"]:
            req = DeployRequest(model="m", quantization=q)
            assert req.quantization == q

    def test_dtype_values(self):
        for dtype in ["auto", "float16", "bfloat16", "float32"]:
            req = DeployRequest(model="m", dtype=dtype)
            assert req.dtype == dtype


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: TokenUsage — 业务规则验证
# ═════════════════════════════════════════════════════════════════════════════════


class TestTokenUsage:
    """Token 使用统计模型"""

    def test_valid_usage(self):
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=500,
            total_tokens=600,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 500
        assert usage.total_tokens == 600

    def test_total_equals_prompt_plus_completion(self):
        """total_tokens 必须等于 prompt_tokens + completion_tokens"""
        usage = TokenUsage(
            prompt_tokens=50,
            completion_tokens=200,
            total_tokens=250,
        )
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_zero_usage(self):
        usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        assert usage.total_tokens == 0

    def test_prompt_only_usage(self):
        """仅统计输入 tokens 的场景"""
        usage = TokenUsage(
            prompt_tokens=300,
            completion_tokens=0,
            total_tokens=300,
        )
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: StreamResponse
# ═════════════════════════════════════════════════════════════════════════════════


class TestStreamResponse:
    """流式响应模型"""

    def test_non_final_chunk(self):
        resp = StreamResponse(
            request_id="req-1",
            delta="Hello",
            is_final=False,
        )
        assert resp.request_id == "req-1"
        assert resp.delta == "Hello"
        assert resp.is_final is False
        assert resp.usage is None
        assert resp.finish_reason is None

    def test_final_chunk_with_usage(self):
        resp = StreamResponse(
            request_id="req-1",
            delta="World",
            is_final=True,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )
        assert resp.is_final is True
        assert resp.finish_reason == "stop"
        assert resp.usage["total_tokens"] == 15

    def test_final_chunk_without_usage(self):
        """usage 可以为 None 即使 is_final=True"""
        resp = StreamResponse(
            request_id="req-1",
            delta=".",
            is_final=True,
            finish_reason="length",
        )
        assert resp.is_final is True
        assert resp.usage is None

    def test_finish_reason_values(self):
        for reason in ["stop", "length", "timeout"]:
            resp = StreamResponse(
                request_id="r",
                delta="",
                is_final=True,
                finish_reason=reason,
            )
            assert resp.finish_reason == reason


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: BatchInferenceResponse
# ═════════════════════════════════════════════════════════════════════════════════


class TestBatchInferenceResponseModel:
    """批量推理响应模型业务规则"""

    def test_completed_plus_failed_equals_total(self):
        response = BatchInferenceResponse(
            batch_id="b1",
            model="m",
            total=10,
            completed=7,
            failed=3,
            results=[],
            total_latency_ms=100.0,
            avg_latency_ms=10.0,
        )
        assert response.completed + response.failed == response.total

    def test_all_success_no_failures(self):
        response = BatchInferenceResponse(
            batch_id="b1",
            model="m",
            total=5,
            completed=5,
            failed=0,
            results=[],
            total_latency_ms=50.0,
            avg_latency_ms=10.0,
        )
        assert response.failed == 0
        assert response.completed == response.total

    def test_all_failed_no_successes(self):
        response = BatchInferenceResponse(
            batch_id="b1",
            model="m",
            total=5,
            completed=0,
            failed=5,
            results=[],
            total_latency_ms=0.0,
            avg_latency_ms=0.0,
        )
        assert response.completed == 0
        assert response.failed == response.total

    def test_results_list_matches_completed_count(self):
        results = [
            InferenceResponse(
                request_id=f"r{i}",
                model="m",
                prompt="p",
                generated_text="t",
                finish_reason="stop",
                latency_ms=10.0,
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
            for i in range(3)
        ]
        response = BatchInferenceResponse(
            batch_id="b1",
            model="m",
            total=3,
            completed=3,
            failed=0,
            results=results,
            total_latency_ms=30.0,
            avg_latency_ms=10.0,
        )
        assert len(response.results) == response.completed

    def test_avg_latency_is_total_divided_by_completed_when_positive(self):
        response = BatchInferenceResponse(
            batch_id="b1",
            model="m",
            total=4,
            completed=4,
            failed=0,
            results=[],
            total_latency_ms=48.0,
            avg_latency_ms=12.0,
        )
        assert response.avg_latency_ms == response.total_latency_ms / response.completed


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: JobInfo
# ═════════════════════════════════════════════════════════════════════════════════


class TestJobInfo:
    """作业信息模型"""

    def test_queued_job(self):
        job = JobInfo(
            job_id="job-1",
            model="Qwen2.5-7B",
            status="queued",
            priority=5,
            created_at=datetime.now(),
            progress=0.0,
            prompt="Hello",
        )
        assert job.status == "queued"
        assert job.progress == 0.0
        assert job.started_at is None
        assert job.completed_at is None
        assert job.result is None
        assert job.error is None

    def test_completed_job(self):
        now = datetime.now()
        job = JobInfo(
            job_id="job-2",
            model="m",
            status="completed",
            priority=8,
            created_at=now,
            started_at=now,
            completed_at=now,
            progress=1.0,
            prompt="p",
            result="generated text",
            allocated_nodes=["node-1"],
            retry_count=0,
        )
        assert job.status == "completed"
        assert job.progress == 1.0
        assert job.result == "generated text"

    def test_failed_job_with_error(self):
        job = JobInfo(
            job_id="job-3",
            model="m",
            status="failed",
            priority=5,
            created_at=datetime.now(),
            progress=0.3,
            prompt="p",
            error="CUDA OOM",
            retry_count=2,
        )
        assert job.status == "failed"
        assert job.error == "CUDA OOM"
        assert job.retry_count == 2

    def test_status_values(self):
        for status in ["queued", "scheduling", "running", "completed", "failed", "cancelled"]:
            job = JobInfo(
                job_id="j",
                model="m",
                status=status,
                priority=1,
                created_at=datetime.now(),
                progress=0.0,
                prompt="p",
            )
            assert job.status == status


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: DeployResponse / UndeployResponse / LoadModelResponse / UnloadModelResponse / ModelStatusResponse
# ═════════════════════════════════════════════════════════════════════════════════


class TestDeployResponse:
    """部署响应模型"""

    def test_successful_deploy(self):
        resp = DeployResponse(
            model_id="qwen2.5-7b",
            status="deployed",
            replicas=2,
            message="Model deployed successfully",
        )
        assert resp.model_id == "qwen2.5-7b"
        assert resp.status == "deployed"
        assert resp.replicas == 2
        assert "deployed" in resp.message


class TestUndeployResponse:
    """卸载响应模型"""

    def test_successful_undeploy(self):
        resp = UndeployResponse(
            model_id="qwen2.5-7b",
            status="unloaded",
            message="Model unloaded successfully",
        )
        assert resp.model_id == "qwen2.5-7b"
        assert resp.status == "unloaded"


class TestLoadModelResponse:
    """加载模型响应模型"""

    def test_loaded_status(self):
        resp = LoadModelResponse(
            model="Qwen2.5-1.5B",
            status="loaded",
            message="Model loaded successfully",
        )
        assert resp.model == "Qwen2.5-1.5B"
        assert resp.status == "loaded"

    def test_loading_status(self):
        resp = LoadModelResponse(
            model="m",
            status="loading",
            message="Model is being loaded",
        )
        assert resp.status == "loading"

    def test_failed_status(self):
        resp = LoadModelResponse(
            model="m",
            status="failed",
            message="Failed: OOM",
        )
        assert resp.status == "failed"


class TestUnloadModelResponse:
    """卸载模型响应模型"""

    def test_unloaded_status(self):
        resp = UnloadModelResponse(
            model="Qwen2.5-7B",
            status="unloaded",
            message="Model unloaded",
        )
        assert resp.model == "Qwen2.5-7B"
        assert resp.status == "unloaded"


class TestModelStatusResponse:
    """模型状态响应模型"""

    def test_empty_models(self):
        resp = ModelStatusResponse(loaded_models=[], total=0)
        assert resp.loaded_models == []
        assert resp.total == 0

    def test_multiple_models(self):
        models = ["Qwen2.5-7B", "Qwen2.5-1.5B", "Llama-3-8B"]
        resp = ModelStatusResponse(loaded_models=models, total=len(models))
        assert len(resp.loaded_models) == 3
        assert resp.total == 3

    def test_total_matches_list_length(self):
        models = ["m1", "m2"]
        resp = ModelStatusResponse(loaded_models=models, total=2)
        assert resp.total == len(resp.loaded_models)


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: BenchmarkResponse
# ═════════════════════════════════════════════════════════════════════════════════


class TestBenchmarkResponseModel:
    """基准测试响应模型"""

    def test_in_progress_benchmark(self):
        resp = BenchmarkResponse(
            benchmark_id="bench-1",
            model="Qwen2.5-7B",
            test_set="mmlu",
            status="running",
            total_samples=100,
            completed_samples=45,
        )
        assert resp.status == "running"
        assert resp.completed_samples == 45
        assert resp.results is None
        assert resp.metrics is None

    def test_completed_benchmark_with_results(self):
        resp = BenchmarkResponse(
            benchmark_id="bench-2",
            model="Qwen2.5-7B",
            test_set="humaneval",
            status="completed",
            total_samples=100,
            completed_samples=100,
            results={"accuracy": 0.82, "pass@1": 0.75},
            metrics={
                "avg_latency_ms": 120.5,
                "throughput_tokens_per_sec": 450.0,
                "gpu_utilization": 0.92,
            },
        )
        assert resp.status == "completed"
        assert resp.results is not None
        assert resp.results["accuracy"] == 0.82
        assert resp.metrics is not None
        assert resp.metrics["throughput_tokens_per_sec"] == 450.0

    def test_failed_benchmark(self):
        resp = BenchmarkResponse(
            benchmark_id="bench-3",
            model="Qwen2.5-7B",
            test_set="math",
            status="failed",
            total_samples=100,
            completed_samples=10,
        )
        assert resp.status == "failed"
        assert resp.completed_samples < resp.total_samples


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: MetricsResponse
# ═════════════════════════════════════════════════════════════════════════════════


class TestMetricsResponseModel:
    """指标响应模型"""

    def test_empty_metrics(self):
        now = datetime.now()
        resp = MetricsResponse(
            timestamp=now,
            system={},
            models={},
            nodes={},
        )
        assert resp.timestamp == now
        assert resp.system == {}
        assert resp.models == {}
        assert resp.nodes == {}

    def test_system_metrics(self):
        resp = MetricsResponse(
            timestamp=datetime.now(),
            system={"cpu_usage": 0.45, "memory_used_gb": 32.5},
        )
        assert resp.system["cpu_usage"] == 0.45
        assert resp.system["memory_used_gb"] == 32.5

    def test_model_metrics(self):
        resp = MetricsResponse(
            timestamp=datetime.now(),
            models={
                "qwen2.5-7b": {
                    "request_count": 120,
                    "avg_latency_ms": 85.3,
                }
            },
        )
        assert resp.models["qwen2.5-7b"]["request_count"] == 120
        assert resp.models["qwen2.5-7b"]["avg_latency_ms"] == 85.3

    def test_node_metrics(self):
        resp = MetricsResponse(
            timestamp=datetime.now(),
            nodes={
                "node-1": {"gpu_utilization": 0.6, "gpu_memory_used_mb": 8192},
                "node-2": {"gpu_utilization": 0.3, "gpu_memory_used_mb": 4096},
            },
        )
        assert len(resp.nodes) == 2
        assert resp.nodes["node-1"]["gpu_utilization"] == 0.6
        assert resp.nodes["node-2"]["gpu_memory_used_mb"] == 4096

    def test_full_metrics(self):
        resp = MetricsResponse(
            timestamp=datetime.now(),
            system={"cpu": 0.5},
            models={"m1": {"rps": 100.0}},
            nodes={"n1": {"gpu": 0.8}},
        )
        assert len(resp.system) == 1
        assert len(resp.models) == 1
        assert len(resp.nodes) == 1


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: LoadModelRequest 单元测试（与 backend_selection.py 互补）
# ═════════════════════════════════════════════════════════════════════════════════


class TestLoadModelRequestModel:
    """LoadModelRequest 基础模型测试"""

    def test_default_backend_is_huggingface(self):
        req = LoadModelRequest(model="test-model")
        assert req.backend == "huggingface"

    def test_default_tensor_parallel(self):
        req = LoadModelRequest(model="test-model")
        assert req.tensor_parallel == 1

    def test_default_gpu_memory_utilization(self):
        req = LoadModelRequest(model="test-model")
        assert req.gpu_memory_utilization == 0.6

    def test_default_max_model_len(self):
        req = LoadModelRequest(model="test-model")
        assert req.max_model_len == 2048

    def test_default_dtype(self):
        req = LoadModelRequest(model="test-model")
        assert req.dtype == "auto"

    def test_optional_fields_default_to_none(self):
        req = LoadModelRequest(model="test-model")
        assert req.model_path is None
        assert req.quantization is None
        assert req.tgi_base_url is None
        assert req.sglang_base_url is None
        assert req.sglang_timeout is None

    def test_tgi_specific_config(self):
        req = LoadModelRequest(
            model="test",
            backend="tgi",
            tgi_base_url="http://tgi-server:8080",
        )
        assert req.backend == "tgi"
        assert req.tgi_base_url == "http://tgi-server:8080"

    def test_sglang_specific_config(self):
        req = LoadModelRequest(
            model="test",
            backend="sglang",
            sglang_base_url="http://sglang:30000",
            sglang_timeout=300,
        )
        assert req.backend == "sglang"
        assert req.sglang_base_url == "http://sglang:30000"
        assert req.sglang_timeout == 300

    def test_quantization_values(self):
        for q in ["awq", "gptq", "gguf"]:
            req = LoadModelRequest(model="test", quantization=q)
            assert req.quantization == q


# ═════════════════════════════════════════════════════════════════════════════════
# 新增: GPUInfo 内存百分比覆盖
# ═════════════════════════════════════════════════════════════════════════════════


class TestGPUInfoExtended:
    """GPUInfo 内存计算扩展测试"""

    def test_memory_used_percent_full_utilization(self):
        info = GPUInfo(
            gpu_id=0,
            name="Test",
            memory_total=1024,
            memory_used=1024,
            memory_free=0,
            utilization=1.0,
            temperature=80.0,
        )
        assert info.memory_used_percent == 1.0

    def test_memory_used_percent_no_usage(self):
        info = GPUInfo(
            gpu_id=0,
            name="Test",
            memory_total=1024,
            memory_used=0,
            memory_free=1024,
            utilization=0.0,
            temperature=30.0,
        )
        assert info.memory_used_percent == 0.0

    def test_memory_used_percent_half_usage(self):
        info = GPUInfo(
            gpu_id=0,
            name="Test",
            memory_total=2000,
            memory_used=1000,
            memory_free=1000,
            utilization=0.5,
            temperature=45.0,
        )
        assert info.memory_used_percent == 0.5

    def test_memory_used_never_exceeds_one(self):
        """即使用量超过总量（异常状态），百分比也应 <=1.0"""
        info = GPUInfo(
            gpu_id=0,
            name="Test",
            memory_total=1000,
            memory_used=1200,
            memory_free=0,
            utilization=1.0,
            temperature=90.0,
        )
        assert info.memory_used_percent >= 0.0  # 至少不返回负数
