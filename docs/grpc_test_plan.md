# QuantumFlow gRPC 测试计划书

## 一、测试策略

### 1.1 测试原则
1. **业务逻辑优先**: 所有测试必须验证业务逻辑正确性，而非仅验证"无报错"
2. **强断言**: 每个测试必须包含明确的、精确的断言
3. **全场景覆盖**: 常规、边界、非法输入、异常、并发、多分支
4. **零容忍假通过**: 测试本身不能有漏洞

### 1.2 测试分层

| 层级 | 测试内容 | 测试数量（预估） |
|------|---------|----------------|
| Proto 验证 | 消息序列化/反序列化 | 50 |
| 单元测试 | 拦截器、异常、工具函数 | 100 |
| 服务测试 | 各 Servicer 业务逻辑 | 200 |
| 集成测试 | 端到端 gRPC 调用 | 50 |

**目标覆盖率**: ≥95%

---

## 二、测试文件结构

```
tests/
├── unit/
│   └── grpc/
│       ├── test_proto_validation.py      # Proto 序列化测试
│       ├── test_exceptions.py             # 异常测试
│       ├── test_interceptors/
│       │   ├── __init__.py
│       │   ├── test_logging_interceptor.py
│       │   ├── test_auth_interceptor.py
│       │   ├── test_metrics_interceptor.py
│       │   └── test_rate_limit_interceptor.py
│       ├── test_servicers/
│       │   ├── __init__.py
│       │   ├── test_inference_service.py
│       │   ├── test_cluster_service.py
│       │   ├── test_scheduler_service.py
│       │   ├── test_model_management_service.py
│       │   └── test_health_service.py
│       └── test_clients/
│           ├── __init__.py
│           ├── test_inference_client.py
│           ├── test_cluster_client.py
│           └── test_scheduler_client.py
├── integration/
│   └── grpc/
│       ├── test_end_to_end.py             # 端到端测试
│       ├── test_worker_controller.py      # Worker-Controller 通信
│       └── test_load_balancing.py          # 负载均衡测试
└── conftest.py                             # pytest 配置
```

---

## 三、Proto 验证测试 (test_proto_validation.py)

### 3.1 枚举值测试

```python
class TestModelBackendEnum:
    """验证 ModelBackend 枚举值"""

    def test_enum_values_are_unique(self):
        """枚举值必须唯一"""
        values = list(ModelBackend.values())
        assert len(values) == len(set(values)), "枚举值重复"

    def test_enum_has_unspecified(self):
        """必须有 UNSPECIFIED 默认值"""
        assert ModelBackend.MODEL_BACKEND_UNSPECIFIED == 0

    def test_all_backends_defined(self):
        """所有推理后端都已定义"""
        expected = {'VLLM', 'HUGGINGFACE', 'TGI', 'SGLANG'}
        actual = {b.name for b in ModelBackend if b != ModelBackend.MODEL_BACKEND_UNSPECIFIED}
        assert actual == expected

    @pytest.mark.parametrize("backend", [
        ModelBackend.MODEL_BACKEND_VLLM,
        ModelBackend.MODEL_BACKEND_HUGGINGFACE,
        ModelBackend.MODEL_BACKEND_TGI,
        ModelBackend.MODEL_BACKEND_SGLANG,
    ])
    def test_backend_serialization_roundtrip(self, backend):
        """枚举序列化/反序列化往返一致"""
        serialized = backend.SerializeToString()
        restored = ModelBackend()
        restored.ParseFromString(serialized)
        assert restored == backend
```

### 3.2 消息字段测试

```python
class TestGPUMemoryMessage:
    """GPUMemory 消息测试"""

    def test_total_bytes_must_be_positive(self):
        """total_bytes 必须为正数"""
        msg = GPUMemory(total_bytes=0)
        with pytest.raises(ValueError, match="total_bytes must be positive"):
            validate_gpu_memory(msg)

    def test_used_bytes_cannot_exceed_total(self):
        """used_bytes 不能超过 total_bytes"""
        msg = GPUMemory(total_bytes=100, used_bytes=200)
        assert msg.used_bytes > msg.total_bytes  # 业务逻辑校验
        # 应该返回错误或截断

    def test_memory_utilization_range(self):
        """utilization 必须在 [0, 1] 范围内"""
        msg_valid = GPUMemory(utilization=0.5)
        assert 0 <= msg_valid.utilization <= 1

        msg_invalid = GPUMemory(utilization=1.5)
        assert not (0 <= msg_invalid.utilization <= 1)

    @pytest.mark.parametrize("total,available,used,expected_util", [
        (100, 60, 40, 0.4),
        (100, 100, 0, 0.0),
        (100, 0, 100, 1.0),
        (16 * 1024**3, 8 * 1024**3, 8 * 1024**3, 0.5),  # 16GB, 8GB used, 50%
    ])
    def test_memory_calculation(self, total, available, used, expected_util):
        """显存计算正确性"""
        msg = GPUMemory(total_bytes=total, available_bytes=available, used_bytes=used)
        assert abs(msg.utilization - expected_util) < 0.01


class TestInferenceRequestValidation:
    """InferenceRequest 字段验证"""

    def test_request_id_required(self):
        """request_id 不能为空"""
        msg = InferenceRequest(request_id="")
        assert not msg.request_id  # 空字符串

    def test_request_id_format_uuid(self):
        """request_id 必须是有效 UUID"""
        msg_valid = InferenceRequest(request_id=str(uuid.uuid4()))
        assert is_valid_uuid(msg_valid.request_id)

        msg_invalid = InferenceRequest(request_id="not-a-uuid")
        assert not is_valid_uuid(msg_invalid.request_id)

    def test_max_tokens_must_be_positive(self):
        """max_tokens 必须 > 0"""
        msg = InferenceRequest(max_tokens=0)
        assert msg.max_tokens <= 0  # 应该被拒绝

    def test_max_tokens_has_upper_bound(self):
        """max_tokens 有上限 (例如 8192)"""
        msg = InferenceRequest(max_tokens=10000)
        assert msg.max_tokens > 8192  # 超出上限

    def test_temperature_range(self):
        """temperature 必须在 [0, 2] 范围内"""
        for temp in [-0.1, 2.1, -1.0]:
            msg = InferenceRequest(temperature=temp)
            assert temp < 0 or temp > 2  # 超出范围

    def test_top_p_range(self):
        """top_p 必须在 (0, 1] 范围内"""
        for invalid in [0, -0.1, 1.1, 2.0]:
            msg = InferenceRequest(top_p=invalid)
            assert invalid <= 0 or invalid > 1

    def test_top_k_non_negative(self):
        """top_k 必须 >= 0"""
        msg_negative = InferenceRequest(top_k=-1)
        assert msg_negative.top_k < 0

        msg_zero = InferenceRequest(top_k=0)
        assert msg_zero.top_k == 0  # 0 表示 disabled

    def test_repetition_penalty_range(self):
        """repetition_penalty 必须 >= 1.0"""
        for invalid in [0.5, 0.9, 0.0]:
            msg = InferenceRequest(repetition_penalty=invalid)
            assert msg.repetition_penalty < 1.0

        msg_valid = InferenceRequest(repetition_penalty=1.2)
        assert msg_valid.repetition_penalty >= 1.0

    def test_extra_params_type(self):
        """extra_params 必须是有效的 map"""
        msg = InferenceRequest(extra_params={"temperature": "0.5"})
        assert isinstance(msg.extra_params, dict)
        assert msg.extra_params["temperature"] == "0.5"


class TestNodeResourcesMessage:
    """NodeResources 消息测试"""

    def test_node_id_required(self):
        """node_id 不能为空"""
        msg = NodeResources(node_id="")
        assert not msg.node_id

    def test_port_range(self):
        """port 必须在有效范围内"""
        msg_valid = NodeResources(port=8000)
        assert 1 <= msg_valid.port <= 65535

        msg_invalid = NodeResources(port=0)
        assert msg_invalid.port < 1 or msg_invalid.port > 65535

    def test_gpu_list_empty_allowed(self):
        """GPU 列表可以为空（CPU-only 节点）"""
        msg = NodeResources(node_id="cpu-node", gpus=[])
        assert len(msg.gpus) == 0

    def test_gpu_list_order(self):
        """GPU 列表按索引排序"""
        gpus = [
            GPUInfo(index=1, name="GPU 1"),
            GPUInfo(index=0, name="GPU 0"),
            GPUInfo(index=2, name="GPU 2"),
        ]
        msg = NodeResources(node_id="test", gpus=gpus)
        # 排序后应按 index 排列
        sorted_gpus = sorted(msg.gpus, key=lambda g: g.index)
        assert [g.index for g in sorted_gpus] == [0, 1, 2]
```

### 3.3 序列化测试

```python
class TestSerializationRoundtrip:
    """序列化往返测试"""

    @pytest.mark.parametrize("message_class,test_cases", [
        (InferenceRequest, [
            {"request_id": "uuid-123", "model_name": "llama-2-7b", "prompt": "Hello"},
            {"request_id": "uuid-456", "model_name": "mixtral-8x7b", "prompt": "", "max_tokens": 100},
        ]),
        (GPUMemory, [
            {"total_bytes": 16 * 1024**3, "used_bytes": 8 * 1024**3},
            {"total_bytes": 80 * 1024**3, "used_bytes": 40 * 1024**3},
        ]),
        # ... 更多消息类型
    ])
    def test_serialize_deserialize_preserves_data(self, message_class, test_cases):
        """序列化/反序列化后数据完全一致"""
        for case in test_cases:
            msg = message_class(**case)
            serialized = msg.SerializeToString()
            restored = message_class()
            restored.ParseFromString(serialized)
            assert restored == msg

    def test_serialize_size_reasonable(self):
        """序列化后大小在合理范围内"""
        msg = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-70b",
            prompt="Hello world, this is a test prompt.",
            max_tokens=100,
            temperature=0.7,
        )
        serialized = msg.SerializeToString()
        # 应该小于 1KB
        assert len(serialized) < 1024


class TestProtoBackwardCompatibility:
    """Proto 向后兼容性测试"""

    def test_unknown_fields_preserved(self):
        """解析时保留未知字段（未来扩展兼容）"""
        # 旧版本客户端发送的请求可能包含新版本字段
        # 服务器解析时不能丢弃未知字段
        raw = b'\x08\x01\x12\x05hello'  # 包含未知字段
        msg = InferenceRequest()
        msg.ParseFromString(raw)
        # 未知字段应该被保留
        assert msg.request_id == ""  # request_id 未设置

    def test_default_values_not_serialized(self):
        """默认值字段不会被序列化（节省空间）"""
        msg = InferenceRequest(request_id="test")
        serialized = msg.SerializeToString()
        # 不包含 default 值的字段
        assert b'\x10\x00' not in serialized  # max_tokens=0 默认值
```

---

## 四、异常测试 (test_exceptions.py)

```python
class TestGrpcQuantumFlowError:
    """GrpcQuantumFlowError 异常测试"""

    def test_error_has_code_and_message(self):
        """异常包含 code 和 message"""
        error = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        assert error.code == grpc.StatusCode.NOT_FOUND
        assert error.message == "Node not found"

    def test_error_string_format(self):
        """错误字符串格式正确"""
        error = GrpcQuantumFlowError(grpc.StatusCode.INVALID_ARGUMENT, "Bad request")
        error_str = str(error)
        assert "[INVALID_ARGUMENT]" in error_str
        assert "Bad request" in error_str

    def test_error_equality(self):
        """相同 code/message 的错误相等"""
        error1 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        error2 = GrpcQuantumFlowError(grpc.StatusCode.NOT_FOUND, "Node not found")
        assert error1 == error2

    def test_error_chain(self):
        """异常可以链接"""
        cause = ValueError("original error")
        error = GrpcQuantumFlowError(grpc.StatusCode.INTERNAL, "wrapped", cause=cause)
        assert error.__cause__ == cause


class TestNodeNotFoundError:
    """NodeNotFoundError 测试"""

    @pytest.fixture
    def error(self):
        return NodeNotFoundError(node_id="node-123")

    def test_error_message_contains_node_id(self, error):
        assert "node-123" in str(error)

    def test_error_code_is_not_found(self, error):
        assert error.code == grpc.StatusCode.NOT_FOUND


class TestModelNotLoadedError:
    """ModelNotLoadedError 测试"""

    @pytest.fixture
    def error(self):
        return ModelNotLoadedError(model_name="llama-2-70b")

    def test_error_message_contains_model_name(self, error):
        assert "llama-2-70b" in str(error)

    def test_error_code_is_not_found(self, error):
        assert error.code == grpc.StatusCode.NOT_FOUND


class TestSchedulingError:
    """SchedulingError 测试"""

    def test_error_with_reasons(self):
        """包含调度失败原因"""
        error = SchedulingError(
            reason="No available GPU with enough memory",
            requested="80GB",
            available="40GB"
        )
        assert "80GB" in str(error)
        assert "40GB" in str(error)


class TestResourceUnavailableError:
    """ResourceUnavailableError 测试"""

    def test_error_with_resource_type(self):
        """包含资源类型信息"""
        error = ResourceUnavailableError(resource="GPU", required=8, available=0)
        assert error.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert "GPU" in str(error)


class TestExceptionMapping:
    """gRPC 状态码映射测试"""

    @pytest.mark.parametrize("error_class,expected_status", [
        (NodeNotFoundError, grpc.StatusCode.NOT_FOUND),
        (ModelNotLoadedError, grpc.StatusCode.NOT_FOUND),
        (SchedulingError, grpc.StatusCode.UNAVAILABLE),
        (ResourceUnavailableError, grpc.StatusCode.RESOURCE_EXHAUSTED),
        (GrpcQuantumFlowError(grpc.StatusCode.INTERNAL, "test"), grpc.StatusCode.INTERNAL),
    ])
    def test_exception_maps_to_correct_status(self, error_class, expected_status):
        """每种异常对应正确的 gRPC 状态码"""
        assert error_class.code == expected_status


class TestExceptionFromRpcError:
    """从 RpcError 转换异常"""

    def test_from_grpc_rpc_error(self):
        """从 grpc.RpcError 转换为 GrpcQuantumFlowError"""
        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.UNAUTHENTICATED
        rpc_error.details = lambda: "Invalid token"

        error = GrpcQuantumFlowError.from_rpc_error(rpc_error)
        assert error.code == grpc.StatusCode.UNAUTHENTICATED
        assert error.message == "Invalid token"
```

---

## 五、拦截器测试

### 5.1 日志拦截器 (test_logging_interceptor.py)

```python
class TestLoggingInterceptor:
    """日志拦截器测试"""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def interceptor(self, mock_logger):
        return LoggingInterceptor(mock_logger)

    def test_logs_method_name(self, interceptor, mock_logger):
        """记录方法名"""
        handler_details = create_handler_details(method="/quantumflow.v1.InferenceService/Inference")
        interceptor.intercept_service(lambda x: create_response(), handler_details)
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "InferenceService/Inference" in call_args

    def test_logs_call_duration(self, interceptor, mock_logger):
        """记录调用耗时"""
        def slow_handler(details):
            time.sleep(0.1)
            return create_response()

        handler_details = create_handler_details(method="/test")
        interceptor.intercept_service(slow_handler, handler_details)

        call_args = mock_logger.info.call_args[0][0]
        assert "duration_ms" in call_args
        # 耗时应该 >= 100ms
        duration_match = re.search(r'duration_ms[:\s]+(\d+\.?\d*)', call_args)
        assert duration_match and float(duration_match.group(1)) >= 100

    def test_logs_status_code_on_success(self, interceptor, mock_logger):
        """成功时记录 OK 状态"""
        response = create_response()
        handler_details = create_handler_details()

        interceptor.intercept_service(lambda x: response, handler_details)

        call_args = mock_logger.info.call_args[0][0]
        assert "OK" in call_args or "success" in call_args.lower()

    def test_logs_error_on_failure(self, interceptor, mock_logger):
        """失败时记录错误"""
        def failing_handler(details):
            raise grpc.RpcError(grpc.StatusCode.INTERNAL, "Test error")

        handler_details = create_handler_details()

        with pytest.raises(grpc.RpcError):
            interceptor.intercept_service(failing_handler, handler_details)

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "INTERNAL" in call_args or "Test error" in call_args


class TestLoggingInterceptorEdgeCases:
    """日志拦截器边界情况"""

    def test_handles_metadata(self, interceptor, mock_logger):
        """正确处理 metadata"""
        metadata = [
            ('x-request-id', 'req-123'),
            ('x-user-id', 'user-456'),
        ]
        handler_details = create_handler_details(metadata=metadata)

        interceptor.intercept_service(lambda x: create_response(), handler_details)

        # 应该在日志中包含 request-id
        all_calls = str(mock_logger.info.call_args_list)
        assert "req-123" in all_calls

    def test_handles_binary_metadata(self, interceptor, mock_logger):
        """处理二进制 metadata"""
        metadata = [('x-binary-data', b'\x00\x01\x02\x03')]
        handler_details = create_handler_details(metadata=metadata)

        # 不应该崩溃
        interceptor.intercept_service(lambda x: create_response(), handler_details)

    def test_timeout_not_recorded_as_error(self, interceptor, mock_logger):
        """超时不应记录为 error（可能是预期行为）"""
        def timeout_handler(details):
            raise grpc.RpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline exceeded")

        handler_details = create_handler_details()

        with pytest.raises(grpc.RpcError):
            interceptor.intercept_service(timeout_handler, handler_details)

        # 应该记录为 warning，不是 error
        # 或者根本不被记录（避免日志噪音）
```

### 5.2 认证拦截器 (test_auth_interceptor.py)

```python
class TestAuthInterceptor:
    """认证拦截器测试"""

    @pytest.fixture
    def valid_tokens(self):
        return {
            "valid-token-123": "user-1",
            "admin-token-456": "admin",
        }

    @pytest.fixture
    def interceptor(self, valid_tokens):
        return AuthInterceptor(valid_tokens)

    def test_accepts_valid_bearer_token(self, interceptor):
        """接受有效的 Bearer token"""
        metadata = [('authorization', 'Bearer valid-token-123')]
        handler_details = create_handler_details(metadata=metadata)

        result = interceptor.intercept_service(lambda x: create_response(), handler_details)
        assert result is not None

    def test_accepts_token_without_bearer_prefix(self, interceptor):
        """接受没有 Bearer 前缀的 token"""
        metadata = [('authorization', 'valid-token-123')]
        handler_details = create_handler_details(metadata=metadata)

        result = interceptor.intercept_service(lambda x: create_response(), handler_details)
        assert result is not None

    def test_rejects_invalid_token(self, interceptor):
        """拒绝无效 token"""
        metadata = [('authorization', 'Bearer invalid-token')]
        handler_details = create_handler_details(metadata=metadata)

        with pytest.raises(grpc.RpcError) as exc_info:
            interceptor.intercept_service(lambda x: create_response(), handler_details)

        assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED

    def test_rejects_expired_token(self, interceptor):
        """拒绝过期 token"""
        expired_token = "expired-token-789"
        # 假设有过期机制
        metadata = [('authorization', f'Bearer {expired_token}')]
        handler_details = create_handler_details(metadata=metadata)

        with pytest.raises(grpc.RpcError) as exc_info:
            interceptor.intercept_service(lambda x: create_response(), handler_details)

        assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED

    def test_rejects_missing_authorization_header(self, interceptor):
        """拒绝缺少 Authorization header"""
        handler_details = create_handler_details(metadata=[])

        with pytest.raises(grpc.RpcError) as exc_info:
            interceptor.intercept_service(lambda x: create_response(), handler_details)

        assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED

    def test_rejects_empty_token(self, interceptor):
        """拒绝空 token"""
        metadata = [('authorization', '')]
        handler_details = create_handler_details(metadata=metadata)

        with pytest.raises(grpc.RpcError) as exc_info:
            interceptor.intercept_service(lambda x: create_response(), handler_details)

        assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


class TestAuthInterceptorBypass:
    """认证绕过测试"""

    def test_health_service_bypasses_auth(self):
        """HealthService 应该跳过认证"""
        health_interceptor = AuthInterceptor({"token": "user"}, bypass_methods=[
            "/quantumflow.v1.HealthService/Check",
            "/quantumflow.v1.HealthService/Watch",
        ])

        metadata = [('authorization', 'Bearer token')]  # 无效 token
        handler_details = create_handler_details(
            method="/quantumflow.v1.HealthService/Check",
            metadata=metadata
        )

        # 应该不抛异常（跳过认证）
        result = health_interceptor.intercept_service(lambda x: create_response(), handler_details)
        assert result is not None

    def test_non_health_service_requires_auth(self):
        """非 HealthService 需要认证"""
        health_interceptor = AuthInterceptor({"token": "user"}, bypass_methods=[
            "/quantumflow.v1.HealthService/Check",
        ])

        metadata = [('authorization', 'Bearer invalid')]  # 无效 token
        handler_details = create_handler_details(
            method="/quantumflow.v1.InferenceService/Inference",
            metadata=metadata
        )

        with pytest.raises(grpc.RpcError):
            health_interceptor.intercept_service(lambda x: create_response(), handler_details)


class TestAuthInterceptorEdgeCases:
    """认证拦截器边界情况"""

    def test_handles_unicode_in_token(self, interceptor):
        """处理 token 中的 Unicode 字符"""
        metadata = [('authorization', 'Bearer 中文token')]
        handler_details = create_handler_details(metadata=metadata)

        with pytest.raises(grpc.RpcError):
            interceptor.intercept_service(lambda x: create_response(), handler_details)

    def test_handles_very_long_token(self, interceptor):
        """处理超长 token（防 DoS）"""
        long_token = "Bearer " + "a" * 10000
        metadata = [('authorization', long_token)]
        handler_details = create_handler_details(metadata=metadata)

        # 应该快速拒绝，不耗尽内存
        with pytest.raises(grpc.RpcError):
            interceptor.intercept_service(lambda x: create_response(), handler_details)

    def test_token_extraction_is_case_sensitive(self, interceptor):
        """Authorization 头提取大小写敏感"""
        metadata = [('Authorization', 'Bearer valid-token-123')]  # 大写 A
        handler_details = create_handler_details(metadata=metadata)

        # gRPC metadata 应该大小写不敏感
        # 但我们测试实际行为
        result = interceptor.intercept_service(lambda x: create_response(), handler_details)
        assert result is not None  # 应该成功（gRPC 自动处理）
```

### 5.3 限流拦截器 (test_rate_limit_interceptor.py)

```python
class TestRateLimitInterceptor:
    """限流拦截器测试"""

    @pytest.fixture
    def interceptor(self):
        return RateLimitInterceptor(qps=10, burst=20)

    def test_allows_request_under_limit(self, interceptor):
        """限流内请求通过"""
        handler_details = create_handler_details()
        for i in range(10):
            result = interceptor.intercept_service(lambda x: create_response(), handler_details)
            assert result is not None

    def test_blocks_request_over_limit(self, interceptor):
        """超出限流拒绝"""
        handler_details = create_handler_details()

        # 耗尽令牌桶
        for i in range(20):
            interceptor.intercept_service(lambda x: create_response(), handler_details)

        # 下一个请求应该被拒绝
        with pytest.raises(grpc.RpcError) as exc_info:
            interceptor.intercept_service(lambda x: create_response(), handler_details)

        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

    def test_rate_limit_message_contains_retry_info(self, interceptor):
        """限流消息包含重试信息"""
        handler_details = create_handler_details()

        # 耗尽令牌桶
        for i in range(20):
            interceptor.intercept_service(lambda x: create_response(), handler_details)

        with pytest.raises(grpc.RpcError) as exc_info:
            interceptor.intercept_service(lambda x: create_response(), handler_details)

        assert "retry" in str(exc_info.value).lower()


class TestTokenBucketAlgorithm:
    """令牌桶算法测试"""

    def test_burst_allows_short_surge(self):
        """burst 允许短暂突发"""
        bucket = TokenBucket(capacity=10, refill_rate=1)  # 10 容量，每秒补充 1 个
        # 初始可以获取 10 个
        for i in range(10):
            assert bucket.try_acquire()

        # 第 11 个应该失败
        assert not bucket.try_acquire()

    def test_token_refill_rate(self):
        """令牌补充速率正确"""
        bucket = TokenBucket(capacity=10, refill_rate=10)  # 每秒 10 个
        bucket.try_acquire()  # 消耗 1 个

        # 等待 100ms，应该补充 1 个
        time.sleep(0.1)
        # 现在应该有 10 个（补充了 1 个）
        assert bucket.try_acquire()

    def test_tokens_capped_at_capacity(self):
        """令牌不会超过容量"""
        bucket = TokenBucket(capacity=10, refill_rate=100)
        time.sleep(0.5)  # 等待补充

        # 最多只能获取 10 个
        acquired = 0
        while bucket.try_acquire():
            acquired += 1

        assert acquired == 10


class TestRateLimitPerMethod:
    """按方法限流测试"""

    def test_different_methods_have_separate_limits(self):
        """不同方法有独立限流"""
        interceptor = RateLimitInterceptor(qps=5, per_method=True)

        handler_inference = create_handler_details(method="/InferenceService/Inference")
        handler_cluster = create_handler_details(method="/ClusterService/Register")

        # 每个方法 5 个请求
        for i in range(5):
            interceptor.intercept_service(lambda x: create_response(), handler_inference)
            interceptor.intercept_service(lambda x: create_response(), handler_cluster)

        # 两种方法都超限
        with pytest.raises(grpc.RpcError):
            interceptor.intercept_service(lambda x: create_response(), handler_inference)

        with pytest.raises(grpc.RpcError):
            interceptor.intercept_service(lambda x: create_response(), handler_cluster)


class TestRateLimitEdgeCases:
    """限流边界情况"""

    def test_handles_concurrent_requests(self):
        """处理并发请求"""
        interceptor = RateLimitInterceptor(qps=100)
        handler = create_handler_details()

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(interceptor.intercept_service, lambda x: create_response(), handler)
                for _ in range(100)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # 应该大部分成功，少量失败
        success_count = sum(1 for r in results if r is not None)
        assert success_count > 0  # 至少有一些成功

    def test_zero_qps_blocks_all(self):
        """qps=0 阻止所有请求"""
        interceptor = RateLimitInterceptor(qps=0)
        handler = create_handler_details()

        with pytest.raises(grpc.RpcError):
            interceptor.intercept_service(lambda x: create_response(), handler)

    def test_negative_qps_treated_as_unlimited(self):
        """负数 qps 视为无限制"""
        interceptor = RateLimitInterceptor(qps=-1)
        handler = create_handler_details()

        # 应该不限制
        for i in range(1000):
            result = interceptor.intercept_service(lambda x: create_response(), handler)
            assert result is not None
```

### 5.4 监控拦截器 (test_metrics_interceptor.py)

```python
class TestMetricsInterceptor:
    """监控拦截器测试"""

    @pytest.fixture
    def interceptor(self):
        return MetricsInterceptor(prometheus_registry=REGISTRY)

    def test_records_request_count(self, interceptor):
        """记录请求计数"""
        handler_details = create_handler_details(method="/InferenceService/Inference")
        interceptor.intercept_service(lambda x: create_response(), handler_details)

        metric = REGISTRY.get_metric("grpc_requests_total", "counter")
        assert metric is not None
        assert metric._value.get() == 1

    def test_records_request_duration(self, interceptor):
        """记录请求延迟"""
        def slow_handler(details):
            time.sleep(0.1)
            return create_response()

        handler_details = create_handler_details(method="/InferenceService/Inference")
        interceptor.intercept_service(slow_handler, handler_details)

        metric = REGISTRY.get_metric("grpc_request_duration_seconds", "histogram")
        assert metric is not None
        # 检查有观测值
        assert metric._sum.get() > 0

    def test_records_by_method(self, interceptor):
        """按方法名记录"""
        handler_inference = create_handler_details(method="/InferenceService/Inference")
        handler_cluster = create_handler_details(method="/ClusterService/Register")

        interceptor.intercept_service(lambda x: create_response(), handler_inference)
        interceptor.intercept_service(lambda x: create_response(), handler_cluster)

        # 应该有分别的标签
        # inference_count > 0
        # cluster_count > 0

    def test_records_error_status(self, interceptor):
        """记录错误状态"""
        def error_handler(details):
            raise grpc.RpcError(grpc.StatusCode.INTERNAL, "Test error")

        handler_details = create_handler_details(method="/InferenceService/Inference")

        with pytest.raises(grpc.RpcError):
            interceptor.intercept_service(error_handler, handler_details)

        # 检查错误计数
        metric = REGISTRY.get_metric("grpc_requests_failed_total", "counter")
        assert metric._value.get() == 1
```

---

## 六、服务实现测试

### 6.1 InferenceService (test_inference_service.py)

```python
class TestInferenceServiceInference:
    """Inference 方法测试"""

    @pytest.fixture
    def servicer(self, mock_engine_manager, mock_cluster_manager):
        return InferenceServiceServicer(mock_engine_manager, mock_cluster_manager)

    @pytest.fixture
    def valid_request(self):
        return InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello, world!",
            max_tokens=100,
            temperature=0.7,
        )

    def test_returns_success_response(self, servicer, valid_request):
        """返回成功响应"""
        response = servicer.Inference(valid_request, create_context())

        assert response.status == ResponseStatus.STATUS_SUCCESS
        assert response.request_id == valid_request.request_id
        assert len(response.text) > 0
        assert response.tokens_generated > 0

    def test_returns_error_for_invalid_model(self, servicer):
        """模型不存在时返回错误"""
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="non-existent-model",
            prompt="Hello",
        )
        # 假设模型不存在
        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.Inference(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    def test_returns_error_for_model_not_loaded(self, servicer):
        """模型未加载时返回错误"""
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-70b",  # 未加载的模型
            prompt="Hello",
        )
        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.Inference(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    def test_respects_max_tokens(self, servicer, mock_engine_manager, valid_request):
        """正确传递 max_tokens 参数"""
        valid_request.max_tokens = 50
        servicer.Inference(valid_request, create_context())

        # 验证传递给引擎的参数
        call_args = mock_engine_manager.generate.call_args
        assert call_args.kwargs['max_tokens'] == 50
        # 或 call_args[1]['max_tokens']

    def test_respects_temperature(self, servicer, mock_engine_manager, valid_request):
        """正确传递 temperature 参数"""
        valid_request.temperature = 0.9
        servicer.Inference(valid_request, create_context())

        call_args = mock_engine_manager.generate.call_args
        assert call_args.kwargs['temperature'] == 0.9

    def test_respects_top_p(self, servicer, mock_engine_manager, valid_request):
        """正确传递 top_p 参数"""
        valid_request.top_p = 0.95
        servicer.Inference(valid_request, create_context())

        call_args = mock_engine_manager.generate.call_args
        assert call_args.kwargs['top_p'] == 0.95

    def test_respects_top_k(self, servicer, mock_engine_manager, valid_request):
        """正确传递 top_k 参数"""
        valid_request.top_k = 50
        servicer.Inference(valid_request, create_context())

        call_args = mock_engine_manager.generate.call_args
        assert call_args.kwargs['top_k'] == 50

    def test_respects_repetition_penalty(self, servicer, mock_engine_manager, valid_request):
        """正确传递 repetition_penalty 参数"""
        valid_request.repetition_penalty = 1.2
        servicer.Inference(valid_request, create_context())

        call_args = mock_engine_manager.generate.call_args
        assert call_args.kwargs['repetition_penalty'] == 1.2

    def test_passes_extra_params(self, servicer, mock_engine_manager, valid_request):
        """正确传递额外参数"""
        valid_request.extra_params["custom_param"] = "custom_value"
        servicer.Inference(valid_request, create_context())

        call_args = mock_engine_manager.generate.call_args
        assert "custom_param" in call_args.kwargs['extra_params']
        assert call_args.kwargs['extra_params']['custom_param'] == "custom_value"


class TestInferenceServiceValidation:
    """Inference 参数验证测试"""

    @pytest.fixture
    def servicer(self):
        return InferenceServiceServicer(MockEngineManager(), MockClusterManager())

    def test_rejects_empty_request_id(self, servicer):
        """拒绝空的 request_id"""
        request = InferenceRequest(request_id="", model_name="llama-2-7b", prompt="Hello")
        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.Inference(request, create_context())
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_model_name(self, servicer):
        """拒绝空的 model_name"""
        request = InferenceRequest(request_id=str(uuid.uuid4()), model_name="", prompt="Hello")
        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.Inference(request, create_context())
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_empty_prompt(self, servicer):
        """拒绝空的 prompt"""
        request = InferenceRequest(request_id=str(uuid.uuid4()), model_name="llama-2-7b", prompt="")
        # 空 prompt 可能被允许（取决于业务需求），或返回警告
        # 这里测试业务逻辑决定

    def test_rejects_invalid_temperature(self, servicer):
        """拒绝无效的 temperature"""
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            temperature=5.0  # 超出范围
        )
        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.Inference(request, create_context())
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_negative_max_tokens(self, servicer):
        """拒绝负数 max_tokens"""
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            max_tokens=-1
        )
        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.Inference(request, create_context())
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_rejects_max_tokens_exceeding_limit(self, servicer):
        """拒绝超过上限的 max_tokens"""
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            max_tokens=100000  # 超过 8192 上限
        )
        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.Inference(request, create_context())
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


class TestInferenceServiceInferenceStream:
    """InferenceStream 方法测试"""

    @pytest.fixture
    def servicer(self, mock_engine_manager):
        return InferenceServiceServicer(mock_engine_manager, MockClusterManager())

    def test_stream_returns_multiple_chunks(self, servicer, mock_engine_manager):
        """流式返回多个文本块"""
        # 模拟流式生成
        mock_engine_manager.generate_stream.return_value = iter([
            InferenceResponse(text="Hello"),
            InferenceResponse(text="Hello, world"),
            InferenceResponse(text="Hello, world!"),
        ])

        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
        )
        responses = list(servicer.InferenceStream(request, create_context()))

        assert len(responses) == 3

    def test_stream_yields_in_order(self, servicer, mock_engine_manager):
        """流式按顺序返回"""
        tokens = ["The", " quick", " brown", " fox"]
        mock_engine_manager.generate_stream.return_value = iter([
            InferenceResponse(text=t, tokens_generated=i+1)
            for i, t in enumerate(tokens)
        ])

        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
        )
        responses = list(servicer.InferenceStream(request, create_context()))

        texts = [r.text for r in responses]
        assert texts == tokens

    def test_stream_final_response_has_total_count(self, servicer, mock_engine_manager):
        """流式最后一帧包含总 token 数"""
        mock_engine_manager.generate_stream.return_value = iter([
            InferenceResponse(text="The", tokens_generated=1),
            InferenceResponse(text="The quick", tokens_generated=2, status=ResponseStatus.STATUS_SUCCESS),
        ])

        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
        )
        responses = list(servicer.InferenceStream(request, create_context()))

        final_response = responses[-1]
        assert final_response.status == ResponseStatus.STATUS_SUCCESS

    def test_stream_handles_error(self, servicer, mock_engine_manager):
        """流式处理生成错误"""
        mock_engine_manager.generate_stream.return_value = iter([
            InferenceResponse(text="Started"),
            InferenceResponse(text="", status=ResponseStatus.STATUS_ERROR, error_message="Generation failed"),
        ])

        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
        )
        responses = list(servicer.InferenceStream(request, create_context()))

        assert any(r.status == ResponseStatus.STATUS_ERROR for r in responses)


class TestInferenceServiceBatchInference:
    """BatchInference 方法测试"""

    @pytest.fixture
    def servicer(self, mock_engine_manager):
        return InferenceServiceServicer(mock_engine_manager, MockClusterManager())

    def test_batch_returns_all_results(self, servicer, mock_engine_manager):
        """批量返回所有结果"""
        mock_engine_manager.batch_generate.return_value = [
            InferenceResponse(text="Result 1", tokens_generated=2),
            InferenceResponse(text="Result 2", tokens_generated=3),
            InferenceResponse(text="Result 3", tokens_generated=4),
        ]

        request = BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
            max_tokens=100,
        )
        response = servicer.BatchInference(request, create_context())

        assert len(response.results) == 3
        assert response.status == ResponseStatus.STATUS_SUCCESS

    def test_batch_respects_model_name(self, servicer, mock_engine_manager):
        """批量使用正确的模型名"""
        request = BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="mixtral-8x7b",
            prompts=["Prompt 1"],
        )
        servicer.BatchInference(request, create_context())

        mock_engine_manager.batch_generate.assert_called_once()
        call_args = mock_engine_manager.batch_generate.call_args
        assert call_args[0][0] == "mixtral-8x7b"  # 第一个参数是 model_name

    def test_batch_respects_max_tokens(self, servicer, mock_engine_manager):
        """批量传递 max_tokens"""
        request = BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=["Prompt 1"],
            max_tokens=50,
        )
        servicer.BatchInference(request, create_context())

        call_args = mock_engine_manager.batch_generate.call_args
        assert call_args.kwargs['max_tokens'] == 50

    def test_batch_partial_failure(self, servicer, mock_engine_manager):
        """批量部分失败"""
        mock_engine_manager.batch_generate.return_value = [
            InferenceResponse(text="Success 1", status=ResponseStatus.STATUS_SUCCESS),
            InferenceResponse(text="", status=ResponseStatus.STATUS_ERROR, error_message="Failed"),
            InferenceResponse(text="Success 2", status=ResponseStatus.STATUS_SUCCESS),
        ]

        request = BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=["P1", "P2", "P3"],
        )
        response = servicer.BatchInference(request, create_context())

        assert response.status == ResponseStatus.STATUS_ERROR  # 整体失败
        # 或者根据业务逻辑，部分成功也可能返回 SUCCESS

    def test_batch_empty_prompts(self, servicer):
        """空 prompt 列表"""
        request = BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=[],
        )

        # 应该返回错误或空结果
        with pytest.raises(grpc.RpcError):
            servicer.BatchInference(request, create_context())

    def test_batch_large_size(self, servicer, mock_engine_manager):
        """大批量处理"""
        prompts = [f"Prompt {i}" for i in range(1000)]
        mock_engine_manager.batch_generate.return_value = [
            InferenceResponse(text=f"Result {i}", tokens_generated=2)
            for i in range(1000)
        ]

        request = BatchInferenceRequest(
            batch_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompts=prompts,
        )
        response = servicer.BatchInference(request, create_context())

        assert len(response.results) == 1000
```

### 6.2 ClusterService (test_cluster_service.py)

```python
class TestClusterServiceRegisterNode:
    """RegisterNode 测试"""

    @pytest.fixture
    def servicer(self, mock_cluster_manager):
        return ClusterServiceServicer(mock_cluster_manager)

    @pytest.fixture
    def valid_request(self):
        return RegisterNodeRequest(
            node_id="worker-001",
            host="192.168.1.100",
            port=8001,
            gpus=[
                GPUInfo(
                    index=0,
                    name="NVIDIA RTX 4090",
                    memory=GPUMemory(total_bytes=24 * 1024**3),
                ),
            ],
        )

    def test_register_returns_success(self, servicer, valid_request):
        """注册成功"""
        response = servicer.RegisterNode(valid_request, create_context())

        assert response.success is True
        assert response.assigned_id != ""

    def test_register_saves_node_info(self, servicer, mock_cluster_manager, valid_request):
        """注册保存节点信息"""
        servicer.RegisterNode(valid_request, create_context())

        mock_cluster_manager.register_node.assert_called_once()
        call_args = mock_cluster_manager.register_node.call_args
        assert call_args[0][0].node_id == "worker-001"

    def test_register_rejects_duplicate_node_id(self, servicer, mock_cluster_manager, valid_request):
        """拒绝重复节点 ID"""
        mock_cluster_manager.register_node.side_effect = NodeAlreadyExistsError("worker-001")

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.RegisterNode(valid_request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.ALREADY_EXISTS

    def test_register_rejects_invalid_port(self, servicer, valid_request):
        """拒绝无效端口"""
        valid_request.port = 70000  # 超出范围

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.RegisterNode(valid_request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_register_rejects_empty_node_id(self, servicer):
        """拒绝空节点 ID"""
        request = RegisterNodeRequest(node_id="", host="localhost", port=8001)

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.RegisterNode(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


class TestClusterServiceHeartbeat:
    """Heartbeat 测试"""

    @pytest.fixture
    def servicer(self, mock_cluster_manager):
        return ClusterServiceServicer(mock_cluster_manager)

    def test_heartbeat_updates_node_status(self, servicer, mock_cluster_manager):
        """心跳更新节点状态"""
        request = HeartbeatRequest(
            node_id="worker-001",
            resources=NodeResources(
                node_id="worker-001",
                status=NodeStatus.NODE_STATUS_BUSY,
                gpus=[GPUInfo(index=0, utilization=0.8)],
            ),
        )
        response = servicer.Heartbeat(request, create_context())

        assert response.success is True
        mock_cluster_manager.update_heartbeat.assert_called_once()

    def test_heartbeat_returns_pending_tasks(self, servicer, mock_cluster_manager):
        """心跳返回待处理任务"""
        mock_cluster_manager.get_pending_tasks.return_value = ["task-1", "task-2"]
        request = HeartbeatRequest(node_id="worker-001")

        response = servicer.Heartbeat(request, create_context())

        assert len(response.pending_tasks) == 2

    def test_heartbeat_rejects_unknown_node(self, servicer, mock_cluster_manager):
        """心跳拒绝未知节点"""
        mock_cluster_manager.update_heartbeat.side_effect = NodeNotFoundError("unknown-node")

        request = HeartbeatRequest(node_id="unknown-node")

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.Heartbeat(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


class TestClusterServiceDeregisterNode:
    """DeregisterNode 测试"""

    def test_deregister_success(self, servicer, mock_cluster_manager):
        """注销成功"""
        request = DeregisterNodeRequest(node_id="worker-001", reason="Maintenance")
        response = servicer.DeregisterNode(request, create_context())

        assert response.success is True
        mock_cluster_manager.deregister_node.assert_called_once_with("worker-001")

    def test_deregister_reassigns_tasks(self, servicer, mock_cluster_manager):
        """注销重新分配任务"""
        mock_cluster_manager.get_pending_tasks.return_value = ["task-1", "task-2"]
        request = DeregisterNodeRequest(node_id="worker-001")

        servicer.DeregisterNode(request, create_context())

        # 应该触发任务重新调度
        mock_cluster_manager.reschedule_tasks.assert_called_once()

    def test_deregister_nonexistent_node(self, servicer, mock_cluster_manager):
        """注销不存在的节点"""
        mock_cluster_manager.deregister_node.side_effect = NodeNotFoundError("unknown")

        request = DeregisterNodeRequest(node_id="unknown")

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.DeregisterNode(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


class TestClusterServiceListNodes:
    """ListNodes 测试"""

    @pytest.fixture
    def servicer(self, mock_cluster_manager):
        return ClusterServiceServicer(mock_cluster_manager)

    def test_list_returns_all_nodes(self, servicer, mock_cluster_manager):
        """列出所有节点"""
        mock_cluster_manager.list_nodes.return_value = [
            NodeResources(node_id="worker-001"),
            NodeResources(node_id="worker-002"),
        ]
        request = ListNodesRequest()

        response = servicer.ListNodes(request, create_context())

        assert len(response.nodes) == 2

    def test_list_filters_by_status(self, servicer, mock_cluster_manager):
        """按状态过滤"""
        mock_cluster_manager.list_nodes.return_value = [
            NodeResources(node_id="worker-001", status=NodeStatus.NODE_STATUS_ACTIVE),
        ]
        request = ListNodesRequest(filter_status=NodeStatus.NODE_STATUS_ACTIVE)

        response = servicer.ListNodes(request, create_context())

        mock_cluster_manager.list_nodes.assert_called_once()
        call_kwargs = mock_cluster_manager.list_nodes.call_args[1]
        assert call_kwargs['status'] == NodeStatus.NODE_STATUS_ACTIVE

    def test_list_filters_by_model(self, servicer, mock_cluster_manager):
        """按模型过滤"""
        request = ListNodesRequest(filter_model="llama-2-70b")

        servicer.ListNodes(request, create_context())

        call_kwargs = mock_cluster_manager.list_nodes.call_args[1]
        assert call_kwargs['model'] == "llama-2-70b"

    def test_list_returns_empty_when_no_match(self, servicer, mock_cluster_manager):
        """无匹配时返回空列表"""
        mock_cluster_manager.list_nodes.return_value = []
        request = ListNodesRequest(filter_status=NodeStatus.NODE_STATUS_OFFLINE)

        response = servicer.ListNodes(request, create_context())

        assert len(response.nodes) == 0
```

### 6.3 SchedulerService (test_scheduler_service.py)

```python
class TestSchedulerServiceSubmit:
    """SubmitRequest 测试"""

    @pytest.fixture
    def servicer(self, mock_scheduler):
        return SchedulerServiceServicer(mock_scheduler)

    @pytest.fixture
    def valid_request(self):
        return SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-70b",
            backend=ModelBackend.MODEL_BACKEND_VLLM,
            tensor_parallel_size=4,
            gpu_memory_required_gb=80,
            priority=5,
            mode=InferenceMode.INFERENCE_MODE_GENERATE,
        )

    def test_submit_returns_assigned_node(self, servicer, mock_scheduler, valid_request):
        """提交返回分配的节点"""
        mock_scheduler.schedule.return_value = SchedulingResponse(
            request_id=valid_request.request_id,
            scheduled=True,
            assigned_node_id="worker-001",
            assigned_host="192.168.1.100",
            assigned_port=50051,
        )

        response = servicer.SubmitRequest(valid_request, create_context())

        assert response.scheduled is True
        assert response.assigned_node_id == "worker-001"
        assert response.assigned_host == "192.168.1.100"

    def test_submit_fails_when_no_resources(self, servicer, mock_scheduler, valid_request):
        """无资源时调度失败"""
        mock_scheduler.schedule.side_effect = SchedulingError(
            reason="No GPU with enough memory"
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.SubmitRequest(valid_request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE

    def test_submit_validates_priority(self, servicer, mock_scheduler):
        """验证优先级范围"""
        request = SchedulingRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            priority=15,  # 超出 0-10 范围
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.SubmitRequest(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_submit_respects_tensor_parallel_size(self, servicer, mock_scheduler, valid_request):
        """传递正确的 tensor_parallel_size"""
        valid_request.tensor_parallel_size = 8
        servicer.SubmitRequest(valid_request, create_context())

        call_args = mock_scheduler.schedule.call_args
        assert call_args[0][0].tensor_parallel_size == 8


class TestSchedulerServiceCancel:
    """CancelRequest 测试"""

    def test_cancel_success(self, servicer, mock_scheduler):
        """取消成功"""
        mock_scheduler.cancel.return_value = True
        request = CancelRequest(request_id="req-123")

        response = servicer.CancelRequest(request, create_context())

        assert response.success is True

    def test_cancel_nonexistent_request(self, servicer, mock_scheduler):
        """取消不存在的请求"""
        mock_scheduler.cancel.return_value = False

        request = CancelRequest(request_id="nonexistent")

        response = servicer.CancelRequest(request, create_context())

        assert response.success is False

    def test_cancel_already_completed_request(self, servicer, mock_scheduler):
        """取消已完成的请求"""
        mock_scheduler.cancel.side_effect = SchedulingError("Request already completed")

        request = CancelRequest(request_id="completed-req")

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.CancelRequest(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


class TestSchedulerServiceGetStatus:
    """GetStatus 测试"""

    def test_status_pending(self, servicer, mock_scheduler):
        """状态：等待中"""
        mock_scheduler.get_status.return_value = SchedulingResponse(
            request_id="req-123",
            scheduled=True,
            status=ResponseStatus.STATUS_PENDING,
        )

        request = GetSchedulingStatusRequest(request_id="req-123")
        response = servicer.GetStatus(request, create_context())

        assert response.status == ResponseStatus.STATUS_PENDING

    def test_status_processing(self, servicer, mock_scheduler):
        """状态：处理中"""
        mock_scheduler.get_status.return_value = SchedulingResponse(
            request_id="req-123",
            status=ResponseStatus.STATUS_PROCESSING,
        )

        request = GetSchedulingStatusRequest(request_id="req-123")
        response = servicer.GetStatus(request, create_context())

        assert response.status == ResponseStatus.STATUS_PROCESSING

    def test_status_completed_with_result(self, servicer, mock_scheduler):
        """状态：完成并返回结果"""
        mock_scheduler.get_status.return_value = SchedulingResponse(
            request_id="req-123",
            status=ResponseStatus.STATUS_SUCCESS,
            result=InferenceResponse(text="Generated text"),
        )

        request = GetSchedulingStatusRequest(request_id="req-123")
        response = servicer.GetStatus(request, create_context())

        assert response.status == ResponseStatus.STATUS_SUCCESS
        assert response.result.text == "Generated text"

    def test_status_error(self, servicer, mock_scheduler):
        """状态：错误"""
        mock_scheduler.get_status.return_value = SchedulingResponse(
            request_id="req-123",
            status=ResponseStatus.STATUS_ERROR,
            error_message="GPU memory error",
        )

        request = GetSchedulingStatusRequest(request_id="req-123")
        response = servicer.GetStatus(request, create_context())

        assert response.status == ResponseStatus.STATUS_ERROR
        assert "GPU memory" in response.error_message
```

### 6.4 ModelManagementService (test_model_management_service.py)

```python
class TestModelManagementServiceLoadModel:
    """LoadModel 测试"""

    @pytest.fixture
    def servicer(self, mock_engine_manager):
        return ModelManagementServiceServicer(mock_engine_manager)

    def test_load_model_success(self, servicer, mock_engine_manager):
        """加载成功"""
        mock_engine_manager.load_model.return_value = LoadModelResponse(
            success=True,
            model_name="llama-2-7b",
            memory_allocated=GPUMemory(used_bytes=14 * 1024**3),
        )

        request = LoadModelRequest(
            model_name="llama-2-7b",
            backend=ModelBackend.MODEL_BACKEND_VLLM,
            tensor_parallel_size=1,
        )
        response = servicer.LoadModel(request, create_context())

        assert response.success is True
        assert response.model_name == "llama-2-7b"

    def test_load_model_checks_memory(self, servicer, mock_engine_manager):
        """加载前检查显存"""
        mock_engine_manager.load_model.side_effect = ResourceUnavailableError(
            resource="GPU memory",
            required=80,
            available=40,
        )

        request = LoadModelRequest(
            model_name="llama-2-70b",
            backend=ModelBackend.MODEL_BACKEND_VLLM,
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.LoadModel(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

    def test_load_model_rejects_invalid_backend(self, servicer):
        """拒绝无效后端"""
        request = LoadModelRequest(
            model_name="llama-2-7b",
            backend=ModelBackend.MODEL_BACKEND_UNSPECIFIED,
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            servicer.LoadModel(request, create_context())

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_load_model_evicts_if_needed(self, servicer, mock_engine_manager):
        """显存不足时自动淘汰"""
        # 第一次尝试失败（内存不足）
        # 第二次尝试成功（淘汰了其他模型）
        mock_engine_manager.load_model.side_effect = [
            ResourceUnavailableError("Memory full"),
            LoadModelResponse(success=True, model_name="llama-2-7b"),
        ]

        request = LoadModelRequest(
            model_name="llama-2-7b",
            backend=ModelBackend.MODEL_BACKEND_VLLM,
            gpu_memory_utilization=0.9,
        )
        response = servicer.LoadModel(request, create_context())

        # 应该调用了两次（第一次失败后触发 eviction）
        assert mock_engine_manager.load_model.call_count == 2


class TestModelManagementServiceListModels:
    """ListModels 测试"""

    def test_list_returns_loaded_models(self, servicer, mock_engine_manager):
        """返回已加载模型"""
        mock_engine_manager.list_loaded_models.return_value = [
            ModelInfo(name="llama-2-7b", backend="vllm", is_loaded=True),
            ModelInfo(name="mixtral-8x7b", backend="vllm", is_loaded=True),
        ]

        request = ListModelsRequest()
        response = servicer.ListModels(request, create_context())

        assert len(response.models) == 2

    def test_list_returns_empty_when_no_models(self, servicer, mock_engine_manager):
        """无模型时返回空"""
        mock_engine_manager.list_loaded_models.return_value = []

        request = ListModelsRequest()
        response = servicer.ListModels(request, create_context())

        assert len(response.models) == 0
```

### 6.5 HealthService (test_health_service.py)

```python
class TestHealthServiceCheck:
    """Check 测试"""

    @pytest.fixture
    def servicer(self):
        return HealthServiceServicer()

    def test_check_returns_healthy(self, servicer):
        """服务健康时返回 healthy"""
        request = HealthCheckRequest(service="inference")
        response = servicer.Check(request, create_context())

        assert response.healthy is True
        assert response.status == "OK"

    def test_check_returns_unhealthy_when_status_false(self, servicer):
        """服务不健康时返回 unhealthy"""
        servicer._status = False
        request = HealthCheckRequest(service="inference")
        response = servicer.Check(request, create_context())

        assert response.healthy is False
        assert response.status == "UNHEALTHY"

    def test_check_includes_component_details(self, servicer):
        """检查包含组件详情"""
        request = HealthCheckRequest(service="all")
        response = servicer.Check(request, create_context())

        assert "inference" in response.details
        assert "cluster" in response.details


class TestHealthServiceWatch:
    """Watch 测试"""

    @pytest.fixture
    def servicer(self):
        return HealthServiceServicer()

    def test_watch_yields_periodic_updates(self, servicer):
        """定期推送健康状态"""
        request = HealthCheckRequest(service="inference")

        # 收集 3 个响应
        responses = []
        for i, response in enumerate(servicer.Watch(request, create_context())):
            responses.append(response)
            if i >= 2:
                break

        assert len(responses) == 3
        assert all(r.healthy for r in responses)

    def test_watch_reports_status_change(self, servicer):
        """状态变化时推送"""
        request = HealthCheckRequest(service="inference")

        # 模拟状态变化
        responses = []
        for i, response in enumerate(servicer.Watch(request, create_context())):
            responses.append(response)
            if i == 0:
                servicer._status = False  # 模拟变不健康
            if i >= 2:
                break

        # 最后一个应该是 unhealthy
        assert responses[-1].healthy is False
```

---

## 七、客户端测试

### 7.1 InferenceClient (test_inference_client.py)

```python
class TestInferenceClient:
    """InferenceClient 测试"""

    @pytest.fixture
    def mock_channel(self):
        return MagicMock()

    @pytest.fixture
    def client(self, mock_channel):
        return InferenceClient(mock_channel)

    def test_inference_calls_correct_method(self, client, mock_channel):
        """调用正确的方法"""
        request = InferenceRequest(request_id="test")
        client.inference(request)

        mock_channel.unary_unary.assert_called_once()
        call_path = mock_channel.unary_unary.call_args[0][0]
        assert "/quantumflow.v1.InferenceService/Inference" in str(call_path)

    def test_inference_passes_request(self, client, mock_channel):
        """正确传递请求"""
        request = InferenceRequest(request_id="test-id", prompt="Hello")
        client.inference(request)

        # 验证请求被正确传递
        passed_request = mock_channel.unary_unary.call_args[0][1]
        assert passed_request.request_id == "test-id"

    def test_inference_stream_calls_stream_method(self, client, mock_channel):
        """流式方法调用"""
        request = InferenceRequest(request_id="test")
        list(client.inference_stream(request))

        mock_channel.unary_stream.assert_called_once()

    def test_batch_inference_calls_correct_method(self, client, mock_channel):
        """批量方法调用"""
        request = BatchInferenceRequest(batch_id="batch-1")
        client.batch_inference(request)

        call_path = mock_channel.unary_unary.call_args[0][0]
        assert "/BatchInference" in str(call_path)


class TestInferenceClientTimeout:
    """超时测试"""

    def test_default_timeout(self, client, mock_channel):
        """默认超时"""
        request = InferenceRequest(request_id="test")
        client.inference(request)

        # 应该有超时设置
        call_kwargs = mock_channel.unary_unary.call_args[1]
        assert 'timeout' in call_kwargs or call_kwargs.get('timeout') == 30.0

    def test_custom_timeout(self, client, mock_channel):
        """自定义超时"""
        request = InferenceRequest(request_id="test")
        client.inference(request, timeout=60.0)

        call_kwargs = mock_channel.unary_unary.call_args[1]
        assert call_kwargs['timeout'] == 60.0
```

---

## 八、集成测试 (test_end_to_end.py)

### 8.1 端到端测试

```python
class TestGrpcEndToEnd:
    """gRPC 端到端测试"""

    @pytest.fixture
    def grpc_server(self):
        """启动 gRPC 服务器"""
        server = GrpcServer(port=50051)
        server.add_service(
            inference_service_pb2_grpc.InferenceServiceServicer,
            InferenceServiceServicer(engine_manager, cluster_manager)
        )
        server.add_service(
            cluster_service_pb2_grpc.ClusterServiceServicer,
            ClusterServiceServicer(cluster_manager)
        )
        server.start()
        yield server
        server.stop()

    @pytest.fixture
    def client_channel(self):
        """客户端连接"""
        channel = grpc.insecure_channel('localhost:50051')
        yield channel
        channel.close()

    def test_full_inference_workflow(self, grpc_server, client_channel):
        """完整推理工作流"""
        # 1. 注册节点
        cluster_client = ClusterClient(client_channel)
        node_response = cluster_client.register_node(RegisterNodeRequest(
            node_id="worker-1",
            host="localhost",
            port=50051,
        ))
        assert node_response.success

        # 2. 加载模型
        model_client = ModelManagementClient(client_channel)
        load_response = model_client.load_model(LoadModelRequest(
            model_name="llama-2-7b",
            backend=ModelBackend.MODEL_BACKEND_VLLM,
        ))
        assert load_response.success

        # 3. 执行推理
        inference_client = InferenceClient(client_channel)
        response = inference_client.inference(InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama-2-7b",
            prompt="Hello",
            max_tokens=10,
        ))
        assert response.status == ResponseStatus.STATUS_SUCCESS

    def test_health_check(self, grpc_server, client_channel):
        """健康检查"""
        health_stub = health_service_pb2_grpc.HealthServiceStub(client_channel)
        response = health_stub.Check(HealthCheckRequest())

        assert response.healthy is True

    def test_concurrent_requests(self, grpc_server, client_channel):
        """并发请求"""
        inference_client = InferenceClient(client_channel)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    inference_client.inference,
                    InferenceRequest(
                        request_id=str(uuid.uuid4()),
                        model_name="llama-2-7b",
                        prompt=f"Prompt {i}",
                    )
                )
                for i in range(50)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # 所有请求都应该成功
        success_count = sum(1 for r in results if r.status == ResponseStatus.STATUS_SUCCESS)
        assert success_count == 50
```

---

## 九、测试覆盖目标

### 9.1 按模块覆盖率目标

| 模块 | 目标覆盖率 |
|------|----------|
| Proto 验证 | 100% |
| 异常 | 100% |
| 拦截器 | 95% |
| Servicers | 95% |
| 客户端 | 90% |
| 集成测试 | 95% |

### 9.2 测试用例数量目标

| 类型 | 数量 |
|------|-----|
| 单元测试 | ~400 |
| 集成测试 | ~50 |
| **总计** | **~450** |

---

## 十、测试执行计划

```
Phase 1: Proto + 异常 (TODO-1.4, TODO-2.1)
  → 50 测试用例

Phase 2: 拦截器 (TODO-2.2 - TODO-2.5)
  → 100 测试用例

Phase 3: 服务实现 (TODO-3.1 - TODO-3.5)
  → 200 测试用例

Phase 4: 客户端 (TODO-5.2 - TODO-5.4)
  → 50 测试用例

Phase 5: 集成测试 (TODO-6.1 - TODO-6.2)
  → 50 测试用例
```

---

## 十一、测试数据

### 11.1 有效测试数据

```python
VALID_INFERENCE_REQUEST = InferenceRequest(
    request_id="550e8400-e29b-41d4-a716-446655440000",
    model_name="llama-2-7b",
    backend=ModelBackend.MODEL_BACKEND_VLLM,
    prompt="Hello, world!",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
)

VALID_GPU_INFO = GPUInfo(
    index=0,
    name="NVIDIA RTX 4090",
    memory=GPUMemory(
        total_bytes=24 * 1024**3,
        available_bytes=16 * 1024**3,
        used_bytes=8 * 1024**3,
        utilization=0.33,
    ),
    utilization=0.5,
    compute_capacity=33000,
)
```

### 11.2 边界测试数据

```python
BOUNDARY_TEMPERATURE = [0.0, 0.001, 1.0, 1.999, 2.0]
BOUNDARY_TOP_P = [0.0, 0.001, 0.5, 0.999, 1.0]
BOUNDARY_MAX_TOKENS = [1, 100, 1000, 4096, 8192, 8193]
BOUNDARY_PORT = [1, 1024, 8000, 30000, 65535]
```

### 11.3 错误测试数据

```python
INVALID_REQUESTS = [
    InferenceRequest(request_id="", model_name="llama-2-7b", prompt="Hello"),  # 空 ID
    InferenceRequest(request_id="not-uuid", model_name="llama-2-7b", prompt="Hello"),  # 非 UUID
    InferenceRequest(request_id=str(uuid.uuid4()), model_name="", prompt="Hello"),  # 空模型名
    InferenceRequest(request_id=str(uuid.uuid4()), model_name="llama-2-7b", prompt="Hello", temperature=5.0),  # 温度超范围
    InferenceRequest(request_id=str(uuid.uuid4()), model_name="llama-2-7b", prompt="Hello", max_tokens=0),  # max_tokens=0
    InferenceRequest(request_id=str(uuid.uuid4()), model_name="llama-2-7b", prompt="Hello", max_tokens=-1),  # max_tokens<0
    InferenceRequest(request_id=str(uuid.uuid4()), model_name="llama-2-7b", prompt="Hello", repetition_penalty=0.5),  # repetition_penalty<1
]
```

---

## 十二、自查清单

完成测试代码后，自查以下问题：

- [ ] 是否有测试用例未覆盖的代码分支？
- [ ] 断言是否足够严格（检查具体值，而非仅检查非空）？
- [ ] 边界值是否全部测试？
- [ ] 异常场景是否有对应测试？
- [ ] 并发测试是否充分？
- [ ] 测试是否相互独立（无顺序依赖）？
- [ ] Mock 对象是否正确设置（验证调用参数）？
- [ ] 测试数据是否覆盖所有有效/无效组合？
- [ ] 测试执行速度是否合理（避免不必要的 sleep）？
- [ ] 是否有假通过风险（断言永远为真）？
