# QuantumFlow gRPC 开发计划书

## 一、现状分析

### 1.1 现有架构
- **通信机制**: HTTP REST API
- **分布式组件**: Controller (API Server) + WorkerNode + Redis Queue
- **调度系统**: DistributedScheduler (基于 Redis 队列)
- **依赖**: `grpcio` 已作为传递依赖安装

### 1.2 迁移目标
将 Controller ↔ Worker 之间的 HTTP 通信迁移到 gRPC，实现：
- 高性能二进制序列化 (Protocol Buffers)
- 双向流式通信
- 强类型 API 契约
- 拦截器支持（认证、日志、监控）

---

## 二、gRPC 服务设计

### 2.1 Proto 定义文件

**文件位置**: `quantumflow/grpc/proto/quantumflow.proto`

```protobuf
syntax = "proto3";

package quantumflow.v1;

option go_package = "quantumflow/v1;quantumflow";
option java_multiple_files = true;
option java_package = "com.quantumflow.grpc";

// ============ 枚举类型 ============

enum ModelBackend {
  MODEL_BACKEND_UNSPECIFIED = 0;
  MODEL_BACKEND_VLLM = 1;
  MODEL_BACKEND_HUGGINGFACE = 2;
  MODEL_BACKEND_TGI = 3;
  MODEL_BACKEND_SGLANG = 4;
}

enum InferenceMode {
  INFERENCE_MODE_UNSPECIFIED = 0;
  INFERENCE_MODE_GENERATE = 1;
  INFERENCE_MODE_STREAM = 2;
  INFERENCE_MODE_BATCH = 3;
}

enum ResponseStatus {
  STATUS_UNSPECIFIED = 0;
  STATUS_SUCCESS = 1;
  STATUS_PENDING = 2;
  STATUS_PROCESSING = 3;
  STATUS_ERROR = 4;
  STATUS_CANCELLED = 5;
}

enum NodeStatus {
  NODE_STATUS_UNSPECIFIED = 0;
  NODE_STATUS_ACTIVE = 1;
  NODE_STATUS_IDLE = 2;
  NODE_STATUS_BUSY = 3;
  NODE_STATUS_OFFLINE = 4;
}

// ============ 公共消息 ============

message GPUMemory {
  uint64 total_bytes = 1;
  uint64 available_bytes = 2;
  uint64 used_bytes = 3;
  float utilization = 4;
}

message GPUInfo {
  int32 index = 1;
  string name = 2;
  GPUMemory memory = 3;
  float utilization = 4;
  uint64 compute_capacity = 5;  // TFLOPS
}

message ModelInfo {
  string name = 1;
  string backend = 2;
  int64 size_bytes = 3;
  int32 tensor_parallel_size = 4;
  GPUMemory memory_required = 5;
  bool is_loaded = 6;
}

message NodeResources {
  string node_id = 1;
  string host = 2;
  int32 port = 3;
  repeated GPUInfo gpus = 4;
  repeated ModelInfo loaded_models = 5;
  NodeStatus status = 6;
  int64 last_heartbeat_timestamp = 7;
}

// ============ 推理服务 ============

message InferenceRequest {
  string request_id = 1;
  string model_name = 2;
  ModelBackend backend = 3;
  string prompt = 4;
  string stream = 5;  // "true" or "false"
  int32 max_tokens = 6;
  float temperature = 7;
  float top_p = 8;
  int32 top_k = 9;
  float repetition_penalty = 10;
  map<string, string> extra_params = 11;
}

message InferenceResponse {
  string request_id = 1;
  ResponseStatus status = 2;
  string text = 3;
  int32 tokens_generated = 4;
  float latency_ms = 5;
  string error_message = 6;
}

message BatchInferenceRequest {
  string batch_id = 1;
  string model_name = 2;
  ModelBackend backend = 3;
  repeated string prompts = 4;
  int32 max_tokens = 5;
  float temperature = 6;
  map<string, string> extra_params = 7;
}

message BatchInferenceResponse {
  string batch_id = 1;
  ResponseStatus status = 2;
  repeated InferenceResponse results = 3;
  float total_latency_ms = 4;
}

// 推理服务定义
service InferenceService {
  // 同步推理
  rpc Inference(InferenceRequest) returns (InferenceResponse);

  // 流式推理
  rpc InferenceStream(InferenceRequest) returns (stream InferenceResponse);

  // 批量推理
  rpc BatchInference(BatchInferenceRequest) returns (BatchInferenceResponse);
}

// ============ 集群管理服务 ============

message RegisterNodeRequest {
  string node_id = 1;
  string host = 2;
  int32 port = 3;
  repeated GPUInfo gpus = 4;
  map<string, string> capabilities = 5;
}

message RegisterNodeResponse {
  bool success = 1;
  string message = 2;
  string assigned_id = 3;
}

message HeartbeatRequest {
  string node_id = 1;
  NodeResources resources = 2;
}

message HeartbeatResponse {
  bool success = 1;
  int64 server_time = 2;
  repeated string pending_tasks = 3;
}

message DeregisterNodeRequest {
  string node_id = 1;
  string reason = 2;
}

message DeregisterNodeResponse {
  bool success = 1;
  string message = 1;
}

message ListNodesRequest {
  NodeStatus filter_status = 1;
  string filter_model = 2;
}

message ListNodesResponse {
  repeated NodeResources nodes = 1;
}

service ClusterService {
  // 节点注册
  rpc RegisterNode(RegisterNodeRequest) returns (RegisterNodeResponse);

  // 节点注销
  rpc DeregisterNode(DeregisterNodeRequest) returns (DeregisterNodeResponse);

  // 心跳
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);

  // 列出节点
  rpc ListNodes(ListNodesRequest) returns (ListNodesResponse);

  // 节点资源更新（推送）
  rpc UpdateNodeResources(stream NodeResources) returns (stream NodeResources);
}

// ============ 调度服务 ============

message SchedulingRequest {
  string request_id = 1;
  string model_name = 2;
  ModelBackend backend = 3;
  int32 tensor_parallel_size = 4;
  int32 gpu_memory_required_gb = 5;
  int32 priority = 6;
  InferenceMode mode = 7;
}

message SchedulingResponse {
  string request_id = 1;
  bool scheduled = 2;
  string assigned_node_id = 3;
  string assigned_host = 4;
  int32 assigned_port = 5;
  string error_message = 6;
}

message CancelRequest {
  string request_id = 1;
}

message CancelResponse {
  bool success = 1;
  string message = 2;
}

message GetSchedulingStatusRequest {
  string request_id = 1;
}

message GetSchedulingStatusResponse {
  string request_id = 1;
  ResponseStatus status = 2;
  InferenceResponse result = 3;
}

service SchedulerService {
  // 提交调度请求
  rpc SubmitRequest(SchedulingRequest) returns (SchedulingResponse);

  // 取消请求
  rpc CancelRequest(CancelRequest) returns (CancelResponse);

  // 获取调度状态
  rpc GetStatus(GetSchedulingStatusRequest) returns (GetSchedulingStatusResponse);

  // 取消调度请求（流式）
  rpc CancelRequestStream(stream CancelRequest) returns (stream CancelResponse);
}

// ============ 健康检查服务 ============

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  bool healthy = 1;
  string status = 2;
  map<string, string> details = 3;
}

service HealthService {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
}

// ============ 模型管理服务 ============

message LoadModelRequest {
  string model_name = 1;
  ModelBackend backend = 2;
  int32 tensor_parallel_size = 3;
  float gpu_memory_utilization = 4;
  map<string, string> backend_config = 5;
}

message LoadModelResponse {
  bool success = 1;
  string model_name = 2;
  string message = 3;
  GPUMemory memory_allocated = 4;
}

message UnloadModelRequest {
  string model_name = 1;
}

message UnloadModelResponse {
  bool success = 1;
  string message = 2;
  uint64 memory_freed_bytes = 3;
}

message ListModelsRequest {}

message ListModelsResponse {
  repeated ModelInfo models = 1;
}

service ModelManagementService {
  rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
  rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
}

// ============ 指标服务 ============

message MetricsRequest {
  string metric_names = 1;  // comma-separated
}

message MetricSample {
  string name = 1;
  double value = 2;
  int64 timestamp = 3;
  map<string, string> labels = 4;
}

message MetricsResponse {
  repeated MetricSample metrics = 1;
}

service MetricsService {
  rpc GetMetrics(MetricsRequest) returns (MetricsResponse);
  rpc StreamMetrics(MetricsRequest) returns (stream MetricsResponse);
}
```

---

## 三、开发任务分解 (TODO List)

### Phase 1: 基础设施

#### TODO-1.1: 创建项目目录结构
```python
# 创建目录
quantumflow/grpc/
├── proto/              # Proto 文件
│   ├── quantumflow.proto
│   └── generate.sh    # 生成脚本
├── generated/          # 生成的 Python 代码
│   ├── quantumflow_pb2.py
│   ├── quantumflow_pb2_grpc.py
│   └── __init__.py
├── interceptors/      # 拦截器
│   ├── __init__.py
│   ├── logging.py     # 日志拦截器
│   ├── auth.py         # 认证拦截器
│   ├── metrics.py      # 监控拦截器
│   └── rate_limit.py   # 限流拦截器
├── channels/          # Channel 管理
│   ├── __init__.py
│   └── pool.py         # 连接池
└── exceptions.py      # gRPC 异常定义
```
**完成标准**: 目录结构创建完成，所有 `__init__.py` 文件存在

#### TODO-1.2: 安装依赖
```python
# requirements.txt 添加
grpcio>=1.60.0
grpcio-tools>=1.60.0
grpcio-reflection>=1.60.0  # 用于健康检查
protobuf>=4.25.0
```
**完成标准**: `pip install -r requirements.txt` 成功

#### TODO-1.3: 生成 Proto 代码
```bash
# quantumflow/grpc/proto/generate.sh
#!/bin/bash
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./generated \
    --grpc_python_out=./generated \
    ./proto/quantumflow.proto
```
**完成标准**: `generated/` 目录生成 `*_pb2.py` 和 `*_pb2_grpc.py`

#### TODO-1.4: 验证生成代码
```python
# tests/unit/grpc/test_proto_generation.py
def test_proto_imports():
    from quantumflow.grpc.generated import quantumflow_pb2
    from quantumflow.grpc.generated import quantumflow_pb2_grpc
    # 验证所有消息类型可导入
    # 验证所有服务可导入
```
**完成标准**: 测试通过

---

### Phase 2: 异常和拦截器

#### TODO-2.1: 定义 gRPC 异常
```python
# quantumflow/grpc/exceptions.py
class GrpcQuantumFlowError(Exception):
    """gRPC 异常基类"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")

class NodeNotFoundError(GrpcQuantumFlowError):
    """节点未找到"""

class ModelNotLoadedError(GrpcQuantumFlowError):
    """模型未加载"""

class SchedulingError(GrpcQuantumFlowError):
    """调度失败"""

class ResourceUnavailableError(GrpcQuantumFlowError):
    """资源不可用"""
```
**完成标准**: 异常类完整，覆盖所有错误场景

#### TODO-2.2: 日志拦截器
```python
# quantumflow/grpc/interceptors/logging.py
class LoggingInterceptor(grpc.UnaryUnaryServerInterceptor):
    """日志拦截器 - 记录所有 gRPC 调用"""
    def __init__(self, logger):
        self.logger = logger

    def intercept_service(self, continuation, handler_call_details):
        # 记录: 方法名、调用者、耗时、状态码
        start_time = time.time()
        response = continuation(handler_call_details)
        duration = time.time() - start_time
        # 记录日志
        return response
```
**完成标准**: 所有 gRPC 调用都有日志记录

#### TODO-2.3: 认证拦截器
```python
# quantumflow/grpc/interceptors/auth.py
class AuthInterceptor(grpc.ServerInterceptor):
    """认证拦截器 - 验证 API Key/Token"""
    def __init__(self, allowed_tokens: Dict[str, str]):
        self.allowed_tokens = allowed_tokens  # token -> user_id

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)
        token = metadata.get('authorization', '').replace('Bearer ', '')
        if token not in self.allowed_tokens:
            raise grpc.RpcError(grpc.StatusCode.UNAUTHENTICATED)
        return continuation(handler_call_details)
```
**完成标准**: 无效 token 被拒绝

#### TODO-2.4: 监控拦截器
```python
# quantumflow/grpc/interceptors/metrics.py
class MetricsInterceptor(grpc.UnaryUnaryServerInterceptor):
    """监控拦截器 - 记录 QPS、延迟"""
    def intercept_service(self, continuation, handler_call_details):
        metric_name = f"grpc_{handler_call_details.method}"
        # 记录到 Prometheus
```
**完成标准**: 指标正确暴露

#### TODO-2.5: 限流拦截器
```python
# quantumflow/grpc/interceptors/rate_limit.py
class RateLimitInterceptor(grpc.ServerInterceptor):
    """限流拦截器 - 基于令牌桶"""
    def __init__(self, qps: int = 100):
        self.rate_limiter = TokenBucket(qps)

    def intercept_service(self, continuation, handler_call_details):
        if not self.rate_limiter.try_acquire():
            raise grpc.RpcError(grpc.StatusCode.RESOURCE_EXHAUSTED)
```
**完成标准**: 超过 QPS 返回 RESOURCE_EXHAUSTED

---

### Phase 3: 服务实现

#### TODO-3.1: InferenceService 实现
```python
# quantumflow/grpc/services/inference.py
class InferenceServiceServicer(inference_service_pb2_grpc.InferenceServiceServicer):
    """推理服务实现"""
    def __init__(self, engine_manager, cluster_manager):
        self.engine_manager = engine_manager
        self.cluster_manager = cluster_manager

    def Inference(self, request, context):
        """同步推理"""
        # 1. 验证请求
        # 2. 查找可用节点/引擎
        # 3. 执行推理
        # 4. 返回结果

    def InferenceStream(self, request, context):
        """流式推理"""
        # 1. 验证请求
        # 2. 查找可用节点/引擎
        # 3. yield 返回每个 token

    def BatchInference(self, request, context):
        """批量推理"""
        # 1. 验证请求
        # 2. 批量提交到引擎
        # 3. 返回所有结果
```
**完成标准**: 支持同步、流式、批量三种模式

#### TODO-3.2: ClusterService 实现
```python
# quantumflow/grpc/services/cluster.py
class ClusterServiceServicer(cluster_service_pb2_grpc.ClusterServiceServicer):
    """集群管理服务"""
    def __init__(self, cluster_manager):
        self.cluster_manager = cluster_manager

    def RegisterNode(self, request, context):
        """节点注册"""
        # 1. 验证节点信息
        # 2. 保存到 ClusterManager
        # 3. 返回分配的资源

    def Heartbeat(self, request, context):
        """心跳处理"""
        # 1. 更新节点状态
        # 2. 返回待处理任务

    def DeregisterNode(self, request, context):
        """节点注销"""
        # 1. 清理节点资源
        # 2. 重新调度任务

    def ListNodes(self, request, context):
        """列出节点"""
        # 1. 过滤条件
        # 2. 返回节点列表
```
**完成标准**: 注册/注销/心跳/列表全部正常工作

#### TODO-3.3: SchedulerService 实现
```python
# quantumflow/grpc/services/scheduler.py
class SchedulerServiceServicer(scheduler_service_pb2_grpc.SchedulerServiceServicer):
    """调度服务"""
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def SubmitRequest(self, request, context):
        """提交调度请求"""
        # 1. 验证请求
        # 2. 选择最优节点
        # 3. 分发任务
        # 4. 返回调度结果

    def GetStatus(self, request, context):
        """获取调度状态"""
        # 1. 查询任务状态
        # 2. 返回结果或进度

    def CancelRequest(self, request, context):
        """取消请求"""
        # 1. 查找任务
        # 2. 取消执行
```
**完成标准**: 调度、状态查询、取消全部正常工作

#### TODO-3.4: ModelManagementService 实现
```python
# quantumflow/grpc/services/model_management.py
class ModelManagementServiceServicer(
    model_management_service_pb2_grpc.ModelManagementServiceServicer
):
    """模型管理服务"""
    def LoadModel(self, request, context):
        # 委托给 EngineManager

    def UnloadModel(self, request, context):
        # 委托给 EngineManager

    def ListModels(self, request, context):
        # 返回已加载模型列表
```
**完成标准**: 模型加载/卸载/列表全部正常工作

#### TODO-3.5: HealthService 实现
```python
# quantumflow/grpc/services/health.py
class HealthServiceServicer(health_service_pb2_grpc.HealthServiceServicer):
    """健康检查服务 - 实现 gRPC 标准健康检查"""
    def __init__(self):
        self._status = True

    def Check(self, request, context):
        return health_service_pb2.HealthCheckResponse(
            healthy=self._status,
            status="OK" if self._status else "UNHEALTHY"
        )

    def Watch(self, request, context):
        """流式健康检查"""
        while True:
            yield health_service_pb2.HealthCheckResponse(
                healthy=self._status,
                status="OK" if self._status else "UNHEALTHY"
            )
            time.sleep(5)
```
**完成标准**: Check 和 Watch 都正常工作

---

### Phase 4: 服务端启动

#### TODO-4.1: gRPC Server 封装
```python
# quantumflow/grpc/server.py
class GrpcServer:
    """gRPC 服务器封装"""
    def __init__(self, port: int = 50051):
        self.port = port
        self.server = None
        self.servicers = {}

    def add_service(self, service, servicer):
        """添加服务"""
        self.servicers[service] = servicer

    def start(self):
        """启动服务器"""
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            interceptors=[
                LoggingInterceptor(),
                AuthInterceptor(),
                MetricsInterceptor(),
                RateLimitInterceptor(),
            ]
        )
        # 注册所有服务
        for service, servicer in self.servicers.items():
            service.add_to_server(servicer, self.server)
        self.server.add_insecure_port(f'[::]:{self.port}')
        self.server.start()

    def stop(self):
        """停止服务器"""
        self.server.stop(grace=5)

    def wait_for_termination(self):
        self.server.wait_for_termination()
```
**完成标准**: 可以启动和停止

#### TODO-4.2: 整合到主服务器
```python
# quantumflow/api/server.py 修改
# 在现有 FastAPI 服务中启动 gRPC
from quantumflow.grpc.server import GrpcServer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动 FastAPI
    # ...
    # 启动 gRPC
    grpc_server = GrpcServer(port=50051)
    grpc_server.add_service(
        inference_service_pb2_grpc.InferenceServiceServicer,
        InferenceServiceServicer(...)
    )
    grpc_server.start()
    yield
    grpc_server.stop()
```
**完成标准**: FastAPI 和 gRPC 同时运行

---

### Phase 5: 客户端

#### TODO-5.1: gRPC Channel 管理
```python
# quantumflow/grpc/channels/pool.py
class GrpcChannelPool:
    """gRPC 连接池"""
    def __init__(self, endpoints: List[str]):
        self.channels = {
            endpoint: grpc.insecure_channel(endpoint)
            for endpoint in endpoints
        }

    def get_channel(self, endpoint: str) -> grpc.Channel:
        return self.channels.get(endpoint)

    def close_all(self):
        for channel in self.channels.values():
            channel.close()

    def get_stats(self) -> Dict[str, int]:
        """获取连接统计"""
        # 连接数、活跃数等
```
**完成标准**: 连接池正确管理连接

#### TODO-5.2: InferenceClient
```python
# quantumflow/grpc/clients/inference.py
class InferenceClient:
    """推理客户端"""
    def __init__(self, channel: grpc.Channel):
        self.stub = inference_service_pb2_grpc.InferenceServiceStub(channel)

    def inference(self, request: InferenceRequest) -> InferenceResponse:
        """同步推理"""
        return self.stub.Inference(request)

    def inference_stream(self, request: InferenceRequest) -> Iterator[InferenceResponse]:
        """流式推理"""
        return self.stub.InferenceStream(request)

    def batch_inference(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """批量推理"""
        return self.stub.BatchInference(request)
```
**完成标准**: 客户端可调用

#### TODO-5.3: ClusterClient
```python
# quantumflow/grpc/clients/cluster.py
class ClusterClient:
    """集群管理客户端"""
    def __init__(self, channel: grpc.Channel):
        self.stub = cluster_service_pb2_grpc.ClusterServiceStub(channel)

    def register_node(self, request: RegisterNodeRequest) -> RegisterNodeResponse:
        return self.stub.RegisterNode(request)

    def heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        return self.stub.Heartbeat(request)

    def list_nodes(self, filter_status=None) -> ListNodesResponse:
        request = ListNodesRequest(filter_status=filter_status)
        return self.stub.ListNodes(request)
```
**完成标准**: 客户端可调用

#### TODO-5.4: SchedulerClient
```python
# quantumflow/grpc/clients/scheduler.py
class SchedulerClient:
    """调度客户端"""
    def __init__(self, channel: grpc.Channel):
        self.stub = scheduler_service_pb2_grpc.SchedulerServiceStub(channel)

    def submit(self, request: SchedulingRequest) -> SchedulingResponse:
        return self.stub.SubmitRequest(request)

    def get_status(self, request_id: str) -> GetSchedulingStatusResponse:
        request = GetSchedulingStatusRequest(request_id=request_id)
        return self.stub.GetStatus(request)

    def cancel(self, request_id: str) -> CancelResponse:
        return CancelRequest(request_id=request_id)
        return self.stub.CancelRequest(request)
```
**完成标准**: 客户端可调用

---

### Phase 6: Worker 集成

#### TODO-6.1: Worker gRPC 客户端
```python
# quantumflow/worker/grpc_client.py
class WorkerGrpcClient:
    """Worker 使用的 gRPC 客户端（连接 Controller）"""
    def __init__(self, controller_endpoint: str):
        self.channel = grpc.insecure_channel(controller_endpoint)
        self.cluster_stub = cluster_service_pb2_grpc.ClusterServiceStub(self.channel)
        self.model_stub = model_management_service_pb2_grpc.ModelManagementServiceStub(self.channel)

    def send_heartbeat(self, resources: NodeResources) -> HeartbeatResponse:
        request = HeartbeatRequest(node_id=self.node_id, resources=resources)
        return self.cluster_stub.Heartbeat(request)

    def report_model_loaded(self, model_name: str):
        # 通知 Controller 模型已加载
```
**完成标准**: Worker 可通过 gRPC 与 Controller 通信

#### TODO-6.2: Worker gRPC 服务
```python
# quantumflow/worker/grpc_service.py
class WorkerGrpcService(inference_service_pb2_grpc.InferenceServiceServicer):
    """Worker 上的 gRPC 推理服务"""
    def __init__(self, worker: WorkerNode):
        self.worker = worker

    def Inference(self, request, context):
        # 调用 worker.engine_manager 执行推理
        result = self.worker.engine_manager.generate(
            request.model_name, request.prompt, params
        )
        return InferenceResponse(
            request_id=request.request_id,
            status=STATUS_SUCCESS,
            text=result.text,
            tokens_generated=result.num_tokens
        )

    def InferenceStream(self, request, context):
        # yield 每个 token
```
**完成标准**: Worker 接收 gRPC 请求

---

## 四、开发顺序

```
Phase 1 (基础设施)
  TODO-1.1 → TODO-1.2 → TODO-1.3 → TODO-1.4

Phase 2 (异常和拦截器)
  TODO-2.1 → TODO-2.2 → TODO-2.3 → TODO-2.4 → TODO-2.5

Phase 3 (服务实现)
  TODO-3.1 → TODO-3.2 → TODO-3.3 → TODO-3.4 → TODO-3.5

Phase 4 (服务端启动)
  TODO-4.1 → TODO-4.2

Phase 5 (客户端)
  TODO-5.1 → TODO-5.2 → TODO-5.3 → TODO-5.4

Phase 6 (Worker 集成)
  TODO-6.1 → TODO-6.2
```

---

## 五、配置项

```yaml
# configs/default.yaml 添加
grpc:
  enabled: true
  port: 50051
  max_workers: 10
  reflection_enabled: true  # 启用 gRPC Reflection

  interceptors:
    logging:
      enabled: true
      log_level: INFO
    auth:
      enabled: false  # 开发环境关闭
      api_keys:
        - key: "dev-key-12345"
          user: "dev"
    rate_limit:
      enabled: true
      qps: 100

  connection_pool:
    max_size: 10
    min_size: 1
    keepalive_ms: 30000
```

---

## 六、部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Client                                 │
│  (HTTP REST + gRPC)                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐
│   FastAPI Server    │   │   gRPC Server       │
│   (HTTP REST)       │   │   (Port 50051)      │
│   (Port 8000)       │   │                     │
│                     │   │  - InferenceService │
│  - /inference       │   │  - ClusterService   │
│  - /cluster         │   │  - SchedulerService │
│  - /scheduler      │   │  - HealthService    │
│  - /metrics        │   │  - ModelMgmtService  │
└─────────────────────┘   └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
             ┌────────────┐   ┌────────────┐   ┌────────────┐
             │  Worker 1  │   │  Worker 2  │   │  Worker 3  │
             │  (gRPC)    │   │  (gRPC)    │   │  (gRPC)    │
             └────────────┘   └────────────┘   └────────────┘
```

---

## 七、向后兼容

- HTTP REST API 继续保留
- gRPC 作为高性能替代方案
- 客户端可选择使用 HTTP 或 gRPC
- 内部通信优先使用 gRPC
