# QuantumFlow API 参考

> REST API（FastAPI）、gRPC API、CLI 命令的统一参考。

**Base URL**：`http://<host>:8000`
**gRPC 端口**：`50051`
**API 版本**：`v1`
**认证**：除 `/health` 外，所有 REST 端点需要 `X-API-Key` Header

---

## 目录

- [REST API](#1-rest-api)
  - [Health](#11-health)
  - [Models](#12-models)
  - [Inference](#13-inference)
  - [Cluster](#14-cluster)
  - [Tenants](#15-tenants)
  - [Metrics](#16-metrics)
- [gRPC API](#2-grpc-api)
- [CLI](#3-cli)
- [错误响应](#4-错误响应)

---

## 1. REST API

### 1.1 Health

#### `GET /api/v1/health`

健康检查。返回服务自身、Redis、集群的整体状态。

**响应**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "checks": {
    "api": "healthy",
    "redis": "healthy",
    "cluster": "healthy"
  }
}
```

#### `GET /api/v1/health/ready`

就绪检查。任一依赖不可用返回 HTTP 503。

```json
{"ready": true}
```

#### `GET /api/v1/health/live`

存活检查。用于 Kubernetes liveness probe。

```json
{"alive": true}
```

---

### 1.2 Models

#### `GET /api/v1/models`

列出所有已注册模型。

**Query**：`status_filter`, `backend`

**响应**：`ModelInfo[]`

#### `GET /api/v1/models/{model_name}`

获取模型详情。404 表示模型不存在。

#### `POST /api/v1/models/deploy`

部署模型。

**请求**

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "backend": "vllm",
  "tensor_parallel": 1,
  "replicas": 1,
  "dtype": "bfloat16",
  "max_model_length": 8192
}
```

**响应**

```json
{
  "model_id": "qwen2.5-7b-1712000000",
  "status": "loading",
  "replicas": 1,
  "message": "Model deployment started"
}
```

#### `POST /api/v1/models/undeploy`

卸载模型。`force: true` 强制卸载（即使有正在执行的请求）。

**请求**

```json
{"model": "Qwen2.5-7B-Instruct", "force": false}
```

#### `POST /api/v1/models/benchmark`

运行基准测试，返回 `benchmark_id` 用于查询进度。

---

### 1.3 Inference

#### `POST /api/v1/inference/generate`

同步文本生成。

**请求**

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "prompt": "Hello, how are you?",
  "sampling_params": {
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 0.9
  }
}
```

**响应**

```json
{
  "request_id": "uuid-here",
  "model": "Qwen2.5-7B-Instruct",
  "generated_text": "I'm doing well, thank you!",
  "finish_reason": "stop",
  "prompt_tokens": 10,
  "completion_tokens": 20,
  "total_tokens": 30
}
```

#### `POST /api/v1/inference/chat`

多轮对话。自动转 ChatML 格式。

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "sampling_params": {"temperature": 0.7, "max_tokens": 100}
}
```

#### `POST /api/v1/inference/generate/stream`

流式生成（SSE）。

```
data: {"delta": "Hello", "is_final": false}
data: {"delta": " world", "is_final": false}
data: {"delta": "!", "is_final": true}
data: [DONE]
```

#### `POST /api/v1/inference/batch`

批量生成。

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "prompts": ["Hello", "Hi there", "How are you?"],
  "sampling_params": {"temperature": 0.7, "max_tokens": 50}
}
```

**响应**

```json
{
  "total": 3,
  "results": [
    {"request_id": "uuid-1", "generated_text": "...", "...": "..."},
    {"request_id": "uuid-2", "generated_text": "...", "...": "..."},
    {"request_id": "uuid-3", "generated_text": "...", "...": "..."}
  ]
}
```

---

### 1.4 Cluster

#### `GET /api/v1/cluster/status`

集群整体状态。

```json
{
  "total_nodes": 3,
  "healthy_nodes": 2,
  "unhealthy_nodes": 1,
  "total_gpus": 12,
  "available_gpus": 8,
  "total_memory_bytes": 512000000000,
  "available_memory_bytes": 256000000000
}
```

#### `GET /api/v1/cluster/nodes`

列出所有节点及详情。

#### `POST /api/v1/cluster/heartbeat`

节点心跳上报（Worker 调用）。

```json
{
  "node_id": "node-1",
  "hostname": "server-1",
  "ip": "192.168.1.100",
  "gpu_count": 4,
  "gpu_info": [],
  "status": "healthy",
  "current_load": 0.5
}
```

---

### 1.5 Tenants

| Method | Path | 说明 |
|--------|------|------|
| POST | `/api/v1/tenants/` | 创建租户（返回 API Key） |
| GET | `/api/v1/tenants/` | 列出所有租户 |
| GET | `/api/v1/tenants/{id}` | 获取租户详情 |
| PATCH | `/api/v1/tenants/{id}` | 更新租户（配额/状态） |
| DELETE | `/api/v1/tenants/{id}` | 软删除租户 |
| GET | `/api/v1/tenants/{id}/usage` | 获取使用量统计 |

---

### 1.6 Metrics

#### `GET /api/v1/metrics`

Prometheus 指标端点。

| 指标 | 类型 | 描述 |
|------|------|------|
| `qf_inference_requests_total` | Counter | 推理请求总数 |
| `qf_inference_latency_seconds` | Histogram | 推理延迟分布 |
| `qf_gpu_utilization` | Gauge | GPU 利用率 |
| `qf_gpu_memory_used_bytes` | Gauge | GPU 显存使用量 |
| `qf_scheduler_queue_size` | Gauge | 调度队列长度 |
| `qf_worker_count` | Gauge | Worker 节点数量 |
| `qf_model_loaded` | Gauge | 已加载模型数量 |

---

## 2. gRPC API

服务定义文件：`quantumflow/grpc/proto/quantumflow.proto`

### 2.1 InferenceService

| RPC | 类型 | 说明 |
|-----|------|------|
| `Inference` | Unary | 同步推理 |
| `InferenceStream` | Server Stream | 流式推理 |
| `BatchInference` | Unary | 批量推理 |

### 2.2 ClusterService

| RPC | 类型 | 说明 |
|-----|------|------|
| `RegisterNode` | Unary | 节点注册 |
| `DeregisterNode` | Unary | 节点注销 |
| `Heartbeat` | Unary | 心跳上报 |
| `ListNodes` | Unary | 列出节点 |
| `UpdateNodeResources` | Bidirectional | 节点资源推送 |

### 2.3 SchedulerService

| RPC | 类型 | 说明 |
|-----|------|------|
| `SubmitRequest` | Unary | 提交调度请求 |
| `CancelRequest` | Unary | 取消请求 |
| `GetStatus` | Unary | 查询调度状态 |

### 2.4 ModelManagementService

| RPC | 说明 |
|-----|------|
| `LoadModel` | 加载模型 |
| `UnloadModel` | 卸载模型 |
| `ListModels` | 列出已加载模型 |

### 2.5 HealthService

| RPC | 类型 | 说明 |
|-----|------|------|
| `Check` | Unary | 健康检查 |
| `Watch` | Server Stream | 流式健康监控 |

---

## 3. CLI

```bash
# 启动 API 服务
python -m quantumflow.cli serve [--host HOST] [--port PORT]

# 集群状态
python -m quantumflow.cli status

# 模型列表
python -m quantumflow.cli models

# 加载模型
python -m quantumflow.cli load <model_name>

# 文本生成
python -m quantumflow.cli generate <model_name> -p <prompt>

# 对话
python -m quantumflow.cli chat <model_name> -p <prompt>

# 交互式终端
python -m quantumflow.cli interactive
```

---

## 4. 错误响应

所有错误统一返回以下结构：

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model Qwen2.5-7B is not loaded",
    "request_id": "uuid-here"
  }
}
```

| HTTP 状态码 | 场景 |
|-------------|------|
| 400 | 请求参数校验失败 |
| 401 | API Key 无效或缺失 |
| 403 | 租户被禁用或配额耗尽 |
| 404 | 模型/节点不存在 |
| 429 | 限流触发 |
| 500 | 内部错误 |
| 503 | 依赖（Redis/集群）不可用 |

---

*系统设计见 [ARCHITECTURE.md](./ARCHITECTURE.md)；部署见 [DEPLOYMENT.md](./DEPLOYMENT.md)。*
