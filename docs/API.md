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
  - [Model Management](#13-model-management)
  - [Inference](#14-inference)
  - [Cluster](#15-cluster)
  - [Tenants](#16-tenants)
  - [Hub](#17-hub)
  - [Scheduler](#18-scheduler)
  - [Metrics](#19-metrics)
- [gRPC API](#2-grpc-api)
- [SDK](#3-sdk)
- [CLI](#4-cli)
- [错误响应](#5-错误响应)

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

> 注：`/models` 前缀下挂了两个 router：`models`（元数据）和 `model_management`（部署/卸载/基准）。完整路径见下两节。

#### `GET /api/v1/models`

列出所有已注册模型。

**Query**：`status_filter`, `backend`

**响应**：`ModelInfo[]`

#### `GET /api/v1/models/{model_name}`

获取模型详情。404 表示模型不存在。

#### `POST /api/v1/models/deploy`

注册 / 部署模型元信息（不实际加载权重——见 1.3 Model Management）。

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

#### `POST /api/v1/models/undeploy`

注销模型元信息。

#### `POST /api/v1/models/benchmark`

注册基准测试任务（实际运行由 `Model Management /benchmark` 触发）。

---

### 1.3 Model Management

> 与 1.2 Models 共享 `/api/v1/models` 前缀，但语义不同：1.2 是元数据 CRUD，本节是**实际加载/卸载/查询**权重到 GPU。

#### `POST /api/v1/models/load`

将指定模型的权重加载到 Worker 节点的 GPU 上。

**请求**

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "backend": "vllm",
  "tensor_parallel": 2,
  "gpu_memory_utilization": 0.85
}
```

**响应**

```json
{
  "status": "loading",
  "message": "Model deployment started on node-1"
}
```

#### `POST /api/v1/models/unload`

卸载模型权重，释放显存。

#### `GET /api/v1/models/status`

查询模型在所有 Worker 上的加载状态（按节点展开）。

#### `GET /api/v1/models/list`

列出当前已加载（占用显存）的模型。

> **Register 重名约定**：`POST /api/v1/models/deploy` 内部调用 `ModelRegistry.register_model`，默认 `overwrite=False`——**重名注册返回 `false`，HTTP 响应 `200` 但 `deployed=false`**。若要覆盖已存在模型，显式传 `"overwrite": true`。

---

### 1.4 Inference

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

#### `POST /api/v1/inference/submit` / `/api/v1/inference/submit/batch`

异步提交（返回 `request_id` 后通过 `/result/{request_id}` 轮询结果）。

#### `GET /api/v1/inference/result/{request_id}`

查询异步任务结果。

#### `GET /api/v1/inference/queue/stats`

队列统计（同步入口）。异步版本走 gRPC `SchedulerService.GetStatus`。

---

### 1.5 Cluster

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

#### `GET /api/v1/cluster/nodes/{node_id}`

单个节点详情。

#### `POST /api/v1/cluster/nodes/{node_id}/action`

节点运维动作（drain / undrain / quarantine / resume）。

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

### 1.6 Tenants

| Method | Path | 说明 |
|--------|------|------|
| POST | `/api/v1/tenants/` | 创建租户（返回 API Key，仅创建时返回明文） |
| GET | `/api/v1/tenants/` | 列出所有租户 |
| GET | `/api/v1/tenants/{id}` | 获取租户详情 |
| PATCH | `/api/v1/tenants/{id}` | 更新租户（配额/状态） |
| DELETE | `/api/v1/tenants/{id}` | 软删除租户 |
| GET | `/api/v1/tenants/{id}/usage` | 获取使用量统计（今日 / 7 日 / 30 日） |

---

### 1.7 Hub（HuggingFace 集成）

通过 `api/services/hub_service.py` 代理 HuggingFace Hub，提供模型发现 / 元数据 / 下载 / 推荐。

| Method | Path | 说明 |
|--------|------|------|
| GET | `/api/v1/hub/trending` | HuggingFace Trending 模型 |
| GET | `/api/v1/hub/search?q=...&limit=...` | 模型搜索 |
| POST | `/api/v1/hub/validate` | 校验本地 / 远端模型路径与可用性 |
| GET | `/api/v1/hub/detail/{model_id}` | 模型详情（参数量、推荐 TP、最小 VRAM） |
| POST | `/api/v1/hub/download` | 异步下载模型（返回 download_id） |
| GET | `/api/v1/hub/download/progress/{download_id}` | 下载进度 |
| GET | `/api/v1/hub/downloaded` | 列出已下载到本地的模型 |
| GET | `/api/v1/hub/recommendations` | 基于 `SystemProfiler`（GPU / 内存 / 磁盘）的模型推荐 |

> `recommendations` 端点会读取 `api/services/system_profiler.py::SystemCapability`（GPU 数量 / 总 VRAM / 可用磁盘 / CUDA 版本），匹配出该硬件能跑的最大模型。

---

### 1.8 Scheduler

#### `GET /api/v1/scheduler/status`

调度器状态：当前队列长度 / 策略 / 节点分布。

---

### 1.9 Metrics

#### `GET /api/v1/metrics`

Prometheus 标准指标端点（`text/plain; version=0.0.4`）。

| 指标 | 类型 | 描述 |
|------|------|------|
| `qf_inference_requests_total` | Counter | 推理请求总数（按 model / status 标签） |
| `qf_inference_latency_seconds` | Histogram | 推理延迟分布 |
| `qf_gpu_utilization` | Gauge | GPU 计算利用率（按 gpu_id） |
| `qf_gpu_memory_used_bytes` | Gauge | GPU 显存使用量（按 gpu_id） |
| `qf_scheduler_queue_size` | Gauge | 调度队列长度（按 priority） |
| `qf_worker_count` | Gauge | Worker 节点数量（按 status） |
| `qf_model_loaded` | Gauge | 已加载模型数量（按 backend） |
| `qf_tenant_concurrent` | Gauge | 当前并发请求数（按 tenant_id） |

Histogram buckets：`[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]` 秒。

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

### 2.6 容灾接入（failover）

> 容灾层（`quantumflow/failover/`）目前**没有独立 gRPC Service**——副本管理 / Leader 选举 / 健康检查统一收敛在以下 RPC：
>
> - 节点健康相关：走 `ClusterService.Heartbeat` / `HealthService.Check/Watch`
> - 模型副本查询 / 主从切换：经 `REST` 端点调用 Controller 内部 `FailoverController`（详见 [ARCHITECTURE.md §10 容灾层](./ARCHITECTURE.md#10-容灾层)）
>
> 后续可能拆出独立的 `FailoverService`（见 [README.md Roadmap §容灾](../README.md#路线图)）。

---

## 3. SDK

Python SDK 位于 `quantumflow/sdk/`，提供同步和异步两套客户端。

### 3.1 安装与导入

```python
from quantumflow import SyncClient, AsyncClient
```

或直接从子模块：

```python
from quantumflow.sdk.client import SyncQuantumFlowClient, AsyncQuantumFlowClient
```

### 3.2 同步客户端

```python
from quantumflow import SyncClient

client = SyncClient(base_url="http://localhost:8000", api_key="qf-dev-xxx")

# 同步生成
result = client.generate(
    model="Qwen2.5-7B-Instruct",
    prompt="用一句话介绍分布式系统",
    sampling_params={"temperature": 0.7, "max_tokens": 100},
)
print(result.generated_text)

# 流式生成
for chunk in client.stream(model="Qwen2.5-7B-Instruct", prompt="写一首诗"):
    print(chunk.delta, end="", flush=True)

# 上下文管理器（自动关闭 httpx.Client）
with SyncClient(base_url=..., api_key=...) as client:
    result = client.generate(...)
```

### 3.3 异步客户端

```python
from quantumflow import AsyncClient

async with AsyncClient(base_url="http://localhost:8000", api_key="qf-xxx") as client:
    # 异步生成
    result = await client.generate(model=..., prompt=...)

    # 异步流式
    async for chunk in client.stream(model=..., prompt=...):
        print(chunk.delta, end="", flush=True)
```

### 3.4 异常体系

| 异常 | HTTP 状态 | 触发场景 |
|------|-----------|----------|
| `APIError` | 4xx/5xx | 服务端返回非 200 |
| `RateLimitError` | 429 | 限流触发 |
| `TimeoutError` | — | httpx 超时 |

### 3.5 当前已知问题

> `tests/unit/sdk/test_client.py` 和 `test_async_client.py` 共 11 个测试在干净 `main` 上失败：mock 使用 `__enter__.return_value.post` 但 `SyncQuantumFlowClient._post` 实际**未走上下文管理器**（直接 `self._client.post(...)`），所以 `status_code` 是 MagicMock 而非 int。SDK 功能本身正常（手工 `curl` 验证 OK），仅是测试代码 mock 写错。详见 [TESTING.md §6 已知失败](./TESTING.md#6-已知失败预存在)。

---

## 4. CLI

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

> CLI 基于 `click` 实现（`quantumflow/cli.py`），用 `rich` 渲染表格输出。子命令通过 `cli.add_command(...)` 注册；具体列表以 `python -m quantumflow.cli --help` 为准。

---

## 5. 错误响应

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
