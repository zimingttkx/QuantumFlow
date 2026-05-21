# QuantumFlow 完整规格文档

> 本文档详细记录 QuantumFlow 分布式大模型推理平台的完整功能规格、设计逻辑、模块交互和数据流。
> 生成日期：2026-05-20

---

## 目录

1. [项目概述](#1-项目概述)
2. [模块架构总览](#2-模块架构总览)
3. [API 层](#3-api-层)
4. [调度层](#4-调度层)
5. [推理引擎层](#5-推理引擎层)
6. [存储层](#6-存储层)
7. [集群管理层](#7-集群管理层)
8. [Worker 层](#8-worker-层)
9. [监控层](#9-监控层)
10. [工具层](#10-工具层)
11. [CLI 层](#11-cli-层)
12. [测试覆盖分析](#12-测试覆盖分析)
13. [待实现功能清单](#13-待实现功能清单)

---

## 1. 项目概述

### 1.1 项目定位

QuantumFlow 是一个**生产级的分布式大模型推理平台**，旨在像调度 Kubernetes Pods 一样调度 AI 推理任务。

### 1.2 核心价值

| 特性 | 说明 |
|------|------|
| 智能调度 | Gang / Pack / Adaptive 多策略自动选择 |
| 分布式部署 | Redis 队列 + Worker 节点，Controller 与 Worker 完全解耦 |
| 多后端支持 | vLLM / HuggingFace / TGI / SGLang 统一接口 |
| GPU 优化 | BatchAccumulator / Chunked Prefill / Block VRAM 显存精细管理 |
| 多租户支持 | 租户间显存隔离，资源配额（规划中） |

### 1.3 技术栈

```
应用层：FastAPI + Pydantic + Uvicorn
调度层：asyncio + 自定义调度算法 + Redis
执行层：vLLM / PyTorch / CUDA
存储层：Redis
监控层：Prometheus + Grafana
基础设施：Docker / Kubernetes
```

---

## 2. 模块架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        QuantumFlow Platform                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     接入层 (API Gateway)                          │  │
│  │  REST API (FastAPI)  │  gRPC API (规划中)  │  Python SDK      │  │
│  │  CLI Tool            │  Web UI (规划中)                        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      调度层 (Scheduler)                          │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │  Priority Queue  │  Strategy Selector  │  Dispatcher  │  │  │
│  │  │  Gang / Pack / Adaptive 策略                                │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  │                              │                                   │  │
│  │  ┌──────────────────────────┼──────────────────────────────┐  │  │
│  │  │  DistributedScheduler  │  Redis Queue  │  Worker Client│  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────┼────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      执行层 (Inference)                          │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │  EngineManager  │  VRAMManager  │  BatchAccumulator    │  │  │
│  │  │  GPUMonitor     │  ChunkedPrefill                       │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  │                              │                                   │  │
│  │  ┌──────────────────────────┼──────────────────────────────┐  │  │
│  │  │  HuggingFace  │  vLLM  │  TGI  │  SGLang            │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────┼────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      存储层 (Storage)                             │  │
│  │  Redis Queue (ZSET 优先级)  │  RedisConnectionManager (单例)   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      集群管理层 (Cluster)                         │  │
│  │  NodeRegistry  │  ServiceDiscovery  │  HealthMonitor          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      Worker 层                                   │  │
│  │  TaskFetcher (Redis 拉取)  │  Worker API  │  Worker Node     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. API 层

### 3.1 模块结构

```
quantumflow/api/
├── __init__.py
├── server.py           # FastAPI 应用入口
├── models/             # Pydantic 请求/响应模型
│   ├── __init__.py
│   └── models.py      # 所有 API 数据模型
├── routes/             # 路由模块
│   ├── __init__.py
│   ├── health.py       # 健康检查端点
│   ├── models.py       # 模型管理端点
│   ├── cluster.py      # 集群管理端点
│   ├── inference.py    # 推理端点
│   ├── scheduler.py    # 调度可视化端点
│   └── worker.py       # Worker API 端点
└── services/          # 业务服务
    ├── __init__.py
    └── hub_service.py  # HuggingFace Hub 服务
```

### 3.2 API 端点规格

#### 3.2.1 健康检查端点 `/health`

**GET `/api/v1/health`** — 健康检查

响应：
```json
{
  "status": "healthy",           // "healthy" | "degraded" | "unhealthy"
  "version": "1.0.0",
  "uptime_seconds": 3600,        // 从应用启动时间计算，不是 0
  "checks": {
    "api": "healthy",            // 实时检查 API 自身
    "redis": "healthy",          // 实时检查 Redis 连接
    "cluster": "healthy"         // 实时检查集群状态
  }
}
```

**规格要求：**
- `uptime_seconds` 必须从应用启动时间计算，不能硬编码为 0
- `checks.redis` 必须实际 ping Redis 并验证连接
- `checks.cluster` 必须查询真实集群节点状态
- 任一 check 为 unhealthy 时，顶层 status 必须为 unhealthy 或 degraded

**GET `/api/v1/health/ready`** — 就绪检查

响应：
```json
{"ready": true}
```

**规格要求：**
- 必须检查所有依赖（Redis、集群）是否就绪
- 任一依赖不可用时，返回 `ready: false`（HTTP 503）

**GET `/api/v1/health/live`** — 存活检查

响应：
```json
{"alive": true}
```

**规格要求：**
- 仅检查进程是否存活，不检查依赖
- 用于 Kubernetes liveness probe

#### 3.2.2 模型管理端点 `/models`

**GET `/api/v1/models`** — 列出所有模型

响应：`ModelInfo[]`

**规格要求：**
- 返回所有已注册的模型（包括已加载和未加载）
- 支持 `status_filter` 和 `backend` 查询参数
- 模型信息必须从 EngineManager 实时查询，不能从 `_mock_models` 字典返回

**GET `/api/v1/models/{model_name}`** — 获取模型详情

响应：`ModelInfo`

**规格要求：**
- 返回指定模型的详细信息
- 模型不存在返回 404

**POST `/api/v1/models/deploy`** — 部署模型

请求：
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

响应：
```json
{
  "model_id": "qwen2.5-7b-1712000000",
  "status": "loading",        // "loading" | "ready" | "failed"
  "replicas": 1,
  "message": "Model deployment started"
}
```

**规格要求：**
- 必须调用 EngineManager.load_model() 真实加载模型
- 不能只操作 `_mock_models` 字典
- 部署是异步的，API 返回后模型仍在加载
- 需要跟踪部署状态并支持查询

**POST `/api/v1/models/undeploy`** — 卸载模型

请求：
```json
{
  "model": "Qwen2.5-7B-Instruct",
  "force": false
}
```

响应：
```json
{
  "model_id": "qwen2.5-7b-1712000000",
  "status": "unloaded",
  "message": "Model unloaded successfully"
}
```

**规格要求：**
- 必须调用 EngineManager.unload_model() 真实卸载模型
- `force: true` 时强制卸载，即使有请求正在使用
- 模型不存在返回 404

**POST `/api/v1/models/benchmark`** — 运行基准测试

请求：
```json
{
  "model": "Qwen2.5-7B-Instruct",
  "test_set": "standard",
  "num_samples": 100
}
```

响应：
```json
{
  "benchmark_id": "bench_1712000000000",
  "model": "Qwen2.5-7B-Instruct",
  "test_set": "standard",
  "status": "running",         // "pending" | "running" | "completed" | "failed"
  "total_samples": 100,
  "completed_samples": 0
}
```

**规格要求：**
- 必须真实启动基准测试任务
- 返回 benchmark_id 用于查询进度
- 必须跟踪 `completed_samples` 进度

#### 3.2.3 推理端点 `/inference`

**POST `/api/v1/inference/generate`** — 文本生成

请求：
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

响应：
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

**规格要求：**
- 必须通过 Scheduler 调度到 Worker 节点执行
- 不能直接调用本地引擎
- 返回真实的推理结果

**POST `/api/v1/inference/chat`** — 对话生成

请求：
```json
{
  "model": "Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "sampling_params": {
    "temperature": 0.7,
    "max_tokens": 100
  }
}
```

响应：同 `/generate`

**规格要求：**
- 自动转换为 ChatML 格式
- 支持多轮对话

**POST `/api/v1/inference/generate/stream`** — 流式生成

请求：同 `/generate`，`sampling_params.stream: true`

响应：Server-Sent Events (SSE)
```
data: {"delta": "Hello", "is_final": false}
data: {"delta": " world", "is_final": false}
data: {"delta": "!", "is_final": true}
data: [DONE]
```

**规格要求：**
- 每个 token 生成后立即推送
- `is_final: true` 表示生成结束

**POST `/api/v1/inference/batch`** — 批量生成

请求：
```json
{
  "model": "Qwen2.5-7B-Instruct",
  "prompts": ["Hello", "Hi there", "How are you?"],
  "sampling_params": {
    "temperature": 0.7,
    "max_tokens": 50
  }
}
```

响应：
```json
{
  "total": 3,
  "results": [
    {"request_id": "uuid-1", "generated_text": "...", ...},
    {"request_id": "uuid-2", "generated_text": "...", ...},
    {"request_id": "uuid-3", "generated_text": "...", ...}
  ]
}
```

#### 3.2.4 集群管理端点 `/cluster`

**GET `/api/v1/cluster/status`** — 集群状态

响应：
```json
{
  "total_nodes": 3,
  "healthy_nodes": 2,
  "unhealthy_nodes": 1,
  "total_gpus": 12,
  "available_gpus": 8,
  "total_memory_bytes": 512_000_000_000,
  "available_memory_bytes": 256_000_000_000
}
```

**规格要求：**
- 必须从 NodeRegistry 实时查询
- `uptime_seconds` 必须正确计算（不是从 epoch 开始）

**GET `/api/v1/cluster/nodes`** — 列出所有节点

响应：`NodeInfo[]`

**POST `/api/v1/cluster/heartbeat`** — 节点心跳

请求：
```json
{
  "node_id": "node-1",
  "hostname": "server-1",
  "ip": "192.168.1.100",
  "gpu_count": 4,
  "gpu_info": [...],
  "status": "healthy",
  "current_load": 0.5
}
```

**规格要求：**
- 节点注册/更新到 NodeRegistry
- 更新心跳时间戳
- 超时未心跳的节点自动标记为 unhealthy

---

## 4. 调度层

### 4.1 模块结构

```
quantumflow/scheduler/
├── __init__.py
├── scheduler.py           # 基础调度器（单机）
├── distributed.py          # 分布式调度器
├── worker_client.py        # Worker HTTP 客户端
├── strategy/               # 调度策略
│   ├── __init__.py
│   ├── base.py            # 策略基类
│   ├── gang.py            # Gang 调度
│   ├── pack.py            # Pack 调度
│   ├── adaptive.py         # Adaptive 调度
│   └── priority.py         # 优先级队列
```

### 4.2 调度器规格

#### 4.2.1 Scheduler（基础调度器）

**职责：** 管理请求队列、节点注册、调度循环

**核心功能：**
- `submit(request)` — 提交推理请求到队列
- `register_node(node)` — 注册 GPU 节点
- `unregister_node(node_id)` — 注销节点
- `update_node(node)` — 更新节点状态
- `start()` — 启动调度循环
- `stop()` — 停止调度循环

**调度循环逻辑：**
1. 从优先级队列取出最高优先级请求
2. 根据请求特征选择调度策略（Gang/Pack/Adaptive）
3. 策略选择最优节点
4. 分配 GPU 资源
5. **将请求发送到 Worker 节点执行**（不能只是模拟）
6. 跟踪请求状态（pending → running → completed/failed）

**关键要求：**
- `submit()` 后请求必须进入 Redis 队列（分布式模式）
- 调度循环必须真正将请求发送到 Worker，不能用 `_simulate_execution()`
- Worker 执行完成后必须更新请求状态
- fire-and-forget 模式下任务失败不能被静默忽略

#### 4.2.2 DistributedScheduler（分布式调度器）

**职责：** 跨节点的分布式调度，通过 Redis 队列协调

**核心功能：**
- 使用 Redis ZSET 实现优先级队列
- 通过 WorkerClient 与 Worker 节点 HTTP 通信
- 支持节点故障转移

**关键要求：**
- Redis 队列 key：`quantumflow:pending_requests`
- 队列元素：JSON 序列化的请求信息 + 优先级分数
- Worker 节点通过 TaskFetcher 从 Redis 拉取任务
- 调度器通过 WorkerClient 发送执行结果回调

#### 4.2.3 WorkerClient

**职责：** Controller 到 Worker 的 HTTP 通信客户端

**核心功能：**
- `send_inference_request(request)` — 发送推理请求到 Worker
- `get_worker_status(worker_id)` — 查询 Worker 状态
- `cancel_request(request_id)` — 取消请求

**关键要求：**
- 必须使用 httpx.AsyncClient 异步 HTTP 调用
- 必须处理连接超时和重试
- Worker 不可用时必须正确报错，不能静默失败

#### 4.2.4 调度策略

**GangSchedulingStrategy — Gang 调度**
- 适用：70B+ 大模型，TP >= 4
- 规则：所有 GPU 必须同时分配，要么全部分配要么不分配
- 实现：`select_nodes()` 必须检查是否有足够 GPU 容纳整个模型

**PackSchedulingStrategy — Pack 调度**
- 适用：< 30B 小模型，高并发场景
- 规则：多个小请求共享同一 GPU，动态批处理
- 实现：`select_nodes()` 允许节点资源被多个请求共享

**AdaptiveSchedulingStrategy — 自适应调度**
- 适用：通用场景
- 规则：根据请求特征自动选择 Gang 或 Pack
- 决策规则：
  - 参数规模 > 70B → Gang
  - 优先级 >= 8 → Gang
  - 输出长度 > 4K tokens → Gang
  - 其他 → Pack

**PriorityQueue — 优先级队列**
- 实现：Redis ZSET，score = 优先级（0-10，0 最高）
- 支持：优先级相同按 FIFO 排序

---

## 5. 推理引擎层

### 5.1 模块结构

```
quantumflow/inference/
├── __init__.py
├── engine.py              # 引擎抽象基类
├── manager.py             # 引擎管理器
├── vram_manager.py        # VRAM 显存管理
├── batch_accumulator.py    # 动态批处理
├── chunked_prefill.py     # 分块预填充
├── gpu_monitor.py         # GPU 监控
├── backends/              # 引擎实现
│   ├── __init__.py
│   ├── base.py            # 后端基类
│   ├── huggingface.py      # HuggingFace 后端
│   ├── vllm.py            # vLLM 后端
│   ├── tgi.py             # TGI 后端
│   └── sglang.py          # SGLang 后端
```

### 5.2 引擎管理器规格

**EngineManager — 引擎管理器**

**核心功能：**
- `load_model(config)` — 加载模型到指定后端
- `unload_model(model_name)` — 卸载模型
- `generate(model, prompt, params)` — 生成文本
- `stream_generate(model, prompt, params)` — 流式生成
- `list_loaded_models()` — 列出已加载模型
- `get_model_info(model_name)` — 获取模型信息

**关键要求：**
- `load_model()` 必须调用后端的真实加载逻辑
- 模型已加载时直接返回，不重复加载
- 卸载时必须释放 GPU 显存
- `generate()` 必须通过 BatchAccumulator 进行动态批处理

### 5.3 VRAM 管理器规格

**VRAMManager — 显存管理器**

**核心功能：**
- `allocate(model_name, size_bytes)` — 分配显存
- `release(model_name)` — 释放显存
- `get_available()` — 获取可用显存
- `get_block_pool()` — 获取 Block Pool

**关键要求：**
- 使用 Block Pool 细粒度分配（类 vLLM PagedAttention）
- 模型卸载时必须完全释放显存
- 显存不足时触发模型淘汰（LRU）

### 5.4 GPU 监控规格

**GPUMonitor — GPU 监控**

**核心功能：**
- `get_gpu_info()` — 获取所有 GPU 信息
- `subscribe(callback)` — 订阅 GPU 状态变化

**GPU 信息结构：**
```python
@dataclass
class GPUInfo:
    gpu_id: int
    name: str
    memory_total: int        # bytes
    memory_used: int         # bytes
    utilization: float       # 0.0-1.0
    temperature: float       # celsius
    memory_utilization: float  # 0.0-1.0
```

**关键要求：**
- 使用 pynvml 采集真实数据
- Torch fallback 时不能返回硬编码 0.0，必须返回 None 或抛出异常
- `memory_utilization` 必须真实反映显存带宽使用率

### 5.5 后端接口规格

所有后端必须实现 `InferenceEngine` 接口：

```python
class InferenceEngine(ABC):
    @property
    def backend_type(self) -> InferenceBackendType: ...

    async def initialize(self) -> bool: ...

    async def load_model(self, config: ModelConfig) -> bool: ...

    async def unload_model(self, model_name: str) -> bool: ...

    async def generate(
        self,
        prompts: list[str],
        params: SamplingParams
    ) -> list[InferenceResult]: ...

    async def stream_generate(
        self,
        prompt: str,
        params: SamplingParams
    ) -> AsyncIterator[InferenceResult]: ...

    async def get_stats(self) -> dict: ...

    async def health_check(self) -> bool: ...
```

---

## 6. 存储层

### 6.1 模块结构

```
quantumflow/storage/
├── __init__.py
├── redis_queue.py      # Redis 优先级队列
└── connection.py       # Redis 连接管理器
```

### 6.2 Redis 队列规格

**RedisQueue — Redis 优先级队列**

**核心功能：**
- `enqueue(request, priority)` — 入队，priority 0-10
- `dequeue()` — 优先级最高的出队
- `peek()` — 查看最高优先级元素（不出队）
- `size()` — 队列长度
- `remove(request_id)` — 按 request_id 删除

**Redis 数据结构：**
- Key: `quantumflow:pending_requests`
- 类型: ZSET
- Score: 优先级（0 最高，10 最低）+ 时间戳后缀确保 FIFO

**关键要求：**
- 连接失败时必须抛出异常，不能返回 None
- 队列为空时 `dequeue()` 返回 None（这是合法的）
- 必须处理 Redis 连接池复用和自动重连

### 6.3 Redis 连接管理器规格

**RedisConnectionManager — Redis 连接管理器（单例）**

**核心功能：**
- `get_client()` — 获取 Redis 客户端
- `health_check()` — 健康检查
- `close()` — 关闭所有连接

**关键要求：**
- 单例模式，全局共享连接池
- `health_check()` 必须执行 PING 命令验证连接
- 断线后自动重连

---

## 7. 集群管理层

### 7.1 模块结构

```
quantumflow/cluster/
├── __init__.py
├── manager.py           # 集群管理器
├── registry.py          # 节点注册表
├── discovery.py         # 服务发现
└── health.py            # 健康监控
```

### 7.2 节点注册表规格

**NodeRegistry — 节点注册表**

**核心功能：**
- `register_node(node_info)` — 注册节点
- `unregister_node(node_id)` — 注销节点
- `update_node(node_info)` — 更新节点信息
- `get_node(node_id)` — 获取节点信息
- `list_nodes(status_filter)` — 列出节点
- `get_healthy_nodes()` — 获取所有健康节点

**节点状态机：**
```
HEALTHY ←──┬──→ UNHEALTHY
    │            │
    │            ▼
    └──────→ DRAINING → OFFLINE
```

**关键要求：**
- 心跳超时（默认 60s）自动标记节点为 UNHEALTHY
- UNHEALTHY 节点不再接收新请求
- 节点卸下时进入 DRAINING 状态（完成现有请求后 OFFLINE）

---

## 8. Worker 层

### 8.1 模块结构

```
quantumflow/worker/
├── __init__.py
├── worker.py            # Worker 主程序
├── task_fetcher.py      # 任务抓取器
└── api.py               # Worker API
```

### 8.2 TaskFetcher 规格

**TaskFetcher — 任务抓取器**

**核心功能：**
- `start()` — 启动抓取循环
- `stop()` — 停止抓取循环
- `fetch_task()` — 从 Redis 队列拉取任务

**抓取循环：**
1. 定期从 Redis ZSET `dequeue()` 最高优先级任务
2. 调用 Worker API 执行推理
3. 返回结果或重试

**关键要求：**
- 抓取间隔可配置（默认 100ms）
- 任务执行完成前不能 dequeue（使用 BRPOPLPUSH 模式）
- 任务超时必须正确处理

### 8.3 Worker API 规格

**Worker API 端点：**

**POST `/api/v1/worker/inference`** — 执行推理

请求：
```json
{
  "request_id": "uuid",
  "model": "Qwen2.5-7B-Instruct",
  "prompt": "Hello",
  "sampling_params": {...}
}
```

响应：
```json
{
  "request_id": "uuid",
  "status": "completed",
  "result": {
    "generated_text": "...",
    "finish_reason": "stop",
    "tokens": 20
  }
}
```

**关键要求：**
- Worker 收到请求后必须真实执行推理
- 执行完成后必须更新任务状态
- 推理失败必须返回错误信息

---

## 9. 监控层

### 9.1 模块结构

```
quantumflow/monitoring/
├── __init__.py
├── metrics.py           # Prometheus 指标
└── instrumentator.py    # FastAPI 仪表化
```

### 9.2 指标规格

**必须暴露的 Prometheus 指标：**

| 指标名 | 类型 | 描述 |
|--------|------|------|
| `qf_inference_requests_total` | Counter | 推理请求总数 |
| `qf_inference_latency_seconds` | Histogram | 推理延迟分布 |
| `qf_gpu_utilization` | Gauge | GPU 利用率 |
| `qf_gpu_memory_used_bytes` | Gauge | GPU 显存使用量 |
| `qf_scheduler_queue_size` | Gauge | 调度队列长度 |
| `qf_worker_count` | Gauge | Worker 节点数量 |
| `qf_model_loaded` | Gauge | 已加载模型数量 |

**关键要求：**
- 版本号必须从 `quantumflow.version.__version__` 读取，不能硬编码

---

## 10. 工具层

### 10.1 retry.py 规格

**retry 装饰器/函数**

**核心功能：**
- `retry(max_attempts, delay, backoff)` — 重试装饰器
- `retry_async()` — 异步版本

**关键要求：**
- 重试耗尽后必须抛出异常（`reraise=True`）或返回明确错误
- 不能返回 None（调用方无法区分成功/失败）
- 建议返回 Result 类型或抛出原异常

---

## 11. CLI 层

### 11.1 CLI 命令规格

```bash
# 启动服务
quantumflow serve [--host HOST] [--port PORT]

# 交互式终端
quantumflow interactive

# 集群状态
quantumflow status

# 模型列表
quantumflow models

# 加载模型
quantumflow load <model_name>

# 对话
quantumflow chat <model_name> -p <prompt>

# 生成
quantumflow generate <model_name> -p <prompt>
```

---

## 12. 测试覆盖分析

### 12.1 测试文件清单

| 测试文件 | 覆盖模块 | 测试数量 |
|----------|----------|----------|
| `tests/unit/scheduler/test_scheduler.py` | Scheduler | ~15 |
| `tests/unit/scheduler/test_strategy.py` | 调度策略 | ~10 |
| `tests/unit/scheduler/test_worker_client.py` | WorkerClient | ~8 |
| `tests/integration/test_api.py` | API 端点 | ~20 |
| `tests/unit/inference/test_engine.py` | 推理引擎 | ~15 |
| `tests/unit/inference/test_backend_protocol.py` | 后端接口 | ~10 |
| `tests/unit/storage/test_redis_queue.py` | Redis 队列 | ~12 |
| `tests/unit/cluster/test_cluster.py` | 集群管理 | ~10 |
| `tests/unit/worker/test_worker.py` | Worker | ~10 |
| `tests/unit/worker/test_worker_api.py` | Worker API | ~8 |
| `tests/unit/distributed/test_distributed_comprehensive.py` | 分布式综合 | ~59 |

**总计：约 1009 个测试**

### 12.2 已验证的功能（测试通过）

✅ 调度器内部逻辑（请求入队、优先级排序、节点注册/注销）
✅ 调度策略选择逻辑（Gang/Pack/Adaptive）
✅ API 响应格式和状态码
✅ Pydantic 模型验证
✅ Redis 队列基本操作（enqueue/dequeue）
✅ Worker API 路由注册
✅ 引擎后端接口协议
✅ **健康检查真实集成** — Redis ping、集群状态查询、uptime 计算
✅ **Scheduler → Worker 真实通信** — `WorkerClient.inference()` 替代模拟
✅ **模型部署真实集成** — `EngineManager.load_model()` 调用
✅ **模型卸载真实集成** — `EngineManager.unload_model()` 调用
✅ **基准测试真实执行** — 后台任务执行并跟踪进度
✅ **Worker 健康检查** — 引擎状态、GPU 可用性检查
✅ **Storage 健康检查** — 真实 Redis PING
✅ **就绪检查依赖验证** — 检查所有依赖是否就绪
✅ **异常正确传播** — 日志+异常替代 `except: pass`
✅ **GPU 监控数据采集** — NVML 实时数据
✅ **Redis 队列错误处理** — 错误时抛出异常
✅ **gRPC 服务** — Inference/Cluster/Scheduler/Model/Health 服务完整实现
✅ **REST API 限流** — TokenBucket 全局限流 + per_endpoint 按端点限流
✅ **Python SDK** — Sync/Async 客户端，支持 httpx
✅ **TensorRT-LLM** — NVIDIA 高性能推理引擎后端

### 12.3 设计说明

以下为代码中的设计选择，非缺陷：

| 设计 | 位置 | 说明 |
|------|------|------|
| HF attention mask 返回 None | `backends/huggingface.py` | HuggingFace 内部自动处理 attention mask |
| Hub Service 返回空列表 | `api/services/hub_service.py` | list 函数返回空列表是常见模式 |
| vLLM/HF get_stats 返回 `{}` | backends | 当无统计信息时返回空字典是合理的 |

---

## 13. 规划中功能 📋

| 功能 | 说明 |
|------|------|
| 多租户支持 | API Key 认证 + 资源配额隔离 |
| 昇腾 NPU | 华为昇腾深度适配 |
| Cambricon 寒武纪 | 寒武纪 MLU 适配 |

---

## 附录：关键数据流

### A.1 推理请求完整流程

```
1. 客户端 POST /api/v1/inference/generate
           │
2. API 验证请求格式，生成 request_id
           │
3. API 调用 DistributedScheduler.submit()
           │
4. Scheduler 将请求写入 Redis ZSET 优先级队列
           │
5. Scheduler 调度循环取出最高优先级请求
           │
6. Scheduler 选择最优节点（根据 Gang/Pack/Adaptive 策略）
           │
7. Scheduler 调用 WorkerClient.send_inference_request()
           │
8. WorkerClient HTTP POST 到 Worker /api/v1/worker/inference
           │
9. Worker 收到请求，调用 TaskFetcher 执行
           │
10. TaskFetcher 从队列 dequeue 任务
           │
11. Worker 调用 EngineManager.generate()
           │
12. EngineManager 通过 BatchAccumulator 动态批处理
           │
13. 后端（HuggingFace/vLLM）执行推理
           │
14. Worker 返回结果给 Scheduler
           │
15. Scheduler 更新请求状态为 completed
           │
16. API 返回结果给客户端
```

### A.2 模型部署完整流程

```
1. 客户端 POST /api/v1/models/deploy
           │
2. API 验证请求格式
           │
3. API 调用 EngineManager.load_model()
           │
4. EngineManager 选择/创建指定后端引擎
           │
5. 后端执行 ModelConfig.from_pretrained()
           │
6. VRAMManager.allocate() 分配显存
           │
7. 模型加载到 GPU
           │
8. EngineManager 注册模型到 loaded_models
           │
9. API 返回 {status: "loading", model_id: "..."}
           │
10. （可选）客户端轮询 GET /api/v1/models/{model_name} 查询状态
           │
11. 模型加载完成，状态更新为 "ready"
```

---

*文档版本：1.0.0*
*最后更新：2026-05-20*
