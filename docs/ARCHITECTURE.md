# QuantumFlow 系统设计

> 项目定位、模块划分、关键设计、数据流、设计决策。
> 本文只讲"为什么这样设计"与"模块如何协作"，不展开 API 端点、CLI、测试内容（分别见 [API.md](./API.md) / [DEPLOYMENT.md](./DEPLOYMENT.md) / [TESTING.md](./TESTING.md)）。

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
10. [多租户层](#10-多租户层)
11. [关键数据流](#11-关键数据流)
12. [设计决策](#12-设计决策)

---

## 1. 项目概述

### 1.1 项目定位

QuantumFlow 是一个**生产级的分布式大模型推理平台**，目标是把"调度 AI 推理任务"做成像"调度 Kubernetes Pods"一样自然。

### 1.2 核心价值

| 特性 | 说明 |
|------|------|
| 智能调度 | Gang / Pack / Adaptive 多策略自动选择 |
| 分布式部署 | Redis 队列 + Worker 节点，Controller 与 Worker 完全解耦 |
| 多后端 | vLLM / HuggingFace / TGI / SGLang / TensorRT-LLM 统一接口 |
| GPU 优化 | BatchAccumulator / Chunked Prefill / Block VRAM 精细管理 |
| 多租户 | 租户间显存隔离、资源配额、API Key 认证、限流 |
| 双协议 | REST 通用 + gRPC 高性能 |

### 1.3 技术栈

```
应用层      FastAPI + Pydantic + Uvicorn
调度层      asyncio + 自定义调度算法 + Redis ZSET
执行层      vLLM / PyTorch / CUDA
存储层      Redis（队列、租户、状态）
RPC 协议    gRPC + Protocol Buffers
监控层      Prometheus + Grafana
基础设施    Docker / Kubernetes
```

### 1.4 设计原则

| 原则 | 含义 |
|------|------|
| 接口与实现分离 | 所有可替换组件（后端、调度策略）通过抽象基类注入 |
| 真实优先 | 不允许用 mock 字典、占位实现"假装"模块工作 |
| 异常显式 | 错误必须抛出或返回明确的错误对象，禁止静默 `None` |
| 状态可观测 | 关键状态（节点、模型、队列、显存）可被查询、可被监控 |
| 插件化 | 新增后端/策略/拦截器无需修改核心代码 |

---

## 2. 模块架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        QuantumFlow Platform                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     接入层 (API Gateway)                          │  │
│  │  REST API (FastAPI)  │  gRPC API  │  CLI  │  Python SDK        │  │
│  │  中间件：Auth → RateLimit → TenantContext                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      调度层 (Scheduler)                          │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │  Priority Queue  │  Strategy Selector  │  Dispatcher  │  │  │
│  │  │  Gang / Pack / Adaptive 策略                                │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────┼──────────────────────────────┐  │  │
│  │  │  DistributedScheduler  │  Redis Queue  │  Worker Client│  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────┼────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      执行层 (Inference)                          │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │  EngineManager  │  VRAMManager  │  BatchAccumulator    │  │  │
│  │  │  GPUMonitor     │  SharedBatchCoordinator              │  │  │
│  │  │  ChunkedPrefill                                          │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────┼──────────────────────────────┐  │  │
│  │  │  HuggingFace  │  vLLM  │  TGI  │  SGLang  │  TensorRT  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────┼────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      存储层 (Storage)                             │  │
│  │  RedisQueue (ZSET 优先级)  │  RedisConnectionManager (单例)   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      集群管理层 (Cluster)                         │  │
│  │  NodeRegistry  │  ServiceDiscovery  │  HealthMonitor          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      Worker 层                                   │  │
│  │  TaskFetcher (Redis 拉取)  │  WorkerNode API  │  Worker gRPC  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      监控层 (Monitoring)                          │  │
│  │  Prometheus Metrics  │  FastAPI Instrumentator                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                       │
│  ┌────────────────────────────┼────────────────────────────────────┐  │
│  │                      多租户层 (Multi-Tenant)                      │  │
│  │  TenantAuthMiddleware  │  RateLimitMiddleware  │  TenantService│  │
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
├── server.py                # FastAPI 应用入口 + lifespan
├── models/                  # Pydantic 请求/响应模型
│   ├── __init__.py
│   └── models.py
├── routes/                  # 路由模块
│   ├── health.py            # 健康检查
│   ├── models.py            # 模型管理
│   ├── cluster.py           # 集群管理
│   ├── inference.py         # 推理端点（同步/流式/批量/Chat）
│   ├── scheduler.py         # 调度可视化
│   ├── worker.py            # Worker API
│   └── tenants.py           # 多租户管理
├── services/                # 业务服务
│   ├── __init__.py
│   └── hub_service.py       # HuggingFace Hub 服务
└── middleware/              # 中间件
    ├── auth.py              # TenantAuthMiddleware
    └── rate_limit.py        # RateLimitMiddleware
```

### 3.2 请求处理链

```
Request
  → TenantAuthMiddleware    提取 X-API-Key，SHA256 校验，注入 TenantContext
  → RateLimitMiddleware     Token Bucket（全局 + per-tenant + per-endpoint）
  → 路由 Handler            业务逻辑
  → Response                统一 JSON 格式
```

### 3.3 关键设计

- **统一响应格式**：所有 API 返回同一 JSON 信封（成功/失败统一结构，详见 [API.md](./API.md)）
- **请求追踪**：自动生成 `X-Request-ID` header，贯穿全链路
- **OpenAPI 自动生成**：FastAPI 原生支持，`/docs` 直接可访问
- **流式响应**：SSE（Server-Sent Events）协议
- **gRPC 共存**：与 REST 在同一进程，通过 lifespan 启动/停止

### 3.4 中间件关键约束

| 中间件 | 关键约束 |
|--------|----------|
| TenantAuth | 必须用 `async-safe` 的 `ContextVar` 注入租户上下文；任何状态校验失败必须立即 401/403 |
| RateLimit | 令牌耗尽时必须返回 429 并保留 `Retry-After`；不允许静默丢弃请求 |

---

## 4. 调度层

### 4.1 模块结构

```
quantumflow/scheduler/
├── __init__.py
├── scheduler.py            # 基础调度器（单机）
├── distributed.py           # DistributedScheduler
├── worker_client.py         # Worker HTTP 客户端（httpx.AsyncClient）
├── dispatcher.py            # 调度分发器
├── strategy/                # 调度策略
│   ├── __init__.py
│   ├── base.py             # SchedulingStrategy 抽象基类
│   ├── gang.py             # GangSchedulingStrategy
│   ├── pack.py             # PackSchedulingStrategy
│   ├── adaptive.py         # AdaptiveSchedulingStrategy
│   └── priority.py         # 优先级队列
└── node.py                  # NodeResource / NodeInfo
```

### 4.2 调度器职责

| 类 | 职责 |
|----|------|
| `Scheduler` | 单机版调度器：管理请求队列、节点注册、调度循环 |
| `DistributedScheduler` | 分布式版：通过 Redis ZSET 协调，通过 WorkerClient 派发 |
| `WorkerClient` | Controller → Worker 的 HTTP 客户端（异步 + 超时 + 重试） |
| `SchedulingStrategy` | 策略抽象基类；定义 `can_handle` / `select_nodes` |

### 4.3 调度循环逻辑

```
loop:
  1. 从优先级队列取出最高优先级请求
  2. 根据请求特征选择策略（Gang / Pack / Adaptive）
  3. 策略选择最优节点
  4. 分配 GPU 资源
  5. 发送请求到 Worker 节点（不允许 _simulate_execution 占位）
  6. 跟踪状态：pending → running → completed / failed
  7. fire-and-forget 模式下失败必须有显式回调
```

### 4.4 DistributedScheduler 关键约束

- Redis 队列 key：`quantumflow:pending_requests`，类型 `ZSET`
- 队列元素：JSON 序列化的请求 + 优先级 score（0 最高，10 最低）
- Worker 通过 `TaskFetcher` 拉取任务，调度器不直接推送
- 调度器通过 `WorkerClient` 接收执行结果回调
- 节点故障时调度器必须重新选择节点或返回错误

### 4.5 WorkerClient 关键约束

- 必须使用 `httpx.AsyncClient` 异步 HTTP 调用
- 必须处理连接超时与重试
- Worker 不可用时必须正确报错，**不允许静默失败**

### 4.6 调度策略

#### 4.6.1 Gang 调度

**适用场景**：>30B 参数的大模型（TP ≥ 4）

```
class GangSchedulingStrategy:
    规则：所有 GPU 必须同时分配，要么全部分配要么不分配
    优势：保证张量并行的完整性，最小化 GPU 间通信开销
    决策：select_nodes() 必须检查是否有足够 GPU 容纳整个模型
```

#### 4.6.2 Pack 调度

**适用场景**：<30B 小模型、高并发场景

```
class PackSchedulingStrategy:
    规则：允许多个请求共享同一 GPU，动态批处理
    优势：最大化 GPU 利用率，支持更高并发
    决策：select_nodes() 允许节点资源被多个请求共享
```

#### 4.6.3 Adaptive 调度

**适用场景**：通用场景，AI 自动决策

| 条件 | 策略 |
|------|------|
| 参数规模 > 70B | Gang |
| 优先级 ≥ 8 | Gang |
| 输出长度 > 4K tokens | Gang |
| 其他 | Pack |

策略可扩展：支持自定义规则（plugin）。

#### 4.6.4 PriorityQueue

| 字段 | 值 |
|------|------|
| 实现 | Redis ZSET |
| Score | 优先级（0 最高，10 最低） |
| 同优先级 | 按 FIFO 排序（时间戳后缀） |
| 接口 | enqueue / dequeue / peek / size / remove |

---

## 5. 推理引擎层

### 5.1 模块结构

```
quantumflow/inference/
├── __init__.py
├── engine.py                # InferenceEngine 抽象基类 + ModelConfig / InferenceResult
├── manager.py               # EngineManager（多后端统一管理）
├── vram_manager.py          # VRAMManager（Block Pool）
├── batch_accumulator.py     # 动态批处理
├── batch_coordinator.py     # SharedBatchCoordinator（多模型共享）
├── priority_queue.py        # 优先级队列
├── batch_scheduler.py       # Per-GPU BatchScheduler
├── chunked_prefill.py       # 分块预填充
├── gpu_monitor.py           # GPUMonitor（NVML 采集）
├── sampling.py              # SamplingParams
└── backends/                # 引擎实现
    ├── __init__.py
    ├── base.py              # Backend 抽象基类
    ├── huggingface.py       # HF Transformers
    ├── vllm.py              # vLLM (PagedAttention)
    ├── tgi.py               # TGI（外部服务）
    ├── sglang.py            # SGLang（外部服务）
    └── tensorrt.py          # TensorRT-LLM
```

### 5.2 InferenceEngine 抽象接口

```
class InferenceEngine(ABC):
    @property
    def backend_type(self) -> InferenceBackendType: ...

    async def initialize(self) -> bool: ...
    async def load_model(self, config: ModelConfig) -> bool: ...
    async def unload_model(self, model_name: str) -> bool: ...
    async def generate(prompts, params) -> list[InferenceResult]: ...
    async def stream_generate(prompt, params) -> AsyncIterator[InferenceResult]: ...
    async def get_stats(self) -> dict: ...
    async def health_check(self) -> bool: ...
```

**关键约束**：
- 所有后端必须实现此接口
- `load_model` 必须真实加载权重到 GPU，不允许只注册到字典
- `unload_model` 必须释放显存并从后端引擎卸载
- `generate` 必须经过 `BatchAccumulator` 动态批处理

### 5.3 EngineManager

**核心职责**：多后端统一管理，按模型名路由到对应后端。

**关键约束**：
- `load_model()` 必须调用后端真实的加载逻辑
- 模型已加载时直接返回，不重复加载
- `unload_model()` 必须释放 GPU 显存
- `generate()` 走 `BatchAccumulator` → 共享协调器

### 5.4 VRAMManager（Block Pool）

**核心职责**：类 vLLM PagedAttention 的细粒度显存管理。

**关键约束**：
- 使用 Block Pool 细粒度分配
- 模型卸载时必须完全释放显存
- 显存不足时触发模型淘汰（LRU）
- 暴露 `get_vram_utilization()` 供动态批处理查询

### 5.5 BatchAccumulator

**核心职责**：50ms 窗口动态批处理。

| 维度 | 设计 |
|------|------|
| 触发 | 时间窗口（默认 50ms）或达到 `max_batch_size` |
| 优先级 | 缓冲区按 (priority, submit_time) 排序 |
| 公平性 | Anti-starvation：每 N 个高优请求后强制处理低优 |
| 动态尺寸 | 根据 VRAM 利用率动态调整 max_batch_size |
| 多模型 | 通过 `SharedBatchCoordinator` 跨模型共享 GPU 资源 |

### 5.6 GPUMonitor

**核心职责**：通过 pynvml 实时采集 GPU 状态。

**GPUInfo 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| gpu_id | int | GPU 编号 |
| name | str | 设备名 |
| memory_total | int (bytes) | 总显存 |
| memory_used | int (bytes) | 已用显存 |
| utilization | float (0-1) | 计算利用率 |
| temperature | float (°C) | 温度 |
| memory_utilization | float (0-1) | 显存带宽利用率 |

**关键约束**：
- 使用 pynvml 采集真实数据
- Torch fallback 时不能返回硬编码 0.0；必须返回 `None` 或抛异常
- `memory_utilization` 必须真实反映显存带宽使用率

### 5.7 ChunkedPrefill

**核心职责**：把长 prompt 切成多块预填充，混合到 decode 批次中，提高 GPU 利用率。

参考 vLLM 分块预填充设计。Phase 1 实现存在缺陷（见 Dev 待办），需重做。

---

## 6. 存储层

### 6.1 模块结构

```
quantumflow/storage/
├── __init__.py
├── redis_queue.py          # RedisQueue（ZSET 优先级队列）
└── connection.py           # RedisConnectionManager（单例）
```

### 6.2 RedisQueue

**核心接口**：

| 方法 | 说明 |
|------|------|
| `enqueue(request, priority)` | 入队，priority 0-10 |
| `dequeue()` | 优先级最高的出队；空时返回 `None` |
| `peek()` | 查看队首（不出队） |
| `size()` | 队列长度 |
| `remove(request_id)` | 按 request_id 删除 |

**Redis 数据结构**：

```
Key:   quantumflow:pending_requests
Type:  ZSET
Score: 优先级（0 最高，10 最低）+ 时间戳后缀保证 FIFO
Value: JSON 序列化的请求
```

**关键约束**：
- 连接失败时必须抛出异常，**不允许返回 None**
- 队列为空时 `dequeue()` 返回 `None`（合法状态）
- 必须处理连接池复用和自动重连

### 6.3 RedisConnectionManager（单例）

**核心接口**：

| 方法 | 说明 |
|------|------|
| `get_client()` | 获取 Redis 客户端 |
| `health_check()` | 执行 PING，验证连接 |
| `close()` | 关闭所有连接 |

**关键约束**：
- 全局单例，共享连接池
- `health_check()` 必须执行真实 PING
- 断线后自动重连

---

## 7. 集群管理层

### 7.1 模块结构

```
quantumflow/cluster/
├── __init__.py
├── manager.py              # ClusterManager（顶层入口）
├── registry.py             # NodeRegistry
├── discovery.py            # ServiceDiscovery
├── health.py               # HealthMonitor
└── node.py                 # NodeInfo / NodeStatus
```

### 7.2 NodeRegistry

**核心接口**：

| 方法 | 说明 |
|------|------|
| `register_node(node_info)` | 注册节点 |
| `unregister_node(node_id)` | 注销节点 |
| `update_node(node_info)` | 更新节点信息（含心跳） |
| `get_node(node_id)` | 获取节点信息 |
| `list_nodes(status_filter)` | 按状态过滤 |
| `get_healthy_nodes()` | 获取所有健康节点 |

### 7.3 节点状态机

```
         JOINING
            │
            ▼
        HEALTHY ──────┬──▶ DRAINING ──▶ OFFLINE
            │         │
            │         ▼
            └───▶ UNHEALTHY
```

| 状态 | 说明 |
|------|------|
| JOINING | 节点启动中，未通过健康检查 |
| HEALTHY | 正常运行，接收新请求 |
| DRAINING | 排空现有请求，不接收新请求 |
| UNHEALTHY | 心跳超时，不再接收请求 |
| OFFLINE | 已注销 / 主动下线 |

**关键约束**：
- 心跳超时（默认 60s）自动标记 `UNHEALTHY`
- `UNHEALTHY` 节点不再接收新请求
- 进入 `DRAINING` 后完成现有请求才转 `OFFLINE`

### 7.4 ServiceDiscovery

**职责**：Worker 启动时自动注册到 Controller；Controller 主动发现集群中的 Worker。

**机制**：
- 启动发现：Worker 启动时调用 Controller `RegisterNode`
- 持续发现：基于 Redis 订阅（`qf:cluster:events`）实现事件驱动
- 失败发现：心跳超时自动转 UNHEALTHY（见 7.3）

### 7.5 HealthMonitor

**职责**：周期性检查节点健康、检测 GPU 异常、触发告警。

**检查项**：
- 心跳超时
- GPU 温度 / 利用率 / 显存
- 引擎后端健康（`engine.health_check()`）
- Redis 连接

---

## 8. Worker 层

### 8.1 模块结构

```
quantumflow/worker/
├── __init__.py
├── worker.py               # WorkerNode（主程序）
├── task_fetcher.py         # TaskFetcher（拉取任务）
├── api.py                  # WorkerNode HTTP API
├── grpc_service.py         # Worker gRPC Service
└── registry.py             # WorkerRegistry（Controller 侧）
```

### 8.2 TaskFetcher

**职责**：从 Redis 拉取任务并执行。

**抓取循环**：

```
loop:
  1. 从 Redis ZSET dequeue() 最高优先级任务
  2. 调用 Worker API / EngineManager 执行推理
  3. 回调 Controller 报告结果
  4. 处理异常与重试
```

**关键约束**：
- 抓取间隔可配置（默认 100ms）
- 任务执行完成前不能 dequeue 下一个（使用 BRPOPLPUSH 模式防丢失）
- 任务超时必须正确处理（移到 dead letter 或重试）

### 8.3 WorkerNode

**职责**：执行推理、维护本地 EngineManager、向 Controller 汇报。

**关键组件**：
- 本地 `EngineManager`（多后端）
- 本地 `VRAMManager`
- 本地 `BatchAccumulator`
- 心跳上报（向 Controller）
- HTTP / gRPC 接收推理请求

### 8.4 Worker 与 Controller 通信

| 方向 | 协议 | 内容 |
|------|------|------|
| Controller → Worker | HTTP / gRPC | 推理请求、模型加载/卸载 |
| Worker → Controller | HTTP | 心跳、结果回调、状态上报 |
| Controller ↔ Worker | Redis | 任务队列（`TaskFetcher` 消费） |

---

## 9. 监控层

### 9.1 模块结构

```
quantumflow/monitoring/
├── __init__.py
├── metrics.py              # Prometheus 指标定义
└── instrumentator.py       # FastAPI 仪表化（HTTP 指标）
```

### 9.2 指标规格

| 指标 | 类型 | 描述 |
|------|------|------|
| `qf_inference_requests_total` | Counter | 推理请求总数（按 model、status 标签） |
| `qf_inference_latency_seconds` | Histogram | 推理延迟分布 |
| `qf_gpu_utilization` | Gauge | GPU 利用率（按 gpu_id） |
| `qf_gpu_memory_used_bytes` | Gauge | GPU 显存使用量（按 gpu_id） |
| `qf_scheduler_queue_size` | Gauge | 调度队列长度（按 priority） |
| `qf_worker_count` | Gauge | Worker 节点数量（按 status） |
| `qf_model_loaded` | Gauge | 已加载模型数量（按 backend） |

**关键约束**：
- 版本号必须从 `quantumflow.version.__version__` 读取，**不允许硬编码**
- 指标必须可按标签维度切片（按模型、后端、租户、节点）
- Histogram buckets 需覆盖典型延迟范围：[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

---

## 10. 多租户层

### 10.1 模块结构

```
quantumflow/tenant/
├── __init__.py
├── auth.py                 # TenantAuthMiddleware
├── rate_limit.py           # RateLimitMiddleware（per-tenant 维度）
├── service.py              # TenantService（CRUD、配额管理）
├── models.py               # Tenant / TenantQuota 数据模型
└── routes.py               # 租户管理 API
```

### 10.2 租户状态

| 状态 | 行为 |
|------|------|
| active | 正常服务 |
| suspended | 暂停（认证通过但拒绝服务） |
| deleted | 软删除（拒绝所有请求） |

### 10.3 租户数据模型

| 字段 | 类型 | 说明 |
|------|------|------|
| tenant_id | str | 租户唯一标识 |
| api_key_hash | str | SHA256(api_key)，不存明文 |
| name | str | 租户名 |
| status | enum | active / suspended / deleted |
| quota.requests_per_minute | int | 每分钟请求上限 |
| quota.burst | int | 突发容量 |
| quota.concurrent | int | 最大并发请求数 |
| created_at | timestamp | 创建时间 |

### 10.4 请求处理链

```
Request
  → TenantAuthMiddleware
      1. 提取 X-API-Key header
      2. SHA256 哈希
      3. 从 Redis 加载租户信息
      4. 设置 TenantContext（async-safe ContextVar）
      5. 校验状态（active / suspended / deleted）
  → RateLimitMiddleware
      6. Token Bucket 算法
      7. 全局限流 + per-endpoint + per-tenant 三层
      8. 扣减租户配额
  → 租户感知调度
      9. DistributedScheduler 强制租户配额
     10. VRAMManager 按租户隔离
```

### 10.5 Redis Key 模式

| Key | 类型 | 说明 |
|-----|------|------|
| `qf:tenant:{api_key_hash}` | Hash | 租户数据 |
| `qf:tenant:id:{tenant_id}` | String | ID → api_key_hash 索引 |
| `qf:tenant:ids` | Set | 所有租户 ID |
| `qf:concurrent:{tenant_id}` | String | 并发请求计数（INCR/DECR） |
| `qf:usage:{tenant_id}:today` | String | 日使用量（带 TTL） |

---

## 11. 关键数据流

### 11.1 推理请求完整流程

```
1.  客户端 POST /api/v1/inference/generate
2.  TenantAuthMiddleware 校验 API Key，注入 TenantContext
3.  RateLimitMiddleware 校验配额
4.  API 验证请求格式，生成 request_id
5.  API 调用 DistributedScheduler.submit()
6.  Scheduler 将请求写入 Redis ZSET（带优先级 score）
7.  Scheduler 调度循环取出最高优先级请求
8.  Scheduler 选择策略（Gang / Pack / Adaptive）→ 选节点
9.  Scheduler 调用 WorkerClient.send_inference_request()
10. WorkerClient HTTP POST 到 Worker /api/v1/worker/inference
11. Worker 收到请求，调用 EngineManager.generate()
12. EngineManager 通过 BatchAccumulator 动态批处理
13. 后端（HuggingFace / vLLM / TGI / SGLang）执行推理
14. Worker 返回结果给 Controller
15. Scheduler 更新请求状态为 completed
16. API 返回结果给客户端
```

### 11.2 模型部署完整流程

```
1.  客户端 POST /api/v1/models/deploy
2.  API 验证请求格式
3.  API 调用 EngineManager.load_model()
4.  EngineManager 选择 / 创建指定后端引擎
5.  后端执行 ModelConfig.from_pretrained()
6.  VRAMManager.allocate() 分配显存
7.  模型权重加载到 GPU
8.  EngineManager 注册到 loaded_models
9.  API 返回 {status: "loading", model_id: "..."}
10. 客户端轮询 GET /api/v1/models/{model_name} 查询状态
11. 模型加载完成，状态更新为 "ready"
```

### 11.3 流式推理流程

```
客户端 ──POST /inference/stream──▶ Scheduler
                                     │
                                     ▼
                                Worker 分配 + 开始流式推理
                                     │
                                     ▼
                                 Token N ──SSE──▶ 客户端
                                 Token N+1 ──SSE──▶ 客户端
                                 ...
                                 [is_final: true] ──SSE──▶ 客户端
                                 [DONE] ──SSE──▶ 客户端
```

### 11.4 多租户隔离流程

```
Request (X-API-Key: xxx)
  → SHA256(xxx) = hash
  → Redis GET qf:tenant:{hash}
  → 加载 Tenant 对象
  → 校验 status == active
  → RateLimitMiddleware 检查 INCR qf:concurrent:{tenant_id}
  → 若超并发上限：DECR + 返回 429
  → 业务执行
  → DECR qf:concurrent:{tenant_id}
  → INCR qf:usage:{tenant_id}:today + EXPIRE 86400
```

---

## 12. 设计决策

### 12.1 关键选型

| 决策 | 备选 | 选择 | 原因 |
|------|------|------|------|
| 队列实现 | List / Stream / ZSET | ZSET | 原生有序、自动持久化、分布式友好 |
| 优先级队列 | 多级队列 | ZSET + score | 单一结构 + 灵活优先级 |
| 显存管理 | 整模型 / 静态切片 | Block Pool | 类 vLLM PagedAttention，减少碎片 |
| 调度协议 | gRPC / HTTP | HTTP + gRPC 双协议 | HTTP 通用、gRPC 高性能；gRPC 主要服务内部 Worker-Controller |
| 限流算法 | 固定窗口 / 滑动窗口 / 令牌桶 | 令牌桶 | 支持突发、平滑限流 |
| 后端调度 | 多后端并存 | 单模型绑定单后端 | 避免跨后端语义差异 |
| 节点发现 | 主动注册 / 服务发现中心 | Redis 事件 + 主动注册 | 与队列共用 Redis，减少组件 |

### 12.2 一致性约束

| 约束 | 含义 |
|------|------|
| **异常显式** | 错误必须抛出或返回明确的错误对象；禁止静默 `None`、禁止吞 `except: pass` |
| **真实优先** | 不允许用 mock 字典或占位实现假装模块工作；`load_model` 必须真实加载权重 |
| **超时必处理** | 所有外部调用（HTTP / gRPC / Redis）必须配置超时 |
| **健康可查** | 任何外部依赖（Redis、引擎、Worker）必须能返回健康状态 |
| **状态可观测** | 关键状态变化（节点状态、模型加载、调度决策）必须可被查询 |

### 12.3 已知缺陷（待改进）

| 缺陷 | 位置 | 改进方向 |
|------|------|----------|
| `ChunkedPrefill` 实现有 bug | `inference/chunked_prefill.py` | 参考 vLLM 分块逻辑重做 |
| `BatchAccumulator` 单模型绑定 | `inference/batch_accumulator.py` | 引入 `SharedBatchCoordinator` 多模型共享 |
| `flash-attn` 与当前 CUDA 不匹配 | 环境依赖 | 升级 CUDA 至 13.0 或用 `xformers` 替代 |
| `ChunkedPrefill` 与 HF streaming 集成不充分 | 推理层 | 解耦 streaming 与 batching |

### 12.4 后续演进方向

| 方向 | 优先级 |
|------|--------|
| 国产 NPU 支持（昇腾、寒武纪） | P2（长期） |
| 模型副本 + 自动故障转移 | P1 |
| 实时 Block 追踪 + Eviction 事件记录 | P2 |
| BatchAccumulator 增强（动态 batch_size + 优先级感知） | P1（已设计，见 `BATCH_ACCUMULATOR_ENHANCEMENT_PLAN.md`，已随文档清理删除） |

---

*接口细节见 [API.md](./API.md)；部署步骤见 [DEPLOYMENT.md](./DEPLOYMENT.md)；测试策略见 [TESTING.md](./TESTING.md)。*
