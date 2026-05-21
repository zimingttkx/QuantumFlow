# QuantumFlow 开发规范与状态

> 本文档记录 QuantumFlow 项目的功能开发状态，作为 README.md 的补充。

---

## 一、已完成功能 ✅

### 1. 核心架构

| 模块 | 状态 | 说明 |
|------|------|------|
| REST API (FastAPI) | ✅ 已完成 | 所有路由已实现，98% 测试覆盖率 |
| CLI 工具 | ✅ 已完成 | 交互式终端 + 命令行，测试覆盖率 19% |
| 配置系统 | ✅ 已完成 | YAML 配置、多环境支持 |
| 日志系统 | ✅ 已完成 | structlog 集成 |

### 2. 推理引擎

| 后端 | 状态 | 说明 |
|------|------|------|
| HuggingFace | ✅ 已完成 | 动态批处理 + torch.compile + Chunked Prefill |
| vLLM | ✅ 已完成 | PagedAttention + Continuous Batching |
| TGI | ✅ 已完成 | Text Generation Inference 协议支持 |
| SGLang | ✅ 已完成 | Structured Generation 语言支持 |

### 3. 调度系统

| 组件 | 状态 | 说明 |
|------|------|------|
| Gang Scheduler | ✅ 已完成 | 大模型 All-or-Nothing 调度 |
| Pack Scheduler | ✅ 已完成 | 小模型紧凑批处理 |
| Adaptive Strategy | ✅ 已完成 | 基于 AI 的自适应策略选择 |
| Priority Queue | ✅ 已完成 | Redis ZSET 优先级队列 |
| DistributedScheduler | ✅ 已完成 | Redis 队列 + Worker HTTP 通信 |

### 4. 集群管理

| 组件 | 状态 | 说明 |
|------|------|------|
| Node Registry | ✅ 已完成 | 节点注册/注销 |
| Heartbeat Monitor | ✅ 已完成 | 节点健康检查 + 超时处理 |
| Service Discovery | ✅ 已完成 | Worker 自动发现 |

### 5. Worker 节点

| 组件 | 状态 | 说明 |
|------|------|------|
| TaskFetcher | ✅ 已完成 | Redis 队列拉取任务 |
| WorkerNode | ✅ 已完成 | HTTP API 服务 |
| WorkerRegistry | ✅ 已完成 | Controller 注册/注销 |

### 6. 存储层

| 组件 | 状态 | 说明 |
|------|------|------|
| RedisQueue | ✅ 已完成 | ZSET 优先级队列实现 |
| RedisConnection | ✅ 已完成 | 单例 + 健康检查 + 自动重连 |

### 7. GPU 优化

| 组件 | 状态 | 说明 |
|------|------|------|
| VRAM Manager | ✅ 已完成 | BlockPool 细粒度显存管理 |
| BatchAccumulator | ✅ 已完成 | 50ms 窗口动态批处理 |
| GPU Monitor | ✅ 已完成 | NVML 采集 |
| Idle Eviction | ✅ 已完成 | 空闲模型自动淘汰 |

### 8. 本地/分布式自适应

| 功能 | 状态 | 说明 |
|------|------|------|
| 自动模式切换 | ✅ 已完成 | 无 Worker 时本地推理，有 Worker 时分布式调度 |
| LocalGenerate | ✅ 已完成 | 本地 BatchAccumulator 直连 |
| DistributedGenerate | ✅ 已完成 | 调度器 + Redis 队列 + 结果等待 |

### 9. 测试覆盖

| 测试类型 | 状态 | 说明 |
|------|------|------|
| 单元测试 | ✅ 已完成 | 核心模块覆盖率 ~98% |
| 集成测试 | ✅ 已完成 | API/Health/E2E 场景 |
| 分布式测试 | ✅ 已完成 | 调度器/队列/Worker 测试 |

---

## 二、规划中功能 📋

### 高优先级

| 功能 | 状态 | 说明 |
|------|------|------|
| gRPC API | ✅ 已完成 | 高性能 RPC 接口，降低延迟 87% 覆盖率 |
| REST API 限流 | ✅ 已完成 | TokenBucket 全局限流 + per_endpoint 按端点限流 |
| Python SDK | ✅ 已完成 | Sync/Async 客户端，支持 httpx |

### gRPC 开发计划 (docs/grpc_development_plan.md)

**Phase 1: 基础设施**
- [x] TODO-1.1 创建 `quantumflow/grpc/` 目录结构
- [x] TODO-1.2 安装 grpcio, grpcio-tools, grpcio-reflection 依赖
- [x] TODO-1.3 生成 Proto 代码 (`quantumflow.proto` → `*_pb2.py`, `*_pb2_grpc.py`)
- [x] TODO-1.4 验证 Proto 生成测试

**Phase 2: 异常和拦截器**
- [x] TODO-2.1 定义 gRPC 异常类 (GrpcQuantumFlowError, NodeNotFoundError, etc.)
- [x] TODO-2.2 日志拦截器 (LoggingInterceptor)
- [x] TODO-2.3 认证拦截器 (AuthInterceptor)
- [x] TODO-2.4 监控拦截器 (MetricsInterceptor)
- [x] TODO-2.5 限流拦截器 (RateLimitInterceptor)

**Phase 3: 服务实现**
- [x] TODO-3.1 InferenceService (同步/流式/批量推理)
- [x] TODO-3.2 ClusterService (注册/注销/心跳/列表)
- [x] TODO-3.3 SchedulerService (提交/取消/状态查询)
- [x] TODO-3.4 ModelManagementService (加载/卸载/列表)
- [x] TODO-3.5 HealthService (健康检查/流式监控)

**Phase 4: 服务端启动**
- [x] TODO-4.1 GrpcServer 封装类
- [x] TODO-4.2 整合到 FastAPI 主服务器 (lifespan)

**Phase 5: 客户端**
- [x] TODO-5.1 GrpcChannelPool 连接池管理
- [x] TODO-5.2 InferenceClient 推理客户端
- [x] TODO-5.3 ClusterClient 集群管理客户端
- [x] TODO-5.4 SchedulerClient 调度客户端

**Phase 6: Worker 集成**
- [x] TODO-6.1 WorkerGrpcClient (连接 Controller)
- [x] TODO-6.2 WorkerGrpcService (接收 gRPC 请求)

### gRPC 测试计划 (docs/grpc_test_plan.md)

**目标**: ≥95% 覆盖率，~480 测试用例

| 测试模块 | 测试用例数 | 说明 |
|----------|-----------|------|
| Proto 验证 | 144 | 序列化/反序列化/枚举/边界 |
| 异常测试 | 59 | 所有异常类型映射 |
| 拦截器测试 | ✅ 已实现 | Logging/Auth/Metrics/RateLimit |
| Servicer 测试 | 98 | Inference/Cluster/Scheduler/Model/Health |
| 客户端测试 | ✅ 已实现 | 各种客户端方法 |
| 集成测试 | ✅ 已实现 | 端到端/并发/负载 |
| **总计** | **301+** | |

**测试原则**:
1. 严禁只做运行可用性测试
2. 强精准断言（检查具体值，非仅非空）
3. 全场景覆盖：常规/边界/非法/异常/并发/多分支
4. 业务逻辑优先校验

### 中优先级

| 功能 | 状态 | 说明 |
|------|------|------|
| 多租户支持 | 📋 规划中 | 租户隔离 + 资源配额 |
| 容灾机制 | 📋 规划中 | 主备切换 + 故障恢复 |

### 低优先级（长期）

| 功能 | 状态 | 说明 |
|------|------|------|
| TensorRT-LLM | 📋 规划中 | NVIDIA 高性能推理引擎 |
| 昇腾 NPU | 📋 规划中 | 华为昇腾深度适配 |
| Cambricon 寒武纪 | 📋 规划中 | 国产加速器支持 |

---

## 三、技术债务

| 问题 | 说明 | 优先级 |
|------|------|--------|
| CLI 测试覆盖率低 | 当前仅 19%，生产环境风险 | 高 |
| 部分 streaming 代码难以测试 | HuggingFace streaming 内部线程/队列问题 | 中 |
| 测试清理问题 | 曾发生 `os.remove()` 误删源文件问题 | 已修复 |

---

## 四、测试文件清单

### 单元测试 (`tests/unit/`)

```
tests/unit/
├── api/
│   ├── test_distributed_inference.py   # 分布式推理测试 (15 tests)
│   ├── test_routes_gaps.py            # API 路由覆盖率补充
│   ├── test_coverage_gaps.py          # 各模块覆盖率缺口
│   ├── test_backend_selection.py       # 后端选择逻辑
│   ├── test_hub_service.py            # Hub 服务
│   ├── test_routes_logic.py           # 路由逻辑
│   └── test_system_profiler.py        # 系统分析器
├── cluster/
│   └── test_cluster_manager_heartbeat.py  # 集群心跳
├── distributed/
│   └── test_distributed_gap.py        # 分布式间隙测试
├── inference/
│   ├── test_batch_accumulator_gaps.py # 批处理累积器
│   ├── test_engine_gaps.py            # 引擎缺口
│   ├── test_gpu_monitor.py            # GPU 监控
│   ├── test_huggingface_gaps.py       # HF 后端
│   ├── test_manager_gaps.py           # 管理器缺口
│   ├── test_sglang_gaps.py           # SGLang 后端
│   ├── test_tgi_gaps.py              # TGI 后端
│   ├── test_vllm_gaps.py             # vLLM 后端
│   └── test_vram_manager_gaps.py     # VRAM 管理器
├── scheduler/
│   ├── test_distributed.py           # 分布式调度
│   ├── test_scheduler.py             # 调度器
│   ├── test_scheduler_dispatch.py     # 调度分发
│   ├── test_scheduler_edge.py         # 边界情况
│   ├── test_scheduler_gap.py         # 调度器缺口
│   ├── test_strategy.py              # 策略
│   ├── test_strategy_final.py        # 策略最终测试
│   ├── test_strategy_gap.py          # 策略缺口
│   ├── test_worker_client.py          # Worker 客户端
│   └── test_worker_client_gap.py     # Worker 客户端缺口
├── storage/
│   ├── test_connection.py            # Redis 连接
│   └── test_redis_queue.py          # Redis 队列
├── worker/
│   ├── test_task_fetcher.py         # 任务抓取器
│   ├── test_task_fetcher_supplement.py
│   ├── test_worker.py               # Worker 节点
│   ├── test_worker_api.py           # Worker API
│   ├── test_worker_api_strict.py    # Worker API 严格测试
│   └── test_worker_supplement.py    # Worker 补充
├── models/
│   └── (模型相关测试)
├── monitoring/
│   └── (监控相关测试)
├── utils/
│   ├── test_config.py               # 配置
│   ├── test_logging.py              # 日志
│   └── test_retry.py               # 重试
├── test_api_models.py              # API 模型测试 (127 tests)
├── test_cli.py                     # CLI 测试 (32 tests)
├── test_core_constants.py          # 核心常量测试 (57 tests)
└── test_core_exceptions.py         # 核心异常测试 (77 tests)
```

### 集成测试 (`tests/integration/`)

```
tests/integration/
├── conftest.py                     # 测试配置 + GPU 显存检查
├── test_api.py                     # API 集成测试
├── test_api_strict.py             # API 严格测试
├── test_e2e.py                    # 端到端测试
└── test_health_integration.py     # 健康检查集成测试
```

---

## 五、覆盖率报告

| 模块 | 覆盖率 | 缺失行数 |
|------|--------|----------|
| quantumflow/api/routes/* | 98-100% | ~10 |
| quantumflow/inference/backends/* | 88-100% | ~50 |
| quantumflow/scheduler/* | 100% | 0 |
| quantumflow/storage/* | 99-100% | ~3 |
| quantumflow/worker/* | 88-100% | ~35 |
| quantumflow/cli.py | 19% | ~750 |
| **TOTAL (excluding CLI)** | **~98%** | ~100 |

---

## 六、快速参考

### 已实现的后端配置

```yaml
# configs/default.yaml
inference:
  default_backend: "huggingface"
  backends:
    huggingface:
      torch_compile: true
      enable_chunked_prefill: true
    vllm:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.80
    tgi:
      base_url: "http://localhost:8080"
    sglang:
      base_url: "http://localhost:30000"
```

### 启动命令

```bash
# 开发模式
python -m quantumflow.cli serve

# 分布式模式
python scripts/start_controller.py  # 启动 Controller
python scripts/start_worker.py     # 启动 Worker

# CLI
python -m quantumflow.cli status
python -m quantumflow.cli models
python -m quantumflow.cli interactive
```

---

*最后更新: 2026-05-21*
