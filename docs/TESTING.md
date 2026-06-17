# QuantumFlow 测试文档

> 测试策略、跑测指南、覆盖率、gRPC 测试。

---

## 目录

- [测试策略](#1-测试策略)
- [目录结构](#2-目录结构)
- [跑测指南](#3-跑测指南)
- [覆盖率](#4-覆盖率)
- [gRPC 测试](#5-grpc-测试)
- [已知失败（预存在）](#6-已知失败预存在)
- [编写测试规范](#7-编写测试规范)

---

## 1. 测试策略

### 1.1 测试原则

| 原则 | 说明 |
|------|------|
| **业务逻辑优先** | 验证业务正确性，而非"无报错" |
| **强断言** | 检查具体值（数值、状态、字段），不只断言"非空" |
| **全场景覆盖** | 常规 / 边界 / 非法输入 / 异常 / 并发 / 多分支 |
| **零容忍假通过** | 测试本身不能有漏洞（无 `except: pass`、无空断言） |
| **失败可见** | 任何吞掉的异常必须有显式日志或重新抛出 |

### 1.2 测试分层

| 层级 | 内容 | 目标覆盖 |
|------|------|----------|
| 单元测试 | 模块、函数、类、工具 | ≥ 90% |
| 集成测试 | API 端点、组件协作 | ≥ 85% |
| 端到端测试 | 关键用户流 | 核心路径 100% |
| 性能基准 | 吞吐、延迟、显存 | 10 个场景 |

---

## 2. 目录结构

```
tests/
├── conftest.py                                # 全局 pytest 配置
├── unit/
│   ├── api/                                   # API 路由、租户、Hub
│   │   ├── test_api_models.py
│   │   ├── test_coverage_gaps.py
│   │   ├── test_hub_service.py
│   │   ├── test_rate_limit_middleware.py
│   │   ├── test_routes_gaps.py
│   │   ├── test_routes_logic.py
│   │   ├── test_tenant_auth.py
│   │   ├── test_tenant_aware.py
│   │   ├── test_tenant_models.py
│   │   ├── test_tenant_rate_limit.py
│   │   ├── test_tenant_redis_contract.py
│   │   ├── test_tenant_routes.py
│   │   ├── test_tenant_sdk.py
│   │   └── test_tenant_vram.py
│   ├── cluster/                               # 集群管理
│   │   ├── test_cluster.py
│   │   ├── test_cluster_manager_heartbeat.py
│   │   ├── test_cluster_manager_strict.py
│   │   └── test_cluster_routes_strict.py
│   ├── distributed/                           # 分布式调度
│   │   ├── test_distributed.py
│   │   ├── test_distributed_comprehensive.py
│   │   └── test_distributed_gap.py
│   ├── failover/                              # 容灾（FailoverController / ReplicaManager / HealthChecker / LeaderElection）
│   │   ├── test_failover_controller.py
│   │   ├── test_failover_controller_extended.py
│   │   ├── test_failover_extended_v3.py
│   │   ├── test_health_checker.py
│   │   ├── test_health_checker_boundaries.py
│   │   ├── test_health_checker_exception.py
│   │   ├── test_health_checker_extended.py
│   │   ├── test_health_checker_extended_v2.py
│   │   ├── test_leader_election.py
│   │   ├── test_leader_election_extended.py
│   │   ├── test_policy.py
│   │   ├── test_replica_manager.py
│   │   ├── test_replica_manager_extended.py
│   │   ├── test_state_store.py
│   │   └── test_state_store_extended.py
│   ├── inference/                             # 引擎、后端、批处理、VRAM、GPU 监控
│   │   ├── test_backend_protocol.py
│   │   ├── test_backend_selection.py
│   │   ├── test_batch_accumulator_gaps.py
│   │   ├── test_batch_accumulator_logic.py
│   │   ├── test_batch_accumulator_priority.py
│   │   ├── test_batch_coordinator.py
│   │   ├── test_batch_dynamic.py
│   │   ├── test_chunked_prefill.py            # 历史接口规范快照
│   │   ├── test_distributed_inference.py
│   │   ├── test_engine.py
│   │   ├── test_engine_gaps.py
│   │   ├── test_engine_manager_backend_integration.py
│   │   ├── test_engine_strict.py
│   │   ├── test_gpu_monitor.py
│   │   ├── test_gpu_monitor_gaps.py
│   │   ├── test_huggingface_behavior.py
│   │   ├── test_huggingface_gaps.py
│   │   ├── test_huggingface_logic.py
│   │   ├── test_manager_gaps.py
│   │   ├── test_manager_logic.py
│   │   ├── test_sglang_behavior.py
│   │   ├── test_sglang_gaps.py
│   │   ├── test_tensorrt_llm_gaps.py
│   │   ├── test_tgi_behavior.py
│   │   ├── test_tgi_gaps.py
│   │   ├── test_vllm_gaps.py
│   │   ├── test_vram_estimation_precision.py
│   │   ├── test_vram_manager_gaps.py
│   │   └── test_vram_manager_logic.py
│   ├── scheduler/                             # 调度器、策略、Worker 客户端
│   │   ├── test_distributed.py
│   │   ├── test_scheduler.py
│   │   ├── test_scheduler_dispatch.py
│   │   ├── test_scheduler_edge.py
│   │   ├── test_scheduler_gap.py
│   │   ├── test_strategy.py
│   │   ├── test_strategy_final.py
│   │   └── test_strategy_gap.py
│   ├── sdk/                                   # Python SDK
│   │   ├── test_async_client.py
│   │   ├── test_client.py
│   │   └── test_sdk_init.py
│   ├── storage/                               # Redis 队列、连接
│   │   └── test_redis_queue.py
│   ├── models/                                # 模型注册表
│   │   └── test_registry.py
│   ├── monitoring/                            # 监控指标
│   │   ├── test_metrics.py
│   │   └── test_system_profiler.py
│   ├── worker/                                # Worker API / TaskFetcher
│   │   ├── test_task_fetcher.py
│   │   ├── test_task_fetcher_supplement.py
│   │   ├── test_worker.py
│   │   ├── test_worker_api.py
│   │   ├── test_worker_api_strict.py
│   │   ├── test_worker_api_supplement.py
│   │   ├── test_worker_client.py
│   │   ├── test_worker_client_gap.py
│   │   └── test_worker_supplement.py
│   └── utils/                                 # config / logging / retry / grpc 集成
│       ├── test_cli.py
│       ├── test_config.py
│       ├── test_connection.py
│       ├── test_core_constants.py
│       ├── test_core_exceptions.py
│       ├── test_exceptions.py
│       ├── test_integration.py
│       ├── test_logging.py
│       ├── test_models.py
│       └── test_retry.py
├── integration/
│   ├── conftest.py                            # GPU 显存检查
│   ├── test_api.py                            # API 集成
│   ├── test_api_strict.py                     # 严格断言
│   ├── test_e2e.py                            # 端到端
│   ├── test_health_integration.py             # 健康检查
│   ├── test_rate_limit_integration.py
│   ├── test_sdk_integration.py
│   ├── test_tenant_cross_component.py         # 跨组件租户
│   ├── test_tenant_isolation.py
│   ├── test_heterogeneous_gpu_scheduling.py   # 活文档：异构 GPU + 多模型 28 个不变量
│   ├── failover/                              # 容灾集成
│   │   ├── test_e2e_scenarios.py
│   │   ├── test_failover_controller_integration.py
│   │   ├── test_performance.py
│   │   ├── test_redis_real.py
│   │   ├── test_state_store_integration.py
│   │   └── test_stress.py
│   └── inference/                             # 推理引擎集成
│       ├── test_engine_manager.py
│       └── test_model_loading.py
└── benchmarks/                                # 性能基准
    ├── A_single.py
    ├── B_chat_concurrent.py
    ├── C_code_generation.py
    ├── D_long_prompt.py
    ├── E_streaming.py
    ├── F_chat_api.py
    ├── G_batch.py
    ├── H_load.py
    ├── I_greedy_vs_random.py
    └── J_repetition_penalty.py
```

> 单元测试当前 **3352 个**（`pytest --collect-only` 报告）；其中 **3310 通过 / 35 失败 / 7 跳过**（详见 §6 已知失败）。

---

## 3. 跑测指南

### 3.1 环境准备

```bash
# 1. 安装测试依赖
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov httpx

# 2. 启动 Redis（必需）
docker run -d -p 6379:6379 --name qf-redis-test redis:7-alpine

# 3. 验证 GPU（可选，E2E 用）
nvidia-smi
```

### 3.2 常用命令

```bash
# 全部单元测试
pytest tests/unit -v

# 全部测试
pytest tests/ -v

# 集成测试（需 Redis + 可选 GPU）
pytest tests/integration -v

# 单个模块
pytest tests/unit/scheduler -v

# 匹配关键字
pytest tests/ -k "test_priority" -v

# 带覆盖率
pytest tests/unit --cov=quantumflow --cov-report=html --cov-report=term

# 并行（pytest-xdist）
pytest tests/unit -n auto

# 失败即停
pytest tests/unit -x
```

### 3.3 标记（Markers）

| 标记 | 用途 |
|------|------|
| `@pytest.mark.gpu` | 需要真实 GPU（跳过若无） |
| `@pytest.mark.integration` | 集成测试 |
| `@pytest.mark.slow` | 耗时测试 |
| `@pytest.mark.grpc` | gRPC 专项 |

```bash
# 仅 gRPC 测试
pytest tests/ -m grpc -v

# 跳过慢测试
pytest tests/ -m "not slow" -v
```

### 3.4 端到端测试

```bash
# 1. 启动服务
python -m quantumflow.cli serve &
python scripts/start_worker.py &

# 2. 跑 E2E
pytest tests/integration/test_e2e.py -v
```

### 3.5 性能基准

```bash
# 单个场景
python tests/benchmarks/B_chat_concurrent.py

# 全部
for f in tests/benchmarks/*.py; do python "$f"; done
```

基准场景：

| 编号 | 场景 | 关注指标 |
|------|------|----------|
| A | 单请求 greedy 短生成 | 延迟基线 |
| B | 8 并发 chat 短生成 | 并发吞吐 |
| C | 代码生成（中等 prompt） | 延迟分布 |
| D | 长 prompt + 长生成 | 长上下文性能 |
| E | 流式生成 | TTFT、token/s |
| F | Chat API 多轮 | 模板开销 |
| G | 批量 API | 批处理吞吐 |
| H | 32 并发负载 | 系统稳定性 |
| I | Greedy vs Random | 采样开销 |
| J | Repetition Penalty 长输出 | 参数开销 |

---

## 4. 覆盖率

### 4.1 当前测试统计（2026-06）

| 维度 | 数值 |
|------|------|
| 单元测试总数 | **3352** |
| 通过 | 3310 |
| 失败（预存在） | 35 |
| 跳过 | 7 |
| 通过率 | 98.75% |

> 失败 35 个全部为**预存在 bug**（详见 [§6 已知失败](#6-已知失败预存在)），本次深度修复后异构 GPU / 多模型 / 调度算法 / 推理后端 / 分布式 / 容灾 / 模型注册 / 租户契约 相关测试**全部通过**。

### 4.2 关键模块覆盖率

| 模块 | 覆盖率 | 状态 |
|------|--------|------|
| `quantumflow/scheduler/*` | ~100% | ✅ |
| `quantumflow/inference/backends/*` | 88-100% | ✅ |
| `quantumflow/failover/*` | ~95% | ✅ |
| `quantumflow/storage/*` | 99-100% | ✅ |
| `quantumflow/api/routes/*` | 95-100% | ✅ |
| `quantumflow/api/middleware/*` | ~95% | ✅ |
| `quantumflow/api/middlewares/*` | ~95% | ✅ |
| `quantumflow/worker/*` | 88-100% | ✅ |
| `quantumflow/grpc/*` | ≥ 95% | ✅ |
| `quantumflow/cli.py` | ~19% | ⚠️ 待提升 |
| `quantumflow/sdk/*` | 待补 | ⚠️ 测试 mock 写错（见 §6） |

### 4.3 查看报告

```bash
# HTML 报告
pytest --cov=quantumflow --cov-report=html
open htmlcov/index.html

# 终端报告
pytest --cov=quantumflow --cov-report=term-missing

# 校验最低覆盖率（CI）
pytest --cov=quantumflow --cov-fail-under=85
```

### 4.4 改进重点

- **CLI**：当前 19%，需补充 `tests/unit/test_cli.py` 覆盖 `serve`/`load`/`chat`/`generate`/`status`/`models` 等子命令
- **SDK**：mock 写错导致 11 个测试失败（详见 §6）
- **流式代码**：HuggingFace 流式内部线程/队列难以测试，需重构或使用 `inference_mode` 解耦

---

## 6. 已知失败（预存在）

> **重要**：本节列出的失败**全部存在于干净 `main` 分支**（已通过 `git stash` 验证），与最近修复无关，无需在 PR 中修复。

### 6.1 SDK 客户端（11 个）

**文件**：`tests/unit/sdk/test_client.py`、`test_async_client.py`

**症状**：

```
TypeError: '>=' not supported between instances of 'MagicMock' and 'int'
APIError: API Error 502:
```

**根因**：测试用 `mock_response.status_code = 200` 配合 `mock_client_instance.__enter__.return_value.post.return_value = mock_response` 设置 mock，但 `SyncQuantumFlowClient._post`（`quantumflow/sdk/client.py:35`）**实际并未走 `with` 上下文管理器**，直接 `self._client.post(path, **kwargs)`。所以 mock 没生效，`response.status_code` 是 MagicMock。

**修复方向**（两选一）：
1. 把测试改成 `mock_client_class.return_value.post.return_value = mock_response`（推荐，影响小）
2. 把 SDK 客户端改成 `with self._client as c: c.post(...)`（改动大，不推荐）

**是否阻塞核心功能**：**否**。SDK 客户端手工 `curl` / 集成测试（`tests/integration/test_sdk_integration.py`）全部通过。仅单元测试 mock 写错。

### 6.2 vLLM gaps（24 个）

**文件**：`tests/unit/inference/test_vllm_gaps.py`

**症状**：

```
ModuleNotFoundError: No module named 'vllm'
ModuleNotFoundError: No module named 'pynvml'
AttributeError: ...FakeVLLM object... does not have the attribute 'SamplingParams'
```

**根因**：vllm / pynvml 是原生依赖，本地环境未安装；测试试图用 `FakeVLLM` 替代 vLLM 的某些属性但 spec 约束不完整。

**修复方向**：
1. 安装 vllm / pynvml（`pip install vllm pynvml`）
2. 完善 `FakeVLLM` 类的 spec（添加 `SamplingParams` 属性 + 完整 `LLM` 接口）

**是否阻塞核心功能**：**否**。vLLM 真实部署由 Worker 节点上的 vLLM 进程负责；Controller 通过 HTTP / gRPC 调用，不依赖本机 vllm 包。

### 6.3 修复原则

- **不要为了"通过"而 mock 掉真实逻辑**——失败测试的存在恰恰说明测试在验证真实行为
- **批量修复**可以放在独立 PR（chore: fix pre-existing test failures），不与功能 PR 混在一起
- **CI 配置**：将 `test_vllm_gaps.py` / `test_sdk/test_client.py` 标记为 `xfail` 或 `@pytest.mark.optional` 即可跳过

---

## 5. gRPC 测试

### 5.1 测试分层

| 层级 | 测试内容 | 目标用例 |
|------|----------|----------|
| Proto 验证 | 消息序列化 / 反序列化 / 枚举 / 边界 | 50+ |
| 异常 | 异常类型映射、状态码 | 50+ |
| 拦截器 | Logging / Auth / Metrics / RateLimit | 100+ |
| Servicer | 5 个服务业务逻辑 | 200+ |
| 客户端 | 各客户端方法 | 80+ |
| 集成 | 端到端 gRPC 调用 | 50+ |
| 负载 | 并发、负载均衡 | 30+ |

**目标覆盖率**：≥ 95%

### 5.2 测试文件位置

```
tests/unit/grpc/
├── test_proto_validation.py
├── test_exceptions.py
├── test_base_service.py
├── test_clients.py
├── test_interceptors/
│   ├── test_logging_interceptor.py
│   ├── test_auth_interceptor.py
│   ├── test_metrics_interceptor.py
│   └── test_rate_limit_interceptor.py
├── test_servicers/
│   ├── test_inference_service.py
│   ├── test_cluster_service.py
│   ├── test_scheduler_service.py
│   ├── test_model_management_service.py
│   ├── test_health_service.py
│   └── test_metrics_service.py
├── test_clients/
│   ├── test_inference_client.py
│   ├── test_cluster_client.py
│   └── test_scheduler_client.py
└── test_channels/
    ├── test_channel_pool.py
    └── test_channel_pool_complete.py
```

### 5.3 关键测试点

**Proto 验证**：
- 必填字段缺失
- 枚举值越界
- 嵌套消息
- 边界数值（int32 最大 / 最小）

**拦截器**：
- Logging：方法名、耗时、状态码
- Auth：有效/无效 token、缺失 token
- Metrics：QPS、延迟、错误率
- RateLimit：突发、超额、令牌恢复

**Servicer 业务逻辑**：
- 同步 / 流式 / 批量 三种推理路径
- 节点注册 → 心跳 → 注销生命周期
- 调度提交 → 状态查询 → 取消
- 模型加载 / 卸载 / 列表
- 健康检查 / 流式监控

**端到端**：
- Client → Controller gRPC → Worker HTTP → Backend → 回传
- 并发请求下的顺序与一致性
- 负载均衡策略验证

### 5.4 跑 gRPC 测试

```bash
# gRPC 单元
pytest tests/unit/grpc -v

# gRPC 集成（需启动 gRPC Server）
python -m quantumflow.cli serve &
pytest tests/integration/grpc -v

# 覆盖率
pytest tests/unit/grpc --cov=quantumflow.grpc --cov-report=term-missing
```

---

## 7. 编写测试规范

### 6.1 命名

- 文件：`test_<module>.py`
- 函数：`test_<unit>_<scenario>_<expected>`
- 示例：`test_priority_queue_fifo_when_same_priority`

### 6.2 结构

```python
def test_<unit>_<scenario>(fixture):
    # Arrange
    ...

    # Act
    result = function_under_test(...)

    # Assert
    assert result.field == expected_value
    assert result.status == "success"
```

### 6.3 反模式（禁止）

```python
# ❌ 仅断言非空
assert result is not None

# ❌ 吞掉异常
try:
    do_something()
except Exception:
    pass

# ❌ 硬编码 sleep
time.sleep(1)
assert is_done()

# ❌ 依赖外部网络
requests.get("https://api.example.com")
```

### 6.4 正确做法

```python
# ✅ 精确断言
assert result.text == "expected output"
assert result.tokens == 42
assert result.finish_reason == "stop"

# ✅ 异常显式传播
with pytest.raises(ValueError, match="invalid model"):
    load_model("")

# ✅ 轮询 + 超时
def wait_until_ready(timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_ready():
            return
        time.sleep(0.05)
    raise TimeoutError("not ready")

# ✅ Mock 外部依赖
with mock.patch("httpx.AsyncClient.get", return_value=mock_resp):
    result = await fetch_data()
```

### 6.5 Fixture 复用

公共 fixture 放在 `conftest.py`：

```python
# tests/conftest.py
@pytest.fixture
def redis_client():
    return RedisConnectionManager.get_client()

@pytest.fixture
def mock_worker():
    return MockWorker(responses=[...])
```

---

*架构见 [ARCHITECTURE.md](./ARCHITECTURE.md)；接口见 [API.md](./API.md)；部署见 [DEPLOYMENT.md](./DEPLOYMENT.md)。*
