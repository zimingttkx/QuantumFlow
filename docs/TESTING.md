# QuantumFlow 测试文档

> 测试策略、跑测指南、覆盖率、gRPC 测试。

---

## 目录

- [测试策略](#1-测试策略)
- [目录结构](#2-目录结构)
- [跑测指南](#3-跑测指南)
- [覆盖率](#4-覆盖率)
- [gRPC 测试](#5-grpc-测试)
- [编写测试规范](#6-编写测试规范)

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
├── conftest.py                          # 全局 pytest 配置
├── unit/
│   ├── api/                             # API 路由、租户、Hub
│   ├── cluster/                         # 集群管理、心跳
│   ├── distributed/                     # 分布式调度
│   ├── inference/                       # 引擎、后端、批处理、VRAM
│   ├── scheduler/                       # 调度器、策略、Worker 客户端
│   ├── storage/                         # Redis 队列、连接
│   ├── worker/                          # Worker 节点、TaskFetcher
│   ├── monitoring/                      # 监控指标
│   ├── models/                          # API 模型
│   └── utils/                           # config / logging / retry
├── integration/
│   ├── conftest.py                      # GPU 显存检查
│   ├── test_api.py                      # API 集成
│   ├── test_api_strict.py               # 严格断言
│   ├── test_e2e.py                      # 端到端
│   ├── test_health_integration.py       # 健康检查
│   └── test_tenant_cross_component.py   # 跨组件租户
└── benchmarks/                          # 性能基准
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

### 4.1 当前覆盖率

| 模块 | 覆盖率 | 状态 |
|------|--------|------|
| `quantumflow/api/routes/*` | 98-100% | ✅ |
| `quantumflow/inference/backends/*` | 88-100% | ✅ |
| `quantumflow/scheduler/*` | 100% | ✅ |
| `quantumflow/storage/*` | 99-100% | ✅ |
| `quantumflow/worker/*` | 88-100% | ✅ |
| `quantumflow/cli.py` | 19% | ⚠️ 待提升 |
| **TOTAL（不含 CLI）** | **~98%** | ✅ |

### 4.2 查看报告

```bash
# HTML 报告
pytest --cov=quantumflow --cov-report=html
open htmlcov/index.html

# 终端报告
pytest --cov=quantumflow --cov-report=term-missing

# 校验最低覆盖率（CI）
pytest --cov=quantumflow --cov-fail-under=85
```

### 4.3 改进重点

- **CLI**：当前 19%，需补充 `tests/unit/test_cli.py` 覆盖 `serve`/`load`/`chat`/`generate`/`status`/`models` 等子命令
- **流式代码**：HuggingFace 流式内部线程/队列难以测试，需重构或使用 `inference_mode` 解耦

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
tests/
├── unit/grpc/
│   ├── test_proto_validation.py
│   ├── test_exceptions.py
│   ├── test_interceptors/
│   │   ├── test_logging_interceptor.py
│   │   ├── test_auth_interceptor.py
│   │   ├── test_metrics_interceptor.py
│   │   └── test_rate_limit_interceptor.py
│   ├── test_servicers/
│   │   ├── test_inference_service.py
│   │   ├── test_cluster_service.py
│   │   ├── test_scheduler_service.py
│   │   ├── test_model_management_service.py
│   │   └── test_health_service.py
│   └── test_clients/
│       ├── test_inference_client.py
│       ├── test_cluster_client.py
│       └── test_scheduler_client.py
└── integration/grpc/
    ├── test_end_to_end.py
    ├── test_worker_controller.py
    └── test_load_balancing.py
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

## 6. 编写测试规范

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
