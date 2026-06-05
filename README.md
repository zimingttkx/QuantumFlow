# QuantumFlow

> 分布式大模型推理平台 — 把 GPU 集群变成可对外服务的推理平台。

[![Python](https://img.shields.io/badge/Python-3.10+-00D9FF?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%20%7C%2013.0-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

[English](README.md) · [中文](README.md)

---

## 项目定位

**QuantumFlow 是一个面向生产环境的分布式大模型推理平台。**

当你拥有 10 张以上 GPU、需要服务多用户 / 多模型 / 多业务时，QuantumFlow 提供：

- **统一接入** — REST / gRPC / Python SDK / CLI / Web Playground 五种入口
- **智能调度** — Gang（All-or-Nothing，大模型）/ Pack（共享，小模型）/ Adaptive（AI 选）三种策略自动切换
- **多后端兼容** — vLLM / HuggingFace / TGI / SGLang / TensorRT-LLM 共 5 种推理引擎
- **多租户 SaaS 化** — API Key 认证、配额、限流、显存隔离、按租户计费
- **可观测** — Prometheus 指标、健康检查、节点心跳、Grafana 接入

**它不是**：单纯的推理引擎（那是 vLLM 的事），也不是容器调度（那是 K8s 的事）。它站在两者中间，把「模型 + GPU 集群 + 多用户」打包成一个可用的产品。

---

## 应用场景

平台适用的核心模式：

**模式 1：多优先级业务混部**

客服对话、内部知识库、批量离线任务同时打在同一批 GPU 上，但它们对延迟的容忍度天差地别——前者要求毫秒级响应，后者可以接受排队甚至中断重试。

通过 `priority` 字段为请求分级。系统按优先级调度，高优请求插队优先执行，低优请求复用空闲算力；调度器带 anti-starvation 机制，长时间得不到服务的低优请求会被自动提权。

**模式 2：小模型池化共享**

业务方需要的模型尺寸不一（1.5B、7B、13B 都有），但单个模型的 QPS 都不高，单独占卡浪费严重。需要的是把一批小模型塞进同一张卡里，热模型常驻、冷模型按需换入。

调度层走 Pack 策略，把多个小模型并发加载到同一块 GPU；推理层走多模型共享的批处理通道；显存不够时按 LRU 淘汰冷模型。

**模式 3：跨业务成本分摊**

多部门共用一套 GPU 集群，但预算独立。需要知道每个部门用了多少资源、应该分摊多少成本。

为每个部门创建独立租户，分配专属 API Key 和配额上限。系统按 token 数 / 请求数记录每个租户的用量，导出计费报表。

**模式 4：私有化部署 + 跨可用区容灾**

对内提供推理服务的 MaaS 平台，敏感数据不出内网；单机房故障要能自动切换到备用机房。

Worker 节点跨机房部署；Controller 通过 Redis ZSET 统一调度；故障节点被自动标记为 UNHEALTHY 并从调度池剔除，副本被重新派发到健康节点。

---

## 核心能力

| 维度 | 能力 |
|------|------|
| **调度** | Gang / Pack / Adaptive 三策略 + Redis ZSET 优先级队列 + 分布式调度器 |
| **推理** | 5 种后端统一接口 + 动态批处理（50ms 窗口）+ 优先级感知 + 动态 batch_size + 多模型共享 |
| **显存** | Block Pool 细粒度分配（类 vLLM PagedAttention）+ LRU 模型淘汰 + VRAM 实时监控 |
| **集群** | 多 Worker 节点 + 心跳健康检查 + 自动故障转移 + 节点状态机（HEALTHY/UNHEALTHY/DRAINING） |
| **多租户** | API Key 认证 + 租户配额 + Token Bucket 限流（全局 / per-tenant / per-endpoint 三层）+ 显存隔离 |
| **协议** | REST (FastAPI) + gRPC (Protobuf) + Python SDK (Sync/Async) + CLI + Web Playground |
| **可观测** | Prometheus 指标 + structlog 结构化日志 + 健康检查（live / ready）+ 心跳上报 |

---

## 系统架构

```
┌────────────────────────────────────────────────────────────────────┐
│                         QuantumFlow Platform                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   接入层      REST API  │  gRPC API  │  Python SDK  │  CLI         │
│   ─────      ──────────────────────────────────────────────────    │
│                中间件：TenantAuth → RateLimit → Handler             │
│                              │                                     │
│   调度层      DistributedScheduler ─ Redis ZSET 优先级队列          │
│   ─────      策略：Gang / Pack / Adaptive / Priority              │
│                              │                                     │
│   执行层      EngineManager ─ VRAMManager ─ BatchAccumulator       │
│   ─────      SharedBatchCoordinator ─ GPUMonitor (NVML)           │
│               后端：vLLM / HuggingFace / TGI / SGLang / TensorRT   │
│                              │                                     │
│   集群层      NodeRegistry ─ ServiceDiscovery ─ HealthMonitor     │
│   ─────      节点状态机：HEALTHY / DRAINING / UNHEALTHY / OFFLINE │
│                              │                                     │
│   监控层      Prometheus Metrics ─ structlog JSON Logs             │
│   ─────                                                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 前置条件

- Python ≥ 3.10
- CUDA 12.4 或 13.0（仅推理需要；纯调度 / 集群管理可省略）
- Redis ≥ 7.0（Worker ↔ Controller 通信）
- Linux 推荐（Windows 需 WSL2，详见 [DEPLOYMENT.md](./docs/DEPLOYMENT.md)）

### 安装

```bash
# 克隆
git clone https://github.com/quantumflow/quantumflow.git
cd QuantumFlow

# 核心依赖（跨平台）
pip install -r requirements.txt

# GPU 加速（仅 Linux）
pip install -r requirements-gpu.txt

# 开发工具（测试 / 格式化 / 类型检查）
pip install -r requirements-dev.txt
```

### 启动

```bash
# 1. 启动 Redis
docker run -d -p 6379:6379 --name qf-redis redis:7-alpine

# 2. 启动 Controller（API Server）
python -m quantumflow.cli serve --host 0.0.0.0 --port 8000

# 3. （可选）启动 Worker 节点
python scripts/start_worker.py

# 4. 打开浏览器
#    http://localhost:8000/docs              → Swagger UI（API 在线试调）
#    http://localhost:8000/static/playground.html  → 交互式 Playground
```

### Python SDK 调用

```python
from quantumflow import SyncClient

client = SyncClient(base_url="http://localhost:8000", api_key="qf-dev-xxx")

# 文本生成
result = client.generate(
    model="Qwen2.5-7B-Instruct",
    prompt="用一句话介绍分布式系统",
    sampling_params={"temperature": 0.7, "max_tokens": 100}
)
print(result.generated_text)

# 流式生成
for chunk in client.stream(model="Qwen2.5-7B-Instruct", prompt="写一首诗"):
    print(chunk.delta, end="", flush=True)
```

### CLI 调用

```bash
# 集群状态
python -m quantumflow.cli status

# 加载模型
python -m quantumflow.cli load Qwen2.5-7B-Instruct

# 推理
python -m quantumflow.cli generate Qwen2.5-7B-Instruct -p "你好"

# 交互式终端
python -m quantumflow.cli interactive
```

---

## 文档

完整文档位于 [docs/](./docs/)：

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](./docs/ARCHITECTURE.md) | 系统设计：模块划分、关键设计、数据流、调度策略、部署拓扑 |
| [API.md](./docs/API.md) | 接口参考：REST API、gRPC API、CLI 命令、错误响应 |
| [DEPLOYMENT.md](./docs/DEPLOYMENT.md) | 部署指南：环境要求、本地与生产（K8s）部署、监控、升级 |
| [TESTING.md](./docs/TESTING.md) | 测试文档：测试策略、跑测指南、覆盖率、gRPC 测试 |

---

## 性能

### 实测基准（RTX 4080 Laptop 12GB / Qwen2.5-1.5B / HuggingFace + BatchAccumulator）

| 场景 | 描述 | GPU 利用率 | P50 延迟 | 吞吐 |
|------|------|:---------:|:--------:|:----:|
| A | 单请求（greedy，短 prompt） | 59% | 509 ms | 76.6 tok/s |
| B | 短对话 8 并发 | 51% | 1058 ms | 65.9 tok/s |
| D | 长 prompt + 生成 | 69% | 2016 ms | 83.4 tok/s |
| H | 高并发 32 并发 | 54% | 462 ms | 85.5 tok/s |

**与业界对比**：
- 目标 GPU 计算利用率 80%+、显存带宽利用率 80%+
- vLLM Continuous Batching 生产环境通常 60-85% GPU 利用率
- 完整 10 场景测试：见仓库历史提交中的基准数据

### GPU 指标

通过 NVIDIA NVML 采集两个独立指标：

| 指标 | 来源 | 含义 |
|------|------|------|
| GPU Compute Utilization | `nvmlDeviceGetUtilizationRates().gpu` | CUDA 核心活跃度 |
| GPU Memory Bandwidth | `nvmlDeviceGetUtilizationRates().memory` | HBM 显存控制器活跃度 |

---

## 部署

### 本地开发（Docker Compose）

```bash
docker run -d -p 6379:6379 --name qf-redis redis:7-alpine
python -m quantumflow.cli serve
```

### 生产环境（Kubernetes）

```bash
# 部署 Redis
kubectl apply -f deploy/k8s/redis.yaml

# 部署 Controller（REST + gRPC）
kubectl apply -f deploy/k8s/controller.yaml

# 部署 Worker 池（A100 / 4090 / H100 按需）
kubectl apply -f deploy/k8s/worker-a100.yaml
kubectl apply -f deploy/k8s/worker-4090.yaml

# 接入 Prometheus
kubectl apply -f deploy/k8s/monitoring.yaml
```

完整 YAML 模板、HPA 配置、监控告警规则、TLS 配置等详见 [DEPLOYMENT.md](./docs/DEPLOYMENT.md)。

---

## 贡献

欢迎 Issue 和 Pull Request。

```bash
# 1. Fork + 克隆
git clone https://github.com/your-fork/quantumflow.git

# 2. 安装开发依赖
pip install -r requirements.txt -r requirements-dev.txt

# 3. 跑测试（须先启动 Redis）
pytest tests/unit -v

# 4. 提交（遵循 Conventional Commits）
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature
```

**代码规范**：
- Python 3.10+，类型注解必填
- 提交前 `black` + `isort` + `ruff` + `mypy` 必须通过
- 任何新功能必须有单元测试（覆盖率 ≥ 80%）
- 详见 [TESTING.md](./docs/TESTING.md) 测试原则

---

## 路线图

| 方向 | 状态 | 说明 |
|------|:----:|------|
| gRPC 高性能接口 | ✅ | 5 个 Service + 拦截器链 |
| Python SDK（Sync/Async） | ✅ | httpx + 完整鉴权 |
| TensorRT-LLM 后端 | ✅ | NVIDIA 高性能推理引擎 |
| 多租户 + 限流 + 配额 | ✅ | Token Bucket 三层限流 |
| 优先级感知批处理 | ✅ | Anti-starvation 机制 |
| 动态 batch_size | ✅ | VRAM 感知自动调参 |
| 多模型共享 | ✅ | SharedBatchCoordinator |
| 容灾（模型副本 + 自动故障转移） | 📋 | P1 |
| 国产 NPU（昇腾 / 寒武纪） | 📋 | P2 长期 |

---

## 许可证

[Apache License 2.0](LICENSE)

---

## 致谢

本项目参考了以下开源项目的优秀设计：

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention + Continuous Batching
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — 高效注意力 kernel
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 推理引擎基础
- [Ray](https://github.com/ray-project/ray) — 分布式计算框架
- [Kubernetes](https://kubernetes.io/) — 容器编排参考

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐**

*Built with care by the QuantumFlow Team*

</div>
