# QuantumFlow

<div align="center">

![QuantumFlow Logo](https://img.shields.io/badge/QuantumFlow-AI%20Inference-6366F1?style=for-the-badge&logo=rocket)
[![Python](https://img.shields.io/badge/Python-3.10+-00D9FF?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-FF6B6B?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/quantumflow/quantumflow?style=flat-square&color=F59E0B)](https://github.com/quantumflow/quantumflow/stargazers)
[![Forks](https://img.shields.io/github/forks/quantumflow/quantumflow?style=flat-square&color=10B981)](https://github.com/quantumflow/quantumflow/network/members)

**🚀 下一代分布式大模型推理平台 — 让千亿参数模型跑在每台机器上**

*「像调度 Kubernetes Pods 一样调度 AI 推理任务」*

[English](README.md) | [中文](README_zh.md)

</div>

---

## ✨ 特性

<div align="center">

| 🎯 核心能力 | 🌟 差异化亮点 | 🔧 技术优势 |
|:---:|:---:|:---:|
| **智能调度** | Gang/Pack/自适应多策略 | 自动选择最优执行路径 |
| **多后端支持** | vLLM / TGI / SGLang | 统一接口，灵活切换 |
| **国产硬件** | 昇腾NPU深度适配 | 打破 NVIDIA 垄断 |
| **企业级** | 多租户 / 限流 / 容灾 | 开箱即用的生产特性 |

</div>

### 🔥 为什么选择 QuantumFlow？

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   传统方式：                                                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                   │
│   │  模型   │───▶│ 手动分配 │───▶│  低效   │                   │
│   └─────────┘    └─────────┘    └─────────┘                   │
│                                                                 │
│   QuantumFlow：                                                   │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                   │
│   │  模型   │───▶│ 智能调度 │───▶│  高效   │                   │
│   └─────────┘    └─────────┘    └─────────┘                   │
│                       │                                        │
│              ┌──────┴──────┐                                  │
│              │ 自适应策略   │                                  │
│              │ • Gang (大模型)│                                  │
│              │ • Pack (小模型)│                                  │
│              │ • Adaptive (AI) │                                  │
│              └─────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 📦 安装

```bash
# 从 PyPI 安装
pip install quantumflow

# 或从源码安装
git clone https://github.com/quantumflow/quantumflow.git
cd quantumflow
pip install -e .
```

### 💻 启动服务

```bash
# 启动 API 服务器
quantumflow serve --port 8000

# 或使用 Python
python -c "from quantumflow.api.server import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

### 🔥 3 行代码开始推理

```python
from quantumflow import QuantumFlow

# 创建客户端
qf = QuantumFlow(api_url="http://localhost:8000")

# 部署模型（自动选择最优配置）
qf.deploy("Qwen2.5-72B-Instruct", tensor_parallel=4)

# 推理！
result = qf.generate(
    model="Qwen2.5-72B-Instruct",
    prompt="解释量子纠缠的基本原理"
)
print(result)
```

### 🛠️ CLI 使用

```bash
# 查看集群状态
quantumflow status

# 部署模型
quantumflow deploy Qwen2.5-72B --tensor-parallel 4 --gpus 0,1,2,3

# 测试生成
quantumflow generate Qwen2.5-72B -p "Hello, world!"

# 批量推理
quantumflow batch Qwen2.5-7B --file prompts.txt
```

---

## 🏗️ 系统架构

```
┌────────────────────────────────────────────────────────────────────────┐
│                         QuantumFlow Platform                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        接入层 (Gateway)                          │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │  │
│  │  │ REST API│  │ gRPC API│  │ Python  │  │   CLI   │         │  │
│  │  │ FastAPI │  │   SDK   │  │  SDK    │  │         │         │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘         │  │
│  └───────┼─────────────┼─────────────┼─────────────┼───────────────┘  │
│          └─────────────┴─────────────┴─────────────┘                     │
│                               │                                         │
│  ┌───────────────────────────┼───────────────────────────────────────┐ │
│  │                    调度层 (Scheduler)                               │ │
│  │  ┌────────────────────────────────────────────────────────────┐  │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │ │
│  │  │  │  Gang    │  │  Pack    │  │Adaptive  │  │ Priority │   │  │ │
│  │  │  │ Scheduler│  │ Scheduler│  │ Strategy │  │ Queue   │   │  │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │ │
│  │  └────────────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                               │                                         │
│  ┌───────────────────────────┼───────────────────────────────────────┐ │
│  │                    集群管理层 (Cluster)                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │ │
│  │  │   Node      │  │  Service    │  │  Health    │              │ │
│  │  │  Registry   │  │  Discovery  │  │  Monitor   │              │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                               │                                         │
│  ┌───────────────────────────┼───────────────────────────────────────┐ │
│  │                    执行层 (Worker Pool)                           │ │
│  │  ┌─────────────────────────────────────────────────────────────┐  │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │ │
│  │  │  │   vLLM   │  │   TGI    │  │  SGLang  │  │  TRT-LLM │   │  │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │ │
│  │  │           ▲            ▲            ▲            ▲        │  │ │
│  │  │            └────────────┴────────────┴────────────┘         │  │ │
│  │  │                    Unified Inference API                   │  │ │
│  │  └─────────────────────────────────────────────────────────────┘  │ │
│  │                                                                  │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │ │
│  │  │Node 1  │ │Node 2  │ │Node 3  │ │ Ascend │ │Cambricon│       │ │
│  │  │ A100×8 │ │4090×4  │ │H100×4  │ │ NPU×4  │ │  NPU×4  │       │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 调度策略

### Gang调度 — 大模型的专属武器

```
┌─────────────────────────────────────────┐
│         Gang Scheduling (大模型)          │
│                                          │
│   Request: 72B Model, TP=8              │
│                                          │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│   │ GPU0│ │ GPU1│ │ GPU2│ │ GPU3│ ... │
│   │  ✗  │ │  ✗  │ │  ✗  │ │  ✗  │     │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘     │
│      └────────┼────────┼────────┘       │
│               ▼                           │
│        All GPUs or Nothing                │
│                                         │
│   ✅ 100B+ 模型的最优选择                 │
│   ✅ 最小化通信开销                       │
│   ✅ 保障模型一致性                       │
└─────────────────────────────────────────┘
```

### Pack调度 — 小模型的效率之王

```
┌─────────────────────────────────────────┐
│         Pack Scheduling (小模型)         │
│                                          │
│   Request: 7B Model × N                  │
│                                          │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│   │Req 1│ │Req 2│ │Req 3│ │Req N│     │
│   │  ✗  │ │  ✗  │ │  ✗  │ │  ✗  │     │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘     │
│      └────────┼────────┼────────┘       │
│               ▼                           │
│      Shared GPU, Batched                 │
│                                         │
│   ✅ 最大化 GPU 利用率                   │
│   ✅ 高并发处理                          │
│   ✅ 降低单请求成本                       │
└─────────────────────────────────────────┘
```

---

## 📊 性能基准

| 模型 | 参数量 | 并行策略 | 吞吐量 | 延迟 | 备注 |
|------|--------|----------|--------|------|------|
| Qwen2.5-7B | 7B | TP=1 | 150 tok/s | 45ms | 消费级GPU |
| Qwen2.5-72B | 72B | TP=4 | 80 tok/s | 120ms | 数据中心 |
| LLaMA-3-70B | 70B | TP=8 | 60 tok/s | 180ms | 8×A100 |
| DeepSeek-V2 | 236B | TP=16 | 40 tok/s | 300ms | 16×H100 |

*测试环境: NVIDIA A100 80GB, Ubuntu 22.04, CUDA 12.1*

---

## 🛠️ 支持的模型

### 开源模型

| 模型 | 参数量 | 推荐配置 | 最低显存 |
|------|--------|----------|----------|
| Qwen2.5-7B | 7B | TP=1 | 16GB |
| Qwen2.5-14B | 14B | TP=1 | 24GB |
| Qwen2.5-72B | 72B | TP=4 | 4×24GB |
| LLaMA-3-8B | 8B | TP=1 | 16GB |
| LLaMA-3-70B | 70B | TP=4 | 4×24GB |
| DeepSeek-V2 | 236B | TP=16 | 8×H100 |
| GLM-4-9B | 9B | TP=1 | 18GB |
| Yi-1.5-34B | 34B | TP=2 | 2×40GB |

### 推理引擎

- ✅ **vLLM** — PagedAttention, 持续批处理
- ✅ **TGI** — HuggingFace官方推理服务器
- ✅ **SGLang** — RadixAttention, 树搜索
- ✅ **TensorRT-LLM** — NVIDIA官方优化
- 🔄 **更多引擎支持中...**

---

## 📁 项目结构

```python
QuantumFlow/
├── quantumflow/              # 🎯 核心包
│   ├── api/                  # REST API
│   │   ├── routes/          # API 路由
│   │   ├── models/          # 请求/响应模型
│   │   └── server.py        # FastAPI 应用
│   │
│   ├── scheduler/           # 🧠 调度器核心
│   │   ├── scheduler.py     # 调度器主逻辑
│   │   └── strategy/        # 调度策略
│   │       ├── gang.py       # Gang策略
│   │       ├── pack.py       # Pack策略
│   │       └── adaptive.py   # 自适应策略
│   │
│   ├── cluster/             # 🖥️ 集群管理
│   │   └── manager.py       # 节点管理
│   │
│   ├── inference/           # ⚡ 推理引擎
│   │   ├── engine.py        # 引擎抽象
│   │   └── backends/        # 引擎实现
│   │       ├── vllm.py       # vLLM后端
│   │       ├── tgi.py        # TGI后端
│   │       └── sglang.py     # SGLang后端
│   │
│   ├── worker/              # 🚀 Worker节点
│   │   └── worker.py        # Worker实现
│   │
│   ├── storage/             # 💾 存储层
│   │   └── redis_queue.py   # Redis队列
│   │
│   ├── models/              # 📦 模型注册表
│   │   └── registry.py      # 模型管理
│   │
│   ├── monitoring/          # 📊 监控指标
│   │   └── metrics.py       # Prometheus指标
│   │
│   └── utils/               # 🛠️ 工具
│       ├── config.py        # 配置管理
│       ├── logging.py       # 日志系统
│       └── retry.py         # 重试机制
│
├── scripts/                  # 📜 启动脚本
│   ├── start_controller.py  # Controller启动
│   ├── start_worker.py      # Worker启动
│   └── quickstart.sh        # 快速启动
│
├── configs/                  # ⚙️ 配置文件
│   ├── default.yaml         # 默认配置
│   ├── development.yaml     # 开发配置
│   └── production.yaml      # 生产配置
│
├── tests/                    # 🧪 测试
│   ├── unit/                # 单元测试
│   └── integration/         # 集成测试
│
├── docs/                     # 📚 文档
│   └── ARCHITECTURE.md      # 架构文档
│
├── pyproject.toml           # 📦 项目配置
├── README.md                 # 📖 说明文档
└── LICENSE                  # 📜 许可证
```

---

## 🔧 配置示例

```yaml
# configs/production.yaml
app:
  name: "QuantumFlow"
  environment: "production"
  log_level: "INFO"

scheduler:
  default_strategy: "adaptive"
  max_concurrent_requests: 5000
  queue_max_size: 50000
  strategies:
    gang:
      enabled: true
      timeout_seconds: 600
    pack:
      enabled: true
      max_batch_size: 64

inference:
  default_backend: "vllm"
  backends:
    vllm:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.92
      max_model_len: 8192
      enable_chunked_prefill: true

cluster:
  heartbeat_interval_seconds: 5
  heartbeat_timeout_seconds: 60
```

---

## 🤝 贡献

我们欢迎所有形式的贡献！

```bash
# 1. Fork 项目
# 2. 创建特性分支
git checkout -b feature/amazing-feature

# 3. 提交更改
git commit -m "feat: add amazing feature"

# 4. 推送分支
git push origin feature/amazing-feature

# 5. 创建 Pull Request
```

### 开发环境

```bash
# 克隆并安装
git clone https://github.com/quantumflow/quantumflow.git
cd quantumflow
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码格式化
black quantumflow/
isort quantumflow/
ruff check quantumflow/

# 类型检查
mypy quantumflow/
```

---

## 🏃 快速部署

### 单节点部署

```bash
# 启动Controller（API服务器）
quantumflow serve --host 0.0.0.0 --port 8000

# 在另一终端启动Worker
quantumflow worker --controller-url http://localhost:8000 --port 8080
```

### 多节点集群部署

```bash
# 启动Controller
quantumflow serve --host 0.0.0.0 --port 8000

# 在每个GPU节点启动Worker
quantumflow worker --controller-url http://localhost:8000 --port 8080 --backend vllm
```

### 使用Docker Compose

```yaml
version: '3.8'
services:
  controller:
    image: quantumflow/quantumflow:latest
    command: serve --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    depends_on:
      - redis
    volumes:
      - ./configs:/app/configs

  worker:
    image: quantumflow/quantumflow:latest
    command: worker --controller-url http://controller:8000 --port 8080
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## 📖 API 使用

### REST API

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 集群状态
curl http://localhost:8000/api/v1/cluster/status

# 列出模型
curl http://localhost:8000/api/v1/models

# 部署模型
curl -X POST http://localhost:8000/api/v1/models/deploy \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct", "tensor_parallel": 1}'

# 推理
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "prompt": "解释量子计算",
    "sampling_params": {
      "temperature": 0.7,
      "max_tokens": 100
    }
  }'
```

### Python SDK

```python
from quantumflow import QuantumFlow

# 创建客户端
qf = QuantumFlow(api_url="http://localhost:8000")

# 查看集群状态
status = qf.cluster.status()
print(f"健康节点: {status['healthy_nodes']}")

# 部署模型
qf.deploy("Qwen2.5-7B-Instruct", tensor_parallel=1)

# 推理
result = qf.generate(
    model="Qwen2.5-7B-Instruct",
    prompt="什么是大语言模型？",
    temperature=0.7,
    max_tokens=200
)
print(result.text)

# 流式推理
for chunk in qf.generate_stream(
    model="Qwen2.5-7B-Instruct",
    prompt="写一首关于AI的诗"
):
    print(chunk, end="", flush=True)
```

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行特定模块测试
pytest tests/unit/scheduler/ -v

# 生成覆盖率报告
pytest tests/ --cov=quantumflow --cov-report=html
```

---

## 📜 许可证

本项目基于 Apache License 2.0 许可证开源。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

本项目站在巨人的肩膀上：

- [vLLM](https://github.com/vllm-project/vllm) — 高效的PagedAttention实现
- [Ray](https://github.com/ray-project/ray) — 分布式计算框架
- [K8s](https://kubernetes.io/) — 容器编排参考
- 所有开源贡献者！

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐**

*Built with ❤️ by the QuantumFlow Team*

</div>
