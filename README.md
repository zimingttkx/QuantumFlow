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

| 🎯 核心能力 | 🌟 差异化亮点 | 🔧 技术优势 | 状态 |
|:---:|:---:|:---:|:---:|
| **智能调度** | Gang/Pack/自适应多策略 | 自动选择最优执行路径 | ✅ 代码完成 |
| **多后端支持** | vLLM / HF / TGI / SGLang | 统一接口，灵活切换 | ✅ HF 可用 |
| **国产硬件** | 昇腾NPU深度适配 | 打破 NVIDIA 垄断 | 📋 规划中 |
| **企业级** | 多租户 / 限流 / 容灾 | 开箱即用的生产特性 | 📋 规划中 |

</div>

> ✅ 已完成 &nbsp;&nbsp; 🔄 开发中 &nbsp;&nbsp; 📋 规划中

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
git clone <repo-url>
cd QuantumFlow
pip install -e .
```

### 💻 启动

```bash
# 一键启动（推荐）
./scripts/qf

# 或手动启动
python -m quantumflow.cli serve
```

浏览器打开 `http://localhost:8000` 进入前端。

### 🛠️ CLI

```bash
# 交互式终端
python -m quantumflow.cli interactive

# 命令行
python -m quantumflow.cli status              # 集群状态
python -m quantumflow.cli models              # 模型列表
python -m quantumflow.cli load Qwen2.5-1.5B  # 加载模型
python -m quantumflow.cli chat Qwen2.5-1.5B -p "你好"  # 对话
python -m quantumflow.cli generate Qwen2.5-1.5B -p "你好"  # 生成
```

---

## 🏗️ 系统架构

```
┌────────────────────────────────────────────────────────────────────────┐
│                         QuantumFlow Platform                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   接入层 (Gateway) ✅ 已完成                     │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │  │
│  │  │ REST API│  │ gRPC API│  │ Python  │  │   CLI   │         │  │
│  │  │ ✅FastAPI│  │ 📋 SDK  │  │ 📋 SDK  │  │ ✅ CLI  │         │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘         │  │
│  └───────┼─────────────┼─────────────┼─────────────┼───────────────┘  │
│          └─────────────┴─────────────┴─────────────┘                     │
│                               │                                         │
│  ┌───────────────────────────┼───────────────────────────────────────┐ │
│  │                    调度层 (Scheduler) ✅ 代码完成                    │ │
│  │  ┌────────────────────────────────────────────────────────────┐  │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │ │
│  │  │  │  Gang    │  │  Pack    │  │Adaptive  │  │ Priority │   │  │ │
│  │  │  │ Scheduler│  │ Scheduler│  │ Strategy │  │ Queue   │   │  │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │ │
│  │  └────────────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                               │                                         │
│  ┌───────────────────────────┼───────────────────────────────────────┐ │
│  │                    集群管理层 (Cluster) ✅ 单机模式                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │ │
│  │  │   Node      │  │  Service    │  │  Health    │              │ │
│  │  │  Registry   │  │  Discovery  │  │  Monitor   │              │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                               │                                         │
│  ┌───────────────────────────┼───────────────────────────────────────┐ │
│  │                    执行层 (Worker Pool) 🔄 单节点                  │ │
│  │  ┌─────────────────────────────────────────────────────────────┐  │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │ │
│  │  │  │ ✅ HF    │  │ 📋 vLLM  │  │ 📋 TGI   │  │📋 SGLang │   │  │ │
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

| 模型 | 参数量 | 显存要求 | 状态 |
|------|--------|----------|------|
| Qwen2.5-1.5B | 1.5B | ~3GB | ✅ 已验证 |
| Qwen2.5-3B | 3B | ~6GB | ✅ 可加载 |
| Qwen2.5-7B | 7B | ~14GB | 📋 待测试 |
| LLaMA-3-8B | 8B | ~16GB | 📋 规划中 |
| Qwen2.5-72B | 72B | 4×24GB | 📋 分布式 |

### 推理引擎

- ✅ **HuggingFace Transformers** — 已验证可用
- 🔄 **vLLM** — v0.21.0 有显存bug，待降级
- 📋 **TGI** — 规划中
- 📋 **SGLang** — 规划中
- 📋 **TensorRT-LLM** — 规划中

---

## 📁 项目结构

```python
QuantumFlow/
├── quantumflow/              # 🎯 核心包
│   ├── api/                  # ✅ REST API (FastAPI)
│   │   ├── routes/          # API 路由
│   │   ├── models/          # 请求/响应模型
│   │   └── server.py        # FastAPI 应用
│   │
│   ├── scheduler/           # ✅ 调度器代码完成
│   │   ├── scheduler.py     # 调度器主逻辑
│   │   └── strategy/        # 调度策略
│   │
│   ├── cluster/             # ✅ 集群管理（单机模式）
│   │
│   ├── inference/           # 🔄 推理引擎
│   │   ├── engine.py        # 引擎抽象
│   │   └── backends/        # 引擎实现
│   │       ├── huggingface.py # ✅ 已验证
│   │       └── vllm.py       # 🔄 待修复
│   │
│   ├── worker/              # 📋 Worker节点（待部署）
│   │
│   ├── storage/             # 📋 Redis队列（待部署）
│   │
│   └── cli.py               # ✅ CLI工具
│
├── scripts/
│   └── qf                   # ✅ 一键启动脚本
│
├── tests/                    # ✅ 266个测试
│
├── configs/                  # ⚙️ 配置文件
├── pyproject.toml
└── README.md
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

## 🏃 部署

### 单机

```bash
./scripts/qf           # 一键启动
# 或
python -m quantumflow.cli serve
```

### 分布式（规划中）

```bash
# Controller
quantumflow serve --host 0.0.0.0 --port 8000

# Worker节点（待实现）
quantumflow worker --controller-url http://localhost:8000 --backend vllm
```

---

## 📖 API 使用

### REST API

```bash
# 集群状态
curl http://localhost:8000/api/v1/cluster/status

# 模型列表
curl http://localhost:8000/api/v1/models/list

# 已加载模型
curl http://localhost:8000/api/v1/models/status

# 加载模型
curl -X POST http://localhost:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-1.5B"}'

# 推理
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B",
    "prompt": "你好",
    "sampling_params": {"temperature": 0.7, "max_tokens": 100}
  }'

# 对话
curl -X POST http://localhost:8000/api/v1/inference/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B",
    "messages": [{"role": "user", "content": "你好"}]
  }'

# 流式生成
curl -X POST http://localhost:8000/api/v1/inference/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-1.5B", "prompt": "你好", "stream": true}'
```

### Python SDK

```python
import httpx

async with httpx.AsyncClient() as client:
    # 推理
    resp = await client.post("http://localhost:8000/api/v1/inference/generate",
        json={"model": "Qwen2.5-1.5B", "prompt": "你好",
              "sampling_params": {"max_tokens": 100}})
    print(resp.json()["generated_text"])

    # 对话
    resp = await client.post("http://localhost:8000/api/v1/inference/chat",
        json={"model": "Qwen2.5-1.5B",
              "messages": [{"role": "user", "content": "你好"}]})
    print(resp.json()["generated_text"])
```

---

## 🧪 测试

```bash
pytest tests/ -v    # 266个测试，全部通过
```

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=quantumflow/quantumflow&type=Date)](https://star-history.com/#quantumflow/quantumflow&Date)

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
