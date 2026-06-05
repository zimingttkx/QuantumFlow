# QuantumFlow

> A distributed LLM inference platform — turn a GPU cluster into a production-ready inference service.

[![Python](https://img.shields.io/badge/Python-3.10+-00D9FF?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%20%7C%2013.0-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

[English](README.en.md) · [中文](README.md)

---

## What is QuantumFlow

**QuantumFlow is a distributed LLM inference platform built for production.**

When you have 10+ GPUs and need to serve multiple users, models, and business lines at the same time, QuantumFlow gives you:

- **Unified access** — REST / gRPC / Python SDK / CLI / Web Playground, five entry points
- **Smart scheduling** — Gang (all-or-nothing, for large models) / Pack (shared, for small models) / Adaptive (AI-chosen), auto-switching
- **Multi-backend** — vLLM / HuggingFace / TGI / SGLang / TensorRT-LLM, 5 inference engines under one interface
- **Multi-tenant SaaS** — API Key auth, quota, rate limiting, VRAM isolation, per-tenant billing
- **Observable** — Prometheus metrics, health checks, node heartbeats, Grafana ready

**What it is not**: it's not just an inference engine (that's vLLM's job), and it's not a container orchestrator (that's K8s). It sits between the two, packaging "model + GPU cluster + multi-user" into a usable product.

---

## Use Cases

The core patterns the platform supports:

**Pattern 1: Multi-priority workload coexistence**

Customer support chat, internal knowledge base queries, and batch offline jobs all hit the same pool of GPUs — but their latency tolerance is wildly different. The first demands millisecond-level response, the second can tolerate queuing, and the third can even be interrupted and retried.

Requests are tiered via a `priority` field. The scheduler runs high-priority jobs first; low-priority jobs reuse idle capacity. An anti-starvation mechanism auto-promotes low-priority jobs that have been waiting too long.

**Pattern 2: Pooled sharing of small models**

Business lines need models of varying sizes (1.5B, 7B, 13B), but the QPS of any single model is low — dedicating a GPU to each one is wasteful. The goal is to pack a batch of small models into one GPU, keep hot models resident, and swap in cold models on demand.

The scheduler uses the Pack strategy to load multiple small models onto the same GPU concurrently; the inference layer routes requests through a shared batching channel; LRU eviction kicks in when VRAM runs short.

**Pattern 3: Cross-business cost allocation**

Multiple departments share a GPU cluster, but each has its own budget. You need to know how much each department consumed and how to split the cost.

Each department gets a dedicated tenant with its own API Key and quota. The system records per-tenant usage by token count / request count and exports billing reports.

**Pattern 4: Private deployment + cross-AZ disaster recovery**

An internal MaaS platform that must keep sensitive data inside the network boundary, with automatic failover when a single datacenter goes down.

Workers span multiple datacenters. The Controller schedules through a Redis ZSET. Failed nodes are auto-marked UNHEALTHY and removed from the scheduling pool; replicas are re-dispatched to healthy nodes.

---

## Core Capabilities

| Dimension | Capability |
|------|------|
| **Scheduling** | Gang / Pack / Adaptive strategies + Redis ZSET priority queue + distributed scheduler |
| **Inference** | Unified interface for 5 backends + dynamic batching (50ms window) + priority-aware + dynamic batch_size + multi-model sharing |
| **VRAM** | Block Pool fine-grained allocation (vLLM-style PagedAttention) + LRU model eviction + real-time VRAM monitoring |
| **Cluster** | Multi-worker nodes + heartbeat health checks + auto-failover + node state machine (HEALTHY/UNHEALTHY/DRAINING) |
| **Multi-tenant** | API Key auth + tenant quota + Token Bucket rate limiting (global / per-tenant / per-endpoint, three layers) + VRAM isolation |
| **Protocols** | REST (FastAPI) + gRPC (Protobuf) + Python SDK (Sync/Async) + CLI + Web Playground |
| **Observability** | Prometheus metrics + structlog structured logs + health checks (live / ready) + heartbeat reporting |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         QuantumFlow Platform                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Access     REST API  │  gRPC API  │  Python SDK  │  CLI          │
│   Layer      ──────────────────────────────────────────────────    │
│                Middleware: TenantAuth → RateLimit → Handler        │
│                              │                                     │
│   Sched.     DistributedScheduler ─ Redis ZSET priority queue      │
│   Layer      Strategies: Gang / Pack / Adaptive / Priority         │
│                              │                                     │
│   Exec.      EngineManager ─ VRAMManager ─ BatchAccumulator       │
│   Layer      SharedBatchCoordinator ─ GPUMonitor (NVML)           │
│               Backends: vLLM / HuggingFace / TGI / SGLang / TRT    │
│                              │                                     │
│   Cluster    NodeRegistry ─ ServiceDiscovery ─ HealthMonitor       │
│   Layer      Node states: HEALTHY / DRAINING / UNHEALTHY / OFFLINE │
│                              │                                     │
│   Monitor    Prometheus Metrics ─ structlog JSON Logs              │
│   Layer                                                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- CUDA 12.4 or 13.0 (only required for inference; pure scheduling / cluster management works without it)
- Redis ≥ 7.0 (for Worker ↔ Controller communication)
- Linux recommended (Windows users need WSL2, see [DEPLOYMENT.md](./docs/DEPLOYMENT.md))

### Installation

```bash
# Clone
git clone https://github.com/quantumflow/quantumflow.git
cd QuantumFlow

# Core dependencies (cross-platform)
pip install -r requirements.txt

# GPU acceleration (Linux only)
pip install -r requirements-gpu.txt

# Development tools (testing / formatting / type checking)
pip install -r requirements-dev.txt
```

### Run

```bash
# 1. Start Redis
docker run -d -p 6379:6379 --name qf-redis redis:7-alpine

# 2. Start the Controller (API Server)
python -m quantumflow.cli serve --host 0.0.0.0 --port 8000

# 3. (Optional) Start Worker nodes
python scripts/start_worker.py

# 4. Open in browser
#    http://localhost:8000/docs              → Swagger UI (live API playground)
#    http://localhost:8000/static/playground.html  → Interactive Playground
```

### Python SDK

```python
from quantumflow import SyncClient

client = SyncClient(base_url="http://localhost:8000", api_key="qf-dev-xxx")

# Text generation
result = client.generate(
    model="Qwen2.5-7B-Instruct",
    prompt="Explain distributed systems in one sentence",
    sampling_params={"temperature": 0.7, "max_tokens": 100}
)
print(result.generated_text)

# Streaming generation
for chunk in client.stream(model="Qwen2.5-7B-Instruct", prompt="Write a short poem"):
    print(chunk.delta, end="", flush=True)
```

### CLI

```bash
# Cluster status
python -m quantumflow.cli status

# Load a model
python -m quantumflow.cli load Qwen2.5-7B-Instruct

# Generate
python -m quantumflow.cli generate Qwen2.5-7B-Instruct -p "Hello"

# Interactive terminal
python -m quantumflow.cli interactive
```

---

## Documentation

Full documentation lives in [docs/](./docs/):

| Document | Content |
|------|------|
| [ARCHITECTURE.md](./docs/ARCHITECTURE.md) | System design: module layout, key designs, data flow, scheduling strategies, deployment topology |
| [API.md](./docs/API.md) | API reference: REST API, gRPC API, CLI commands, error responses |
| [DEPLOYMENT.md](./docs/DEPLOYMENT.md) | Deployment guide: environment requirements, local and production (K8s) deployment, monitoring, upgrades |
| [TESTING.md](./docs/TESTING.md) | Testing docs: test strategy, running tests, coverage, gRPC testing |

---

## Performance

### Benchmarks (RTX 4080 Laptop 12GB / Qwen2.5-1.5B / HuggingFace + BatchAccumulator)

| Scenario | Description | GPU Util | P50 Latency | Throughput |
|------|------|:---------:|:--------:|:----:|
| A | Single request (greedy, short prompt) | 59% | 509 ms | 76.6 tok/s |
| B | Short dialog 8 concurrent | 51% | 1058 ms | 65.9 tok/s |
| D | Long prompt + generation | 69% | 2016 ms | 83.4 tok/s |
| H | High concurrency 32 concurrent | 54% | 462 ms | 85.5 tok/s |

**Industry comparison**:
- Target: 80%+ GPU compute utilization, 80%+ VRAM bandwidth utilization
- vLLM Continuous Batching in production: typically 60-85% GPU utilization
- Full 10-scenario test: see benchmark data in the repository's git history

### GPU Metrics

Two independent metrics collected via NVIDIA NVML:

| Metric | Source | Meaning |
|------|------|------|
| GPU Compute Utilization | `nvmlDeviceGetUtilizationRates().gpu` | CUDA core activity |
| GPU Memory Bandwidth | `nvmlDeviceGetUtilizationRates().memory` | HBM memory controller activity |

---

## Deployment

### Local Development (Docker Compose)

```bash
docker run -d -p 6379:6379 --name qf-redis redis:7-alpine
python -m quantumflow.cli serve
```

### Production (Kubernetes)

```bash
# Deploy Redis
kubectl apply -f deploy/k8s/redis.yaml

# Deploy Controller (REST + gRPC)
kubectl apply -f deploy/k8s/controller.yaml

# Deploy Worker pool (A100 / 4090 / H100 as needed)
kubectl apply -f deploy/k8s/worker-a100.yaml
kubectl apply -f deploy/k8s/worker-4090.yaml

# Hook up Prometheus
kubectl apply -f deploy/k8s/monitoring.yaml
```

For full YAML templates, HPA config, monitoring & alerting rules, and TLS setup, see [DEPLOYMENT.md](./docs/DEPLOYMENT.md).

---

## Star History

<a href="https://www.star-history.com/?repos=zimingttkx%2FQuantumFlow&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=zimingttkx/QuantumFlow&type=date&theme=dark&logscale&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=zimingttkx/QuantumFlow&type=date&logscale&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=zimingttkx/QuantumFlow&type=date&logscale&legend=top-left" />
 </picture>
</a>

---

## Contributing

Issues and Pull Requests are welcome.

```bash
# 1. Fork + clone
git clone https://github.com/your-fork/quantumflow.git

# 2. Install dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Run tests (start Redis first)
pytest tests/unit -v

# 4. Commit (follow Conventional Commits)
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature
```

**Code standards**:
- Python 3.10+, type annotations required
- `black` + `isort` + `ruff` + `mypy` must pass before commit
- Every new feature must have unit tests (coverage ≥ 80%)
- See [TESTING.md](./docs/TESTING.md) for testing principles

---

## Roadmap

| Direction | Status | Description |
|------|:----:|------|
| gRPC high-performance interface | ✅ | 5 Services + interceptor chain |
| Python SDK (Sync/Async) | ✅ | httpx + full authentication |
| TensorRT-LLM backend | ✅ | NVIDIA high-performance inference engine |
| Multi-tenant + rate limit + quota | ✅ | Token Bucket three-layer rate limiting |
| Priority-aware batching | ✅ | Anti-starvation mechanism |
| Dynamic batch_size | ✅ | VRAM-aware auto-tuning |
| Multi-model sharing | ✅ | SharedBatchCoordinator |
| Disaster recovery (model replicas + auto-failover) | 📋 | P1 |
| Domestic NPU (Ascend / Cambricon) | 📋 | P2 long-term |

---

## License

[Apache License 2.0](LICENSE)

---

## Acknowledgments

This project draws inspiration from these excellent open source projects:

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention + Continuous Batching
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — efficient attention kernel
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — inference engine foundation
- [Ray](https://github.com/ray-project/ray) — distributed computing framework
- [Kubernetes](https://kubernetes.io/) — container orchestration reference

---

<div align="center">

**If this project helps you, please give us a ⭐**

*Built with care by the QuantumFlow Team*

</div>
