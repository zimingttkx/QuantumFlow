# QuantumFlow

> Production-grade distributed LLM inference platform.
> 像调度 Kubernetes Pods 一样调度 AI 推理任务。

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 系统设计：模块划分、数据流、调度策略、容灾层、部署拓扑 |
| [API.md](./API.md) | 接口文档：REST API、gRPC API、SDK、CLI 命令 |
| [DEPLOYMENT.md](./DEPLOYMENT.md) | 部署文档：环境要求、本地与生产部署、监控、多后端端口 |
| [TESTING.md](./TESTING.md) | 测试文档：测试策略、跑测指南、覆盖率、已知失败 |

---

## 项目简介

QuantumFlow 提供：

- **多策略调度**：Gang / Pack / Adaptive 自动选择
- **多后端推理**：vLLM / HuggingFace / TGI / SGLang / TensorRT-LLM
- **分布式部署**：Redis 队列 + Worker 节点，Controller 与 Worker 解耦
- **GPU 优化**：BatchAccumulator / Block VRAM 精细管理
- **多租户隔离**：API Key 认证、租户配额、显存隔离、限流
- **企业级容灾**：FailoverController / ReplicaManager / LeaderElection / HealthChecker
- **gRPC 高性能接口**：二进制序列化、流式 RPC、拦截器

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Redis
docker run -d -p 6379:6379 --name qf-redis redis:7-alpine

# 3. 启动 Controller（API Server）
python -m quantumflow.cli serve --host 0.0.0.0 --port 8000

# 4. 启动 Worker
python scripts/start_worker.py
```

详细步骤见 [DEPLOYMENT.md](./DEPLOYMENT.md)。

---

## 技术栈

| 层 | 技术 |
|----|------|
| Web 框架 | FastAPI + Pydantic + Uvicorn |
| 调度 | asyncio + 自定义算法 + Redis ZSET |
| 推理 | vLLM / PyTorch / CUDA |
| 存储 | Redis（队列、租户、状态） |
| RPC | gRPC + Protocol Buffers |
| 监控 | Prometheus + Grafana |
| 基础设施 | Docker / Kubernetes |

---

*最后更新：2026-06-17*
