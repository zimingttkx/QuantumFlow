# QuantumFlow 部署文档

> 环境要求、本地开发部署、生产部署（Kubernetes）、监控接入。

---

## 目录

- [环境要求](#1-环境要求)
- [本地开发部署](#2-本地开发部署)
- [生产部署（Kubernetes）](#3-生产部署kubernetes)
- [配置说明](#4-配置说明)
- [监控与日志](#5-监控与日志)
- [升级与回滚](#6-升级与回滚)
- [常见问题](#7-常见问题)

---

## 1. 环境要求

### 1.1 硬件

| 角色 | 最低配置 | 推荐配置 |
|------|----------|----------|
| Controller | 4 vCPU / 8 GB | 8 vCPU / 16 GB |
| Worker（小模型） | 1×GPU 16GB | 1×RTX 4090 24GB |
| Worker（大模型 70B+） | 4×GPU 40GB | 8×A100 80GB |
| Redis | 2 vCPU / 4 GB | 4 vCPU / 8 GB |

### 1.2 软件

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| CUDA | 12.4 或 13.0 |
| cuDNN | 匹配 CUDA |
| Docker | ≥ 24.0 |
| Kubernetes | ≥ 1.27 |
| Redis | ≥ 7.0 |
| NVIDIA Driver | ≥ 535 |

### 1.3 Python 依赖

```bash
pip install -r requirements.txt
```

核心依赖：FastAPI、uvicorn、pydantic、redis、httpx、grpcio、grpcio-tools、vllm、torch、transformers、prometheus-client、pynvml。

> 注：FlashAttention 需要 PyTorch 与系统 CUDA 严格匹配。若 `flash-attn` 安装失败，使用 `attn_implementation="xformers"` 替代。

---

## 2. 本地开发部署

### 2.1 单机开发（Docker Compose）

```bash
# 1. 启动 Redis
docker run -d \
  --name qf-redis \
  -p 6379:6379 \
  redis:7-alpine

# 2. 启动 Controller
python -m quantumflow.cli serve --host 0.0.0.0 --port 8000

# 3. 启动 Worker（新终端）
python scripts/start_worker.py

# 4. 验证
curl http://localhost:8000/api/v1/health
```

### 2.2 分布式开发（多进程）

```bash
# 终端 1：Controller
python -m quantumflow.cli serve --host 0.0.0.0 --port 8000

# 终端 2：Worker 节点 1
WORKER_ID=worker-1 WORKER_PORT=8001 \
  python scripts/start_worker.py

# 终端 3：Worker 节点 2
WORKER_ID=worker-2 WORKER_PORT=8002 \
  python scripts/start_worker.py
```

### 2.3 本地开发架构

```
┌──────────────────────────────────────┐
│         Development Machine            │
├──────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐         │
│  │Controller │  │ Worker   │         │
│  │  :8000    │  │  :8001   │         │
│  └─────┬─────┘  └────┬─────┘         │
│        │             │               │
│        └──────┬──────┘               │
│               ▼                       │
│  ┌──────────────────────────────┐    │
│  │      Redis :6379              │    │
│  └──────────────────────────────┘    │
│               │                       │
│               ▼                       │
│  ┌──────────────────────────────┐    │
│  │  GPU: 1×RTX 4090              │    │
│  │  vLLM + Qwen2.5-7B           │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
```

---

## 3. 生产部署（Kubernetes）

### 3.1 命名空间与配置

```bash
kubectl create namespace quantumflow
kubectl -n quantumflow create secret generic qf-redis \
  --from-literal=password=<your-redis-password>

kubectl -n quantumflow create configmap qf-config \
  --from-file=configs/production.yaml
```

### 3.2 Controller Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qf-controller
  namespace: quantumflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qf-controller
  template:
    metadata:
      labels:
        app: qf-controller
    spec:
      containers:
      - name: controller
        image: quantumflow/controller:latest
        ports:
        - containerPort: 8000
        - containerPort: 50051
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: qf-redis
              key: password
        - name: GRPC_ENABLED
          value: "true"
        - name: GRPC_PORT
          value: "50051"
        resources:
          requests: {cpu: "2", memory: "4Gi"}
          limits:   {cpu: "4", memory: "8Gi"}
        livenessProbe:
          httpGet: {path: /api/v1/health/live, port: 8000}
        readinessProbe:
          httpGet: {path: /api/v1/health/ready, port: 8000}
---
apiVersion: v1
kind: Service
metadata:
  name: qf-controller
  namespace: quantumflow
spec:
  selector: {app: qf-controller}
  ports:
  - {name: rest, port: 8000,  targetPort: 8000}
  - {name: grpc, port: 50051, targetPort: 50051}
```

### 3.3 Worker Deployment（GPU 节点池）

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qf-worker-a100
  namespace: quantumflow
spec:
  replicas: 4
  selector:
    matchLabels:
      app: qf-worker
      pool: a100
  template:
    metadata:
      labels:
        app: qf-worker
        pool: a100
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
      containers:
      - name: worker
        image: quantumflow/worker:latest
        env:
        - name: CONTROLLER_URL
          value: "http://qf-controller:8000"
        - name: REDIS_URL
          valueFrom: {secretKeyRef: {name: qf-redis, key: password}}
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: 256Gi
        volumeMounts:
        - {name: models, mountPath: /data/models}
      volumes:
      - {name: models, persistentVolumeClaim: {claimName: qf-models}}
```

### 3.4 Redis 集群

生产建议使用 Redis Sentinel 或 Redis Cluster（3 节点）。最低 3 副本 + 1 Sentinel。

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: quantumflow
spec:
  replicas: 3
  serviceName: redis
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports: [{containerPort: 6379}]
        resources:
          requests: {cpu: "1", memory: "2Gi"}
          limits:   {cpu: "2", memory: "4Gi"}
        volumeMounts:
        - {name: data, mountPath: /data}
  volumeClaimTemplates:
  - metadata: {name: data}
    spec:
      accessModes: [ReadWriteOnce]
      resources: {requests: {storage: 50Gi}}
```

### 3.5 部署命令

```bash
# 1. 应用清单
kubectl -n quantumflow apply -f k8s/redis.yaml
kubectl -n quantumflow apply -f k8s/controller.yaml
kubectl -n quantumflow apply -f k8s/worker-a100.yaml
kubectl -n quantumflow apply -f k8s/worker-4090.yaml

# 2. 等待就绪
kubectl -n quantumflow wait --for=condition=ready pod -l app=qf-controller --timeout=300s
kubectl -n quantumflow wait --for=condition=ready pod -l app=qf-worker     --timeout=300s

# 3. 验证
kubectl -n quantumflow port-forward svc/qf-controller 8000:8000
curl http://localhost:8000/api/v1/health
```

### 3.6 生产部署架构

```
┌──────────────────────────────────────────────────┐
│              Kubernetes Cluster                    │
├──────────────────────────────────────────────────┤
│                                                    │
│   Control Plane                                    │
│   ├── Controller Pods ×3 (REST + gRPC)             │
│   ├── Redis Cluster ×3                             │
│   └── Prometheus + Grafana                         │
│                                                    │
│   GPU Node Pool A (A100 8×)                        │
│   └── Worker Pods (Qwen-72B / LLaMA-70B / DeepSeek)│
│                                                    │
│   GPU Node Pool B (RTX 4090 4×)                    │
│   └── Worker Pods (Qwen-7B / 小模型)              │
│                                                    │
└──────────────────────────────────────────────────┘
```

---

## 4. 配置说明

配置文件：`configs/default.yaml`

```yaml
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

scheduler:
  strategy: "adaptive"          # gang | pack | adaptive
  redis_url: "redis://localhost:6379/0"
  queue_key: "quantumflow:pending_requests"

grpc:
  enabled: true
  port: 50051
  max_workers: 10
  interceptors:
    logging:    {enabled: true}
    auth:       {enabled: false}
    rate_limit: {enabled: true, qps: 100}
  connection_pool:
    max_size: 10
    keepalive_ms: 30000

monitoring:
  prometheus_port: 9090
  metrics_path: /api/v1/metrics
```

多环境配置：

```bash
CONFIG_FILE=configs/production.yaml python -m quantumflow.cli serve
```

---

## 5. 监控与日志

### 5.1 Prometheus 抓取配置

```yaml
scrape_configs:
  - job_name: 'quantumflow'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: qf-(controller|worker)
        action: keep
      - source_labels: [__meta_kubernetes_pod_port_number]
        regex: "8000"
        action: keep
    metrics_path: /api/v1/metrics
    scrape_interval: 15s
```

### 5.2 关键告警

| 指标 | 告警条件 |
|------|----------|
| `qf_gpu_memory_used_bytes` | > 90% 持续 5min |
| `qf_scheduler_queue_size` | > 1000 持续 2min |
| `qf_inference_latency_seconds` (P99) | > 5s 持续 5min |
| Controller Pod 不可用 | down > 1min |
| Redis 不可达 | PING 失败 |

### 5.3 日志

使用 `structlog` 输出 JSON 格式日志至 stdout。生产环境建议：

- **Loki** 聚合 Controller / Worker 日志
- **Elasticsearch + Fluentd** 全文检索
- 关键字段：`request_id`、`tenant_id`、`model`、`latency_ms`

---

## 6. 升级与回滚

### 6.1 滚动升级

```bash
# 设置新镜像
kubectl -n quantumflow set image deployment/qf-controller \
  controller=quantumflow/controller:v1.1.0

# 观察
kubectl -n quantumflow rollout status deployment/qf-controller
```

### 6.2 回滚

```bash
kubectl -n quantumflow rollout undo deployment/qf-controller
kubectl -n quantumflow rollout undo deployment/qf-worker-a100
```

### 6.3 兼容性

REST API 向后兼容。gRPC 遵循 `package quantumflow.v1;` 语义化版本管理。

---

## 7. 常见问题

| 问题 | 解决方案 |
|------|----------|
| `flash-attn` 安装失败 | 改用 `xformers`，或升级 CUDA 至 13.0+ |
| Controller 启动报 Redis 连接失败 | 检查 `REDIS_URL`，确认 Redis 可达且无密码错误 |
| Worker 注册后 60s 变 UNHEALTHY | 检查 Worker 与 Controller 网络，确认心跳端口可达 |
| `vllm` OOM | 调低 `gpu_memory_utilization`（如 0.7）或增大 `tensor_parallel_size` |
| 多租户 403 | 检查 `X-API-Key` 是否在 `qf:tenant:ids` 集合中存在 |
| gRPC 端口冲突 | 修改 `grpc.port` 配置并同步 Service |

---

*架构见 [ARCHITECTURE.md](./ARCHITECTURE.md)；接口见 [API.md](./API.md)。*
