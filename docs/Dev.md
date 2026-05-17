# 开发文档 — 待开发内容

本文档记录 QuantumFlow 尚未完成的功能规划。

---

## 📋 待开发功能

### 1. 多后端完善

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **TGI 后端** | Text Generation Inference 后端适配 | 📋 规划中 |
| **SGLang 后端** | SGLang 后端适配 | 📋 规划中 |
| **TensorRT-LLM 后端** | NVIDIA TensorRT-LLM 推理引擎 | 📋 规划中 |

### 2. 分布式部署

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **Worker 节点** | 支持多节点横向扩展 | 📋 规划中 |
| **Controller-Worker 通信** | Controller 与 Worker 之间的高效通信 | 📋 规划中 |
| **Redis 队列** | 分布式任务队列（替代当前单机队列） | 📋 规划中 |

### 3. 高级调度策略

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **Gang Scheduler** | All-or-Nothing 调度，适合 100B+ 大模型分布式推理 | 📋 规划中 |
| **Adaptive Scheduler** | AI 驱动的自适应调度策略 | 📋 规划中 |
| **Priority Queue** | 带优先级的任务队列 | 📋 规划中 |

### 4. 企业级特性

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **多租户隔离** | 租户间显存隔离，资源配额 | 📋 规划中 |
| **限流保护** | Token bucket / Leaky bucket 限流 | 📋 规划中 |
| **容灾备份** | 模型副本、自动故障转移 | 📋 规划中 |

### 5. 硬件支持

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **昇腾 NPU** | 华为昇腾 NPU 深度适配，打破 NVIDIA 垄断 | 📋 规划中 |
| **Cambricon NPU** | 寒武纪 NPU 支持 | 📋 规划中 |

### 6. FlashAttention 集成

当前环境 CUDA 版本不匹配（PyTorch 编译用 CUDA 13.0，系统 12.4），`flash-attn` 无法安装。

**解决方案**：
- 方案一：重新编译 PyTorch 匹配系统 CUDA 12.4 版本后安装
- 方案二：等待环境升级到 CUDA 13+
- 方案三：使用 `xformers` 作为替代（`model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="xformers")`）

**收益**：显存占用减少 2-4x，长序列速度提升 1.5-3x

### 7. 调度可视化完善

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **Scheduler 集成** | 将 Scheduler 真正接入推理请求路径（目前 Scheduler 仅作为状态展示） | 📋 规划中 |
| **实时 Block 追踪** | 每个请求的 Block 分配/释放实时可视化 | 📋 规划中 |
| **Eviction 事件记录** | 模型淘汰历史记录 | 📋 规划中 |

### 8. BatchAccumulator 增强

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **多模型共享** | 支持不同模型共享 BatchAccumulator | 📋 规划中 |
| **动态 batch_size** | 根据 GPU 显存动态调整 max_batch_size | 📋 规划中 |
| **优先级感知** | 高优先级请求优先插入 batch | 📋 规划中 |

### 9. 测试覆盖

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **vLLM 后端测试** | vLLM backend 集成测试 | 📋 规划中 |
| **并发压力测试** | 高并发下的稳定性和性能测试 | 📋 规划中 |
| **Eviction 单元测试** | 模型淘汰逻辑测试 | 📋 规划中 |

---

## 🔧 技术债务

- Torch fallback 路径下的 `memory_util_pct` 仍为 0（需要确保始终走 pynvml 路径）
- `memory_util_pct` 在 benchmark 数据中显示为 0，需修复 GPU Monitor 的 Torch fallback

---

## 📊 Benchmark 扩展

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **vLLM 基准测试** | 在更大显存机器上重新测试 vLLM high concurrency | 📋 规划中 |
| **7B 模型测试** | Qwen2.5-7B 推理性能测试 | 📋 规划中 |
| **FlashAttention 对比** | 安装 FlashAttention 后与标准 Attention 对比 | 📋 规划中 |
| **多 GPU 测试** | 多卡并行推理性能测试 | 📋 规划中 |
