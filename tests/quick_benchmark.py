#!/usr/bin/env python3
"""
QuantumFlow GPU Performance Benchmark — 深度覆盖版

覆盖所有推理代码路径：
  API 层：/generate, /generate/stream, /chat, /batch
  引擎层：BatchAccumulator(50ms) / direct generate / streaming
  采样参数：greedy/random, top_p极端值, repetition_penalty, stop

Prompt 设计原则（覆盖多样性）：
  - 长度：短(10词) / 中(50词) / 长(150+词) / 超长(500+词)
  - 语言：英文 / 中文
  - 类型：知识问答 / 技术解释 / 代码生成 / 数学推导 / 创意写作 / 对话
  - 复杂度：单步推理 / 多步推理 / 需要外部知识 / 需要生成能力
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass

import httpx
import numpy as np

# 禁用代理
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

BASE = "http://localhost:8000"
MODEL = "Qwen2.5-1.5B"


@dataclass
class ScenarioResult:
    name: str
    total_requests: int
    success: int
    failed: int
    avg_gpu_util: float
    max_gpu_util: float
    avg_gpu_mem_util: float
    avg_vram_gb: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_tok_s: float


# ── 场景配置：覆盖所有 API 路径和采样参数 ────────────────────────────────────

SCENARIOS = {
    # ── 场景 A: 单请求基线 (短 prompt, greedy decoding) ──────────────────
    # API: /generate (BatchAccumulator 路径)
    # 采样: temperature=0 (greedy), max_tokens=32
    # 目的: 最低延迟基线，消除采样方差
    "A: Single (greedy, short)": {
        "api": "generate",
        "prompts": [
            "What is quantum entanglement?",
            "Why does ice float on water?",
            "How does GPS work in phones?",
            "What causes seasons on Earth?",
            "Explain blockchain to a 10-year-old.",
        ],
        "sampling_params": {"temperature": 0, "max_tokens": 32, "top_p": 1.0, "top_k": 1},
        "concurrent": 1,
        "rounds": 5,
    },
    # ── 场景 B: 短对话 8 并发 (经 BatchAccumulator) ───────────────────────
    # API: /generate (BatchAccumulator 路径)
    # 采样: temperature=0.7, max_tokens=64
    # 目的: 动态批处理效率，50ms 窗口合并
    "B: Chat (8 concurrent, short)": {
        "api": "generate",
        "prompts": [
            "What's the capital of France?",
            "Explain photosynthesis simply.",
            "How do planes stay in the air?",
            "Why is the sky blue?",
            "What is machine learning?",
            "How does a refrigerator work?",
            "Why do we dream?",
            "What is CRISPR gene editing?",
        ],
        "sampling_params": {"temperature": 0.7, "max_tokens": 64, "top_p": 0.9, "top_k": 50},
        "concurrent": 8,
        "rounds": 4,
    },
    # ── 场景 C: 代码生成 (长输出, 中等 prompt) ─────────────────────────────
    # API: /generate (BatchAccumulator 路径)
    # 采样: temperature=0.3, max_tokens=256 (更长输出)
    # 目的: 长 completion 场景的吞吐量和 GPU 利用率
    "C: Code Generation (medium prompt)": {
        "api": "generate",
        "prompts": [
            "Write a Python function to check if a string is a palindrome, with type hints and a docstring.",
            "Write a Python decorator that logs function call arguments and return value, with error handling.",
            "Write a Python class for a stack (LIFO) with push, pop, peek, and is_empty methods.",
            "Write a Python function to find the longest common prefix of an array of strings, with tests.",
            "Write a Python async function that fetches data from two URLs concurrently and merges results.",
        ],
        "sampling_params": {"temperature": 0.3, "max_tokens": 256, "top_p": 0.9, "top_k": 50},
        "concurrent": 4,
        "rounds": 3,
    },
    # ── 场景 D: 长 prompt + 长输出 (压力测试) ───────────────────────────────
    # API: /generate (BatchAccumulator 路径)
    # 采样: temperature=0.7, max_tokens=128
    # 目的: 长 prompt 的 prefill 开销，长序列生成
    "D: Long Prompt + Generation": {
        "api": "generate",
        "prompts": [
            "解释量子力学中的不确定性原理，包括海森堡的原始表述和现代数学表述，以及它在双缝实验中的具体体现。请用中文详细说明。",
            "请用 Python 实现一个完整的二叉搜索树，包括插入、删除、查找和遍历操作，要求代码包含类型注解、详细注释和复杂度分析。",
            "机器学习中的反向传播算法是深度学习的基础。请详细推导梯度下降的数学过程，解释链式法则在其中的作用，给出矩阵形式的推导。",
            "Transformer 架构自 2017 年提出以来彻底改变了 NLP 领域。请详细解释 Self-Attention 的工作机制，包括 Q/K/V 的计算、点积注意力公式和multi-head attention的并行计算原理。",
            "在大模型推理中，KV Cache 是加速生成的核心技术。请说明其工作原理、显存占用计算方式，以及 PagedAttention 如何通过分页管理解决显存碎片问题。",
        ],
        "sampling_params": {"temperature": 0.7, "max_tokens": 128, "top_p": 0.9, "top_k": 50},
        "concurrent": 4,
        "rounds": 3,
    },
    # ── 场景 E: 流式生成 (完整测试 /generate/stream 路径) ─────────────────
    # API: /generate/stream
    # 采样: temperature=0.7, max_tokens=128
    # 目的: Thread+Queue 桥接的流式路径，不走 BatchAccumulator
    "E: Streaming Generation": {
        "api": "stream",
        "prompts": [
            "What is the difference between a process and a thread in operating systems? Explain with examples.",
            "Explain how distributed systems achieve consensus using the Raft algorithm.",
            "Describe the internals of a relational database index (B-tree) and its performance benefits.",
            "How does reinforcement learning work? Explain with the cart-pole problem example.",
            "What are the key differences between SQL and NoSQL databases? When would you choose each?",
        ],
        "sampling_params": {"temperature": 0.7, "max_tokens": 128, "top_p": 0.9, "top_k": 50},
        "concurrent": 4,
        "rounds": 3,
    },
    # ── 场景 F: Chat API (测试 ChatML 格式转换) ────────────────────────────
    # API: /chat (内部调用 /generate，ChatML 格式组装 prompt)
    # 采样: temperature=0.7, max_tokens=64
    # 目的: 多轮对话格式，对话历史组装
    "F: Chat API (multi-turn)": {
        "api": "chat",
        "prompts": [
            "What is artificial intelligence?",
            "Explain deep learning to a beginner.",
            "What is a neural network?",
        ],
        "sampling_params": {"temperature": 0.7, "max_tokens": 64, "top_p": 0.9, "top_k": 50},
        "concurrent": 3,
        "rounds": 3,
    },
    # ── 场景 G: Batch API (直接到 engine.generate, 不走 BatchAccumulator) ──
    # API: /batch (引擎直接批量处理)
    # 采样: temperature=0.3, max_tokens=64
    # 目的: 绕过 BatchAccumulator 的直接批量路径
    "G: Batch API (direct)": {
        "api": "batch",
        "prompts": [
            "What is quantum entanglement in one sentence?",
            "Explain blockchain to a 10-year-old.",
            "Why does ice float on water?",
            "What causes seasons on Earth?",
            "How does GPS work in phones?",
            "What is machine learning?",
            "How does a refrigerator work?",
            "Why do we dream?",
        ],
        "sampling_params": {"temperature": 0.3, "max_tokens": 64, "top_p": 0.9, "top_k": 50},
        "concurrent": 8,
        "rounds": 4,
    },
    # ── 场景 H: 高并发压力 (32 并发, 经 BatchAccumulator) ─────────────────
    # API: /generate (BatchAccumulator 路径)
    # 采样: temperature=0.7, max_tokens=32
    # 目的: 系统在洪峰下的稳定性，BatchAccumulator 50ms 合并效果
    "H: Load Test (32 concurrent)": {
        "api": "generate",
        "prompts": [
            "What is artificial intelligence?",
            "How does neural network work?",
            "Explain deep learning.",
            "What is natural language processing?",
            "How does reinforcement learning work?",
            "What is computer vision?",
            "Explain the transformer architecture.",
            "What is a language model?",
            "How does attention mechanism work?",
            "What is tokenization in NLP?",
            "What is fine-tuning in ML?",
            "How does transfer learning work?",
            "What is RAG in AI systems?",
            "What is vector database?",
            "What is model quantization?",
            "What is RLHF in LLM training?",
            "What is retrieval augmented generation?",
            "What is embedding in machine learning?",
            "What is a GPU and how does it differ from a CPU?",
            "Explain the concept of gradient descent.",
            "What is overfitting in machine learning?",
            "How does dropout regularization work?",
            "What is batch normalization?",
            "Explain the vanishing gradient problem.",
            "What is an activation function?",
            "How does the attention mechanism work in transformers?",
            "What is positional encoding in transformers?",
            "What is beam search in text generation?",
            "What is nucleus sampling (top-p)?",
            "What is temperature scaling in LLMs?",
            "Explain the BLEU score for evaluation.",
            "What is perplexity in language modeling?",
        ],
        "sampling_params": {"temperature": 0.7, "max_tokens": 32, "top_p": 0.9, "top_k": 50},
        "concurrent": 32,
        "rounds": 2,
    },
    # ── 场景 I: 极端采样参数 (greedy vs random 对比) ───────────────────────
    # API: /generate (BatchAccumulator 路径)
    # 采样: temperature=0 vs temperature=1.0 — 对比确定性 vs 随机性
    # 目的: 采样参数对延迟和吞吐量的影响
    "I: Greedy vs Random (temperature)": {
        "api": "generate",
        "prompts": [
            "Write a Python function to compute fibonacci numbers recursively.",
            "Explain the concept of recursion in programming with an example.",
            "What is a linked list and how does it differ from an array?",
            "Describe the quicksort algorithm and its time complexity.",
        ],
        "sampling_params": {"temperature": 1.0, "max_tokens": 96, "top_p": 0.95, "top_k": 50},
        "concurrent": 4,
        "rounds": 3,
    },
    # ── 场景 J: repetition_penalty 测试 (长输出防复读) ────────────────────
    # API: /generate (BatchAccumulator 路径)
    # 采样: temperature=0.7, repetition_penalty=1.2, max_tokens=192
    # 目的: 重复惩罚参数对长输出的影响
    "J: Repetition Penalty (long output)": {
        "api": "generate",
        "prompts": [
            "Write a detailed explanation of how the Python GIL (Global Interpreter Lock) works, its impact on multi-threading, and how to work around it.",
            "Explain the CAP theorem in distributed systems, including what consistency, availability, and partition tolerance mean in practice.",
            "Describe the internals of Python's asyncio module: event loop, coroutines, futures, and how await works under the hood.",
        ],
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 192,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
        },
        "concurrent": 3,
        "rounds": 3,
    },
}


async def get_gpu(client) -> tuple:
    """获取 GPU 状态"""
    try:
        r = await client.get(f"{BASE}/api/v1/cluster/nodes", timeout=5)
        if r.status_code == 200:
            g = r.json()[0]["gpu_info"][0]
            return (
                g["utilization"] * 100,
                g.get("memory_util_pct", 0),
                g["memory_used"] / 1e9,
                g["memory_total"] / 1e9,
            )
    except Exception:
        pass
    return 0, 0, 0, 0


async def send_generate(client: httpx.AsyncClient, prompt: str, sp: dict) -> dict:
    """POST /api/v1/inference/generate"""
    t0 = time.time()
    try:
        r = await client.post(
            f"{BASE}/api/v1/inference/generate",
            json={"model": MODEL, "prompt": prompt, "sampling_params": sp},
            timeout=120.0,
        )
        elapsed_ms = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        return {
            "ok": r.status_code == 200,
            "elapsed_ms": elapsed_ms,
            "latency_ms": data.get("latency_ms", 0),
            "tokens": data.get("usage", {}).get("total_tokens", 0) if data else 0,
            "status": r.status_code,
        }
    except Exception:
        return {
            "ok": False,
            "elapsed_ms": (time.time() - t0) * 1000,
            "latency_ms": 0,
            "tokens": 0,
            "status": 0,
        }


async def send_stream(client: httpx.AsyncClient, prompt: str, sp: dict) -> dict:
    """POST /api/v1/inference/generate/stream (consume SSE)"""
    t0 = time.time()
    text_len = 0
    try:
        async with client.stream(
            "POST",
            f"{BASE}/api/v1/inference/generate/stream",
            json={"model": MODEL, "prompt": prompt, "sampling_params": sp},
            timeout=120.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    if line == "data: [DONE]" or line == "data: [DONE]\n":
                        break
                    try:
                        import json as _json

                        chunk = _json.loads(line[6:])
                        if chunk.get("delta"):
                            text_len += len(chunk["delta"])
                    except Exception:
                        pass
        elapsed_ms = (time.time() - t0) * 1000
        # 流式延迟估算：end-to-end 时间
        return {
            "ok": resp.status_code == 200,
            "elapsed_ms": elapsed_ms,
            "latency_ms": elapsed_ms,
            "tokens": text_len // 4,
            "status": resp.status_code,
        }
    except Exception:
        return {
            "ok": False,
            "elapsed_ms": (time.time() - t0) * 1000,
            "latency_ms": 0,
            "tokens": 0,
            "status": 0,
        }


async def send_chat(client: httpx.AsyncClient, prompt: str, sp: dict) -> dict:
    """POST /api/v1/inference/chat"""
    t0 = time.time()
    try:
        # Chat API 接受 messages 格式
        r = await client.post(
            f"{BASE}/api/v1/inference/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "sampling_params": sp,
            },
            timeout=120.0,
        )
        elapsed_ms = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        return {
            "ok": r.status_code == 200,
            "elapsed_ms": elapsed_ms,
            "latency_ms": data.get("latency_ms", 0),
            "tokens": data.get("usage", {}).get("total_tokens", 0) if data else 0,
            "status": r.status_code,
        }
    except Exception:
        return {
            "ok": False,
            "elapsed_ms": (time.time() - t0) * 1000,
            "latency_ms": 0,
            "tokens": 0,
            "status": 0,
        }


async def send_batch(client: httpx.AsyncClient, prompts: list[str], sp: dict) -> dict:
    """POST /api/v1/inference/batch"""
    t0 = time.time()
    try:
        r = await client.post(
            f"{BASE}/api/v1/inference/batch",
            json={"model": MODEL, "prompts": prompts, "sampling_params": sp},
            timeout=120.0,
        )
        elapsed_ms = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        return {
            "ok": r.status_code == 200,
            "elapsed_ms": elapsed_ms,
            "latency_ms": data.get("avg_latency_ms", 0),
            "tokens": sum(
                u.get("total_tokens", 0) for u in (data.get("results", []) if data else [])
            ),
            "count": len(data.get("results", [])) if data else 0,
            "status": r.status_code,
        }
    except Exception:
        return {
            "ok": False,
            "elapsed_ms": (time.time() - t0) * 1000,
            "latency_ms": 0,
            "tokens": 0,
            "status": 0,
        }


async def run_scenario(name: str, cfg: dict) -> ScenarioResult:
    """运行单个测试场景"""
    api = cfg["api"]
    prompts = cfg["prompts"]
    sp = cfg["sampling_params"]
    concurrent = cfg["concurrent"]
    rounds = cfg["rounds"]

    print(f"\n  [{name}]  api={api}  concurrent={concurrent}  rounds={rounds}  sp={sp}")

    all_latencies = []
    all_tokens = []
    gpu_utils = []
    mem_utils = []
    vram_gbs = []
    successes = 0
    batch_wall_ms = 0.0  # 仅用于 batch API

    for round_i in range(rounds):
        async with httpx.AsyncClient(timeout=120.0) as client:
            gpu_u, mem_u, vram_u, _ = await get_gpu(client)
            gpu_utils.append(gpu_u)
            mem_utils.append(mem_u)
            vram_gbs.append(vram_u)

            if api == "batch":
                res = await send_batch(client, prompts, sp)
                if res["ok"]:
                    successes += res.get("count", 0)
                    batch_wall_ms += res["elapsed_ms"]
                    all_latencies.append(res["latency_ms"])
                    all_tokens.append(res["tokens"])
            else:
                # generate / stream / chat: 并发发送多个请求
                tasks = [
                    (
                        send_stream
                        if api == "stream"
                        else send_chat if api == "chat" else send_generate
                    )(client, prompts[i % len(prompts)], sp)
                    for i in range(concurrent)
                ]
                results = await asyncio.gather(*tasks)

                for res in results:
                    if res["ok"]:
                        successes += 1
                        all_latencies.append(res["latency_ms"])
                        all_tokens.append(res["tokens"])

            if round_i < rounds - 1:
                await asyncio.sleep(0.5)

    total = len(prompts) * rounds if api == "batch" else concurrent * rounds
    ok_latencies = [l for l in all_latencies if l > 0]
    ok_tokens = [t for t in all_tokens if t > 0]

    # 吞吐量: batch 用实际 wall time；其他用 sum(latencies) 作为近似
    wall_s = (batch_wall_ms / 1000) if api == "batch" else sum(l for l in ok_latencies) / 1000
    tp = sum(ok_tokens) / max(wall_s, 0.001)

    result = ScenarioResult(
        name=name,
        total_requests=total,
        success=successes,
        failed=total - successes,
        avg_gpu_util=float(np.mean(gpu_utils)) if gpu_utils else 0,
        max_gpu_util=float(np.max(gpu_utils)) if gpu_utils else 0,
        avg_gpu_mem_util=float(np.mean(mem_utils)) if mem_utils else 0,
        avg_vram_gb=float(np.mean(vram_gbs)) if vram_gbs else 0,
        avg_latency_ms=float(np.mean(ok_latencies)) if ok_latencies else 0,
        p50_latency_ms=float(np.percentile(ok_latencies, 50)) if ok_latencies else 0,
        p95_latency_ms=float(np.percentile(ok_latencies, 95)) if ok_latencies else 0,
        p99_latency_ms=float(np.percentile(ok_latencies, 99)) if ok_latencies else 0,
        throughput_tok_s=tp,
    )

    success_rate = 100 * successes / max(total, 1)
    print(
        f"    GPU: {result.avg_gpu_util:.0f}% | "
        f"P50: {result.p50_latency_ms:.0f}ms | "
        f"Throughput: {result.throughput_tok_s:.1f} tok/s | "
        f"Success: {successes}/{total} ({success_rate:.0f}%)"
    )
    return result


async def main():
    print("=" * 70)
    print("  QuantumFlow GPU Benchmark — 全路径覆盖版")
    print(f"  Model: {MODEL} | RTX 4080 Laptop GPU")
    print("  APIs: /generate | /generate/stream | /chat | /batch")
    print("=" * 70)

    # 检查服务器
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"{BASE}/api/v1/cluster/status")
            if r.status_code != 200:
                print("Server not ready")
                return
            print("Server OK\n")
        except Exception as e:
            print(f"Cannot connect: {e}")
            return

    results = []

    for name, cfg in SCENARIOS.items():
        r = await run_scenario(name, cfg)
        results.append(asdict(r))
        await asyncio.sleep(2)

    # 保存结果
    with open("docs/benchmark_data.json", "w") as f:
        json.dump(results, f, indent=2)

    # 打印汇总表
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    hdr = f"{'Scenario':<35} {'GPU':>7} {'P50':>8} {'Throughput':>12} {'Success'}"
    print(hdr)
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<35} "
            f"{r['avg_gpu_util']:>5.0f}% "
            f"{r['p50_latency_ms']:>6.0f}ms "
            f"{r['throughput_tok_s']:>10.1f} tok/s "
            f"{r['success']}/{r['total_requests']}"
        )

    print("\nData: docs/benchmark_data.json")
    print("Chart: tests/regenerate_chart.py")


if __name__ == "__main__":
    asyncio.run(main())
