#!/usr/bin/env python3
"""
GPU性能基准测试 — 对比动态批处理前后的性能差异
生成可视化图表并保存到 docs/benchmarks.png
"""

import asyncio
import httpx
import time
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# 禁用代理
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE_URL = "http://localhost:8000"
MODEL = "Qwen2.5-1.5B"

PROMPTS = [
    "解释量子力学中的不确定性原理，包括海森堡的原始表述和现代数学表述，以及它在双缝实验中的具体体现。",
    "用Python实现一个完整的二叉搜索树，包括插入、删除、查找和遍历操作，要求代码包含类型注解和详细注释。",
    "机器学习中的反向传播算法是深度学习的基础。请详细推导梯度下降的数学过程，解释链式法则在其中的作用。",
    "请用中文写一首关于人工智能与人类关系的现代诗，要求包含科技元素和人文思考，韵律优美意境深远。",
    "Transformer架构自2017年提出以来彻底改变了NLP领域。请详细解释Self-Attention的工作机制。",
    "GPU的显存带宽对深度学习训练速度有决定性影响。请解释HBM2、HBM3显存的工作原理。",
    "在大模型推理中，KV Cache是加速生成的核心技术。请说明其工作原理以及PagedAttention的优势。",
    "Python的asyncio模块如何实现协程？请解释事件循环、Future、Task的概念以及await语法。",
]


class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.gpu_util_samples: List[float] = []
        self.gpu_mem_util_samples: List[float] = []
        self.vram_used_gb: List[float] = []
        self.latencies_ms: List[float] = []
        self.throughput_tokens_per_sec: List[float] = []
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0

    def add_gpu_sample(self, gpu_util: float, mem_util: float, vram_gb: float):
        self.gpu_util_samples.append(gpu_util)
        self.gpu_mem_util_samples.append(mem_util)
        self.vram_used_gb.append(vram_gb)

    def add_request(self, latency_ms: float, tokens: int, wall_ms: float):
        self.total_requests += 1
        if latency_ms > 0:
            self.success_requests += 1
            self.latencies_ms.append(latency_ms)
            if wall_ms > 0 and latency_ms > 0:
                self.throughput_tokens_per_sec.append(tokens / (wall_ms / 1000))
        else:
            self.failed_requests += 1

    def summary(self) -> dict:
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "success": self.success_requests,
            "failed": self.failed_requests,
            "avg_gpu_util": np.mean(self.gpu_util_samples) if self.gpu_util_samples else 0,
            "max_gpu_util": np.max(self.gpu_util_samples) if self.gpu_util_samples else 0,
            "avg_gpu_mem_util": np.mean(self.gpu_mem_util_samples) if self.gpu_mem_util_samples else 0,
            "avg_vram_gb": np.mean(self.vram_used_gb) if self.vram_used_gb else 0,
            "avg_latency_ms": np.mean(self.latencies_ms) if self.latencies_ms else 0,
            "p50_latency_ms": np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0,
            "p95_latency_ms": np.percentile(self.latencies_ms, 95) if self.latencies_ms else 0,
            "p99_latency_ms": np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0,
            "throughput_tok_s": np.mean(self.throughput_tokens_per_sec) if self.throughput_tokens_per_sec else 0,
        }


async def send_request(client: httpx.AsyncClient, prompt: str, max_tokens: int = 256) -> dict:
    t0 = time.time()
    try:
        resp = await client.post(
            f"{BASE_URL}/api/v1/inference/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "sampling_params": {"temperature": 0.7, "max_tokens": max_tokens, "top_p": 0.9},
            },
            timeout=120.0,
        )
        wall_ms = (time.time() - t0) * 1000
        result = resp.json() if resp.status_code == 200 else None
        latency_ms = result.get("latency_ms", 0) if result else 0
        tokens = result.get("usage", {}).get("total_tokens", 0) if result else 0
        return {"success": resp.status_code == 200, "latency_ms": latency_ms, "tokens": tokens, "wall_ms": wall_ms}
    except Exception as e:
        wall_ms = (time.time() - t0) * 1000
        return {"success": False, "latency_ms": 0, "tokens": 0, "wall_ms": wall_ms, "error": str(e)}


async def get_gpu_sample(client: httpx.AsyncClient) -> dict:
    try:
        resp = await client.get(f"{BASE_URL}/api/v1/cluster/nodes", timeout=5.0)
        if resp.status_code == 200:
            nodes = resp.json()
            if nodes and nodes[0].get("gpu_info"):
                g = nodes[0]["gpu_info"][0]
                return {
                    "gpu_util": g["utilization"] * 100,
                    "mem_util": g.get("memory_util_pct", 0),
                    "vram_gb": g["memory_used"] / 1e9,
                }
    except:
        pass
    return {"gpu_util": 0, "mem_util": 0, "vram_gb": 0}


async def get_scheduler_status(client: httpx.AsyncClient) -> dict:
    try:
        resp = await client.get(f"{BASE_URL}/api/v1/scheduler/status", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return {}


async def run_scenario(
    name: str,
    concurrent: int,
    rounds: int,
    delay_between_rounds: float,
    max_tokens: int,
    monitor_interval: float,
) -> BenchmarkResult:
    """运行一个测试场景"""
    print(f"\n  Scenario: {name}")
    print(f"  Concurrency: {concurrent} | Rounds: {rounds} | max_tokens: {max_tokens}")

    result = BenchmarkResult(name)
    monitor_task = None

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Ensure model is loaded
        status = await get_scheduler_status(client)
        vram = status.get("vram", {})
        loaded = vram.get("loaded_count", 0)
        print(f"  Loaded models: {loaded}")

        # Start GPU monitoring background task
        monitor_samples = []

        async def monitor_loop():
            async with httpx.AsyncClient(timeout=5.0) as mc:
                for _ in range(200):  # 最多采样200次
                    await asyncio.sleep(monitor_interval)
                    sample = await get_gpu_sample(mc)
                    if sample["gpu_util"] > 0:
                        monitor_samples.append(sample)

        monitor_task = asyncio.create_task(monitor_loop())

        # 等待监控启动
        await asyncio.sleep(0.5)

        for r in range(rounds):
            # Send concurrent requests
            tasks = [send_request(client, PROMPTS[i % len(PROMPTS)], max_tokens) for i in range(concurrent)]
            results = await asyncio.gather(*tasks)

            for res in results:
                result.add_request(res["latency_ms"], res["tokens"], res["wall_ms"])

            succ = sum(1 for r in results if r["success"])
            print(f"  Round {r+1}/{rounds}: {succ}/{concurrent} succeeded")

            if r < rounds - 1:
                await asyncio.sleep(delay_between_rounds)

        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Collect GPU samples
        for s in monitor_samples:
            result.add_gpu_sample(s["gpu_util"], s["mem_util"], s["vram_gb"])

        # Print summary
        s = result.summary()
        print(f"\n  Results:")
        print(f"    GPU Util:        avg={s['avg_gpu_util']:.1f}%  max={s['max_gpu_util']:.1f}%")
        print(f"    GPU Mem BW:      avg={s['avg_gpu_mem_util']:.1f}%")
        print(f"    Latency:         avg={s['avg_latency_ms']:.0f}ms  p50={s['p50_latency_ms']:.0f}ms  p95={s['p95_latency_ms']:.0f}ms")
        print(f"    Throughput:      {s['throughput_tok_s']:.0f} tokens/s")

    return result


async def run_benchmarks() -> List[BenchmarkResult]:
    """Run all benchmark scenarios."""
    results = []

    # Scenario 1: Single-request baseline
    results.append(await run_scenario(
        name="Baseline (single)",
        concurrent=1,
        rounds=5,
        delay_between_rounds=2.0,
        max_tokens=256,
        monitor_interval=0.5,
    ))

    await asyncio.sleep(3)

    # Scenario 2: Low concurrency (8 concurrent requests)
    results.append(await run_scenario(
        name="Low (8 concurrent)",
        concurrent=8,
        rounds=5,
        delay_between_rounds=3.0,
        max_tokens=256,
        monitor_interval=0.5,
    ))

    await asyncio.sleep(3)

    # Scenario 3: High concurrency (24 concurrent requests)
    results.append(await run_scenario(
        name="High (24 concurrent)",
        concurrent=24,
        rounds=5,
        delay_between_rounds=3.0,
        max_tokens=256,
        monitor_interval=0.3,
    ))

    return results


def generate_charts(results: List[BenchmarkResult], output_path: str):
    """Generate performance comparison charts (Nature/Science journal style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    names = [r.name for r in results]
    summaries = [r.summary() for r in results]

    # Nature / Science journal color palette
    C = {
        "blue":   "#3B4992",
        "orange": "#EE7733",
        "red":    "#CC0000",
        "green":  "#009988",
        "purple": "#7B3F00",
        "gray":   "#BBBBBB",
        "bg":     "#FFFFFF",
        "grid":   "#DDDDDD",
        "text":   "#222222",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["bg"],
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.88, bottom=0.10, hspace=0.50, wspace=0.32)

    x = np.arange(len(names))
    xw = 0.5

    # ── A  GPU Compute Utilization ──
    ax = axes[0, 0]
    avg_u = [s["avg_gpu_util"] for s in summaries]
    max_u = [s["max_gpu_util"] for s in summaries]
    bars = ax.bar(x, avg_u, width=xw, color=C["blue"], alpha=0.85, zorder=3)
    ax.errorbar(x, avg_u, yerr=[max(0, m - a) for m, a in zip(max_u, avg_u)],
                fmt="none", color=C["text"], capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
    ax.set_ylabel("GPU Compute Utilization (%)")
    ax.set_title("A   GPU Compute Utilization", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, max(max_u) * 1.3 if max_u else 130)
    ax.axhline(y=80, color=C["orange"], linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
    ax.text(x[-1] + 0.05, 80, "80% target", color=C["orange"], fontsize=7, va="center")
    for bar, val in zip(bars, avg_u):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── B  P50 / P95 / P99 Latency ──
    ax = axes[0, 1]
    p50 = [s["p50_latency_ms"] for s in summaries]
    p95 = [s["p95_latency_ms"] for s in summaries]
    p99 = [s["p99_latency_ms"] for s in summaries]
    w = xw / 3.2
    b1 = ax.bar(x - w, p50, width=w, label="P50", color=C["blue"], alpha=0.85, zorder=3)
    b2 = ax.bar(x,     p95, width=w, label="P95", color=C["orange"], alpha=0.85, zorder=3)
    b3 = ax.bar(x + w, p99, width=w, label="P99", color=C["red"], alpha=0.85, zorder=3)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("B   Inference Latency Distribution", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.legend(loc="upper left", framealpha=0.4)
    max_p = max(max(p50), max(p95), max(p99))
    for bars_g, vals in [(b1, p50), (b2, p95), (b3, p99)]:
        for bar, val in zip(bars_g, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_p * 0.01,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)

    # ── C  Throughput ──
    ax = axes[0, 2]
    tp = [s["throughput_tok_s"] for s in summaries]
    colors_tp = [C["blue"], C["orange"], C["red"]]
    bars = ax.bar(x, tp, width=xw, color=colors_tp, alpha=0.85, zorder=3)
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("C   Generation Throughput", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, max(tp) * 1.25 if tp else 100)
    for bar, val in zip(bars, tp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tp) * 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── D  GPU Memory Bandwidth Utilization ──
    ax = axes[1, 0]
    mu = [s["avg_gpu_mem_util"] for s in summaries]
    bars = ax.bar(x, mu, width=xw, color=C["green"], alpha=0.85, zorder=3)
    ax.set_ylabel("Memory Bandwidth Util. (%)")
    ax.set_title("D   GPU Memory Bandwidth Utilization", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, mu):
        lbl = f"{val:.1f}%" if val > 0 else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                lbl, ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── E  VRAM Usage ──
    ax = axes[1, 1]
    vram = [s["avg_vram_gb"] for s in summaries]
    bars = ax.bar(x, vram, width=xw, color=C["purple"], alpha=0.85, zorder=3)
    ax.set_ylabel("VRAM Usage (GB)")
    ax.set_title("E   VRAM Footprint", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, max(vram) * 1.3 if vram else 20)
    for bar, val in zip(bars, vram):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f} GB", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── F  Summary Table ──
    ax = axes[1, 2]
    ax.axis("off")

    col_labels = ["Metric", "Baseline\n(single)", "Low\n(8 req)", "High + Batching\n(24 req)"]
    row_data = [
        ["GPU Util (avg)", f"{summaries[0]['avg_gpu_util']:.1f}%", f"{summaries[1]['avg_gpu_util']:.1f}%", f"{summaries[2]['avg_gpu_util']:.1f}%"],
        ["GPU Util (max)", f"{summaries[0]['max_gpu_util']:.1f}%", f"{summaries[1]['max_gpu_util']:.1f}%", f"{summaries[2]['max_gpu_util']:.1f}%"],
        ["P50 Latency", f"{summaries[0]['p50_latency_ms']:.0f} ms", f"{summaries[1]['p50_latency_ms']:.0f} ms", f"{summaries[2]['p50_latency_ms']:.0f} ms"],
        ["P95 Latency", f"{summaries[0]['p95_latency_ms']:.0f} ms", f"{summaries[1]['p95_latency_ms']:.0f} ms", f"{summaries[2]['p95_latency_ms']:.0f} ms"],
        ["Throughput", f"{summaries[0]['throughput_tok_s']:.1f} tok/s", f"{summaries[1]['throughput_tok_s']:.1f} tok/s", f"{summaries[2]['throughput_tok_s']:.1f} tok/s"],
        ["Success Rate", f"{100*summaries[0]['success']/max(summaries[0]['total_requests'],1):.0f}%",
                          f"{100*summaries[1]['success']/max(summaries[1]['total_requests'],1):.0f}%",
                          f"{100*summaries[2]['success']/max(summaries[2]['total_requests'],1):.0f}%"],
    ]

    table = ax.table(cellText=row_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.15, 2.1)

    for j in range(4):
        table[(0, j)].set_facecolor(C["blue"])
        table[(0, j)].set_text_props(color="white", fontweight="bold")
        table[(0, j)].set_height(0.18)

    row_colors = [C["bg"], "#F5F5F5"]
    for i in range(1, 7):
        for j in range(4):
            table[(i, j)].set_facecolor(row_colors[(i - 1) % 2])
            table[(i, j)].set_text_props(color=C["text"])
            table[(i, j)].set_height(0.14)
            if j == 0:
                table[(i, j)].set_text_props(fontweight="bold")

    ax.set_title("F   Performance Summary", fontweight="bold", pad=6)

    fig.suptitle(
        "QuantumFlow GPU Performance Benchmark\n"
        "Dynamic Batching Optimization — Qwen2.5-1.5B on NVIDIA RTX 4080 Laptop GPU",
        fontsize=13, fontweight="bold", y=0.97, color=C["text"],
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"\nChart saved: {output_path}")

    # Nature/Science 期刊配色 — 沉稳学术感
    C = {
        "blue":   "#3B4992",   # 深蓝 (主色)
        "orange": "#EE7733",   # 橙色
        "red":    "#CC0000",   # 深红
        "green":  "#009988",   # 青绿
        "purple": "#7B3F00",   # 棕色
        "gray":   "#BBBBBB",   # 灰
        "bg":     "#FFFFFF",   # 白色背景
        "grid":   "#DDDDDD",   # 浅灰网格
        "text":   "#222222",   # 深灰文字
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["bg"],
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "ytick.minor.width": 0.4,
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.10, hspace=0.45, wspace=0.30)

    x = np.arange(len(names))
    xw = 0.5  # bar width

    # ── ① GPU 计算利用率 ──
    ax = axes[0, 0]
    avg_u = [s["avg_gpu_util"] for s in summaries]
    max_u = [s["max_gpu_util"] for s in summaries]
    bars = ax.bar(x, avg_u, width=xw, color=C["blue"], alpha=0.85, zorder=3)
    # error bar: max - avg
    ax.errorbar(x, avg_u, yerr=[max(0, m - a) for m, a in zip(max_u, avg_u)],
                fmt="none", color=C["text"], capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
    ax.set_ylabel("GPU Compute Utilization (%)")
    ax.set_title("A   GPU Compute Utilization", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, max(max_u) * 1.25 if max_u else 120)
    ax.axhline(y=80, color=C["orange"], linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
    ax.text(x[-1] + 0.1, 80, "80% target", color=C["orange"], fontsize=7, va="center")
    for bar, val in zip(bars, avg_u):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=C["text"])

    # ── ② P50 / P95 / P99 延迟 ──
    ax = axes[0, 1]
    p50 = [s["p50_latency_ms"] for s in summaries]
    p95 = [s["p95_latency_ms"] for s in summaries]
    p99 = [s["p99_latency_ms"] for s in summaries]
    w = xw / 3.2
    b1 = ax.bar(x - w, p50, width=w, label="P50", color=C["blue"], alpha=0.85, zorder=3)
    b2 = ax.bar(x,     p95, width=w, label="P95", color=C["orange"], alpha=0.85, zorder=3)
    b3 = ax.bar(x + w, p99, width=w, label="P99", color=C["red"], alpha=0.85, zorder=3)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("B   Inference Latency (ms)", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.legend(loc="upper left", framealpha=0.4)
    for bars_group in [b1, b2, b3]:
        for bar in bars_group:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(p95) * 0.01,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=6.5, color=C["text"])

    # ── ③ 吞吐量 (Tokens/s) ──
    ax = axes[0, 2]
    tp = [s["throughput_tok_s"] for s in summaries]
    bars = ax.bar(x, tp, width=xw, color=[C["blue"], C["orange"], C["red"]], alpha=0.85, zorder=3)
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("C   Generation Throughput", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, max(tp) * 1.2 if tp else 100)
    for bar, val in zip(bars, tp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tp) * 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=C["text"])

    # ── ④ GPU 显存带宽利用率 ──
    ax = axes[1, 0]
    mu = [s["avg_gpu_mem_util"] for s in summaries]
    bars = ax.bar(x, mu, width=xw, color=C["green"], alpha=0.85, zorder=3)
    ax.set_ylabel("Memory Bandwidth Utilization (%)")
    ax.set_title("D   GPU Memory Bandwidth Utilization", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, mu):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=C["text"])

    # ── ⑤ VRAM 占用 & 请求成功率 ──
    ax = axes[1, 1]
    vram = [s["avg_vram_gb"] for s in summaries]
    bars = ax.bar(x, vram, width=xw, color=C["purple"], alpha=0.85, zorder=3)
    ax.set_ylabel("VRAM Usage (GB)")
    ax.set_title("E   VRAM Footprint", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, max(vram) * 1.25 if vram else 20)
    for bar, val in zip(bars, vram):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f} GB", ha="center", va="bottom", fontsize=8, fontweight="bold", color=C["text"])

    # ── ⑥ 综合性能对比表格 ──
    ax = axes[1, 2]
    ax.axis("off")

    col_labels = ["Metric", "Baseline\n(single)", "Low\n(8 req)", "High + Batching\n(24 req)"]
    row_data = [
        ["GPU Util (avg)", f"{summaries[0]['avg_gpu_util']:.1f}%", f"{summaries[1]['avg_gpu_util']:.1f}%", f"{summaries[2]['avg_gpu_util']:.1f}%"],
        ["GPU Mem BW Util", f"{summaries[0]['avg_gpu_mem_util']:.1f}%", f"{summaries[1]['avg_gpu_mem_util']:.1f}%", f"{summaries[2]['avg_gpu_mem_util']:.1f}%"],
        ["P50 Latency", f"{summaries[0]['p50_latency_ms']:.0f} ms", f"{summaries[1]['p50_latency_ms']:.0f} ms", f"{summaries[2]['p50_latency_ms']:.0f} ms"],
        ["P95 Latency", f"{summaries[0]['p95_latency_ms']:.0f} ms", f"{summaries[1]['p95_latency_ms']:.0f} ms", f"{summaries[2]['p95_latency_ms']:.0f} ms"],
        ["Throughput", f"{summaries[0]['throughput_tok_s']:.1f} tok/s", f"{summaries[1]['throughput_tok_s']:.1f} tok/s", f"{summaries[2]['throughput_tok_s']:.1f} tok/s"],
        ["Success Rate", f"{100*summaries[0]['success']/max(summaries[0]['total_requests'],1):.0f}%", f"{100*summaries[1]['success']/max(summaries[1]['total_requests'],1):.0f}%", f"{100*summaries[2]['success']/max(summaries[2]['total_requests'],1):.0f}%"],
    ]

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.15, 2.1)

    # 表头
    for j in range(4):
        table[(0, j)].set_facecolor(C["blue"])
        table[(0, j)].set_text_props(color="white", fontweight="bold")
        table[(0, j)].set_height(0.18)

    # 数据行
    row_colors = [C["bg"], "#F5F5F5"]
    for i in range(1, 7):
        for j in range(4):
            table[(i, j)].set_facecolor(row_colors[(i - 1) % 2])
            table[(i, j)].set_text_props(color=C["text"])
            table[(i, j)].set_height(0.14)
            if j == 0:
                table[(i, j)].set_text_props(fontweight="bold")

    ax.set_title("F   Performance Summary", fontweight="bold", pad=6)

    # 总标题
    fig.suptitle(
        "QuantumFlow GPU Performance Benchmark\n"
        "Dynamic Batching Optimization — Qwen2.5-1.5B on NVIDIA RTX 4080 Laptop GPU",
        fontsize=13, fontweight="bold", y=0.98, color=C["text"],
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    print(f"\n✅ 图表已保存: {output_path}")

    # 打印总结报告
    print("\n" + "="*60)
    print("  基准测试总结")
    print("="*60)
    print(f"\n{'指标':<20} {'单请求基线':<15} {'低并发(8)':<15} {'高并发(24)+批处理':<15}")
    print("-"*70)
    print(f"{'GPU计算利用率':<20} {summaries[0]['avg_gpu_util']:.1f}%{'':<9} {summaries[1]['avg_gpu_util']:.1f}%{'':<9} {summaries[2]['avg_gpu_util']:.1f}%")
    print(f"{'GPU显存带宽':<20} {summaries[0]['avg_gpu_mem_util']:.1f}%{'':<9} {summaries[1]['avg_gpu_mem_util']:.1f}%{'':<9} {summaries[2]['avg_gpu_mem_util']:.1f}%")
    print(f"{'P50延迟':<20} {summaries[0]['p50_latency_ms']:.0f}ms{'':<9} {summaries[1]['p50_latency_ms']:.0f}ms{'':<9} {summaries[2]['p50_latency_ms']:.0f}ms")
    print(f"{'P95延迟':<20} {summaries[0]['p95_latency_ms']:.0f}ms{'':<9} {summaries[1]['p95_latency_ms']:.0f}ms{'':<9} {summaries[2]['p95_latency_ms']:.0f}ms")
    print(f"{'吞吐量(tokens/s)':<20} {summaries[0]['throughput_tok_s']:.0f}{'':<14} {summaries[1]['throughput_tok_s']:.0f}{'':<14} {summaries[2]['throughput_tok_s']:.0f}")
    print(f"{'成功率':<20} {summaries[0]['success']/max(summaries[0]['total_requests'],1)*100:.0f}%{'':<9} {summaries[1]['success']/max(summaries[1]['total_requests'],1)*100:.0f}%{'':<9} {summaries[2]['success']/max(summaries[2]['total_requests'],1)*100:.0f}%")

    # 计算提升倍数
    if summaries[0]['avg_gpu_util'] > 0:
        util_improvement = summaries[2]['avg_gpu_util'] / summaries[0]['avg_gpu_util']
        print(f"\n  📈 GPU利用率提升: {util_improvement:.1f}x ({summaries[0]['avg_gpu_util']:.0f}% → {summaries[2]['avg_gpu_util']:.0f}%)")
    if summaries[0]['throughput_tok_s'] > 0:
        tp_improvement = summaries[2]['throughput_tok_s'] / summaries[0]['throughput_tok_s']
        print(f"  📈 吞吐量提升: {tp_improvement:.1f}x ({summaries[0]['throughput_tok_s']:.0f} → {summaries[2]['throughput_tok_s']:.0f} tok/s)")

    return summaries


async def main():
    print("=" * 60)
    print("  QuantumFlow GPU Performance Benchmark")
    print("  Dynamic Batching Optimization Analysis")
    print("=" * 60)

    # Check server
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{BASE_URL}/api/v1/cluster/status")
            if resp.status_code != 200:
                print("Server unavailable")
                return
            print("Server OK")
        except Exception as e:
            print(f"Connection failed: {e}")
            return

    # 运行基准测试
    results = await run_benchmarks()

    # 生成图表
    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmarks.png")

    summaries = generate_charts(results, output_path)

    # 保存JSON数据
    json_path = os.path.join(output_dir, "benchmark_data.json")
    with open(json_path, "w") as f:
        json.dump([s for r in results for s in [r.summary()]], f, indent=2)
    print(f"📊 JSON数据已保存: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
