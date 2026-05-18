#!/usr/bin/env python3
"""Regenerate benchmark charts: 2×3 fixed layout, 6 scenarios max."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("docs/benchmark_data.json") as f:
    data = json.load(f)

summaries = data

if len(summaries) > 6:
    indices = [0, 1, 2, 3, 4, 7]  # A, B, C, D, E, H
    summaries = [summaries[i] for i in indices if i < len(summaries)]

enames = [s["name"].replace(" ", "\n", 1) for s in summaries]
# 简短标签用于分组柱图
short_labels = [s["name"].split(":")[0].strip() for s in summaries]

C = {
    "blue":   "#3B4992",
    "orange": "#EE7733",
    "red":    "#CC0000",
    "green":  "#009988",
    "purple": "#7B3F00",
    "bg":     "#FFFFFF",
    "grid":   "#DDDDDD",
    "text":   "#222222",
}

SCENE_COLORS = [C["blue"], C["orange"], C["red"], C["green"], C["purple"], C["blue"]]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 9,
    "legend.fontsize": 7.5,
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

n_scenes = len(summaries)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.14, hspace=0.60, wspace=0.30)

x = np.arange(n_scenes)
xw = 0.5

# ── A: GPU Compute Utilization ──────────────────────────────────────
ax = axes[0, 0]
avg_u = [s["avg_gpu_util"] for s in summaries]
max_u = [s["max_gpu_util"] for s in summaries]
bars = ax.bar(x, avg_u, width=xw, color=SCENE_COLORS[:n_scenes], alpha=0.85, zorder=3)
ax.errorbar(x, avg_u, yerr=[max(0, m - a) for m, a in zip(max_u, avg_u)],
            fmt="none", color=C["text"], capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
ax.set_ylabel("GPU Compute (%)")
ax.set_title("A   GPU Compute Utilization", fontweight="bold", pad=6)
ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C", "D", "E", "H"][:n_scenes], fontsize=8)
ax.set_ylim(0, max(max_u) * 1.3 if max_u else 130)
ax.axhline(y=80, color=C["orange"], linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
for bar, val in zip(bars, avg_u):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# ── B: Latency (P50 / P95 / P99) ──────────────────────────────────
ax = axes[0, 1]
p50 = [s["p50_latency_ms"] for s in summaries]
p95 = [s["p95_latency_ms"] for s in summaries]
p99 = [s["p99_latency_ms"] for s in summaries]
w = 0.5 / 4.0
b1 = ax.bar(x - w, p50, width=w, label="P50", color=C["blue"],   alpha=0.85, zorder=3)
b2 = ax.bar(x,      p95, width=w, label="P95", color=C["orange"], alpha=0.85, zorder=3)
b3 = ax.bar(x + w, p99, width=w, label="P99", color=C["red"],   alpha=0.85, zorder=3)
ax.set_ylabel("Latency (ms)")
ax.set_title("B   Inference Latency", fontweight="bold", pad=6)
ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C", "D", "E", "H"][:n_scenes], fontsize=8)
ax.legend(loc="upper left", framealpha=0.4, fontsize=7)
max_p = max(max(p50), max(p95), max(p99))
for bar, val in zip(b1, p50):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_p * 0.01,
            f"{val:.0f}", ha="center", va="bottom", fontsize=6)

# ── C: Throughput ──────────────────────────────────────────────────
ax = axes[0, 2]
tp = [s["throughput_tok_s"] for s in summaries]
bars = ax.bar(x, tp, width=xw, color=SCENE_COLORS[:n_scenes], alpha=0.85, zorder=3)
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("C   Generation Throughput", fontweight="bold", pad=6)
ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C", "D", "E", "H"][:n_scenes], fontsize=8)
mx_tp = max(tp) if tp else 100
ax.set_ylim(0, mx_tp * 1.3)
for bar, val in zip(bars, tp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + mx_tp * 0.01,
            f"{val:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# ── D: GPU Memory Bandwidth (全0显示N/A) ─────────────────────────
ax = axes[1, 0]
mu = [s["avg_gpu_mem_util"] for s in summaries]
all_zero = all(v == 0 for v in mu)
if all_zero:
    ax.text(0.5, 0.5, "N/A\n(GPU Monitor\nTorch fallback)", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=C["text"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#DDDDDD"))
    ax.set_ylim(0, 1)
else:
    bars = ax.bar(x, mu, width=xw, color=C["green"], alpha=0.85, zorder=3)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, mu):
        lbl = f"{val:.1f}%" if val > 0 else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                lbl, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_ylabel("Memory BW (%)")
ax.set_title("D   GPU Memory Bandwidth", fontweight="bold", pad=6)
ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C", "D", "E", "H"][:n_scenes], fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── E: VRAM Usage ──────────────────────────────────────────────────
ax = axes[1, 1]
vram = [s["avg_vram_gb"] for s in summaries]
bars = ax.bar(x, vram, width=xw, color=C["purple"], alpha=0.85, zorder=3)
ax.set_ylabel("VRAM (GB)")
ax.set_title("E   VRAM Footprint", fontweight="bold", pad=6)
ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C", "D", "E", "H"][:n_scenes], fontsize=8)
mx_v = max(vram) if vram else 20
ax.set_ylim(0, mx_v * 1.3)
for bar, val in zip(bars, vram):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + mx_v * 0.01,
            f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# ── F: Success Rate (替换 table) ───────────────────────────────────
ax = axes[1, 2]
rates = [100 * s["success"] / max(s["total_requests"], 1) for s in summaries]
colors_sr = [C["green"] if r == 100 else C["orange"] for r in rates]
bars = ax.bar(x, rates, width=xw, color=colors_sr, alpha=0.85, zorder=3)
ax.set_ylabel("Success Rate (%)")
ax.set_title("F   Request Success Rate", fontweight="bold", pad=6)
ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C", "D", "E", "H"][:n_scenes], fontsize=8)
ax.set_ylim(0, 115)
for bar, val, s in zip(bars, rates, summaries):
    label = f"100%" if val == 100 else f"{val:.0f}%"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            label, ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(y=100, color=C["green"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

fig.suptitle(
    "QuantumFlow GPU Performance Benchmark\n"
    "HuggingFace + BatchAccumulator — Qwen2.5-1.5B on NVIDIA RTX 4080 Laptop GPU",
    fontsize=12, fontweight="bold", y=0.97, color=C["text"],
)

output_path = "docs/benchmarks.png"
plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"Chart saved: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.0f} KB")
