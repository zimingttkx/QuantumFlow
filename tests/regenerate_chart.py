#!/usr/bin/env python3
"""Regenerate benchmark charts from saved JSON data."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("docs/benchmark_data.json") as f:
    data = json.load(f)

summaries = data
enames = ["Baseline\n(single)", "Low\n(8 concurrent)", "High\n(24 concurrent)"]

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

x = np.arange(len(enames))
xw = 0.5

# A: GPU Compute Utilization
ax = axes[0, 0]
avg_u = [s["avg_gpu_util"] for s in summaries]
max_u = [s["max_gpu_util"] for s in summaries]
bars = ax.bar(x, avg_u, width=xw, color=C["blue"], alpha=0.85, zorder=3)
ax.errorbar(x, avg_u, yerr=[max(0, m - a) for m, a in zip(max_u, avg_u)],
            fmt="none", color=C["text"], capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
ax.set_ylabel("GPU Compute Utilization (%)")
ax.set_title("A   GPU Compute Utilization", fontweight="bold", pad=8)
ax.set_xticks(x)
ax.set_xticklabels(enames, fontsize=8)
ax.set_ylim(0, max(max_u) * 1.3 if max_u else 130)
ax.axhline(y=80, color=C["orange"], linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
ax.text(x[-1] + 0.05, 80, "80% target", color=C["orange"], fontsize=7, va="center")
for bar, val in zip(bars, avg_u):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

# B: P50/P95/P99 Latency
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
ax.set_xticklabels(enames, fontsize=8)
ax.legend(loc="upper left", framealpha=0.4)
max_p = max(max(p50), max(p95), max(p99))
for bars_g, vals in [(b1, p50), (b2, p95), (b3, p99)]:
    for bar, val in zip(bars_g, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_p * 0.01,
                f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)

# C: Throughput
ax = axes[0, 2]
tp = [s["throughput_tok_s"] for s in summaries]
colors_tp = [C["blue"], C["orange"], C["red"]]
bars = ax.bar(x, tp, width=xw, color=colors_tp, alpha=0.85, zorder=3)
ax.set_ylabel("Throughput (tokens/s)")
ax.set_title("C   Generation Throughput", fontweight="bold", pad=8)
ax.set_xticks(x)
ax.set_xticklabels(enames, fontsize=8)
ax.set_ylim(0, max(tp) * 1.25 if tp else 100)
for bar, val in zip(bars, tp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tp) * 0.01,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# D: GPU Memory Bandwidth Utilization
ax = axes[1, 0]
mu = [s["avg_gpu_mem_util"] for s in summaries]
bars = ax.bar(x, mu, width=xw, color=C["green"], alpha=0.85, zorder=3)
ax.set_ylabel("Memory Bandwidth Util. (%)")
ax.set_title("D   GPU Memory Bandwidth Utilization", fontweight="bold", pad=8)
ax.set_xticks(x)
ax.set_xticklabels(enames, fontsize=8)
ax.set_ylim(0, 110)
for bar, val in zip(bars, mu):
    lbl = f"{val:.1f}%" if val > 0 else "N/A"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            lbl, ha="center", va="bottom", fontsize=8, fontweight="bold")

# E: VRAM Usage
ax = axes[1, 1]
vram = [s["avg_vram_gb"] for s in summaries]
bars = ax.bar(x, vram, width=xw, color=C["purple"], alpha=0.85, zorder=3)
ax.set_ylabel("VRAM Usage (GB)")
ax.set_title("E   VRAM Footprint", fontweight="bold", pad=8)
ax.set_xticks(x)
ax.set_xticklabels(enames, fontsize=8)
ax.set_ylim(0, max(vram) * 1.3 if vram else 20)
for bar, val in zip(bars, vram):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.1f} GB", ha="center", va="bottom", fontsize=8, fontweight="bold")

# F: Summary Table
ax = axes[1, 2]
ax.axis("off")

col_labels = ["Metric", "Baseline\n(single)", "Low\n(8 req)", "High\n(24 req)"]
row_data = [
    ["GPU Util (avg)",
     f"{summaries[0]['avg_gpu_util']:.1f}%",
     f"{summaries[1]['avg_gpu_util']:.1f}%",
     f"{summaries[2]['avg_gpu_util']:.1f}%"],
    ["GPU Util (max)",
     f"{summaries[0]['max_gpu_util']:.1f}%",
     f"{summaries[1]['max_gpu_util']:.1f}%",
     f"{summaries[2]['max_gpu_util']:.1f}%"],
    ["P50 Latency",
     f"{summaries[0]['p50_latency_ms']:.0f} ms",
     f"{summaries[1]['p50_latency_ms']:.0f} ms",
     f"{summaries[2]['p50_latency_ms']:.0f} ms"],
    ["P95 Latency",
     f"{summaries[0]['p95_latency_ms']:.0f} ms",
     f"{summaries[1]['p95_latency_ms']:.0f} ms",
     f"{summaries[2]['p95_latency_ms']:.0f} ms"],
    ["Throughput",
     f"{summaries[0]['throughput_tok_s']:.1f} tok/s",
     f"{summaries[1]['throughput_tok_s']:.1f} tok/s",
     f"{summaries[2]['throughput_tok_s']:.1f} tok/s"],
    ["Success Rate",
     f"{100 * summaries[0]['success'] / max(summaries[0]['total_requests'], 1):.0f}%",
     f"{100 * summaries[1]['success'] / max(summaries[1]['total_requests'], 1):.0f}%",
     f"{100 * summaries[2]['success'] / max(summaries[2]['total_requests'], 1):.0f}%"],
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

output_path = "docs/benchmarks.png"
plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"Chart saved: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.0f} KB")
