#!/usr/bin/env python3
"""GPU Performance Benchmark — vLLM Backend"""
import os, asyncio, httpx, time, json
import numpy as np

for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

BASE = "http://localhost:8000"
MODEL = "Qwen2.5-1.5B"
MAX_TOKENS = 128

PROMPTS = [
    "Explain quantum uncertainty principle in simple terms.",
    "Write a Python quicksort with type hints and comments.",
    "What is the Transformer architecture in deep learning?",
    "Describe how gradient descent works in machine learning.",
    "What is KV Cache in LLM inference and why does it matter?",
]


async def get_gpu(client):
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


async def send_one(client, prompt):
    t0 = time.time()
    try:
        r = await client.post(
            f"{BASE}/api/v1/inference/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "sampling_params": {"temperature": 0.7, "max_tokens": MAX_TOKENS, "top_p": 0.9},
            },
            timeout=60.0,
        )
        elapsed_ms = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        return {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "latency_ms": data.get("latency_ms", 0),
            "tokens": data.get("usage", {}).get("total_tokens", 0) if data else 0,
        }
    except Exception as e:
        return {"ok": False, "elapsed_ms": (time.time() - t0) * 1000, "latency_ms": 0, "tokens": 0}


async def run_scenario(name, concurrent, rounds):
    print(f"\n  {name}  (concurrent={concurrent}, rounds={rounds})")

    latencies, all_tokens = [], []
    gpu_utils, mem_utils, vram_gbs = [], [], []
    successes = 0

    for r in range(rounds):
        async with httpx.AsyncClient(timeout=60.0) as client:
            gpu_u, mem_u, vram_u, _ = await get_gpu(client)
            gpu_utils.append(gpu_u)
            mem_utils.append(mem_u)
            vram_gbs.append(vram_u)

            tasks = [send_one(client, PROMPTS[i % len(PROMPTS)]) for i in range(concurrent)]
            results = await asyncio.gather(*tasks)

            for res in results:
                if res["ok"]:
                    successes += 1
                    latencies.append(res["latency_ms"])
                    all_tokens.append(res["tokens"])

            await asyncio.sleep(0.5)

    total = concurrent * rounds
    wall_s = sum(r["elapsed_ms"] for r in results if r["ok"]) / 1000
    tp = sum(all_tokens) / max(wall_s, 0.001)

    return {
        "name": name,
        "total_requests": total,
        "success": successes,
        "failed": total - successes,
        "avg_gpu_util": float(np.mean(gpu_utils)) if gpu_utils else 0,
        "max_gpu_util": float(np.max(gpu_utils)) if gpu_utils else 0,
        "avg_gpu_mem_util": float(np.mean(mem_utils)) if mem_utils else 0,
        "avg_vram_gb": float(np.mean(vram_gbs)) if vram_gbs else 0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0,
        "throughput_tok_s": tp,
    }


async def main():
    print("=" * 60)
    print("  QuantumFlow GPU Benchmark — vLLM Backend")
    print("  Model: Qwen2.5-1.5B | RTX 4080 Laptop GPU")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"{BASE}/api/v1/cluster/status")
            if r.status_code != 200:
                print("Server not ready"); return
            print("Server OK")
        except Exception as e:
            print(f"Cannot connect: {e}"); return

    results = []

    r1 = await run_scenario("Baseline (single)", concurrent=1, rounds=3)
    results.append(r1)
    print(f"    GPU: {r1['avg_gpu_util']:.0f}% | P50: {r1['p50_latency_ms']:.0f}ms | Throughput: {r1['throughput_tok_s']:.1f} tok/s | Success: {r1['success']}/{r1['total_requests']}")

    await asyncio.sleep(2)

    r2 = await run_scenario("Low (8 concurrent)", concurrent=8, rounds=3)
    results.append(r2)
    print(f"    GPU: {r2['avg_gpu_util']:.0f}% | P50: {r2['p50_latency_ms']:.0f}ms | Throughput: {r2['throughput_tok_s']:.1f} tok/s | Success: {r2['success']}/{r2['total_requests']}")

    await asyncio.sleep(2)

    r3 = await run_scenario("High (24 concurrent)", concurrent=24, rounds=3)
    results.append(r3)
    print(f"    GPU: {r3['avg_gpu_util']:.0f}% | P50: {r3['p50_latency_ms']:.0f}ms | Throughput: {r3['throughput_tok_s']:.1f} tok/s | Success: {r3['success']}/{r3['total_requests']}")

    with open("docs/benchmark_data.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    hdr = f"{'Scenario':<25} {'GPU Util':<10} {'MemBW':<8} {'P50 (ms)':<10} {'Throughput':<14} {'Success'}"
    print(hdr)
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25} {r['avg_gpu_util']:.1f}%{'':>5} {r['avg_gpu_mem_util']:.1f}%{'':>3} {r['p50_latency_ms']:.0f}{'':>7} {r['throughput_tok_s']:.1f} tok/s{'':>4} {r['success']}/{r['total_requests']}")

    print(f"\nData: docs/benchmark_data.json")
    print(f"Chart: tests/regenerate_chart.py")


if __name__ == "__main__":
    asyncio.run(main())
