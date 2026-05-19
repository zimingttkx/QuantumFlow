#!/usr/bin/env python3
"""GPU单卡压测脚本 — 触发动态批处理，观察GPU利用率"""

import asyncio
import os
import random
import time
from datetime import datetime

import httpx

# 禁用代理环境变量
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)


MODEL = "Qwen2.5-1.5B"
BASE_URL = "http://localhost:8000"
CONCURRENT_REQUESTS = 24  # 同时24个请求，触发 max_batch_size=8 的合并
MAX_TOKENS = 256  # 增加输出长度，让GPU有更多工作
PROMPTS = [
    "请详细解释量子力学中的不确定性原理，包括海森堡的原始表述和现代数学表述，以及它在双缝实验中的具体体现。请举例说明测量对量子系统的影响。",
    "用Python实现一个完整的二叉搜索树，包括插入、删除、查找和遍历操作。要求代码包含类型注解和详细的注释说明。",
    "机器学习中的反向传播算法是深度学习的基础。请详细推导梯度下降的数学过程，解释链式法则在其中的作用，以及如何避免梯度消失和梯度爆炸问题。",
    "大模型的涌现能力是指什么？请从缩放定律的角度分析这一问题，并讨论grokking、in-context learning等新兴能力的出现条件。",
    "请用中文写一首关于人工智能与人类关系的现代诗，要求包含科技元素和人文思考，韵律优美，意境深远。",
    "Transformer架构自2017年提出以来彻底改变了NLP领域。请详细解释Self-Attention的工作机制，包括Query、Key、Value矩阵的计算过程。",
    "GPU的显存带宽对深度学习训练速度有决定性影响。请解释HBM2、HBM3显存的工作原理，以及它与传统GDDR显存的区别。",
    "在大模型推理中，KV Cache是加速生成的核心技术。请说明其工作原理，以及PagedAttention如何解决显存碎片化问题。",
    "Python的asyncio模块如何实现协程？请解释事件循环、Future、Task的概念，以及await语法的异步执行机制。",
    "大模型的上下文窗口长度受到哪些因素限制？请分析Softmax注意力计算的复杂度，以及各种长度外推技术的工作原理。",
    "分布式训练中的数据并行、模型并行和流水线并行各有优缺点。请比较FSDP、DeepSpeed ZeRO、Megatron-LM等主流方案的实现差异。",
    "混合专家模型(MoE)通过稀疏激活来突破模型规模的限制。请解释MoE的工作原理，包括门控网络的设计和负载均衡的必要性。",
    "大模型推理中的批处理优化对吞吐量至关重要。请比较FIFO批处理、动态批处理和contiguous batching的优缺点。",
    "强化学习中的人类反馈(RLHF)是如何让大模型与人类偏好对齐的？请详细说明Reward Model训练和PPO优化的过程。",
    "模型量化通过降低权重精度来减少显存占用。请比较INT8、INT4量化的实现方式，以及GPTQ、AWQ等量化算法的特点。",
    "向量数据库在RAG系统中扮演关键角色。请解释HNSW、IVF等索引算法的工作原理，以及如何选择合适的距离度量。",
    "GPU并行计算的核心是SIMT架构。请解释CUDA线程层次结构，以及warp、block、grid的组织方式和调度机制。",
    "Python中的GIL对多线程性能有什么影响？异步编程如何绕过这一限制？请结合实际场景分析并发模型的选择。",
    "大模型推理服务中的推测解码(Speculative Decoding)如何加速生成？请分析其正确性保证和效率权衡。",
    "从技术发展角度，分析Transformer架构的局限性和未来可能的研究方向。",
]


async def send_request(client: httpx.AsyncClient, prompt: str, request_id: int):
    """发送单个推理请求"""
    t0 = time.time()
    try:
        resp = await client.post(
            f"{BASE_URL}/api/v1/inference/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": MAX_TOKENS,
                    "top_p": 0.9,
                },
            },
            timeout=120.0,
        )
        elapsed = (time.time() - t0) * 1000
        result = resp.json() if resp.status_code == 200 else None
        return {
            "id": request_id,
            "prompt": prompt[:20],
            "elapsed_ms": elapsed,
            "success": resp.status_code == 200,
            "latency_ms": result.get("latency_ms", 0) if result else 0,
            "tokens": result.get("usage", {}).get("total_tokens", 0) if result else 0,
        }
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return {
            "id": request_id,
            "prompt": prompt[:20],
            "elapsed_ms": elapsed,
            "success": False,
            "error": str(e),
        }


async def check_scheduler_status(client: httpx.AsyncClient):
    """获取调度状态"""
    try:
        resp = await client.get(f"{BASE_URL}/api/v1/scheduler/status", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


async def get_gpu_status(client: httpx.AsyncClient):
    """获取GPU状态"""
    try:
        resp = await client.get(f"{BASE_URL}/api/v1/cluster/nodes", timeout=5.0)
        if resp.status_code == 200:
            nodes = resp.json()
            if nodes and nodes[0].get("gpu_info"):
                g = nodes[0]["gpu_info"][0]
                return {
                    "name": g["name"],
                    "utilization": f"{g['utilization']*100:.0f}%",
                    "memory_utilization": f"{g['memory_util_pct']:.0f}%",
                    "memory_used": f"{g['memory_used']/1e9:.1f}GB",
                    "memory_total": f"{g['memory_total']/1e9:.1f}GB",
                    "memory_pct": f"{g['memory_used']/g['memory_total']*100:.0f}%",
                }
    except:
        pass
    return None


async def run_batch_test(round_num: int):
    """运行一轮并发测试"""
    print(f"\n{'='*60}")
    print(f"  第 {round_num} 轮并发测试 — {CONCURRENT_REQUESTS} 个请求同时发出")
    print(f"{'='*60}")

    async with httpx.AsyncClient() as client:
        # 并发发送所有请求
        tasks = [
            send_request(client, random.choice(PROMPTS), i) for i in range(CONCURRENT_REQUESTS)
        ]
        results = await asyncio.gather(*tasks)

        # 统计
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]
        latencies = [r["latency_ms"] for r in successes]
        wall_times = [r["elapsed_ms"] for r in successes]

        print(f"\n  成功: {len(successes)}/{CONCURRENT_REQUESTS} | 失败: {len(failures)}")
        if successes:
            print(
                f"  端到端延迟:  min={min(wall_times):.0f}ms  avg={sum(wall_times)/len(wall_times):.0f}ms  max={max(wall_times):.0f}ms"
            )
            print(
                f"  模型推理延迟: min={min(latencies):.0f}ms  avg={sum(latencies)/len(latencies):.0f}ms  max={max(latencies):.0f}ms"
            )

        # 获取调度状态
        sched = await check_scheduler_status(client)
        if sched:
            batch = sched.get("batch", {})
            if batch:
                print("\n  📦 批处理统计:")
                for key, stats in batch.items():
                    print(
                        f"     {key}: {stats['total_requests']} 请求 / {stats['total_batches']} 批次 / 平均批量 {stats['avg_batch_size']}"
                    )

            vram = sched.get("vram", {})
            if vram:
                print(
                    f"\n  🧠 VRAM: 可用 {vram['available_vram_gb']}GB | 安全系数 {vram['safety_factor']*100:.0f}% | 加载模型 {vram['loaded_count']}个"
                )
                for m in vram.get("loaded_models", []):
                    status = "🔄 推理中" if m["in_use"] else "💤 空闲"
                    print(
                        f"     - {m['model_name']}: {m['estimated_vram_gb']}GB | {status} | 空闲 {m['idle_seconds']:.1f}s"
                    )

        # 获取GPU状态
        gpu = await get_gpu_status(client)
        if gpu:
            print(f"\n  🎮 GPU: {gpu['name']}")
            print(
                f"     计算利用率: {gpu['utilization']} | 显存带宽利用率: {gpu['memory_utilization']} | 显存: {gpu['memory_used']}/{gpu['memory_total']} ({gpu['memory_pct']})"
            )

        return successes, failures


async def monitor_loop():
    """后台监控GPU利用率变化"""
    print("\n[后台监控] 开始监控GPU利用率变化...")
    async with httpx.AsyncClient() as client:
        for i in range(20):  # 监控20次
            await asyncio.sleep(2)
            try:
                resp = await client.get(f"{BASE_URL}/api/v1/cluster/nodes", timeout=3.0)
                if resp.status_code == 200:
                    nodes = resp.json()
                    if nodes and nodes[0].get("gpu_info"):
                        g = nodes[0]["gpu_info"][0]
                        util = g["utilization"]
                        mem_used = g["memory_used"] / 1e9
                        mem_total = g["memory_total"] / 1e9
                        ts = datetime.now().strftime("%H:%M:%S")
                        bar = "█" * int(util * 20) + "░" * (20 - int(util * 20))
                        print(
                            f"  [{ts}] GPU利用率: [{bar}] {util*100:.0f}% | 显存: {mem_used:.1f}/{mem_total:.1f}GB"
                        )
            except:
                pass


async def main():
    print("=" * 60)
    print("  QuantumFlow 单卡GPU利用率压测")
    print("=" * 60)
    print(f"  模型: {MODEL}")
    print(f"  并发: {CONCURRENT_REQUESTS} 请求/轮")
    print(f"  max_tokens: {MAX_TOKENS}")
    print("  批处理: max_batch_size=8, max_delay=50ms")
    print()

    # 先检查服务是否可用
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BASE_URL}/api/v1/cluster/status", timeout=5.0)
            if resp.status_code != 200:
                print("❌ 无法连接到服务器")
                return
            sched_resp = await client.get(f"{BASE_URL}/api/v1/scheduler/status", timeout=5.0)
            print(f"✅ 服务器正常 | 调度API: {'✓' if sched_resp.status_code == 200 else '✗'}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 运行3轮压测
    for i in range(1, 4):
        await run_batch_test(i)
        if i < 3:
            await asyncio.sleep(3)  # 轮次间隔

    print(f"\n{'='*60}")
    print("  压测完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
