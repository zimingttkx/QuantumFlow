"""深度验证测试：模型加载 → 推理 → 流式 → 卸载 全流程

注意：本脚本是手动运行集成测试，不是 pytest 测试文件。
用法: python tests/deep_test.py
"""
import asyncio
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
pytestmark = pytest.mark.skip(reason="手动集成测试脚本，需 GPU + 模型下载，通过 python tests/deep_test.py 运行")

from quantumflow.inference.engine import ModelConfig, SamplingParams, InferenceResult
from quantumflow.inference.backends.huggingface import HuggingFaceEngine
from quantumflow.inference.manager import EngineManager
from quantumflow.core.constants import InferenceBackendType

PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        msg = f"  ✗ {name} — {detail}"
        print(msg)
        ERRORS.append(msg)


async def test_model(engine, model_name, model_path, prompts):
    """测试单个模型的完整推理流程"""
    print(f"\n{'='*60}")
    print(f"📦 测试: {model_name} ({model_path})")
    print(f"{'='*60}")

    config = ModelConfig(
        model_name=model_name,
        model_path=model_path,
        trust_remote_code=True,
        max_model_len=2048,
    )

    # 1. 加载模型
    print("\n[1] 加载模型...")
    t0 = time.time()
    ok = await engine.load_model(config)
    elapsed = time.time() - t0
    check("模型加载成功", ok, f"耗时 {elapsed:.1f}s")
    if not ok:
        return False
    check("加载耗时 < 60s", elapsed < 60, f"实际 {elapsed:.1f}s")
    check("模型在 loaded_models 中", await engine.is_model_loaded(model_name))

    # 2. 基础生成
    print("\n[2] 基础生成...")
    sampling = SamplingParams(temperature=0.1, max_tokens=30, top_p=0.9, top_k=50)
    t0 = time.time()
    results = await engine.generate(model_name, prompts[:1], sampling)
    elapsed = time.time() - t0

    check("返回非空结果", len(results) > 0)
    if results:
        r = results[0]
        check("有生成文本", len(r.outputs[0]) > 0, f"文本: {r.outputs[0][:80]}...")
        check("有 prompt_tokens", r.prompt_tokens > 0, f"tokens: {r.prompt_tokens}")
        check("有 completion_tokens", r.completion_tokens > 0, f"tokens: {r.completion_tokens}")
        check("有 finish_reason", r.finish_reason in ("stop", "length"), f"reason: {r.finish_reason}")
        check("有 latency_ms", r.latency_ms > 0, f"latency: {r.latency_ms:.0f}ms")
        check("completion_tokens <= max_tokens", r.completion_tokens <= 30, f"实际: {r.completion_tokens}")
        print(f"    输出: {r.outputs[0][:100]}")

    # 3. 批量生成
    print("\n[3] 批量生成...")
    results = await engine.generate(model_name, prompts, sampling)
    check("返回正确数量", len(results) == len(prompts), f"期望 {len(prompts)}, 实际 {len(results)}")
    for i, r in enumerate(results):
        check(f"  结果[{i}] 有文本", len(r.outputs[0]) > 0)

    # 4. 流式生成
    print("\n[4] 流式生成...")
    chunks = []
    async for chunk in engine.generate_stream(model_name, prompts[0], sampling):
        chunks.append(chunk)
    check("流式有输出", len(chunks) > 0, f"chunks: {len(chunks)}")
    full_text = "".join(chunks)
    check("流式输出非空", len(full_text) > 0, f"文本: {full_text[:80]}...")

    # 5. 统计
    print("\n[5] 引擎统计...")
    stats = await engine.get_stats(model_name)
    check("返回 dict", isinstance(stats, dict))

    # 6. 卸载
    print("\n[6] 卸载模型...")
    ok = await engine.unload_model(model_name)
    check("卸载成功", ok)
    check("不在 loaded_models 中", not await engine.is_model_loaded(model_name))

    return True


async def test_sampling_params():
    """测试采样参数的各种组合"""
    print(f"\n{'='*60}")
    print(f"🧪 测试: 采样参数边界")
    print(f"{'='*60}")

    # 加载一个小模型
    engine = HuggingFaceEngine()
    await engine.initialize()

    config = ModelConfig(
        model_name="test-sampling",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
        max_model_len=512,
    )
    await engine.load_model(config)

    prompt = ["Hello"]

    # temperature=0 (greedy)
    print("\n[1] Greedy (temperature=0)...")
    sp = SamplingParams(temperature=0, max_tokens=10)
    results = await engine.generate("test-sampling", prompt, sp)
    greedy_text = results[0].outputs[0] if results else ""
    check("Greedy 有输出", len(greedy_text) > 0, f"文本: {greedy_text}")

    # temperature=1.0 (random)
    print("\n[2] Random (temperature=1.0)...")
    sp = SamplingParams(temperature=1.0, max_tokens=10)
    results = await engine.generate("test-sampling", prompt, sp)
    random_text = results[0].outputs[0] if results else ""
    check("Random 有输出", len(random_text) > 0, f"文本: {random_text}")

    # high top_k
    print("\n[3] top_k=5...")
    sp = SamplingParams(temperature=0.7, top_k=5, max_tokens=10)
    results = await engine.generate("test-sampling", prompt, sp)
    check("top_k=5 有输出", len(results) > 0 and len(results[0].outputs[0]) > 0)

    # low top_p
    print("\n[4] top_p=0.1...")
    sp = SamplingParams(temperature=0.7, top_p=0.1, max_tokens=10)
    results = await engine.generate("test-sampling", prompt, sp)
    check("top_p=0.1 有输出", len(results) > 0 and len(results[0].outputs[0]) > 0)

    # repetition_penalty
    print("\n[5] repetition_penalty=1.2...")
    sp = SamplingParams(temperature=0.7, repetition_penalty=1.2, max_tokens=10)
    results = await engine.generate("test-sampling", prompt, sp)
    check("repetition_penalty 有输出", len(results) > 0 and len(results[0].outputs[0]) > 0)

    # max_tokens=1
    print("\n[6] max_tokens=1...")
    sp = SamplingParams(temperature=0, max_tokens=1)
    results = await engine.generate("test-sampling", prompt, sp)
    if results:
        check("max_tokens=1 生成1个token", results[0].completion_tokens <= 1)

    # max_tokens=500
    print("\n[7] max_tokens=500...")
    sp = SamplingParams(temperature=0.7, max_tokens=500)
    t0 = time.time()
    results = await engine.generate("test-sampling", prompt, sp)
    elapsed = time.time() - t0
    check("max_tokens=500 有输出", len(results) > 0 and len(results[0].outputs[0]) > 0)
    check("max_tokens=500 不超时", elapsed < 30, f"耗时 {elapsed:.1f}s")

    await engine.unload_model("test-sampling")
    return True


async def test_engine_manager():
    """测试 EngineManager 单例和路由逻辑"""
    print(f"\n{'='*60}")
    print(f"🏭 测试: EngineManager")
    print(f"{'='*60}")

    mgr1 = EngineManager()
    mgr2 = EngineManager()
    check("单例模式", mgr1 is mgr2)

    ok = await mgr1.initialize(InferenceBackendType.HUGGINGFACE)
    check("初始化成功", ok)
    check("is_model_loaded 返回 False", not mgr1.is_model_loaded("nonexistent"))

    return True


async def test_error_handling():
    """测试错误处理路径"""
    print(f"\n{'='*60}")
    print(f"🛡️  测试: 错误处理")
    print(f"{'='*60}")

    engine = HuggingFaceEngine()
    await engine.initialize()

    # 未加载模型时 generate
    print("\n[1] 未加载模型 generate...")
    sp = SamplingParams(max_tokens=10)
    results = await engine.generate("nonexistent", ["Hello"], sp)
    # 模型未加载时应返回包含错误信息的 InferenceResult 列表，不是空列表
    check("未加载模型返回错误结果", len(results) == 1 and "模型未加载" in results[0].outputs[0])

    # 未加载模型时 generate_stream
    print("\n[2] 未加载模型 generate_stream...")
    chunks = []
    async for chunk in engine.generate_stream("nonexistent", "Hello", sp):
        chunks.append(chunk)
    check("未加载模型流式无输出", len(chunks) == 0)

    # 未加载模型时 unload
    print("\n[3] 未加载模型 unload...")
    ok = await engine.unload_model("nonexistent")
    check("未加载模型 unload 返回 False", not ok)

    # 重复加载相同模型
    print("\n[4] 重复加载相同模型...")
    config = ModelConfig(
        model_name="test-dup",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
        max_model_len=512,
    )
    ok = await engine.load_model(config)
    check("首次加载成功", ok)
    ok2 = await engine.load_model(config)
    check("重复加载覆盖成功", ok2)
    await engine.unload_model("test-dup")

    return True


async def test_model_switch():
    """测试模型切换：加载A→推理→加载B→推理→卸载B→推理A"""
    print(f"\n{'='*60}")
    print(f"🔄 测试: 模型切换")
    print(f"{'='*60}")

    engine = HuggingFaceEngine()
    await engine.initialize()

    sp = SamplingParams(temperature=0.1, max_tokens=20)

    # 加载 A
    config_a = ModelConfig(model_name="model-a", model_path="Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, max_model_len=256)
    await engine.load_model(config_a)
    check("加载 model-a", await engine.is_model_loaded("model-a"))

    # 推理 A
    results_a = await engine.generate("model-a", ["Hello"], sp)
    check("model-a 推理成功", len(results_a) > 0 and len(results_a[0].outputs[0]) > 0)

    # 加载 B
    config_b = ModelConfig(model_name="model-b", model_path="Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True, max_model_len=256)
    await engine.load_model(config_b)
    check("加载 model-b", await engine.is_model_loaded("model-b"))

    # 推理 B
    results_b = await engine.generate("model-b", ["Hello"], sp)
    check("model-b 推理成功", len(results_b) > 0 and len(results_b[0].outputs[0]) > 0)

    # 推理 A (仍然可用)
    results_a2 = await engine.generate("model-a", ["Hello"], sp)
    check("卸载B后 model-a 仍可用", len(results_a2) > 0 and len(results_a2[0].outputs[0]) > 0)

    # 卸载 B
    await engine.unload_model("model-b")
    check("卸载 model-b", not await engine.is_model_loaded("model-b"))
    check("卸载B后 model-a 仍在", await engine.is_model_loaded("model-a"))

    await engine.unload_model("model-a")
    return True


async def main():
    global PASS, FAIL, ERRORS

    print("╔══════════════════════════════════════════════════╗")
    print("║   QuantumFlow 深度验证测试                        ║")
    print("╚══════════════════════════════════════════════════╝")

    engine = HuggingFaceEngine()
    ok = await engine.initialize()
    if not ok:
        print("❌ HF引擎初始化失败，终止测试")
        return 1

    # 测试模型列表（用已有缓存的小模型）
    models = [
        ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("Phi-3-mini-4k", "microsoft/Phi-3-mini-4k-instruct"),
    ]

    prompts = [
        "What is the capital of France?",
        "Write a haiku about coding:",
        "1 + 1 =",
    ]

    for name, path in models:
        try:
            await test_model(engine, name, path, prompts)
        except Exception as e:
            FAIL += 1
            msg = f"  ✗ 测试 {name} 异常: {e}"
            print(msg)
            ERRORS.append(msg)
            import traceback
            traceback.print_exc()
            # 尝试卸载
            try:
                await engine.unload_model(name)
            except Exception:
                pass

    # 采样参数测试
    try:
        await test_sampling_params()
    except Exception as e:
        FAIL += 1
        print(f"  ✗ 采样参数测试异常: {e}")
        import traceback
        traceback.print_exc()

    # 错误处理测试
    try:
        await test_error_handling()
    except Exception as e:
        FAIL += 1
        print(f"  ✗ 错误处理测试异常: {e}")

    # EngineManager 测试
    try:
        await test_engine_manager()
    except Exception as e:
        FAIL += 1
        print(f"  ✗ EngineManager 测试异常: {e}")

    # 模型切换测试
    try:
        await test_model_switch()
    except Exception as e:
        FAIL += 1
        print(f"  ✗ 模型切换测试异常: {e}")

    # 最终报告
    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"📊 测试报告: {PASS}/{total} 通过, {FAIL} 失败")
    print(f"{'='*60}")
    if ERRORS:
        print("\n❌ 失败项:")
        for e in ERRORS:
            print(f"  {e}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
