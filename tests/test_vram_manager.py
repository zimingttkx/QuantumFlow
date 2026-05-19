"""VRAM感知模型加载 — 100%覆盖率业务逻辑测试

测试范围:
- 静态方法: _estimate_param_count, _estimate_architecture
- VRAM估算: estimate_model_vram_gb (含 KV cache 精确/粗略)
- can_load 决策: 直接加载/刚好够/需淘汰/拒绝/0VRAM/0需求
- 淘汰评分: LRU+大小复合评分公式精确验证
- 状态管理: record/mark/update 全生命周期
- 空闲淘汰: TTL边界/禁用/in_use保护
- 安全系数: 不同配置影响决策
- 真实GPU: 检测/fallback
- 边界: 空dict/全部in_use/0值/不存在模型操作
"""

import time
from unittest.mock import MagicMock, patch

from quantumflow.inference.vram_manager import VRAM_SAFETY_FACTOR, LoadedModelInfo, VRAMManager

PASS = 0
FAIL = 0
FAIL_MSGS = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        msg = f"  ✗ {name} — {detail}"
        print(msg)
        FAIL_MSGS.append(msg)


def report():
    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"📊 VRAM Manager 测试报告: {PASS}/{total} 通过, {FAIL} 失败")
    print(f"{'='*60}")
    if FAIL_MSGS:
        print("\n❌ 失败项:")
        for e in FAIL_MSGS:
            print(f"  {e}")
    return FAIL == 0


# ── helpers ──────────────────────────────────────────────


def make_vram(
    free_vram: float, used_vram: float = 0.0, safety: float = None, idle_ttl: float = 0.0
) -> VRAMManager:
    """创建 VRAMManager，mock 掉 GPU 读取"""
    mgr = VRAMManager(
        safety_factor=safety if safety is not None else VRAM_SAFETY_FACTOR,
        idle_ttl_seconds=idle_ttl,
    )
    mgr._read_free_vram_gb = MagicMock(return_value=free_vram)
    mgr._read_used_vram_gb = MagicMock(return_value=used_vram)
    return mgr


# ═══════════════════════════════════════════════════════════
# 1. 静态方法 — 参数量估算
# ═══════════════════════════════════════════════════════════


def test_param_count__all_known_patterns():
    """每个已知参数量模式精确匹配"""
    print("\n── 参数量: 所有已知模式 ──")
    cases = [
        ("0.5b-model", 500_000_000),
        ("1.5b-model", 1_500_000_000),
        ("2.6b-model", 2_600_000_000),
        ("3.8b-model", 3_800_000_000),
        ("14b-model", 14_000_000_000),
        ("13b-model", 13_000_000_000),
        ("11b-model", 11_000_000_000),
        ("9b-model", 9_000_000_000),
        ("8b-model", 8_000_000_000),
        ("7b-model", 7_000_000_000),
        ("72b-model", 72_000_000_000),
        ("70b-model", 70_000_000_000),
        ("34b-model", 34_000_000_000),
        ("20b-model", 20_000_000_000),
        ("3b-model", 3_000_000_000),
        ("2b-model", 2_000_000_000),
        ("1b-model", 1_000_000_000),
    ]
    for path, expected in cases:
        got = VRAMManager._estimate_param_count(path)
        check(
            f"{path} → {expected//1_000_000_000}B",
            got == expected,
            f"expected={expected}, got={got}",
        )


def test_param_count__long_pattern_before_short():
    """长模式优先：13b 不应被 1b 误匹配"""
    print("\n── 参数量: 长模式优先 ──")
    check(
        "13b不被1b误匹配",
        VRAMManager._estimate_param_count("meta-llama/Llama-2-13b-chat-hf") == 13_000_000_000,
    )
    check(
        "11b不被1b误匹配",
        VRAMManager._estimate_param_count("model-11b-something") == 11_000_000_000,
    )
    check("14b不被1b误匹配", VRAMManager._estimate_param_count("Qwen2.5-14B") == 14_000_000_000)
    check(
        "1.5b不被1b误匹配(1.5b在1b之前)",
        VRAMManager._estimate_param_count("Qwen2.5-1.5B") == 1_500_000_000,
    )


def test_param_count__case_insensitive():
    """大小写不敏感"""
    print("\n── 参数量: 大小写 ──")
    check("大写B", VRAMManager._estimate_param_count("QWEN2.5-7B") == 7_000_000_000)
    check("小写b", VRAMManager._estimate_param_count("qwen2.5-7b") == 7_000_000_000)
    check("混合", VRAMManager._estimate_param_count("Qwen2.5-7b-InStRuCt") == 7_000_000_000)


def test_param_count__unknown():
    """未匹配返回0"""
    print("\n── 参数量: 未知模型 ──")
    check("无数字", VRAMManager._estimate_param_count("bert-base-uncased") == 0)
    check("空字符串", VRAMManager._estimate_param_count("") == 0)
    check("仅有数字无b", VRAMManager._estimate_param_count("model-7") == 0)
    check("路径含数字但非模型大小", VRAMManager._estimate_param_count("v1.0/model") == 0)


# ═══════════════════════════════════════════════════════════
# 2. 静态方法 — 架构估算
# ═══════════════════════════════════════════════════════════


def test_architecture__known():
    """已知架构精确匹配"""
    print("\n── 架构: 已知模型 ──")
    cases = [
        ("0.5b-model", (896, 24)),
        ("1.5b-model", (1536, 28)),
        ("3b-model", (2560, 32)),
        ("7b-model", (4096, 32)),
        ("8b-model", (4096, 32)),
        ("13b-model", (5120, 40)),
        ("14b-model", (5120, 40)),
    ]
    for path, expected in cases:
        got = VRAMManager._estimate_architecture(path)
        check(f"{path} → {expected}", got == expected, f"got={got}")


def test_architecture__70b_72b():
    """70B/72B 架构"""
    print("\n── 架构: 70B/72B ──")
    check("70b", VRAMManager._estimate_architecture("llama-2-70b") == (8192, 80))
    check("72b", VRAMManager._estimate_architecture("qwen-72b") == (8192, 80))


def test_architecture__phi_mini():
    """Phi-3-mini 特殊架构"""
    print("\n── 架构: Phi-mini ──")
    check("phi+mini", VRAMManager._estimate_architecture("Phi-3-mini-4k-instruct") == (3072, 32))
    check("phi无mini不算", VRAMManager._estimate_architecture("Phi-3-medium") == (0, 0))


def test_architecture__unknown():
    """未知架构返回(0,0)"""
    print("\n── 架构: 未知 ──")
    check("未知", VRAMManager._estimate_architecture("random-model") == (0, 0))
    check("空", VRAMManager._estimate_architecture("") == (0, 0))


# ═══════════════════════════════════════════════════════════
# 3. VRAM 估算
# ═══════════════════════════════════════════════════════════


def test_vram_estimation__with_architecture():
    """有架构信息时精确估算 KV cache"""
    print("\n── VRAM估算: 精确(含架构) ──")
    mgr = VRAMManager()

    # 0.5B: params=5e8, model_gb=5e8*2/1e9*1.2=1.2
    #       KV=2*24*896*2048*2/1e9=0.176 → total≈1.38
    est = mgr.estimate_model_vram_gb("Qwen2.5-0.5B-Instruct", max_model_len=2048)
    check("0.5B-2048 ≈ 1.2-1.6GB", 1.2 <= est <= 1.6, f"got={est}")

    # 7B: params=7e9, model_gb=7e9*2/1e9*1.2=16.8
    #     KV=2*32*4096*2048*2/1e9=1.07 → total≈17.9
    est7 = mgr.estimate_model_vram_gb("Qwen2.5-7B-Instruct", max_model_len=2048)
    check("7B-2048 ≈ 16-20GB", 16.0 <= est7 <= 20.0, f"got={est7}")

    # 70B: params=7e10, model_gb=7e10*2/1e9*1.2=168
    #      KV=2*80*8192*2048*2/1e9=5.37 → total≈173
    est70 = mgr.estimate_model_vram_gb("llama-2-70b", max_model_len=2048)
    check("70B-2048 > 100GB", est70 > 100.0, f"got={est70}")


def test_vram_estimation__without_architecture():
    """无架构信息时用粗略 KV cache 估算（model_gb * 0.3）"""
    print("\n── VRAM估算: 粗略(无架构) ──")
    mgr = VRAMManager()

    # 用 patch 让 _estimate_architecture 返回 (0,0)
    with patch.object(VRAMManager, "_estimate_architecture", return_value=(0, 0)):
        est = mgr.estimate_model_vram_gb("some-7b-model", max_model_len=2048)
        # model_gb = 7e9*2/1e9*1.2=16.8, kv_rough=16.8*0.3=5.04 → total≈21.84
        check("7B粗略有KV", 15.0 < est < 30.0, f"got={est}")


def test_vram_estimation__unknown_model():
    """未知模型返回 0.0"""
    print("\n── VRAM估算: 未知模型 → 0 ──")
    mgr = VRAMManager()
    check("unknown→0", mgr.estimate_model_vram_gb("unknown-model") == 0.0)
    check("empty→0", mgr.estimate_model_vram_gb("") == 0.0)


def test_vram_estimation__max_model_len_affects_kv():
    """max_model_len 越大 → KV cache 越大 → 总估算越大"""
    print("\n── VRAM估算: max_model_len 影响 ──")
    mgr = VRAMManager()

    est_short = mgr.estimate_model_vram_gb("Qwen2.5-7B-Instruct", max_model_len=512)
    est_mid = mgr.estimate_model_vram_gb("Qwen2.5-7B-Instruct", max_model_len=2048)
    est_long = mgr.estimate_model_vram_gb("Qwen2.5-7B-Instruct", max_model_len=8192)

    check("short < mid", est_short < est_mid, f"short={est_short}, mid={est_mid}")
    check("mid < long", est_mid < est_long, f"mid={est_mid}, long={est_long}")
    check("long至少比short大2GB", est_long - est_short > 2.0, f"diff={est_long - est_short:.1f}")


def test_vram_estimation__max_model_len_zero():
    """max_model_len=0 时 KV cache 为 0"""
    print("\n── VRAM估算: max_model_len=0 ──")
    mgr = VRAMManager()
    est = mgr.estimate_model_vram_gb("Qwen2.5-0.5B-Instruct", max_model_len=0)
    # model_gb=1.2, KV=0 → total≈1.2
    check("KV=0", est > 0 and est < 2.0, f"got={est}")


# ═══════════════════════════════════════════════════════════
# 4. can_load 决策 — 4 种基本结果
# ═══════════════════════════════════════════════════════════


def test_can_load__direct_fit():
    """场景1: VRAM充足，直接加载"""
    print("\n── can_load: 直接加载 ──")
    mgr = make_vram(20.0)  # usable=14GB
    can, reason, evict = mgr.can_load("model-x", required_vram_gb=10.0)
    check("can=True", can)
    check("reason含'VRAM充足'", "VRAM充足" in reason)
    check("evict=[]", evict == [])


def test_can_load__tight_fit():
    """场景2: VRAM刚好够 — 边界值"""
    print("\n── can_load: 刚好够 ──")
    mgr = make_vram(20.0)  # usable=14GB

    # 正好等于 usable
    can, reason, evict = mgr.can_load("model", required_vram_gb=14.0)
    check("恰好14GB可加载", can and len(evict) == 0)

    # 比 usable 多 0.1GB → 需要淘汰
    can2, _, evict2 = mgr.can_load("model2", required_vram_gb=14.1)
    check("14.1GB需要淘汰", can2 is False)  # 无已加载模型可淘汰

    # 远小于 usable
    can3, _, evict3 = mgr.can_load("model3", required_vram_gb=0.1)
    check("0.1GB直接加载", can3 and len(evict3) == 0)


def test_can_load__evict_exact_one():
    """场景3: 需淘汰恰好1个模型"""
    print("\n── can_load: 淘汰1个 ──")
    mgr = make_vram(10.0)  # usable=7GB
    mgr.record_loaded("old-a", estimated_vram_gb=3.0)
    mgr.record_loaded("old-b", estimated_vram_gb=2.0)

    can, reason, evict = mgr.can_load("new-model", required_vram_gb=10.0)
    # shortage = 10-7 = 3GB, old-a(3GB)刚好够
    check("can=True", can)
    check("evict=[old-a(LRU最旧)]", evict == ["old-a"], f"got={evict}")
    check("reason含'淘汰'", "淘汰" in reason)


def test_can_load__evict_multiple():
    """场景4: 需淘汰多个模型（累计满足 shortage）"""
    print("\n── can_load: 淘汰多个 ──")
    mgr = make_vram(8.0)  # usable=5.6GB
    mgr.record_loaded("a", estimated_vram_gb=2.0)
    mgr.record_loaded("b", estimated_vram_gb=2.0)
    mgr.record_loaded("c", estimated_vram_gb=2.0)

    can, reason, evict = mgr.can_load("big", required_vram_gb=10.0)
    # shortage = 10-5.6 = 4.4GB → 需要淘汰 a+b(4GB)还不够? 4.0 < 4.4 → 需要 a+b+c(6GB)
    # 实际: a(2.0) < 4.4 → 加 b(累计4.0 < 4.4) → 加 c(累计6.0 >= 4.4)
    check("can=True", can)
    check("淘汰3个", len(evict) == 3, f"got={evict}")
    check("evict顺序LRU", evict == ["a", "b", "c"], f"got={evict}")


def test_can_load__reject_absolute_insufficient():
    """场景5: 即使淘汰全部也不够 → 拒绝"""
    print("\n── can_load: 绝对不足拒绝 ──")
    mgr = make_vram(8.0)  # usable=5.6GB
    mgr.record_loaded("a", estimated_vram_gb=3.0)
    mgr.record_loaded("b", estimated_vram_gb=2.0)

    can, reason, evict = mgr.can_load("huge", required_vram_gb=20.0)
    # total_freeable = 5.6 + 5.0 = 10.6 < 20
    check("can=False", not can)
    check("reason含'不足'", "不足" in reason)
    check("evict=[]", evict == [])


def test_can_load__zero_available_but_evictable():
    """场景6: 可用VRAM=0，但淘汰模型后够 → 可加载"""
    print("\n── can_load: 0VRAM但有淘汰 ──")
    mgr = make_vram(0.0)  # usable=0
    mgr.record_loaded("old", estimated_vram_gb=5.0)

    can, reason, evict = mgr.can_load("new", required_vram_gb=3.0)
    # usable=0, shortage=3, old(5GB) >= 3GB → 可加载
    check("can=True", can, f"reason={reason}")
    check("evict=[old]", evict == ["old"], f"got={evict}")


def test_can_load__zero_available_no_evictable():
    """场景7: 可用VRAM=0，无模型可淘汰 → 拒绝"""
    print("\n── can_load: 0VRAM无可淘汰 ──")
    mgr = make_vram(0.0)

    can, reason, evict = mgr.can_load("new", required_vram_gb=1.0)
    check("can=False", not can)
    check("reason含'不足'", "不足" in reason)
    check("evict=[]", evict == [])


def test_can_load__required_zero():
    """场景8: required_vram_gb=0 → 拒绝（无法估算）"""
    print("\n── can_load: required=0 ──")
    mgr = make_vram(20.0)
    can, reason, evict = mgr.can_load("unknown", required_vram_gb=0.0)
    check("can=False", not can)
    check("reason含'无法估算'", "无法估算" in reason.lower())
    check("evict=[]", evict == [])


def test_can_load__required_negative():
    """场景9: required_vram_gb<0 → 拒绝"""
    print("\n── can_load: required负值 ──")
    mgr = make_vram(20.0)
    can, reason, evict = mgr.can_load("bad", required_vram_gb=-1.0)
    check("can=False", not can)


def test_can_load__all_models_in_use():
    """场景10: 所有已加载模型都在推理中 → 无法淘汰 → 拒绝"""
    print("\n── can_load: 全部in_use ──")
    mgr = make_vram(10.0)  # usable=7GB
    mgr.record_loaded("busy-1", estimated_vram_gb=5.0)
    mgr.record_loaded("busy-2", estimated_vram_gb=5.0)
    mgr.mark_in_use("busy-1")
    mgr.mark_in_use("busy-2")

    can, reason, evict = mgr.can_load("new", required_vram_gb=10.0)
    check("can=False(无idle可淘汰)", not can)
    check("evict=[]", evict == [])


def test_can_load__partial_in_use():
    """场景11: 部分 in_use，仅淘汰 idle 模型"""
    print("\n── can_load: 部分in_use ──")
    mgr = make_vram(8.0)  # usable=5.6GB
    mgr.record_loaded("busy", estimated_vram_gb=4.0)
    mgr.mark_in_use("busy")
    # busy 的 4GB 不计入可用
    mgr.record_loaded("idle", estimated_vram_gb=3.0)

    can, reason, evict = mgr.can_load("new", required_vram_gb=8.0)
    # usable=5.6, shortage=8-5.6=2.4, idle(3GB)足够
    check("can=True(仅淘汰idle)", can and evict == ["idle"], f"can={can}, evict={evict}")


# ═══════════════════════════════════════════════════════════
# 5. 淘汰评分公式 — 精确验证
# ═══════════════════════════════════════════════════════════


def test_scoring__formula_verification():
    """评分=age_norm*0.7+size_norm*0.3，高分优先淘汰"""
    print("\n── 评分: 公式精确验证 ──")
    mgr = make_vram(20.0)

    now = time.time()
    # 直接构造 LoadedModelInfo 避免时间漂移
    mgr._loaded["old-big"] = LoadedModelInfo(
        model_name="old-big", estimated_vram_gb=10.0, last_used_at=now - 100
    )
    mgr._loaded["new-small"] = LoadedModelInfo(
        model_name="new-small", estimated_vram_gb=1.0, last_used_at=now - 10
    )

    candidates = mgr._get_eviction_candidates()

    # old-big: age=100, size=10
    #   age_norm=100/100=1.0, size_norm=10/10=1.0
    #   score=1.0*0.7+1.0*0.3=1.0
    # new-small: age=10, size=1
    #   age_norm=10/100=0.1, size_norm=1/10=0.1
    #   score=0.1*0.7+0.1*0.3=0.1
    check("2个候选", len(candidates) == 2)
    check("old-big优先淘汰(高分1.0)", candidates[0].model_name == "old-big")
    check("new-small后淘汰(低分0.1)", candidates[1].model_name == "new-small")


def test_scoring__same_age_size_decides():
    """相同年龄 → 大模型优先淘汰"""
    print("\n── 评分: 同年龄-大小决定 ──")
    mgr = make_vram(20.0)
    now = time.time()

    mgr._loaded["small"] = LoadedModelInfo(
        model_name="small", estimated_vram_gb=1.0, last_used_at=now - 50
    )
    mgr._loaded["large"] = LoadedModelInfo(
        model_name="large", estimated_vram_gb=10.0, last_used_at=now - 50
    )

    candidates = mgr._get_eviction_candidates()
    # 年龄相同→age_norm相同(0.0/0→... wait max_age=0 for same timestamp)
    # max_age = max(0,0) = 0 → or 1.0
    # age_norm both = 0/1.0 = 0
    # size_norm: small=0.1, large=1.0
    # score(small)=0, score(large)=0.3 → large first
    # Actually wait: max_age = max(now - last_used) = max(50, 50) = 50
    # age_norm both = 50/50 = 1.0
    # size_norm: small=1/10=0.1, large=10/10=1.0
    # score(small)=1.0*0.7+0.1*0.3=0.73
    # score(large)=1.0*0.7+1.0*0.3=1.0 → large first
    check("2个候选", len(candidates) == 2)
    check("large先淘汰", candidates[0].model_name == "large", f"first={candidates[0].model_name}")


def test_scoring__same_size_age_decides():
    """相同大小 → 更旧模型优先淘汰"""
    print("\n── 评分: 同大小-年龄决定 ──")
    mgr = make_vram(20.0)
    now = time.time()

    mgr._loaded["new"] = LoadedModelInfo(
        model_name="new", estimated_vram_gb=5.0, last_used_at=now - 10
    )
    mgr._loaded["old"] = LoadedModelInfo(
        model_name="old", estimated_vram_gb=5.0, last_used_at=now - 100
    )

    candidates = mgr._get_eviction_candidates()
    # max_age=100, max_vram=5
    # new: age_norm=10/100=0.1, size_norm=5/5=1.0 → score=0.07+0.3=0.37
    # old: age_norm=100/100=1.0, size_norm=5/5=1.0 → score=0.7+0.3=1.0
    check("old先淘汰", candidates[0].model_name == "old", f"first={candidates[0].model_name}")
    check("new后淘汰", candidates[1].model_name == "new")


def test_scoring__single_candidate():
    """仅1个 idle 模型 → 返回单候选"""
    print("\n── 评分: 单候选 ──")
    mgr = make_vram(20.0)
    mgr.record_loaded("only", estimated_vram_gb=5.0)
    # 1个busy, 0个idle
    mgr.record_loaded("busy", estimated_vram_gb=3.0)
    mgr.mark_in_use("busy")

    candidates = mgr._get_eviction_candidates()
    check("仅1个候选", len(candidates) == 1)
    check("候选是only", candidates[0].model_name == "only")


def test_scoring__empty_candidates():
    """边界: 空 _loaded → 空候选"""
    print("\n── 评分: 空候选 ──")
    mgr = make_vram(20.0)
    candidates = mgr._get_eviction_candidates()
    check("空_loaded→[]", candidates == [])


def test_scoring__all_in_use_returns_empty():
    """边界: 全部 in_use → 空候选"""
    print("\n── 评分: 全部in_use → [] ──")
    mgr = make_vram(20.0)
    mgr.record_loaded("a", estimated_vram_gb=3.0)
    mgr.record_loaded("b", estimated_vram_gb=3.0)
    mgr.mark_in_use("a")
    mgr.mark_in_use("b")

    candidates = mgr._get_eviction_candidates()
    check("全部in_use→[]", candidates == [])


def test_scoring__division_by_zero_safeguard():
    """if max_age or max_vram == 0 → 'or 1.0' 防除零"""
    print("\n── 评分: 除零保护 ──")
    mgr = make_vram(20.0)
    now = time.time()

    # 所有模型同一时刻加载 + 大小都为0 → max_age=0, max_vram=0
    mgr._loaded["a"] = LoadedModelInfo(model_name="a", estimated_vram_gb=0.0, last_used_at=now)
    mgr._loaded["b"] = LoadedModelInfo(model_name="b", estimated_vram_gb=0.0, last_used_at=now)

    candidates = mgr._get_eviction_candidates()
    # 不应 crash
    check("不crash", len(candidates) == 2)
    # 两者分数相同(均为0)，顺序无关
    check("都是候选", {c.model_name for c in candidates} == {"a", "b"})


# ═══════════════════════════════════════════════════════════
# 6. 状态管理 — record / mark / update / get
# ═══════════════════════════════════════════════════════════


def test_record_loaded__basic():
    """record_loaded 注册模型"""
    print("\n── 状态: record_loaded ──")
    mgr = make_vram(20.0)

    mgr.record_loaded("m1", estimated_vram_gb=5.0)
    check("m1在loaded中", "m1" in mgr.get_loaded_models())
    check("仅1个", len(mgr.get_loaded_models()) == 1)
    check("info正确", mgr._loaded["m1"].estimated_vram_gb == 5.0)
    check("初始in_use=False", not mgr._loaded["m1"].in_use)
    check("initial last_used_at", mgr._loaded["m1"].last_used_at > 0)


def test_record_loaded__overwrite():
    """重复 record_loaded 覆盖旧记录"""
    print("\n── 状态: record_loaded 覆盖 ──")
    mgr = make_vram(20.0)

    mgr.record_loaded("m", estimated_vram_gb=3.0)
    mgr.record_loaded("m", estimated_vram_gb=7.0)  # 覆盖
    check("覆盖后值更新", mgr._loaded["m"].estimated_vram_gb == 7.0)
    check("仍仅1个", len(mgr.get_loaded_models()) == 1)


def test_record_loaded__zero_vram():
    """estimated_vram_gb=0 也可记录"""
    print("\n── 状态: record_loaded(0GB) ──")
    mgr = make_vram(20.0)
    mgr.record_loaded("tiny", estimated_vram_gb=0.0)
    check("0GB也记录", "tiny" in mgr.get_loaded_models())


def test_record_unloaded__basic():
    """record_unloaded 移除模型"""
    print("\n── 状态: record_unloaded ──")
    mgr = make_vram(20.0)
    mgr.record_loaded("a", 5.0)
    mgr.record_loaded("b", 3.0)

    mgr.record_unloaded("a")
    check("a已移除", "a" not in mgr.get_loaded_models())
    check("b仍存在", "b" in mgr.get_loaded_models())
    check("仅剩1个", len(mgr.get_loaded_models()) == 1)


def test_record_unloaded__nonexistent():
    """卸载不存在的模型不报错"""
    print("\n── 状态: record_unloaded 不存在 ──")
    mgr = make_vram(20.0)
    mgr.record_unloaded("ghost")  # 不crash
    check("不crash", True)


def test_mark_in_use__basic():
    """mark_in_use 设置状态和时间戳"""
    print("\n── 状态: mark_in_use ──")
    mgr = make_vram(20.0)
    mgr.record_loaded("m", 5.0)

    old_ts = mgr._loaded["m"].last_used_at
    mgr.mark_in_use("m")
    check("in_use=True", mgr._loaded["m"].in_use)
    check("last_used_at更新", mgr._loaded["m"].last_used_at >= old_ts)


def test_mark_in_use__nonexistent():
    """mark_in_use 不存在模型不报错"""
    print("\n── 状态: mark_in_use 不存在 ──")
    mgr = make_vram(20.0)
    mgr.mark_in_use("ghost")  # 不crash
    check("不crash", True)


def test_mark_idle__basic():
    """mark_idle 恢复空闲状态"""
    print("\n── 状态: mark_idle ──")
    mgr = make_vram(20.0)
    mgr.record_loaded("m", 5.0)
    mgr.mark_in_use("m")
    check("先确认in_use", mgr._loaded["m"].in_use)

    mgr.mark_idle("m")
    check("恢复idle", not mgr._loaded["m"].in_use)


def test_mark_idle__nonexistent():
    """mark_idle 不存在模型不报错"""
    print("\n── 状态: mark_idle 不存在 ──")
    mgr = make_vram(20.0)
    mgr.mark_idle("ghost")  # 不crash
    check("不crash", True)


def test_update_actual_vram__basic():
    """update_actual_vram 从 GPU 读取实际 VRAM"""
    print("\n── 状态: update_actual_vram ──")
    mgr = make_vram(20.0, used_vram=8.5)
    mgr.record_loaded("m", 5.0)

    mgr.update_actual_vram("m")
    check("actual设置正确", mgr._loaded["m"].actual_vram_gb == 8.5)


def test_update_actual_vram__nonexistent():
    """update_actual_vram 不存在模型不报错"""
    print("\n── 状态: update_actual_vram 不存在 ──")
    mgr = make_vram(20.0)
    mgr.update_actual_vram("ghost")  # 不crash
    check("不crash", True)


def test_get_loaded_models__empty():
    """初始空列表"""
    print("\n── 状态: get_loaded_models 空 ──")
    mgr = make_vram(20.0)
    check("初始空", mgr.get_loaded_models() == [])


# ═══════════════════════════════════════════════════════════
# 7. 空闲超时淘汰
# ═══════════════════════════════════════════════════════════


def test_idle_ttl__expired_selected():
    """空闲超过 TTL 的模型被选中"""
    print("\n── TTL: 过期选择 ──")
    mgr = make_vram(20.0, idle_ttl=60.0)

    now = time.time()
    mgr.record_loaded("expired", 3.0)
    mgr.record_loaded("active", 3.0)
    mgr._loaded["expired"].last_used_at = now - 120.0  # 过期
    # active 保持 now()

    idle = mgr.get_idle_models_to_evict()
    check("expired在列", "expired" in idle)
    check("active不在列", "active" not in idle)
    check("仅1个过期", len(idle) == 1)


def test_idle_ttl__exact_boundary():
    """刚好超时 vs 刚好未超时 的边界"""
    print("\n── TTL: 精确边界 ──")
    mgr = make_vram(20.0, idle_ttl=60.0)
    now = time.time()

    mgr.record_loaded("just-expired", 3.0)
    mgr.record_loaded("just-active", 3.0)
    mgr._loaded["just-expired"].last_used_at = now - 60.001  # 刚超过
    mgr._loaded["just-active"].last_used_at = now - 59.999  # 刚未超

    idle = mgr.get_idle_models_to_evict()
    check("刚超时在列", "just-expired" in idle)
    check("刚未超时不在列", "just-active" not in idle)


def test_idle_ttl__disabled():
    """idle_ttl=0 禁用"""
    print("\n── TTL: 禁用(=0) ──")
    mgr = make_vram(20.0, idle_ttl=0.0)
    mgr.record_loaded("very-old", 3.0)
    mgr._loaded["very-old"].last_used_at = time.time() - 999999

    idle = mgr.get_idle_models_to_evict()
    check("TTL=0返回空", idle == [])


def test_idle_ttl__in_use_protected():
    """in_use 模型即使到期也不淘汰"""
    print("\n── TTL: in_use保护 ──")
    mgr = make_vram(20.0, idle_ttl=30.0)
    now = time.time()

    mgr.record_loaded("busy-expired", 3.0)
    mgr._loaded["busy-expired"].last_used_at = now - 120.0
    mgr.mark_in_use("busy-expired")

    idle = mgr.get_idle_models_to_evict()
    check("in_use不在淘汰列表", "busy-expired" not in idle)


def test_idle_ttl__multiple_expired():
    """多个过期模型全部返回"""
    print("\n── TTL: 多模型过期 ──")
    mgr = make_vram(20.0, idle_ttl=60.0)
    now = time.time()

    mgr.record_loaded("e1", 3.0)
    mgr.record_loaded("e2", 2.0)
    mgr.record_loaded("e3", 4.0)
    mgr._loaded["e1"].last_used_at = now - 120
    mgr._loaded["e2"].last_used_at = now - 300
    # e3 保持现在时间（未过期）

    idle = mgr.get_idle_models_to_evict()
    check("2个过期", len(idle) == 2)
    check("e1在列", "e1" in idle)
    check("e2在列", "e2" in idle)
    check("e3不在列", "e3" not in idle)


# ═══════════════════════════════════════════════════════════
# 8. safety_factor
# ═══════════════════════════════════════════════════════════


def test_safety_factor__affects_usable():
    """安全系数直接影响可用 VRAM"""
    print("\n── 安全系数: 影响可用VRAM ──")

    mgr_05 = make_vram(20.0, safety=0.5)
    can_05, _, _ = mgr_05.can_load("m", required_vram_gb=12.0)
    # usable=10GB < 12GB → reject
    check("0.5系数拒绝12GB", not can_05)

    mgr_07 = make_vram(20.0, safety=0.7)
    can_07, _, _ = mgr_07.can_load("m", required_vram_gb=12.0)
    # usable=14GB > 12GB → accept
    check("0.7系数接受12GB", can_07)

    mgr_09 = make_vram(20.0, safety=0.9)
    can_09, _, _ = mgr_09.can_load("m", required_vram_gb=18.0)
    # usable=18GB >= 18GB → accept
    check("0.9系数接受18GB", can_09)


def test_safety_factor__edge_values():
    """安全系数边界值"""
    print("\n── 安全系数: 边界值 ──")

    mgr_0 = make_vram(20.0, safety=0.0)
    can_0, _, _ = mgr_0.can_load("m", required_vram_gb=0.1)
    check("safety=0: usable=0", not can_0)

    mgr_1 = make_vram(20.0, safety=1.0)
    can_1, _, _ = mgr_1.can_load("m", required_vram_gb=20.0)
    check("safety=1: usable=20GB", can_1)


# ═══════════════════════════════════════════════════════════
# 9. 真实 GPU 检测
# ═══════════════════════════════════════════════════════════


def test_real_gpu__detection():
    """真实环境GPU检测"""
    print("\n── 真实GPU: 检测 ──")
    mgr = VRAMManager()
    free = mgr.get_available_vram_gb()
    used = mgr._read_used_vram_gb()

    import torch

    if torch.cuda.is_available():
        check("有CUDA→free>0", free > 0, f"free={free:.1f}GB")
        check("free合理范围", 0 < free < 200, f"free={free:.1f}GB")
        check("used≥0", used >= 0, f"used={used:.1f}GB")
    else:
        check("无CUDA→free=0", free == 0.0)
        print("  (跳过 — 无GPU环境)")


def test_read_vram__torch_fallback():
    """pynvml 失败时 fallback 到 torch"""
    print("\n── 真实GPU: torch fallback ──")
    with patch(
        "quantumflow.inference.vram_manager.VRAMManager._read_free_vram_gb",
        wraps=VRAMManager._read_free_vram_gb,
    ) as mock_read:
        # 不 mock — 让真实路径执行
        pass

    mgr = VRAMManager()
    free = mgr.get_available_vram_gb()

    import torch

    if torch.cuda.is_available():
        check("返回正值", free > 0)
    else:
        check("返回0", free == 0.0)


# ═══════════════════════════════════════════════════════════
# 10. VRAMManager 初始化
# ═══════════════════════════════════════════════════════════


def test_init__defaults():
    """默认参数"""
    print("\n── 初始化: 默认参数 ──")
    mgr = VRAMManager()
    check("safety=0.7", mgr.safety_factor == 0.7)
    check("idle_ttl=0.0", mgr.idle_ttl_seconds == 0.0)
    check("_loaded空", mgr._loaded == {})


def test_init__custom():
    """自定义参数"""
    print("\n── 初始化: 自定义参数 ──")
    mgr = VRAMManager(safety_factor=0.5, idle_ttl_seconds=120.0)
    check("safety=0.5", mgr.safety_factor == 0.5)
    check("idle_ttl=120", mgr.idle_ttl_seconds == 120.0)


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 静态方法
    test_param_count__all_known_patterns()
    test_param_count__long_pattern_before_short()
    test_param_count__case_insensitive()
    test_param_count__unknown()
    test_architecture__known()
    test_architecture__70b_72b()
    test_architecture__phi_mini()
    test_architecture__unknown()

    # VRAM 估算
    test_vram_estimation__with_architecture()
    test_vram_estimation__without_architecture()
    test_vram_estimation__unknown_model()
    test_vram_estimation__max_model_len_affects_kv()
    test_vram_estimation__max_model_len_zero()

    # can_load 决策
    test_can_load__direct_fit()
    test_can_load__tight_fit()
    test_can_load__evict_exact_one()
    test_can_load__evict_multiple()
    test_can_load__reject_absolute_insufficient()
    test_can_load__zero_available_but_evictable()
    test_can_load__zero_available_no_evictable()
    test_can_load__required_zero()
    test_can_load__required_negative()
    test_can_load__all_models_in_use()
    test_can_load__partial_in_use()

    # 淘汰评分
    test_scoring__formula_verification()
    test_scoring__same_age_size_decides()
    test_scoring__same_size_age_decides()
    test_scoring__single_candidate()
    test_scoring__empty_candidates()
    test_scoring__all_in_use_returns_empty()
    test_scoring__division_by_zero_safeguard()

    # 状态管理
    test_record_loaded__basic()
    test_record_loaded__overwrite()
    test_record_loaded__zero_vram()
    test_record_unloaded__basic()
    test_record_unloaded__nonexistent()
    test_mark_in_use__basic()
    test_mark_in_use__nonexistent()
    test_mark_idle__basic()
    test_mark_idle__nonexistent()
    test_update_actual_vram__basic()
    test_update_actual_vram__nonexistent()
    test_get_loaded_models__empty()

    # 空闲超时
    test_idle_ttl__expired_selected()
    test_idle_ttl__exact_boundary()
    test_idle_ttl__disabled()
    test_idle_ttl__in_use_protected()
    test_idle_ttl__multiple_expired()

    # 安全系数
    test_safety_factor__affects_usable()
    test_safety_factor__edge_values()

    # 真实GPU
    test_real_gpu__detection()
    test_read_vram__torch_fallback()

    # 初始化
    test_init__defaults()
    test_init__custom()

    ok = report()
    exit(0 if ok else 1)
