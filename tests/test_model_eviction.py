"""模型缓存/淘汰策略 — 100%覆盖率业务逻辑测试

测试范围:
- 复合评分: age*0.7+size*0.3 精确计算验证
- 淘汰优先级: 同年龄/同大小/混合场景
- in_use保护: 推理中模型不出现在淘汰候选
- 空闲超时: TTL边界/禁用/in_use不淘汰
- can_load集成: 淘汰决策链/部分in_use/全in_use
- 状态转换: mark_in_use更新last_used_at/初始时间戳
- 综合场景: 多模型共存/顺序淘汰/partial evict

设计说明:
- 本文件所有断言走 `check()` 工具, 失败时立即抛 AssertionError,
  绝不在模块级 PASS/FAIL 全局变量里累积(那样会导致 pytest 看不到失败)。
- 这样 pytest 在该文件下会因 0 个被 pytest 发现的测试函数而报
  "no tests ran", 提示团队把这些 test_* 包装成真正的 pytest 函数。
- main() 入口保留以便独立运行: `python tests/test_model_eviction.py`。
"""

import sys
import time
import traceback
from pathlib import Path
from unittest.mock import MagicMock

# 支持两种运行方式:
#   - pytest: conftest 自动注入 PYTHONPATH, 这里不需要动作
#   - 直接 python tests/test_model_eviction.py: 需要把项目根加到 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantumflow.inference.vram_manager import LoadedModelInfo, VRAMManager


def check(name: str, condition: bool, detail: str = "") -> None:
    """断言: 失败时立即抛出 AssertionError, 带上名称与上下文。

    关键: 不能在全局变量里累积, 必须抛 — 这是让 pytest 看到失败的唯一方法。
    """
    if not condition:
        suffix = f" — {detail}" if detail else ""
        raise AssertionError(f"{name}{suffix}")


def make_mgr(free_vram=20.0, idle_ttl=0.0):
    mgr = VRAMManager(safety_factor=0.7, idle_ttl_seconds=idle_ttl)
    mgr._read_free_vram_gb = MagicMock(return_value=free_vram)
    mgr._read_used_vram_gb = MagicMock(return_value=0.0)
    return mgr


# ═══════════════════════════════════════════════════════════
# 1. 复合评分公式 — 精确验证
# ═══════════════════════════════════════════════════════════


def test_weighted_scoring_formula():
    """评分=age_norm*0.7+size_norm*0.3，高分优先淘汰。
    用精确构造的 last_used_at 验证公式计算值。"""
    mgr = make_mgr()
    now = time.time()

    # 构造3个模型，手动控制时间戳和大小
    mgr._loaded["A"] = LoadedModelInfo(
        model_name="A", estimated_vram_gb=10.0, last_used_at=now - 100
    )
    mgr._loaded["B"] = LoadedModelInfo(model_name="B", estimated_vram_gb=5.0, last_used_at=now - 50)
    mgr._loaded["C"] = LoadedModelInfo(model_name="C", estimated_vram_gb=1.0, last_used_at=now - 10)

    candidates = mgr._get_eviction_candidates()

    # 手动计算分数:
    # max_age=100, max_vram=10
    # A: age_norm=100/100=1.0, size_norm=10/10=1.0 → score=1.0*0.7+1.0*0.3=1.0
    # B: age_norm=50/100=0.5,  size_norm=5/10=0.5  → score=0.5*0.7+0.5*0.3=0.5
    # C: age_norm=10/100=0.1,  size_norm=1/10=0.1  → score=0.1*0.7+0.1*0.3=0.1
    # 淘汰顺序: A(1.0) > B(0.5) > C(0.1)

    check("3个候选", len(candidates) == 3)
    check(
        "A优先淘汰(score=1.0)", candidates[0].model_name == "A", f"first={candidates[0].model_name}"
    )
    check("B第二(score=0.5)", candidates[1].model_name == "B", f"second={candidates[1].model_name}")
    check("C最后(score=0.1)", candidates[2].model_name == "C", f"third={candidates[2].model_name}")


def test_scoring__age_dominates_when_size_similar():
    """大小相近时 → 年龄权重更大(70%)决定淘汰顺序"""
    mgr = make_mgr()
    now = time.time()

    mgr._loaded["old-small"] = LoadedModelInfo(
        model_name="old-small", estimated_vram_gb=2.0, last_used_at=now - 200
    )
    mgr._loaded["new-big"] = LoadedModelInfo(
        model_name="new-big", estimated_vram_gb=3.0, last_used_at=now - 1
    )

    candidates = mgr._get_eviction_candidates()

    # old-small: age_norm=1.0, size_norm=2/3≈0.67 → score=0.7+0.2=0.9
    # new-big:   age_norm=1/200≈0.005, size_norm=1.0 → score≈0.0035+0.3=0.3035
    # old-small 分更高 → 优先淘汰
    check(
        "old-small优先淘汰(年龄权重)",
        candidates[0].model_name == "old-small",
        f"first={candidates[0].model_name}",
    )
    check("new-big受保护", candidates[1].model_name == "new-big")


def test_scoring__size_breaks_tie():
    """年龄相同时 → 大小决定（30%权重）"""
    mgr = make_mgr()
    now = time.time()

    mgr._loaded["small"] = LoadedModelInfo(
        model_name="small", estimated_vram_gb=1.0, last_used_at=now - 50
    )
    mgr._loaded["large"] = LoadedModelInfo(
        model_name="large", estimated_vram_gb=10.0, last_used_at=now - 50
    )

    candidates = mgr._get_eviction_candidates()
    # 同年龄 → age_norm相同 → 大模型size_norm=1.0 > 小模型size_norm=0.1
    check(
        "large先淘汰(同年龄大者优先)",
        candidates[0].model_name == "large",
        f"first={candidates[0].model_name}",
    )


# ═══════════════════════════════════════════════════════════
# 2. in_use 保护
# ═══════════════════════════════════════════════════════════


def test_in_use_excluded_from_candidates():
    """正在推理的模型不在淘汰候选列表中"""
    mgr = make_mgr()

    mgr.record_loaded("busy", estimated_vram_gb=5.0)
    mgr.record_loaded("idle-1", estimated_vram_gb=1.0)
    mgr.record_loaded("idle-2", estimated_vram_gb=1.0)
    mgr.mark_in_use("busy")

    candidates = mgr._get_eviction_candidates()
    names = {c.model_name for c in candidates}
    check("busy不在候选", "busy" not in names)
    check("idle-1在候选", "idle-1" in names)
    check("idle-2在候选", "idle-2" in names)
    check("2个候选", len(candidates) == 2)


def test_all_in_use__empty_candidates():
    """所有模型都在推理 → 淘汰候选为空"""
    mgr = make_mgr()
    mgr.record_loaded("a", 5.0)
    mgr.record_loaded("b", 3.0)
    mgr.mark_in_use("a")
    mgr.mark_in_use("b")

    candidates = mgr._get_eviction_candidates()
    check("空候选", candidates == [])


def test_mark_idle_returns_to_candidates():
    """推理完成后模型回到可淘汰池"""
    mgr = make_mgr()
    mgr.record_loaded("m", 5.0)
    mgr.mark_in_use("m")
    check("推理中不在候选", len(mgr._get_eviction_candidates()) == 0)

    mgr.mark_idle("m")
    candidates = mgr._get_eviction_candidates()
    check("idle后在候选", len(candidates) == 1)
    check("候选是m", candidates[0].model_name == "m")


# ═══════════════════════════════════════════════════════════
# 3. 空闲超时淘汰
# ═══════════════════════════════════════════════════════════


def test_idle_ttl__selects_expired():
    """空闲超过 TTL 的模型被选中淘汰"""
    mgr = make_mgr(idle_ttl=60.0)
    now = time.time()

    mgr.record_loaded("active", 3.0)
    mgr.record_loaded("expired", 3.0)
    mgr._loaded["expired"].last_used_at = now - 120.0

    idle = mgr.get_idle_models_to_evict()
    check("expired在列", "expired" in idle)
    check("active不在列", "active" not in idle)
    check("仅1个", len(idle) == 1)


def test_idle_ttl__multiple_expired():
    """多个过期模型全部返回"""
    mgr = make_mgr(idle_ttl=60.0)
    now = time.time()

    mgr.record_loaded("e1", 2.0)
    mgr.record_loaded("e2", 3.0)
    mgr.record_loaded("fresh", 4.0)
    mgr._loaded["e1"].last_used_at = now - 300
    mgr._loaded["e2"].last_used_at = now - 120

    idle = mgr.get_idle_models_to_evict()
    check("2个过期", len(idle) == 2)
    check("e1在列", "e1" in idle)
    check("e2在列", "e2" in idle)
    check("fresh不在列", "fresh" not in idle)


def test_idle_ttl__exact_boundary():
    """TTL精确边界: 刚好超时 vs 远未超时

    边界值不要用 0.001s 这种微小时差 — 测试执行期间 time.time()
    的漂移就足以让 TTL 比较穿过临界。改用 5s 间隔, 既验证 "> TTL"
    严格大于语义, 又对执行耗时完全免疫。
    """
    mgr = make_mgr(idle_ttl=60.0)
    now = time.time()

    mgr.record_loaded("just-over", 3.0)
    mgr.record_loaded("well-under", 3.0)
    mgr._loaded["just-over"].last_used_at = now - 60.5   # 超过 TTL 0.5s
    mgr._loaded["well-under"].last_used_at = now - 30.0  # TTL 的一半

    idle = mgr.get_idle_models_to_evict()
    check("刚超时在列", "just-over" in idle)
    check("未超时不在列", "well-under" not in idle)


def test_idle_ttl__zero_disabled():
    """TTL=0 禁用空闲淘汰"""
    mgr = make_mgr(idle_ttl=0.0)
    mgr.record_loaded("ancient", 3.0)
    mgr._loaded["ancient"].last_used_at = time.time() - 999999

    idle = mgr.get_idle_models_to_evict()
    check("TTL=0返回[]", idle == [])


def test_idle_ttl__negative_disabled():
    """TTL<0 应等同于禁用"""
    mgr = make_mgr(idle_ttl=-1.0)
    mgr.record_loaded("old", 3.0)
    mgr._loaded["old"].last_used_at = time.time() - 999999

    idle = mgr.get_idle_models_to_evict()
    check("TTL<0返回[]", idle == [])


def test_idle_ttl__in_use_protected():
    """推理中模型即使过期也不淘汰"""
    mgr = make_mgr(idle_ttl=30.0)
    now = time.time()

    mgr.record_loaded("busy-expired", 3.0)
    mgr._loaded["busy-expired"].last_used_at = now - 120.0
    mgr.mark_in_use("busy-expired")

    idle = mgr.get_idle_models_to_evict()
    check("busy-expired不在列", "busy-expired" not in idle)


def test_idle_ttl__no_expired_empty():
    """无过期模型 → 空列表"""
    mgr = make_mgr(idle_ttl=60.0)
    mgr.record_loaded("fresh", 3.0)
    # 刚加载，未过期

    idle = mgr.get_idle_models_to_evict()
    check("无过期→[]", idle == [])


# ═══════════════════════════════════════════════════════════
# 4. can_load 集成淘汰
# ═══════════════════════════════════════════════════════════


def test_can_load__evicts_lru_idle():
    """加载新模型时淘汰最旧的idle模型"""
    mgr = make_mgr(free_vram=10.0)  # usable=7GB

    mgr.record_loaded("old-a", estimated_vram_gb=3.0)
    time.sleep(0.02)
    mgr.record_loaded("old-b", estimated_vram_gb=3.0)

    can, reason, evict = mgr.can_load("new-model", required_vram_gb=8.0)
    # shortage=8-7=1GB → old-a(3GB)足够
    check("can=True", can)
    check("淘汰old-a(LRU最旧)", evict == ["old-a"], f"got={evict}")


def test_can_load__skip_in_use():
    """淘汰时跳过推理中的模型"""
    mgr = make_mgr(free_vram=5.0)  # usable=3.5GB

    mgr.record_loaded("busy", estimated_vram_gb=3.0)
    mgr.record_loaded("idle", estimated_vram_gb=2.0)
    mgr.mark_in_use("busy")

    can, reason, evict = mgr.can_load("new", required_vram_gb=5.0)
    check("can=True", can)
    check("busy不在淘汰列表", "busy" not in evict)
    check("仅淘汰idle", evict == ["idle"], f"got={evict}")


def test_can_load__reject_when_busy_only():
    """唯一idle模型是busy → 无法淘汰 → 拒绝"""
    mgr = make_mgr(free_vram=3.0)  # usable=2.1GB

    mgr.record_loaded("busy-only", estimated_vram_gb=5.0)
    mgr.mark_in_use("busy-only")

    can, reason, evict = mgr.can_load("new", required_vram_gb=5.0)
    check("can=False", not can)
    check("evict=[]", evict == [])


def test_can_load__partial_evict():
    """shortage小时只淘汰必要模型，不淘汰多余模型"""
    mgr = make_mgr(free_vram=8.0)  # usable=5.6GB

    mgr.record_loaded("a-2g", estimated_vram_gb=2.0)
    time.sleep(0.01)
    mgr.record_loaded("b-2g", estimated_vram_gb=2.0)
    time.sleep(0.01)
    mgr.record_loaded("c-4g", estimated_vram_gb=4.0)

    can, reason, evict = mgr.can_load("new", required_vram_gb=7.0)
    # usable=5.6, shortage=7-5.6=1.4GB → a-2g(2GB >= 1.4)够了
    check("can=True", can)
    check("仅淘汰1个", len(evict) == 1, f"got={evict}")
    check("淘汰a-2g(LRU最旧)", evict[0] == "a-2g", f"evict={evict}")


def test_can_load__multiple_evictions_needed():
    """shortage较大 → 需淘汰多个模型（累计满足）"""
    mgr = make_mgr(free_vram=8.0)  # usable=5.6GB

    mgr.record_loaded("a-1g", estimated_vram_gb=1.0)
    time.sleep(0.01)
    mgr.record_loaded("b-1g", estimated_vram_gb=1.0)
    time.sleep(0.01)
    mgr.record_loaded("c-1g", estimated_vram_gb=1.0)
    time.sleep(0.01)
    mgr.record_loaded("d-3g", estimated_vram_gb=3.0)

    can, reason, evict = mgr.can_load("new", required_vram_gb=9.0)
    # usable=5.6, shortage=9-5.6=3.4GB
    # a(1): 1.0 < 3.4
    # a+b(2): 2.0 < 3.4
    # a+b+c(3): 3.0 < 3.4
    # a+b+c+d(6): 6.0 >= 3.4 → 淘汰4个
    check("can=True", can)
    check("淘汰4个", len(evict) == 4, f"got={len(evict)}: {evict}")


# ═══════════════════════════════════════════════════════════
# 5. 状态转换
# ═══════════════════════════════════════════════════════════


def test_mark_in_use_updates_timestamp():
    """mark_in_use 更新 last_used_at"""
    mgr = make_mgr()
    mgr.record_loaded("test", 3.0)

    old_ts = mgr._loaded["test"].last_used_at
    time.sleep(0.01)
    mgr.mark_in_use("test")
    new_ts = mgr._loaded["test"].last_used_at
    check("timestamp已更新", new_ts > old_ts)


def test_record_loaded__initial_timestamp():
    """加载时设置初始时间戳"""
    mgr = make_mgr()
    before = time.time()
    mgr.record_loaded("fresh", 3.0)
    after = time.time()

    ts = mgr._loaded["fresh"].last_used_at
    check(
        "时间戳在加载时间范围内",
        before <= ts <= after,
        f"before={before:.6f}, ts={ts:.6f}, after={after:.6f}",
    )


def test_record_loaded__sets_in_use_false():
    """新加载的模型 in_use=False"""
    mgr = make_mgr()
    mgr.record_loaded("m", 3.0)
    check("初始in_use=False", not mgr._loaded["m"].in_use)


# ═══════════════════════════════════════════════════════════
# 6. 综合场景
# ═══════════════════════════════════════════════════════════


def test_full_scenario__3_models_load_4th():
    """3模型共存，加载第4个触发淘汰 + 第5个无法加载"""
    mgr = make_mgr(free_vram=15.0)  # usable=10.5GB

    # 加载3模型
    mgr.record_loaded("model-a", estimated_vram_gb=4.0)
    time.sleep(0.01)
    mgr.record_loaded("model-b", estimated_vram_gb=3.0)
    time.sleep(0.01)
    mgr.record_loaded("model-c", estimated_vram_gb=3.0)

    # model-a 推理中
    mgr.mark_in_use("model-a")

    # 加载 model-d(6GB): usable=10.5 > 6 → 直接加载
    can, _, evict = mgr.can_load("model-d", required_vram_gb=6.0)
    check("VRAM充足直接加载", can and len(evict) == 0)

    # 加载 big-model(12GB): usable=10.5, shortage=1.5GB
    # idle: model-b(3GB,oldest), model-c(3GB)
    # model-b(3GB) >= 1.5 → 仅淘汰model-b
    can2, _, evict2 = mgr.can_load("big-model", required_vram_gb=12.0)
    check("big-model需要淘汰", can2)
    check("model-a(busy)不在淘汰", "model-a" not in evict2)
    check("淘汰model-b(LRU最旧idle)", evict2 == ["model-b"], f"evict={evict2}")

    # 加载 huge-model(30GB): usable(10.5)+idle(3+3)=16.5 < 30 → 拒绝
    can3, _, evict3 = mgr.can_load("huge-model", required_vram_gb=30.0)
    check("huge-model拒绝", not can3)
    check("evict=[]", evict3 == [])


def test_scenario__sequential_eviction_with_mark_idle():
    """推理完成→idle→被后续模型淘汰"""
    mgr = make_mgr(free_vram=8.0)  # usable=5.6GB

    mgr.record_loaded("worker", estimated_vram_gb=5.0)
    mgr.record_loaded("standby", estimated_vram_gb=2.0)

    # worker推理中 → 不可淘汰
    mgr.mark_in_use("worker")
    can1, _, evict1 = mgr.can_load("new1", required_vram_gb=6.0)
    # usable=5.6, shortage=0.4, standby(2GB)够
    check("淘汰standby(worker in_use)", can1 and evict1 == ["standby"], f"evict={evict1}")

    # worker推理完成
    mgr.mark_idle("worker")

    # 模拟EngineManager实际淘汰standby（can_load只建议，EngineManager执行）
    mgr.record_unloaded("standby")

    # 现在只剩worker(idle,5GB)
    can2, _, evict2 = mgr.can_load("new2", required_vram_gb=6.0)
    # usable=5.6, shortage=0.4, worker(5GB)够
    check("standby已淘汰后淘汰worker", can2 and evict2 == ["worker"], f"evict={evict2}")


def test_scenario__no_models_loaded():
    """无已加载模型时直接拒绝大模型"""
    mgr = make_mgr(free_vram=4.0)  # usable=2.8GB

    can, reason, evict = mgr.can_load("big", required_vram_gb=10.0)
    check("can=False(无可淘汰)", not can)
    check("evict=[]", evict == [])
    check("reason含'不足'", "不足" in reason)


def test_scenario__small_vram_many_small_models():
    """VRAM紧张时精确淘汰多个小模型"""
    mgr = make_mgr(free_vram=5.0)  # usable=3.5GB

    # 加载10个小模型
    for i in range(10):
        mgr.record_loaded(f"m{i}", estimated_vram_gb=1.0)
        time.sleep(0.001)

    can, reason, evict = mgr.can_load("medium", required_vram_gb=8.0)
    # usable=3.5, shortage=8-3.5=4.5GB → 需淘汰5个(5*1=5GB >= 4.5)
    check("淘汰5个", len(evict) == 5, f"got={len(evict)}: {evict}")
    # 前5个是最旧的(LRU顺序)
    check("最旧的5个", evict == ["m0", "m1", "m2", "m3", "m4"], f"got={evict}")


# ═══════════════════════════════════════════════════════════
# main: 独立运行入口 (跑完所有用例,任一失败 exit code 1)
# ═══════════════════════════════════════════════════════════

_TEST_FUNCS = [
    test_weighted_scoring_formula,
    test_scoring__age_dominates_when_size_similar,
    test_scoring__size_breaks_tie,
    test_in_use_excluded_from_candidates,
    test_all_in_use__empty_candidates,
    test_mark_idle_returns_to_candidates,
    test_idle_ttl__selects_expired,
    test_idle_ttl__multiple_expired,
    test_idle_ttl__exact_boundary,
    test_idle_ttl__zero_disabled,
    test_idle_ttl__negative_disabled,
    test_idle_ttl__in_use_protected,
    test_idle_ttl__no_expired_empty,
    test_can_load__evicts_lru_idle,
    test_can_load__skip_in_use,
    test_can_load__reject_when_busy_only,
    test_can_load__partial_evict,
    test_can_load__multiple_evictions_needed,
    test_mark_in_use_updates_timestamp,
    test_record_loaded__initial_timestamp,
    test_record_loaded__sets_in_use_false,
    test_full_scenario__3_models_load_4th,
    test_scenario__sequential_eviction_with_mark_idle,
    test_scenario__no_models_loaded,
    test_scenario__small_vram_many_small_models,
]


if __name__ == "__main__":
    failed = 0
    for fn in _TEST_FUNCS:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {fn.__name__} — {e}")
            traceback.print_exc()
    total = len(_TEST_FUNCS)
    print(f"\n{'='*60}")
    print(f"模型淘汰策略 测试报告: {total - failed}/{total} 通过, {failed} 失败")
    print(f"{'='*60}")
    exit(0 if failed == 0 else 1)

