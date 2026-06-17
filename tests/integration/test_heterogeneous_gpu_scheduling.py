"""异构 GPU + 多模型端到端调度集成测试骨架（活文档）

说明
----
本文件**不直接运行**，而是作为"活文档"描述 QuantumFlow 修复完成后的
期望行为。每个测试对应调度算法、GPU 拓扑、容错链路中的一个关键不变量。

如果你修改了：
- ``quantumflow/scheduler/strategy/gang.py`` / ``pack.py`` / ``adaptive.py``
- ``quantumflow/scheduler/scheduler.py``
- ``quantumflow/cluster/manager.py``
- ``quantumflow/inference/backends/{vllm,sglang,tgi,huggingface}.py``
- ``quantumflow/failover/replica_manager.py``
- ``quantumflow/scheduler/distributed.py``

请回来检查本文件的断言是否仍然成立。任何一个不变量被破坏都意味着回归。

运行方式
--------
需要真实集群时按名字运行：
    pytest tests/integration/test_heterogeneous_gpu_scheduling.py --collect-only

本文件不依赖外部集群，全部用 ``pytest.skip()`` 标记为 living document。
仅作 ``git grep`` 友好的行为规范。
"""

from __future__ import annotations

import pytest


# ────────────────────────────────────────────────────────────────────────
# 场景 1: 异构 GPU 亲和度
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document — 需要真实异构集群才能执行")
async def test_70b_model_lands_on_h100_node():
    """72B AWQ 模型必须优先选 H100 节点，避开 RTX 4090。

    断言要点：
    - 集群中有 3 个节点：A（8×H100）, B（8×A100）, C（8×RTX 4090 24GB）
    - 请求 model_id=llama-3-70b-awq, recommended_tensor_parallel=8
    - 调度结果必须落在节点 A（hopper 家族 + 80GB 显存）
    - 若仅有 B / C 可用，必须返回 NoSchedulableError 而非强派到 B/C
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_consumer_gpu_rejected_for_large_model():
    """consumer（RTX 4090）模型不能承接 > 13B 模型。

    断言要点：
    - RTX 4090（24GB, model_family=consumer）接到 70B 请求
    - ``can_fit_model()`` 必须返回 False
    - ``can_serve_model()`` 因为 family 不匹配返回 False
    - 调度器不进入 candidate 列表
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_preferred_gpu_families_respected():
    """请求声明 preferred_gpu_families=["hopper"] 时，必须只挑 H100。

    断言要点：
    - 集群中同时存在 A100 与 H100
    - 请求带 preferred_gpu_families=["hopper"]
    - 选中的所有 GPU 都来自 H100 节点
    - 不允许 fallback 到 A100（除非无可用 H100，此时记录 warning）
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 2: 模型亲和度（已在 worker 上加载）
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_7b_model_lands_on_worker_that_loaded_it():
    """Qwen-7B 请求只派到 loaded_models 含 Qwen-7B 的 worker。

    断言要点：
    - Worker A 已加载 Qwen-7B（loaded_models=["qwen-7b"]）
    - Worker B 已加载 Llama-3-8B
    - Worker C 未加载任何模型
    - Qwen-7B 请求被派到 Worker A
    - 不能落到 Worker B 或 C
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_unloaded_model_falls_back_to_replica_manager():
    """无 worker 加载目标模型时，请求交给 ReplicaManager 拉取。

    断言要点：
    - 集群中无 worker 加载 Mistral-7B
    - 请求 model_id=mistral-7b
    - 调度器不直接 dispatch，而是标记 ``requires_loading=True``
    - 触发 ``ReplicaManager.copy_model()``（使用 CopyStrategy）
    - 加载完成后调度器重新评估并 dispatch
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 3: 故障隔离
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_worker_failure_quarantines_node():
    """连续 3 次失败后节点被隔离 60 秒，调度器不再派请求。

    断言要点：
    - 节点 N 在 10s 内连续 3 次推理失败
    - ``_quarantined_nodes[N]`` 必须存在，过期时间 = now + 60s
    - 接下来的请求候选节点列表中不再包含 N
    - 60s 后 N 自动恢复，重新进入候选
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_worker_recovery_releases_reservation():
    """Worker 重启成功后保留的 GPU 必须被释放。

    断言要点：
    - Worker N 之前 ``reserved_gpus={0,1,2,3}``, ``reserved_memory_bytes=80GB``
    - Worker 心跳恢复（health_check 成功）
    - reserved_* 字段被清空（``release_reservation`` 调用）
    - 下个调度周期 N 重新可用
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_endpoint_circuit_breaker_trips_after_failures():
    """单个 endpoint 失败次数超过阈值后被熔断。

    断言要点：
    - ``WorkerEndpoint.failure_count`` 累计 >= 5
    - ``can_serve()`` 返回 False
    - ``record_success()`` 重置 failure_count
    - ``WorkerClient.inference()`` 不会路由到熔断的 endpoint
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 4: 租户配额
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_tenant_quota_decrements_on_completion():
    """请求完成后并发计数必须 decrement，避免泄漏。

    断言要点：
    - 初始 Redis ``tenant:q1:concurrent`` = 0
    - 提交 5 个请求，调度成功后计数 = 5
    - 5 个请求完成后计数 = 0（DECR 被调用 5 次）
    - 失败路径也必须 DECR（即使 inference 抛错）
    - 使用 Lua 脚本保证原子性，永不为负
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_tenant_quota_overflow_returns_error_not_partial():
    """请求超过 tenant 并发上限时，必须立即拒绝而不是排队。

    断言要点：
    - ``tenant:q1.max_concurrent`` = 4
    - 当前并发 = 4
    - 第 5 个请求进来时 INCR 返回 5 > 4，立即 DECR 回 4 并抛 ``QuotaExceededError``
    - 该请求不会进入 ``asyncio.PriorityQueue``
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 5: 张量并行与 NVLink 拓扑
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_gang_tp8_prefers_single_nvlink_domain():
    """TP=8 必须在同一 NVLink 域内选 8 卡，不能跨域。

    断言要点：
    - 节点 X 有 2 个 NVLink domain：domain-A 包含 {0,1,2,3}，domain-B 包含 {4,5,6,7}
    - 请求 ``recommended_tensor_parallel=8``
    - Gang 调度挑出 domain-A 的 4 卡 + domain-B 的 4 卡
      或者同一个 domain 的 8 卡（如果存在）
    - 不允许跨 domain 拼凑（除非同 domain 不够）
    - 跨域时记录 warning 日志
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_gang_cross_domain_fallback_logs_warning():
    """同 NVLink 域 GPU 不够时，记录警告并允许跨域。

    断言要点：
    - 节点 X 只有一个 domain 包含 4 个 GPU（max=4）
    - 请求 TP=8
    - 调度结果：4 卡 from domain-A + 4 卡 from 其他（PCIe）
    - 日志中出现 ``nvlink_cross_domain_fallback`` 事件
    - estimated_throughput 按 0.6 折扣
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 6: Pack vs Gang 路由
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_pack_skips_overloaded_node():
    """Pack 调度应跳过显存不足的节点，而不是返回 success。

    断言要点：
    - 节点 A 显存使用率 95%（几乎打满）
    - 节点 B 显存使用率 30%
    - 7B 请求 → Pack 必须选 B 而不是 A
    - 如果所有节点都 < threshold，返回 NoSchedulableError
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_pack_yields_to_gang_for_tp_requests():
    """Pack 看到 TP>=2 的请求时直接拒绝，交给 Gang。

    断言要点：
    - 节点 X 有 8 个空闲 A100
    - 请求 ``recommended_tensor_parallel=4``
    - ``Pack.can_handle()`` 返回 False
    - Adaptive 路由必须把请求送到 Gang 策略
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 7: 后端协议正确性
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_vllm_streaming_returns_first_token_within_500ms():
    """vLLM 真流式首 token 应 < 500ms（不是假流式的整段延迟）。

    断言要点：
    - 启动 vLLM AsyncLLMEngine（mock 也可以）
    - 调用 ``generate_stream()``
    - 第一个 yield 在 500ms 内返回
    - 后续 yield 间隔 ≈ 25ms / token
    - 与假流式（先生成完整 response 再 yield 整段）有明显差异
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_tgi_error_response_keeps_result_length_aligned():
    """TGI 出错时仍返回 len(prompts) 长度的 InferenceResult 列表。

    断言要点：
    - ``generate(prompts=[p1, p2, p3])``
    - TGI 服务返回 HTTP 500
    - 返回的 list 长度 == 3
    - 每条 ``finish_reason == "error"``
    - caller 用 ``zip(prompts, results)`` 不会 IndexError
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_sglang_concurrency_capped_by_semaphore():
    """SGLang 客户端通过 Semaphore 限制单实例并发，避免被打爆。

    断言要点：
    - max_concurrent=4
    - 同时投递 20 个请求
    - 任何时刻活跃请求数 <= 4
    - 其余请求排队等待
    - 所有请求最终都能完成（或受超时影响）
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_hf_use_cache_enabled_for_throughput():
    """HuggingFace 后端 KV cache 必须开启（use_cache=True）。

    断言要点：
    - 生成 100 tokens
    - 不开启 KV cache：每步 forward 都跑整段
    - 开启 KV cache：增量 forward，速度显著提升
    - 修复前 use_cache=False 是性能 bug，必须回归测试
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 8: 模型完整性
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_replica_checksum_is_deterministic():
    """ReplicaManager 的 SHA-256 必须确定性，不依赖时间戳。

    断言要点：
    - 对同一目录两次调用 ``_calculate_checksum()``
    - 返回的 hex digest 完全相同
    - 即使两次调用间隔 1 秒（修复前会包含时间戳 → 不一致）
    - 添加/删除任意文件后 checksum 必然变化
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_copy_strategy_injection_overrides_local_copy():
    """CopyStrategy 可注入 SSH/P2P 实现，绕过默认本地 copytree。

    断言要点：
    - ``ReplicaManager(set_copy_strategy=SSHCopyStrategy(...))``
    - ``_copy_model()`` 调用 SSH 策略而不是 shutil.copytree
    - 默认 ``LocalCopyStrategy`` 行为与原实现一致
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 9: 调度器循环行为
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_scheduler_backpressure_when_queue_empty():
    """调度循环在队列空时退避，避免空转消耗 CPU。

    断言要点：
    - 队列空时调度间隔从 100ms 增长到 1s
    - 新请求到达后间隔立即降回 100ms
    - 连续空转 5s 不会触发任何 I/O
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_multi_worker_dispatch_uses_gather_not_first():
    """多 worker 派发必须用 ``asyncio.gather`` 而不是只取第一个。

    断言要点：
    - 请求 recommended_tensor_parallel=4，需要 4 个 worker
    - Worker A / B / C / D 都匹配
    - 4 个 inference 调用必须并发执行
    - 总耗时 ≈ max(A, B, C, D) 而非 sum(A, B, C, D)
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 10: 自适应策略
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_adaptive_routes_large_models_to_gang():
    """模型 > 70B 或 TP>=2 时，Adaptive 必须走 Gang。

    断言要点：
    - 请求 parameter_count=72e9
    - Adaptive 评估后 selected_strategy="gang"
    - 不走 Pack（Pack 单节点不够 72B）
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_adaptive_routes_small_models_to_pack():
    """模型 <= 13B 且单卡可放下时，Adaptive 走 Pack。

    断言要点：
    - 请求 Qwen-7B（parameter_count=7e9）
    - Adaptive 评估后 selected_strategy="pack"
    - 选中节点内最大可用 GPU（pack 语义）
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 11: NVML 单例
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_nvml_manager_is_singleton():
    """NVMLManager 必须是单例，避免重复 init/shutdown。

    断言要点：
    - ``_get_nvml_manager()`` 两次调用返回同一对象
    - pynvml.nvmlInit() 只调用一次
    - ``shutdown()`` 后所有 handle 失效，必须 lazy re-init
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_nvml_stats_iterates_all_gpus_not_just_zero():
    """多卡 stats 必须迭代所有 GPU，不能只读 GPU 0。

    断言要点：
    - 节点有 8 张 GPU
    - ``get_stats()`` 返回的 dict 必须聚合 8 张卡的利用率
    - 而不是仅反映 GPU 0
    """


# ────────────────────────────────────────────────────────────────────────
# 场景 12: 配置与生命周期
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_model_registry_loads_from_external_json():
    """ModelRegistry 必须支持从 JSON 文件加载（而不是 hardcode）。

    断言要点：
    - ``registry.load_from_file("models.json")`` 加载 5 个模型
    - 每个模型包含 quantization, model_family, max_model_len 字段
    - ``estimate_memory_per_gpu(name, tp=2, quant="awq")`` 返回合理估算
    - 注册表可以在不修改代码的情况下扩展新模型
    """


@pytest.mark.integration
@pytest.mark.skip(reason="living document")
async def test_scheduler_stop_releases_all_reservations():
    """``Scheduler.stop()`` 必须释放所有节点的 GPU 预留。

    断言要点：
    - 调度过程中节点 N 累积 reserved_gpus={0,1,2,3}
    - ``await scheduler.stop()``
    - 所有节点的 reserved_gpus, reserved_memory_bytes 必须清零
    - 不留悬挂状态导致下次启动时误判
    """