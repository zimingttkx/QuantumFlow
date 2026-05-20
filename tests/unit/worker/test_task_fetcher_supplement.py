"""TaskFetcher 补充测试 — 覆盖现有测试未触达的代码路径

目标覆盖缺失行:
- 118-122: stop() 中 fetch_task cancel + CancelledError
- 126-133: stop() 中等待 active_tasks (含 TimeoutError)
- 151-162: _fetch_loop 中的槽位检查和 dequeue 逻辑
- 166-168: _fetch_loop 异常处理
- 202: _execute_task 的 default_handler 分支 (无注册 handler)
- 234-235: _default_handler 无引擎
- 319: _on_task_done 的任务异常日志
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantumflow.inference.engine import InferenceResult
from quantumflow.storage.redis_queue import QueuedRequest
from quantumflow.worker.task_fetcher import TaskFetcher, TaskFetcherConfig


def _make_queued_request(
    request_id: str = "req-001",
    model_name: str = "test-model",
    prompt: str = "Hello",
    priority: int = 5,
    metadata: dict | None = None,
) -> QueuedRequest:
    """辅助: 创建 QueuedRequest"""
    from datetime import datetime

    return QueuedRequest(
        request_id=request_id,
        model_name=model_name,
        prompt=prompt,
        priority=priority,
        created_at=datetime.now(),
        metadata=metadata or {},
    )


def _make_fetcher(max_retries=3, active_tasks=10, poll_interval_ms=100, batch_size=1):
    """辅助: 创建 TaskFetcher"""
    config = TaskFetcherConfig(
        node_id="test-node",
        max_retries=max_retries,
        active_tasks=active_tasks,
        poll_interval_ms=poll_interval_ms,
        batch_size=batch_size,
    )
    return TaskFetcher(config=config)


# ═══════════════════════════════════════════════════════════════════════════════
# stop() — fetch_task cancel + CancelledError (lines 118-122)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStopFetchTaskCancel:
    """stop() 中取消 fetch_task 的逻辑"""

    @pytest.mark.asyncio
    async def test_stop_cancels_fetch_task(self):
        """lines 117-122: stop 取消 fetch_task 并处理 CancelledError"""
        fetcher = _make_fetcher()
        fetcher._running = True
        fetcher._active_tasks = {}

        # 创建真实的 asyncio.Task 以便 cancel
        async def dummy_loop():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise

        fetcher._fetch_task = asyncio.create_task(dummy_loop())
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.disconnect = AsyncMock()

        await fetcher.stop()

        assert fetcher._running is False
        assert fetcher._fetch_task.cancelled() or fetcher._fetch_task.done()

    @pytest.mark.asyncio
    async def test_stop_no_fetch_task(self):
        """无 fetch_task 时 stop 正常执行"""
        fetcher = _make_fetcher()
        fetcher._running = True
        fetcher._fetch_task = None
        fetcher._active_tasks = {}
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.disconnect = AsyncMock()

        await fetcher.stop()
        assert fetcher._running is False


# ═══════════════════════════════════════════════════════════════════════════════
# stop() — 等待 active_tasks (lines 126-133)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStopActiveTasksWait:
    """stop() 等待活跃任务完成的逻辑"""

    @pytest.mark.asyncio
    async def test_stop_waits_for_active_tasks(self):
        """lines 126-131: 等待 active_tasks 完成"""
        fetcher = _make_fetcher()
        fetcher._running = True

        # 创建一个快速完成的任务
        async def fast_task():
            return True

        fetcher._active_tasks = {
            "task-1": asyncio.create_task(fast_task()),
        }
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.disconnect = AsyncMock()

        await fetcher.stop()

        assert fetcher._running is False

    @pytest.mark.asyncio
    async def test_stop_active_tasks_timeout(self):
        """lines 132-133: 等待 active_tasks 超时 (TimeoutError)"""
        fetcher = _make_fetcher()
        fetcher._running = True

        # 创建一个永不完成的任务
        async def never_ending():
            await asyncio.sleep(999)

        fetcher._active_tasks = {
            "slow-task": asyncio.create_task(never_ending()),
        }
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.disconnect = AsyncMock()

        # 使用很短的超时来触发 TimeoutError
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            await fetcher.stop()

        assert fetcher._running is False

    @pytest.mark.asyncio
    async def test_stop_no_active_tasks(self):
        """无活跃任务时 stop 跳过等待逻辑"""
        fetcher = _make_fetcher()
        fetcher._running = True
        fetcher._active_tasks = {}
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.disconnect = AsyncMock()

        await fetcher.stop()
        assert fetcher._running is False


# ═══════════════════════════════════════════════════════════════════════════════
# _fetch_loop — 内部逻辑 (lines 151-162, 166-168)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchLoop:
    """_fetch_loop 的详细逻辑"""

    @pytest.mark.asyncio
    async def test_fetch_loop_at_capacity_skips_dequeue(self):
        """lines 151-152: 达到容量上限时跳过 dequeue"""
        fetcher = _make_fetcher(active_tasks=2)
        fetcher._running = True

        # 塞满活跃任务
        fetcher._active_tasks = {
            "task-1": AsyncMock(),
            "task-2": AsyncMock(),
        }

        mock_queue = AsyncMock()
        mock_queue.dequeue = AsyncMock()
        fetcher._redis_queue = mock_queue

        # 运行一次循环迭代，让它退出
        iteration = 0

        async def controlled_sleep(duration):
            nonlocal iteration
            iteration += 1
            if iteration >= 3:
                fetcher._running = False

        with patch("asyncio.sleep", controlled_sleep):
            await fetcher._fetch_loop()

        # dequeue 不应被调用，因为容量已满
        assert not mock_queue.dequeue.called

    @pytest.mark.asyncio
    async def test_fetch_loop_dequeues_task(self):
        """lines 155-162: 从 Redis 队列获取并处理任务"""
        fetcher = _make_fetcher(active_tasks=10, batch_size=2)
        fetcher._running = True
        fetcher._active_tasks = {}

        mock_queue = AsyncMock()
        mock_queue.dequeue = AsyncMock(return_value=_make_queued_request("req-deq"))
        mock_queue.store_result = AsyncMock()
        mock_queue.requeue = AsyncMock()
        fetcher._redis_queue = mock_queue

        # 注册成功 handler
        async def ok_handler(r):
            return True
        fetcher.register_handler("test-model", ok_handler)

        # 运行一次拉取后停止
        iteration = 0

        async def controlled_sleep(duration):
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                fetcher._running = False

        with patch("asyncio.sleep", controlled_sleep):
            await fetcher._fetch_loop()

        assert mock_queue.dequeue.called

    @pytest.mark.asyncio
    async def test_fetch_loop_dequeue_returns_none(self):
        """dequeue 返回 None 时继续循环"""
        fetcher = _make_fetcher(poll_interval_ms=10)
        fetcher._running = True
        fetcher._active_tasks = {}

        mock_queue = AsyncMock()
        mock_queue.dequeue = AsyncMock(return_value=None)
        fetcher._redis_queue = mock_queue

        iteration = 0

        async def controlled_sleep(duration):
            nonlocal iteration
            iteration += 1
            if iteration >= 3:
                fetcher._running = False

        with patch("asyncio.sleep", controlled_sleep):
            await fetcher._fetch_loop()

    @pytest.mark.asyncio
    async def test_fetch_loop_no_redis_queue(self):
        """无 redis_queue 时不尝试 dequeue"""
        fetcher = _make_fetcher()
        fetcher._running = True
        fetcher._redis_queue = None

        iteration = 0

        async def controlled_sleep(duration):
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                fetcher._running = False

        with patch("asyncio.sleep", controlled_sleep):
            await fetcher._fetch_loop()

    @pytest.mark.asyncio
    async def test_fetch_loop_handles_exception(self):
        """lines 166-168: fetch_loop 中的异常被捕获"""
        fetcher = _make_fetcher(poll_interval_ms=10)
        fetcher._running = True
        fetcher._active_tasks = {}

        # redis_queue.dequeue 抛异常
        mock_queue = AsyncMock()
        mock_queue.dequeue = AsyncMock(side_effect=RuntimeError("Redis connection lost"))
        fetcher._redis_queue = mock_queue

        iteration = 0

        async def controlled_sleep(duration):
            nonlocal iteration
            iteration += 1
            if iteration >= 3:
                fetcher._running = False

        with patch("asyncio.sleep", controlled_sleep):
            await fetcher._fetch_loop()

        # 不应崩溃，循环应继续

    @pytest.mark.asyncio
    async def test_fetch_loop_stops_when_not_running(self):
        """_running=False 时循环不执行"""
        fetcher = _make_fetcher()
        fetcher._running = False

        await fetcher._fetch_loop()
        # 应立即退出

    @pytest.mark.asyncio
    async def test_fetch_loop_batch_size_multiple(self):
        """batch_size > 1 时拉取多个任务"""
        fetcher = _make_fetcher(active_tasks=10, batch_size=3)
        fetcher._running = True
        fetcher._active_tasks = {}

        call_count = 0

        async def dequeue_side_effect(timeout=1):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                fetcher._running = False
            return _make_queued_request(f"req-{call_count}")

        mock_queue = AsyncMock()
        mock_queue.dequeue = dequeue_side_effect
        mock_queue.store_result = AsyncMock()
        mock_queue.requeue = AsyncMock()
        fetcher._redis_queue = mock_queue

        async def ok_handler(r):
            return True
        fetcher.register_handler("test-model", ok_handler)

        async def mock_sleep(duration):
            pass

        with patch("asyncio.sleep", mock_sleep):
            await fetcher._fetch_loop()

        assert call_count >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# _execute_task — default_handler 分支 (line 202)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskDefaultHandler:
    """_execute_task 使用默认处理器的场景"""

    @pytest.mark.asyncio
    async def test_execute_task_uses_default_handler_when_no_registered(self):
        """line 202: 无注册 handler 时使用默认处理器"""
        fetcher = _make_fetcher()
        engine = AsyncMock()
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-dh", outputs=["default output"],
                prompt_tokens=1, completion_tokens=2,
                latency_ms=10, finish_reason="stop",
            )
        ])
        fetcher.engine = engine

        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        request = _make_queued_request(request_id="req-dh", prompt="Hello")

        result = await fetcher._execute_task(request)

        assert result is True
        # 验证调用了 default_handler (通过 engine.generate)
        engine.generate.assert_called_once()
        assert fetcher.stats["tasks_completed"] == 1

    @pytest.mark.asyncio
    async def test_execute_task_default_handler_model_not_loaded(self):
        """默认处理器中模型未加载 → 失败 + retry"""
        fetcher = _make_fetcher()
        engine = AsyncMock()
        engine.is_model_loaded = AsyncMock(return_value=False)
        fetcher.engine = engine

        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        request = _make_queued_request(request_id="req-noload")

        result = await fetcher._execute_task(request)

        assert result is False
        assert fetcher.stats["tasks_failed"] == 1

    @pytest.mark.asyncio
    async def test_execute_task_default_handler_no_engine(self):
        """默认处理器无引擎 → 失败 (lines 234-235)"""
        fetcher = _make_fetcher()
        fetcher.engine = None
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        request = _make_queued_request(request_id="req-noeng")

        result = await fetcher._execute_task(request)

        assert result is False
        assert fetcher.stats["tasks_failed"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# _on_task_done — 异常任务日志 (line 319)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOnTaskDone:
    """_on_task_done 回调逻辑"""

    @pytest.mark.asyncio
    async def test_on_task_done_removes_from_active_tasks(self):
        """任务完成时从 active_tasks 移除"""
        fetcher = _make_fetcher()

        async def simple_task():
            return True

        task = asyncio.create_task(simple_task())
        fetcher._active_tasks["req-001"] = task

        await task  # 等待完成

        fetcher._on_task_done("req-001", task)
        assert "req-001" not in fetcher._active_tasks

    @pytest.mark.asyncio
    async def test_on_task_done_with_task_exception(self):
        """line 319: 任务抛异常时记录日志"""
        fetcher = _make_fetcher()

        async def failing_task():
            raise ValueError("task error")

        task = asyncio.create_task(failing_task())
        fetcher._active_tasks["req-fail"] = task

        # 等待任务完成（含异常）
        await asyncio.sleep(0.05)

        # 回调应记录异常但不抛出新异常
        fetcher._on_task_done("req-fail", task)
        assert "req-fail" not in fetcher._active_tasks
        assert task.exception() is not None

    @pytest.mark.asyncio
    async def test_on_task_done_unknown_request_id(self):
        """对不存在的 request_id 回调不抛异常"""
        fetcher = _make_fetcher()

        async def simple_task():
            return True

        task = asyncio.create_task(simple_task())
        await task

        # request_id 不在 active_tasks 中
        fetcher._on_task_done("unknown-id", task)
        # 不抛异常


# ═══════════════════════════════════════════════════════════════════════════════
# _default_handler — 无引擎 (lines 234-235 独立测试)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefaultHandlerNoEngine:
    """_default_handler 无引擎的独立测试"""

    @pytest.mark.asyncio
    async def test_default_handler_no_engine_returns_false(self):
        """lines 234-235: 无引擎时返回 False 并记录错误"""
        fetcher = _make_fetcher()
        fetcher.engine = None
        request = _make_queued_request(request_id="req-ne")

        result = await fetcher._default_handler(request)

        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# _process_task — done_callback 验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestProcessTaskCallback:
    """_process_task 添加的完成回调"""

    @pytest.mark.asyncio
    async def test_process_task_adds_done_callback(self):
        """_process_task 为任务添加完成回调"""
        fetcher = _make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        async def handler(r):
            return True
        fetcher.register_handler("test-model", handler)

        request = _make_queued_request()

        await fetcher._process_task(request)
        await asyncio.sleep(0.05)

        # 任务应已完成并从 active_tasks 移除
        assert request.request_id not in fetcher._active_tasks


# ═══════════════════════════════════════════════════════════════════════════════
# _retry_task — 无 redis_queue 时 (line 284-285)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetryNoRedis:
    """_retry_task 无 redis_queue 场景"""

    @pytest.mark.asyncio
    async def test_retry_without_redis_queue(self):
        """lines 284-285: 无 redis_queue 时 _retry_task 直接返回"""
        fetcher = _make_fetcher()
        fetcher._redis_queue = None

        request = _make_queued_request(metadata={"retry_count": 0})
        await fetcher._retry_task(request)
        # 不抛异常，不执行任何操作


# ═══════════════════════════════════════════════════════════════════════════════
# _execute_task — 注册 handler 路径验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskRegisteredHandler:
    """_execute_task 使用注册 handler"""

    @pytest.mark.asyncio
    async def test_execute_task_with_registered_handler_success(self):
        """注册 handler 成功处理"""
        fetcher = _make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        async def custom_handler(r):
            assert r.request_id == "req-custom"
            return True
        fetcher.register_handler("custom-model", custom_handler)

        request = _make_queued_request(request_id="req-custom", model_name="custom-model")

        result = await fetcher._execute_task(request)

        assert result is True
        assert fetcher.stats["tasks_completed"] == 1
        fetcher._redis_queue.store_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task_with_registered_handler_failure(self):
        """注册 handler 失败处理"""
        fetcher = _make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        async def bad_handler(r):
            return False
        fetcher.register_handler("bad-model", bad_handler)

        request = _make_queued_request(request_id="req-bad", model_name="bad-model")

        result = await fetcher._execute_task(request)

        assert result is False
        assert fetcher.stats["tasks_failed"] == 1
        fetcher._redis_queue.requeue.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task_with_wildcard_handler(self):
        """通配符 handler 兜底"""
        fetcher = _make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        async def catch_all(r):
            return True
        fetcher.register_handler("*", catch_all)

        request = _make_queued_request(request_id="req-any", model_name="unknown-model")

        result = await fetcher._execute_task(request)

        assert result is True
        assert fetcher.stats["tasks_completed"] == 1

    @pytest.mark.asyncio
    async def test_execute_task_registered_handler_raises(self):
        """注册 handler 抛异常 → 失败 + retry"""
        fetcher = _make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        async def crashy(r):
            raise RuntimeError("handler bug")

        fetcher.register_handler("crashy-model", crashy)

        request = _make_queued_request(request_id="req-crash", model_name="crashy-model")

        result = await fetcher._execute_task(request)

        assert result is False
        assert fetcher.stats["tasks_failed"] == 1
        fetcher._redis_queue.requeue.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# start() — 重复启动 (line 91-92)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStartAlreadyRunning:
    """start 重复调用"""

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """_running=True 时 start 直接返回"""
        fetcher = _make_fetcher()
        fetcher._running = True

        await fetcher.start()
        # 无副作用


# ═══════════════════════════════════════════════════════════════════════════════
# _execute_task — 无 redis_queue 时的 store_result 调用保护
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskNoRedisQueue:
    """_execute_task 无 redis_queue 的场景"""

    @pytest.mark.asyncio
    async def test_execute_task_success_no_redis_queue(self):
        """无 redis_queue 时成功执行不崩溃"""
        fetcher = _make_fetcher()
        fetcher._redis_queue = None

        async def ok_handler(r):
            return True
        fetcher.register_handler("test-model", ok_handler)

        request = _make_queued_request()
        result = await fetcher._execute_task(request)

        assert result is True
        assert fetcher.stats["tasks_completed"] == 1
        # store_result 不会被调用 (redis_queue 为 None)

    @pytest.mark.asyncio
    async def test_execute_task_failure_no_redis_queue(self):
        """无 redis_queue 时失败执行不崩溃"""
        fetcher = _make_fetcher()
        fetcher._redis_queue = None

        async def fail_handler(r):
            return False
        fetcher.register_handler("test-model", fail_handler)

        request = _make_queued_request()
        result = await fetcher._execute_task(request)

        assert result is False
        assert fetcher.stats["tasks_failed"] == 1
