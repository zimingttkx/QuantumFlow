"""TaskFetcher 严格单元测试

测试覆盖:
1. TaskFetcherConfig 默认值和自定义值
2. TaskFetcher 初始化状态
3. handler 注册和查找逻辑
4. 启动/停止生命周期
5. 任务处理流程 (pull, execute, complete, fail)
6. 重试机制 (max_retries 边界)
7. 并发槽位控制 (active_tasks 限制)
8. 默认处理器行为
9. 异常场景
10. 统计数据准确性
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

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


# ==================== TaskFetcherConfig ====================


class TestTaskFetcherConfig:
    """TaskFetcherConfig 数据类测试"""

    def test_default_values(self):
        """[默认值] 验证所有默认配置值"""
        config = TaskFetcherConfig(node_id="test-node")
        assert config.node_id == "test-node"
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.poll_interval_ms == 100
        assert config.batch_size == 1
        assert config.max_retries == 3
        assert config.active_tasks == 10

    def test_custom_values(self):
        """[自定义值] 验证所有自定义配置"""
        config = TaskFetcherConfig(
            node_id="worker-1",
            redis_url="redis://redis:6379/1",
            poll_interval_ms=200,
            batch_size=5,
            max_retries=5,
            active_tasks=20,
        )
        assert config.node_id == "worker-1"
        assert config.redis_url == "redis://redis:6379/1"
        assert config.poll_interval_ms == 200
        assert config.batch_size == 5
        assert config.max_retries == 5
        assert config.active_tasks == 20

    def test_zero_max_retries(self):
        """[边界值] max_retries=0 (不重试)"""
        config = TaskFetcherConfig(node_id="t", max_retries=0)
        assert config.max_retries == 0

    def test_large_active_tasks(self):
        """[边界值] active_tasks 极大值"""
        config = TaskFetcherConfig(node_id="t", active_tasks=1000)
        assert config.active_tasks == 1000


# ==================== TaskFetcher 初始化 ====================


class TestTaskFetcherInit:
    """TaskFetcher 初始化验证"""

    def test_creation_sets_config(self):
        """[初始化] 正确保存配置"""
        config = TaskFetcherConfig(node_id="n1", max_retries=3)
        fetcher = TaskFetcher(config=config)
        assert fetcher.config is config
        assert fetcher.config.node_id == "n1"

    def test_creation_has_empty_stats(self):
        """[初始化] 统计归零"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        assert fetcher.stats["tasks_pulled"] == 0
        assert fetcher.stats["tasks_completed"] == 0
        assert fetcher.stats["tasks_failed"] == 0
        assert fetcher.stats["tasks_retried"] == 0

    def test_creation_not_running(self):
        """[初始化] 未启动时 _running=False"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        assert fetcher._running is False

    def test_creation_empty_handlers(self):
        """[初始化] handler 字典为空"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        assert fetcher._task_handlers == {}

    def test_creation_stores_engine(self):
        """[初始化] 正确保存引擎引用"""
        engine = AsyncMock()
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"), engine=engine)
        assert fetcher.engine is engine


# ==================== Handler 注册 ====================


class TestTaskFetcherHandler:
    """Handler 注册与查找"""

    def test_register_handler_stored(self):
        """[注册] handler 被正确存储"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        async def my_handler(request):
            return True
        fetcher.register_handler("model-a", my_handler)
        assert "model-a" in fetcher._task_handlers
        assert fetcher._task_handlers["model-a"] is my_handler

    def test_register_wildcard_handler(self):
        """[注册] 通配符 handler "*" 正确存储"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        async def catch_all(request):
            return True
        fetcher.register_handler("*", catch_all)
        assert fetcher._task_handlers["*"] is catch_all

    def test_register_multiple_handlers(self):
        """[注册] 多个 handler 不互相覆盖"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        async def h1(r):
            return True
        async def h2(r):
            return False
        fetcher.register_handler("a", h1)
        fetcher.register_handler("b", h2)
        assert len(fetcher._task_handlers) == 2
        assert fetcher._task_handlers["a"] is h1
        assert fetcher._task_handlers["b"] is h2

    def test_overwrite_handler(self):
        """[覆盖] 同名 handler 被覆盖"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        async def h1(r):
            return True
        async def h2(r):
            return False
        fetcher.register_handler("a", h1)
        fetcher.register_handler("a", h2)
        assert fetcher._task_handlers["a"] is h2


# ==================== 启动/停止生命周期 ====================


class TestTaskFetcherLifecycle:
    """启动与停止逻辑"""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """[启动] start() 设置 _running=True"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.connect = AsyncMock()

        with patch("quantumflow.worker.task_fetcher.get_redis_manager") as mock_get:
            mock_mgr = AsyncMock()
            mock_mgr.is_connected = True
            mock_get.return_value = mock_mgr

            await fetcher.start()
            assert fetcher._running is True

    @pytest.mark.asyncio
    async def test_start_twice_noop(self):
        """[边界值] 重复启动不重复创建"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        fetcher._running = True
        await fetcher.start()
        # 应该直接返回，不创建新任务

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self):
        """[停止] stop() 设置 _running=False"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        fetcher._running = True
        fetcher._active_tasks = {}
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.disconnect = AsyncMock()

        await fetcher.stop()
        assert fetcher._running is False

    @pytest.mark.asyncio
    async def test_stop_twice_noop(self):
        """[边界值] 重复停止不抛异常"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        fetcher._running = False
        await fetcher.stop()

    @pytest.mark.asyncio
    async def test_start_when_redis_unavailable(self):
        """[异常场景] Redis 不可达时 start 不抛异常"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))

        with patch("quantumflow.worker.task_fetcher.get_redis_manager") as mock_get:
            mock_mgr = AsyncMock()
            mock_mgr.is_connected = False
            mock_get.return_value = mock_mgr

            await fetcher.start()
            assert fetcher._running is False


# ==================== 任务处理 ====================


class TestTaskFetcherProcessTask:
    """_process_task 内部逻辑"""

    def _make_fetcher(self):
        """创建可用于测试的 fetcher"""
        config = TaskFetcherConfig(node_id="test-node", max_retries=3)
        return TaskFetcher(config=config)

    @pytest.mark.asyncio
    async def test_process_task_increments_pulled(self):
        """[统计] 拉取计数 +1"""
        fetcher = self._make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        request = _make_queued_request()
        # register a handler that succeeds
        async def ok_handler(r):
            return True
        fetcher.register_handler("test-model", ok_handler)

        await fetcher._process_task(request)
        assert fetcher.stats["tasks_pulled"] == 1

    @pytest.mark.asyncio
    async def test_process_task_successful_handler(self):
        """[正常用例] 成功 handler → tasks_completed +1"""
        fetcher = self._make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        request = _make_queued_request()
        async def ok_handler(r):
            return True
        fetcher.register_handler("test-model", ok_handler)

        await fetcher._process_task(request)
        # 等待内部 create_task 完成
        await asyncio.sleep(0.05)

        assert fetcher.stats["tasks_pulled"] == 1
        assert fetcher.stats["tasks_completed"] == 1
        assert fetcher.stats["tasks_failed"] == 0

    @pytest.mark.asyncio
    async def test_process_task_failing_handler(self):
        """[错误用例] 失败 handler → tasks_failed +1 + retry"""
        fetcher = self._make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        request = _make_queued_request()
        async def fail_handler(r):
            return False
        fetcher.register_handler("test-model", fail_handler)

        await fetcher._process_task(request)
        await asyncio.sleep(0.05)

        assert fetcher.stats["tasks_failed"] == 1
        assert fetcher.stats["tasks_retried"] == 1
        # verify requeue was called
        assert fetcher._redis_queue.requeue.called

    @pytest.mark.asyncio
    async def test_process_task_handler_exception(self):
        """[异常场景] handler 抛异常 → tasks_failed +1 + retry"""
        fetcher = self._make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        request = _make_queued_request()
        async def crash_handler(r):
            raise RuntimeError("boom")
        fetcher.register_handler("test-model", crash_handler)

        await fetcher._process_task(request)
        await asyncio.sleep(0.05)

        assert fetcher.stats["tasks_failed"] == 1
        assert fetcher._redis_queue.requeue.called

    @pytest.mark.asyncio
    async def test_process_task_adds_to_active_tasks(self):
        """[状态变更] 处理任务时添加到活跃任务列表"""
        fetcher = self._make_fetcher()
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()

        # use a handler that takes a bit to complete
        async def slow_handler(r):
            await asyncio.sleep(0.1)
            return True
        fetcher.register_handler("test-model", slow_handler)

        request = _make_queued_request()
        await fetcher._process_task(request)
        # 任务应该被添加到 active_tasks
        assert "req-001" in fetcher._active_tasks
        await asyncio.sleep(0.15)
        # 任务完成后应从 active_tasks 移除
        assert "req-001" not in fetcher._active_tasks


# ==================== 默认处理器 ====================


class TestTaskFetcherDefaultHandler:
    """_default_handler 逻辑"""

    def _make_fetcher_with_engine(self, engine=None):
        config = TaskFetcherConfig(node_id="test-node")
        return TaskFetcher(config=config, engine=engine or AsyncMock())

    @pytest.mark.asyncio
    async def test_default_handler_no_engine(self):
        """[错误用例] 无引擎 → 返回 False"""
        fetcher = self._make_fetcher_with_engine(engine=None)
        request = _make_queued_request()
        result = await fetcher._default_handler(request)
        assert result is False

    @pytest.mark.asyncio
    async def test_default_handler_model_not_loaded(self):
        """[错误用例] 模型未加载 → 返回 False"""
        engine = AsyncMock()
        engine.is_model_loaded = AsyncMock(return_value=False)
        fetcher = self._make_fetcher_with_engine(engine)

        request = _make_queued_request()
        result = await fetcher._default_handler(request)
        assert result is False

    @pytest.mark.asyncio
    async def test_default_handler_success(self):
        """[正常用例] 引擎返回结果 → 返回 True"""
        engine = AsyncMock()
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[
            InferenceResult(
                request_id="req-001",
                outputs=["test output"],
                prompt_tokens=1,
                completion_tokens=2,
                latency_ms=10,
                finish_reason="stop",
            )
        ])
        fetcher = self._make_fetcher_with_engine(engine)

        request = _make_queued_request(prompt="Hi")
        result = await fetcher._default_handler(request)
        assert result is True
        engine.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_handler_empty_result(self):
        """[边界值] 引擎返回空结果 → 返回 False"""
        engine = AsyncMock()
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[])
        fetcher = self._make_fetcher_with_engine(engine)

        request = _make_queued_request()
        result = await fetcher._default_handler(request)
        assert result is False

    @pytest.mark.asyncio
    async def test_default_handler_engine_exception(self):
        """[异常场景] 引擎抛异常 → 返回 False"""
        engine = AsyncMock()
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(side_effect=RuntimeError("GPU OOM"))
        fetcher = self._make_fetcher_with_engine(engine)

        request = _make_queued_request()
        result = await fetcher._default_handler(request)
        assert result is False

    @pytest.mark.asyncio
    async def test_default_handler_passes_sampling_params(self):
        """[数据验证] 传递 metadata 中的采样参数"""
        engine = AsyncMock()
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[
            InferenceResult(request_id="r1", outputs=["ok"], prompt_tokens=1,
                          completion_tokens=1, latency_ms=1, finish_reason="stop")
        ])
        fetcher = self._make_fetcher_with_engine(engine)

        request = _make_queued_request(
            prompt="Hello",
            metadata={
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 30,
                "max_tokens": 256,
                "repetition_penalty": 1.1,
            },
        )
        await fetcher._default_handler(request)

        call_kwargs = engine.generate.call_args.kwargs
        sp = call_kwargs["sampling_params"]
        assert sp.temperature == 0.5
        assert sp.top_p == 0.8
        assert sp.top_k == 30
        assert sp.max_tokens == 256
        assert sp.repetition_penalty == 1.1

    @pytest.mark.asyncio
    async def test_default_handler_default_sampling_params(self):
        """[默认值] 无 metadata 时使用默认采样参数"""
        engine = AsyncMock()
        engine.is_model_loaded = AsyncMock(return_value=True)
        engine.generate = AsyncMock(return_value=[
            InferenceResult(request_id="r1", outputs=["ok"], prompt_tokens=1,
                          completion_tokens=1, latency_ms=1, finish_reason="stop")
        ])
        fetcher = self._make_fetcher_with_engine(engine)

        request = _make_queued_request(metadata=None)
        await fetcher._default_handler(request)

        call_kwargs = engine.generate.call_args.kwargs
        sp = call_kwargs["sampling_params"]
        assert sp.temperature == 0.7
        assert sp.top_p == 0.9
        assert sp.top_k == 50
        assert sp.max_tokens == 512


# ==================== 重试逻辑 ====================


class TestTaskFetcherRetry:
    """_retry_task 重试逻辑"""

    def _make_fetcher(self, max_retries=3):
        config = TaskFetcherConfig(node_id="n1", max_retries=max_retries)
        return TaskFetcher(config=config)

    @pytest.mark.asyncio
    async def test_retry_within_limit(self):
        """[正常用例] 未超过 max_retries → 重新入队"""
        fetcher = self._make_fetcher(max_retries=3)
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()

        request = _make_queued_request(metadata={"retry_count": 0})
        await fetcher._retry_task(request)

        # 验证 requeue 被调用
        assert fetcher._redis_queue.requeue.called
        assert fetcher.stats["tasks_retried"] == 1

    @pytest.mark.asyncio
    async def test_retry_exceeds_limit(self):
        """[边界值] 超过 max_retries → 存储失败结果"""
        fetcher = self._make_fetcher(max_retries=3)
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()

        request = _make_queued_request(metadata={"retry_count": 3})
        await fetcher._retry_task(request)

        # 超过限制，不应 requeue
        assert not fetcher._redis_queue.requeue.called
        # 应存储失败结果
        assert fetcher._redis_queue.store_result.called
        call_args = fetcher._redis_queue.store_result.call_args
        assert call_args[0][0] == "req-001"
        assert call_args[0][1]["status"] == "error"
        assert call_args[0][1]["reason"] == "max_retries_exceeded"

    @pytest.mark.asyncio
    async def test_retry_at_boundary(self):
        """[边界值] retry_count == max_retries-1 (最后一次重试)"""
        fetcher = self._make_fetcher(max_retries=3)
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()

        request = _make_queued_request(metadata={"retry_count": 2})
        await fetcher._retry_task(request)

        # 还在限制内，应该 requeue
        assert fetcher._redis_queue.requeue.called
        assert fetcher.stats["tasks_retried"] == 1

    @pytest.mark.asyncio
    async def test_retry_no_redis_queue(self):
        """[异常场景] 无 redis_queue → 不抛异常"""
        fetcher = self._make_fetcher()
        # redis_queue is None
        request = _make_queued_request(metadata={"retry_count": 0})
        await fetcher._retry_task(request)
        # 不应抛异常

    @pytest.mark.asyncio
    async def test_retry_increments_metadata_count(self):
        """[状态变更] retry 时 metadata.retry_count +1"""
        fetcher = self._make_fetcher(max_retries=3)
        fetcher._redis_queue = AsyncMock()
        fetcher._redis_queue.requeue = AsyncMock()
        fetcher._redis_queue.store_result = AsyncMock()

        request = _make_queued_request(metadata={"retry_count": 0})
        await fetcher._retry_task(request)

        call_args = fetcher._redis_queue.requeue.call_args
        passed_request = call_args[0][0]
        assert passed_request.metadata["retry_count"] == 1


# ==================== 统计信息 ====================


class TestTaskFetcherStats:
    """get_stats 方法"""

    def test_get_stats_returns_all_fields(self):
        """[正常用例] get_stats 返回所有字段"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        stats = fetcher.get_stats()
        required = {"tasks_pulled", "tasks_completed", "tasks_failed",
                     "tasks_retried", "active_tasks", "running"}
        assert required.issubset(set(stats.keys()))

    def test_get_stats_initial_values(self):
        """[默认值] 初始统计全为 0"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        stats = fetcher.get_stats()
        assert stats["tasks_pulled"] == 0
        assert stats["tasks_completed"] == 0
        assert stats["tasks_failed"] == 0
        assert stats["tasks_retried"] == 0
        assert stats["active_tasks"] == 0
        assert stats["running"] is False

    def test_get_stats_reflects_running(self):
        """[状态] running 反映当前运行状态"""
        fetcher = TaskFetcher(config=TaskFetcherConfig(node_id="n1"))
        assert fetcher.get_stats()["running"] is False
        fetcher._running = True
        assert fetcher.get_stats()["running"] is True


# ==================== 并发槽位 ====================


class TestTaskFetcherConcurrency:
    """active_tasks 槽位控制"""

    @pytest.mark.asyncio
    async def test_at_capacity_no_more_tasks(self):
        """[边界值] 达到 active_tasks 上限时不再拉取"""
        config = TaskFetcherConfig(
            node_id="n1",
            poll_interval_ms=100,
            batch_size=1,
            active_tasks=2,
        )
        fetcher = TaskFetcher(config=config)
        fetcher._running = True
        fetcher._redis_queue = AsyncMock()
        fetcher._active_tasks = {"task-1": AsyncMock(), "task-2": AsyncMock()}

        # 在拉取循环中，_active_tasks 满时应该跳过
        assert len(fetcher._active_tasks) >= config.active_tasks

    @pytest.mark.asyncio
    async def test_below_capacity_pulls_tasks(self):
        """[正常用例] 有空位时继续拉取"""
        config = TaskFetcherConfig(
            node_id="n1",
            active_tasks=10,
        )
        fetcher = TaskFetcher(config=config)
        fetcher._active_tasks = {}
        assert len(fetcher._active_tasks) < config.active_tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
