"""重试机制 - 严格单元测试

测试覆盖:
1. RetryConfig 数据类
2. retry 同步装饰器: 正常通过、异常重试、重试耗尽、reraise、异常过滤、jitter、on_retry
3. async_retry 异步装饰器: 同上 + 异步 on_retry 回调
4. RetryContext 异步上下文管理器: 成功退出、异常重试、不匹配异常、max_attempts 耗尽
5. 退避计算验证: exponential backoff, max_delay capping, jitter randomness
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantumflow.utils.retry import (
    RetryConfig,
    RetryContext,
    async_retry,
    retry,
)


# ==================== RetryConfig 数据类测试 ====================


class TestRetryConfig:
    """RetryConfig 数据类测试"""

    def test_default_values(self):
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.initial_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.backoff_factor == 2.0
        assert cfg.jitter is True
        assert cfg.exceptions == (Exception,)
        assert cfg.on_retry is None

    def test_custom_values(self):
        callback = MagicMock()
        cfg = RetryConfig(max_attempts=5, initial_delay=2.0, max_delay=30.0,
                          backoff_factor=3.0, jitter=False,
                          exceptions=(ValueError, TypeError), on_retry=callback)
        assert cfg.max_attempts == 5
        assert cfg.initial_delay == 2.0
        assert cfg.max_delay == 30.0
        assert cfg.backoff_factor == 3.0
        assert cfg.jitter is False
        assert cfg.exceptions == (ValueError, TypeError)
        assert cfg.on_retry is callback

    def test_zero_max_attempts(self):
        """[边界值] max_attempts=0"""
        cfg = RetryConfig(max_attempts=0)
        assert cfg.max_attempts == 0

    def test_large_delay_values(self):
        """[边界值] 极大延迟值"""
        cfg = RetryConfig(initial_delay=100, max_delay=300)
        assert cfg.initial_delay == 100
        assert cfg.max_delay == 300


# ==================== retry 同步装饰器测试 ====================


class TestSyncRetryDecorator:
    """retry 同步装饰器测试"""

    def test_success_first_attempt_no_retry(self):
        """[正常用例] 首次成功不触发重试"""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_matching_exception(self):
        """[核心功能] 匹配异常时触发重试"""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01, exceptions=(ValueError,))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "recovered"

        result = flaky()
        assert result == "recovered"
        assert call_count == 3

    def test_retry_exhausted_reraise_true(self):
        """[核心功能] 重试耗尽且 reraise=True 时重新抛出异常"""
        call_count = 0

        @retry(max_attempts=2, initial_delay=0.01, reraise=True, exceptions=(ValueError,))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("persistent error")

        with pytest.raises(ValueError, match="persistent error"):
            always_fail()
        assert call_count == 2

    def test_retry_exhausted_reraise_false_returns_none(self):
        """[核心功能] 重试耗尽且 reraise=False 时返回 None"""
        call_count = 0

        @retry(max_attempts=2, initial_delay=0.01, reraise=False, exceptions=(ValueError,))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("persistent error")

        result = always_fail()
        assert result is None
        assert call_count == 2

    def test_non_matching_exception_not_retried(self):
        """[核心功能] 不匹配的异常不触发重试"""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01, exceptions=(ValueError,))
        def fail_with_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not a ValueError")

        with pytest.raises(TypeError, match="not a ValueError"):
            fail_with_type_error()
        assert call_count == 1

    def test_multiple_exception_types(self):
        """[核心功能] 多个异常类型都应触发重试"""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01, exceptions=(ValueError, TypeError))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise TypeError("type error")
            if call_count < 3:
                raise ValueError("value error")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 3

    def test_on_retry_callback_called(self):
        """[核心功能] on_retry 回调在每次重试时被调用"""
        callback = MagicMock()
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01, on_retry=callback, exceptions=(ValueError,))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("error")
            return "ok"

        flaky()
        assert callback.call_count == 2
        for call_args in callback.call_args_list:
            assert isinstance(call_args[0][0], Exception)
            assert isinstance(call_args[0][1], int)

    @patch("time.sleep", return_value=None)
    def test_exponential_backoff_delay_sequence(self, mock_sleep):
        """[核心功能] 退避延迟序列验证"""
        call_count = 0

        @retry(max_attempts=4, initial_delay=1.0, backoff_factor=2.0, jitter=False,
               reraise=False, exceptions=(ValueError,))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        always_fail()

        assert mock_sleep.call_count == 3
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0
        assert mock_sleep.call_args_list[2][0][0] == 4.0

    @patch("time.sleep", return_value=None)
    def test_delay_capped_by_max_delay(self, mock_sleep):
        """[核心功能] 延迟不超过 max_delay"""
        @retry(max_attempts=5, initial_delay=1.0, max_delay=3.0, backoff_factor=10.0,
               jitter=False, reraise=False, exceptions=(ValueError,))
        def always_fail():
            raise ValueError("error")

        always_fail()

        for call in mock_sleep.call_args_list:
            assert call[0][0] <= 3.0

    @patch("time.sleep", return_value=None)
    def test_jitter_affects_delay(self, mock_sleep):
        """[核心功能] jitter=True 时延迟有随机变化"""
        call_count = 0

        @retry(max_attempts=10, initial_delay=2.0, backoff_factor=2.0, jitter=True,
               reraise=False, exceptions=(ValueError,))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        always_fail()

        delays = [call[0][0] for call in mock_sleep.call_args_list]
        theoretical = [2.0 * (2.0**i) for i in range(9)]
        different_from_theoretical = any(
            abs(d - t) > 0.001 for d, t in zip(delays, theoretical[:len(delays)])
        )
        assert different_from_theoretical, (
            f"jitter=True 时延迟应有变化: delays={delays}, theoretical={theoretical}"
        )

    def test_func_metadata_preserved(self):
        """[核心功能] 装饰器保留原函数元数据"""
        @retry(max_attempts=2, initial_delay=0.01)
        def my_function():
            """My docstring"""
            return 42

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring"

    def test_func_arguments_passed_correctly(self):
        """[核心功能] 函数参数正确传递"""
        captured = []

        @retry(max_attempts=2, initial_delay=0.01)
        def func_with_args(a, b, *, c=None):
            captured.append((a, b, c))
            return a + b + (c or 0)

        result = func_with_args(1, 2, c=3)
        assert result == 6
        assert captured == [(1, 2, 3)]

    def test_max_attempts_1_means_no_retry(self):
        """[边界用例] max_attempts=1 表示不重试"""
        call_count = 0

        @retry(max_attempts=1, initial_delay=0.01, reraise=True, exceptions=(ValueError,))
        def fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("only one shot")

        with pytest.raises(ValueError):
            fail()
        assert call_count == 1

    @patch("time.sleep", return_value=None)
    def test_max_delay_cap_on_large_backoff(self, mock_sleep):
        """[边界值] 大 backoff 被 max_delay 截断"""
        @retry(max_attempts=2, initial_delay=100, max_delay=5, backoff_factor=10,
               jitter=False, reraise=False)
        def always_fail():
            raise ValueError("fail")

        always_fail()
        # sleep called once with min(100,5)=5 or min(1000,5)=5
        assert mock_sleep.call_args_list[0][0][0] <= 5.0


# ==================== async_retry 异步装饰器测试 ====================


class TestAsyncRetryDecorator:
    """async_retry 异步装饰器测试"""

    @pytest.mark.asyncio
    async def test_success_first_attempt_no_retry(self):
        """[正常用例] 首次成功不触发重试"""
        call_count = 0

        @async_retry(max_attempts=3, initial_delay=0.01)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "async success"

        result = await succeed()
        assert result == "async success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_matching_exception(self):
        """[核心功能] 匹配异常时触发异步重试"""
        call_count = 0

        @async_retry(max_attempts=3, initial_delay=0.01, exceptions=(ValueError,))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "async recovered"

        result = await flaky()
        assert result == "async recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_reraise_true(self):
        """[核心功能] 重试耗尽且 reraise=True 时重新抛出"""
        call_count = 0

        @async_retry(max_attempts=2, initial_delay=0.01, reraise=True, exceptions=(ValueError,))
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("persistent")

        with pytest.raises(ValueError):
            await always_fail()
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_reraise_false_returns_none(self):
        """[核心功能] reraise=False 时返回 None"""
        call_count = 0

        @async_retry(max_attempts=2, initial_delay=0.01, reraise=False, exceptions=(ValueError,))
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("persistent")

        result = await always_fail()
        assert result is None
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_matching_exception_not_retried(self):
        """[核心功能] 不匹配的异常不触发重试"""
        call_count = 0

        @async_retry(max_attempts=3, initial_delay=0.01, exceptions=(ValueError,))
        async def fail_with_key_error():
            nonlocal call_count
            call_count += 1
            raise KeyError("not matched")

        with pytest.raises(KeyError):
            await fail_with_key_error()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_sync_on_retry_callback_called(self):
        """[核心功能] 同步 on_retry 回调被调用"""
        callback = MagicMock()
        call_count = 0

        @async_retry(max_attempts=3, initial_delay=0.01, on_retry=callback,
                     exceptions=(ValueError,))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("error")
            return "ok"

        await flaky()
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_async_on_retry_callback_called(self):
        """[核心功能] 异步 on_retry 回调被 await"""
        async_calls = []

        async def async_on_retry(exc, attempt):
            async_calls.append((str(exc), attempt))
            await asyncio.sleep(0.001)

        call_count = 0

        @async_retry(max_attempts=3, initial_delay=0.01, on_retry=async_on_retry,
                     exceptions=(ValueError,))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("async retry error")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert len(async_calls) == 2

    @pytest.mark.asyncio
    async def test_async_func_metadata_preserved(self):
        """[核心功能] 装饰器保留异步函数元数据"""
        @async_retry(max_attempts=2, initial_delay=0.01)
        async def my_async_func():
            """Async docstring"""
            return 42

        assert my_async_func.__name__ == "my_async_func"
        assert my_async_func.__doc__ == "Async docstring"

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_exponential_backoff_async(self, mock_sleep):
        """[核心功能] 异步退避延迟序列验证"""
        call_count = 0

        @async_retry(max_attempts=4, initial_delay=1.0, backoff_factor=2.0,
                     jitter=False, reraise=False, exceptions=(ValueError,))
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        await always_fail()

        assert mock_sleep.call_count == 3
        expected = [1.0, 2.0, 4.0]
        for i, call in enumerate(mock_sleep.call_args_list):
            assert call[0][0] == expected[i], (
                f"delay[{i}] 期望 {expected[i]}, 实际 {call[0][0]}"
            )


# ==================== RetryContext 异步上下文管理器测试 ====================


class TestRetryContext:
    """RetryContext 异步上下文管理器测试"""

    @pytest.mark.asyncio
    async def test_successful_context(self):
        """[正常用例] 无异常时正常退出"""
        ctx = RetryContext(max_attempts=3)
        async with ctx:
            pass
        assert ctx.attempt == 1
        assert ctx.last_exception is None

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_matching_exception_triggers_retry(self, mock_sleep):
        """[核心功能] 匹配异常触发重试"""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01, jitter=False,
                           exceptions=(ValueError,))
        attempts = 0

        for _ in range(3):
            async with ctx:
                attempts += 1
                if attempts < 3:
                    raise ValueError("retry me")

        assert attempts == 3
        assert ctx.attempt == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_non_matching_exception_propagates(self, mock_sleep):
        """[核心功能] 不匹配的异常直接传播"""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01, exceptions=(ValueError,))

        with pytest.raises(KeyError):
            async with ctx:
                raise KeyError("not retriable")

        assert ctx.attempt == 1
        assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_max_attempts_exceeded_raises(self, mock_sleep):
        """[核心功能] 超过最大尝试次数后异常传播"""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01, jitter=False,
                           exceptions=(ValueError,))
        captured = []

        with pytest.raises(ValueError):
            for _ in range(4):
                async with ctx:
                    captured.append(ctx.attempt)
                    raise ValueError("fail")

        assert len(captured) == 3
        assert ctx.attempt == 3

    def test_should_retry_property_inital(self):
        """[核心功能] 初始状态 should_retry=False"""
        ctx = RetryContext(max_attempts=3)
        assert ctx.should_retry is False

    @pytest.mark.asyncio
    async def test_should_retry_after_exception(self):
        """[核心功能] 异常后 should_retry=True"""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01, exceptions=(ValueError,))
        async with ctx:
            raise ValueError("first")
        assert ctx.should_retry is True

    @pytest.mark.asyncio
    async def test_should_retry_false_at_max(self):
        """[核心功能] 达到上限后 should_retry=False"""
        ctx = RetryContext(max_attempts=1, initial_delay=0.001, exceptions=(ValueError,))
        try:
            async with ctx:
                raise ValueError("final")
        except ValueError:
            pass
        assert ctx.should_retry is False

    @pytest.mark.asyncio
    async def test_last_exception_tracks_errors(self):
        """[核心功能] last_exception 跟踪最近异常"""
        ctx = RetryContext(max_attempts=3, initial_delay=0.001, exceptions=(ValueError,))
        async with ctx:
            raise ValueError("first error")
        assert isinstance(ctx.last_exception, ValueError)

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_exponential_backoff_in_context(self, mock_sleep):
        """[核心功能] 上下文管理器退避延迟序列"""
        ctx = RetryContext(max_attempts=5, initial_delay=1.0, backoff_factor=2.0,
                           jitter=False, exceptions=(ValueError,))

        for _ in range(4):
            async with ctx:
                raise ValueError("fail")
        # 第5次尝试会抛出异常（超过max_attempts）
        try:
            async with ctx:
                raise ValueError("fail")
        except ValueError:
            pass

        assert mock_sleep.call_count == 4
        expected = [1.0, 2.0, 4.0, 8.0]
        for i, call in enumerate(mock_sleep.call_args_list):
            assert abs(call[0][0] - expected[i]) < 0.001

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_delay_capped_at_max_delay(self, mock_sleep):
        """[核心功能] 延迟不超过 max_delay"""
        ctx = RetryContext(max_attempts=5, initial_delay=1.0, max_delay=3.0,
                           backoff_factor=10.0, jitter=False, exceptions=(ValueError,))

        for _ in range(4):
            async with ctx:
                raise ValueError("fail")
        try:
            async with ctx:
                raise ValueError("fail")
        except ValueError:
            pass

        for call in mock_sleep.call_args_list:
            assert call[0][0] <= 3.0

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_jitter_adds_randomness(self, mock_sleep):
        """[核心功能] jitter=True 引入随机变化"""
        ctx = RetryContext(max_attempts=5, initial_delay=2.0, backoff_factor=2.0,
                           jitter=True, exceptions=(ValueError,))

        for _ in range(4):
            async with ctx:
                raise ValueError("fail")
        try:
            async with ctx:
                raise ValueError("fail")
        except ValueError:
            pass

        delays = [call[0][0] for call in mock_sleep.call_args_list]
        theoretical = [2.0 * (2.0**i) for i in range(4)]
        different = any(abs(d - t) > 0.001 for d, t in zip(delays, theoretical))
        assert different, f"jitter=True 延迟应有变化: delays={delays}"

    @pytest.mark.asyncio
    async def test_attempt_counter_increments_correctly(self):
        """[核心功能] attempt 计数器正确递增"""
        ctx = RetryContext(max_attempts=3, initial_delay=0.001, exceptions=(ValueError,))
        async with ctx:
            raise ValueError("1")
        assert ctx.attempt == 1
        async with ctx:
            raise ValueError("2")
        assert ctx.attempt == 2
        async with ctx:
            pass
        assert ctx.attempt == 3

    @pytest.mark.asyncio
    async def test_multiple_retries_then_success(self):
        """[完整流程] 多次重试后成功"""
        ctx = RetryContext(max_attempts=5, initial_delay=0.001, jitter=False,
                           exceptions=(ValueError,))
        failures = 0
        for _ in range(5):
            async with ctx:
                failures += 1
                if failures < 4:
                    raise ValueError(f"fail #{failures}")
            if failures >= 4:
                break

        assert failures == 4
        assert ctx.attempt == 4
        assert isinstance(ctx.last_exception, ValueError)


# ==================== GAP-FILLING: max_attempts=0 边界用例 (lines 100, 179) ====================


class TestRetryMaxAttemptsZero:
    """retry/async_retry max_attempts=0 边界用例（覆盖 line 100 和 line 179）"""

    def test_sync_retry_max_attempts_zero_returns_none(self):
        """[边界用例] max_attempts=0 时循环不执行, 返回 None (line 100)"""
        call_count = 0

        @retry(max_attempts=0, initial_delay=0.001, reraise=False, exceptions=(ValueError,))
        def never_called():
            nonlocal call_count
            call_count += 1
            return "should not reach"

        result = never_called()
        assert result is None
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_async_retry_max_attempts_zero_returns_none(self):
        """[边界用例] async retry max_attempts=0 时循环不执行, 返回 None (line 179)"""
        call_count = 0

        @async_retry(max_attempts=0, initial_delay=0.001, reraise=False, exceptions=(ValueError,))
        async def never_called():
            nonlocal call_count
            call_count += 1
            return "should not reach"

        result = await never_called()
        assert result is None
        assert call_count == 0
