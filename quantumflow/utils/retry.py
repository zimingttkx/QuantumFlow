"""重试机制"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Optional, Tuple, Type, Union
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class RetryConfig:
    """重试配置"""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
    on_retry: Optional[Callable[[Exception, int], None]] = None


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    reraise: bool = True,
):
    """同步重试装饰器

    Args:
        max_attempts: 最大重试次数
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        backoff_factor: 退避因子
        jitter: 是否添加随机抖动
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        reraise: 是否在重试次数耗尽后重新抛出异常

    Returns:
        装饰后的函数
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        if reraise:
                            raise
                        return None

                    # 计算延迟
                    import random

                    delay = min(current_delay, max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        "retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=delay,
                        error=str(e),
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    import time

                    time.sleep(delay)
                    current_delay *= backoff_factor

            return None

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    reraise: bool = True,
):
    """异步重试装饰器

    Args:
        max_attempts: 最大重试次数
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        backoff_factor: 退避因子
        jitter: 是否添加随机抖动
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        reraise: 是否在重试次数耗尽后重新抛出异常

    Returns:
        装饰后的异步函数
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        if reraise:
                            raise
                        return None

                    # 计算延迟
                    import random

                    delay = min(current_delay, max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        "retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=delay,
                        error=str(e),
                    )

                    if on_retry:
                        if asyncio.iscoroutinefunction(on_retry):
                            await on_retry(e, attempt)
                        else:
                            on_retry(e, attempt)

                    await asyncio.sleep(delay)
                    current_delay *= backoff_factor

            return None

        return wrapper

    return decorator


class RetryContext:
    """可配置的异步重试上下文管理器"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.exceptions = exceptions

        self.attempt = 0
        self.last_exception: Optional[Exception] = None

    async def __aenter__(self) -> "RetryContext":
        self.attempt += 1
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_val: Optional[Exception],
        exc_tb: Any,
    ) -> bool:
        if exc_val is None:
            return True

        if not isinstance(exc_val, self.exceptions):
            return False

        self.last_exception = exc_val

        if self.attempt >= self.max_attempts:
            logger.error(
                "retry_exhausted",
                attempts=self.max_attempts,
                error=str(exc_val),
            )
            return False

        # 计算延迟
        import random

        current_delay = min(self.initial_delay * (self.backoff_factor ** (self.attempt - 1)), self.max_delay)
        if self.jitter:
            current_delay = current_delay * (0.5 + random.random())

        logger.warning(
            "retry_attempt",
            attempt=self.attempt,
            max_attempts=self.max_attempts,
            delay=current_delay,
            error=str(exc_val),
        )

        await asyncio.sleep(current_delay)
        return True

    @property
    def should_retry(self) -> bool:
        """是否应该重试"""
        return self.attempt < self.max_attempts and self.last_exception is not None
