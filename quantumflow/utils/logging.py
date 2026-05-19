"""日志配置模块"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import Processor

from quantumflow.version import __version__

# 类型别名
BoundLogger = structlog.BoundLogger


def _drop_key(key: str) -> Processor:
    """返回一个删除指定键的处理器"""

    def drop_processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        event_dict.pop(key, None)
        return event_dict

    return drop_processor


# 自定义JSON渲染器
class JsonRenderer:
    """自定义JSON日志渲染器"""

    def __init__(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        include_timestamp: bool = True,
    ):
        self.indent = indent
        self.ensure_ascii = ensure_ascii
        self.include_timestamp = include_timestamp

    def __call__(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> str:
        """渲染日志事件为JSON字符串"""
        # 移除structlog内部字段
        filtered = {k: v for k, v in event_dict.items() if not k.startswith("_") and v is not None}

        # 格式化时间戳
        if "timestamp" in filtered and isinstance(filtered["timestamp"], float):
            import datetime

            filtered["timestamp"] = datetime.datetime.fromtimestamp(
                filtered["timestamp"]
            ).isoformat()

        # 添加版本信息
        filtered["version"] = __version__

        return json.dumps(filtered, indent=self.indent, ensure_ascii=self.ensure_ascii)


# 颜色渲染器（用于控制台）
class ConsoleRenderer:
    """控制台彩色日志渲染器"""

    COLORS = {
        "critical": "\033[91m",  # 红色
        "error": "\033[91m",
        "warning": "\033[93m",  # 黄色
        "warn": "\033[93m",
        "info": "\033[92m",  # 绿色
        "debug": "\033[94m",  # 蓝色
        "notset": "\033[0m",
        "reset": "\033[0m",
    }

    def __init__(self, *, colors: bool = True, compact: bool = False):
        self.colors = colors and sys.stdout.isatty()
        self.compact = compact

    def __call__(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> str:
        """渲染日志事件为控制台格式"""
        # 获取颜色
        color = self.COLORS.get(method_name, self.COLORS["reset"])
        reset = self.COLORS["reset"] if self.colors else ""

        # 提取关键信息
        timestamp = event_dict.pop("timestamp", None)
        level = event_dict.pop("level", method_name.upper())
        event = event_dict.pop("event", "")
        request_id = event_dict.pop("request_id", None)
        component = event_dict.pop("component", None)

        # 构建输出
        parts = []

        # 时间戳
        if timestamp:
            if isinstance(timestamp, float):
                import datetime

                timestamp = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            parts.append(f"[{timestamp}]")

        # 级别
        parts.append(f"{color}{level:8}{reset}")

        # 组件
        if component:
            parts.append(f"[{component}]")

        # 请求ID
        if request_id:
            parts.append(f"({request_id})")

        # 事件
        parts.append(f"{color if not event else ''}{event}{reset}")

        # 额外信息
        if event_dict and not self.compact:
            extra = " ".join(f"{k}={v!r}" for k, v in event_dict.items())
            parts.append(f"│ {extra}")

        return " ".join(parts)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    log_file: str | Path | None = None,
    component: str | None = None,
    include_process_info: bool = False,
    include_thread_info: bool = False,
) -> None:
    """配置structlog日志系统

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日志格式 (console, json)
        log_file: 日志文件路径（可选）
        component: 组件名称
        include_process_info: 是否包含进程信息
        include_thread_info: 是否包含线程信息
    """
    # 基础处理器
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        # structlog.stdlib.add_logger_name,  # 不与PrintLogger兼容
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        (
            structlog.processors.StackInfoRenderer()
            if include_process_info
            else _drop_key("stack_info")
        ),
        structlog.processors.UnicodeDecoder(),
    ]

    # 添加组件名到上下文
    if component:
        structlog.contextvars.bind_contextvars(qf_component=component)

    # 添加调用位置信息（仅在DEBUG模式）
    if log_level == "DEBUG":
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            )
        )

    # 异常处理
    processors.append(structlog.processors.ExceptionRenderer())

    # 渲染器
    if log_format == "json":
        processors.append(JsonRenderer(include_timestamp=False))
    else:
        processors.append(ConsoleRenderer(colors=True))

    # 配置structlog - 使用简化的wrapper
    try:
        wrapper_class = structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        )
    except Exception:
        wrapper_class = structlog.BoundLogger

    structlog.configure(
        processors=processors,
        wrapper_class=wrapper_class,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # 配置标准库日志
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # 设置第三方库日志级别
    for logger_name in ["uvicorn", "fastapi", "httpx", "httpcore", "h11"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None, **kwargs: Any) -> BoundLogger:
    """获取logger实例

    Args:
        name: logger名称
        **kwargs: 额外的绑定变量

    Returns:
        structlog.BoundLogger实例
    """
    logger = structlog.get_logger()

    if name:
        logger = logger.bind(component=name)

    if kwargs:
        logger = logger.bind(**kwargs)

    return logger


class LoggerMixin:
    """为类添加日志功能的Mixin"""

    @property
    def logger(self) -> BoundLogger:
        """获取logger实例"""
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return structlog.get_logger().bind(component=name)


# 便捷函数
def debug(msg: str, **kwargs: Any) -> None:
    """记录DEBUG级别日志"""
    structlog.get_logger().debug(msg, **kwargs)


def info(msg: str, **kwargs: Any) -> None:
    """记录INFO级别日志"""
    structlog.get_logger().info(msg, **kwargs)


def warning(msg: str, **kwargs: Any) -> None:
    """记录WARNING级别日志"""
    structlog.get_logger().warning(msg, **kwargs)


def error(msg: str, **kwargs: Any) -> None:
    """记录ERROR级别日志"""
    structlog.get_logger().error(msg, **kwargs)


def critical(msg: str, **kwargs: Any) -> None:
    """记录CRITICAL级别日志"""
    structlog.get_logger().critical(msg, **kwargs)
