"""Pytest配置"""

import logging

import pytest

# 配置标准库日志
logging.basicConfig(
    format="%(message)s",
    stream=None,
    level=logging.WARNING,
)


@pytest.fixture(autouse=True)
def reset_logging():
    """每个测试后重置日志配置"""
    yield
    # 测试后不执行任何操作
