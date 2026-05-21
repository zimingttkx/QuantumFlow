"""gRPC 通道池管理

提供 gRPC Channel 的连接池管理：
- 连接创建和复用
- 连接健康检查
- 自动重连
- 负载均衡支持
"""

import threading
import time
from typing import Dict, List, Optional, Set

import grpc

from quantumflow.utils.logging import get_logger

logger = get_logger(__name__)


class GrpcChannel:
    """gRPC Channel 封装

    提供单个 Channel 的健康检查和重连功能。
    """

    def __init__(
        self,
        target: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Args:
            target: 服务器地址 (host:port)
            timeout: 默认超时时间
            max_retries: 最大重试次数
        """
        self.target = target
        self.timeout = timeout
        self.max_retries = max_retries
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[any] = None
        self._last_used: float = time.time()
        self._lock = threading.Lock()

    def get_channel(self) -> grpc.Channel:
        """获取 Channel"""
        with self._lock:
            if self._channel is None or not self._channel.is_alive():
                self._create_channel()
            self._last_used = time.time()
            return self._channel

    def _create_channel(self) -> None:
        """创建 Channel"""
        self._channel = grpc.insecure_channel(
            self.target,
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
            ],
        )
        logger.info(f"Created gRPC channel to {self.target}")

    def is_alive(self) -> bool:
        """检查 Channel 是否存活"""
        with self._lock:
            return self._channel is not None and self._channel.is_alive()

    def close(self) -> None:
        """关闭 Channel"""
        with self._lock:
            if self._channel is not None:
                self._channel.close()
                self._channel = None
                logger.info(f"Closed gRPC channel to {self.target}")

    @property
    def last_used(self) -> float:
        """获取最后使用时间"""
        return self._last_used


class GrpcChannelPool:
    """gRPC Channel 连接池

    管理多个 Channel，支持连接复用和自动清理。
    """

    def __init__(
        self,
        max_size: int = 10,
        max_idle_seconds: float = 300.0,
        cleanup_interval: float = 60.0,
    ):
        """
        Args:
            max_size: 最大连接数
            max_idle_seconds: 空闲连接最大存活时间
            cleanup_interval: 清理间隔（秒）
        """
        self.max_size = max_size
        self.max_idle_seconds = max_idle_seconds
        self.cleanup_interval = cleanup_interval
        self._channels: Dict[str, GrpcChannel] = {}
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

    def get_channel(self, target: str, timeout: float = 30.0) -> GrpcChannel:
        """获取 Channel

        Args:
            target: 服务器地址
            timeout: 超时时间

        Returns:
            GrpcChannel 实例
        """
        with self._lock:
            if target not in self._channels:
                if len(self._channels) >= self.max_size:
                    # 移除最旧的连接
                    self._evict_oldest()
                self._channels[target] = GrpcChannel(target, timeout)
                logger.info(f"Added channel to pool: {target}")

            channel = self._channels[target]
            return channel

    def _evict_oldest(self) -> None:
        """移除最旧的连接"""
        if not self._channels:
            return

        oldest_target = min(
            self._channels.keys(),
            key=lambda t: self._channels[t].last_used,
        )
        self._channels[oldest_target].close()
        del self._channels[oldest_target]
        logger.info(f"Evicted oldest channel: {oldest_target}")

    def remove_channel(self, target: str) -> None:
        """移除 Channel

        Args:
            target: 服务器地址
        """
        with self._lock:
            if target in self._channels:
                self._channels[target].close()
                del self._channels[target]

    def close_all(self) -> None:
        """关闭所有 Channel"""
        with self._lock:
            for channel in self._channels.values():
                channel.close()
            self._channels.clear()

    def start_cleanup(self) -> None:
        """启动清理线程"""
        if self._cleanup_thread is not None:
            return

        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def stop_cleanup(self) -> None:
        """停止清理线程"""
        if self._cleanup_thread is None:
            return

        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=5.0)
        self._cleanup_thread = None

    def _cleanup_loop(self) -> None:
        """清理循环"""
        while not self._stop_cleanup.is_set():
            self._cleanup_idle()
            self._stop_cleanup.wait(self.cleanup_interval)

    def _cleanup_idle(self) -> None:
        """清理空闲连接"""
        now = time.time()
        with self._lock:
            to_remove = []
            for target, channel in self._channels.items():
                if now - channel.last_used > self.max_idle_seconds:
                    if not channel.is_alive():
                        to_remove.append(target)

            for target in to_remove:
                self._channels[target].close()
                del self._channels[target]
                logger.info(f"Cleaned up idle channel: {target}")

    def get_stats(self) -> Dict[str, int]:
        """获取连接统计

        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                "total_channels": len(self._channels),
                "max_size": self.max_size,
                "active_channels": sum(
                    1 for c in self._channels.values() if c.is_alive()
                ),
            }

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close_all()
        self.stop_cleanup()


# 全局连接池实例
_default_pool: Optional[GrpcChannelPool] = None
_pool_lock = threading.Lock()


def get_default_pool() -> GrpcChannelPool:
    """获取默认连接池

    Returns:
        GrpcChannelPool 实例
    """
    global _default_pool
    with _pool_lock:
        if _default_pool is None:
            _default_pool = GrpcChannelPool()
            _default_pool.start_cleanup()
        return _default_pool
