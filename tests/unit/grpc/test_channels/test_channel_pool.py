"""gRPC Channel Pool 测试

严格测试连接池的业务逻辑：
- 连接创建和复用
- 连接健康检查
- 空闲清理
- 统计信息
"""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch

from quantumflow.grpc.channels.pool import GrpcChannel, GrpcChannelPool


class TestGrpcChannelLifecycle:
    """GrpcChannel 生命周期测试"""

    def test_get_channel_creates_channel(self):
        """获取 channel 时创建 channel"""
        channel = GrpcChannel("localhost:50051")

        with patch("quantumflow.grpc.channels.pool.grpc.insecure_channel") as mock_channel:
            mock_channel_instance = MagicMock()
            mock_channel.return_value = mock_channel_instance

            result = channel.get_channel()

            mock_channel.assert_called_once()
            assert result is mock_channel_instance

    def test_close_closes_channel(self):
        """close 关闭 channel"""
        channel = GrpcChannel("localhost:50051")

        with patch("quantumflow.grpc.channels.pool.grpc.insecure_channel") as mock_channel:
            mock_channel_instance = MagicMock()
            mock_channel.return_value = mock_channel_instance

            channel.get_channel()
            channel.close()

            mock_channel_instance.close.assert_called_once()


class TestGrpcChannelPoolBasics:
    """GrpcChannelPool 基础测试"""

    def test_default_initialization(self):
        """默认初始化"""
        pool = GrpcChannelPool()

        assert pool.max_size == 10
        assert pool.max_idle_seconds == 300.0
        assert pool.cleanup_interval == 60.0

    def test_custom_initialization(self):
        """自定义初始化"""
        pool = GrpcChannelPool(
            max_size=20,
            max_idle_seconds=600.0,
            cleanup_interval=120.0,
        )

        assert pool.max_size == 20
        assert pool.max_idle_seconds == 600.0
        assert pool.cleanup_interval == 120.0


class TestGrpcChannelPoolGetChannel:
    """获取 channel 测试"""

    def test_get_channel_returns_same_for_same_target(self):
        """同一 target 返回同一个 channel"""
        pool = GrpcChannelPool()

        channel1 = pool.get_channel("localhost:50051")
        channel2 = pool.get_channel("localhost:50051")

        assert channel1 is channel2

    def test_get_channel_different_for_different_targets(self):
        """不同 target 返回不同 channel"""
        pool = GrpcChannelPool()

        channel1 = pool.get_channel("host1:50051")
        channel2 = pool.get_channel("host2:50051")

        assert channel1 is not channel2


class TestGrpcChannelPoolRemoveChannel:
    """移除 channel 测试"""

    def test_remove_channel(self):
        """移除 channel"""
        pool = GrpcChannelPool()

        channel = pool.get_channel("localhost:50051")
        pool.remove_channel("localhost:50051")

        assert "localhost:50051" not in pool._channels

    def test_remove_nonexistent_channel(self):
        """移除不存在的 channel 不报错"""
        pool = GrpcChannelPool()

        # 不应该抛出异常
        pool.remove_channel("nonexistent:50051")


class TestGrpcChannelPoolCloseAll:
    """关闭所有测试"""

    def test_close_all_closes_all_channels(self):
        """关闭所有 channel"""
        pool = GrpcChannelPool()

        pool.get_channel("host1:50051")
        pool.get_channel("host2:50051")
        pool.close_all()

        assert len(pool._channels) == 0


class TestGrpcChannelPoolCleanup:
    """清理测试"""

    def test_start_cleanup_starts_thread(self):
        """启动清理线程"""
        pool = GrpcChannelPool(cleanup_interval=0.1)

        pool.start_cleanup()

        assert pool._cleanup_thread is not None
        assert pool._cleanup_thread.daemon is True

        pool.stop_cleanup()

    def test_stop_cleanup_stops_thread(self):
        """停止清理线程"""
        pool = GrpcChannelPool(cleanup_interval=0.1)

        pool.start_cleanup()
        pool.stop_cleanup()

        # 线程应该已停止
        assert pool._cleanup_thread is None or not pool._cleanup_thread.is_alive()


class TestGrpcChannelPoolStats:
    """统计测试"""

    def test_get_stats_empty_pool(self):
        """空池统计"""
        pool = GrpcChannelPool()

        stats = pool.get_stats()

        assert stats["total_channels"] == 0
        assert stats["max_size"] == 10


class TestGrpcChannelPoolEviction:
    """驱逐策略测试"""

    def test_eviction_preserves_newer_channels(self):
        """驱逐时保留较新的 channel"""
        pool = GrpcChannelPool(max_size=2)

        pool.get_channel("old:50051")
        time.sleep(0.1)
        pool.get_channel("new:50051")

        pool.get_channel("newest:50051")

        assert "old:50051" not in pool._channels
        assert "new:50051" in pool._channels
        assert "newest:50051" in pool._channels


class TestGrpcChannelPoolConcurrency:
    """并发测试"""

    def test_concurrent_get_channel(self):
        """并发获取 channel"""
        pool = GrpcChannelPool()

        def get_channel(target):
            return pool.get_channel(target)

        threads = [threading.Thread(target=get_channel, args=(f"host{i}:50051",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 应该创建了 10 个 channel
        assert len(pool._channels) == 10
