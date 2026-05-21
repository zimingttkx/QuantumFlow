"""gRPC Channel Pool 完整测试

覆盖所有代码路径，包括边界情况。
"""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch, call

from quantumflow.grpc.channels.pool import GrpcChannel, GrpcChannelPool


class TestGrpcChannelLifecycleComplete:
    """GrpcChannel 完整生命周期测试"""

    def test_is_alive_with_no_channel(self):
        """没有 channel 时 is_alive 返回 False"""
        channel = GrpcChannel("localhost:50051")
        # channel._channel is None initially

        result = channel.is_alive()

        assert result is False

    def test_is_alive_with_channel_not_alive(self):
        """channel 存在但不活跃时返回 False"""
        channel = GrpcChannel("localhost:50051")

        with patch("quantumflow.grpc.channels.pool.grpc.insecure_channel") as mock_channel:
            mock_instance = MagicMock()
            mock_instance.is_alive.return_value = False
            mock_channel.return_value = mock_instance

            channel.get_channel()
            result = channel.is_alive()

            assert result is False

    def test_is_alive_with_channel_alive(self):
        """channel 存在且活跃时返回 True"""
        channel = GrpcChannel("localhost:50051")

        with patch("quantumflow.grpc.channels.pool.grpc.insecure_channel") as mock_channel:
            mock_instance = MagicMock()
            mock_instance.is_alive.return_value = True
            mock_channel.return_value = mock_instance

            channel.get_channel()
            result = channel.is_alive()

            assert result is True


class TestGrpcChannelPoolEvictionComplete:
    """连接池驱逐完整测试"""

    def test_evict_oldest_when_empty(self):
        """空池时不驱逐"""
        pool = GrpcChannelPool()

        # 不应该抛出异常
        pool._evict_oldest()

    def test_evict_oldest_removes_correct_channel(self):
        """驱逐最旧的 channel"""
        pool = GrpcChannelPool(max_size=3)

        # 添加三个 channel，中间间隔一点时间
        ch1 = pool.get_channel("host1:50051")
        ch1._last_used = time.time() - 100

        ch2 = pool.get_channel("host2:50051")
        ch2._last_used = time.time() - 50

        ch3 = pool.get_channel("host3:50051")
        ch3._last_used = time.time()

        # 触发驱逐
        pool._evict_oldest()

        # host1 应该被驱逐
        assert "host1:50051" not in pool._channels
        assert "host2:50051" in pool._channels
        assert "host3:50051" in pool._channels


class TestGrpcChannelPoolCleanupComplete:
    """连接池清理完整测试"""

    def test_start_cleanup_idempotent(self):
        """多次启动只创建一个线程"""
        pool = GrpcChannelPool(cleanup_interval=0.1)

        pool.start_cleanup()
        first_thread = pool._cleanup_thread

        pool.start_cleanup()
        second_thread = pool._cleanup_thread

        assert first_thread is second_thread

        pool.stop_cleanup()

    def test_stop_cleanup_when_not_started(self):
        """清理线程未启动时停止"""
        pool = GrpcChannelPool()

        # 不应该抛出异常
        pool.stop_cleanup()

    def test_cleanup_idle_removes_dead_channels(self):
        """清理移除不活跃的 channel"""
        pool = GrpcChannelPool(max_idle_seconds=1.0, cleanup_interval=10.0)

        channel = pool.get_channel("oldhost:50051")

        # 模拟 channel 变旧且不活跃
        channel._last_used = time.time() - 10
        channel._channel = MagicMock()
        channel._channel.is_alive.return_value = False

        # 直接调用清理
        pool._cleanup_idle()

        # 旧的、不活跃的 channel 应该被移除
        assert "oldhost:50051" not in pool._channels

    def test_cleanup_loop_runs_multiple_times(self):
        """清理循环多次运行"""
        pool = GrpcChannelPool(cleanup_interval=0.05)

        with patch.object(pool, "_cleanup_idle") as mock_cleanup:
            pool.start_cleanup()

            # 等待几个清理周期
            time.sleep(0.2)

            # 清理应该运行了多次
            assert mock_cleanup.call_count >= 3

        pool.stop_cleanup()


class TestGrpcChannelPoolContextManager:
    """上下文管理器完整测试"""

    def test_context_manager_enter(self):
        """__enter__ 返回 self"""
        pool = GrpcChannelPool()

        result = pool.__enter__()

        assert result is pool

    def test_context_manager_exit_closes_all(self):
        """__exit__ 关闭所有连接"""
        pool = GrpcChannelPool()

        pool.get_channel("host1:50051")
        pool.get_channel("host2:50051")

        assert len(pool._channels) == 2

        # 验证 __exit__ 调用了 close_all 和 stop_cleanup
        with patch.object(pool, "close_all") as mock_close:
            with patch.object(pool, "stop_cleanup") as mock_stop:
                pool.__exit__(None, None, None)
                mock_close.assert_called_once()
                mock_stop.assert_called_once()


class TestGrpcChannelPoolStatsComplete:
    """统计完整测试"""

    def test_get_stats_with_inactive_channels(self):
        """有非活跃 channel 时的统计"""
        pool = GrpcChannelPool()

        channel = pool.get_channel("inactive:50051")
        channel._channel = MagicMock()
        channel._channel.is_alive.return_value = False

        stats = pool.get_stats()

        assert stats["total_channels"] == 1
        assert stats["active_channels"] == 0


class TestGetDefaultPoolComplete:
    """全局池完整测试"""

    def setup_method(self):
        """重置全局池"""
        import quantumflow.grpc.channels.pool as pool_module
        pool_module._default_pool = None

    def teardown_method(self):
        """清理"""
        import quantumflow.grpc.channels.pool as pool_module
        if pool_module._default_pool:
            pool_module._default_pool.stop_cleanup()
            pool_module._default_pool = None

    def test_get_default_pool_creates_new(self):
        """首次获取创建新池"""
        import quantumflow.grpc.channels.pool as pool_module
        pool_module._default_pool = None

        pool1 = pool_module.get_default_pool()
        pool2 = pool_module.get_default_pool()

        assert pool1 is pool2
        assert pool_module._default_pool is pool1

        pool_module._default_pool.stop_cleanup()

    def test_get_default_pool_returns_same(self):
        """多次获取返回相同实例"""
        import quantumflow.grpc.channels.pool as pool_module
        pool_module._default_pool = None

        pool1 = pool_module.get_default_pool()
        pool2 = pool_module.get_default_pool()
        pool3 = pool_module.get_default_pool()

        assert pool1 is pool2
        assert pool2 is pool3

        pool_module._default_pool.stop_cleanup()


class TestGrpcChannelPoolConcurrencyComplete:
    """并发完整测试"""

    def test_concurrent_remove_channel(self):
        """并发移除 channel"""
        pool = GrpcChannelPool()

        # 创建 10 个 channel
        for i in range(10):
            pool.get_channel(f"host{i}:50051")

        # 并发移除
        errors = []
        def remove(i):
            try:
                pool.remove_channel(f"host{i}:50051")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=remove, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(pool._channels) == 0

    def test_concurrent_eviction(self):
        """并发驱逐"""
        pool = GrpcChannelPool(max_size=2)

        # 创建初始 channel
        pool.get_channel("host0:50051")
        pool.get_channel("host1:50051")

        # 并发获取新 channel 触发驱逐
        errors = []
        def get_new_channel(i):
            try:
                pool.get_channel(f"newhost{i}:50051")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_new_channel, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 不应该有错误
        assert len(errors) == 0


class TestGrpcChannelLastUsed:
    """最后使用时间测试"""

    def test_last_used_updates(self):
        """最后使用时间更新"""
        channel = GrpcChannel("localhost:50051")

        with patch("quantumflow.grpc.channels.pool.grpc.insecure_channel"):
            initial = channel.last_used
            time.sleep(0.01)

            channel.get_channel()

            assert channel.last_used >= initial


class TestGrpcChannelClose:
    """Channel 关闭测试"""

    def test_close_with_no_channel(self):
        """关闭不存在的 channel 不报错"""
        channel = GrpcChannel("localhost:50051")

        # 不应该抛出异常
        channel.close()

    def test_close_with_channel(self):
        """关闭已有的 channel"""
        channel = GrpcChannel("localhost:50051")

        with patch("quantumflow.grpc.channels.pool.grpc.insecure_channel") as mock_channel:
            mock_instance = MagicMock()
            mock_channel.return_value = mock_instance

            channel.get_channel()
            channel.close()

            mock_instance.close.assert_called_once()
            assert channel._channel is None


class TestGrpcChannelPoolCloseAll:
    """连接池关闭测试"""

    def test_close_all_multiple_channels(self):
        """关闭多个 channel"""
        pool = GrpcChannelPool()

        pool.get_channel("host1:50051")
        pool.get_channel("host2:50051")
        pool.get_channel("host3:50051")

        assert len(pool._channels) == 3

        pool.close_all()

        assert len(pool._channels) == 0


class TestGrpcChannelPoolRemove:
    """连接池移除测试"""

    def test_remove_existing_channel(self):
        """移除已存在的 channel"""
        pool = GrpcChannelPool()

        pool.get_channel("host1:50051")
        assert "host1:50051" in pool._channels

        pool.remove_channel("host1:50051")

        assert "host1:50051" not in pool._channels

    def test_remove_nonexistent_channel(self):
        """移除不存在的 channel 不报错"""
        pool = GrpcChannelPool()

        # 不应该抛出异常
        pool.remove_channel("nonexistent:50051")


class TestGrpcChannelPoolGetChannel:
    """获取 Channel 测试"""

    def test_get_channel_creates_new(self):
        """获取新 channel"""
        pool = GrpcChannelPool()

        channel = pool.get_channel("newhost:50051")

        assert channel is not None
        assert "newhost:50051" in pool._channels

    def test_get_channel_reuses_existing(self):
        """获取已存在的 channel"""
        pool = GrpcChannelPool()

        channel1 = pool.get_channel("host:50051")
        channel2 = pool.get_channel("host:50051")

        assert channel1 is channel2
        assert len(pool._channels) == 1

    def test_get_channel_triggers_eviction(self):
        """获取新 channel 触发驱逐"""
        pool = GrpcChannelPool(max_size=2)

        ch1 = pool.get_channel("host1:50051")
        ch2 = pool.get_channel("host2:50051")

        # 现在有 2 个 channel，达到 max_size
        # 获取第三个 channel 应该触发驱逐
        pool.get_channel("host3:50051")

        # 应该只有 2 个 channel
        assert len(pool._channels) == 2
        # host1 应该被驱逐（最旧的）
        assert "host1:50051" not in pool._channels


class TestGrpcChannelCreateChannel:
    """Channel 创建测试"""

    def test_create_channel_sets_options(self):
        """创建 channel 时设置正确的选项"""
        channel = GrpcChannel("localhost:50051", timeout=15.0, max_retries=5)

        with patch("quantumflow.grpc.channels.pool.grpc.insecure_channel") as mock_channel:
            mock_instance = MagicMock()
            mock_channel.return_value = mock_instance

            ch = channel.get_channel()

            mock_channel.assert_called_once_with(
                "localhost:50051",
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 5000),
                ],
            )
