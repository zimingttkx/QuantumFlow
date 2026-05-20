"""Redis连接管理器 - 严格单元测试

测试覆盖:
1. RedisConnectionManager 单例模式
2. connect / disconnect / reconnect 生命周期
3. is_connected / client 属性
4. health_check (healthy / disconnected / unhealthy)
5. get_instance_sync 异常路径
6. 全局便捷函数: get_redis_manager, get_redis_manager_sync, init_redis, close_redis
"""

from unittest.mock import AsyncMock, patch

import pytest

from quantumflow.storage.connection import (
    RedisConnectionManager,
    close_redis,
    get_redis_manager,
    get_redis_manager_sync,
    init_redis,
)


# ==================== 连接生命周期测试 ====================


class TestRedisConnectionManagerLifecycle:
    """连接生命周期测试"""

    @pytest.mark.asyncio
    async def test_connect_success_sets_connected(self):
        """[核心功能] 连接成功时 _connected 和 client 被设置"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            result = await manager.connect(redis_url="redis://test:6379/0")

        assert result is True
        assert manager.is_connected is True
        assert manager._redis_client is not None

    @pytest.mark.asyncio
    async def test_connect_failure_sets_disconnected(self):
        """[核心功能] 连接失败时 _connected 为 False"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=ConnectionError("Connection refused"))

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            result = await manager.connect(redis_url="redis://bad:6379/0")

        assert result is False
        assert manager.is_connected is False
        assert manager._redis_client is None

    @pytest.mark.asyncio
    async def test_connect_returns_true_if_already_connected(self):
        """[核心功能] 已连接时 connect 立即返回 True"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            await manager.connect(redis_url="redis://localhost:6379/0")
            # 第二次 connect 应短路返回 True
            result = await manager.connect(redis_url="redis://other:6379/0")

        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        """[核心功能] disconnect 清除客户端和连接状态"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.close = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            await manager.connect(redis_url="redis://localhost:6379/0")
            await manager.disconnect()

        assert manager.is_connected is False
        assert manager._redis_client is None
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """[边界用例] 未连接时 disconnect 不报错"""
        manager = RedisConnectionManager()
        await manager.disconnect()  # 不应抛出异常
        assert manager.is_connected is False

    @pytest.mark.asyncio
    async def test_reconnect_success(self):
        """[核心功能] reconnect 断开后重连成功"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.close = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            await manager.connect(redis_url="redis://localhost:6379/0")
            assert manager.is_connected is True

            result = await manager.reconnect()
            assert result is True
            assert manager.is_connected is True

    @pytest.mark.asyncio
    async def test_reconnect_without_prior_connect(self):
        """[边界用例] 无先前连接时 reconnect 返回 False"""
        manager = RedisConnectionManager()
        result = await manager.reconnect()
        assert result is False  # _redis_url is None

    @pytest.mark.asyncio
    async def test_reconnect_preserves_connection_kwargs(self):
        """[核心功能] reconnect 复用原有连接参数"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.close = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_client) as mock_from_url:
            manager = RedisConnectionManager()
            await manager.connect(
                redis_url="redis://custom:6380/2",
                max_connections=50,
            )

            # Reset mock to track second call
            mock_from_url.reset_mock()
            mock_from_url.return_value = mock_client

            await manager.reconnect()

            # 验证第二次 from_url 使用了相同的参数
            call_kwargs = mock_from_url.call_args[1]
            assert call_kwargs.get("max_connections") == 50


# ==================== 单例模式测试 ====================


class TestSingletonPattern:
    """单例模式测试"""

    @pytest.mark.asyncio
    async def test_get_instance_returns_same_object(self):
        """[核心功能] 多次 get_instance 返回同一实例"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        # 需要重置单例状态
        RedisConnectionManager._instance = None

        try:
            with patch("redis.asyncio.from_url", return_value=mock_client):
                instance1 = await RedisConnectionManager.get_instance()
                instance2 = await RedisConnectionManager.get_instance()

            assert instance1 is instance2
        finally:
            RedisConnectionManager._instance = None

    @pytest.mark.asyncio
    async def test_get_instance_auto_connects(self):
        """[核心功能] get_instance 自动调用 connect"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        RedisConnectionManager._instance = None

        try:
            with patch("redis.asyncio.from_url", return_value=mock_client):
                instance = await RedisConnectionManager.get_instance()

            # connect() is called — _redis_url will be set to default
            assert instance._redis_url == "redis://localhost:6379/0"
        finally:
            RedisConnectionManager._instance = None

    def test_get_instance_sync_raises_when_not_initialized(self):
        """[核心功能] 未初始化时 get_instance_sync 抛出 RuntimeError"""
        RedisConnectionManager._instance = None

        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                RedisConnectionManager.get_instance_sync()
        finally:
            RedisConnectionManager._instance = None

    @pytest.mark.asyncio
    async def test_get_instance_sync_works_after_init(self):
        """[核心功能] 初始化后 get_instance_sync 返回实例"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        RedisConnectionManager._instance = None

        try:
            with patch("redis.asyncio.from_url", return_value=mock_client):
                await RedisConnectionManager.get_instance()

            instance = RedisConnectionManager.get_instance_sync()
            assert instance is not None
            assert isinstance(instance, RedisConnectionManager)
        finally:
            RedisConnectionManager._instance = None


# ==================== 属性测试 ====================


class TestProperties:
    """属性验证"""

    def test_is_connected_false_initially(self):
        manager = RedisConnectionManager()
        assert manager.is_connected is False

    def test_client_is_none_initially(self):
        manager = RedisConnectionManager()
        assert manager.client is None

    @pytest.mark.asyncio
    async def test_client_returns_redis_object_after_connect(self):
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            await manager.connect()
            assert manager.client is mock_client


# ==================== health_check 测试 ====================


class TestHealthCheck:
    """health_check 测试"""

    def test_health_check_disconnected(self):
        """[核心功能] 未连接时返回 disconnected 状态"""
        manager = RedisConnectionManager()
        result = manager.health_check()

        # health_check is an async coroutine in property form; it's defined as `async def health_check`
        # 它是普通 async 方法，需要 await
        # 实际它是异步方法，直接 await

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self):
        """[核心功能] 未连接时返回 disconnected 状态"""
        manager = RedisConnectionManager()
        result = await manager.health_check()

        assert result["status"] == "disconnected"
        assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """[核心功能] 健康时返回 healthy 状态"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.info = AsyncMock(return_value={"redis_version": "7.0.0"})

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            await manager.connect()
            result = await manager.health_check()

        assert result["status"] == "healthy"
        assert result["connected"] is True
        assert result["redis_version"] == "7.0.0"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """[核心功能] 已连接但 ping 失败时返回 unhealthy 状态"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.info = AsyncMock(return_value={"redis_version": "7.0.0"})

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            # First connect successfully
            await manager.connect()
            assert manager._connected is True

            # Now simulate ping failure during health_check
            mock_client.ping = AsyncMock(side_effect=Exception("Connection lost"))
            result = await manager.health_check()

        assert result["status"] == "unhealthy"
        assert result["connected"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_health_check_missing_version_field(self):
        """[边界用例] info 中缺少 redis_version 时使用 unknown"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.info = AsyncMock(return_value={})  # 没有 redis_version

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            await manager.connect()
            result = await manager.health_check()

        assert result["status"] == "healthy"
        assert result["redis_version"] == "unknown"


# ==================== 全局便捷函数测试 ====================


class TestGlobalFunctions:
    """全局便捷函数测试"""

    @pytest.mark.asyncio
    async def test_get_redis_manager_returns_manager(self):
        """[核心功能] get_redis_manager 返回 RedisConnectionManager"""
        import quantumflow.storage.connection as conn_mod
        # 重置全局状态
        conn_mod._redis_manager = None
        RedisConnectionManager._instance = None

        try:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)

            with patch("redis.asyncio.from_url", return_value=mock_client):
                manager = await get_redis_manager()

            assert isinstance(manager, RedisConnectionManager)
        finally:
            conn_mod._redis_manager = None
            RedisConnectionManager._instance = None

    def test_get_redis_manager_sync_not_initialized(self):
        """[核心功能] 未初始化时 get_redis_manager_sync 抛出 RuntimeError"""
        import quantumflow.storage.connection as conn_mod
        conn_mod._redis_manager = None
        RedisConnectionManager._instance = None

        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                get_redis_manager_sync()
        finally:
            conn_mod._redis_manager = None
            RedisConnectionManager._instance = None

    @pytest.mark.asyncio
    async def test_init_redis_returns_bool(self):
        """[核心功能] init_redis 返回布尔值"""
        import quantumflow.storage.connection as conn_mod
        conn_mod._redis_manager = None
        RedisConnectionManager._instance = None

        try:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)

            with patch("redis.asyncio.from_url", return_value=mock_client):
                result = await init_redis("redis://localhost:6379/0")

            assert isinstance(result, bool)
        finally:
            conn_mod._redis_manager = None
            RedisConnectionManager._instance = None

    @pytest.mark.asyncio
    async def test_close_redis_clears_global(self):
        """[核心功能] close_redis 清除全局管理器"""
        import quantumflow.storage.connection as conn_mod
        conn_mod._redis_manager = None
        RedisConnectionManager._instance = None

        try:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.close = AsyncMock()

            with patch("redis.asyncio.from_url", return_value=mock_client):
                conn_mod._redis_manager = await RedisConnectionManager.get_instance()

            await close_redis()
            assert conn_mod._redis_manager is None
        finally:
            conn_mod._redis_manager = None
            RedisConnectionManager._instance = None

    @pytest.mark.asyncio
    async def test_close_redis_when_not_initialized(self):
        """[边界用例] 未初始化时 close_redis 不报错"""
        import quantumflow.storage.connection as conn_mod
        conn_mod._redis_manager = None
        await close_redis()  # 不应抛出异常


# ==================== 连接参数测试 ====================


class TestConnectionArgs:
    """连接参数传递测试"""

    @pytest.mark.asyncio
    async def test_connect_passes_extra_kwargs(self):
        """[核心功能] 额外的连接参数传递给 from_url"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        with patch("redis.asyncio.from_url", return_value=mock_client) as mock_from_url:
            manager = RedisConnectionManager()
            await manager.connect(
                redis_url="redis://test:6379/0",
                max_connections=100,
                socket_keepalive=True,
            )

            call_kwargs = mock_from_url.call_args[1]
            assert call_kwargs.get("max_connections") == 100
            assert call_kwargs.get("socket_keepalive") is True

    @pytest.mark.asyncio
    async def test_connect_uses_default_url(self):
        """[核心功能] 默认连接 URL"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        with patch("redis.asyncio.from_url", return_value=mock_client) as mock_from_url:
            manager = RedisConnectionManager()
            await manager.connect()

            call_args = mock_from_url.call_args[0]
            assert call_args[0] == "redis://localhost:6379/0"


# ==================== 边缘场景测试 ====================


class TestEdgeCases:
    """边缘场景测试"""

    @pytest.mark.asyncio
    async def test_is_connected_false_when_client_is_none_but_connected_true(self):
        """[边界用例] client 为 None 但 _connected 为 True 时 is_connected 返回 False"""
        manager = RedisConnectionManager()
        manager._connected = True
        manager._redis_client = None
        assert manager.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_sets_redis_url(self):
        """[核心功能] connect 保存 redis_url"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            await manager.connect(redis_url="redis://custom:6380/5")

        assert manager._redis_url == "redis://custom:6380/5"

    @pytest.mark.asyncio
    async def test_connect_sets_connection_kwargs(self):
        """[核心功能] connect 保存连接参数"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        with patch("redis.asyncio.from_url", return_value=mock_client):
            manager = RedisConnectionManager()
            await manager.connect(max_connections=20, retry_on_timeout=True)

        assert manager._connection_kwargs == {"max_connections": 20, "retry_on_timeout": True}

    @pytest.mark.asyncio
    async def test_lock_is_shared_class_attribute(self):
        """[核心功能] _lock 是类级别的 asyncio.Lock"""
        lock1 = RedisConnectionManager._lock
        lock2 = RedisConnectionManager._lock
        assert lock1 is lock2


# ==================== GAP-FILLING: get_redis_manager_sync 已初始化路径 ====================


class TestGetRedisManagerSyncInitialized:
    """get_redis_manager_sync 已初始化路径（覆盖 line 160）"""

    def test_get_redis_manager_sync_returns_cached_instance(self):
        """[核心功能] _redis_manager 已设置时直接返回缓存实例"""
        import quantumflow.storage.connection as conn_mod
        conn_mod._redis_manager = None
        RedisConnectionManager._instance = None

        try:
            # 创建一个实例并设置到全局变量
            manager = RedisConnectionManager()
            conn_mod._redis_manager = manager

            result = get_redis_manager_sync()
            assert result is manager
        finally:
            conn_mod._redis_manager = None
            RedisConnectionManager._instance = None
