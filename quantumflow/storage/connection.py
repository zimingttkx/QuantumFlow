"""Redis连接管理器

提供全局Redis连接单例，支持分布式存储和任务队列。
"""

from typing import Optional
import asyncio
import structlog

logger = structlog.get_logger().bind(component="redis_manager")


class RedisConnectionManager:
    """
    Redis连接管理器（单例）

    负责：
    - Redis连接生命周期管理
    - 连接池维护
    - 自动重连
    """

    _instance: Optional["RedisConnectionManager"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        self._redis_client = None
        self._connected = False
        self._redis_url: Optional[str] = None
        self._connection_kwargs: dict = {}

    @classmethod
    async def get_instance(cls) -> "RedisConnectionManager":
        """获取单例实例（异步安全）"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.connect()
        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> "RedisConnectionManager":
        """同步获取单例实例（仅在已初始化时使用）"""
        if cls._instance is None:
            raise RuntimeError("RedisConnectionManager not initialized. Call get_instance() first.")
        return cls._instance

    async def connect(
        self,
        redis_url: str = "redis://localhost:6379/0",
        **kwargs,
    ) -> bool:
        """
        连接到Redis

        Args:
            redis_url: Redis连接URL
            **kwargs: 额外连接参数

        Returns:
            是否连接成功
        """
        if self._connected and self._redis_client is not None:
            return True

        self._redis_url = redis_url
        self._connection_kwargs = kwargs

        try:
            import redis.asyncio as redis

            self._redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                **kwargs,
            )

            # 测试连接
            await self._redis_client.ping()
            self._connected = True

            logger.info("redis_manager_connected", redis_url=redis_url)
            return True

        except Exception as e:
            logger.error("redis_manager_connect_failed", error=str(e), redis_url=redis_url)
            self._connected = False
            self._redis_client = None
            return False

    async def disconnect(self):
        """断开Redis连接"""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
            self._connected = False
            logger.info("redis_manager_disconnected")

    async def reconnect(self) -> bool:
        """重新连接"""
        if self._redis_url is None:
            return False

        await self.disconnect()
        return await self.connect(self._redis_url, **self._connection_kwargs)

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected and self._redis_client is not None

    @property
    def client(self):
        """获取Redis客户端"""
        return self._redis_client

    async def health_check(self) -> dict:
        """健康检查"""
        if not self.is_connected:
            return {"status": "disconnected", "connected": False}

        try:
            import redis.asyncio as redis

            info = await self._redis_client.info("server")
            await self._redis_client.ping()

            return {
                "status": "healthy",
                "connected": True,
                "redis_version": info.get("redis_version", "unknown"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }


# 全局获取函数
_redis_manager: Optional[RedisConnectionManager] = None


async def get_redis_manager() -> RedisConnectionManager:
    """获取Redis连接管理器"""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = await RedisConnectionManager.get_instance()
    return _redis_manager


def get_redis_manager_sync() -> RedisConnectionManager:
    """同步获取Redis连接管理器"""
    global _redis_manager
    if _redis_manager is None:
        return RedisConnectionManager.get_instance_sync()
    return _redis_manager


async def init_redis(redis_url: str = "redis://localhost:6379/0") -> bool:
    """初始化Redis连接"""
    manager = await get_redis_manager()
    return manager.is_connected


async def close_redis():
    """关闭Redis连接"""
    global _redis_manager
    if _redis_manager:
        await _redis_manager.disconnect()
        _redis_manager = None
