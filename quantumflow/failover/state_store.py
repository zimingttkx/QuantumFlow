"""节点状态存储

使用 Redis 存储节点故障转移状态，支持分布式环境下的状态同步。
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import structlog

from quantumflow.failover.models import (
    FailoverEvent,
    ModelReplica,
    NodeFailoverState,
    ReplicaRole,
)
from quantumflow.storage.connection import RedisConnectionManager

logger = structlog.get_logger().bind(component="failover_state_store")

# Redis key 前缀
FAILOVER_PREFIX = "qf:failover"
NODE_STATE_KEY = f"{FAILOVER_PREFIX}:node:{{node_id}}:state"
REPLICA_INDEX_KEY = f"{FAILOVER_PREFIX}:replica:index:{{model_name}}"
FAILOVER_LOCK_KEY = f"{FAILOVER_PREFIX}:lock:{{resource}}"
FAILOVER_EVENT_KEY = f"{FAILOVER_PREFIX}:events"
FAILOVER_LEADER_KEY = f"{FAILOVER_PREFIX}:leader"


class NodeStateStore:
    """
    节点状态存储

    使用 Redis 存储：
    - 节点故障转移状态
    - 模型副本索引
    - 分布式锁
    - 故障事件日志
    """

    def __init__(self, redis_manager: RedisConnectionManager | None = None):
        self._redis_manager = redis_manager
        self._initialized = False

    async def initialize(self) -> None:
        """初始化状态存储"""
        if self._initialized:
            return

        if self._redis_manager is None:
            from quantumflow.storage.connection import get_redis_manager

            self._redis_manager = await get_redis_manager()

        self._initialized = True
        logger.info("failover_state_store_initialized")

    def _get_redis(self):
        """获取 Redis 客户端"""
        if self._redis_manager is None:
            raise RuntimeError("State store not initialized. Call initialize() first.")
        return self._redis_manager.get_client()

    # ==================== 节点状态 ====================

    async def save_node_state(self, state: NodeFailoverState) -> bool:
        """
        保存节点状态

        Args:
            state: 节点故障转移状态

        Returns:
            是否保存成功
        """
        try:
            redis = self._get_redis()
            key = NODE_STATE_KEY.format(node_id=state.node_id)
            data = state.to_dict()
            redis.set(key, json.dumps(data))
            logger.debug("node_state_saved", node_id=state.node_id)
            return True
        except Exception as e:
            logger.error("node_state_save_failed", node_id=state.node_id, error=str(e))
            return False

    async def load_node_state(self, node_id: str) -> NodeFailoverState | None:
        """
        加载节点状态

        Args:
            node_id: 节点 ID

        Returns:
            节点状态，不存在返回 None
        """
        try:
            redis = self._get_redis()
            key = NODE_STATE_KEY.format(node_id=node_id)
            data = redis.get(key)
            if data is None:
                return None

            if isinstance(data, bytes):
                data = data.decode()

            return NodeFailoverState.from_dict(json.loads(data))
        except Exception as e:
            logger.error(
                "node_state_load_failed", node_id=node_id, error=str(e)
            )
            return None

    async def delete_node_state(self, node_id: str) -> bool:
        """
        删除节点状态

        Args:
            node_id: 节点 ID

        Returns:
            是否删除成功
        """
        try:
            redis = self._get_redis()
            key = NODE_STATE_KEY.format(node_id=node_id)
            redis.delete(key)
            logger.debug("node_state_deleted", node_id=node_id)
            return True
        except Exception as e:
            logger.error(
                "node_state_delete_failed", node_id=node_id, error=str(e)
            )
            return False

    async def get_all_node_states(self) -> list[NodeFailoverState]:
        """
        获取所有节点状态

        Returns:
            所有节点状态列表
        """
        try:
            redis = self._get_redis()
            pattern = NODE_STATE_KEY.format(node_id="*")
            keys = redis.keys(pattern)

            states = []
            for key in keys:
                data = redis.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode()
                    states.append(NodeFailoverState.from_dict(json.loads(data)))

            return states
        except Exception as e:
            logger.error("get_all_node_states_failed", error=str(e))
            return []

    # ==================== 模型副本索引 ====================

    async def save_replica_index(self, replica: ModelReplica) -> bool:
        """
        保存模型副本索引

        Args:
            replica: 模型副本信息

        Returns:
            是否保存成功
        """
        try:
            redis = self._get_redis()
            key = REPLICA_INDEX_KEY.format(model_name=replica.model_name)
            data = replica.to_dict()
            redis.set(key, json.dumps(data))
            logger.debug("replica_index_saved", model_name=replica.model_name)
            return True
        except Exception as e:
            logger.error(
                "replica_index_save_failed",
                model_name=replica.model_name,
                error=str(e),
            )
            return False

    async def load_replica_index(
        self, model_name: str
    ) -> ModelReplica | None:
        """
        加载模型副本索引

        Args:
            model_name: 模型名称

        Returns:
            模型副本信息，不存在返回 None
        """
        try:
            redis = self._get_redis()
            key = REPLICA_INDEX_KEY.format(model_name=model_name)
            data = redis.get(key)
            if data is None:
                return None

            if isinstance(data, bytes):
                data = data.decode()

            return ModelReplica.from_dict(json.loads(data))
        except Exception as e:
            logger.error(
                "replica_index_load_failed", model_name=model_name, error=str(e)
            )
            return None

    async def delete_replica_index(self, model_name: str) -> bool:
        """
        删除模型副本索引

        Args:
            model_name: 模型名称

        Returns:
            是否删除成功
        """
        try:
            redis = self._get_redis()
            key = REPLICA_INDEX_KEY.format(model_name=model_name)
            redis.delete(key)
            logger.debug("replica_index_deleted", model_name=model_name)
            return True
        except Exception as e:
            logger.error(
                "replica_index_delete_failed",
                model_name=model_name,
                error=str(e),
            )
            return False

    async def get_all_replica_indexes(self) -> list[ModelReplica]:
        """
        获取所有模型副本索引

        Returns:
            所有模型副本信息列表
        """
        try:
            redis = self._get_redis()
            pattern = REPLICA_INDEX_KEY.format(model_name="*")
            keys = redis.keys(pattern)

            replicas = []
            for key in keys:
                data = redis.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode()
                    replicas.append(ModelReplica.from_dict(json.loads(data)))

            return replicas
        except Exception as e:
            logger.error("get_all_replica_indexes_failed", error=str(e))
            return []

    # ==================== 分布式锁 ====================

    async def acquire_lock(
        self, resource: str, owner: str, ttl_seconds: int = 30
    ) -> bool:
        """
        获取分布式锁

        Args:
            resource: 资源名称
            owner: 锁持有者 ID
            ttl_seconds: 锁过期时间（秒）

        Returns:
            是否成功获取锁
        """
        try:
            redis = self._get_redis()
            key = FAILOVER_LOCK_KEY.format(resource=resource)
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl_seconds)

            lock_data = json.dumps({
                "owner": owner,
                "acquired_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
            })

            # 使用 SET NX（不存在时才设置）实现原子性
            result = redis.set(key, lock_data, nx=True, ex=ttl_seconds)
            if result:
                logger.debug("lock_acquired", resource=resource, owner=owner)
            return bool(result)
        except Exception as e:
            logger.error("lock_acquire_failed", resource=resource, error=str(e))
            return False

    RELEASE_LOCK_LUA = """
    local data = redis.call('GET', KEYS[1])
    if data then
        local decoded = cjson.decode(data)
        if decoded and decoded['owner'] == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
    end
    return 0
    """

    async def release_lock(self, resource: str, owner: str) -> bool:
        """
        释放分布式锁（仅当持有者匹配时）

        Args:
            resource: 资源名称
            owner: 锁持有者 ID

        Returns:
            是否成功释放锁
        """
        try:
            redis = self._get_redis()
            key = FAILOVER_LOCK_KEY.format(resource=resource)

            # 尝试用 Lua 脚本原子性地检查 owner 并删除
            try:
                result = redis.eval(self.RELEASE_LOCK_LUA, 1, key, owner)
                if result:
                    logger.debug("lock_released", resource=resource, owner=owner)
                    return True
                else:
                    logger.warning(
                        "lock_release_denied",
                        resource=resource,
                        owner=owner,
                    )
                    return False
            except Exception:
                # fakeredis 等环境可能不支持 Lua eval，回退到非原子操作
                data = redis.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode()
                    lock_info = json.loads(data)
                    if lock_info.get("owner") == owner:
                        redis.delete(key)
                        logger.debug("lock_released_fallback", resource=resource, owner=owner)
                        return True
                logger.warning(
                    "lock_release_denied_fallback",
                    resource=resource,
                    owner=owner,
                )
                return False
        except Exception as e:
            logger.error("lock_release_failed", resource=resource, error=str(e))
            return False

    EXTEND_LOCK_LUA = """
    local data = redis.call('GET', KEYS[1])
    if data then
        local decoded = cjson.decode(data)
        if decoded and decoded['owner'] == ARGV[1] then
            return redis.call('EXPIRE', KEYS[1], ARGV[2])
        end
    end
    return 0
    """

    async def extend_lock(
        self, resource: str, owner: str, ttl_seconds: int = 30
    ) -> bool:
        """
        延长锁的过期时间

        Args:
            resource: 资源名称
            owner: 锁持有者 ID
            ttl_seconds: 新的过期时间（秒）

        Returns:
            是否成功延长
        """
        try:
            redis = self._get_redis()
            key = FAILOVER_LOCK_KEY.format(resource=resource)

            # 尝试用 Lua 脚本原子性地检查 owner 并延长 TTL
            try:
                result = redis.eval(self.EXTEND_LOCK_LUA, 1, key, owner, ttl_seconds)
                if result:
                    logger.debug("lock_extended", resource=resource, owner=owner)
                    return True
                else:
                    return False
            except Exception:
                # fakeredis 等环境可能不支持 Lua eval，回退到非原子操作
                data = redis.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode()
                    lock_info = json.loads(data)
                    if lock_info.get("owner") == owner:
                        redis.expire(key, ttl_seconds)
                        logger.debug("lock_extended_fallback", resource=resource, owner=owner)
                        return True
                return False
        except Exception as e:
            logger.error("lock_extend_failed", resource=resource, error=str(e))
            return False

    async def get_lock_info(self, resource: str) -> dict[str, Any] | None:
        """
        获取锁信息

        Args:
            resource: 资源名称

        Returns:
            锁信息，不存在返回 None
        """
        try:
            redis = self._get_redis()
            key = FAILOVER_LOCK_KEY.format(resource=resource)
            data = redis.get(key)
            if data is None:
                return None

            if isinstance(data, bytes):
                data = data.decode()

            return json.loads(data)
        except Exception as e:
            logger.error("get_lock_info_failed", resource=resource, error=str(e))
            return None

    # ==================== Leader 选举 ====================

    async def set_leader(self, node_id: str, term: int) -> bool:
        """
        设置当前 Leader

        Args:
            node_id: Leader 节点 ID
            term: 任期号

        Returns:
            是否设置成功
        """
        try:
            redis = self._get_redis()
            data = json.dumps({
                "node_id": node_id,
                "term": term,
                "timestamp": datetime.now().isoformat(),
            })
            redis.set(FAILOVER_LEADER_KEY, data)
            logger.info("leader_set", node_id=node_id, term=term)
            return True
        except Exception as e:
            logger.error("set_leader_failed", error=str(e))
            return False

    async def get_leader(self) -> tuple[str | None, int]:
        """
        获取当前 Leader

        Returns:
            (leader_node_id, term)
        """
        try:
            redis = self._get_redis()
            data = redis.get(FAILOVER_LEADER_KEY)
            if data is None:
                return None, 0

            if isinstance(data, bytes):
                data = data.decode()

            info = json.loads(data)
            return info["node_id"], info["term"]
        except Exception as e:
            logger.error("get_leader_failed", error=str(e))
            return None, 0

    # ==================== 故障事件 ====================

    async def save_failover_event(self, event: FailoverEvent) -> bool:
        """
        保存故障转移事件

        Args:
            event: 故障转移事件

        Returns:
            是否保存成功
        """
        try:
            redis = self._get_redis()
            event_data = event.to_dict()
            # 使用 List 存储事件，LPUSH 添加到列表头部
            redis.lpush(FAILOVER_EVENT_KEY, json.dumps(event_data))
            # 保留最近 1000 条事件
            redis.ltrim(FAILOVER_EVENT_KEY, 0, 999)
            logger.debug("failover_event_saved", event_id=event.event_id)
            return True
        except Exception as e:
            logger.error(
                "failover_event_save_failed",
                event_id=event.event_id,
                error=str(e),
            )
            return False

    async def get_failover_events(
        self, limit: int = 100
    ) -> list[FailoverEvent]:
        """
        获取故障转移事件

        Args:
            limit: 返回事件数量

        Returns:
            故障转移事件列表
        """
        try:
            redis = self._get_redis()
            events_data = redis.lrange(FAILOVER_EVENT_KEY, 0, limit - 1)

            events = []
            for data in events_data:
                if isinstance(data, bytes):
                    data = data.decode()
                events.append(FailoverEvent.from_dict(json.loads(data)))

            return events
        except Exception as e:
            logger.error("get_failover_events_failed", error=str(e))
            return []

    # ==================== 工具方法 ====================

    def generate_event_id(self) -> str:
        """生成唯一事件 ID"""
        return f"fe_{uuid.uuid4().hex[:12]}"

    async def close(self) -> None:
        """关闭状态存储"""
        # Redis 连接由 RedisConnectionManager 管理，这里不需要额外清理
        self._initialized = False
        logger.info("failover_state_store_closed")


# 全局单例
_state_store: NodeStateStore | None = None


async def get_state_store() -> NodeStateStore:
    """获取全局状态存储实例"""
    global _state_store
    if _state_store is None:
        _state_store = NodeStateStore()
        await _state_store.initialize()
    return _state_store
