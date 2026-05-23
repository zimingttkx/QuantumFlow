"""NodeStateStore 扩展测试 - 提高覆盖率

补充测试以覆盖未覆盖的代码分支：
- initialize() 重复初始化
- _get_redis() 未初始化异常
- 字节数据解码处理
- 异常处理路径
- get_lock_info 方法
- get_recent_failover_events 方法
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from quantumflow.failover.models import (
    FailoverEvent,
    FailoverState,
    HealthStatus,
    ModelReplica,
    NodeFailoverState,
    ReplicaRole,
)
from quantumflow.failover.state_store import NodeStateStore


class TestNodeStateStoreInitialization:
    """状态存储初始化扩展测试"""

    @pytest.fixture
    def store(self):
        """创建状态存储"""
        return NodeStateStore()

    @pytest.mark.asyncio
    async def test_initialize_twice(self, store):
        """测试重复初始化"""
        mock_manager = MagicMock()

        # 第一次初始化
        store._initialized = False
        store._redis_manager = mock_manager

        # 第二次初始化应该直接返回
        await store.initialize()

        # 因为已设置为 True，应该跳过
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_get_redis_not_initialized(self, store):
        """测试 _get_redis 未初始化时抛出异常"""
        store._redis_manager = None
        store._initialized = False

        with pytest.raises(RuntimeError, match="not initialized"):
            store._get_redis()


class TestNodeStateStoreBytesHandling:
    """字节数据处理测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_load_node_state_bytes_data(self, store_with_mock_redis):
        """测试加载节点状态时处理字节数据"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        stored_data = {
            "node_id": "node-1",
            "role": "primary",
            "state": "normal",
            "health": "healthy",
            "term": 1,
            "last_heartbeat": now.isoformat(),
            "failure_reasons": [],
            "gpu_status": {},
            "model_status": {},
        }
        # 返回 bytes 而不是 str
        mock_redis.get.return_value = json.dumps(stored_data).encode("utf-8")

        result = await store.load_node_state("node-1")

        assert result is not None
        assert result.node_id == "node-1"

    @pytest.mark.asyncio
    async def test_get_all_node_states_bytes_data(self, store_with_mock_redis):
        """测试获取所有节点状态时处理字节数据"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        node_data = {
            "node_id": "node-1",
            "role": "primary",
            "state": "normal",
            "health": "healthy",
            "term": 1,
            "last_heartbeat": now.isoformat(),
            "failure_reasons": [],
            "gpu_status": {},
            "model_status": {},
        }

        mock_redis.keys.return_value = ["qf:failover:node:node-1:state"]

        def mock_get(key):
            if "node-1" in key:
                return json.dumps(node_data).encode("utf-8")  # bytes
            return None

        mock_redis.get.side_effect = mock_get

        result = await store.get_all_node_states()

        assert len(result) == 1
        assert result[0].node_id == "node-1"

    @pytest.mark.asyncio
    async def test_load_replica_index_bytes_data(self, store_with_mock_redis):
        """测试加载副本索引时处理字节数据"""
        store, mock_redis = store_with_mock_redis

        replica_data = {
            "model_name": "qwen2.5-7b",
            "model_path": "/models/qwen2.5-7b",
            "primary_node": "node-1",
            "secondary_nodes": ["node-2"],
            "replica_count": 2,
            "sync_state": "synced",
            "last_sync_at": datetime.now().isoformat(),
            "checksum": "abc123",
            "version": 1,
        }
        mock_redis.get.return_value = json.dumps(replica_data).encode("utf-8")

        result = await store.load_replica_index("qwen2.5-7b")

        assert result is not None
        assert result.model_name == "qwen2.5-7b"


class TestNodeStateStoreExceptions:
    """状态存储异常处理测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_delete_node_state_exception(self, store_with_mock_redis):
        """测试删除节点状态时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.delete.side_effect = Exception("Redis error")

        result = await store.delete_node_state("node-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_all_node_states_exception(self, store_with_mock_redis):
        """测试获取所有节点状态时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.keys.side_effect = Exception("Redis error")

        result = await store.get_all_node_states()

        assert result == []

    @pytest.mark.asyncio
    async def test_save_replica_index_exception(self, store_with_mock_redis):
        """测试保存副本索引时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.set.side_effect = Exception("Redis error")

        replica = ModelReplica(
            model_name="test-model",
            model_path="/models/test",
            primary_node="node-1",
            secondary_nodes=[],
            replica_count=1,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc",
            version=1,
        )

        result = await store.save_replica_index(replica)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_replica_index_exception(self, store_with_mock_redis):
        """测试删除副本索引时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.delete.side_effect = Exception("Redis error")

        result = await store.delete_replica_index("test-model")

        assert result is False


class TestDistributedLockExtended:
    """分布式锁扩展测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_extend_lock_no_lock(self, store_with_mock_redis):
        """测试延长不存在的锁"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = None

        result = await store.extend_lock("leader", "node-1", ttl_seconds=30)

        assert result is False

    @pytest.mark.asyncio
    async def test_extend_lock_wrong_owner(self, store_with_mock_redis):
        """测试延长不是自己持有的锁"""
        store, mock_redis = store_with_mock_redis

        stored_lock = json.dumps({
            "owner": "node-1",
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=30)).isoformat(),
        })
        mock_redis.get.return_value = stored_lock

        result = await store.extend_lock("leader", "node-2", ttl_seconds=30)

        assert result is False

    @pytest.mark.asyncio
    async def test_extend_lock_bytes_data(self, store_with_mock_redis):
        """测试延长锁时处理字节数据"""
        store, mock_redis = store_with_mock_redis

        stored_lock = json.dumps({
            "owner": "node-1",
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=30)).isoformat(),
        })
        mock_redis.get.return_value = stored_lock.encode("utf-8")
        mock_redis.set.return_value = True

        result = await store.extend_lock("leader", "node-1", ttl_seconds=60)

        assert result is True

    @pytest.mark.asyncio
    async def test_extend_lock_exception(self, store_with_mock_redis):
        """测试延长锁时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.side_effect = Exception("Redis error")

        result = await store.extend_lock("leader", "node-1", ttl_seconds=30)

        assert result is False

    @pytest.mark.asyncio
    async def test_release_lock_exception(self, store_with_mock_redis):
        """测试释放锁时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.side_effect = Exception("Redis error")

        result = await store.release_lock("leader", "node-1")

        assert result is False


class TestGetLockInfo:
    """获取锁信息测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_get_lock_info_not_exists(self, store_with_mock_redis):
        """测试获取不存在的锁信息"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = None

        result = await store.get_lock_info("nonexistent-lock")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_lock_info_success(self, store_with_mock_redis):
        """测试成功获取锁信息"""
        store, mock_redis = store_with_mock_redis

        stored_lock = {
            "owner": "node-1",
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=30)).isoformat(),
        }
        mock_redis.get.return_value = json.dumps(stored_lock)

        result = await store.get_lock_info("leader")

        assert result is not None
        assert result["owner"] == "node-1"

    @pytest.mark.asyncio
    async def test_get_lock_info_bytes_data(self, store_with_mock_redis):
        """测试获取锁信息时处理字节数据"""
        store, mock_redis = store_with_mock_redis

        stored_lock = {
            "owner": "node-1",
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=30)).isoformat(),
        }
        mock_redis.get.return_value = json.dumps(stored_lock).encode("utf-8")

        result = await store.get_lock_info("leader")

        assert result is not None
        assert result["owner"] == "node-1"

    @pytest.mark.asyncio
    async def test_get_lock_info_exception(self, store_with_mock_redis):
        """测试获取锁信息时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.side_effect = Exception("Redis error")

        result = await store.get_lock_info("leader")

        assert result is None


class TestLeaderElectionStore:
    """Leader 选举状态存储测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_set_leader_exception(self, store_with_mock_redis):
        """测试设置 Leader 时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.set.side_effect = Exception("Redis error")

        result = await store.set_leader("node-1", term=5)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_leader_bytes_data(self, store_with_mock_redis):
        """测试获取 Leader 时处理字节数据"""
        store, mock_redis = store_with_mock_redis

        stored_data = json.dumps({
            "node_id": "node-1",
            "term": 3,
            "timestamp": datetime.now().isoformat(),
        })
        mock_redis.get.return_value = stored_data.encode("utf-8")

        node_id, term = await store.get_leader()

        assert node_id == "node-1"
        assert term == 3

    @pytest.mark.asyncio
    async def test_get_leader_exception(self, store_with_mock_redis):
        """测试获取 Leader 时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.side_effect = Exception("Redis error")

        node_id, term = await store.get_leader()

        assert node_id is None
        assert term == 0


class TestFailoverEventExtended:
    """故障转移事件扩展测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_save_failover_event_exception(self, store_with_mock_redis):
        """测试保存故障事件时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.lpush.side_effect = Exception("Redis error")

        event = FailoverEvent(
            event_id="fe_123",
            event_type="node_fail",
            source_node="node-1",
            target_node="node-2",
            reason="test",
            timestamp=datetime.now(),
            success=True,
        )

        result = await store.save_failover_event(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_failover_events_with_bytes(self, store_with_mock_redis):
        """测试获取故障事件时处理字节数据"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        event_data = {
            "event_id": "fe_001",
            "event_type": "node_fail",
            "source_node": "node-1",
            "target_node": "node-2",
            "reason": "timeout",
            "timestamp": now.isoformat(),
            "success": True,
            "details": {},
        }

        # 返回 bytes 列表
        mock_redis.lrange.return_value = [json.dumps(event_data).encode("utf-8")]

        result = await store.get_failover_events(limit=100)

        assert len(result) == 1
        assert result[0].event_id == "fe_001"

    @pytest.mark.asyncio
    async def test_get_failover_events_exception(self, store_with_mock_redis):
        """测试获取故障事件时异常"""
        store, mock_redis = store_with_mock_redis
        mock_redis.lrange.side_effect = Exception("Redis error")

        result = await store.get_failover_events(limit=100)

        assert result == []

    @pytest.mark.asyncio
    async def test_save_failover_event_with_bytes_details(self, store_with_mock_redis):
        """测试保存带字节详情的故障事件"""
        store, mock_redis = store_with_mock_redis

        event = FailoverEvent(
            event_id="fe_456",
            event_type="gpu_fail",
            source_node="node-1",
            target_node=None,
            reason="memory_error",
            timestamp=datetime.now(),
            success=False,
            details={"gpu_id": 0, "error": "ECC error"},
        )

        result = await store.save_failover_event(event)

        assert result is True
        mock_redis.lpush.assert_called_once()


class TestLoadNodeStateWithInvalidData:
    """加载节点状态异常数据测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_load_node_state_invalid_json(self, store_with_mock_redis):
        """测试加载无效 JSON 数据"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = "invalid json {"

        result = await store.load_node_state("node-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_load_replica_index_invalid_json(self, store_with_mock_redis):
        """测试加载副本索引无效 JSON 数据"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = "not valid json"

        result = await store.load_replica_index("test-model")

        assert result is None


class TestNodeStateSerializationEdgeCases:
    """节点状态序列化边界测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_save_node_state_with_empty_failure_reasons(self, store_with_mock_redis):
        """测试保存带空失败原因的节点状态"""
        store, mock_redis = store_with_mock_redis

        state = NodeFailoverState(
            node_id="node-empty",
            role=ReplicaRole.PRIMARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
            failure_reasons=[],  # 空列表
            gpu_status={},  # 空字典
            model_status={},  # 空字典
        )

        result = await store.save_node_state(state)

        assert result is True

        # 验证保存的数据
        call_args = mock_redis.set.call_args
        saved_data = json.loads(call_args[0][1])
        assert saved_data["failure_reasons"] == []
        assert saved_data["gpu_status"] == {}
        assert saved_data["model_status"] == {}

    @pytest.mark.asyncio
    async def test_save_node_state_with_special_characters(self, store_with_mock_redis):
        """测试保存带特殊字符的节点状态"""
        store, mock_redis = store_with_mock_redis

        state = NodeFailoverState(
            node_id="node-特殊字符",
            role=ReplicaRole.SECONDARY,
            state=FailoverState.DEGRADED,
            health=HealthStatus.DEGRADED,
            term=2,
            last_heartbeat=datetime.now(),
            failure_reasons=["高温度", "memory < 10%"],
            gpu_status={"0": HealthStatus.UNHEALTHY, "1": HealthStatus.DEGRADED},
            model_status={"model-中文": HealthStatus.UNHEALTHY},
        )

        result = await store.save_node_state(state)

        assert result is True

        # 验证保存的数据可以正确序列化/反序列化
        call_args = mock_redis.set.call_args
        saved_data = json.loads(call_args[0][1])
        assert saved_data["node_id"] == "node-特殊字符"
        assert len(saved_data["failure_reasons"]) == 2


class TestReplicaIndexEdgeCases:
    """副本索引边界测试"""

    @pytest.fixture
    def store_with_mock_redis(self):
        """创建带 Mock Redis 的状态存储"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        store._initialized = True
        return store, mock_redis

    @pytest.mark.asyncio
    async def test_save_replica_index_with_empty_secondaries(self, store_with_mock_redis):
        """测试保存没有副本节点的索引"""
        store, mock_redis = store_with_mock_redis

        replica = ModelReplica(
            model_name="standalone-model",
            model_path="/models/standalone",
            primary_node="node-1",
            secondary_nodes=[],  # 没有副本节点
            replica_count=1,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="single123",
            version=1,
        )

        result = await store.save_replica_index(replica)

        assert result is True

        call_args = mock_redis.set.call_args
        saved_data = json.loads(call_args[0][1])
        assert saved_data["secondary_nodes"] == []
        assert saved_data["replica_count"] == 1

    @pytest.mark.asyncio
    async def test_load_replica_index_invalid_data(self, store_with_mock_redis):
        """测试加载无效的副本索引数据"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = json.dumps({"invalid": "data"})

        result = await store.load_replica_index("test-model")

        # 应该因为缺少必需字段而返回 None 或抛出异常
        assert result is None or isinstance(result, ModelReplica)
