"""NodeStateStore 状态存储测试

验证 Redis 状态存储的业务逻辑正确性：
- 节点状态 CRUD 操作
- 模型副本索引管理
- 分布式锁的原子性和互斥性
- Leader 选举状态管理
- 故障事件记录
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
    """状态存储初始化测试"""

    def test_init_without_redis_manager(self):
        """测试不使用 Redis Manager 初始化"""
        store = NodeStateStore()
        assert store._redis_manager is None
        assert store._initialized is False

    def test_init_with_redis_manager(self):
        """测试使用 Redis Manager 初始化"""
        mock_redis = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client.return_value = mock_redis

        store = NodeStateStore(redis_manager=mock_manager)
        assert store._redis_manager is mock_manager
        assert store._initialized is False


class TestNodeStateCRUD:
    """节点状态 CRUD 操作测试"""

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
    async def test_save_node_state_success(self, store_with_mock_redis):
        """测试保存节点状态成功"""
        store, mock_redis = store_with_mock_redis

        state = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.PRIMARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
        )

        result = await store.save_node_state(state)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert "qf:failover:node:node-1:state" in call_args[0]

    @pytest.mark.asyncio
    async def test_save_node_state_serializes_correctly(self, store_with_mock_redis):
        """测试节点状态序列化正确性"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        state = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.PRIMARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=5,
            last_heartbeat=now,
            failure_reasons=["gpu_overheat", "memory_full"],
            gpu_status={"0": HealthStatus.DEGRADED},
        )

        await store.save_node_state(state)

        call_args = mock_redis.set.call_args
        saved_data = json.loads(call_args[0][1])

        # 验证序列化完整性
        assert saved_data["node_id"] == "node-1"
        assert saved_data["role"] == "primary"
        assert saved_data["state"] == "normal"
        assert saved_data["health"] == "healthy"
        assert saved_data["term"] == 5
        assert len(saved_data["failure_reasons"]) == 2
        assert saved_data["gpu_status"]["0"] == "degraded"

    @pytest.mark.asyncio
    async def test_load_node_state_not_found(self, store_with_mock_redis):
        """测试加载不存在的节点状态"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = None

        result = await store.load_node_state("nonexistent-node")

        assert result is None

    @pytest.mark.asyncio
    async def test_load_node_state_deserializes_correctly(self, store_with_mock_redis):
        """测试节点状态反序列化正确性"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        stored_data = {
            "node_id": "node-1",
            "role": "secondary",
            "state": "degraded",
            "health": "degraded",
            "term": 3,
            "last_heartbeat": now.isoformat(),
            "failure_reasons": ["timeout"],
            "gpu_status": {"0": "healthy", "1": "unhealthy"},
            "model_status": {"model-a": "healthy"},
        }
        mock_redis.get.return_value = json.dumps(stored_data)

        result = await store.load_node_state("node-1")

        assert result is not None
        assert result.node_id == "node-1"
        assert result.role == ReplicaRole.SECONDARY
        assert result.state == FailoverState.DEGRADED
        assert result.health == HealthStatus.DEGRADED
        assert result.term == 3
        assert len(result.gpu_status) == 2
        assert result.gpu_status["1"] == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_delete_node_state_success(self, store_with_mock_redis):
        """测试删除节点状态成功"""
        store, mock_redis = store_with_mock_redis

        result = await store.delete_node_state("node-1")

        assert result is True
        mock_redis.delete.assert_called_once_with("qf:failover:node:node-1:state")

    @pytest.mark.asyncio
    async def test_get_all_node_states_empty(self, store_with_mock_redis):
        """测试获取所有节点状态为空"""
        store, mock_redis = store_with_mock_redis
        mock_redis.keys.return_value = []

        result = await store.get_all_node_states()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_node_states_multiple_nodes(self, store_with_mock_redis):
        """测试获取多个节点状态"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        node1_data = {
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
        node2_data = {
            "node_id": "node-2",
            "role": "secondary",
            "state": "degraded",
            "health": "degraded",
            "term": 1,
            "last_heartbeat": now.isoformat(),
            "failure_reasons": ["gpu_fail"],
            "gpu_status": {},
            "model_status": {},
        }

        def mock_get(key):
            if "node-1" in key:
                return json.dumps(node1_data)
            elif "node-2" in key:
                return json.dumps(node2_data)
            return json.dumps(node1_data)

        mock_redis.keys.return_value = [
            "qf:failover:node:node-1:state",
            "qf:failover:node:node-2:state",
        ]
        mock_redis.get.side_effect = mock_get

        result = await store.get_all_node_states()

        assert len(result) == 2
        node_ids = {r.node_id for r in result}
        assert "node-1" in node_ids
        assert "node-2" in node_ids


class TestModelReplicaIndex:
    """模型副本索引测试"""

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
    async def test_save_replica_index_success(self, store_with_mock_redis):
        """测试保存副本索引成功"""
        store, mock_redis = store_with_mock_redis

        replica = ModelReplica(
            model_name="qwen2.5-7b",
            model_path="/models/qwen2.5-7b",
            primary_node="node-1",
            secondary_nodes=["node-2", "node-3"],
            replica_count=3,
            sync_state="synced",
            last_sync_at=datetime.now(),
            checksum="abc123def456",
            version=1,
        )

        result = await store.save_replica_index(replica)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert "qf:failover:replica:index:qwen2.5-7b" in call_args[0]

    @pytest.mark.asyncio
    async def test_save_replica_index_serializes_correctly(self, store_with_mock_redis):
        """测试副本索引序列化正确性"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        replica = ModelReplica(
            model_name="llama-7b",
            model_path="/models/llama-7b",
            primary_node="node-a",
            secondary_nodes=["node-b"],
            replica_count=2,
            sync_state="syncing",
            last_sync_at=now,
            checksum="xyz789",
            version=5,
        )

        await store.save_replica_index(replica)

        call_args = mock_redis.set.call_args
        saved_data = json.loads(call_args[0][1])

        assert saved_data["model_name"] == "llama-7b"
        assert saved_data["primary_node"] == "node-a"
        assert saved_data["secondary_nodes"] == ["node-b"]
        assert saved_data["sync_state"] == "syncing"
        assert saved_data["version"] == 5

    @pytest.mark.asyncio
    async def test_load_replica_index_not_found(self, store_with_mock_redis):
        """测试加载不存在的副本索引"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = None

        result = await store.load_replica_index("nonexistent-model")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_replica_index_success(self, store_with_mock_redis):
        """测试删除副本索引成功"""
        store, mock_redis = store_with_mock_redis

        result = await store.delete_replica_index("qwen2.5-7b")

        assert result is True
        mock_redis.delete.assert_called_once()


class TestDistributedLock:
    """分布式锁测试"""

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
    async def test_acquire_lock_success(self, store_with_mock_redis):
        """测试成功获取锁"""
        store, mock_redis = store_with_mock_redis
        mock_redis.set.return_value = True

        result = await store.acquire_lock("leader", "node-1", ttl_seconds=30)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        # 验证 NX 和 EX 参数
        assert call_args[1]["nx"] is True
        assert call_args[1]["ex"] == 30

    @pytest.mark.asyncio
    async def test_acquire_lock_failure_already_held(self, store_with_mock_redis):
        """测试获取锁失败（已被持有）"""
        store, mock_redis = store_with_mock_redis
        mock_redis.set.return_value = None  # 锁已被持有

        result = await store.acquire_lock("leader", "node-2", ttl_seconds=30)

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_lock_serializes_data_correctly(self, store_with_mock_redis):
        """测试锁数据序列化正确性"""
        store, mock_redis = store_with_mock_redis
        mock_redis.set.return_value = True

        await store.acquire_lock("test-resource", "node-1", ttl_seconds=60)

        call_args = mock_redis.set.call_args
        saved_data = json.loads(call_args[0][1])

        assert saved_data["owner"] == "node-1"
        assert "acquired_at" in saved_data
        assert "expires_at" in saved_data

    @pytest.mark.asyncio
    async def test_release_lock_success(self, store_with_mock_redis):
        """测试成功释放锁"""
        store, mock_redis = store_with_mock_redis

        stored_lock = json.dumps({
            "owner": "node-1",
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=30)).isoformat(),
        })
        mock_redis.get.return_value = stored_lock

        result = await store.release_lock("leader", "node-1")

        assert result is True
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_lock_denied_wrong_owner(self, store_with_mock_redis):
        """测试释放锁被拒绝（不是持有者）"""
        store, mock_redis = store_with_mock_redis

        stored_lock = json.dumps({
            "owner": "node-1",
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=30)).isoformat(),
        })
        mock_redis.get.return_value = stored_lock

        # node-2 尝试释放 node-1 的锁
        result = await store.release_lock("leader", "node-2")

        assert result is False
        mock_redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_release_lock_already_expired(self, store_with_mock_redis):
        """测试释放已过期的锁"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = None  # 锁已不存在

        result = await store.release_lock("leader", "node-1")

        assert result is True  # 视为成功

    @pytest.mark.asyncio
    async def test_extend_lock_success(self, store_with_mock_redis):
        """测试成功延长锁"""
        store, mock_redis = store_with_mock_redis

        stored_lock = json.dumps({
            "owner": "node-1",
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=10)).isoformat(),
        })
        mock_redis.get.return_value = stored_lock
        mock_redis.set.return_value = True

        result = await store.extend_lock("leader", "node-1", ttl_seconds=60)

        assert result is True

    @pytest.mark.asyncio
    async def test_extend_lock_wrong_owner(self, store_with_mock_redis):
        """测试延长锁失败（不是持有者）"""
        store, mock_redis = store_with_mock_redis

        stored_lock = json.dumps({
            "owner": "node-1",
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=30)).isoformat(),
        })
        mock_redis.get.return_value = stored_lock

        result = await store.extend_lock("leader", "node-2", ttl_seconds=60)

        assert result is False


class TestLeaderElection:
    """Leader 选举状态测试"""

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
    async def test_set_leader_success(self, store_with_mock_redis):
        """测试设置 Leader 成功"""
        store, mock_redis = store_with_mock_redis

        result = await store.set_leader("node-1", term=5)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "qf:failover:leader"

        saved_data = json.loads(call_args[0][1])
        assert saved_data["node_id"] == "node-1"
        assert saved_data["term"] == 5

    @pytest.mark.asyncio
    async def test_get_leader_exists(self, store_with_mock_redis):
        """测试获取存在的 Leader"""
        store, mock_redis = store_with_mock_redis

        stored_data = json.dumps({
            "node_id": "node-1",
            "term": 3,
            "timestamp": datetime.now().isoformat(),
        })
        mock_redis.get.return_value = stored_data

        node_id, term = await store.get_leader()

        assert node_id == "node-1"
        assert term == 3

    @pytest.mark.asyncio
    async def test_get_leader_not_exists(self, store_with_mock_redis):
        """测试获取不存在的 Leader"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = None

        node_id, term = await store.get_leader()

        assert node_id is None
        assert term == 0


class TestFailoverEvent:
    """故障转移事件测试"""

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
    async def test_save_failover_event_success(self, store_with_mock_redis):
        """测试保存故障转移事件成功"""
        store, mock_redis = store_with_mock_redis

        event = FailoverEvent(
            event_id="fe_123456",
            event_type="node_fail",
            source_node="node-1",
            target_node="node-2",
            reason="heartbeat_timeout",
            timestamp=datetime.now(),
            success=True,
            details={"model": "qwen2.5-7b"},
        )

        result = await store.save_failover_event(event)

        assert result is True
        mock_redis.lpush.assert_called_once()
        mock_redis.ltrim.assert_called_once_with("qf:failover:events", 0, 999)

    @pytest.mark.asyncio
    async def test_save_failover_event_serializes_correctly(self, store_with_mock_redis):
        """测试故障事件序列化正确性"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        event = FailoverEvent(
            event_id="fe_abc789",
            event_type="gpu_fail",
            source_node="node-2",
            target_node=None,
            reason="temperature_exceeded",
            timestamp=now,
            success=False,
            details={"gpu_id": 0, "temperature": 95.0},
        )

        await store.save_failover_event(event)

        call_args = mock_redis.lpush.call_args
        saved_data = json.loads(call_args[0][1])

        assert saved_data["event_id"] == "fe_abc789"
        assert saved_data["event_type"] == "gpu_fail"
        assert saved_data["source_node"] == "node-2"
        assert saved_data["target_node"] is None
        assert saved_data["success"] is False
        assert saved_data["details"]["temperature"] == 95.0

    @pytest.mark.asyncio
    async def test_get_failover_events_empty(self, store_with_mock_redis):
        """测试获取空事件列表"""
        store, mock_redis = store_with_mock_redis
        mock_redis.lrange.return_value = []

        result = await store.get_failover_events(limit=100)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_failover_events_multiple(self, store_with_mock_redis):
        """测试获取多个事件"""
        store, mock_redis = store_with_mock_redis

        now = datetime.now()
        event1 = FailoverEvent(
            event_id="fe_001",
            event_type="node_fail",
            source_node="node-1",
            target_node="node-2",
            reason="timeout",
            timestamp=now,
            success=True,
        )
        event2 = FailoverEvent(
            event_id="fe_002",
            event_type="manual",
            source_node="node-2",
            target_node=None,
            reason="admin_request",
            timestamp=now,
            success=False,
        )

        mock_redis.lrange.return_value = [
            json.dumps(event1.to_dict()),
            json.dumps(event2.to_dict()),
        ]

        result = await store.get_failover_events(limit=10)

        assert len(result) == 2
        assert result[0].event_id == "fe_001"
        assert result[1].event_id == "fe_002"


class TestEventIdGeneration:
    """事件 ID 生成测试"""

    @pytest.fixture
    def store(self):
        """创建状态存储"""
        return NodeStateStore()

    def test_generate_event_id_format(self, store):
        """测试事件 ID 格式"""
        event_id = store.generate_event_id()

        assert event_id.startswith("fe_")
        # "fe_" (3 chars) + 12 hex chars from uuid = 15 chars total
        assert len(event_id) == 15

    def test_generate_event_id_unique(self, store):
        """测试事件 ID 唯一性"""
        ids = {store.generate_event_id() for _ in range(100)}

        assert len(ids) == 100  # 全部唯一


class TestEdgeCases:
    """边界情况测试"""

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
    async def test_load_node_state_corrupted_data(self, store_with_mock_redis):
        """测试加载损坏的节点状态数据"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = "invalid json data"

        result = await store.load_node_state("node-1")

        assert result is None  # 应该返回 None 而不是崩溃

    @pytest.mark.asyncio
    async def test_load_replica_index_corrupted_data(self, store_with_mock_redis):
        """测试加载损坏的副本索引数据"""
        store, mock_redis = store_with_mock_redis
        mock_redis.get.return_value = "{ invalid json }"

        result = await store.load_replica_index("model-x")

        assert result is None

    @pytest.mark.asyncio
    async def test_save_node_state_redis_error(self, store_with_mock_redis):
        """测试 Redis 错误时保存节点状态"""
        store, mock_redis = store_with_mock_redis
        mock_redis.set.side_effect = Exception("Redis connection error")

        state = NodeFailoverState(
            node_id="node-1",
            role=ReplicaRole.PRIMARY,
            state=FailoverState.NORMAL,
            health=HealthStatus.HEALTHY,
            term=1,
            last_heartbeat=datetime.now(),
        )

        result = await store.save_node_state(state)

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_lock_redis_error(self, store_with_mock_redis):
        """测试 Redis 错误时获取锁"""
        store, mock_redis = store_with_mock_redis
        mock_redis.set.side_effect = Exception("Redis error")

        result = await store.acquire_lock("resource", "node-1")

        assert result is False
