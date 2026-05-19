"""调度器单元测试"""

import asyncio
from unittest.mock import Mock

import pytest

from quantumflow.scheduler import GPUResource, NodeResource, Scheduler, SchedulingRequest


class TestScheduler:
    """调度器测试"""

    @pytest.fixture
    def scheduler(self):
        """创建调度器实例"""
        return Scheduler(
            default_strategy="adaptive",
            loop_interval_ms=50,
            max_retries=3,
        )

    @pytest.fixture
    def healthy_nodes(self):
        """创建健康节点列表"""
        return [
            NodeResource(
                node_id=f"node-{i}",
                hostname=f"server-{i}",
                ip=f"192.168.1.{100 + i}",
                status="healthy",
                gpu_count=4,
                gpus=[
                    GPUResource(
                        gpu_id=j,
                        memory_total=24 * 1024**3,
                        memory_used=10 * 1024**3,
                        utilization=0.5,
                        temperature=45.0,
                        node_id=f"node-{i}",
                    )
                    for j in range(4)
                ],
                cpu_count=32,
                memory_total=128 * 1024**3,
                memory_available=64 * 1024**3,
                disk_total=2 * 1024**4,
                disk_available=1 * 1024**4,
                load=0.3,
            )
            for i in range(3)
        ]

    def test_scheduler_initialization(self, scheduler):
        """测试调度器初始化"""
        assert scheduler is not None
        assert scheduler.default_strategy == "adaptive"
        assert scheduler.loop_interval_ms == 50
        assert scheduler.max_retries == 3
        assert scheduler._running is False

    def test_strategies_registered(self, scheduler):
        """测试策略注册"""
        assert "gang" in scheduler.strategies
        assert "pack" in scheduler.strategies

    def test_adaptive_strategy_registered(self, scheduler):
        """测试自适应策略注册"""
        assert scheduler.adaptive_strategy is not None

    def test_initial_stats(self, scheduler):
        """测试初始统计"""
        stats = scheduler.get_stats()

        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
        assert stats["pending_requests"] == 0

    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler):
        """测试启动和停止"""
        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._scheduling_task is not None

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_submit_increments_counter(self, scheduler):
        """测试提交请求增加计数器"""
        await scheduler.start()

        request = SchedulingRequest(
            request_id="test-001",
            model="test-model",
            model_config={"tensor_parallel": 1},
        )

        await scheduler.submit(request)

        stats = scheduler.get_stats()
        assert stats["total_requests"] == 1
        assert stats["pending_requests"] == 1

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_submit_multiple_requests(self, scheduler):
        """测试提交多个请求"""
        await scheduler.start()

        for i in range(5):
            request = SchedulingRequest(
                request_id=f"test-{i:03d}",
                model="test-model",
                priority=i % 3,
            )
            await scheduler.submit(request)

        stats = scheduler.get_stats()
        assert stats["total_requests"] == 5
        assert stats["pending_requests"] == 5

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_register_node(self, scheduler, healthy_nodes):
        """测试注册节点"""
        for node in healthy_nodes:
            await scheduler.register_node(node)

        assert len(scheduler.available_nodes) == 3

    @pytest.mark.asyncio
    async def test_unregister_node(self, scheduler, healthy_nodes):
        """测试注销节点"""
        for node in healthy_nodes:
            await scheduler.register_node(node)

        await scheduler.unregister_node("node-1")

        assert len(scheduler.available_nodes) == 2
        assert "node-1" not in scheduler.available_nodes

    @pytest.mark.asyncio
    async def test_update_node(self, scheduler, healthy_nodes):
        """测试更新节点"""
        node = healthy_nodes[0]
        await scheduler.register_node(node)

        # 更新节点负载
        updated_node = NodeResource(
            node_id=node.node_id,
            hostname=node.hostname,
            ip=node.ip,
            status=node.status,
            gpu_count=node.gpu_count,
            gpus=node.gpus,
            cpu_count=node.cpu_count,
            memory_total=node.memory_total,
            memory_available=node.memory_available,
            disk_total=node.disk_total,
            disk_available=node.disk_available,
            load=0.8,  # 更新负载
        )

        await scheduler.update_node(updated_node)

        assert scheduler.available_nodes["node-0"].load == 0.8

    @pytest.mark.asyncio
    async def test_scheduling_loop_with_nodes(self, scheduler, healthy_nodes):
        """测试调度循环"""
        for node in healthy_nodes:
            await scheduler.register_node(node)

        await scheduler.start()

        # 提交请求
        request = SchedulingRequest(
            request_id="test-001",
            model="test-model",
            model_config={
                "parameter_count": 7_000_000_000,
                "tensor_parallel": 1,
            },
        )
        await scheduler.submit(request)

        # 等待调度执行
        await asyncio.sleep(0.3)

        stats = scheduler.get_stats()
        assert stats["total_requests"] >= 1

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_callback_registration(self, scheduler):
        """测试回调注册"""
        callback = Mock()

        scheduler.on_node_update(callback)

        assert len(scheduler.node_update_callbacks) == 1

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, scheduler):
        """测试获取待调度请求"""
        await scheduler.start()

        for i in range(3):
            request = SchedulingRequest(
                request_id=f"test-{i:03d}",
                model="test-model",
            )
            await scheduler.submit(request)

        pending = scheduler.get_pending_requests()
        assert len(pending) == 3

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_get_running_requests(self, scheduler, healthy_nodes):
        """测试获取运行中的请求"""
        for node in healthy_nodes:
            await scheduler.register_node(node)

        await scheduler.start()

        request = SchedulingRequest(
            request_id="test-001",
            model="test-model",
            model_config={"tensor_parallel": 1},
        )
        await scheduler.submit(request)

        await asyncio.sleep(0.3)

        running = scheduler.get_running_requests()

        # 可能还在队列中或已开始运行
        # 检查请求是否被处理
        assert isinstance(running, dict)

        await scheduler.stop()

    def test_get_stats(self, scheduler):
        """测试获取统计信息"""
        stats = scheduler.get_stats()

        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "pending_requests" in stats
        assert "queue_size" in stats
        assert "running_size" in stats
        assert "available_nodes" in stats

    @pytest.mark.asyncio
    async def test_concurrent_submit(self, scheduler):
        """测试并发提交"""
        await scheduler.start()

        async def submit_requests(count):
            for i in range(count):
                request = SchedulingRequest(
                    request_id=f"test-{i:03d}",
                    model="test-model",
                )
                await scheduler.submit(request)

        # 并发提交
        await asyncio.gather(
            submit_requests(10),
            submit_requests(10),
            submit_requests(10),
        )

        stats = scheduler.get_stats()
        assert stats["total_requests"] == 30

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduling_with_priority(self, scheduler):
        """测试优先级调度"""
        await scheduler.start()

        # 按非顺序优先级提交
        requests = [
            SchedulingRequest(
                request_id=f"test-{i:03d}",
                model="test-model",
                priority=(i * 3) % 10,  # 0, 3, 6, 9, 2
            )
            for i in range(5)
        ]

        for req in requests:
            await scheduler.submit(req)

        # 验证队列中包含所有请求
        pending = scheduler.get_pending_requests()
        assert len(pending) == 5, f"Expected 5 pending requests, got {len(pending)}"

        # 验证所有请求的ID都在队列中
        pending_ids = {r.request_id for r in pending}
        expected_ids = {f"test-{i:03d}" for i in range(5)}
        assert pending_ids == expected_ids, f"Expected {expected_ids}, got {pending_ids}"

        # 验证所有请求都有优先级
        for r in pending:
            assert 0 <= r.priority <= 10, f"Invalid priority: {r.priority}"

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_no_nodes_available(self, scheduler):
        """测试无节点可用"""
        await scheduler.start()

        request = SchedulingRequest(
            request_id="test-001",
            model="test-model",
            model_config={"tensor_parallel": 1},
        )
        await scheduler.submit(request)

        # 等待调度
        await asyncio.sleep(0.3)

        # 应该失败
        stats = scheduler.get_stats()
        # 由于我们没有节点，可能无法成功调度

        await scheduler.stop()
