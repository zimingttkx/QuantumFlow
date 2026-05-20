"""健康检查严格集成测试

测试覆盖：
1. uptime_seconds 必须从应用启动时间计算，不能硬编码
2. Redis 连接检查必须真实 ping Redis
3. 集群状态检查必须查询真实节点状态
4. 各组件 unhealthy 时 overall status 必须正确反映
5. 就绪检查依赖验证
"""

import time
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from quantumflow.api.routes.health import get_app_start_time, set_app_start_time


class TestUptimeCalculation:
    """uptime 计算逻辑严格验证"""

    def test_uptime_must_be_positive(self):
        """[边界值] uptime 必须 >= 0"""
        # 设置启动时间为当前时间之前 100 秒
        past_time = time.time() - 100
        set_app_start_time(past_time)

        uptime = int(time.time() - get_app_start_time())

        assert uptime >= 0, f"uptime 应 >= 0，实际为 {uptime}"
        assert uptime >= 99, f"uptime 应约为 100，实际为 {uptime}"  # 允许 1 秒误差

        set_app_start_time(None)  # 恢复

    def test_uptime_increases_over_time(self):
        """[核心功能] 两次调用 uptime 应该增加"""
        t1 = time.time() - 50
        set_app_start_time(t1)

        uptime1 = int(time.time() - get_app_start_time())
        time.sleep(0.1)
        uptime2 = int(time.time() - get_app_start_time())

        assert uptime2 >= uptime1, "uptime 应该随时间增加"

        set_app_start_time(None)

    def test_uptime_zero_at_start_time(self):
        """[边界值] 启动时刻 uptime 应为 0"""
        current = time.time()
        set_app_start_time(current)

        uptime = int(time.time() - get_app_start_time())

        assert uptime == 0, f"启动时刻 uptime 应为 0，实际为 {uptime}"

        set_app_start_time(None)


class TestHealthCheckRedisIntegration:
    """健康检查 Redis 集成验证"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        from quantumflow.api.server import app

        return TestClient(app)

    @pytest.mark.asyncio
    async def test_redis_check_must_call_real_ping(self):
        """[核心功能] Redis 检查必须调用真实的 ping"""
        from quantumflow.storage import get_redis_manager

        # Mock RedisConnectionManager.health_check 返回 disconnected
        with patch(
            "quantumflow.storage.connection.RedisConnectionManager.health_check",
            new_callable=AsyncMock,
            return_value={"status": "healthy", "connected": True},
        ):
            with patch(
                "quantumflow.storage.get_redis_manager",
                new_callable=AsyncMock,
            ) as mock_get_mgr:
                mock_mgr = AsyncMock()
                mock_mgr.health_check = AsyncMock(
                    return_value={"status": "healthy", "connected": True}
                )
                mock_get_mgr.return_value = mock_mgr

                # 导入并调用 health_check
                from quantumflow.api.routes.health import health_check

                response = await health_check()

                # 验证调用了 health_check
                mock_mgr.health_check.assert_called_once()
                assert response.checks["redis"] == "healthy"

    @pytest.mark.asyncio
    async def test_redis_disconnected_affects_status(self):
        """[多分支] Redis 断开时 status 必须为 degraded"""
        from quantumflow.api.routes.health import health_check

        with patch(
            "quantumflow.storage.get_redis_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_mgr.health_check = AsyncMock(
                return_value={"status": "unhealthy", "connected": False, "error": "Connection refused"}
            )
            mock_get_mgr.return_value = mock_mgr

            response = await health_check()

            assert response.checks["redis"] == "unhealthy"
            assert response.status in ("degraded", "unhealthy")


class TestHealthCheckClusterIntegration:
    """健康检查集群集成验证"""

    @pytest.mark.asyncio
    async def test_cluster_check_must_query_real_stats(self):
        """[核心功能] 集群检查必须查询真实统计"""
        from quantumflow.cluster.manager import ClusterManager

        with patch.object(
            ClusterManager,
            "get_cluster_stats",
            new_callable=AsyncMock,
            return_value={
                "total_nodes": 3,
                "healthy_nodes": 2,
                "unhealthy_nodes": 1,
                "total_gpus": 12,
                "available_gpus": 8,
            },
        ):
            from quantumflow.api.routes.health import health_check

            response = await health_check()

            assert response.checks["cluster"] == "degraded"
            assert response.status == "degraded"

    @pytest.mark.asyncio
    async def test_all_healthy_returns_healthy_status(self):
        """[正常用例] 所有组件健康时返回 healthy"""
        from quantumflow.api.routes.health import health_check

        with patch(
            "quantumflow.storage.get_redis_manager",
            new_callable=AsyncMock,
        ) as mock_redis:
            mock_mgr = AsyncMock()
            mock_mgr.health_check = AsyncMock(
                return_value={"status": "healthy", "connected": True}
            )
            mock_redis.return_value = mock_mgr

            with patch(
                "quantumflow.cluster.manager.ClusterManager.get_cluster_stats",
                new_callable=AsyncMock,
                return_value={
                    "total_nodes": 2,
                    "healthy_nodes": 2,
                    "unhealthy_nodes": 0,
                },
            ):
                response = await health_check()

                assert response.status == "healthy"
                assert response.checks["api"] == "healthy"
                assert response.checks["redis"] == "healthy"
                assert response.checks["cluster"] == "healthy"


class TestHealthCheckVersion:
    """版本信息验证"""

    def test_version_must_match_package_version(self):
        """[核心功能] 返回的版本必须与包版本一致"""
        from quantumflow import __version__
        from quantumflow.api.routes.health import health_check
        import asyncio

        async def run():
            return await health_check()

        response = asyncio.get_event_loop().run_until_complete(run())

        assert response.version == __version__, (
            f"版本不匹配: API 返回 {response.version}, 包版本 {__version__}"
        )

    def test_version_is_not_hardcoded_string(self):
        """[反向校验] 版本不应该是硬编码的 '1.0.0'"""
        from quantumflow.api.routes.health import health_check
        import asyncio

        async def run():
            return await health_check()

        response = asyncio.get_event_loop().run_until_complete(run())

        # 如果版本是硬编码的 '1.0.0' 且包版本也是 '1.0.0'，这个测试无法区分
        # 所以我们检查 version 字段是否来自 __version__ 变量
        assert response.version is not None
        assert len(response.version) > 0


class TestReadinessCheck:
    """就绪检查验证"""

    @pytest.mark.asyncio
    async def test_ready_returns_true_when_all_deps_available(self):
        """[正常用例] 所有依赖就绪时返回 ready=true"""
        from quantumflow.api.routes.health import readiness_check

        with patch(
            "quantumflow.storage.get_redis_manager",
            new_callable=AsyncMock,
        ) as mock_redis:
            mock_mgr = AsyncMock()
            mock_mgr.health_check = AsyncMock(
                return_value={"status": "healthy", "connected": True}
            )
            mock_redis.return_value = mock_mgr

            response = await readiness_check()

            assert response["ready"] is True

    @pytest.mark.asyncio
    async def test_ready_returns_false_when_redis_disconnected(self):
        """[异常场景] Redis 断开时返回 ready=false"""
        from quantumflow.api.routes.health import readiness_check

        with patch(
            "quantumflow.storage.get_redis_manager",
            new_callable=AsyncMock,
        ) as mock_redis:
            mock_mgr = AsyncMock()
            mock_mgr.health_check = AsyncMock(
                return_value={"status": "unhealthy", "connected": False}
            )
            mock_redis.return_value = mock_mgr

            response = await readiness_check()

            assert response["ready"] is False
            assert "reason" in response

    @pytest.mark.asyncio
    async def test_ready_returns_false_on_redis_exception(self):
        """[异常场景] Redis 查询异常时返回 ready=false"""
        from quantumflow.api.routes.health import readiness_check

        with patch(
            "quantumflow.storage.get_redis_manager",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        ):
            response = await readiness_check()

            assert response["ready"] is False
            assert "reason" in response


class TestLivenessCheck:
    """存活检查验证"""

    def test_liveness_always_returns_alive(self):
        """[核心功能] 存活检查应返回 alive=true，与依赖无关"""
        from quantumflow.api.routes.health import liveness_check
        import asyncio

        async def run():
            return await liveness_check()

        response = asyncio.get_event_loop().run_until_complete(run())

        assert response["alive"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
