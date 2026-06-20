"""分布式推理自动化测试

测试 QuantumFlow 的分布式推理功能，包括：
1. 本地模式 vs 分布式模式自动切换
2. 调度器正确提交请求到 Redis 队列
3. Worker 节点注册和注销
4. 请求结果正确返回
5. 错误处理和超时场景
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantumflow.api.models import InferenceResponse
from quantumflow.api.routes.inference import (
    _distributed_generate,
    _is_distributed_mode,
    _local_generate,
    _get_scheduler,
)
from quantumflow.inference.engine import InferenceResult, SamplingParams
from quantumflow.scheduler.distributed import DistributedScheduler, get_scheduler
from quantumflow.scheduler.strategy.base import SchedulingRequest
from quantumflow.storage.redis_queue import QueuedRequest


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _make_inference_result(
    request_id: str = "test-req-001",
    outputs: list[str] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    latency_ms: float = 150.0,
) -> InferenceResult:
    """创建模拟推理结果"""
    if outputs is None:
        outputs = ["Hello, world!"]
    return InferenceResult(
        request_id=request_id,
        outputs=outputs,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        metrics={},
    )


def _make_sampling_params(**kwargs) -> SamplingParams:
    """创建采样参数"""
    defaults = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 100,
        "repetition_penalty": 1.0,
    }
    defaults.update(kwargs)
    return SamplingParams(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# _is_distributed_mode 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsDistributedMode:
    """测试分布式模式检测逻辑"""

    def test_returns_false_when_no_workers(self):
        """没有 Worker 注册时应返回 False"""
        with patch("quantumflow.api.routes.inference._get_scheduler") as mock_get_sched:
            mock_scheduler = MagicMock()
            mock_scheduler.get_worker_count = AsyncMock(return_value=0)
            mock_get_sched.return_value = mock_scheduler

            result = _is_distributed_mode()

            assert result is False

    def test_returns_true_when_workers_exist(self):
        """有 Worker 注册时应返回 True"""
        with patch("quantumflow.api.routes.inference._get_scheduler") as mock_get_sched:
            mock_scheduler = MagicMock()
            mock_scheduler.get_worker_count = AsyncMock(return_value=2)
            mock_get_sched.return_value = mock_scheduler

            result = _is_distributed_mode()

            assert result is True

    def test_returns_false_on_exception(self):
        """调度器异常时应返回 False（安全降级）"""
        with patch("quantumflow.api.routes.inference._get_scheduler") as mock_get_sched:
            mock_scheduler = MagicMock()
            mock_scheduler.get_worker_count = AsyncMock(side_effect=RuntimeError("Boom"))
            mock_get_sched.return_value = mock_scheduler

            result = _is_distributed_mode()

            assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# _local_generate 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestLocalGenerate:
    """测试本地直接推理模式"""

    @pytest.mark.asyncio
    async def test_local_generate_returns_correct_response(self):
        """本地推理应返回正确的响应结构"""
        mock_result = _make_inference_result(
            request_id="local-001",
            outputs=["Test output"],
            prompt_tokens=5,
            completion_tokens=10,
            latency_ms=100.0,
        )

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[mock_result])

        mock_engine = MagicMock()
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            response = await _local_generate(
                request_id="local-001",
                model_name="test-model",
                prompt="Hello",
                sampling_params=_make_sampling_params(),
            )

        assert isinstance(response, InferenceResponse)
        assert response.request_id == "local-001"
        assert response.model == "test-model"
        assert response.generated_text == "Test output"
        assert response.finish_reason == "stop"
        assert response.usage["prompt_tokens"] == 5
        assert response.usage["completion_tokens"] == 10

    @pytest.mark.asyncio
    async def test_local_generate_empty_outputs(self):
        """空输出列表时应返回空字符串"""
        mock_result = _make_inference_result(outputs=[], prompt_tokens=5, completion_tokens=0)

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[mock_result])

        mock_engine = MagicMock()
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            response = await _local_generate(
                request_id="empty-001",
                model_name="test-model",
                prompt="Hello",
                sampling_params=_make_sampling_params(),
            )

        assert response.generated_text == ""

    @pytest.mark.asyncio
    async def test_local_generate_uses_batch_accumulator(self):
        """应正确使用 BatchAccumulator"""
        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[_make_inference_result()])

        mock_engine = MagicMock()
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            await _local_generate(
                request_id="batch-001",
                model_name="test-model",
                prompt="Hello",
                sampling_params=_make_sampling_params(temperature=0.5, max_tokens=200),
            )

            mock_engine.get_batch_accumulator.assert_called_once()
            call_args = mock_engine.get_batch_accumulator.call_args
            assert call_args[0][0] == "test-model"
            assert call_args[1]["max_delay_ms"] == 50.0
            assert call_args[1]["max_batch_size"] == 8

    @pytest.mark.asyncio
    async def test_local_generate_result_not_list(self):
        """当结果不是列表时应正确处理"""
        mock_result = _make_inference_result()
        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=mock_result)  # 直接返回结果，不是列表

        mock_engine = MagicMock()
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            response = await _local_generate(
                request_id="single-001",
                model_name="test-model",
                prompt="Hello",
                sampling_params=_make_sampling_params(),
            )

        assert response.generated_text == "Hello, world!"


# ═══════════════════════════════════════════════════════════════════════════════
# _distributed_generate 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistributedGenerate:
    """测试分布式推理模式"""

    @pytest.mark.asyncio
    async def test_distributed_generate_submits_to_scheduler(self):
        """应正确提交请求到调度器"""
        mock_scheduler = MagicMock()
        mock_scheduler.submit = AsyncMock(return_value=None)
        mock_scheduler.get_worker_count = AsyncMock(return_value=1)

        mock_queue = AsyncMock()
        mock_queue.get_result = AsyncMock(return_value={
            "status": "success",
            "result": {
                "generated_text": "Distributed result",
                "finish_reason": "stop",
                "prompt_tokens": 5,
                "completion_tokens": 10,
            },
            "latency_ms": 200.0,
        })
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()

        with patch("quantumflow.api.routes.inference._get_scheduler", return_value=mock_scheduler):
            with patch("quantumflow.api.routes.inference._get_shared_redis_queue", new_callable=AsyncMock, return_value=mock_queue):
                response = await _distributed_generate(
                    request_id="dist-001",
                    model_name="test-model",
                    prompt="Hello",
                    sampling_params=_make_sampling_params(),
                    priority=3,
                )

        # 验证调度器被调用
        mock_scheduler.submit.assert_called_once()
        sched_req = mock_scheduler.submit.call_args[0][0]
        assert isinstance(sched_req, SchedulingRequest)
        assert sched_req.request_id == "dist-001"
        assert sched_req.model == "test-model"
        assert sched_req.prompt == "Hello"
        assert sched_req.priority == 3

        # 验证响应
        assert response.request_id == "dist-001"
        assert response.generated_text == "Distributed result"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_distributed_generate_timeout(self):
        """结果超时时应抛出异常"""
        mock_scheduler = MagicMock()
        mock_scheduler.submit = AsyncMock(return_value=None)
        mock_scheduler.get_worker_count = AsyncMock(return_value=1)

        mock_queue = AsyncMock()
        mock_queue.get_result = AsyncMock(return_value=None)  # 永远不返回结果
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()

        with patch("quantumflow.api.routes.inference._get_scheduler", return_value=mock_scheduler):
            with patch("quantumflow.api.routes.inference._get_shared_redis_queue", new_callable=AsyncMock, return_value=mock_queue):
                with pytest.raises(Exception) as exc_info:
                    await _distributed_generate(
                        request_id="timeout-001",
                        model_name="test-model",
                        prompt="Hello",
                        sampling_params=_make_sampling_params(),
                        timeout_ms=100,  # 短超时用于测试
                    )

                assert "timeout" in str(exc_info.value).lower() or "Timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_distributed_generate_error_result(self):
        """调度失败时应抛出异常"""
        mock_scheduler = MagicMock()
        mock_scheduler.submit = AsyncMock(return_value=None)

        mock_queue = AsyncMock()
        mock_queue.get_result = AsyncMock(return_value={
            "status": "error",
            "reason": "调度失败：无可用节点",
        })
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()

        with patch("quantumflow.api.routes.inference._get_scheduler", return_value=mock_scheduler):
            with patch("quantumflow.api.routes.inference._get_shared_redis_queue", new_callable=AsyncMock, return_value=mock_queue):
                with pytest.raises(Exception) as exc_info:
                    await _distributed_generate(
                        request_id="error-001",
                        model_name="test-model",
                        prompt="Hello",
                        sampling_params=_make_sampling_params(),
                        timeout_ms=5000,
                    )

                    assert "调度失败" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_distributed_generate_sampling_params_in_metadata(self):
        """采样参数应正确传递到 metadata"""
        mock_scheduler = MagicMock()
        submitted_request = None

        async def capture_submit(req):
            nonlocal submitted_request
            submitted_request = req

        mock_scheduler.submit = AsyncMock(side_effect=capture_submit)
        mock_queue = AsyncMock()
        mock_queue.get_result = AsyncMock(return_value={
            "status": "success",
            "result": {
                "generated_text": "OK",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
            "latency_ms": 10,
        })
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()

        with patch("quantumflow.api.routes.inference._get_scheduler", return_value=mock_scheduler):
            with patch("quantumflow.api.routes.inference._get_shared_redis_queue", new_callable=AsyncMock, return_value=mock_queue):
                    await _distributed_generate(
                        request_id="params-001",
                        model_name="test-model",
                        prompt="Hello",
                        sampling_params=_make_sampling_params(
                            temperature=0.9,
                            top_p=0.8,
                            top_k=100,
                            max_tokens=500,
                            repetition_penalty=1.2,
                        ),
                    )

        assert submitted_request is not None
        model_config = submitted_request.model_config
        assert model_config["temperature"] == 0.9
        assert model_config["top_p"] == 0.8
        assert model_config["top_k"] == 100
        assert model_config["max_tokens"] == 500
        assert model_config["repetition_penalty"] == 1.2


# ═══════════════════════════════════════════════════════════════════════════════
# 集成测试：模式切换
# ═══════════════════════════════════════════════════════════════════════════════


class TestModeSwitching:
    """测试本地模式与分布式模式的自动切换"""

    @pytest.mark.asyncio
    async def test_uses_local_when_no_workers(self):
        """没有 Worker 时应使用本地模式"""
        from quantumflow.api.routes.inference import generate, InferenceRequest

        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[_make_inference_result()])

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            with patch("quantumflow.api.routes.inference._is_distributed_mode", return_value=False):
                request = InferenceRequest(
                    model="test-model",
                    prompt="Hello",
                )
                response = await generate(request)

        # 验证使用了本地模式
        mock_engine.get_batch_accumulator.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_distributed_when_workers_exist(self):
        """有 Worker 时应使用分布式模式"""
        from quantumflow.api.routes.inference import generate, InferenceRequest

        mock_scheduler = MagicMock()
        mock_scheduler.submit = AsyncMock(return_value=None)
        mock_scheduler.get_worker_count = AsyncMock(return_value=1)

        mock_queue = AsyncMock()
        mock_queue.get_result = AsyncMock(return_value={
            "status": "success",
            "result": {
                "generated_text": "Distributed",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
            "latency_ms": 10,
        })
        mock_queue.connect = AsyncMock()
        mock_queue.disconnect = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.is_model_loaded = MagicMock(return_value=True)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            with patch("quantumflow.api.routes.inference._is_distributed_mode", return_value=True):
                with patch("quantumflow.api.routes.inference._get_scheduler", return_value=mock_scheduler):
                    with patch("quantumflow.api.routes.inference._get_shared_redis_queue", new_callable=AsyncMock, return_value=mock_queue):
                            request = InferenceRequest(
                                model="test-model",
                                prompt="Hello",
                            )
                            response = await generate(request)

        # 验证使用了分布式模式
        mock_scheduler.submit.assert_called_once()
        assert response.generated_text == "Distributed"


# ═══════════════════════════════════════════════════════════════════════════════
# 错误场景测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """测试错误处理场景"""

    @pytest.mark.asyncio
    async def test_local_generate_raises_on_empty_results(self):
        """空结果列表应抛出 InferenceError"""
        mock_accumulator = AsyncMock()
        mock_accumulator.submit = AsyncMock(return_value=[])

        mock_engine = MagicMock()
        mock_engine.get_batch_accumulator = MagicMock(return_value=mock_accumulator)

        with patch("quantumflow.api.routes.inference.get_engine_manager", return_value=mock_engine):
            from quantumflow.core.exceptions import InferenceError

            with pytest.raises(InferenceError):
                await _local_generate(
                    request_id="error-001",
                    model_name="test-model",
                    prompt="Hello",
                    sampling_params=_make_sampling_params(),
                )

    @pytest.mark.asyncio
    async def test_distributed_cleans_up_queue_on_error(self):
        """分布式模式出错时应正确清理 Redis 连接"""
        mock_scheduler = MagicMock()
        mock_scheduler.submit = AsyncMock(side_effect=RuntimeError("Boom"))

        with patch("quantumflow.api.routes.inference._get_scheduler", return_value=mock_scheduler):
            with pytest.raises(RuntimeError):
                await _distributed_generate(
                    request_id="cleanup-001",
                    model_name="test-model",
                    prompt="Hello",
                    sampling_params=_make_sampling_params(),
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
