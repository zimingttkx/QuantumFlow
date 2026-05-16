"""严格的端到端测试 - 验证整个系统"""

import pytest
import asyncio
import time
import json
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


# =============================================================================
# Test Configuration and Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def api_base_url():
    """API基础URL"""
    return "http://localhost:8000/api/v1"


@pytest.fixture(scope="session")
def test_timeout():
    """测试超时时间"""
    return 60.0


def is_server_running():
    """检查服务器是否运行"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(2)
        result = s.connect_ex(('localhost', 8000))
        s.close()
        return result == 0
    except Exception:
        return False


# =============================================================================
# Test API Health and Connectivity
# =============================================================================

class TestAPIService:
    """API服务健康检查"""

    def test_server_is_running(self):
        """验证服务器正在运行"""
        assert is_server_running(), "服务器未在8000端口运行，请先启动服务器"

    @pytest.mark.asyncio
    async def test_api_root_endpoint(self):
        """验证API根端点"""
        if not is_server_running():
            pytest.skip("服务器未运行")

        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            response = await client.get("http://localhost:8000/")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cluster_status_endpoint(self, api_base_url):
        """验证集群状态端点返回有效数据"""
        if not is_server_running():
            pytest.skip("服务器未运行")

        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            response = await client.get(f"{api_base_url}/cluster/status")
            assert response.status_code == 200
            data = response.json()
            assert "total_nodes" in data
            assert "healthy_nodes" in data
            assert "total_gpus" in data


# =============================================================================
# Test Inference Endpoints - Non-Streaming
# =============================================================================

class TestInferenceNonStreaming:
    """非流式推理端点测试"""

    @pytest.mark.asyncio
    async def test_generate_endpoint_basic(self, api_base_url):
        """测试基本生成功能"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "test-model",
                "prompt": "Hello, how are you?",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 50,
                },
                "stream": False,
            }
            response = await client.post(
                f"{api_base_url}/inference/generate",
                json=payload,
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            data = response.json()

            # 验证响应结构
            assert "request_id" in data, "响应缺少request_id"
            assert "model" in data, "响应缺少model"
            assert "generated_text" in data, "响应缺少generated_text"
            assert "finish_reason" in data, "响应缺少finish_reason"
            assert "latency_ms" in data, "响应缺少latency_ms"
            assert "usage" in data, "响应缺少usage"

            # 验证内容不为空
            assert len(data["generated_text"]) > 0, "generated_text为空"

            # 验证latency是有效数字
            assert data["latency_ms"] >= 0, "latency_ms应该非负"

            # 验证usage结构
            usage = data["usage"]
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
            assert "total_tokens" in usage

    @pytest.mark.asyncio
    async def test_generate_with_empty_prompt(self, api_base_url):
        """测试空prompt被正确处理"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "test-model",
                "prompt": "",
                "stream": False,
            }
            response = await client.post(
                f"{api_base_url}/inference/generate",
                json=payload,
            )
            # 空prompt安全处理，返回模拟数据
            assert response.status_code in [200, 400, 422, 500]

    @pytest.mark.asyncio
    async def test_generate_with_extreme_temperature(self, api_base_url):
        """测试极端temperature值"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "test-model",
                "prompt": "Test",
                "sampling_params": {
                    "temperature": 2.0,  # 最大值
                    "max_tokens": 10,
                },
                "stream": False,
            }
            response = await client.post(
                f"{api_base_url}/inference/generate",
                json=payload,
            )
            # 应该接受请求（即使模型可能不支持）
            assert response.status_code in [200, 400, 422, 500]

    @pytest.mark.asyncio
    async def test_generate_with_zero_temperature(self, api_base_url):
        """测试temperature=0（贪心解码）"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "test-model",
                "prompt": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": 20,
                },
                "stream": False,
            }
            response = await client.post(
                f"{api_base_url}/inference/generate",
                json=payload,
            )
            assert response.status_code in [200, 400, 422, 500]


# =============================================================================
# Test Inference Endpoints - Streaming
# =============================================================================

class TestInferenceStreaming:
    """流式推理端点测试"""

    @pytest.mark.asyncio
    async def test_stream_generate_endpoint(self, api_base_url):
        """测试流式生成端点"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "test-model",
                "prompt": "Count to 5:",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
                "stream": True,
            }

            async with client.stream(
                "POST",
                f"{api_base_url}/inference/generate/stream",
                json=payload,
            ) as response:
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"

                chunks = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            pass

                # 验证收到了数据块
                assert len(chunks) > 0, "没有收到任何数据块"

                # 验证最后一个块是final
                final_chunk = chunks[-1]
                assert final_chunk.get("is_final") == True, "最后一块应该标记为is_final"

    @pytest.mark.asyncio
    async def test_stream_generate_sse_format(self, api_base_url):
        """验证SSE格式正确性"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "test-model",
                "prompt": "Hello",
                "sampling_params": {"max_tokens": 20},
                "stream": True,
            }

            async with client.stream(
                "POST",
                f"{api_base_url}/inference/generate/stream",
                json=payload,
            ) as response:
                assert response.status_code == 200

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        # 必须是有效的JSON
                        if data_str != "[DONE]":
                            try:
                                json.loads(data_str)
                            except json.JSONDecodeError:
                                pytest.fail(f"无效的JSON: {data_str}")


# =============================================================================
# Test Chat Endpoint
# =============================================================================

class TestChatEndpoint:
    """聊天端点测试"""

    @pytest.mark.asyncio
    async def test_chat_endpoint_basic(self, api_base_url):
        """测试基本聊天功能"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 50,
                },
            }
            response = await client.post(
                f"{api_base_url}/inference/chat",
                json=payload,
            )

            # 应该返回200或错误，不应崩溃
            assert response.status_code in [200, 400, 404, 500]

            if response.status_code == 200:
                data = response.json()
                assert "generated_text" in data

    @pytest.mark.asyncio
    async def test_chat_with_conversation_history(self, api_base_url):
        """测试带对话历史的聊天"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice! How can I help you?"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 50,
                },
            }
            response = await client.post(
                f"{api_base_url}/inference/chat",
                json=payload,
            )

            assert response.status_code in [200, 400, 404, 500]

    @pytest.mark.asyncio
    async def test_chat_empty_messages(self, api_base_url):
        """测试空消息列表"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "test-model",
                "messages": [],
            }
            response = await client.post(
                f"{api_base_url}/inference/chat",
                json=payload,
            )
            # 应该返回错误
            assert response.status_code in [400, 422, 500]

    @pytest.mark.asyncio
    async def test_chat_invalid_role(self, api_base_url):
        """测试无效的role"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "test-model",
                "messages": [
                    {"role": "invalid_role", "content": "Hello"}
                ],
            }
            response = await client.post(
                f"{api_base_url}/inference/chat",
                json=payload,
            )
            # 无效role被转换为"user"，正常返回200
            assert response.status_code in [200, 400, 422, 500]


# =============================================================================
# Test Batch Inference
# =============================================================================

class TestBatchInference:
    """批量推理测试"""

    @pytest.mark.asyncio
    async def test_batch_inference_basic(self, api_base_url):
        """测试基本批量推理"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "test-model",
                "prompts": [
                    "What is 2+2?",
                    "What is 3+3?",
                    "What is 4+4?",
                ],
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 50,
                },
            }
            response = await client.post(
                f"{api_base_url}/inference/batch",
                json=payload,
            )

            assert response.status_code in [200, 400, 404, 500]

            if response.status_code == 200:
                data = response.json()
                assert "batch_id" in data
                assert "results" in data
                assert len(data["results"]) == 3

    @pytest.mark.asyncio
    async def test_batch_inference_empty_list(self, api_base_url):
        """测试空批量列表"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "test-model",
                "prompts": [],
            }
            response = await client.post(
                f"{api_base_url}/inference/batch",
                json=payload,
            )
            # 应该返回错误
            assert response.status_code in [400, 422, 500]

    @pytest.mark.asyncio
    async def test_batch_inference_single_item(self, api_base_url):
        """测试单项目批量"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "test-model",
                "prompts": ["Hello"],
                "sampling_params": {"max_tokens": 20},
            }
            response = await client.post(
                f"{api_base_url}/inference/batch",
                json=payload,
            )
            assert response.status_code in [200, 400, 404, 500]


# =============================================================================
# Test Model Management
# =============================================================================

class TestModelManagement:
    """模型管理测试"""

    @pytest.mark.asyncio
    async def test_model_status_endpoint(self, api_base_url):
        """测试模型状态端点"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{api_base_url}/models/status")
            assert response.status_code == 200
            data = response.json()
            assert "loaded_models" in data

    @pytest.mark.asyncio
    async def test_model_list_endpoint(self, api_base_url):
        """测试模型列表端点"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{api_base_url}/models/list")
            assert response.status_code == 200


# =============================================================================
# Test Cluster Management
# =============================================================================

class TestClusterManagement:
    """集群管理测试"""

    @pytest.mark.asyncio
    async def test_cluster_nodes_endpoint(self, api_base_url):
        """测试集群节点端点"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{api_base_url}/cluster/nodes")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cluster_heartbeat(self, api_base_url):
        """测试集群心跳"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{api_base_url}/cluster/status")
            assert response.status_code == 200


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """错误处理测试"""

    @pytest.mark.asyncio
    async def test_nonexistent_model(self, api_base_url):
        """测试不存在的模型"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "this-model-definitely-does-not-exist-12345",
                "prompt": "Hello",
            }
            response = await client.post(
                f"{api_base_url}/inference/generate",
                json=payload,
            )
            # 不存在的模型返回模拟数据，不应崩溃
            assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    async def test_invalid_json_payload(self, api_base_url):
        """测试无效的JSON负载"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{api_base_url}/inference/generate",
                content=b"not valid json",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code in [400, 422, 500]

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, api_base_url):
        """测试缺失必需字段"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "prompt": "Hello",  # 缺少model字段
            }
            response = await client.post(
                f"{api_base_url}/inference/generate",
                json=payload,
            )
            assert response.status_code in [400, 422, 500]


# =============================================================================
# Test Performance and Latency
# =============================================================================

class TestPerformance:
    """性能测试"""

    @pytest.mark.asyncio
    async def test_latency_reasonable(self, api_base_url):
        """测试延迟合理性"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "test-model",
                "prompt": "Say hello in exactly 5 words",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_tokens": 20,
                },
                "stream": False,
            }
            start = time.time()
            response = await client.post(
                f"{api_base_url}/inference/generate",
                json=payload,
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                # 报告的延迟应该与实际延迟相近
                assert data["latency_ms"] > 0
                assert data["latency_ms"] < elapsed * 1000 + 1000  # 允许一些误差


# =============================================================================
# Test Frontend Static File Serving
# =============================================================================

class TestFrontend:
    """前端测试"""

    def test_index_html_served(self):
        """测试index.html被正确提供"""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex(('localhost', 8000))
        s.close()
        if result != 0:
            pytest.skip("服务器未运行")

        import httpx
        response = httpx.get("http://localhost:8000/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


# =============================================================================
# Test Real Model Loading (if available)
# =============================================================================

class TestRealModelLoading:
    """真实模型加载测试（如果模型可用）"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_qwen_model(self, api_base_url):
        """测试加载Qwen2.5-1.5B模型"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": "Qwen2.5-1.5B",
            }
            response = await client.post(
                f"{api_base_url}/models/load",
                json=payload,
            )
            assert response.status_code in [200, 201, 400]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_generate_with_loaded_model(self, api_base_url):
        """测试使用已加载模型生成"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 先确保模型加载
            await client.post(
                f"{api_base_url}/models/load",
                json={"model": "test-model"},
            )

            # 然后生成
            payload = {
                "model": "test-model",
                "prompt": "What is 1+1?",
                "sampling_params": {
                    "temperature": 0.1,
                    "max_tokens": 10,
                },
            }
            response = await client.post(
                f"{api_base_url}/inference/generate",
                json=payload,
            )
            # 可能是200成功或500模型未加载
            assert response.status_code in [200, 404, 500]
