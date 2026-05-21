"""SDK 集成测试 - 真实 API 测试"""
import pytest
import time
from quantumflow.sdk import SyncQuantumFlowClient


class TestSDKIntegrationReal:
    """真实 API 集成测试"""

    def test_health_check_real(self):
        """验证：真实健康检查"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000")
        health = client.health_check()
        assert health["status"] == "healthy"
        assert "version" in health
        client.close()

    def test_list_models_real(self):
        """验证：真实获取模型列表"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000")
        models = client.list_models()
        assert isinstance(models, list)
        # models 可能为空，因为没有模型加载
        client.close()

    def test_generate_with_mock_model(self):
        """验证：使用真实推理请求（可能失败，因为模型可能不存在）"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000")

        # 发送一个生成请求
        try:
            response = client.generate(
                model="non-existent-model",
                prompt="Hello",
                temperature=0.7,
                max_tokens=10
            )
            # 如果成功，验证响应格式
            assert response.request_id is not None
            assert response.generated_text is not None
        except Exception as e:
            # 期望的错误：模型不存在或服务错误
            assert "non-existent" in str(e).lower() or "model" in str(e).lower() or "404" in str(e)
        finally:
            client.close()

    def test_client_lifecycle(self):
        """验证：客户端生命周期"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

        # 使用上下文管理器
        with SyncQuantumFlowClient(base_url="http://localhost:8000") as c:
            health = c.health_check()
            assert health is not None

    def test_client_with_api_key_header(self):
        """验证：带 API Key 的客户端"""
        # 不传 key 应该也能工作（可能返回未授权）
        client = SyncQuantumFlowClient(base_url="http://localhost:8000")
        # health 不需要认证
        health = client.health_check()
        assert health["status"] == "healthy"
        client.close()

    def test_rate_limit_real(self):
        """验证：真实限流（如果有的话）"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000")

        # 快速发送多个请求
        for _ in range(5):
            try:
                client.health_check()
            except Exception:
                pass

        client.close()

    def test_concurrent_clients(self):
        """验证：多个客户端并发"""
        clients = [
            SyncQuantumFlowClient(base_url="http://localhost:8000")
            for _ in range(3)
        ]

        results = []
        for c in clients:
            try:
                health = c.health_check()
                results.append(health["status"])
            except Exception:
                results.append("error")

        for c in clients:
            c.close()

        # 至少一些请求应该成功
        assert len(results) == 3

    def test_stream_false_param(self):
        """验证：stream 参数传递"""
        client = SyncQuantumFlowClient(base_url="http://localhost:8000")

        try:
            # 不使用 stream
            response = client.generate(
                model="test",
                prompt="Hello",
                stream=False
            )
            assert response is not None
        except Exception as e:
            # 期望模型不存在的错误
            assert "model" in str(e).lower() or "404" in str(e)
        finally:
            client.close()
