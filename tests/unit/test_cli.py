"""CLI命令行工具测试"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from quantumflow.cli import cli


@pytest.fixture
def runner():
    """创建 CLI 测试运行器"""
    return CliRunner()


class TestVersionCommand:
    """测试 version 命令"""

    def test_version_displays_correctly(self, runner):
        """version 命令应正确显示版本"""
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "QuantumFlow" in result.output
        assert "1.0.0" in result.output or "version" in result.output.lower()


class TestLoadCommand:
    """测试 load 命令"""

    def test_load_model_success(self, runner):
        """成功加载模型"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "loading"}
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "load", "Qwen2.5-1.5B",
                "--backend", "vllm",
                "--url", "http://localhost:8000"
            ])
            assert result.exit_code == 0

    def test_load_model_with_custom_params(self, runner):
        """带自定义参数加载模型"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "loading"}
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "load", "Qwen2.5-1.5B",
                "--backend", "huggingface",
                "--tensor-parallel", "2",
                "--gpu-memory", "0.9"
            ])
            assert result.exit_code == 0


class TestUnloadCommand:
    """测试 unload 命令"""

    def test_unload_model_success(self, runner):
        """成功卸载模型"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "unload", "Qwen2.5-1.5B",
                "--url", "http://localhost:8000"
            ])
            assert result.exit_code == 0


class TestStatusCommand:
    """测试 status 命令"""

    def test_status_one_shot(self, runner):
        """单次状态查看"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "total_nodes": 1,
                "healthy_nodes": 1,
                "total_gpus": 1,
                "available_gpus": 1,
                "active_models": 0
            }

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["status", "--url", "http://localhost:8000"])
            assert result.exit_code == 0
            assert "集群状态" in result.output or "QuantumFlow" in result.output


class TestGenerateCommand:
    """测试 generate 命令"""

    def test_generate_basic(self, runner):
        """基本生成测试"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_text": "Hello back",
                "latency_ms": 100
            }
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "generate", "Qwen2.5-1.5B",
                "--prompt", "Hello world",
                "--max-tokens", "50"
            ])
            assert result.exit_code == 0
            assert "生成结果" in result.output or "Hello back" in result.output

    def test_generate_with_temperature(self, runner):
        """带温度参数的生成"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_text": "Test output",
                "latency_ms": 50
            }
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "generate", "Qwen2.5-1.5B",
                "--temperature", "0.8"
            ])
            assert result.exit_code == 0


class TestChatCommand:
    """测试 chat 命令"""

    def test_chat_basic(self, runner):
        """基本对话测试"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_text": "Hello! How can I help?",
            }
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "chat", "Qwen2.5-1.5B",
                "--prompt", "Hello"
            ])
            assert result.exit_code == 0


class TestModelsCommand:
    """测试 models 命令"""

    def test_models_list(self, runner):
        """列出模型"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response_available = MagicMock()
            mock_response_available.status_code = 200
            mock_response_available.json.return_value = {
                "available_models": ["Qwen2.5-1.5B", "Qwen2.5-7B"],
                "mappings": {"Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct"}
            }

            mock_response_status = MagicMock()
            mock_response_status.status_code = 200
            mock_response_status.json.return_value = {"loaded_models": ["Qwen2.5-1.5B"]}

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.side_effect = [mock_response_available, mock_response_status]
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["models", "--url", "http://localhost:8000"])
            assert result.exit_code == 0


class TestQueueCommands:
    """测试 queue 子命令"""

    def test_queue_stats(self, runner):
        """队列统计"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "connected": True,
                "queue_stats": {
                    "size": 5,
                    "pending": 2,
                    "processing": 1,
                    "completed": 100,
                    "failed": 3
                },
                "metrics": {
                    "enqueue_rate": 10.5,
                    "dequeue_rate": 10.2,
                    "avg_wait_time_ms": 150,
                    "success_rate": 0.97
                }
            }

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["queue", "stats", "--url", "http://localhost:8000"])
            assert result.exit_code == 0

    def test_queue_submit_no_wait(self, runner):
        """提交请求（不等待）"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "request_id": "test-req-123",
                "status": "pending"
            }
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "queue", "submit", "Qwen2.5-1.5B",
                "--prompt", "Hello",
                "--url", "http://localhost:8000"
            ])
            assert result.exit_code == 0
            assert "test-req-123" in result.output or "pending" in result.output

    def test_queue_result_success(self, runner):
        """查询成功结果"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "success",
                "result": {
                    "result": {
                        "generated_text": "Success output"
                    }
                }
            }

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "queue", "result", "test-request-123",
                "--url", "http://localhost:8000"
            ])
            assert result.exit_code == 0

    def test_queue_result_pending(self, runner):
        """查询等待中结果"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "pending"
            }

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "queue", "result", "test-request-123",
                "--url", "http://localhost:8000"
            ])
            assert result.exit_code == 0


class TestWorkerCommands:
    """测试 worker 子命令"""

    def test_worker_register(self, runner):
        """注册 Worker"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "worker", "register", "worker-1",
                "--host", "localhost",
                "--port", "8080"
            ])
            assert result.exit_code == 0
            assert "worker-1" in result.output

    def test_worker_unregister(self, runner):
        """注销 Worker"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = ""

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.delete.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "worker", "unregister", "worker-1",
                "--url", "http://localhost:8000"
            ])
            assert result.exit_code == 0

    def test_workers_list(self, runner):
        """列出 Workers"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "node_id": "worker-1",
                    "hostname": "localhost",
                    "port": 8080,
                    "status": "healthy",
                    "gpu_count": 1,
                    "loaded_models": []
                }
            ]

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["workers", "--url", "http://localhost:8000"])
            assert result.exit_code == 0


class TestMonitorCommand:
    """测试 monitor 命令"""

    def test_monitor_one_shot(self, runner):
        """单次监控"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_cluster_response = MagicMock()
            mock_cluster_response.status_code = 200
            mock_cluster_response.json.return_value = {
                "total_nodes": 1,
                "healthy_nodes": 1,
                "total_gpus": 1,
                "available_gpus": 1,
                "active_models": 1
            }

            mock_scheduler_response = MagicMock()
            mock_scheduler_response.status_code = 200
            mock_scheduler_response.json.return_value = {}

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.side_effect = [mock_cluster_response, mock_scheduler_response]
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["monitor", "--url", "http://localhost:8000"])
            assert result.exit_code == 0


class TestHubCommands:
    """测试 hub 相关命令"""

    def test_hub_trending(self, runner):
        """热门模型"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"model_id": "Qwen/Qwen2.5-1.5B", "downloads": 1000000, "pipeline_tag": "text-generation"}
                ]
            }

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["hub", "--url", "http://localhost:8000"])
            assert result.exit_code == 0

    def test_search_models(self, runner):
        """搜索模型"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"model_id": "Qwen/Qwen2.5-7B", "downloads": 500000, "author": "Qwen"}
                ]
            }

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "search", "qwen",
                "--url", "http://localhost:8000"
            ])
            assert result.exit_code == 0


class TestDownloadCommand:
    """测试 download 命令"""

    def test_download_validation(self, runner):
        """下载前验证"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_validate_response = MagicMock()
            mock_validate_response.status_code = 200
            mock_validate_response.json.return_value = {"valid": True}

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_validate_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, [
                "download", "Qwen/Qwen2.5-1.5B-Instruct",
                "--url", "http://localhost:8000"
            ])
            assert result.exit_code == 0


class TestRecommendCommand:
    """测试 recommend 命令"""

    def test_recommend_models(self, runner):
        """推荐模型"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "system": {
                    "gpu_names": ["NVIDIA RTX 4080"],
                    "gpu_count": 1,
                    "total_vram_gb": 16,
                    "free_vram_gb": 8,
                    "ram_total_gb": 32
                },
                "recommendations": [
                    {"name": "Qwen2.5-1.5B", "params": 1.5, "vram_gb": 4, "status": "compatible", "description": "Test model"}
                ],
                "summary": {"compatible_count": 1, "can_run_7b": True}
            }

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["recommend", "--url", "http://localhost:8000"])
            assert result.exit_code == 0


class TestCLIHelp:
    """测试 CLI 帮助信息"""

    def test_help_shows_all_commands(self, runner):
        """帮助信息应显示所有命令"""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "QuantumFlow" in result.output

    def test_load_help(self, runner):
        """load 命令帮助"""
        result = runner.invoke(cli, ["load", "--help"])
        assert result.exit_code == 0
        assert "MODEL" in result.output

    def test_queue_help(self, runner):
        """queue 命令帮助"""
        result = runner.invoke(cli, ["queue", "--help"])
        assert result.exit_code == 0

    def test_worker_help(self, runner):
        """worker 命令帮助"""
        result = runner.invoke(cli, ["worker", "--help"])
        assert result.exit_code == 0

    def test_generate_help(self, runner):
        """generate 命令帮助"""
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0

    def test_chat_help(self, runner):
        """chat 命令帮助"""
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0

    def test_models_help(self, runner):
        """models 命令帮助"""
        result = runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0

    def test_hub_help(self, runner):
        """hub 命令帮助"""
        result = runner.invoke(cli, ["hub", "--help"])
        assert result.exit_code == 0


class TestCLIEdgeCases:
    """测试 CLI 边界情况"""

    def test_load_command_failure(self, runner):
        """加载模型失败"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.post.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["load", "InvalidModel"])
            assert result.exit_code == 0  # CLI 命令本身不抛异常

    def test_generate_command_connection_error(self, runner):
        """生成命令连接错误"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.side_effect = Exception("Connection refused")
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["generate", "Qwen2.5-1.5B", "--prompt", "Hello"])
            assert result.exit_code == 1

    def test_queue_result_not_found(self, runner):
        """查询不存在的请求"""
        with patch("quantumflow.cli.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"

            mock_async_client = AsyncMock()
            mock_async_client.__aenter__.return_value = mock_async_client
            mock_async_client.get.return_value = mock_response
            mock_async_client.__aexit__.return_value = None
            mock_client.return_value = mock_async_client

            result = runner.invoke(cli, ["queue", "result", "nonexistent"])
            assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
