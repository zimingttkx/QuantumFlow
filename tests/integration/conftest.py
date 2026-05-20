"""集成测试配置"""

import socket

import pytest


def is_server_running():
    """检查服务器是否运行"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(2)
        result = s.connect_ex(("localhost", 8000))
        s.close()
        return result == 0
    except Exception:
        return False


def get_free_gpu_memory_gb():
    """获取当前可用的 GPU 显存（GB），失败返回 0"""
    try:
        import torch
        if torch.cuda.is_available():
            # 先清理缓存，获取真实的可用显存
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0]
            return free_mem / (1024 ** 3)
    except Exception:
        pass
    return 0


def pytest_collection_modifyitems(items):
    """自动为需要服务器的测试添加跳过条件"""
    server_running = is_server_running()

    for item in items:
        # 检查测试是否标记为需要服务器
        if "test_api_strict" in str(item.fspath):
            # test_server_is_running 需要服务器运行，所以它不应该被跳过
            if "test_server_is_running" not in item.name:
                if not server_running:
                    item.add_marker(pytest.mark.skip("服务器未运行"))


def pytest_runtest_setup(item):
    """在每个测试运行前检查 GPU 显存"""
    # 跳过需要大显存的模型加载测试
    if "test_deploy_model" in item.name or "test_model_state_persistence" in item.name:
        free_gpu_gb = get_free_gpu_memory_gb()
        if free_gpu_gb < 8.0:
            pytest.skip(f"GPU 显存不足: {free_gpu_gb:.1f}GB < 8GB")
