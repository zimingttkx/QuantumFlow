"""集成测试配置"""

import socket
import subprocess
import signal
import time
import os

import pytest


# ── Test API Key for integration tests ──────────────────────────────────────

TEST_API_KEY = "qk_integration_test_key_0000000000000000"


@pytest.fixture(autouse=True)
def _setup_test_tenant():
    """在每个集成测试前注册测试租户到缓存，确保 API 请求不被 401 拦截。"""
    from quantumflow.api.middleware.auth import (
        _tenant_cache,
        _tenant_cache_times,
        _cache_lock,
        hash_api_key,
    )
    from quantumflow.api.models.tenant import Tenant, TenantStatus, QuotaConfig

    key_hash = hash_api_key(TEST_API_KEY)
    tenant = Tenant(
        id="test-integration",
        name="Integration Test Tenant",
        api_key_hash=key_hash,
        api_key_prefix=TEST_API_KEY[:8],
        status=TenantStatus.ACTIVE,
        quota=QuotaConfig(),
        priority=10,
    )
    with _cache_lock:
        _tenant_cache[key_hash] = tenant
        _tenant_cache_times[key_hash] = time.time()
    yield
    with _cache_lock:
        _tenant_cache.pop(key_hash, None)
        _tenant_cache_times.pop(key_hash, None)


# ── Server detection ─────────────────────────────────────────────────────────

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
            torch.cuda.synchronize()
            free_mem, total_mem = torch.cuda.mem_get_info()
            return free_mem / (1024 ** 3)
    except Exception:
        pass
    return 0


def get_total_gpu_memory_gb():
    """获取 GPU 总显存（GB），失败返回 0"""
    try:
        import torch
        if torch.cuda.is_available():
            _, total_mem = torch.cuda.mem_get_info()
            return total_mem / (1024 ** 3)
    except Exception:
        pass
    return 0


@pytest.fixture(scope="session")
def integration_server():
    """启动集成测试服务器（session-scoped）"""
    if is_server_running():
        yield  # Server already running
        return

    server_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "quantumflow", "api", "server.py"
    )

    proc = subprocess.Popen(
        ["python", server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "QUANTUMFLOW_ENV": "testing"},
    )

    # Wait for server to be ready
    for _ in range(60):
        if is_server_running():
            break
        time.sleep(0.5)
    else:
        proc.terminate()
        proc.wait(timeout=10)
        pytest.skip("Server failed to start within 30s")

    yield

    # Teardown
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


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
        total_gpu_gb = get_total_gpu_memory_gb()
        # RTX 4080 12GB: 降低阈值到 6GB (total >= 10GB 时)
        threshold = 6.0 if total_gpu_gb >= 10.0 else 8.0
        if free_gpu_gb < threshold:
            pytest.skip(f"GPU 显存不足: {free_gpu_gb:.1f}GB < {threshold}GB")
