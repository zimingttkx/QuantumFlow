"""QuantumFlow SDK 初始化与公共契约测试

本文件早期版本只断言 `import X; assert X is not None` —— 这种断言
对任何代码变更都无法失败 (Python import 时对象一定存在)。重写为真正
的行为契约测试:

1. `__all__` 与实际导出符号一致 (防止有人改了 import 但忘了 __all__)
2. 异常类层级正确 (RateLimitError → APIError → QuantumFlowError)
3. 异常构造器产生标准消息格式
4. 客户端默认值符合文档 (base_url 去尾斜杠, 注入 X-API-Key / X-Tenant-ID)
5. 客户端响应码契约: 429 → RateLimitError, 4xx/5xx → APIError, 超时 → TimeoutError

每个测试都断言**具体值**, 而非"非空"或"存在"。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import quantumflow.sdk as sdk
from quantumflow.sdk import (
    APIError,
    AsyncQuantumFlowClient,
    QuantumFlowError,
    QuantumFlowSDK,
    RateLimitError,
    SyncQuantumFlowClient,
    TimeoutError,
    ValidationError,
)


# ─────────────────────────────────────────────────────────────────
# 1. 公共导出
# ─────────────────────────────────────────────────────────────────


def test_all_exports_resolve():
    """`__all__` 中每个名字必须能在模块上取到 — 不是 `is not None` 的空断言,
    而是逐个 name 实际 get, get 不到就 AttributeError."""
    assert hasattr(sdk, "__all__"), "quantumflow.sdk 未声明 __all__"
    for name in sdk.__all__:
        obj = getattr(sdk, name)
        assert obj is not None, f"quantumflow.sdk.{name} 解析为 None"


def test_all_exports_are_public_api_symbols():
    """`__all__` 名单不能含私有符号 (下划线开头) — 防止 export 内部实现细节."""
    assert hasattr(sdk, "__all__")
    private = [n for n in sdk.__all__ if n.startswith("_")]
    assert private == [], f"__all__ 含私有符号: {private}"


def test_quantumflow_sdk_is_alias_for_sync_client():
    """`QuantumFlowSDK` 是 `SyncQuantumFlowClient` 的别名 — 验证两者是同一类
    (不是同名拷贝), 否则 `isinstance(client, QuantumFlowSDK)` 与
    `isinstance(client, SyncQuantumFlowClient)` 会得出不同结论."""
    assert QuantumFlowSDK is SyncQuantumFlowClient, (
        f"QuantumFlowSDK 应为 SyncQuantumFlowClient 的别名, "
        f"实际 QuantumFlowSDK={QuantumFlowSDK!r}, "
        f"SyncQuantumFlowClient={SyncQuantumFlowClient!r}"
    )


# ─────────────────────────────────────────────────────────────────
# 2. 异常类层级
# ─────────────────────────────────────────────────────────────────


def test_exception_hierarchy_is_correct():
    """异常类型必须形成稳定的 catch 顺序:
    RateLimitError → APIError → QuantumFlowError → Exception.
    这是 SDK 调用方 try/except 的基础契约, 任何错位都会让用户抓不到异常。"""
    assert issubclass(RateLimitError, APIError), (
        "RateLimitError 必须继承 APIError — 调用方常写 except APIError 捕获 4xx"
    )
    assert issubclass(APIError, QuantumFlowError), (
        "APIError 必须继承 QuantumFlowError — SDK 顶层 except QuantumFlowError 应能兜住所有 SDK 异常"
    )
    assert issubclass(QuantumFlowError, Exception)
    # TimeoutError 与 ValidationError 直接继承 QuantumFlowError (非 APIError)
    assert issubclass(TimeoutError, QuantumFlowError)
    assert issubclass(ValidationError, QuantumFlowError)
    # 但 TimeoutError 不应是 APIError (4xx/5xx 才是)
    assert not issubclass(TimeoutError, APIError), (
        "TimeoutError 不应继承 APIError — 否则 except APIError 会吞掉超时, 误导调用方"
    )


def test_api_error_message_format_is_stable():
    """APIError 消息格式 'API Error {code}: {message}' 是公开契约 —
    Sentry/日志告警/客服工单都依赖这个格式做正则匹配。改它就会全链路炸。"""
    err = APIError(status_code=502, message="Bad Gateway")
    assert err.status_code == 502
    assert err.message == "Bad Gateway"
    assert str(err) == "API Error 502: Bad Gateway"


def test_rate_limit_error_uses_429_by_default():
    """RateLimitError 不传参时 status_code=429 — 这是限流的语义标识。
    改成其他状态码会让现有客户端把 503 也当成限流处理, 错乱。"""
    err = RateLimitError()
    assert err.status_code == 429
    assert isinstance(err, APIError), "RateLimitError 必须是 APIError 的子类"
    # 默认消息
    assert err.message == "Rate limit exceeded"
    assert "429" in str(err)


def test_rate_limit_error_accepts_custom_message():
    """RateLimitError 可覆盖默认消息 — 例如网关可能返回 'quota exhausted'."""
    err = RateLimitError(message="Monthly quota exhausted")
    assert err.message == "Monthly quota exhausted"
    assert err.status_code == 429
    assert "Monthly quota exhausted" in str(err)


# ─────────────────────────────────────────────────────────────────
# 3. Sync 客户端默认值
# ─────────────────────────────────────────────────────────────────


def test_sync_client_default_base_url_strips_trailing_slash():
    """base_url 末尾斜杠会被去除 — 否则 `base_url + '/api/v1/models'` 会变成
    'http://localhost:8000//api/v1/models', httpx 兼容但语义不清, 监控埋点会乱。"""
    client = SyncQuantumFlowClient(base_url="http://localhost:8000///")
    try:
        assert client.base_url == "http://localhost:8000", (
            f"expected trailing slashes stripped, got {client.base_url!r}"
        )
    finally:
        client.close()


def test_sync_client_injects_api_key_header_when_provided():
    """传 api_key 时, httpx client 必须带上 X-API-Key 头 — 否则服务端 401."""
    client = SyncQuantumFlowClient(base_url="http://localhost:8000", api_key="qf-test-key")
    try:
        assert client.api_key == "qf-test-key"
        assert client._client.headers.get("X-API-Key") == "qf-test-key", (
            f"X-API-Key header not injected, headers={dict(client._client.headers)!r}"
        )
    finally:
        client.close()


def test_sync_client_injects_tenant_header_when_provided():
    """传 tenant_id 时, httpx client 必须带上 X-Tenant-ID 头 — 否则服务端
    不知道走哪个租户的配额 / 限流 / 计费."""
    client = SyncQuantumFlowClient(
        base_url="http://localhost:8000", api_key="qf-k", tenant_id="tenant-42"
    )
    try:
        assert client.tenant_id == "tenant-42"
        assert client._client.headers.get("X-Tenant-ID") == "tenant-42"
        assert client._client.headers.get("X-API-Key") == "qf-k"
    finally:
        client.close()


def test_sync_client_omits_api_key_header_when_not_provided():
    """不传 api_key 时, 不应该有 X-API-Key 头 (允许匿名端点;
    但有 key 时必须注入 — 见上一个测试). 防止 'X-API-Key: None' 这种错。"""
    client = SyncQuantumFlowClient(base_url="http://localhost:8000")
    try:
        assert client.api_key is None
        assert "X-API-Key" not in client._client.headers, (
            f"未传 api_key 时不应注入 X-API-Key, headers={dict(client._client.headers)!r}"
        )
        assert "X-Tenant-ID" not in client._client.headers
    finally:
        client.close()


# ─────────────────────────────────────────────────────────────────
# 4. Sync 客户端 _post 响应码契约 (mock httpx, 不发真实请求)
# ─────────────────────────────────────────────────────────────────


def _mock_httpx_response(status_code: int, body: dict | str = ""):
    """构造一个 httpx.Response-like mock."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = body if isinstance(body, str) else str(body)
    resp.json.return_value = body if isinstance(body, dict) else {}
    return resp


def test_sync_post_returns_parsed_json_on_2xx():
    """2xx 响应 → 返回 json() 结果."""
    client = SyncQuantumFlowClient(base_url="http://localhost:8000", api_key="k")
    try:
        ok = _mock_httpx_response(200, {"result": "ok"})
        with patch.object(client._client, "post", return_value=ok):
            result = client._post("/api/v1/health")
        assert result == {"result": "ok"}, f"got {result!r}"
    finally:
        client.close()


def test_sync_post_raises_rate_limit_error_on_429():
    """429 → RateLimitError (HTTP 限流语义), 不是通用 APIError。
    这一点很关键 — 客户端 SDK 通常需要识别 429 做退避重试。"""
    client = SyncQuantumFlowClient(base_url="http://localhost:8000", api_key="k")
    try:
        resp = _mock_httpx_response(429, "Too Many Requests")
        with patch.object(client._client, "post", return_value=resp):
            with pytest.raises(RateLimitError) as exc_info:
                client._post("/api/v1/inference/generate")
        # RateLimitError 是 APIError 子类, 应该带 status_code=429
        assert exc_info.value.status_code == 429
    finally:
        client.close()


def test_sync_post_raises_api_error_on_4xx():
    """非 429 的 4xx → APIError, status_code 必须如实传递."""
    client = SyncQuantumFlowClient(base_url="http://localhost:8000", api_key="k")
    try:
        resp = _mock_httpx_response(404, "Model not found")
        with patch.object(client._client, "post", return_value=resp):
            with pytest.raises(APIError) as exc_info:
                client._post("/api/v1/models/missing")
        assert exc_info.value.status_code == 404
        assert "Model not found" in str(exc_info.value)
        # 不应是 RateLimitError (那是 429 专属)
        assert not isinstance(exc_info.value, RateLimitError)
    finally:
        client.close()


def test_sync_post_raises_api_error_on_5xx():
    """5xx → APIError, status_code 透传 — 客户端不需要区分 4xx/5xx 业务处理,
    监控告警区分即可."""
    client = SyncQuantumFlowClient(base_url="http://localhost:8000", api_key="k")
    try:
        resp = _mock_httpx_response(503, "Service Unavailable")
        with patch.object(client._client, "post", return_value=resp):
            with pytest.raises(APIError) as exc_info:
                client._post("/api/v1/inference/generate")
        assert exc_info.value.status_code == 503
    finally:
        client.close()


def test_sync_post_translates_httpx_timeout_to_sdk_timeout():
    """httpx.TimeoutException → SDK TimeoutError, 不让 httpx 异常泄漏到调用方 —
    否则调用方需要 import httpx 才能 catch, 违背 SDK 封装原则。"""
    client = SyncQuantumFlowClient(base_url="http://localhost:8000", api_key="k")
    try:
        with patch.object(
            client._client, "post", side_effect=httpx.TimeoutException("timed out")
        ):
            with pytest.raises(TimeoutError) as exc_info:
                client._post("/api/v1/inference/generate")
        # 消息含路径, 方便定位
        assert "/api/v1/inference/generate" in str(exc_info.value)
    finally:
        client.close()


# ─────────────────────────────────────────────────────────────────
# 5. Async 客户端契约 (与 Sync 对照, 防止两边行为漂移)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_client_default_base_url_strips_trailing_slash():
    """Async 客户端必须与 Sync 一致地处理 base_url."""
    client = AsyncQuantumFlowClient(base_url="http://localhost:8000//")
    assert client.base_url == "http://localhost:8000"


@pytest.mark.asyncio
async def test_async_client_stores_credentials():
    """Async 客户端保存 api_key / tenant_id, 用于后续每个请求注入 — 不存就丢,
    存错就 401."""
    client = AsyncQuantumFlowClient(
        base_url="http://localhost:8000", api_key="k", tenant_id="t"
    )
    assert client.api_key == "k"
    assert client.tenant_id == "t"
    assert client.timeout == 30.0  # 默认 30s


@pytest.mark.asyncio
async def test_async_arequest_returns_parsed_json_on_2xx():
    """2xx → 返回 json() 结果."""
    client = AsyncQuantumFlowClient(base_url="http://localhost:8000", api_key="k")
    ok = _mock_httpx_response(200, {"models": []})
    with patch("quantumflow.sdk.client.httpx.AsyncClient") as MockClient:
        # `async with httpx.AsyncClient(...)` 要求 __aenter__/__aexit__
        # 都是 awaitable coroutine, MagicMock 不是。AsyncMock 是。
        mock_instance = MagicMock()
        mock_instance.request = AsyncMock(return_value=ok)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance
        result = await client._arequest("GET", "/api/v1/models")
    assert result == {"models": []}, f"got {result!r}"


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────


def _async_value(value):
    """构造一个 awaitable: `await _async_value(x)` 返回 x。MagicMock 的
    `return_value` 只能给同步调用, await 必须用 Awaitable。"""
    async def _coro():
        return value
    return _coro()
