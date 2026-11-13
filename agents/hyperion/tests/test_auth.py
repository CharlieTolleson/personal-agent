"""
Bearer-token gate tests for Hyperion's API (:4100) and MCP server (:4101).

Covers ``hyperion.server.auth`` as wired into ``hyperion.server.api`` and
``hyperion.server.mcp``:
  * every registered API route, plus docs, ``/openapi.json`` and unknown paths,
    refuses a request with no token, a wrong token, a wrong scheme, or a token in
    the query string (401, never the route's own answer);
  * an authenticated request is served;
  * a missing, short, or placeholder server token fails closed (503) and the
    entrypoints refuse to start;
  * CORS preflight is the only thing answered without a token, and reveals nothing;
  * the MCP endpoint refuses without the token and serves with it, and its
    approve/feedback proxy forwards the token to the API;
  * the token (and a wrong one) never appears in logs or startup errors;
  * the compare goes through ``hmac.compare_digest``; WebSockets are refused too.
"""

from __future__ import annotations

import logging
import re

import httpx
import pytest
from auth_helpers import AUTH_HEADERS, TEST_TOKEN
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from hyperion.server import auth
from hyperion.server.api import app

WRONG_TOKEN = "wrong-token-" + "f" * 40


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only."""
    return "asyncio"


def _route_calls() -> list[tuple[str, str]]:
    """Every (method, concrete path) the API registers, path params filled in.

    Returns:
        One entry per route and method (HEAD skipped), with ``{param}`` and
        ``{param:path}`` replaced by a dummy value.
    """
    calls = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = re.sub(r"\{[^}]+\}", "dummy", route.path)
        for method in sorted(route.methods - {"HEAD"}):
            calls.append((method, path))
    # Framework routes and a path that does not exist must be gated too.
    calls += [("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json"),
              ("GET", "/health"), ("GET", "/no-such-route"), ("OPTIONS", "/agents")]
    return calls


def _client(headers: dict[str, str] | None = None) -> AsyncClient:
    """Build an in-process client for the API app.

    Args:
        headers: Default headers for every request (e.g. ``AUTH_HEADERS``).

    Returns:
        An ``AsyncClient`` bound to the app through ``ASGITransport``.
    """
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=headers)


def test_route_table_is_not_empty() -> None:
    """The route sweep below covers the sensitive routes (guards against a vacuous pass)."""
    paths = {p for _, p in _route_calls()}
    for expected in ("/tasks", "/config", "/config/import", "/admin/files/dummy",
                     "/tasks/dummy/save-to-notion", "/tools/search", "/agents"):
        assert expected in paths


@pytest.mark.anyio
@pytest.mark.parametrize(
    "headers",
    [
        None,
        {"Authorization": f"Bearer {WRONG_TOKEN}"},
        {"Authorization": f"Basic {TEST_TOKEN}"},
        {"Authorization": "Bearer"},
        {"Authorization": f"Bearer {TEST_TOKEN[:-1]}"},
    ],
    ids=["none", "wrong-token", "wrong-scheme", "empty-bearer", "prefix-of-token"],
)
async def test_every_route_refuses_bad_credentials(headers) -> None:
    """Each route class (reads, task starts, config import/export, admin files,
    Notion save, tools, docs, unknown paths) answers 401 to a bad credential."""
    async with _client(headers) as client:
        for method, path in _route_calls():
            resp = await client.request(method, path)
            assert resp.status_code == 401, (method, path, resp.status_code)
            assert resp.json() == {"detail": "unauthorized"}
            assert resp.headers["www-authenticate"] == "Bearer"


@pytest.mark.anyio
async def test_token_in_query_string_is_not_accepted() -> None:
    """A token passed as a query parameter is ignored (header only), so 401."""
    async with _client() as client:
        for name in ("token", "access_token", "api_key"):
            resp = await client.get("/agents", params={name: TEST_TOKEN})
            assert resp.status_code == 401


@pytest.mark.anyio
async def test_authenticated_requests_are_served() -> None:
    """With the right token, read routes and the OpenAPI document answer 200."""
    async with _client(AUTH_HEADERS) as client:
        assert (await client.get("/agents")).status_code == 200
        assert (await client.get("/openapi.json")).status_code == 200
        # Scheme is case-insensitive per RFC 7235.
        lower = {"Authorization": f"bearer {TEST_TOKEN}"}
        assert (await client.get("/agents", headers=lower)).status_code == 200


@pytest.mark.anyio
@pytest.mark.parametrize(
    "server_token",
    [None, "", "short-token", "change-me-to-a-strong-random-secret-value-xyz"],
    ids=["unset", "empty", "too-short", "placeholder"],
)
async def test_unconfigured_server_fails_closed(monkeypatch, server_token) -> None:
    """Without a usable server token every request gets 503, even one that sends
    a header matching the (bad) configured value."""
    if server_token is None:
        monkeypatch.delenv(auth.TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(auth.TOKEN_ENV, server_token)
    sent = {"Authorization": f"Bearer {server_token or 'x'}"}
    async with _client(sent) as client:
        for path in ("/agents", "/config", "/openapi.json"):
            resp = await client.get(path)
            assert resp.status_code == 503
            assert resp.json() == {"detail": "auth not configured"}


@pytest.mark.anyio
async def test_cors_preflight_is_answered_without_token_and_leaks_nothing() -> None:
    """The browser UI's preflight succeeds without a token (it cannot send one),
    returns no data, and the real request after it is still refused."""
    origin = {"Origin": "http://localhost:4102"}
    async with _client() as client:
        pre = await client.options(
            "/agents",
            headers={**origin, "Access-Control-Request-Method": "GET",
                     "Access-Control-Request-Headers": "authorization"},
        )
        assert pre.status_code == 200
        assert pre.headers["access-control-allow-origin"] == "http://localhost:4102"
        assert "agent" not in pre.text.lower()
        real = await client.get("/agents", headers=origin)
        assert real.status_code == 401
        # The 401 carries CORS headers so the UI can read it and ask for a token.
        assert real.headers["access-control-allow-origin"] == "http://localhost:4102"


def test_compare_is_constant_time(monkeypatch) -> None:
    """``is_authorized`` decides through ``hmac.compare_digest`` on equal-length digests."""
    seen = []
    real = auth.hmac.compare_digest

    def spy(a, b):
        seen.append((len(a), len(b)))
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    assert auth.is_authorized(f"Bearer {TEST_TOKEN}".encode())
    assert not auth.is_authorized(b"Bearer nope")
    assert seen == [(32, 32), (32, 32)]


@pytest.mark.anyio
async def test_websocket_is_refused_without_token() -> None:
    """A WebSocket handshake without the token is closed (1008) before the app runs."""
    called = []

    async def inner(scope, receive, send):
        called.append(scope)

    sent = []

    async def send(message):
        sent.append(message)

    gate = auth.BearerTokenMiddleware(inner)
    await gate({"type": "websocket", "path": "/ws", "headers": []}, None, send)
    assert sent == [{"type": "websocket.close", "code": 1008}]
    assert called == []


@pytest.mark.anyio
async def test_token_never_logged(caplog) -> None:
    """Refusals and successes log no token: neither the real one nor a wrong one."""
    caplog.set_level(logging.DEBUG)
    async with _client() as client:
        await client.get("/agents", headers={"Authorization": f"Bearer {WRONG_TOKEN}"})
        await client.get("/agents", headers=AUTH_HEADERS)
        await client.get("/no-such-route")
    assert "auth: refused GET /agents" in caplog.text  # the refusal was logged...
    assert TEST_TOKEN not in caplog.text                # ...without any token
    assert WRONG_TOKEN not in caplog.text


def test_startup_refuses_without_token_and_names_no_value(monkeypatch) -> None:
    """Both entrypoints raise before uvicorn starts when the token is unusable, and
    the error names the variable but never echoes the bad value."""
    import uvicorn

    from hyperion.server import api as api_mod
    from hyperion.server import mcp as mcp_mod

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append(a))
    bad = "short-secret-value"
    monkeypatch.setenv(auth.TOKEN_ENV, bad)
    for entry in (api_mod.main, mcp_mod.main):
        with pytest.raises(auth.TokenNotConfigured) as err:
            entry()
        assert auth.TOKEN_ENV in str(err.value)
        assert bad not in str(err.value)
    assert started == []

    monkeypatch.setenv(auth.TOKEN_ENV, TEST_TOKEN)
    api_mod.main()
    assert len(started) == 1  # a good token lets the API start


# --------------------------------------------------------------------------- MCP

_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "test", "version": "0"}},
}
_MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def test_mcp_refuses_without_token() -> None:
    """The MCP endpoint and any other path on :4101 answer 401 without the token."""
    from hyperion.server.mcp import make_starlette_app

    with TestClient(make_starlette_app()) as client:
        assert client.post("/mcp", json=_INIT, headers=_MCP_HEADERS).status_code == 401
        bad = {**_MCP_HEADERS, "Authorization": f"Bearer {WRONG_TOKEN}"}
        assert client.post("/mcp", json=_INIT, headers=bad).status_code == 401
        assert client.get("/").status_code == 401
        assert client.get("/health").status_code == 401


def test_mcp_serves_with_token() -> None:
    """With the token, an MCP ``initialize`` call succeeds."""
    from hyperion.server.mcp import make_starlette_app

    with TestClient(make_starlette_app()) as client:
        resp = client.post("/mcp", json=_INIT, headers={**_MCP_HEADERS, **AUTH_HEADERS})
        assert resp.status_code == 200
        assert "serverInfo" in resp.text


@pytest.mark.anyio
async def test_mcp_proxy_forwards_token_to_api(monkeypatch) -> None:
    """``_post_api`` (approve/feedback) sends the bearer token to the API."""
    from hyperion.server import mcp as mcp_mod

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"task_id": "t1", "status": "running"})

    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    out = await mcp_mod._post_api("/tasks/t1/approve", {"action": "approve"})
    assert captured["auth"] == f"Bearer {TEST_TOKEN}"
    assert "task_id: t1" in out[0].text
