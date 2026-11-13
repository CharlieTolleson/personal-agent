"""Phase 9 tests: agent card, callback SSRF guard, config export/import round-trip.

This suite exercises the public-facing hardening and interoperability features
added to the Hyperion FastAPI server (``hyperion.server.api``) in Phase 9:

* **SSRF guard** (``hyperion.server.webhooks.validate_callback_url``) — defends the
  task-completion callback feature against Server-Side Request Forgery. A
  user-supplied ``callback_url`` is resolved and rejected if it points at a
  public/non-loopback host or uses a non-HTTP scheme, unless the
  ``hyperion_callback_ssrf_guard`` setting is turned ``off``. These tests cover
  both the standalone validator and its enforcement at the ``POST /tasks``
  endpoint.
* **Agent card** — the ``/.well-known/agent.json`` discovery descriptor that lets
  other agents/clients introspect Hyperion's name, schema version, endpoints, and
  skills.
* **Task submission validation** — ``POST /tasks`` must reject unknown request
  schema versions and unsafe callback URLs with HTTP 422.
* **Config export/import round-trip** — ``GET /config/export`` produces a zip of
  the agent registry; ``POST /config/import`` ingests it back, allowing config to
  survive a wipe.

Testing notes / non-obvious context:

* Tests that hit the live DNS resolver use ``monkeypatch`` to stub
  ``webhooks.socket.getaddrinfo``, so the SSRF outcome is deterministic and does
  not depend on the test host's network or DNS.
* Async tests are marked ``@pytest.mark.anyio`` and driven over an in-process
  ASGI transport (no real socket), with the backend pinned to asyncio via the
  ``anyio_backend`` fixture.
* Several tests reach into module-level state (``settings.tasks_dir``,
  ``settings.config_dir``) to redirect the SQLite DB and config directory into a
  per-test ``tmp_path``, keeping tests isolated and side-effect free.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from httpx import ASGITransport, AsyncClient

from auth_helpers import AUTH_HEADERS  # every route needs the bearer token

import hyperion.server.api as api
from hyperion.server import webhooks
from hyperion.server.api import app
from hyperion.server.webhooks import UnsafeCallbackURL, validate_callback_url


@pytest.fixture
def anyio_backend():
    """Pin anyio's backend to asyncio for the ``@pytest.mark.anyio`` async tests.

    Returns:
        str: The backend name ``"asyncio"``, preventing anyio from also
        parametrizing the async tests over trio.
    """
    return "asyncio"


async def _client():
    """Build an in-process HTTP client bound to the Hyperion FastAPI app.

    Uses httpx's ``ASGITransport`` so requests are dispatched directly to the
    ASGI ``app`` object in-memory — no real network socket or running server is
    involved.

    Returns:
        AsyncClient: An httpx client targeting the app at base URL
        ``http://test``. The caller is responsible for entering it as an async
        context manager (``async with await _client() as client: ...``) so it is
        properly closed.
    """
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS)


# --- SSRF guard -------------------------------------------------------------


def test_callback_guard_allows_loopback():
    """Loopback callback URLs pass the SSRF guard without raising.

    A ``127.0.0.1`` target is considered safe, so ``validate_callback_url``
    should return normally (no ``UnsafeCallbackURL``).
    """
    validate_callback_url("http://127.0.0.1:9000/hook")  # no raise


def test_callback_guard_rejects_public_host(monkeypatch):
    """A URL resolving to a public IP is rejected as an SSRF risk.

    The resolver is stubbed to return a public address so the outcome is
    deterministic; the guard must raise ``UnsafeCallbackURL``.
    """
    # Force a public-looking resolution regardless of the test host's DNS.
    monkeypatch.setattr(
        webhooks.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 80))],
    )
    with pytest.raises(UnsafeCallbackURL):
        validate_callback_url("http://example.com/hook")


def test_callback_guard_rejects_non_http_scheme():
    """Non-HTTP schemes (e.g. ``file://``) are rejected by the SSRF guard.

    Only http/https callbacks are permitted; a ``file:///etc/passwd`` URL must
    raise ``UnsafeCallbackURL`` to block local-file/scheme abuse.
    """
    with pytest.raises(UnsafeCallbackURL):
        validate_callback_url("file:///etc/passwd")


def test_callback_guard_off_skips_check(monkeypatch):
    """With the guard disabled, even a public URL is accepted unchecked.

    Setting ``hyperion_callback_ssrf_guard`` to ``"off"`` short-circuits all
    validation, so a normally-unsafe public URL returns without raising.
    """
    from hyperion.config import settings

    monkeypatch.setattr(settings, "hyperion_callback_ssrf_guard", "off")
    validate_callback_url("http://example.com/anything")  # no raise


# --- agent card -------------------------------------------------------------


@pytest.mark.anyio
async def test_agent_card_descriptor():
    """The agent card endpoint returns a well-formed discovery descriptor.

    ``GET /.well-known/agent.json`` should respond 200 with schema_version 1,
    name "Hyperion", a "submit" entry in ``endpoints``, and a list of ``skills``.
    """
    async with await _client() as client:
        resp = await client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["schema_version"] == 1
    assert card["name"] == "Hyperion"
    assert "submit" in card["endpoints"]
    assert isinstance(card["skills"], list)


# --- submit validation ------------------------------------------------------


@pytest.mark.anyio
async def test_submit_rejects_bad_schema_version(tmp_path):
    """``POST /tasks`` rejects an unknown request ``schema_version`` with 422.

    Redirects the SQLite DB into ``tmp_path`` for isolation, then submits a task
    with an unsupported ``schema_version`` (99); the request must be rejected as
    unprocessable.
    """
    # Point the server at a throwaway DB under tmp_path so the test is isolated;
    # close the freshly-opened connection immediately (we only need the file).
    api.settings.tasks_dir = tmp_path
    await (await api._get_db()).close()
    async with await _client() as client:
        resp = await client.post("/tasks", json={"task": "x", "schema_version": 99})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_submit_rejects_unsafe_callback(tmp_path, monkeypatch):
    """``POST /tasks`` rejects an unsafe ``callback_url`` with 422.

    The resolver is stubbed to map the callback host to a public IP, so the SSRF
    guard fires during task submission and the endpoint returns 422.
    """
    # Isolate state in tmp_path; close the just-opened connection (file only).
    api.settings.tasks_dir = tmp_path
    await (await api._get_db()).close()
    monkeypatch.setattr(
        webhooks.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 80))],
    )
    async with await _client() as client:
        resp = await client.post(
            "/tasks", json={"task": "x", "callback_url": "http://evil.example/hook"}
        )
    assert resp.status_code == 422


# --- config export / import round-trip --------------------------------------


@pytest.mark.anyio
async def test_config_export_import_roundtrip(tmp_path, monkeypatch):
    """Exported config zip can be re-imported to fully reconstruct the registry.

    Seeds two agents, exports them via ``GET /config/export`` (verifying the zip
    content-type and that ``agents/planner.json`` is present), wipes the on-disk
    store, then re-imports the same zip via ``POST /config/import``. The import
    must report both agent ids as imported and recreate ``planner.json`` on disk.
    """
    from hyperion.config import settings

    # Redirect the config store to tmp_path so seeding/wiping is sandboxed.
    monkeypatch.setattr(settings, "config_dir", tmp_path)

    # Seed a couple of persona agents.
    from hyperion.agents.registry import AgentRecord, save_agent

    save_agent(AgentRecord(id="planner", name="Planner",
                           role="r", goal="g", backstory="b"))
    save_agent(AgentRecord(id="synth", name="Synth",
                           role="r", goal="g", backstory="b"))

    async with await _client() as client:
        export = await client.get("/config/export")
        assert export.status_code == 200
        assert export.headers["content-type"] == "application/zip"

        names = zipfile.ZipFile(io.BytesIO(export.content)).namelist()
        assert "agents/planner.json" in names

        # Wipe the store, then import the exported zip back.
        for p in tmp_path.glob("agents/*.json"):
            p.unlink()
        resp = await client.post(
            "/config/import",
            files={"file": ("cfg.zip", export.content, "application/zip")},
        )
    assert resp.status_code == 200
    assert set(resp.json()["imported"]) == {"planner", "synth"}
    assert (tmp_path / "agents" / "planner.json").exists()


@pytest.mark.anyio
async def test_config_import_rejects_bad_zip():
    """``POST /config/import`` rejects a payload that is not a valid zip with 422.

    Uploading raw bytes that are not a zip archive must fail validation rather
    than crash, returning HTTP 422.
    """
    async with await _client() as client:
        resp = await client.post(
            "/config/import",
            files={"file": ("x.zip", b"not a zip", "application/zip")},
        )
    assert resp.status_code == 422
