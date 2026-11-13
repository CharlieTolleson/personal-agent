"""Phase 8 endpoint tests: /thresholds, /tasks (paginated), /metrics.

Async integration tests for Hyperion's observability/operations HTTP API
(``hyperion.server.api``). Each test drives the FastAPI ``app`` in-process via
httpx's ``ASGITransport`` — no network socket or running server is required.

What the suite covers:
- ``GET /thresholds``   — shape of the budget/cap config payload (global + per-agent).
- ``PUT /thresholds``   — updating a global cap mutates live ``settings`` and is
                          persisted to ``<config_dir>/thresholds.json``; non-positive
                          caps are rejected with HTTP 422.
- ``GET /tasks``        — limit/offset pagination over the SQLite ``tasks`` table,
                          including the ``total`` count and per-item field shape.
- ``GET /metrics``      — aggregation of per-agent activation/error counts derived
                          from each task's ``routing`` JSON and terminal status.

Key design / setup notes:
- Tests that hit the DB use ``_seed_db`` to repoint the API at a fresh per-test
  SQLite file under pytest's ``tmp_path``, guaranteeing isolation.
- Tests that mutate config use ``monkeypatch`` to redirect ``settings.config_dir``
  into ``tmp_path`` so writes never touch the real repo config.
- The ``anyio_backend`` fixture pins anyio to asyncio; tests are marked
  ``@pytest.mark.anyio`` to run as coroutines.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from auth_helpers import AUTH_HEADERS  # every route needs the bearer token

import hyperion.server.api as api
from hyperion.server.api import app


@pytest.fixture
def anyio_backend():
    """Pin anyio to the asyncio backend so ``@pytest.mark.anyio`` tests run on asyncio.

    Returns:
        str: The backend name ``"asyncio"`` (anyio reads this fixture to choose
            its event-loop implementation, avoiding a trio dependency).
    """
    return "asyncio"


async def _client():
    """Build an httpx client wired directly to the FastAPI app via ASGI (no network).

    Returns:
        AsyncClient: An httpx ``AsyncClient`` whose transport invokes ``app`` in
            process; ``base_url`` is a placeholder since requests never leave the
            test. Use as an async context manager so the client is closed cleanly.
    """
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS)


async def _seed_db(tmp_path, rows):
    """Point the API at a fresh temp DB and insert (task_id, status, request, routing).

    Repoints ``api.settings.tasks_dir`` at ``tmp_path`` so the dynamically
    resolved DB path (``settings.tasks_dir / state.db``) lands on the temp
    database for subsequent API requests, then bulk-inserts the given rows into
    the ``tasks`` table and commits.

    Args:
        tmp_path: pytest's per-test temp directory; the DB is created as
            ``state.db`` inside it for isolation between tests.
        rows: Iterable of ``(task_id, status, request, routing)`` tuples to insert.
            ``routing`` is the raw JSON string (or ``None``) describing agent
            selection for that task.

    Returns:
        Path: The path to the created SQLite database file.

    Side effects:
        Reassigns ``api.settings.tasks_dir`` (global state) and writes/closes a
        SQLite file.
    """
    db_path = tmp_path / "state.db"
    api.settings.tasks_dir = tmp_path
    db = await api._get_db()
    for task_id, status, request, routing in rows:
        await db.execute(
            "INSERT INTO tasks (task_id, status, request, routing) VALUES (?,?,?,?)",
            (task_id, status, request, routing),
        )
    await db.commit()
    await db.close()
    return db_path


@pytest.mark.anyio
async def test_get_thresholds_shape():
    """GET /thresholds returns 200 with global+per-agent sections and a global wall-clock cap."""
    async with await _client() as client:
        resp = await client.get("/thresholds")
    assert resp.status_code == 200
    body = resp.json()
    assert "global" in body and "agents" in body
    assert "cap_wall_seconds" in body["global"]


@pytest.mark.anyio
async def test_put_thresholds_updates_global_cap(tmp_path, monkeypatch):
    """PUT /thresholds updates the live setting, echoes it, and persists thresholds.json."""
    from hyperion.config import settings

    monkeypatch.setattr(settings, "config_dir", tmp_path)
    async with await _client() as client:
        resp = await client.put("/thresholds", json={"cap_wall_seconds": 123})
    assert resp.status_code == 200
    assert resp.json()["global"]["cap_wall_seconds"] == 123
    assert settings.cap_wall_seconds == 123
    saved = json.loads((tmp_path / "thresholds.json").read_text())
    assert saved["cap_wall_seconds"] == 123


@pytest.mark.anyio
async def test_put_thresholds_rejects_nonpositive(monkeypatch, tmp_path):
    """PUT /thresholds with a non-positive cap (e.g. -5) is rejected with HTTP 422."""
    from hyperion.config import settings

    monkeypatch.setattr(settings, "config_dir", tmp_path)
    async with await _client() as client:
        resp = await client.put("/thresholds", json={"cap_input_tokens": -5})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_tasks_pagination(tmp_path):
    """GET /tasks honors limit/offset: total reflects all rows while items is capped to limit."""
    await _seed_db(
        tmp_path,
        [
            ("aaa", "done", "first", None),
            ("bbb", "failed", "second", None),
            ("ccc", "running", "third", None),
        ],
    )
    async with await _client() as client:
        resp = await client.get("/tasks", params={"limit": 2, "offset": 0})
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert {"task_id", "status", "request", "langfuse_url"} <= set(body["items"][0])


@pytest.mark.anyio
async def test_metrics_counts_activations_and_errors(tmp_path):
    """GET /metrics aggregates per-agent activations, errors, and error_rate from task routing+status.

    Two tasks both activate "researcher"; one is "failed", so the agent shows 2
    activations, 1 error, and an error_rate of 0.5.
    """
    # Both seeded tasks select the same agent so counts roll up to one agent entry.
    routing = json.dumps({"selected_agents": ["researcher"], "skipped": [], "dag": {}})
    await _seed_db(
        tmp_path,
        [
            ("t1", "done", "ok run", routing),
            ("t2", "failed", "bad run", routing),
        ],
    )
    async with await _client() as client:
        resp = await client.get("/metrics")
    body = resp.json()
    assert body["tasks_total"] == 2
    by_id = {a["id"]: a for a in body["agents"]}
    assert by_id["researcher"]["activations"] == 2
    assert by_id["researcher"]["errors"] == 1
    assert by_id["researcher"]["error_rate"] == 0.5
