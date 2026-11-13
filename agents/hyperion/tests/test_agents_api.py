"""Integration tests for Hyperion's agent-management HTTP API (Phase 5).

Exercises the FastAPI app exposed by ``hyperion.server.api`` end-to-end via an
in-process ASGI transport (no network sockets), covering the agent CRUD surface
and the supporting "options" endpoints that the Hyperion UI relies on:

  * ``GET    /agents``                  — list all configured agents
  * ``GET    /agents/{id}``             — fetch a single agent (404 when missing)
  * ``POST   /agents``                  — create an agent, with validation:
        duplicate id -> 409, unknown tool -> 422
  * ``DELETE /agents/{id}``             — delete an agent
  * ``POST   /agents/{id}/duplicate``   — clone an agent under a "-copy" id
  * ``GET    /tools`` / ``GET /models`` — enumerate available tools / model aliases
  * ``PUT    /config``                  — reassign per-role models, with model-id
        validation against the LiteLLM model catalog

Design notes / non-obvious context:
  * Every test runs against an isolated, on-disk agent store rooted at pytest's
    ``tmp_path``. The ``_env`` context manager patches ``settings.config_dir`` to
    that path and seeds a minimal valid agent set, so tests never touch the
    developer's real config and stay independent of one another.
  * ``hyperion.server.api._litellm_model_ids`` is stubbed with an ``AsyncMock`` so
    model validation never makes a real LiteLLM network call. Returning ``[]``
    effectively disables unknown-model rejection (an empty catalog accepts any
    id); tests that need rejection (e.g. ``test_put_config_rejects_unknown_model``)
    set up their own patch with a populated catalog instead of using ``_env``.
  * Tests are async and use the ``anyio`` plugin; the ``anyio_backend`` fixture
    pins the backend to asyncio.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from auth_helpers import AUTH_HEADERS  # every route needs the bearer token

from hyperion.agents.registry import AgentRecord, save_agent
from hyperion.server.api import app


@pytest.fixture
def anyio_backend():
    """Pin the anyio test backend to asyncio.

    Returns:
        str: The backend name ("asyncio") that ``@pytest.mark.anyio`` tests run on.
            Without this, anyio would also parametrize tests over trio.
    """
    return "asyncio"


def _rec(id_, active=True, tools=None):
    """Build a minimal valid ``AgentRecord`` for use as test fixture data / request bodies.

    Args:
        id_: Agent identifier; also reused as ``name``/``role`` to keep records terse.
        active: Whether the agent is active. Defaults to True.
        tools: Optional list of tool names to attach; ``None`` becomes ``[]``.

    Returns:
        AgentRecord: A persona record with placeholder goal/backstory. Agents carry
        no ordering/activation metadata — that lives on the workflow that uses them.
    """
    return AgentRecord(
        id=id_, name=id_, role=id_, goal="g", backstory="b",
        active=active, tools=tools or [],
    )


@contextlib.contextmanager
def _env(tmp_path):
    """Patch config_dir to an isolated store seeded with a few agents, and stub the
    LiteLLM model lookup so model validation never hits the network.

    Seeds three persona agents (planner, researcher with the web_search tool, and
    synthesizer) so individual tests can focus on the behavior under test.

    Args:
        tmp_path: pytest-provided temp directory used as the isolated config_dir.

    Yields:
        None: Control returns to the test body while both patches are active; the
            patches and the temp store are torn down on exit.

    Side effects:
        Writes agent JSON files under ``tmp_path`` via ``save_agent``. Patches
        ``settings.config_dir`` and ``_litellm_model_ids`` for the duration of the
        ``with`` block (the stub returns ``[]``, an empty model catalog).
    """
    from hyperion.config import settings

    with patch.object(settings, "config_dir", tmp_path), \
         patch("hyperion.server.api._litellm_model_ids", new=AsyncMock(return_value=[])):
        save_agent(_rec("planner"))
        save_agent(_rec("researcher", tools=["web_search"]))
        save_agent(_rec("synthesizer"))
        yield


def _client():
    """Construct an httpx ``AsyncClient`` bound to the FastAPI app in-process.

    Uses ``ASGITransport`` so requests are dispatched directly to ``app`` without
    opening a real network socket; the ``base_url`` is an arbitrary placeholder.

    Returns:
        AsyncClient: An un-entered client; callers use it via ``async with``.
    """
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS)


@pytest.mark.anyio
async def test_list_and_get_agents(tmp_path):
    """GET /agents lists the seeded ids; GET /agents/{id} returns one or 404s when absent."""
    with _env(tmp_path):
        async with _client() as c:
            resp = await c.get("/agents")
            assert resp.status_code == 200
            assert {a["id"] for a in resp.json()} == {"planner", "researcher", "synthesizer"}
            assert (await c.get("/agents/planner")).json()["role"] == "planner"
            assert (await c.get("/agents/ghost")).status_code == 404


@pytest.mark.anyio
async def test_create_agent_participates(tmp_path):
    """POST /agents creates a valid agent (201) and it then appears in the agent list."""
    with _env(tmp_path):
        async with _client() as c:
            body = _rec("developer", tools=["web_search"]).model_dump()
            resp = await c.post("/agents", json=body)
            assert resp.status_code == 201
            ids = {a["id"] for a in (await c.get("/agents")).json()}
            assert "developer" in ids


@pytest.mark.anyio
async def test_create_duplicate_id_conflicts(tmp_path):
    """POST /agents with an id that already exists (planner) is rejected with 409 Conflict."""
    with _env(tmp_path):
        async with _client() as c:
            resp = await c.post("/agents", json=_rec("planner").model_dump())
            assert resp.status_code == 409


@pytest.mark.anyio
async def test_create_unknown_tool_rejected(tmp_path):
    """POST /agents referencing a non-existent tool returns 422 with an 'unknown tool' detail."""
    with _env(tmp_path):
        async with _client() as c:
            body = _rec("badtool", tools=["does_not_exist"]).model_dump()
            resp = await c.post("/agents", json=body)
            assert resp.status_code == 422
            assert "unknown tool" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_delete_agent_ok(tmp_path):
    """DELETE of any agent succeeds (200) and echoes the deleted id. Agents are pure
    personas — there is no cross-record invariant forbidding a deletion."""
    with _env(tmp_path):
        async with _client() as c:
            resp = await c.delete("/agents/synthesizer")
            assert resp.status_code == 200
            assert resp.json()["deleted"] == "synthesizer"


@pytest.mark.anyio
async def test_duplicate_agent(tmp_path):
    """POST /agents/{id}/duplicate clones the agent (201) under the conventional '<id>-copy' id."""
    with _env(tmp_path):
        async with _client() as c:
            resp = await c.post("/agents/researcher/duplicate")
            assert resp.status_code == 201
            assert resp.json()["id"] == "researcher-copy"


@pytest.mark.anyio
async def test_tools_and_models_endpoints(tmp_path):
    """GET /tools exposes registered tools (web_search) and GET /models exposes aliases + current per-stage assignments."""
    with _env(tmp_path):
        async with _client() as c:
            tools = (await c.get("/tools")).json()
            assert "web_search" in {t["name"] for t in tools}
            models = (await c.get("/models")).json()
            assert "smart" in models["aliases"]
            assert "planner" in models["current"]


@pytest.mark.anyio
async def test_put_config_reassigns_model(tmp_path):
    """PUT /config updates the planner model alias (200), mutates live settings, and persists to models.json."""
    from hyperion.config import settings

    with _env(tmp_path):
        async with _client() as c:
            resp = await c.put("/config", json={"model_planner": "fast"})
            assert resp.status_code == 200
            assert settings.model_planner == "fast"
            # persisted for next boot
            assert (tmp_path / "models.json").exists()


@pytest.mark.anyio
async def test_put_config_rejects_unknown_model(tmp_path):
    """PUT /config with a model id absent from a populated LiteLLM catalog is rejected with 422.

    Unlike the other tests, this one bypasses ``_env`` and sets up its own patches so the
    stubbed ``_litellm_model_ids`` returns a non-empty catalog (["gpt-4o"]); validation only
    rejects unknown ids when the catalog is non-empty.
    """
    # With a populated model list, an unknown id is rejected.
    from hyperion.config import settings

    with patch.object(settings, "config_dir", tmp_path), \
         patch("hyperion.server.api._litellm_model_ids", new=AsyncMock(return_value=["gpt-4o"])):
        save_agent(_rec("planner"))
        save_agent(_rec("synthesizer"))
        async with _client() as c:
            resp = await c.put("/config", json={"model_worker": "nope-model"})
            assert resp.status_code == 422
