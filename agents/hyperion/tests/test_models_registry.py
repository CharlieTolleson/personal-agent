"""Tests for the operator-editable role/alias registry (models_registry) and its API.

Covers three layers:

  * ``models_registry`` module — default seeding, one-time migration from ``settings``
    (``seed_from_settings_if_missing``), boot re-apply to ``settings``
    (``apply_roles_to_settings``), and ``validate_registry`` rejection rules.
  * The FastAPI surface — ``GET/PUT /roles`` and ``GET/PUT/DELETE /aliases`` — exercised
    in-process via an ASGI transport, with the LiteLLM write-through stubbed so no network
    call is made.
  * ``tools.litellm_admin.reconcile_alias`` idempotency — repeated reconciles converge
    (only missing deployments are added, only stale ones removed).

Design notes:
  * Every test isolates ``settings.config_dir`` to ``tmp_path`` so the developer's real
    registry is never touched.
  * ``_litellm_model_ids`` is stubbed with a populated catalog so concrete-id validation
    is actually exercised (an empty catalog would relax those checks).
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from auth_helpers import AUTH_HEADERS  # every route needs the bearer token

from hyperion import models_registry
from hyperion.config import settings
from hyperion.server.api import app

# A catalog covering every model referenced by the default aliases, so validation passes
# for the seeded registry and only rejects genuinely-unknown ids.
_CATALOG = [
    "gpt-4o", "gpt-4o-mini", "claude-opus-4-6", "claude-sonnet-4-6",
    "claude-haiku-4-5", "gemini-2.5-pro", "gemini-2.5-flash",
]


@pytest.fixture
def anyio_backend():
    """Pin the anyio backend to asyncio (otherwise tests also parametrize over trio)."""
    return "asyncio"


def _client():
    """An un-entered in-process httpx client bound to the FastAPI app."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS)


@contextlib.contextmanager
def _env(tmp_path):
    """Isolate config_dir to tmp_path and stub the LiteLLM model catalog + write-through.

    Patches:
      - ``settings.config_dir`` -> tmp_path (registry file lands here);
      - ``_litellm_model_ids`` -> ``_CATALOG`` (validation sees a populated catalog);
      - ``reconcile_alias`` / ``alias_routing_status`` -> AsyncMocks so alias edits never
        touch the proxy admin API.
    """
    with patch.object(settings, "config_dir", tmp_path), \
         patch("hyperion.server.api._litellm_model_ids", new=AsyncMock(return_value=_CATALOG)), \
         patch("hyperion.tools.litellm_admin.reconcile_alias",
               new=AsyncMock(return_value={"status": "applied"})), \
         patch("hyperion.tools.litellm_admin.alias_routing_status",
               new=AsyncMock(return_value={})):
        yield


# ---------------------------------------------------------------------------
# models_registry module
# ---------------------------------------------------------------------------


def test_defaults_when_no_file(tmp_path):
    """A missing registry file yields the built-in default roles and aliases."""
    with patch.object(settings, "config_dir", tmp_path):
        reg = models_registry.load_registry()
        assert [r["name"] for r in reg["roles"]] == ["planner", "worker", "cheap"]
        assert set(models_registry.alias_names()) == {"smart", "worker", "cheap", "fast"}


def test_seed_and_apply_roundtrip(tmp_path):
    """seed_from_settings_if_missing captures settings; apply_roles_to_settings restores them."""
    with patch.object(settings, "config_dir", tmp_path), \
         patch.object(settings, "model_planner", "fast"):
        # First boot: no file -> capture current settings into the registry.
        models_registry.seed_from_settings_if_missing()
        assert models_registry._registry_path().exists()
        assert models_registry.role_model("planner") == "fast"

        # A later boot with a different in-memory value re-applies the persisted choice.
        with patch.object(settings, "model_planner", "smart"):
            models_registry.apply_roles_to_settings()
            assert settings.model_planner == "fast"

        # Seeding again is a no-op (does not clobber the saved file).
        models_registry.seed_from_settings_if_missing()
        assert models_registry.role_model("planner") == "fast"


def test_alias_details_annotates_provider(tmp_path):
    """alias_details renders each model with an inferred provider label for display."""
    with patch.object(settings, "config_dir", tmp_path):
        details = models_registry.alias_details()
        assert details["smart"][0] == "claude-opus-4-6 (anthropic)"


def test_validate_rejects_unknown_alias_model(tmp_path):
    """validate_registry rejects an alias chain entry absent from the proxy catalog."""
    reg = models_registry.load_registry()
    reg["aliases"]["smart"] = ["not-a-real-model"]
    with pytest.raises(ValueError, match="unknown model"):
        models_registry.validate_registry(reg, _CATALOG)


def test_validate_rejects_removing_builtin_role(tmp_path):
    """validate_registry refuses a roles list missing a built-in role name."""
    reg = models_registry.load_registry()
    reg["roles"] = [r for r in reg["roles"] if r["name"] != "cheap"]
    with pytest.raises(ValueError, match="cheap"):
        models_registry.validate_registry(reg, _CATALOG)


# ---------------------------------------------------------------------------
# API: /roles
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_roles_adds_custom_role(tmp_path):
    """PUT /roles persists a new custom role alongside the built-ins (200)."""
    with _env(tmp_path):
        async with _client() as c:
            roles = models_registry.roles() + [{"name": "critic", "note": "review", "model": "worker"}]
            resp = await c.put("/roles", json={"roles": roles})
            assert resp.status_code == 200
            assert "critic" in {r["name"] for r in resp.json()["roles"]}
            assert models_registry.role_model("critic") == "worker"


@pytest.mark.anyio
async def test_put_roles_rejects_missing_builtin(tmp_path):
    """PUT /roles that drops a built-in role is rejected with 422."""
    with _env(tmp_path):
        async with _client() as c:
            roles = [r for r in models_registry.roles() if r["name"] != "planner"]
            resp = await c.put("/roles", json={"roles": roles})
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# API: /aliases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_upsert_and_use_new_alias(tmp_path):
    """PUT /aliases/{name} defines a new routable alias; it then appears in /models aliases."""
    with _env(tmp_path):
        async with _client() as c:
            resp = await c.put("/aliases/vision", json={"models": ["gpt-4o", "gemini-2.5-pro"]})
            assert resp.status_code == 200
            assert resp.json()["status"] == {"status": "applied"}
            assert "vision" in (await c.get("/models")).json()["aliases"]


@pytest.mark.anyio
async def test_delete_builtin_alias_refused(tmp_path):
    """DELETE /aliases/{builtin} is refused with 422 (built-ins come from litellm_config.yaml)."""
    with _env(tmp_path):
        async with _client() as c:
            assert (await c.delete("/aliases/smart")).status_code == 422


@pytest.mark.anyio
async def test_delete_alias_referenced_by_role_refused(tmp_path):
    """DELETE /aliases/{name} is refused while a role still points at it."""
    with _env(tmp_path):
        async with _client() as c:
            await c.put("/aliases/vision", json={"models": ["gpt-4o"]})
            roles = models_registry.roles() + [{"name": "viz", "note": "", "model": "vision"}]
            await c.put("/roles", json={"roles": roles})
            resp = await c.delete("/aliases/vision")
            assert resp.status_code == 422
            assert "viz" in resp.json()["detail"]


@pytest.mark.anyio
async def test_delete_unreferenced_alias_ok(tmp_path):
    """DELETE /aliases/{name} succeeds for a user-defined, unreferenced alias."""
    with _env(tmp_path):
        async with _client() as c:
            await c.put("/aliases/vision", json={"models": ["gpt-4o"]})
            resp = await c.delete("/aliases/vision")
            assert resp.status_code == 200
            assert "vision" not in models_registry.alias_names()


# ---------------------------------------------------------------------------
# tools.litellm_admin.reconcile_alias — idempotency
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reconcile_alias_idempotent():
    """Reconciling twice adds deployments once, then makes no further changes."""
    from hyperion.tools import litellm_admin

    # The proxy starts with the two concrete models registered (from the YAML base).
    base_info = [
        {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}, "model_info": {"id": "d1"}},
        {"model_name": "gemini-2.5-pro", "litellm_params": {"model": "gemini/gemini-2.5-pro"}, "model_info": {"id": "d2"}},
    ]
    # Mutable view the fake admin API mutates as deployments are added.
    state = {"info": [dict(d) for d in base_info]}

    async def fake_model_info(_client):
        return state["info"]

    async def fake_add(_client, alias, upstream):
        state["info"].append({
            "model_name": alias,
            "litellm_params": {"model": upstream},
            "model_info": {"id": f"{alias}-{upstream}"},
        })

    add_mock = AsyncMock(side_effect=fake_add)
    del_mock = AsyncMock()
    with patch.object(litellm_admin, "_model_info", new=fake_model_info), \
         patch.object(litellm_admin, "_add_deployment", new=add_mock), \
         patch.object(litellm_admin, "_delete_deployment", new=del_mock):
        first = await litellm_admin.reconcile_alias("vision", ["gpt-4o", "gemini-2.5-pro"])
        assert first == {"status": "applied"}
        assert add_mock.await_count == 2  # both deployments added on first pass

        add_mock.reset_mock()
        second = await litellm_admin.reconcile_alias("vision", ["gpt-4o", "gemini-2.5-pro"])
        assert second == {"status": "applied"}
        assert add_mock.await_count == 0  # already present -> no-op
        assert del_mock.await_count == 0


@pytest.mark.anyio
async def test_reconcile_delete_retries_until_worker_synced():
    """Delete path retries /model/info so it finds the id even when a worker lags.

    Simulates LiteLLM's multi-worker cache: the first two ``/model/info`` reads miss the
    just-created deployment (as if served by an unsynced worker), the third shows it. The
    retry loop must keep reading until it finds the id, then delete it — otherwise the
    deployment orphans in the DB (the bug this guards against).
    """
    from hyperion.tools import litellm_admin

    synced = {
        "model_name": "vision",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"id": "vis-1"},
    }
    reads = {"n": 0}

    async def lagging_model_info(_client):
        reads["n"] += 1
        return [] if reads["n"] < 3 else [synced]  # first two reads miss it

    del_mock = AsyncMock()
    # Shorten the retry delay so the test is fast.
    with patch.object(litellm_admin, "_model_info", new=lagging_model_info), \
         patch.object(litellm_admin, "_delete_deployment", new=del_mock), \
         patch.object(litellm_admin, "_DELETE_LOOKUP_DELAY", 0.0):
        result = await litellm_admin.reconcile_alias("vision", None)
        assert result == {"status": "deleted"}
        assert reads["n"] >= 3  # kept retrying past the two empty reads
        # _delete_deployment is called as (client, dep_id); assert it deleted the right id.
        del_mock.assert_awaited_once()
        assert del_mock.call_args.args[1] == "vis-1"
