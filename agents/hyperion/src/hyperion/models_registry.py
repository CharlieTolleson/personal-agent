"""
models_registry.py — operator-editable registry of role models and model aliases.

Role in the system:
  Hyperion historically hard-coded two things: the three "role" model slots
  (``model_planner``/``model_worker``/``model_cheap`` on ``settings``) and the four
  multi-provider "aliases" (``smart``/``worker``/``cheap``/``fast``) that those roles
  point at. The aliases were mirrored in code (``agents.registry.MODEL_ALIASES`` and
  ``server.api._ALIAS_DETAILS``) and defined for real in
  ``ai-router/litellm_config.yaml``. This module turns both into first-class,
  operator-editable data so the Hyperion UI can add/rename/remove roles and define new
  aliases with custom fallback chains.

Source of truth & persistence:
  The registry is a small JSON document at ``{config_dir}/model_registry.json`` with two
  keys — ``roles`` (an ordered list of ``{name, note, model}``) and ``aliases`` (a map of
  ``name -> ordered list of concrete model ids``). It is read fresh on every access (no
  process-level cache) so a patched ``config_dir`` in tests/Docker is always respected,
  mirroring the ``_model_override_path`` pattern in ``server.api``. When the file is
  missing or malformed the built-in defaults below are returned, so a fresh install
  behaves exactly like the previous hard-coded setup.

Relationship to LiteLLM:
  An alias's ordered model list is the *intended* within-group provider order. LiteLLM
  actually routes by registering one deployment per concrete model under the alias's
  ``model_name`` (see ``tools.litellm_admin``); the role aliases shipped in
  ``litellm_config.yaml`` are the bootstrap base, and UI-defined aliases are layered on
  top via the admin API. Defining/editing an alias here does not by itself change proxy
  routing — the API layer calls the reconcile step for that.

Key design decisions:
  - ``roles`` is a list (ordered, renamable) rather than three fixed fields, but the three
    built-in role names (``planner``/``worker``/``cheap``) are still consumed by the
    hard-coded LLM factory functions in ``llms.py``; ``apply_roles_to_settings`` copies
    their chosen models back onto ``settings`` on boot so those call sites are unchanged.
  - Built-in roles and built-in aliases cannot be deleted (the factory functions and the
    proxy's YAML reference them), but their target model / chain may be edited.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from hyperion.config import settings


class Role(TypedDict):
    """A logical model slot the orchestrator selects by intent.

    Keys:
        name:  Short identifier (slug). The three built-ins — ``planner``,
               ``worker``, ``cheap`` — are consumed by the LLM factory functions.
        note:  Human-readable hint describing what the role is used for.
        model: The alias or concrete model id this role resolves to.
    """

    name: str
    note: str
    model: str


# Built-in role names consumed by the hard-coded factory functions in ``llms.py``
# (``make_planner_llm`` -> ``settings.model_planner``, etc.) and by the
# compiler/runner. These names may be re-pointed but never deleted.
BUILTIN_ROLES: tuple[str, ...] = ("planner", "worker", "cheap")

# Built-in alias names defined in ``ai-router/litellm_config.yaml``. The proxy serves
# these regardless of the registry, so they can be re-seeded but never deleted.
BUILTIN_ALIASES: tuple[str, ...] = ("smart", "worker", "cheap", "fast")

# Default roles — mirror the ``model_planner``/``model_worker``/``model_cheap`` defaults
# in ``config.py``. Used when ``model_registry.json`` is absent.
_DEFAULT_ROLES: list[Role] = [
    {"name": "planner", "note": "high-stakes planning (Planner agent)", "model": "smart"},
    {"name": "worker", "note": "research + synthesis (Researcher + Synthesizer agents)", "model": "worker"},
    {"name": "cheap", "note": "summarization sub-calls (tool compression)", "model": "cheap"},
]

# Default aliases — the within-group provider order from ``litellm_config.yaml`` (and the
# old ``_ALIAS_DETAILS`` mirror), as bare concrete model ids.
_DEFAULT_ALIASES: dict[str, list[str]] = {
    "smart": ["claude-opus-4-6", "gemini-2.5-pro", "gpt-4o"],
    "worker": ["claude-sonnet-4-6", "gemini-2.5-pro", "gpt-4o"],
    "cheap": ["claude-haiku-4-5", "gemini-2.5-flash", "gpt-4o-mini"],
    "fast": ["gemini-2.5-flash", "claude-haiku-4-5", "gpt-4o-mini"],
}

# Slug rule for role / alias names — lowercase alphanumerics, dashes, underscores.
_NAME_RE = re.compile(r"^[a-z0-9_-]+$")


def _registry_path() -> Path:
    """Path of the persisted registry file.

    Returns:
        ``{config_dir}/model_registry.json``. Computed at call time (not cached) so a
        patched ``config_dir`` in tests/Docker is respected.
    """
    return settings.config_dir / "model_registry.json"


def load_registry() -> dict:
    """Load the registry, merging any persisted file over the built-in defaults.

    A missing or malformed file yields the defaults (so a fresh install reproduces the
    previous hard-coded behavior). Each top-level key is validated independently, so a
    file that only overrides ``aliases`` keeps the default ``roles`` and vice versa.

    Returns:
        ``{"roles": list[Role], "aliases": dict[str, list[str]]}``.
    """
    data: dict = {
        "roles": [dict(r) for r in _DEFAULT_ROLES],
        "aliases": {k: list(v) for k, v in _DEFAULT_ALIASES.items()},
    }
    path = _registry_path()
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8")) or {}
        except (json.JSONDecodeError, OSError):
            loaded = {}
        if isinstance(loaded.get("roles"), list):
            data["roles"] = loaded["roles"]
        if isinstance(loaded.get("aliases"), dict):
            data["aliases"] = loaded["aliases"]
    return data


def save_registry(data: dict) -> None:
    """Persist the registry document to ``model_registry.json``.

    Args:
        data: ``{"roles": [...], "aliases": {...}}``. Written verbatim (callers are
            responsible for validation via :func:`validate_registry`).

    Side effects:
        Creates ``config_dir`` if needed and overwrites the file.
    """
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def roles() -> list[Role]:
    """Return the current list of roles (built-ins first by default)."""
    return load_registry()["roles"]


def aliases() -> dict[str, list[str]]:
    """Return the current ``alias name -> ordered concrete model ids`` map."""
    return load_registry()["aliases"]


def alias_names() -> tuple[str, ...]:
    """Return all known alias names (built-in + user-defined), order preserved.

    Replaces the former ``agents.registry.MODEL_ALIASES`` constant as the canonical
    set of valid alias handles used by agent/role validation.
    """
    return tuple(load_registry()["aliases"].keys())


def role_model(name: str) -> str | None:
    """Return the model a role resolves to, or ``None`` if the role is absent."""
    for r in load_registry()["roles"]:
        if r.get("name") == name:
            return r.get("model")
    return None


def _provider_of(model_id: str) -> str:
    """Best-effort provider label for a concrete model id (display only).

    Infers the vendor from the id family so the UI/`/config` can show
    ``claude-opus-4-6 (anthropic)`` the way the old ``_ALIAS_DETAILS`` did, without a
    hard-coded per-model table. Unknown families fall back to ``"?"``.
    """
    if model_id.startswith("claude"):
        return "anthropic"
    if model_id.startswith("gpt") or model_id.startswith("o1") or "embedding" in model_id:
        return "openai"
    if model_id.startswith("gemini"):
        return "gemini"
    return "?"


def alias_details() -> dict[str, list[str]]:
    """Annotated alias chains for display (``name -> ["model (provider)", ...]``).

    Replaces the former hard-coded ``server.api._ALIAS_DETAILS``; the annotation is
    derived from :func:`_provider_of` so user-defined aliases render consistently.
    """
    return {
        name: [f"{m} ({_provider_of(m)})" for m in chain]
        for name, chain in load_registry()["aliases"].items()
    }


def seed_from_settings_if_missing() -> None:
    """One-time migration: create the registry file seeded from current ``settings``.

    On the first boot after this feature lands, no ``model_registry.json`` exists yet but
    an install may already have ``model_planner``/``model_worker``/``model_cheap`` set via
    env or the legacy ``models.json`` (PUT /config). This captures those into the built-in
    roles and persists the registry, so the migration is lossless. No-op once the file
    exists.

    Must run *after* ``server.api._apply_model_overrides`` so the legacy ``models.json``
    values are already on ``settings``.

    Side effects:
        Writes ``model_registry.json`` on first run only.
    """
    if _registry_path().exists():
        return
    data = load_registry()  # built-in defaults
    field_by_role = {"planner": "model_planner", "worker": "model_worker", "cheap": "model_cheap"}
    for r in data["roles"]:
        field = field_by_role.get(r.get("name", ""))
        if field:
            r["model"] = getattr(settings, field)
    save_registry(data)


def apply_roles_to_settings() -> None:
    """Copy the built-in roles' chosen models onto ``settings`` (boot-time).

    The LLM factory functions read ``settings.model_planner``/``model_worker``/
    ``model_cheap``; this keeps them in sync with the registry's same-named roles so the
    existing call sites need no change. Roles other than the three built-ins are ignored
    here (they have no fixed factory consumer). A missing built-in role leaves the
    corresponding ``settings`` field at its env/default value.

    Side effects:
        Mutates the global ``settings`` object.
    """
    field_by_role = {
        "planner": "model_planner",
        "worker": "model_worker",
        "cheap": "model_cheap",
    }
    for r in load_registry()["roles"]:
        field = field_by_role.get(r.get("name", ""))
        if field and r.get("model"):
            setattr(settings, field, r["model"])


def validate_registry(data: dict, known_models: list[str]) -> None:
    """Validate a candidate registry document; raise ``ValueError`` on any problem.

    Checks:
      - every role/alias name is a slug and unique within its collection;
      - the three built-in roles and four built-in aliases are still present;
      - each alias chain is a non-empty list of concrete model ids that the proxy
        reports (skipped when ``known_models`` is empty — i.e. proxy unreachable —
        so offline edits aren't blocked);
      - each role's ``model`` is a known alias name or a known concrete model id
        (also skipped when ``known_models`` is empty for concrete-id checks).

    Args:
        data: Candidate ``{"roles": [...], "aliases": {...}}`` document.
        known_models: Concrete model ids the proxy reports (from ``_litellm_model_ids``).
            Empty means "can't validate concrete ids" and relaxes those checks.

    Raises:
        ValueError: with a human-readable message describing the first failure.
    """
    roles_in = data.get("roles", [])
    aliases_in = data.get("aliases", {})

    if not isinstance(roles_in, list):
        raise ValueError("roles must be a list")
    if not isinstance(aliases_in, dict):
        raise ValueError("aliases must be an object")

    alias_keys = list(aliases_in.keys())
    seen_aliases: set[str] = set()
    for name in alias_keys:
        if not _NAME_RE.match(name):
            raise ValueError(f"Alias name {name!r} must be a slug ([a-z0-9_-])")
        if name in seen_aliases:
            raise ValueError(f"Duplicate alias {name!r}")
        seen_aliases.add(name)
        chain = aliases_in[name]
        if not isinstance(chain, list) or not chain:
            raise ValueError(f"Alias {name!r} must have a non-empty list of models")
        for m in chain:
            if not isinstance(m, str) or not m:
                raise ValueError(f"Alias {name!r} has an invalid model entry")
            if known_models and m not in known_models:
                raise ValueError(
                    f"Alias {name!r} references unknown model {m!r}. "
                    f"Use a concrete model id the proxy reports."
                )

    for builtin in BUILTIN_ALIASES:
        if builtin not in seen_aliases:
            raise ValueError(f"Built-in alias {builtin!r} cannot be removed")

    seen_roles: set[str] = set()
    for r in roles_in:
        if not isinstance(r, dict):
            raise ValueError("Each role must be an object")
        name = r.get("name", "")
        if not _NAME_RE.match(name):
            raise ValueError(f"Role name {name!r} must be a slug ([a-z0-9_-])")
        if name in seen_roles:
            raise ValueError(f"Duplicate role {name!r}")
        seen_roles.add(name)
        model = r.get("model", "")
        if not model:
            raise ValueError(f"Role {name!r} must name a model")
        # Concrete-id targets can only be checked when the proxy list is available.
        if model not in alias_keys and known_models and model not in known_models:
            raise ValueError(
                f"Role {name!r} references unknown model {model!r}. "
                f"Use an alias or a concrete model id."
            )

    for builtin in BUILTIN_ROLES:
        if builtin not in seen_roles:
            raise ValueError(f"Built-in role {builtin!r} cannot be removed")
