"""Guards that every shipped agent record only names tools that exist.

``build_tools`` raises ``ValueError`` on an unknown tool name, but only when a run
builds that agent, so a stale name in a record surfaces as a failed run rather than
at startup. This happened for real when Qdrant was removed (2026-10-26): the
``second_brain`` and ``recall_similar_tasks`` tools left the registry and the
researcher and planner records had to change with it. These tests make that kind of
drift fail in CI instead.

Covered:
  * the git-tracked records in ``config/agents/*.json`` (what the container mounts);
  * the literal ``SEEDS`` in ``scripts/seed_agents.py`` (what a fresh store gets);
  * the follow-up chat's required tools in ``hyperion.followup``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from hyperion.agents.registry import TOOL_REGISTRY

_ROOT = Path(__file__).resolve().parents[1]
_RECORD_FILES = sorted((_ROOT / "config" / "agents").glob("*.json"))


def _load_seeds() -> list:
    """Import ``scripts/seed_agents.py`` by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("seed_agents", _ROOT / "scripts" / "seed_agents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SEEDS


def test_record_files_exist():
    """The config/agents directory ships at least one record (guards a bad glob path)."""
    assert _RECORD_FILES, f"no agent records found under {_ROOT / 'config' / 'agents'}"


@pytest.mark.parametrize("record_file", _RECORD_FILES, ids=lambda p: p.stem)
def test_shipped_record_tools_are_registered(record_file: Path):
    """Each config/agents/*.json record lists only tools present in TOOL_REGISTRY."""
    tools = json.loads(record_file.read_text(encoding="utf-8")).get("tools", [])
    missing = [t for t in tools if t not in TOOL_REGISTRY]
    assert not missing, f"{record_file.name} lists unregistered tools: {missing}"


def test_seed_record_tools_are_registered():
    """Every record in seed_agents.SEEDS lists only tools present in TOOL_REGISTRY."""
    problems = {
        rec.id: [t for t in rec.tools if t not in TOOL_REGISTRY]
        for rec in _load_seeds()
    }
    problems = {k: v for k, v in problems.items() if v}
    assert not problems, f"seed records list unregistered tools: {problems}"


def test_followup_required_tools_are_registered():
    """The follow-up chat's always-on retrieval tools all exist in TOOL_REGISTRY."""
    from hyperion.followup import _FOLLOWUP_REQUIRED_TOOLS

    missing = [t for t in _FOLLOWUP_REQUIRED_TOOLS if t not in TOOL_REGISTRY]
    assert not missing, f"follow-up requires unregistered tools: {missing}"
