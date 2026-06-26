"""
Task-scoped context blackboard (Phase 4).

A per-task ``tasks/{id}/context.json`` key/value store that every stage can read
and write. It is the cross-stage channel for facts that aren't notes or artifacts —
e.g. the auto-discovered ``context_brief``, recalled prior-task ids, or a fact the
researcher wants the synthesizer to see without re-deriving.

``context_put`` / ``context_get`` are exposed both as plain functions (used by the
runner) and as tools (granted to agent records via the registry).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hyperion.config import settings

logger = logging.getLogger(__name__)


def _context_path(task_id: str):
    """Resolve the on-disk path of a task's context blackboard JSON file.

    The path is ``{settings.tasks_dir}/{task_id}/context.json``.

    Args:
        task_id: The task identifier. Used as a directory name segment.

    Returns:
        A ``pathlib.Path`` pointing at the task's ``context.json``. The file is
        not guaranteed to exist.

    Raises:
        ValueError: If ``task_id`` is empty or contains path-traversal characters
            (``/`` or ``..``). This guard prevents a malicious or malformed
            ``task_id`` from escaping ``settings.tasks_dir``.
    """
    # Reject path-traversal / nested ids so the task_id can only name a single
    # subdirectory directly under tasks_dir.
    if not task_id or "/" in task_id or ".." in task_id:
        raise ValueError(f"Invalid task_id: {task_id!r}")
    return settings.tasks_dir / task_id / "context.json"


def _load(task_id: str) -> dict[str, Any]:
    """Load a task's context blackboard from disk as a plain dict.

    Args:
        task_id: The task identifier whose context file should be read.

    Returns:
        The parsed key/value mapping. Returns an empty dict when the file does
        not exist, is empty, contains invalid JSON, or cannot be read — callers
        treat a missing/corrupt blackboard the same as an empty one rather than
        failing a stage.
    """
    path = _context_path(task_id)
    if not path.exists():
        return {}
    try:
        # ``or {}`` guards against a file whose JSON content is literally null.
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable file is treated as an empty blackboard.
        return {}


def context_put(task_id: str, key: str, value: Any) -> None:
    """Merge a single key into the task's context blackboard."""
    data = _load(task_id)
    data[key] = value
    path = _context_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def context_get(task_id: str, key: str | None = None) -> Any:
    """Read one key, or the whole blackboard when ``key`` is None."""
    data = _load(task_id)
    if key is None:
        return data
    return data.get(key)


# ---------------------------------------------------------------------------
# Tool wrappers (task-scoped via the registry factory)
# ---------------------------------------------------------------------------


class ContextPutTool:
    name = "context_put"
    description = (
        "Save a fact to the shared task context so later stages can read it. "
        "Input: a 'key' (string) and a 'value' (string)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "The context key to set."},
            "value": {"type": "string", "description": "The value to store."},
        },
        "required": ["key", "value"],
    }

    def __init__(self, task_id: str):
        self.task_id = task_id

    def _run(self, key: str, value: str = "") -> str:
        context_put(self.task_id, key, value)
        return f"Saved context key {key!r}."


class ContextGetTool:
    name = "context_get"
    description = (
        "Read shared task context written by earlier stages. "
        "Input: a key name, or empty to list all keys."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "The key to read, or empty for all keys."}
        },
    }

    def __init__(self, task_id: str):
        self.task_id = task_id

    def _run(self, key: str = "") -> str:
        if not key.strip():
            data = context_get(self.task_id)
            return json.dumps(data, ensure_ascii=False) if data else "(context is empty)"
        value = context_get(self.task_id, key)
        return "(not set)" if value is None else json.dumps(value, ensure_ascii=False)


class RecallSimilarTasksTool:
    name = "recall_similar_tasks"
    description = (
        "Search memory of past completed tasks for ones similar to a query. "
        "Use before planning to reuse prior work. Input: a natural-language query."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language query."}
        },
        "required": ["query"],
    }

    def __init__(self, task_id: str = ""):
        self.task_id = task_id

    def _run(self, query: str) -> str:
        from hyperion.memory.episodic import recall_similar_tasks

        hits = recall_similar_tasks(query, limit=5)
        if not hits:
            return "(no similar past tasks found)"
        lines = ["## Similar past tasks"]
        for h in hits:
            status = "ok" if h.get("success") else "failed"
            lines.append(
                f"- [{h.get('task_id')}] ({status}, score {h.get('score')}): "
                f"{h.get('request', '')[:160]}"
            )
        return "\n".join(lines)
