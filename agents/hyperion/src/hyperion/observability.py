"""
observability.py — Langfuse tracing helper.

Tracing is configured at the LiteLLM proxy level (success_callback: ["langfuse"]
in litellm_config.yaml). Every LLM call from any agent is automatically logged
as a generation in Langfuse, no SDK glue required on the client side.

This module exposes a helper that builds the LiteLLM `metadata` dict so all
LLM calls for a given task share a session_id (= task_id) and tags. The dict
should be passed to the LLM call via `extra_body={"metadata": ...}` (the owned
agent loop threads it into every `litellm.completion`).
"""

from __future__ import annotations


def trace_metadata(task_id: str, agent_role: str | None = None) -> dict:
    """
    Build LiteLLM metadata that groups every call for this task under one
    Langfuse session, with optional per-agent tagging.

    The returned dict is meant to be threaded into the underlying LLM call so
    that Langfuse (configured proxy-side via ``success_callback: ["langfuse"]``)
    can correlate all generations belonging to the same Hyperion task. Pass it
    to the LLM call via ``extra_body={"metadata": ...}``.

    Args:
        task_id: Unique identifier for the Hyperion task/run. Used as the
            Langfuse ``session_id`` so every LLM call across all agents in the
            same task collapses into one session/trace group.
        agent_role: Optional role of the agent making the call (e.g.
            "researcher", "critic"). When omitted, defaults to "crew" so
            crew-level / non-agent-specific calls remain attributable.

    Returns:
        A plain ``dict`` of LiteLLM/Langfuse metadata fields:
        ``generation_name``, ``trace_user_id``, ``session_id``, and ``tags``.

    Side effects:
        None — this is a pure builder function and performs no I/O.
    """
    meta: dict = {
        # Human-readable name for this generation in the Langfuse UI; namespaced
        # by agent role (falls back to "crew" when no specific role is given).
        "generation_name": f"hyperion/{agent_role or 'crew'}",
        # Constant user id so all Hyperion traffic is attributed to one "user".
        "trace_user_id": "hyperion",
        # task_id == session_id is the key that groups every call in a task.
        "session_id": task_id,
        # Tags enable filtering in Langfuse by product ("hyperion") and by role.
        "tags": ["hyperion", agent_role or "crew"],
    }
    return meta
