"""
usage.py — per-task / per-agent token accounting and cap enforcement (Phase 8).

Every Hyperion LLM call already tags its LiteLLM request with a metadata block
(``session_id`` = task_id, ``tags`` = ["hyperion", agent_role]; see ``llms.py``).
A LiteLLM ``CustomLogger`` reads that block to:

  * accumulate input/output tokens per (task_id, agent_role) on each success, and
  * raise ``CapExceeded`` *before* the next call once an agent's record-level
    ``thresholds.max_input_tokens`` / ``max_output_tokens`` is reached.

The runner already catches ``CapExceeded`` and aborts the run, so enforcement is
additive — no runner edit needed. Tracking powers ``GET /metrics`` usage bars.

Token attribution is best-effort: if the metadata block cannot be located in the
LiteLLM callback kwargs (provider/version differences), accounting is simply
skipped for that call rather than guessed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

# (task_id, agent_role) -> [input_tokens, output_tokens]
_usage: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Accounting API (also used directly by tests)
# ---------------------------------------------------------------------------


def record(task_id: str, agent_role: str, in_tokens: int, out_tokens: int) -> None:
    """Add a single LLM call's token counts to the in-memory usage bucket.

    Args:
        task_id: The run identifier (LiteLLM ``session_id``). Empty/falsy values
            are ignored so unattributable calls do not pollute accounting.
        agent_role: The agent role tag. Falls back to ``"unknown"`` if empty.
        in_tokens: Prompt (input) tokens for this call; coerced to int, None -> 0.
        out_tokens: Completion (output) tokens for this call; coerced to int, None -> 0.

    Returns:
        None.

    Side effects:
        Mutates the module-global ``_usage`` map under ``_lock`` (thread-safe).
    """
    # Drop calls we cannot attribute to a run rather than guessing.
    if not task_id:
        return
    with _lock:
        bucket = _usage[(task_id, agent_role or "unknown")]
        bucket[0] += int(in_tokens or 0)
        bucket[1] += int(out_tokens or 0)


def agent_totals(task_id: str, agent_role: str) -> tuple[int, int]:
    """Return accumulated ``(input_tokens, output_tokens)`` for one agent in one task.

    Args:
        task_id: The run identifier.
        agent_role: The agent role to look up.

    Returns:
        A ``(input_tokens, output_tokens)`` tuple; ``(0, 0)`` if no usage recorded.

    Side effects:
        Reads ``_usage`` under ``_lock``.
    """
    with _lock:
        bucket = _usage.get((task_id, agent_role), [0, 0])
        return bucket[0], bucket[1]


def task_totals(task_id: str) -> dict[str, Any]:
    """Aggregate usage for one task: totals plus a per-agent breakdown.

    Args:
        task_id: The run identifier to aggregate.

    Returns:
        Dict with keys ``input_tokens`` (int), ``output_tokens`` (int), and
        ``by_agent`` mapping each agent role to ``{"input": int, "output": int}``.

    Side effects:
        Reads ``_usage`` under ``_lock``.
    """
    with _lock:
        total_in = total_out = 0
        by_agent: dict[str, dict[str, int]] = {}
        for (tid, role), (i, o) in _usage.items():
            if tid != task_id:
                continue
            total_in += i
            total_out += o
            by_agent[role] = {"input": i, "output": o}
    return {"input_tokens": total_in, "output_tokens": total_out, "by_agent": by_agent}


def all_agent_totals() -> dict[str, dict[str, int]]:
    """Lifetime token totals per agent across all tasks (for /metrics).

    Returns:
        Dict mapping each agent role to ``{"input": int, "output": int}``,
        summed over every task currently held in ``_usage``.

    Side effects:
        Reads ``_usage`` under ``_lock``.

    Note:
        Totals are only as complete as the in-memory state; ``reset_task`` and
        process restarts discard history.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
    with _lock:
        for (_tid, role), (i, o) in _usage.items():
            out[role]["input"] += i
            out[role]["output"] += o
    return dict(out)


def reset_task(task_id: str) -> None:
    """Forget all accumulated usage for a single task.

    Args:
        task_id: The run identifier whose buckets should be removed.

    Returns:
        None.

    Side effects:
        Deletes every ``_usage`` entry whose key's first element matches
        ``task_id``, under ``_lock``.
    """
    with _lock:
        for key in [k for k in _usage if k[0] == task_id]:
            del _usage[key]


def _agent_caps(agent_role: str) -> tuple[int | None, int | None]:
    """Look up an agent record's token thresholds. Non-record roles (e.g. the
    'cheap' sub-call alias) have no record → no caps.

    Args:
        agent_role: The agent role to resolve to a registry record.

    Returns:
        ``(max_input_tokens, max_output_tokens)``; either element (or both) is
        ``None`` when no cap applies or the role has no registry record.

    Note:
        ``load_agent`` is imported lazily here to avoid a module-load import cycle
        and any failure (missing record, registry error) is swallowed and treated
        as "no caps".
    """
    try:
        from hyperion.agents.registry import load_agent

        th = load_agent(agent_role).thresholds
        return th.max_input_tokens, th.max_output_tokens
    except Exception:
        return None, None


def check_agent_cap(task_id: str, agent_role: str) -> None:
    """Raise CapExceeded if the agent has blown its per-record token threshold.

    Called from the pre-API-call hook so enforcement happens *before* the next
    LLM request fires.

    Args:
        task_id: The run identifier; no-op if empty.
        agent_role: The agent role to check; no-op if empty.

    Returns:
        None when the agent is within its caps (or has none).

    Raises:
        CapExceeded: If accumulated input or output tokens have reached the
            agent record's threshold.
    """
    if not task_id or not agent_role:
        return
    cap_in, cap_out = _agent_caps(agent_role)
    if cap_in is None and cap_out is None:
        return
    used_in, used_out = agent_totals(task_id, agent_role)
    if cap_in is not None and used_in >= cap_in:
        _raise(agent_role, "input", used_in, cap_in)
    if cap_out is not None and used_out >= cap_out:
        _raise(agent_role, "output", used_out, cap_out)


def _raise(agent_role: str, kind: str, used: int, cap: int) -> None:
    """Construct and raise a descriptive ``CapExceeded`` exception.

    Args:
        agent_role: The offending agent role (for the message).
        kind: Either ``"input"`` or ``"output"`` (which cap was hit).
        used: Tokens consumed so far.
        cap: The threshold that was reached.

    Raises:
        CapExceeded: Always.
    """
    # Lazy import keeps this module free of a runner import cycle.
    from hyperion.crews.runner import CapExceeded

    raise CapExceeded(
        f"Agent '{agent_role}' exceeded its {kind}-token cap "
        f"({used} >= {cap}). Aborting run."
    )


# ---------------------------------------------------------------------------
# LiteLLM callback metadata extraction
# ---------------------------------------------------------------------------


def _dig_metadata(kwargs: dict) -> dict:
    """Find the metadata block we set via extra_body, wherever LiteLLM stashed it.

    LiteLLM relocates caller-supplied metadata to different kwarg paths depending
    on provider and version, so several known locations are probed in order.

    Args:
        kwargs: The raw kwargs dict passed to a LiteLLM callback hook.

    Returns:
        The first candidate dict that looks like ours (contains ``session_id`` or
        ``tags``), or an empty dict if none match.
    """
    # Probe known LiteLLM stash locations, most-specific first.
    candidates = [
        kwargs.get("litellm_params", {}).get("metadata"),
        kwargs.get("metadata"),
        kwargs.get("optional_params", {}).get("extra_body", {}).get("metadata"),
        kwargs.get("optional_params", {}).get("metadata"),
    ]
    for c in candidates:
        if isinstance(c, dict) and ("session_id" in c or "tags" in c):
            return c
    return {}


def _attribution(kwargs: dict) -> tuple[str, str]:
    """Derive ``(task_id, agent_role)`` from a LiteLLM callback's metadata block.

    Args:
        kwargs: The raw kwargs dict passed to a LiteLLM callback hook.

    Returns:
        ``(task_id, agent_role)``. ``task_id`` is the metadata ``session_id``
        (empty string if absent); ``agent_role`` is the first tag that is not
        the literal ``"hyperion"``, defaulting to ``"unknown"``.
    """
    meta = _dig_metadata(kwargs)
    task_id = str(meta.get("session_id") or "")
    role = "unknown"
    tags = meta.get("tags") or []
    # tags is ["hyperion", agent_role]; take the first non-"hyperion" tag.
    for t in tags:
        if t and t != "hyperion":
            role = str(t)
            break
    return task_id, role


def _prompt_preview(kwargs: dict) -> str:
    """Extract a plain-text preview of the last user/prompt message.

    Args:
        kwargs: The raw kwargs dict passed to a LiteLLM callback hook; expected
            to contain a ``messages`` list.

    Returns:
        The text content of the final message as a string; empty string if there
        are no messages. Multi-part content blocks are flattened into a single
        space-joined string.
    """
    messages = kwargs.get("messages") or []
    if not messages:
        return ""
    content = messages[-1].get("content") or ""
    if isinstance(content, list):  # multi-part content blocks
        content = " ".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    return str(content)


def _write_trace_event(
    kwargs: dict, response_obj: Any, start_time: Any, end_time: Any,
    task_id: str, role: str, in_tok: int, out_tok: int,
) -> None:
    """Persist one LLM call as a trace_events row. Must never raise into the
    main path — tracing is best-effort only.

    Args:
        kwargs: The raw LiteLLM callback kwargs (used for metadata, model,
            messages).
        response_obj: The completion response object (for content, tool calls,
            cost).
        start_time: Call start timestamp (datetime-like); used for ``started_at``
            and duration.
        end_time: Call end timestamp (datetime-like); used for duration.
        task_id: Run identifier to attribute the row to.
        role: Agent role to attribute the row to.
        in_tok: Input/prompt token count for the call.
        out_tok: Output/completion token count for the call.

    Returns:
        None.

    Side effects:
        Inserts one row into the ``trace_events`` table of the SQLite state DB at
        ``settings.tasks_dir/state.db`` and commits. All exceptions are swallowed
        so tracing failures never propagate into the LLM call path.
    """
    try:
        from hyperion.config import settings

        meta = _dig_metadata(kwargs)
        tags = meta.get("tags") or []
        # Workflow node this call ran under (None for meta-prompt / non-workflow
        # calls). Lets the trace UI attribute calls to the exact node.
        node_id = meta.get("node_id")
        # Classify the call so the UI can separate internal meta-prompts from
        # the user-facing conversation turns.
        prompt_type = "meta-prompt" if "meta-prompt" in tags else "user-facing"

        prompt_preview = _prompt_preview(kwargs)

        # Response content / tool calls are best-effort; shape varies by provider.
        response_preview = ""
        tools_used = "[]"
        try:
            choice = response_obj.choices[0]
            response_preview = choice.message.content or ""
            tc = getattr(choice.message, "tool_calls", None) or []
            tools_used = json.dumps([t.function.name for t in tc])
        except Exception:
            pass

        try:
            import litellm

            cost = litellm.completion_cost(completion_response=response_obj) or 0.0
        except Exception:
            cost = 0.0

        try:
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
        except Exception:
            duration_ms = None
        try:
            started_at = start_time.isoformat()
        except Exception:
            started_at = None

        db_path = str(settings.tasks_dir / "state.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """INSERT INTO trace_events
                   (task_id, agent_role, node_id, prompt_type, model, input_tokens, output_tokens,
                    cost_usd, prompt_preview, response_preview, tools_used,
                    started_at, duration_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (task_id, role, node_id, prompt_type, kwargs.get("model", ""),
                 in_tok, out_tok, cost,
                 prompt_preview, response_preview, tools_used,
                 started_at, duration_ms),
            )
            conn.commit()
    except Exception:
        pass


def _usage_from_response(response_obj: Any) -> tuple[int, int]:
    """Pull ``(prompt_tokens, completion_tokens)`` from a completion response.

    Handles both object-style responses (``response_obj.usage.prompt_tokens``)
    and dict-style responses (``response_obj["usage"]["prompt_tokens"]``).

    Args:
        response_obj: The LiteLLM/OpenAI completion response (object or dict).

    Returns:
        ``(input_tokens, output_tokens)``; ``(0, 0)`` if no usage block is found.
    """
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    return int(get("prompt_tokens", 0) or 0), int(get("completion_tokens", 0) or 0)


class HyperionUsageLogger(CustomLogger):
    """Accumulate token usage and enforce per-agent caps via LiteLLM callbacks."""

    def log_pre_api_call(self, model, messages, kwargs):  # noqa: D401 (sync hook)
        """LiteLLM hook fired before each request — enforces token caps.

        Args:
            model: Model name (unused; required by the hook signature).
            messages: Outgoing messages (unused; required by the signature).
            kwargs: Raw LiteLLM kwargs carrying our metadata block.

        Returns:
            None.

        Raises:
            CapExceeded: Re-raised so the runner can abort the run when an agent
                has exceeded its cap. Any other exception is swallowed so a
                metadata/attribution glitch never blocks a legitimate call.
        """
        try:
            task_id, role = _attribution(kwargs)
            check_agent_cap(task_id, role)
        except Exception as exc:  # CapExceeded must propagate; others must not.
            from hyperion.crews.runner import CapExceeded

            if isinstance(exc, CapExceeded):
                raise

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """LiteLLM hook fired after a successful request — records usage + trace.

        Args:
            kwargs: Raw LiteLLM kwargs carrying our metadata block.
            response_obj: The completion response (for token usage and trace).
            start_time: Call start timestamp.
            end_time: Call end timestamp.

        Returns:
            None.

        Side effects:
            Updates in-memory token accounting via ``record`` and, when the call
            is attributable to a task, writes a ``trace_events`` row. All
            exceptions are swallowed — accounting must never break the call path.
        """
        try:
            task_id, role = _attribution(kwargs)
            in_tok, out_tok = _usage_from_response(response_obj)
            record(task_id, role, in_tok, out_tok)
            if task_id:
                _write_trace_event(
                    kwargs, response_obj, start_time, end_time,
                    task_id, role, in_tok, out_tok,
                )
        except Exception:
            pass

    async def async_log_pre_api_call(self, model, messages, kwargs):
        """Async variant of ``log_pre_api_call``; delegates to the sync impl.

        Required because LiteLLM dispatches to the async hook for async
        completions. See ``log_pre_api_call`` for behavior and raised errors.
        """
        self.log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async variant of ``log_success_event``; delegates to the sync impl.

        Required because LiteLLM dispatches to the async hook for async
        completions. See ``log_success_event`` for behavior and side effects.
        """
        self.log_success_event(kwargs, response_obj, start_time, end_time)


_REGISTERED = False
_LOGGER: HyperionUsageLogger | None = None


def get_logger() -> HyperionUsageLogger:
    """Return the process-wide usage logger singleton.

    Installed on LiteLLM's global callback list by :func:`register`; exposed as an
    accessor for callers that need the instance directly."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = HyperionUsageLogger()
    return _LOGGER


def register() -> None:
    """Install the usage logger on LiteLLM's callback list exactly once.

    Covers every ``litellm.completion`` call in the process — the owned agent loop
    and direct tool calls alike — so token accounting and cap enforcement always
    apply. Called once at startup (see ``server/api.py``)."""
    global _REGISTERED
    if _REGISTERED:
        return
    import litellm

    logger = get_logger()
    existing = list(getattr(litellm, "callbacks", []) or [])
    if logger not in existing:
        litellm.callbacks = existing + [logger]
    _REGISTERED = True
