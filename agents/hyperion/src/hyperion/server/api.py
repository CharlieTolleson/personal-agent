"""
FastAPI service for Hyperion — the HTTP control plane for the multi-agent orchestrator.

This module is the single public surface (mounted at http://localhost:4100 in the
docker-compose stack) for everything the Hyperion UI (nginx :4102) and external
callers (n8n, other agents) do: submitting tasks, polling/streaming progress,
approving plans, sending feedback, CRUD on agents/workflows, live model/threshold
reconfiguration, metrics, and config export/import.

Architecture / key design decisions:
  - State is a single SQLite file (``state.db`` under ``settings.tasks_dir``) accessed
    via ``aiosqlite``. The schema is created lazily on first connect and migrated
    in-place (``_migrate``) by ALTER-ing in nullable columns, so an existing DB from
    an earlier phase keeps working without a destructive migration.
  - Task execution runs in fire-and-forget ``asyncio.create_task`` background
    coroutines (``_run_and_update`` / ``_resume_and_update``). The HTTP handlers
    return ``202`` / a ``running`` status immediately; clients poll ``GET
    /tasks/{id}`` or subscribe to the SSE stream.
  - Progress lines live in the in-memory ``_PROGRESS`` dict AND are appended to a
    per-task ``progress.log`` on disk, so a server restart can rehydrate streams and
    a paused (human-in-the-loop) task survives the restart and can still be resumed.
  - Human-in-the-loop (HITL): a task can pause at ``awaiting_approval`` (plan gate)
    or ``awaiting_input`` (an ``ask_user`` question). All pause/resume state needed
    to continue is persisted to the DB, never only in memory.
  - Live reconfiguration: ``PUT /config`` (role→model) and ``PUT /thresholds``
    (token/wall caps) mutate the in-process ``settings`` object AND persist to JSON
    files under ``settings.config_dir`` (``models.json`` / ``thresholds.json``),
    which are re-applied on the next boot (``_apply_*_overrides``).
  - Per the workspace convention, all LLM traffic flows through the LiteLLM proxy;
    this service never calls provider APIs directly.

Endpoints (non-exhaustive — see the route decorators below):
  GET  /config                    → current model assignments + provider key status
  POST /tasks                     → submit a task
  GET  /tasks/{task_id}           → status + links
  GET  /tasks/{task_id}/stream    → SSE progress stream
  GET  /tasks/{task_id}/artifacts/{name} → static file
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Optional

import aiosqlite
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from hyperion.agents.registry import (
    TOOL_REGISTRY,
    AgentRecord,
    delete_agent,
    load_agent,
    load_all_agents,
    save_agent,
    validate_agent,
)
from hyperion import models_registry, scheduler, usage
from hyperion.config import settings
from hyperion.crews.plan_contract import parse_plan
from hyperion.crews.runner import resume_task, run_task
from hyperion.crews.workflows import WorkflowRecord
from hyperion.memory.episodic import store_episode
from hyperion.server.affordances import Affordance, AffordanceOption
from hyperion.server.webhooks import UnsafeCallbackURL, fire_callback, validate_callback_url
from hyperion.tools.second_brain import SecondBrainTool
from hyperion.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """ASGI lifespan: run startup state-init, then cleanly stop the scheduler.

    Replaces the deprecated ``@app.on_event("startup"/"shutdown")`` hooks with the
    modern single context manager. The detailed logic lives in ``_startup`` /
    ``_shutdown`` (defined later in the module) so the ordering rationale stays
    documented next to the code it governs; both are resolved at call time.
    """
    await _startup()
    try:
        yield
    finally:
        await _shutdown()


app = FastAPI(title="Hyperion", version="0.1.0", lifespan=lifespan)

# The Phase 7 web UI is served from a different origin (nginx :4102) and calls this
# API directly from the browser. Allow the local UI origins only — this is a
# localhost developer tool, not a public service.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROGRESS: dict[str, list[str]] = {}  # task_id → list of progress lines

# task_id → the in-flight background asyncio.Task running it. Populated at every
# spawn site (via ``_spawn_tracked``) so ``POST /tasks/{id}/stop`` has a handle to
# cancel. NOTE: cancelling the asyncio task does NOT kill a blocking LLM call
# already running in an executor thread (a running thread can't be cancelled) — so
# the stop endpoint also writes status=cancelled to the DB directly for an
# immediate, responsive UI; the orphaned thread, if any, drains in the background.
_RUNNING: dict[str, "asyncio.Task[Any]"] = {}

# Statuses that mean a run has finished and will not change further.
_TERMINAL: tuple[str, ...] = ("done", "failed", "cancelled")


def _spawn_tracked(task_id: str, coro: Any) -> None:
    """Spawn a background run coroutine and register it for cancellation.

    Stores the created ``asyncio.Task`` in ``_RUNNING`` keyed by ``task_id`` and
    attaches a done-callback that removes it once the run finishes (or is
    cancelled), so the registry never leaks completed tasks.

    Args:
        task_id: The task id this coroutine runs (the registry key).
        coro: The ``_run_and_update`` / ``_resume_and_update`` coroutine to run.

    Returns:
        None.
    """
    t = asyncio.create_task(coro)
    _RUNNING[task_id] = t
    t.add_done_callback(lambda _t: _RUNNING.pop(task_id, None))


def _db_path() -> Path:
    """Resolve the SQLite state DB path from the *current* ``settings.tasks_dir``.

    Computed per call rather than captured at import so that redirecting
    ``settings.tasks_dir`` (e.g. tests patching it to a tmp dir) also moves the
    state DB, keeping runs hermetic instead of writing to the real task store.
    """
    return settings.tasks_dir / "state.db"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


# All columns added in Phase 0 (the implementation plan §4.4); later phases only populate them.
_NEW_COLUMNS = {
    "hitl": "TEXT",
    "pending_stage": "TEXT",
    "pending_payload": "TEXT",
    "resume_token": "TEXT",
    "routing": "TEXT",
    "callback_url": "TEXT",  # (Phase 9) outbound completion webhook
    "workflow": "TEXT",      # workflow id this task runs (null → server default)
}


async def _migrate(db: aiosqlite.Connection) -> None:
    """Bring an existing ``tasks`` table up to the current schema and ensure
    ``trace_events`` exists.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op once the table exists, so newer
    columns must be added explicitly. This inspects the live column set via
    ``PRAGMA table_info`` and ALTER-adds any missing column from ``_NEW_COLUMNS``
    (all nullable, so the migration is non-destructive and order-independent).

    Args:
        db: An open aiosqlite connection. The connection is committed before return.

    Returns:
        None.

    Side effects:
        Issues ALTER TABLE / CREATE TABLE statements and commits the transaction.
    """
    async with db.execute("PRAGMA table_info(tasks)") as cur:
        existing = {row[1] for row in await cur.fetchall()}
    for name, sqltype in _NEW_COLUMNS.items():
        if name not in existing:
            await db.execute(f"ALTER TABLE tasks ADD COLUMN {name} {sqltype}")
    await db.execute(
        """CREATE TABLE IF NOT EXISTS trace_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id          TEXT    NOT NULL,
            agent_role       TEXT    NOT NULL,
            node_id          TEXT,
            prompt_type      TEXT    NOT NULL DEFAULT 'user-facing',
            model            TEXT,
            input_tokens     INTEGER DEFAULT 0,
            output_tokens    INTEGER DEFAULT 0,
            cost_usd         REAL    DEFAULT 0.0,
            prompt_preview   TEXT,
            response_preview TEXT,
            tools_used       TEXT,
            started_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
            duration_ms      INTEGER
        )"""
    )
    # Add node_id to a pre-existing trace_events table (nullable → non-destructive).
    async with db.execute("PRAGMA table_info(trace_events)") as cur:
        trace_cols = {row[1] for row in await cur.fetchall()}
    if "node_id" not in trace_cols:
        await db.execute("ALTER TABLE trace_events ADD COLUMN node_id TEXT")
    await db.commit()


async def _get_db() -> aiosqlite.Connection:
    """Open the tasks DB, creating + migrating the schema on the way out.

    Ensures the ``tasks`` table exists (with the full current column set) and then
    runs ``_migrate`` so an older on-disk DB gains any newly-added columns.

    Returns:
        An open aiosqlite connection. The caller owns it and must ``await db.close()``.

    Side effects:
        Creates/migrates the SQLite schema and commits.
    """
    db = await aiosqlite.connect(_db_path())
    await db.execute(
        """CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status  TEXT NOT NULL DEFAULT 'queued',
            request TEXT,
            error   TEXT,
            result_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            hitl TEXT,
            pending_stage TEXT,
            pending_payload TEXT,
            resume_token TEXT,
            routing TEXT,
            callback_url TEXT,
            workflow TEXT
        )"""
    )
    await db.commit()
    await _migrate(db)
    return db


@asynccontextmanager
async def _db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Schema-ensured connection as an async context manager.

    Wraps ``_get_db`` (which creates/migrates the schema) so read-path endpoints
    never assume a prior write or the startup hook ran — a fresh DB is fully
    usable on first access. Closes the connection on exit.
    """
    db = await _get_db()
    try:
        yield db
    finally:
        await db.close()


async def _update_task(task_id: str, **kwargs: Any) -> None:
    """Patch arbitrary columns of one task row and bump ``updated_at``.

    Builds the SET clause from the kwargs keys (column names are trusted callers'
    constants, never user input) and binds the values positionally.

    Args:
        task_id: The task to update (WHERE task_id=?).
        **kwargs: column=value pairs to write. ``updated_at`` is always set to
            CURRENT_TIMESTAMP in addition.

    Returns:
        None.

    Side effects:
        Opens its own short-lived connection, executes the UPDATE, and commits.
    """
    sets = ", ".join(f"{k}=?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    async with _db() as db:
        await db.execute(
            f"UPDATE tasks SET {sets}, updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
            values,
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Progress persistence — survives an API restart so a paused task can be polled
# and resumed even after the in-memory _PROGRESS dict is gone.
# ---------------------------------------------------------------------------


def _progress_log_path(task_id: str) -> Path:
    """Return the on-disk progress log path for a task.

    Args:
        task_id: The task whose log path to compute.

    Returns:
        ``{tasks_dir}/{task_id}/progress.log`` (not guaranteed to exist).
    """
    return settings.tasks_dir / task_id / "progress.log"


def _append_progress(task_id: str, line: str) -> None:
    """Record one progress line both in memory and on disk.

    Appends to the in-memory ``_PROGRESS`` list (read by the SSE stream and status
    endpoint) and best-effort appends to the per-task ``progress.log`` so the line
    survives a restart.

    Args:
        task_id: The task this progress line belongs to.
        line: The text to record (no trailing newline needed).

    Returns:
        None.

    Side effects:
        Mutates ``_PROGRESS`` and writes to disk. Disk errors are swallowed —
        progress logging must never break a running task.
    """
    _PROGRESS.setdefault(task_id, []).append(line)
    try:
        path = _progress_log_path(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # progress logging must never break a task
        pass


def _rehydrate_progress() -> None:
    """Reload progress lines from disk so a restarted API can keep streaming.

    Called once on startup. Scans every task directory under ``settings.tasks_dir``
    and repopulates ``_PROGRESS`` from each ``progress.log``.

    Returns:
        None.

    Side effects:
        Repopulates the module-level ``_PROGRESS`` dict. Per-file read errors are
        ignored so one corrupt log can't block startup.
    """
    if not settings.tasks_dir.exists():
        return
    for task_dir in settings.tasks_dir.iterdir():
        log = task_dir / "progress.log"
        if log.exists():
            try:
                _PROGRESS[task_dir.name] = log.read_text(encoding="utf-8").splitlines()
            except Exception:
                pass


def _plan_affordance(task_id: str) -> dict:
    """Build the choice affordance shown while a task awaits plan approval.

    Parses the task's persisted plan and turns each plan option into a selectable
    choice. If the plan exposed no options, falls back to a single "Proceed with
    the plan" default so the UI always has something to render.

    Args:
        task_id: The paused task whose plan options to surface.

    Returns:
        A serialized ``Affordance`` dict (type="choice") for the API response.
    """
    fm = parse_plan(task_id)
    options = [
        AffordanceOption(id=o.id, label=o.summary or o.id, description=o.summary or "")
        for o in fm.options
    ]
    if not options:
        options = [AffordanceOption(id="default", label="Proceed with the plan", description="")]
    return Affordance(
        type="choice",
        prompt="Review the plan and choose how to proceed.",
        options=options,
        agent_id="planner",
        stage="plan",
    ).model_dump()


def _question_affordance(task_id: str) -> Optional[dict]:
    """Build the question affordance for a task paused on an ``ask_user`` request.

    Looks up the most recent pending affordance recorded by the feedback subsystem
    and renders it as a free-text question for the UI.

    Args:
        task_id: The paused task awaiting a user answer.

    Returns:
        A serialized ``Affordance`` dict (type="question"), or ``None`` if no
        pending question exists for this task.
    """
    from hyperion.feedback import latest_pending_affordance

    pending = latest_pending_affordance(task_id)
    if pending is None:
        return None
    return Affordance(
        type="question",
        prompt=pending.get("prompt", "The agent needs more information."),
        agent_id=pending.get("agent_id") or "planner",
        stage=pending.get("stage") or "plan",
    ).model_dump()


def _pending_affordance(task_id: str, status: str) -> Optional[dict]:
    """Surface the right affordance for whichever pending state a task is in.

    Dispatches on status: ``awaiting_approval`` → plan-choice affordance,
    ``awaiting_input`` → question affordance, anything else → no affordance.

    Args:
        task_id: The task being inspected.
        status: The task's current status string.

    Returns:
        A serialized affordance dict, or ``None`` when the task is not paused for
        human input.
    """
    if status == "awaiting_approval":
        return _plan_affordance(task_id)
    if status == "awaiting_input":
        return _question_affordance(task_id)
    return None


async def _persist_result(task_id: str, result: dict) -> None:
    """Write a runner/resume result back to the tasks table.

    Branches on the result status:
      - paused states (``awaiting_approval`` / ``awaiting_input``): stores the
        pending stage + payload and mints a short ``resume_token`` so the task can
        be continued later (even across a restart).
      - terminal states: stores error/result_path and the JSON-encoded routing,
        and clears the pending stage/payload fields.

    Args:
        task_id: The task to update.
        result: The dict returned by ``run_task`` / ``resume_task`` (must contain
            a ``status`` key; other keys are read defensively).

    Returns:
        None.

    Side effects:
        Persists task state via ``_update_task``.
    """
    status = result["status"]
    if status in ("awaiting_approval", "awaiting_input"):
        await _update_task(
            task_id,
            status=status,
            pending_stage=result.get("pending_stage"),
            pending_payload=json.dumps(result.get("pending_payload") or {}),
            resume_token=str(uuid.uuid4())[:8],
        )
    else:
        routing = result.get("routing")
        await _update_task(
            task_id,
            status=status,
            error=result.get("error"),
            result_path=result.get("result_path"),
            routing=json.dumps(routing) if routing is not None else None,
            pending_stage=None,
            pending_payload=None,
        )


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


_MODEL_FIELDS = ("model_planner", "model_worker", "model_cheap")


def _model_override_path() -> Path:
    """Path of the persisted ``PUT /config`` model-override file.

    Returns:
        ``{config_dir}/models.json``. Computed at call time (not cached) so a
        patched ``config_dir`` in tests/Docker is always respected.
    """
    # Computed at call time so a patched config_dir (tests/Docker) is respected.
    return settings.config_dir / "models.json"


def _apply_model_overrides() -> None:
    """Re-apply persisted PUT /config model choices to in-memory settings on boot.

    Reads ``models.json`` (if present) and copies any role-model fields plus
    ``default_workflow`` onto the live ``settings`` object. A missing or malformed
    file is silently ignored (settings keep their env/default values).

    Returns:
        None.

    Side effects:
        Mutates the global ``settings`` object.
    """
    path = _model_override_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError):
        return
    for field in _MODEL_FIELDS:
        if data.get(field):
            setattr(settings, field, data[field])
    if data.get("default_workflow"):
        settings.default_workflow = data["default_workflow"]


# Global cap fields (token/wall/loop budgets) editable via PUT /thresholds.
_CAP_FIELDS = ("cap_input_tokens", "cap_output_tokens", "cap_tool_loop", "cap_wall_seconds")


def _thresholds_path() -> Path:
    """Path of the persisted ``PUT /thresholds`` global-caps file.

    Returns:
        ``{config_dir}/thresholds.json`` (computed at call time, like the model
        override path, so a patched ``config_dir`` is respected).
    """
    return settings.config_dir / "thresholds.json"


def _apply_threshold_overrides() -> None:
    """Re-apply persisted PUT /thresholds global caps to settings on boot.

    Reads ``thresholds.json`` (if present) and copies any of the global cap fields
    onto the live ``settings`` object. Only integer values are applied; a missing
    or malformed file is ignored.

    Returns:
        None.

    Side effects:
        Mutates the global ``settings`` object.
    """
    path = _thresholds_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError):
        return
    for field in _CAP_FIELDS:
        if isinstance(data.get(field), int):
            setattr(settings, field, data[field])


async def _reconcile_orphaned_runs() -> None:
    """Mark runs left ``running``/``queued`` by a restart as ``failed``.

    A fire-and-forget background coroutine does not survive an API restart, so any
    row still in an active (non-paused, non-terminal) state on boot is orphaned and
    can never finish. Flip those to ``failed`` with a clear error so they stop
    polling as live. ``awaiting_approval``/``awaiting_input`` are left untouched —
    those are genuine HITL pauses that ``resume_task`` can pick back up.

    Returns:
        None.
    """
    async with _db() as db:
        await db.execute(
            "UPDATE tasks SET status='failed', "
            "error='interrupted by server restart', updated_at=CURRENT_TIMESTAMP "
            "WHERE status IN ('running','queued')",
        )
        await db.commit()


async def _startup() -> None:
    """Initialize all runtime state (invoked by the ``lifespan`` context manager).

    Ordering matters: the tasks dir and DB schema must exist before anything else,
    progress is rehydrated so paused tasks can stream, persisted config/threshold
    overrides are re-applied to ``settings``, usage accounting is registered, and
    finally the background scheduler loop is launched (and stashed on ``app.state``
    so ``_shutdown`` can stop it).

    Returns:
        None.

    Side effects:
        Creates directories/DB, mutates ``settings``, registers usage hooks, and
        spawns the scheduler asyncio task.
    """
    settings.tasks_dir.mkdir(parents=True, exist_ok=True)
    # Ensure DB schema exists
    db = await _get_db()
    await db.close()
    # Finalize runs orphaned by this restart. A ``running``/``queued`` row has no
    # surviving background coroutine after a restart, so it would otherwise poll as
    # "running" forever; mark it failed. ``awaiting_*`` rows are deliberately left
    # alone — they are legitimately paused and resumable.
    await _reconcile_orphaned_runs()
    # Reload progress for tasks that were running/paused before this restart.
    _rehydrate_progress()
    # Re-apply any persisted model assignments (PUT /config) and caps (PUT /thresholds).
    _apply_model_overrides()
    # Seed the role/alias registry from settings on first run (lossless migration of
    # legacy env / models.json role choices), then make the registry authoritative for
    # the built-in role -> model fields used by the LLM factory functions.
    models_registry.seed_from_settings_if_missing()
    models_registry.apply_roles_to_settings()
    _apply_threshold_overrides()
    # Install per-agent token-cap enforcement + usage accounting (Phase 8).
    usage.register()
    # Start the schedule-trigger loop (Phase 8). Cancelled on shutdown.
    app.state.scheduler_stop = asyncio.Event()
    app.state.scheduler_task = asyncio.create_task(
        scheduler.run_scheduler(_enqueue_scheduled, stop_event=app.state.scheduler_stop)
    )


async def _shutdown() -> None:
    """Stop the background scheduler cleanly (invoked by the ``lifespan`` manager).

    Signals the scheduler's stop event and waits up to 5 seconds for the loop to
    exit; if it doesn't finish in time (or is already cancelled), the task is
    force-cancelled.

    Returns:
        None.

    Side effects:
        Sets the stop event and may cancel the scheduler asyncio task.
    """
    stop = getattr(app.state, "scheduler_stop", None)
    task = getattr(app.state, "scheduler_task", None)
    if stop is not None:
        stop.set()
    if task is not None:
        try:
            await asyncio.wait_for(task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()


async def _enqueue_scheduled(record: AgentRecord) -> None:
    """Scheduler callback: start a normal pipeline task originated by a
    schedule-trigger agent. Mirrors POST /tasks so scheduled runs are first-class.

    Mints a task id, derives the request text from the agent record, inserts a
    ``queued`` row (with HITL forced off — scheduled runs are unattended), and
    spawns the background runner.

    Args:
        record: The schedule-trigger agent whose configured task to enqueue.

    Returns:
        None.

    Side effects:
        Inserts a tasks row and starts a background ``_run_and_update`` coroutine.
    """
    task_id = str(uuid.uuid4())[:8]
    request = scheduler.scheduled_task_request(record)
    async with _db() as db:
        await db.execute(
            "INSERT INTO tasks (task_id, status, request, hitl) VALUES (?,?,?,?)",
            (task_id, "queued", request, "off"),
        )
        await db.commit()
    _spawn_tracked(
        task_id, _run_and_update(task_id, request, False, None, None, None, "off")
    )


# ---------------------------------------------------------------------------
# Config endpoint
# ---------------------------------------------------------------------------


@app.get("/config")
async def get_config() -> dict:
    """
    Show current model assignments and which provider API keys are configured.

    Model aliases (smart / worker / cheap) are multi-provider groups in LiteLLM —
    the proxy automatically picks from whichever providers have valid keys.
    Override any assignment by setting MODEL_PLANNER / MODEL_WORKER / MODEL_CHEAP
    in agents/hyperion/.env and restarting.

    Returns:
        A dict with current model aliases (+ env var names + notes), per-provider
        key presence, the hard-coded alias fallback order, global caps, the default
        workflow, and the LiteLLM base URL.
    """
    providers = settings.provider_keys_present()
    # Roles are now operator-editable (models_registry); the three built-ins still map to
    # the MODEL_* env vars for back-compat, user-added roles report env_var=None.
    _env_by_role = {"planner": "MODEL_PLANNER", "worker": "MODEL_WORKER", "cheap": "MODEL_CHEAP"}
    return {
        "models": {
            r["name"]: {
                "alias": r["model"],
                "note": r.get("note", ""),
                "env_var": _env_by_role.get(r["name"]),
            }
            for r in models_registry.roles()
        },
        "roles": models_registry.roles(),
        "aliases": models_registry.aliases(),
        "providers": {
            name: {"key_present": present, "status": "available" if present else "no key — alias will skip"}
            for name, present in providers.items()
        },
        "alias_fallback_order": models_registry.alias_details(),
        "caps": {
            "input_tokens": settings.cap_input_tokens,
            "output_tokens": settings.cap_output_tokens,
            "tool_loop": settings.cap_tool_loop,
            "wall_seconds": settings.cap_wall_seconds,
        },
        "default_workflow": settings.default_workflow,
        "litellm_url": settings.litellm_base_url,
    }


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TaskRequest(BaseModel):
    """Input body for ``POST /tasks``.

    Fields:
        task: The natural-language task to run (required).
        schema_version: External-caller contract version; the server only speaks 1.
        hitl: Human-in-the-loop mode — "off", "plan" (pause for plan approval), or
            "full".
        workflow: DAG id to run; ``None`` falls back to the server default workflow.
        workflow_prompt: Plain-language description of how the agents should work
            together (e.g. "research, then have the critic review, then synthesize").
            When set (and ``workflow`` is not), it is compiled into an ad-hoc
            workflow DAG, persisted, and run. Ignored if ``workflow`` is given.
        callback_url: Optional outbound webhook POSTed once on terminal status.
        cap_wall_seconds / cap_input_tokens / cap_output_tokens: Per-run budget
            overrides; ``None`` uses the global caps.
    """
    task: str
    schema_version: int = 1                              # (Phase 9) external-caller contract
    hitl: Literal["off", "plan", "full"] = "off"         # (filled: Phase 3)
    workflow: Optional[str] = None                       # DAG id; null → server default
    workflow_prompt: Optional[str] = None                # NL → compiled ad-hoc DAG (req 4.1)
    callback_url: Optional[str] = None                   # (filled: Phase 9)
    cap_wall_seconds: Optional[int] = None
    cap_input_tokens: Optional[int] = None
    cap_output_tokens: Optional[int] = None


class TaskResponse(BaseModel):
    """Output body for the task endpoints (submit / status / approve / feedback).

    Fields:
        task_id: The 8-char task identifier.
        status: queued|running|awaiting_approval|awaiting_input|done|failed.
        error: Error message on a failed task, else ``None``.
        result_path: Path to the primary result artifact when available.
        progress_lines: Accumulated progress log lines from ``_PROGRESS``.
        routing: Persisted routing decision (which agents ran), populated post-route.
        pending_stage: For paused tasks, the stage the task is paused at.
        pending_affordance: For paused tasks, the UI affordance to render.
    """
    task_id: str
    status: str                                          # queued|running|awaiting_approval|awaiting_input|done|failed
    error: Optional[str] = None
    result_path: Optional[str] = None
    progress_lines: list[str] = []
    routing: Optional[dict] = None                       # (filled: Phase 2)
    pending_stage: Optional[str] = None                  # (filled: Phase 3)
    pending_affordance: Optional[dict] = None            # (filled: Phase 3)


class ApproveRequest(BaseModel):
    """Input body for ``POST /tasks/{id}/approve`` (the plan gate).

    Fields:
        action: "approve" to run, "revise" to send edits back to the planner, or
            "reject" to abort.
        chosen_option: Which plan option to run (for multi-option plans).
        edits: Free-text revision feedback handed to the planner on "revise".
    """
    action: Literal["approve", "revise", "reject"] = "approve"
    chosen_option: Optional[str] = None                  # which plan option to run
    edits: Optional[str] = None                          # revision feedback for the planner


class FeedbackRequest(BaseModel):
    """Input body for ``POST /tasks/{id}/feedback``.

    Fields:
        message: Free-text human feedback — either queued for a running task or
            used to answer an ``ask_user`` question on a paused task.
    """
    message: str


class ChatRequest(BaseModel):
    """Input body for ``POST /tasks/{id}/chat`` (post-run follow-up conversation).

    Fields:
        message: The user's follow-up message about a completed run's result.
    """
    message: str


class ChatResponse(BaseModel):
    """Output body for ``POST /tasks/{id}/chat``.

    Fields:
        task_id: The run being discussed.
        reply: The assistant's grounded answer for this turn.
    """
    task_id: str
    reply: str


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------


def _maybe_store_episode(task_id: str, request: str, result: dict, started: float) -> None:
    """Persist a one-line episode summary so future tasks can recall similar work.
    Best-effort — failure to store should never affect the task result.

    Reads up to the first 2000 chars of the result artifact as the summary (falling
    back to the error or a placeholder) and records it in episodic memory along with
    success and elapsed duration.

    Args:
        task_id: The finished task.
        request: The original task request text.
        result: The terminal result dict (status + result_path/error).
        started: Event-loop timestamp captured when the task began (for duration).

    Returns:
        None.

    Side effects:
        Writes to episodic memory; any exception is logged and swallowed.
    """
    try:
        summary = ""
        result_path = result.get("result_path")
        if result_path and Path(result_path).exists():
            summary = Path(result_path).read_text(encoding="utf-8")[:2000]
        store_episode(
            task_id=task_id,
            original_request=request,
            final_summary=summary or (result.get("error") or "(no output)"),
            success=(result["status"] == "done"),
            duration_seconds=asyncio.get_event_loop().time() - started,
        )
    except Exception as exc:
        logger.warning("Episode store skipped for %s: %s", task_id, exc)


async def _maybe_fire_callback(task_id: str, result: dict) -> None:
    """POST the terminal result to the task's callback_url, once, if one was given.
    Looked up from the DB so it survives a pause/resume across an API restart.

    No-ops when the task has no ``callback_url``. Otherwise delivers a small JSON
    payload (task_id/status/error/result_path) and appends a progress line noting
    whether delivery succeeded.

    Args:
        task_id: The finished task.
        result: The terminal result dict.

    Returns:
        None.

    Side effects:
        Reads the callback URL from SQLite, performs an outbound HTTP POST, and
        appends a progress line.
    """
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT callback_url FROM tasks WHERE task_id=?", (task_id,)
        ) as cur:
            row = await cur.fetchone()
    url = row["callback_url"] if row else None
    if not url:
        return
    payload = {
        "task_id": task_id,
        "status": result.get("status"),
        "error": result.get("error"),
        "result_path": result.get("result_path"),
    }
    ok = await fire_callback(url, payload)
    _append_progress(task_id, f"[hyperion] callback {'delivered' if ok else 'failed'} → {url}")


async def _run_and_update(
    task_id: str,
    request: str,
    cap_wall_seconds: Optional[int],
    cap_input_tokens: Optional[int],
    cap_output_tokens: Optional[int],
    hitl: str = "off",
    workflow: Optional[str] = None,
) -> None:
    """Background coroutine that runs a fresh task end-to-end and persists outcome.

    Resets the in-memory progress buffer, marks the task ``running``, invokes the
    crew runner with the given caps/HITL/workflow, persists the result, and emits a
    final status progress line. If the task pauses for human input it returns early
    (the episode is stored only once the task truly finishes); otherwise it stores
    an episode and fires the completion callback.

    Args:
        task_id: The task to run.
        request: The task request text.
        cap_wall_seconds / cap_input_tokens / cap_output_tokens: Per-run budget
            overrides (``None`` → global caps).
        hitl: Human-in-the-loop mode.
        workflow: Workflow id to run (``None`` → server default).

    Returns:
        None — this is a fire-and-forget background task; results land in SQLite.

    Side effects:
        Mutates ``_PROGRESS``, updates the tasks row, may write an episode and POST
        a callback.
    """
    _PROGRESS[task_id] = []
    started = asyncio.get_event_loop().time()

    def _progress(line: str) -> None:
        """Forward a runner progress line to ``_append_progress`` for this task."""
        _append_progress(task_id, line)

    await _update_task(task_id, status="running")
    result = await run_task(
        task_id=task_id,
        request=request,
        progress_callback=_progress,
        cap_wall_seconds=cap_wall_seconds,
        cap_input_tokens=cap_input_tokens,
        cap_output_tokens=cap_output_tokens,
        hitl=hitl,
        workflow=workflow,
    )
    await _persist_result(task_id, result)
    _progress(f"[hyperion] status={result['status']}")

    if result["status"] in ("awaiting_approval", "awaiting_input"):
        return  # paused for human input; episode stored after the task finishes
    _maybe_store_episode(task_id, request, result, started)
    await _maybe_fire_callback(task_id, result)


async def _resume_and_update(
    task_id: str,
    request: str,
    action: str,
    chosen_option: Optional[str],
    edits: Optional[str],
    payload: dict,
) -> None:
    """Background coroutine that resumes a paused task and persists outcome.

    Reconstructs the runner call from the persisted pending ``payload`` (hitl,
    revise_count, workflow, resume_node, caps) plus the approval/answer inputs,
    runs ``resume_task``, persists the result, and emits a status line. Re-pauses
    return early; terminal results store an episode and fire the callback.

    Args:
        task_id: The paused task to resume.
        request: The original task request text.
        action: One of "approve"/"revise"/"reject" (or "revise" for an answered
            question).
        chosen_option: Selected plan option, if any.
        edits: Revision feedback / answer text, if any.
        payload: The persisted ``pending_payload`` dict carrying resume context.

    Returns:
        None — fire-and-forget; results land in SQLite.

    Side effects:
        Mutates ``_PROGRESS``, updates the tasks row, may write an episode and POST
        a callback.
    """
    started = asyncio.get_event_loop().time()

    def _progress(line: str) -> None:
        """Forward a runner progress line to ``_append_progress`` for this task."""
        _append_progress(task_id, line)

    caps = payload.get("caps") or {}
    result = await resume_task(
        task_id=task_id,
        request=request,
        action=action,
        chosen_option=chosen_option,
        edits=edits,
        hitl=payload.get("hitl", "off"),
        revise_count=payload.get("revise_count", 0),
        workflow=payload.get("workflow"),
        resume_node=payload.get("resume_node"),
        progress_callback=_progress,
        cap_wall_seconds=caps.get("cap_wall_seconds"),
        cap_input_tokens=caps.get("cap_input_tokens"),
        cap_output_tokens=caps.get("cap_output_tokens"),
    )
    await _persist_result(task_id, result)
    _progress(f"[hyperion] status={result['status']}")

    if result["status"] in ("awaiting_approval", "awaiting_input"):
        return  # re-paused after a revision / new question
    _maybe_store_episode(task_id, request, result, started)
    await _maybe_fire_callback(task_id, result)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(body: TaskRequest) -> TaskResponse:
    """Submit a new task and start it running in the background.

    Validates the schema version, callback URL (SSRF guard), and workflow id before
    inserting a ``queued`` row and spawning ``_run_and_update``. Returns
    immediately with ``202`` — clients poll/stream for progress.

    Args:
        body: The validated ``TaskRequest``.

    Returns:
        A ``TaskResponse`` with the new task_id and status "queued".

    Raises:
        HTTPException 422: unsupported schema_version, unsafe callback_url,
            unknown workflow id, or a ``workflow_prompt`` that could not be compiled.
        HTTPException 502: the workflow-compilation LLM call failed (proxy down).

    Side effects:
        Inserts a tasks row, may persist a compiled ad-hoc workflow, and launches a
        background runner coroutine.
    """
    if body.schema_version != 1:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported schema_version {body.schema_version} (this server speaks v1)",
        )
    if body.callback_url:
        try:
            validate_callback_url(body.callback_url)
        except UnsafeCallbackURL as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    # Resolve which workflow this run uses. Precedence: an explicit `workflow` id,
    # else a compiled-from-prompt ad-hoc DAG (req 4.1), else the server default.
    workflow_id = body.workflow
    if workflow_id:
        from hyperion.crews.workflows import load_workflow

        try:
            load_workflow(workflow_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Unknown workflow: {exc}")
    elif body.workflow_prompt:
        from hyperion.crews.compiler import WorkflowCompileError, compile_workflow
        from hyperion.crews.workflows import load_all_workflows, save_workflow

        try:
            compiled = compile_workflow(
                body.workflow_prompt, load_all_agents(), load_all_workflows()
            )
        except WorkflowCompileError as exc:
            raise HTTPException(
                status_code=422, detail=f"Could not build a workflow from your prompt: {exc}"
            )
        except Exception as exc:  # proxy/LLM failure — not the caller's fault
            raise HTTPException(status_code=502, detail=f"Workflow compilation failed: {exc}")
        save_workflow(compiled)  # persist so HITL resume + the trace UI can reload it
        workflow_id = compiled.id

    task_id = str(uuid.uuid4())[:8]
    db = await _get_db()  # creates + migrates the table (adds callback_url if absent)
    try:
        await db.execute(
            "INSERT INTO tasks (task_id, status, request, hitl, callback_url, workflow) "
            "VALUES (?,?,?,?,?,?)",
            (task_id, "queued", body.task, body.hitl, body.callback_url, workflow_id),
        )
        await db.commit()
    finally:
        await db.close()

    _spawn_tracked(
        task_id,
        _run_and_update(
            task_id,
            body.task,
            body.cap_wall_seconds,
            body.cap_input_tokens,
            body.cap_output_tokens,
            body.hitl,
            workflow_id,
        ),
    )
    return TaskResponse(task_id=task_id, status="queued")


@app.post("/tasks/{task_id}/stop", response_model=TaskResponse)
async def stop_task(task_id: str) -> TaskResponse:
    """Cancel a non-terminal run, marking it ``cancelled``.

    Cancels the in-flight background ``asyncio.Task`` (if still registered) and
    writes ``status=cancelled`` to the DB immediately so the UI reflects the stop
    without waiting on the run loop.

    Important: cancelling the asyncio task does NOT abort a blocking LLM call that
    is already executing in an executor thread (a running thread cannot be
    cancelled). The DB status flips right away, but any such orphaned call drains
    in the background and is discarded — its result is never persisted because the
    cancelled coroutine never reaches ``_persist_result``. This is also why a
    stop is the only reliable way to clear a run wedged on a hung completion.

    Args:
        task_id: The task to cancel.

    Returns:
        The task's ``TaskResponse`` after cancellation (or its unchanged terminal
        state if it had already finished — the call is idempotent).

    Raises:
        HTTPException 404: no task with this id.
    """
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT status FROM tasks WHERE task_id=?", (task_id,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    # Already finished — nothing to cancel; return current state (idempotent).
    if row["status"] in _TERMINAL:
        return await get_task(task_id)

    t = _RUNNING.get(task_id)
    if t is not None and not t.done():
        t.cancel()

    await _update_task(task_id, status="cancelled", error="Cancelled by user")
    _append_progress(task_id, "[hyperion] status=cancelled")
    return await get_task(task_id)


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Return the current status, progress, routing, and pending affordance of a task.

    Args:
        task_id: The task to look up.

    Returns:
        A fully-populated ``TaskResponse``.

    Raises:
        HTTPException 404: no task with this id.
    """
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT task_id, status, error, result_path, pending_stage, routing "
            "FROM tasks WHERE task_id=?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    routing = json.loads(row["routing"]) if row["routing"] else None
    affordance = _pending_affordance(task_id, row["status"])
    return TaskResponse(
        task_id=row["task_id"],
        status=row["status"],
        error=row["error"],
        result_path=row["result_path"],
        progress_lines=_PROGRESS.get(task_id, []),
        routing=routing,
        pending_stage=row["pending_stage"],
        pending_affordance=affordance,
    )


@app.get("/tasks/{task_id}/trace")
async def get_task_trace(task_id: str) -> dict:
    """Return trace events + DAG structure for the Trace Flow UI.

    Loads the task's request/status/routing plus its ordered ``trace_events`` rows
    (one per agent LLM call), decoding each event's ``tools_used`` JSON column into
    a list (defaulting to ``[]`` on bad JSON).

    Args:
        task_id: The task whose trace to return.

    Returns:
        A dict with ``task_id``, ``request``, ``status``, ``routing``, and the
        ``events`` list.

    Raises:
        HTTPException 404: no task with this id.
    """
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT request, status, routing FROM tasks WHERE task_id=?", (task_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")

        request_text = row["request"] or ""
        routing = json.loads(row["routing"]) if row["routing"] else None

        async with db.execute(
            """SELECT agent_role, node_id, prompt_type, model, input_tokens, output_tokens,
                      cost_usd, prompt_preview, response_preview, tools_used,
                      started_at, duration_ms
               FROM trace_events
               WHERE task_id=?
               ORDER BY started_at, id""",
            (task_id,),
        ) as cur:
            events = [dict(r) for r in await cur.fetchall()]

    for ev in events:
        try:
            ev["tools_used"] = json.loads(ev["tools_used"] or "[]")
        except Exception:
            ev["tools_used"] = []

    return {
        "task_id": task_id,
        "request": request_text,
        "status": row["status"],
        "routing": routing,
        "events": events,
    }


@app.post("/tasks/{task_id}/approve", response_model=TaskResponse)
async def approve_task(task_id: str, body: ApproveRequest) -> TaskResponse:
    """Resume a task paused at the plan gate.

    Reads the pending payload from the DB (not in-memory) so a restarted API can
    still resume. action ∈ {approve, revise, reject}.

    Args:
        task_id: The paused task to resume.
        body: The ``ApproveRequest`` (action + optional chosen_option / edits).

    Returns:
        A ``TaskResponse`` with status "running" — resumption happens in the
        background.

    Raises:
        HTTPException 404: no such task.
        HTTPException 409: task is not in ``awaiting_approval``.

    Side effects:
        Marks the task running and spawns ``_resume_and_update``.
    """
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT status, request, pending_payload FROM tasks WHERE task_id=?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    if row["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Task is not awaiting approval (status={row['status']})",
        )
    payload = json.loads(row["pending_payload"]) if row["pending_payload"] else {}

    await _update_task(task_id, status="running")
    _spawn_tracked(
        task_id,
        _resume_and_update(
            task_id, row["request"], body.action, body.chosen_option, body.edits, payload
        ),
    )
    return TaskResponse(task_id=task_id, status="running")


@app.post("/tasks/{task_id}/feedback", response_model=TaskResponse)
async def feedback_task(task_id: str, body: FeedbackRequest) -> TaskResponse:
    """Push free-text human feedback to a task.

    For a *running* task the runner drains it between stages (changes behavior). For a
    task paused on an ``ask_user`` question (status=awaiting_input) the message answers
    the affordance and resumes the plan stage with the answer injected.

    Args:
        task_id: The target task.
        body: The ``FeedbackRequest`` carrying the human message.

    Returns:
        A ``TaskResponse``: status "running" if the message resumed a paused task,
        otherwise the task's unchanged status (feedback queued).

    Raises:
        HTTPException 404: no such task.

    Side effects:
        Either answers the pending affordance + spawns ``_resume_and_update``, or
        appends the message to the task's feedback queue.
    """
    from hyperion.feedback import answer_affordance, append_feedback

    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT status, request, pending_payload FROM tasks WHERE task_id=?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    if row["status"] == "awaiting_input":
        # The message answers the planner's question; resume by re-running the plan
        # stage with the answer fed back as revision feedback.
        answer_affordance(task_id, body.message)
        payload = json.loads(row["pending_payload"]) if row["pending_payload"] else {}
        await _update_task(task_id, status="running")
        _spawn_tracked(
            task_id,
            _resume_and_update(task_id, row["request"], "revise", None, body.message, payload),
        )
        return TaskResponse(task_id=task_id, status="running")

    # Running (or any other) task: queue the feedback for the next stage drain.
    append_feedback(task_id, body.message)
    return TaskResponse(task_id=task_id, status=row["status"])


@app.post("/tasks/{task_id}/chat", response_model=ChatResponse)
async def chat_task(task_id: str, body: ChatRequest) -> ChatResponse:
    """Hold a follow-up conversation about a finished run's result.

    Unlike ``/feedback`` (which steers a *running* task), this answers questions about a
    task that has already reached ``done``/``failed``. It grounds a lightweight single
    agent on the run's distilled outputs (result + node index) and lets it retrieve any
    node's full detail on demand — without re-running the workflow. The conversation is
    persisted to ``chat.jsonl`` so it is multi-turn and survives an API restart.

    Args:
        task_id: The completed task to discuss.
        body: The ``ChatRequest`` carrying the user's message.

    Returns:
        A ``ChatResponse`` with the assistant's grounded reply for this turn.

    Raises:
        HTTPException 404: no such task.
        HTTPException 409: the task hasn't finished yet (still queued/running/paused).

    Side effects:
        Appends the user and assistant turns to the task's ``chat.jsonl`` and issues
        one (tool-augmented) LLM call.
    """
    from hyperion.followup import append_turn, load_history, run_followup_chat

    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT status, request FROM tasks WHERE task_id=?", (task_id,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    if row["status"] not in ("done", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Task is {row['status']}; follow-up chat is available once it finishes.",
        )

    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=422, detail="Empty message")

    history = load_history(task_id)
    # Run the (blocking, LiteLLM) chat off the event loop so the server stays responsive.
    reply = await asyncio.to_thread(
        run_followup_chat, task_id, row["request"], history, message
    )
    append_turn(task_id, "user", message)
    append_turn(task_id, "assistant", reply)
    return ChatResponse(task_id=task_id, reply=reply)


@app.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str) -> StreamingResponse:
    """Server-Sent Events stream of a task's progress lines until it terminates.

    Args:
        task_id: The task to stream.

    Returns:
        A ``StreamingResponse`` (text/event-stream) yielding each new progress line
        as a ``data:`` event, terminated by a ``[DONE]`` event when the task
        reaches done/failed.
    """
    async def _gen() -> AsyncGenerator[str, None]:
        """Yield new progress lines once per second, polling the DB for terminal status.

        Tracks how many lines have been emitted (``seen``) to send only the delta,
        then ends the stream with a ``[DONE]`` event when the task is done/failed.
        """
        seen = 0
        while True:
            lines = _PROGRESS.get(task_id, [])
            for line in lines[seen:]:
                yield f"data: {line}\n\n"
                seen += 1
            # Check if task is done
            async with _db() as db:
                async with db.execute(
                    "SELECT status FROM tasks WHERE task_id=?", (task_id,)
                ) as cur:
                    row = await cur.fetchone()
            if row and row[0] in ("done", "failed"):
                yield f"data: [DONE] status={row[0]}\n\n"
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/tasks/{task_id}/artifacts/{name:path}")
async def get_artifact(task_id: str, name: str) -> FileResponse:
    """Serve a static artifact file produced by a task.

    Hardened against path traversal: the ``task_id`` shape is validated and the
    resolved artifact path must be inside the task's directory (``is_relative_to``,
    which — unlike ``startswith`` — also blocks sibling-dir escapes).

    Args:
        task_id: The owning task.
        name: Artifact path relative to the task's ``artifacts/`` dir (may contain
            subdirs via the ``{name:path}`` route converter).

    Returns:
        A ``FileResponse`` for the requested artifact.

    Raises:
        HTTPException 400: malformed task_id or a path that escapes the task dir.
        HTTPException 404: the artifact does not exist.
    """
    # Validate task_id shape to prevent traversal via the path param itself
    if not task_id or "/" in task_id or ".." in task_id:
        raise HTTPException(status_code=400, detail="Invalid task_id")
    base = (settings.tasks_dir / task_id).resolve()
    artifact_path = (base / "artifacts" / name).resolve()
    # Use is_relative_to (Python 3.9+) — startswith allows sibling-dir attacks
    if not artifact_path.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(artifact_path)


# ---------------------------------------------------------------------------
# Agent CRUD + options API (Phase 5) — additive endpoints, no model restructuring.
# ---------------------------------------------------------------------------


async def _litellm_model_ids() -> list[str]:
    """Concrete model ids the proxy reports. Best-effort — empty on any failure.

    GETs ``{litellm_base_url}/models`` with the configured API key (5s timeout).

    Returns:
        The list of model id strings, or ``[]`` if the proxy is unreachable or the
        response can't be parsed (logged at WARNING). An empty list is treated by
        callers as "can't validate" rather than "no models", so they don't block
        offline edits.
    """
    import httpx

    url = settings.litellm_base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {settings.llm_api_key}"})
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
    except Exception as exc:
        logger.warning("Could not fetch LiteLLM models: %s", exc)
        return []


async def _assert_model_alias_valid(record: AgentRecord) -> None:
    """Reject an unknown model_alias — but only when we can actually see the model
    list (offline edits aren't blocked by a transient proxy outage).

    Accepts any of the well-known role aliases immediately; otherwise checks the
    record's alias against the live proxy model list, and only rejects when that
    list is non-empty (so a proxy outage doesn't block the edit).

    Args:
        record: The agent record being created/updated.

    Returns:
        None.

    Raises:
        HTTPException 422: the alias is neither a known role alias nor a concrete
            model id reported by the (reachable) proxy.
    """
    known_aliases = models_registry.alias_names()
    if record.model_alias in known_aliases:
        return
    known = await _litellm_model_ids()
    if known and record.model_alias not in known:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model_alias {record.model_alias!r}. "
                   f"Use one of {list(known_aliases)} or a concrete model id.",
        )


@app.get("/agents")
async def list_agents(group: Optional[str] = None) -> list[dict]:
    """List all agent records, optionally filtered to one group.

    Args:
        group: If given, only agents whose ``group`` matches are returned.

    Returns:
        A list of serialized agent record dicts.
    """
    records = load_all_agents()
    if group:
        records = [r for r in records if r.group == group]
    return [r.model_dump() for r in records]


@app.get("/groups")
async def list_groups() -> list[str]:
    """Distinct agent groups, for the UI group filter.

    Returns:
        Sorted unique group names across all agent records.
    """
    return sorted({r.group for r in load_all_agents()})


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str) -> dict:
    """Fetch a single agent record by id.

    Args:
        agent_id: The agent to load.

    Returns:
        The serialized agent record dict.

    Raises:
        HTTPException 404: no agent with this id.
    """
    try:
        return load_agent(agent_id).model_dump()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No agent {agent_id!r}")


@app.post("/agents", status_code=201)
async def create_agent(record: AgentRecord) -> dict:
    """Create a new agent record after full validation.

    Checks id uniqueness, single-record validity, model alias validity, and the
    whole-store invariants on the prospective set before persisting.

    Args:
        record: The new agent record.

    Returns:
        The persisted record (serialized).

    Raises:
        HTTPException 409: an agent with this id already exists.
        HTTPException 422: the record or resulting collection is invalid, or the
            model alias is unknown.

    Side effects:
        Writes the agent JSON file.
    """
    existing = {r.id: r for r in load_all_agents()}
    if record.id in existing:
        raise HTTPException(status_code=409, detail=f"Agent {record.id!r} already exists")
    try:
        validate_agent(record)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await _assert_model_alias_valid(record)
    save_agent(record)
    return record.model_dump()


@app.put("/agents/{agent_id}")
async def update_agent(agent_id: str, record: AgentRecord) -> dict:
    """Replace an existing agent record after validation.

    Args:
        agent_id: The id from the URL (must equal ``record.id``).
        record: The new full record to store.

    Returns:
        The persisted record (serialized).

    Raises:
        HTTPException 422: body id mismatch, invalid record, unknown model alias,
            or a resulting collection that breaks invariants.
        HTTPException 404: no agent with this id.

    Side effects:
        Overwrites the agent JSON file.
    """
    if record.id != agent_id:
        raise HTTPException(status_code=422, detail="Body id must match the URL id")
    existing = {r.id: r for r in load_all_agents()}
    if agent_id not in existing:
        raise HTTPException(status_code=404, detail=f"No agent {agent_id!r}")
    try:
        validate_agent(record)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await _assert_model_alias_valid(record)
    save_agent(record)
    return record.model_dump()


@app.delete("/agents/{agent_id}")
async def remove_agent(agent_id: str) -> dict:
    """Delete an agent record if removing it keeps the store valid.

    Args:
        agent_id: The agent to delete.

    Returns:
        ``{"deleted": agent_id}``.

    Raises:
        HTTPException 404: no agent with this id.
        HTTPException 422: deletion would break store invariants (e.g. removing the
            last plan/synthesize agent).

    Side effects:
        Removes the agent JSON file.
    """
    existing = {r.id: r for r in load_all_agents()}
    if agent_id not in existing:
        raise HTTPException(status_code=404, detail=f"No agent {agent_id!r}")
    delete_agent(agent_id)
    return {"deleted": agent_id}


@app.post("/agents/{agent_id}/duplicate", status_code=201)
async def duplicate_agent(agent_id: str, new_id: Optional[str] = None) -> dict:
    """Clone an agent under a new id (default ``{id}-copy``).

    Args:
        agent_id: The source agent to copy.
        new_id: Target id for the clone; defaults to ``{agent_id}-copy``.

    Returns:
        The new clone record (serialized).

    Raises:
        HTTPException 404: source agent not found.
        HTTPException 409: target id already exists.
        HTTPException 422: the clone fails single-record validation.

    Side effects:
        Writes a new agent JSON file.
    """
    existing = {r.id: r for r in load_all_agents()}
    if agent_id not in existing:
        raise HTTPException(status_code=404, detail=f"No agent {agent_id!r}")
    src = existing[agent_id]
    target = new_id or f"{agent_id}-copy"
    if target in existing:
        raise HTTPException(status_code=409, detail=f"Agent {target!r} already exists")
    clone = src.model_copy(update={"id": target, "name": f"{src.name} (copy)"})
    try:
        validate_agent(clone)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    save_agent(clone)
    return clone.model_dump()


# ---------------------------------------------------------------------------
# Workflow CRUD — named DAGs of agent nodes, picked per-run or as the default.
# ---------------------------------------------------------------------------


def _validate_workflow_record(record) -> None:
    """Structural + agent-reference validation, surfaced as a 422.

    Validates the workflow's DAG structure and that every node references a known
    agent id.

    Args:
        record: The ``WorkflowRecord`` to validate.

    Returns:
        None.

    Raises:
        HTTPException 422: the workflow is structurally invalid or references an
            unknown agent.
    """
    from hyperion.crews.workflows import (
        load_all_workflows,
        load_workflow,
        validate_workflow,
    )

    known = {r.id for r in load_all_agents()}
    known_workflows = {w.id for w in load_all_workflows()}
    try:
        # Pass the workflow registry + a loader so subworkflow refs are checked
        # for existence and cross-workflow cycles (A -> B -> A), not just structure.
        validate_workflow(record, known, known_workflows, load_workflow)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/workflows")
async def list_workflows() -> list[dict]:
    """List all defined workflow DAGs.

    Returns:
        A list of serialized ``WorkflowRecord`` dicts.
    """
    from hyperion.crews.workflows import load_all_workflows

    return [w.model_dump() for w in load_all_workflows()]


@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str) -> dict:
    """Fetch a single workflow DAG by id.

    Args:
        workflow_id: The workflow to load.

    Returns:
        The serialized ``WorkflowRecord`` dict.

    Raises:
        HTTPException 404: no workflow with this id.
        HTTPException 422: the stored workflow file is malformed.
    """
    from hyperion.crews.workflows import load_workflow

    try:
        return load_workflow(workflow_id).model_dump()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No workflow {workflow_id!r}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/workflows", status_code=201)
async def create_workflow(record: WorkflowRecord) -> dict:
    """Create a new workflow DAG after structural validation.

    Args:
        record: The new ``WorkflowRecord``.

    Returns:
        The persisted record (serialized).

    Raises:
        HTTPException 409: a workflow with this id already exists.
        HTTPException 422: the workflow is invalid.

    Side effects:
        Writes the workflow JSON file.
    """
    from hyperion.crews.workflows import load_all_workflows, save_workflow

    existing = {w.id for w in load_all_workflows()}
    if record.id in existing:
        raise HTTPException(status_code=409, detail=f"Workflow {record.id!r} already exists")
    _validate_workflow_record(record)
    save_workflow(record)
    return record.model_dump()


@app.put("/workflows/{workflow_id}")
async def update_workflow(workflow_id: str, record: WorkflowRecord) -> dict:
    """Replace an existing workflow DAG after validation.

    Args:
        workflow_id: The id from the URL (must equal ``record.id``).
        record: The new full workflow record.

    Returns:
        The persisted record (serialized).

    Raises:
        HTTPException 422: body id mismatch or invalid workflow.
        HTTPException 404: no workflow with this id.

    Side effects:
        Overwrites the workflow JSON file.
    """
    from hyperion.crews.workflows import load_all_workflows, save_workflow

    if record.id != workflow_id:
        raise HTTPException(status_code=422, detail="Body id must match the URL id")
    if workflow_id not in {w.id for w in load_all_workflows()}:
        raise HTTPException(status_code=404, detail=f"No workflow {workflow_id!r}")
    _validate_workflow_record(record)
    save_workflow(record)
    return record.model_dump()


@app.delete("/workflows/{workflow_id}")
async def remove_workflow(workflow_id: str) -> dict:
    """Delete a workflow DAG, guarding the last-one and default invariants.

    Args:
        workflow_id: The workflow to delete.

    Returns:
        ``{"deleted": workflow_id}``.

    Raises:
        HTTPException 404: no workflow with this id.
        HTTPException 409: it's the only workflow, or it's the current default
            workflow (set a different default first).

    Side effects:
        Removes the workflow JSON file.
    """
    from hyperion.crews.workflows import delete_workflow, load_all_workflows

    existing = {w.id for w in load_all_workflows()}
    if workflow_id not in existing:
        raise HTTPException(status_code=404, detail=f"No workflow {workflow_id!r}")
    if len(existing) == 1:
        raise HTTPException(status_code=409, detail="Cannot delete the last workflow")
    if workflow_id == settings.default_workflow:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete the default workflow; set a different default first",
        )
    delete_workflow(workflow_id)
    return {"deleted": workflow_id}


@app.post("/workflows/{workflow_id}/duplicate", status_code=201)
async def duplicate_workflow(workflow_id: str, new_id: Optional[str] = None) -> dict:
    """Clone a workflow DAG under a new id (default ``{id}-copy``).

    Args:
        workflow_id: The source workflow to copy.
        new_id: Target id for the clone; defaults to ``{workflow_id}-copy``.

    Returns:
        The new clone record (serialized).

    Raises:
        HTTPException 404: source workflow not found.
        HTTPException 409: target id already exists.
        HTTPException 422: the clone fails validation.

    Side effects:
        Writes a new workflow JSON file.
    """
    from hyperion.crews.workflows import load_all_workflows, load_workflow, save_workflow

    existing = {w.id for w in load_all_workflows()}
    if workflow_id not in existing:
        raise HTTPException(status_code=404, detail=f"No workflow {workflow_id!r}")
    target = new_id or f"{workflow_id}-copy"
    if target in existing:
        raise HTTPException(status_code=409, detail=f"Workflow {target!r} already exists")
    src = load_workflow(workflow_id)
    clone = src.model_copy(update={"id": target, "name": f"{src.name} (copy)"})
    _validate_workflow_record(clone)
    save_workflow(clone)
    return clone.model_dump()


@app.get("/tools")
async def list_tools() -> list[dict]:
    """Tool names available to agents, with a description sampled from each factory."""
    out = []
    for name in sorted(TOOL_REGISTRY):
        desc = ""
        try:
            desc = getattr(TOOL_REGISTRY[name]("_probe"), "description", "")
        except Exception:
            pass
        out.append({"name": name, "description": desc})
    return out


# ---------------------------------------------------------------------------
# Tool endpoints — expose Hyperion's tools over HTTP so OWUI models and other
# callers can use them without delegating a full orchestration task.
#
# This is the canonical implementation for web search and second-brain lookup:
# all logic lives here in the Hyperion tools layer. OWUI plugins call these
# endpoints and are intentionally thin HTTP wrappers — updating the tool logic
# here automatically propagates to every caller.
# ---------------------------------------------------------------------------

# NOTE: the alias → fallback-chain mapping that the agent editor and settings page
# render now lives in the operator-editable registry (``models_registry.alias_details()``),
# not a hard-coded constant. Callers below derive it from there.


@app.get("/tools/search")
async def tool_web_search(
    q: str,
    top_k: int = 10,
    categories: str = "general,news",
) -> dict:
    """Web search via SearXNG + Infinity reranker.

    Delegates to the same WebSearchTool that Hyperion agents use internally, so
    the prompt-injection defenses, reranking, and snippet sanitization are
    identical whether the caller is an OWUI model or an in-process Hyperion agent.

    Args:
        q: Search query string.
        top_k: Maximum number of results to return after reranking.
        categories: Comma-separated SearXNG category list.

    Returns:
        {"query": str, "result": str} — result is Markdown-formatted web results.
    """
    tool = WebSearchTool(top_k=top_k, categories=categories)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, tool._run, q)
    return {"query": q, "result": result}


@app.get("/tools/second-brain")
async def tool_second_brain(q: str, limit: int = 5) -> dict:
    """Semantic search over the Qdrant second-brain collection.

    Delegates to the same SecondBrainTool that Hyperion agents use internally,
    including the Infinity reranker pass and the per-call token-budget trim.

    Args:
        q: Natural-language search query.
        limit: Maximum number of results to return after reranking.

    Returns:
        {"query": str, "result": str} — result is Markdown-formatted excerpts.
    """
    tool = SecondBrainTool(top_k=limit)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, tool._run, q)
    return {"query": q, "result": result}


@app.get("/models")
async def list_models() -> dict:
    """Role aliases plus the concrete models the proxy currently exposes."""
    return {
        "aliases": list(models_registry.alias_names()),
        "models": await _litellm_model_ids(),
        "current": {
            "planner": settings.model_planner,
            "worker": settings.model_worker,
            "cheap": settings.model_cheap,
        },
        # Full operator-editable registry (roles + alias chains), so the settings page
        # and agent editor can render and edit everything without reading litellm_config.yaml.
        "roles": models_registry.roles(),
        "aliases_detail": models_registry.aliases(),
        # Annotated fallback chain for each alias — shown in the agent editor and settings.
        "alias_details": models_registry.alias_details(),
    }


class ConfigUpdate(BaseModel):
    model_planner: Optional[str] = None
    model_worker: Optional[str] = None
    model_cheap: Optional[str] = None
    default_workflow: Optional[str] = None


@app.put("/config")
async def update_config(body: ConfigUpdate) -> dict:
    """Reassign role models / default workflow live (no restart) and persist for boot."""
    known = await _litellm_model_ids()
    updates: dict[str, str] = {}
    for field in _MODEL_FIELDS:
        value = getattr(body, field)
        if not value:
            continue
        if value not in models_registry.alias_names() and known and value not in known:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown model {value!r} for {field}. "
                       f"Use one of {list(models_registry.alias_names())} or a concrete model id.",
            )
        updates[field] = value

    if body.default_workflow:
        from hyperion.crews.workflows import load_workflow

        try:
            load_workflow(body.default_workflow)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Unknown workflow: {exc}")
        updates["default_workflow"] = body.default_workflow

    for field, value in updates.items():
        setattr(settings, field, value)  # live effect

    if updates:
        path = _model_override_path()
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8")) or {}
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

        # Keep the registry (the canonical source of role models) in sync so the new
        # Roles UI and a future reboot reflect changes made through this legacy endpoint.
        role_field_models = {
            "planner": updates.get("model_planner"),
            "worker": updates.get("model_worker"),
            "cheap": updates.get("model_cheap"),
        }
        if any(role_field_models.values()):
            reg = models_registry.load_registry()
            for role in reg["roles"]:
                new_model = role_field_models.get(role.get("name"))
                if new_model:
                    role["model"] = new_model
            models_registry.save_registry(reg)

    return await get_config()


# ---------------------------------------------------------------------------
# Role + alias registry (operator-editable model roles and alias fallback chains)
# ---------------------------------------------------------------------------


class RoleIn(BaseModel):
    """One role row in a ``PUT /roles`` request body."""

    name: str
    note: str = ""
    model: str


class RolesUpdate(BaseModel):
    """``PUT /roles`` body — the full, ordered roles list (replaces the current set)."""

    roles: list[RoleIn]


class AliasUpdate(BaseModel):
    """``PUT /aliases/{name}`` body — the alias's ordered list of concrete model ids."""

    models: list[str]


async def _save_registry_validated(reg: dict) -> None:
    """Validate a candidate registry against the live proxy model list and persist it.

    Args:
        reg: ``{"roles": [...], "aliases": {...}}`` candidate document.

    Raises:
        HTTPException 422: if :func:`models_registry.validate_registry` rejects it.

    Side effects:
        Writes ``model_registry.json`` and re-applies role models to ``settings``.
    """
    known = await _litellm_model_ids()
    try:
        models_registry.validate_registry(reg, known)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    models_registry.save_registry(reg)
    models_registry.apply_roles_to_settings()


async def _reconcile_alias_safe(name: str, models: list[str] | None) -> dict:
    """Best-effort LiteLLM write-through for one alias; never raises.

    Calls the admin reconcile (``tools.litellm_admin``) so a new/edited alias actually
    routes through the proxy. Returns a status dict the UI can surface. ``models=None``
    means "delete this alias's deployments". Failures are captured as
    ``{"status": "error", ...}`` rather than failing the whole request — the registry
    edit still succeeds and can be re-reconciled later.
    """
    try:
        from hyperion.tools.litellm_admin import reconcile_alias

        return await reconcile_alias(name, models)
    except Exception as exc:  # pragma: no cover - defensive; admin API optional
        logger.warning("LiteLLM reconcile for alias %r failed: %s", name, exc)
        return {"status": "error", "detail": str(exc)}


@app.get("/roles")
async def list_roles() -> dict:
    """Return the operator-editable roles list."""
    return {"roles": models_registry.roles()}


@app.put("/roles")
async def update_roles(body: RolesUpdate) -> dict:
    """Replace the roles list (add / rename / remove / re-point), validated + persisted.

    The three built-in roles (planner/worker/cheap) must remain present; their chosen
    model flows back onto ``settings`` so the LLM factory functions pick it up live.
    """
    reg = models_registry.load_registry()
    reg["roles"] = [r.model_dump() for r in body.roles]
    await _save_registry_validated(reg)
    return {"roles": models_registry.roles()}


@app.get("/aliases")
async def list_aliases() -> dict:
    """Return alias chains plus each alias's live routing status from the proxy."""
    aliases = models_registry.aliases()
    try:
        from hyperion.tools.litellm_admin import alias_routing_status

        statuses = await alias_routing_status(aliases)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not fetch alias routing status: %s", exc)
        statuses = {}
    return {
        "aliases": aliases,
        "builtins": list(models_registry.BUILTIN_ALIASES),
        "status": statuses,
    }


@app.put("/aliases/{name}")
async def upsert_alias(name: str, body: AliasUpdate) -> dict:
    """Create or replace an alias's ordered model chain, then write through to LiteLLM."""
    reg = models_registry.load_registry()
    reg["aliases"][name] = list(body.models)
    await _save_registry_validated(reg)
    status = await _reconcile_alias_safe(name, body.models)
    return {"name": name, "models": body.models, "status": status}


@app.delete("/aliases/{name}")
async def delete_alias(name: str) -> dict:
    """Delete a user-defined alias (refused for built-ins or referenced aliases)."""
    if name in models_registry.BUILTIN_ALIASES:
        raise HTTPException(status_code=422, detail=f"Built-in alias {name!r} cannot be deleted")
    reg = models_registry.load_registry()
    if name not in reg["aliases"]:
        raise HTTPException(status_code=404, detail=f"Unknown alias {name!r}")

    # Refuse deletion while any role or agent still references the alias.
    referencing_roles = [r["name"] for r in reg["roles"] if r.get("model") == name]
    if referencing_roles:
        raise HTTPException(
            status_code=422,
            detail=f"Alias {name!r} is used by role(s): {', '.join(referencing_roles)}",
        )
    referencing_agents = [
        a.id for a in load_all_agents()
        if a.model_alias == name or a.fallback_alias == name
    ]
    if referencing_agents:
        raise HTTPException(
            status_code=422,
            detail=f"Alias {name!r} is used by agent(s): {', '.join(referencing_agents)}",
        )

    del reg["aliases"][name]
    await _save_registry_validated(reg)
    status = await _reconcile_alias_safe(name, None)  # remove deployments from proxy
    return {"name": name, "deleted": True, "status": status}


# ---------------------------------------------------------------------------
# Phase 8 — thresholds, paginated tasks, metrics, monitoring deep-links
# ---------------------------------------------------------------------------


def _langfuse_session_url(task_id: str) -> Optional[str]:
    """Deep-link to the Langfuse session whose id == task_id (set in llms.py).
    Fully resolvable only when langfuse_project_id is configured."""
    host = (settings.langfuse_public_url or settings.langfuse_host or "").rstrip("/")
    if not host:
        return None
    if settings.langfuse_project_id:
        return f"{host}/project/{settings.langfuse_project_id}/sessions/{task_id}"
    return f"{host}/sessions/{task_id}"


class ThresholdUpdate(BaseModel):
    cap_input_tokens: Optional[int] = None
    cap_output_tokens: Optional[int] = None
    cap_tool_loop: Optional[int] = None
    cap_wall_seconds: Optional[int] = None
    # Per-agent token/activation overrides: {agent_id: {max_input_tokens: int, ...}}
    agents: Optional[dict[str, dict[str, Optional[int]]]] = None


@app.get("/thresholds")
async def get_thresholds() -> dict:
    """Global caps plus each agent's per-record token/activation thresholds."""
    agents = {
        r.id: r.thresholds.model_dump()
        for r in load_all_agents()
    }
    return {
        "global": {field: getattr(settings, field) for field in _CAP_FIELDS},
        "agents": agents,
    }


@app.put("/thresholds")
async def update_thresholds(body: ThresholdUpdate) -> dict:
    """Update global caps (live + persisted) and/or per-agent thresholds (written
    back into the agent records). Additive — leaves unspecified fields untouched."""
    updates: dict[str, int] = {}
    for field in _CAP_FIELDS:
        value = getattr(body, field)
        if value is None:
            continue
        if not isinstance(value, int) or value <= 0:
            raise HTTPException(status_code=422, detail=f"{field} must be a positive integer")
        updates[field] = value

    for field, value in updates.items():
        setattr(settings, field, value)  # live effect

    if updates:
        path = _thresholds_path()
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8")) or {}
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # Per-agent threshold overrides are written into the agent JSON records.
    for agent_id, fields in (body.agents or {}).items():
        record = load_agent(agent_id)  # 404s naturally if missing
        th = record.thresholds.model_dump()
        for key in ("max_input_tokens", "max_output_tokens", "max_activations_per_day"):
            if key in fields:
                th[key] = fields[key]
        record.thresholds = type(record.thresholds)(**th)
        save_agent(record)

    return await get_thresholds()


def _iso_utc(ts: Optional[str]) -> Optional[str]:
    """SQLite CURRENT_TIMESTAMP is naive UTC ('YYYY-MM-DD HH:MM:SS'). Tag it as
    ISO-8601 UTC ('...TZ') so the browser renders it in the viewer's local
    timezone instead of mis-parsing the naive string as local time."""
    if not ts or "T" in ts or ts.endswith("Z") or "+" in ts:
        return ts
    return ts.replace(" ", "T", 1) + "Z"


@app.get("/tasks")
async def list_tasks(limit: int = 50, offset: int = 0) -> dict:
    """Paginated run history, newest first, for the monitoring page."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT COUNT(*) AS n FROM tasks") as cur:
            total = (await cur.fetchone())["n"]
        async with db.execute(
            "SELECT task_id, status, request, error, created_at, updated_at, hitl "
            "FROM tasks ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()
    items = []
    for r in rows:
        items.append(
            {
                "task_id": r["task_id"],
                "status": r["status"],
                "request": (r["request"] or "")[:200],
                "error": r["error"],
                "created_at": _iso_utc(r["created_at"]),
                "updated_at": _iso_utc(r["updated_at"]),
                "hitl": r["hitl"],
                "langfuse_url": _langfuse_session_url(r["task_id"]),
            }
        )
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@app.get("/metrics")
async def get_metrics() -> dict:
    """Per-agent activation counts + error rate (from the routing column) and token
    usage (summed from the persisted trace_events table). Powers the monitoring tiles.

    Token totals come from trace_events rather than the in-memory usage accountant so
    the bars stay durable across API restarts and consistent with the DB-derived
    activation counts (the in-memory accountant is reset on every process restart)."""
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT status, routing FROM tasks") as cur:
            rows = await cur.fetchall()
        # Durable per-agent token totals from the trace log (keyed by agent id).
        async with db.execute(
            "SELECT agent_role, SUM(input_tokens) AS input, "
            "SUM(output_tokens) AS output FROM trace_events GROUP BY agent_role"
        ) as cur:
            token_rows = await cur.fetchall()
    tokens = {
        tr["agent_role"]: {"input": tr["input"] or 0, "output": tr["output"] or 0}
        for tr in token_rows
    }

    status_counts: dict[str, int] = {}
    per_agent: dict[str, dict[str, int]] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        if not r["routing"]:
            continue
        try:
            routing = json.loads(r["routing"])
        except (json.JSONDecodeError, TypeError):
            continue
        # Routing is persisted by the runner as {"selected_agents": [id, ...], ...}.
        selected = routing.get("selected_agents") or routing.get("selected") or []
        for a in selected:
            aid = a.get("id") if isinstance(a, dict) else a
            if not aid:
                continue
            bucket = per_agent.setdefault(aid, {"activations": 0, "errors": 0})
            bucket["activations"] += 1
            if r["status"] == "failed":
                bucket["errors"] += 1

    caps = {field: getattr(settings, field) for field in _CAP_FIELDS}
    agents = []
    for record in load_all_agents():
        stats = per_agent.get(record.id, {"activations": 0, "errors": 0})
        acts = stats["activations"]
        errs = stats["errors"]
        agents.append(
            {
                "id": record.id,
                "name": record.name,
                "group": record.group,
                "active": record.active,
                "activations": acts,
                "errors": errs,
                "error_rate": round(errs / acts, 3) if acts else 0.0,
                "tokens": tokens.get(record.id, {"input": 0, "output": 0}),
                "thresholds": record.thresholds.model_dump(),
            }
        )
    # Reranker health: fail-soft means a chronic reranker outage is otherwise invisible
    # (results still return, just unranked). Surface the in-process counters so a rising
    # degraded share is visible here. Best-effort import — never fail /metrics over it.
    try:
        from hyperion.tools.reranker import health_stats as _rerank_health
        reranker_stats = _rerank_health()
    except Exception:
        reranker_stats = {}

    return {
        "tasks_total": sum(status_counts.values()),
        "by_status": status_counts,
        "caps": caps,
        "agents": agents,
        "reranker": reranker_stats,
    }


# ---------------------------------------------------------------------------
# Phase 9 — orchestration & polish: agent card, follow-ups, config export/import
# ---------------------------------------------------------------------------


@app.get("/.well-known/agent.json")
async def agent_card() -> dict:
    """A2A-style agent descriptor so external orchestrators (n8n, other agents)
    can discover Hyperion's contract. Skills are the active agents;
    the input contract is POST /tasks with schema_version:1."""
    agents = load_all_agents()
    skills = [
        {"id": r.id, "name": r.name, "description": r.description or r.goal}
        for r in agents
        if r.active
    ]
    return {
        "schema_version": 1,
        "name": "Hyperion",
        "description": "Local multi-agent orchestrator (plan → work → synthesize).",
        "url": settings.hyperion_api_url,
        "version": app.version,
        "capabilities": {"streaming": True, "humanInTheLoop": True, "webhooks": True},
        "endpoints": {
            "submit": "POST /tasks",
            "status": "GET /tasks/{task_id}",
            "stream": "GET /tasks/{task_id}/stream",
            "approve": "POST /tasks/{task_id}/approve",
            "feedback": "POST /tasks/{task_id}/feedback",
        },
        "input_modes": ["text"],
        "output_modes": ["text", "file"],
        "skills": skills,
    }


def _result_text(task_id: str, result_path: Optional[str]) -> str:
    """Read a finished task's primary artifact for the save-to-notion follow-up."""
    if result_path and Path(result_path).exists():
        return Path(result_path).read_text(encoding="utf-8")
    return ""


class SaveToNotionRequest(BaseModel):
    title: Optional[str] = None


@app.post("/tasks/{task_id}/save-to-notion")
async def save_task_to_notion(task_id: str, body: SaveToNotionRequest) -> dict:
    """Synthesizer follow-up affordance: write a completed task's result to Notion."""
    from hyperion.tools.notion import create_notion_page

    async with _db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT status, request, result_path FROM tasks WHERE task_id=?", (task_id,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    if row["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Task is {row['status']}, not done")

    text = _result_text(task_id, row["result_path"])
    if not text:
        raise HTTPException(status_code=409, detail="Task has no readable result artifact")

    title = body.title or f"Hyperion: {(row['request'] or task_id)[:80]}"
    result = create_notion_page(title, text)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@app.get("/config/export")
async def export_config() -> Response:
    """Download the agent + workflow store as a zip — back up or move config."""
    from hyperion.agents.registry import _agents_dir
    from hyperion.crews.workflows import _workflows_dir

    buf = io.BytesIO()
    agents_dir = _agents_dir()
    workflows_dir = _workflows_dir()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if agents_dir.exists():
            for path in sorted(agents_dir.glob("*.json")):
                zf.write(path, arcname=f"agents/{path.name}")
        if workflows_dir.exists():
            for path in sorted(workflows_dir.glob("*.json")):
                zf.write(path, arcname=f"workflows/{path.name}")
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="hyperion-config.zip"'},
    )


@app.post("/config/import")
async def import_config(file: UploadFile = File(...)) -> dict:
    """Restore an agent + workflow store from an exported zip. Validates every record
    before writing and rejects the whole import if any record is malformed (atomic).
    Entries are routed by path prefix: ``agents/*.json`` and ``workflows/*.json``;
    a flat ``*.json`` (legacy export) is treated as an agent record."""
    from hyperion.crews.workflows import (
        load_all_workflows,
        save_workflow,
        validate_workflow,
    )

    raw = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid zip")

    records: list[AgentRecord] = []
    workflows: list[WorkflowRecord] = []
    for name in zf.namelist():
        if not name.endswith(".json") or name.endswith("/"):
            continue
        text = zf.read(name).decode("utf-8")
        if name.startswith("workflows/"):
            try:
                workflows.append(WorkflowRecord.model_validate_json(text))
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid workflow {name!r}: {exc}")
            continue
        try:
            record = AgentRecord.model_validate_json(text)
            validate_agent(record)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid record {name!r}: {exc}")
        records.append(record)

    if not records:
        raise HTTPException(status_code=422, detail="No agent records found in archive")

    # Validate workflows against the imported agent set (cross-reference must
    # resolve). Subworkflow refs may target a workflow elsewhere in this archive or
    # one already on disk, so the known set + resolver span both.
    from hyperion.crews.workflows import load_workflow

    known_agents = {r.id for r in records}
    imported_by_id = {wf.id: wf for wf in workflows}
    known_workflows = set(imported_by_id) | {w.id for w in load_all_workflows()}

    def _resolve_imported(wf_id: str) -> WorkflowRecord:
        """Prefer a workflow from this import batch, else fall back to disk."""
        if wf_id in imported_by_id:
            return imported_by_id[wf_id]
        return load_workflow(wf_id)

    for wf in workflows:
        try:
            validate_workflow(wf, known_agents, known_workflows, _resolve_imported)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid workflow {wf.id!r}: {exc}")

    for record in records:
        save_agent(record)
    for wf in workflows:
        save_workflow(wf)
    return {
        "imported": [r.id for r in records],
        "count": len(records),
        "workflows": [w.id for w in workflows],
    }


# ---------------------------------------------------------------------------
# Admin — source file read/write (dev mode, requires src volume mount)
# ---------------------------------------------------------------------------

_SRC_ROOT = Path("/app/src")


def _resolve_src_path(rel: str) -> Path:
    """Resolve a relative path inside /app/src, rejecting traversal attempts."""
    path = (_SRC_ROOT / rel).resolve()
    if not path.is_relative_to(_SRC_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return path


@app.get("/admin/files")
async def list_src_files(prefix: str = "") -> dict:
    """Recursively list .py files under /app/src (or a sub-prefix)."""
    base = _resolve_src_path(prefix) if prefix else _SRC_ROOT
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {prefix!r}")
    files = sorted(str(p.relative_to(_SRC_ROOT)) for p in base.rglob("*.py"))
    return {"root": str(_SRC_ROOT), "files": files}


@app.get("/admin/files/{path:path}")
async def read_src_file(path: str) -> dict:
    """Read a source file. path is relative to /app/src."""
    fp = _resolve_src_path(path)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path!r}")
    return {"path": path, "content": fp.read_text(encoding="utf-8")}


class FileWriteBody(BaseModel):
    content: str


@app.put("/admin/files/{path:path}", status_code=200)
async def write_src_file(path: str, body: FileWriteBody) -> dict:
    """Overwrite a source file. uvicorn --reload picks up the change automatically."""
    fp = _resolve_src_path(path)
    if not fp.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path!r} (use a path returned by list_src_files)")
    fp.write_text(body.content, encoding="utf-8")
    return {"path": path, "bytes": len(body.content.encode())}


def main() -> None:
    import uvicorn

    uvicorn.run("hyperion.server.api:app", host="0.0.0.0", port=4100, reload=False, log_level="info")
