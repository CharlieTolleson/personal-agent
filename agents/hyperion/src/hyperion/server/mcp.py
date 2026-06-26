"""
Hyperion MCP server — exposes Hyperion's multi-agent orchestrator to MCP clients
(e.g. Claude Code) as a small set of callable tools.

Role in the system
------------------
This module is the Model Context Protocol (MCP) front door to Hyperion. Hyperion's
"real" HTTP control plane is the FastAPI service (``settings.hyperion_api_url``,
typically http://localhost:4100); this MCP server is a separate, lightweight ASGI
app that lets an MCP-aware agent submit and inspect Hyperion tasks without speaking
the REST API directly.

Transport: streamable HTTP, mounted at the ``/mcp`` route and served on port 4101
(see ``main``). Sessions are stateless (see ``make_starlette_app``): each tool call
is self-contained and no per-session state is retained between requests.

Tools exposed (see ``list_tools`` for full schemas):
  hyperion_run(task[, workflow])           → task_id (runs the crew in the background)
  hyperion_status(task_id)                 → status + recent progress lines
  hyperion_artifact(task_id[, name])       → artifact file contents as text
  hyperion_approve(task_id[, action, ...]) → resume a plan-approval gate
  hyperion_feedback(task_id, message)      → send free-text feedback / answer a question

Design notes
------------
- Task lifecycle state is persisted in a local SQLite database (``_db_path()``) so
  that ``hyperion_status`` can report on tasks even though crew execution happens in
  a detached asyncio task. The schema is created lazily on first use (``_ensure_db``).
- ``hyperion_run`` / ``hyperion_status`` / ``hyperion_artifact`` are handled
  in-process here. ``hyperion_approve`` / ``hyperion_feedback`` are proxied to the
  FastAPI service via ``_post_api`` because that service owns the human-in-the-loop
  (HITL) orchestration and resume tokens — this MCP process does not.
- ``_PROGRESS`` is an in-memory, per-process buffer of progress lines; it is NOT
  persisted and is lost on restart (only the DB-backed status survives).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from typing import AsyncIterator

import aiosqlite
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from hyperion.config import settings
from hyperion.crews.default import run_crew

logger = logging.getLogger(__name__)

# The MCP server instance; tool registration happens via its decorators below.
server = Server("hyperion")

# In-memory, per-process map of task_id -> ordered list of progress strings.
# Populated by the progress callback in hyperion_run; read by hyperion_status.
# Not persisted: cleared on process restart (the SQLite row is the durable record).
_PROGRESS: dict[str, list[str]] = {}

# SQLite database holding durable task lifecycle state, colocated with task workdirs.
def _db_path():
    """Resolve the state-DB path from the *current* ``settings.tasks_dir``.

    Computed per call (not captured at import) so that redirecting
    ``settings.tasks_dir`` also moves the DB — matching ``api._db_path`` and
    keeping the API and MCP entry points pointed at the same file.
    """
    return settings.tasks_dir / "state.db"


async def _ensure_db() -> None:
    """Create the tasks directory and the ``tasks`` table if they do not yet exist.

    Idempotent: safe to call before every tool invocation (``call_tool`` does so).
    Uses ``CREATE TABLE IF NOT EXISTS`` so concurrent callers do not conflict.

    Side effects:
        - Creates ``settings.tasks_dir`` (and parents) on disk.
        - Creates/ensures the ``tasks`` table in the SQLite DB at ``_db_path()``.

    Returns:
        None.
    """
    settings.tasks_dir.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_db_path()) as db:
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
                routing TEXT
            )"""
        )
        await db.commit()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the MCP tool catalog advertised to clients.

    Invoked by the MCP runtime in response to a ``tools/list`` request. Each
    returned ``Tool`` carries a JSON Schema (``inputSchema``) describing its
    arguments; ``call_tool`` dispatches on the tool ``name``.

    Returns:
        list[Tool]: The five Hyperion tools (run, status, artifact, approve, feedback).
    """
    return [
        Tool(
            name="hyperion_run",
            description=(
                "Submit a task to Hyperion (a multi-agent system). "
                "It plans, researches, and synthesizes a result. "
                "Returns a task_id immediately; use hyperion_status to poll."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Natural-language task description"},
                    "workflow": {
                        "type": "string",
                        "description": (
                            "Optional workflow (DAG) id to run. Omit to use the server "
                            "default. List ids via the /workflows API or the UI."
                        ),
                    },
                    "workflow_prompt": {
                        "type": "string",
                        "description": (
                            "Optional plain-language description of how the agents should "
                            "work together (e.g. 'research, then have the critic review, then "
                            "synthesize'). Compiled into an ad-hoc workflow DAG and run. "
                            "Ignored if 'workflow' is given."
                        ),
                    },
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="hyperion_status",
            description="Get the status and result of a Hyperion task.",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        ),
        Tool(
            name="hyperion_artifact",
            description="Retrieve the content of an artifact file from a completed Hyperion task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "name": {"type": "string", "default": "result.md"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="hyperion_approve",
            description=(
                "Resume a task paused at the plan-approval gate. action ∈ "
                "{approve, revise, reject}. For 'approve' you may pass chosen_option "
                "(a plan option id); for 'revise' pass edits (feedback for the planner)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["approve", "revise", "reject"],
                        "default": "approve",
                    },
                    "chosen_option": {"type": "string"},
                    "edits": {"type": "string"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="hyperion_feedback",
            description=(
                "Send free-text feedback to a Hyperion task. A running task drains it "
                "between stages; a task paused on a question (awaiting_input) treats the "
                "message as the answer and resumes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["task_id", "message"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch an MCP ``tools/call`` request to the matching Hyperion handler.

    Ensures the task DB exists, then branches on ``name`` to the appropriate
    behavior. Run/status/artifact are served locally; approve/feedback are proxied
    to the FastAPI service (see ``_post_api``).

    Args:
        name: The tool name (one of the five advertised by ``list_tools``).
        arguments: Decoded JSON arguments conforming to that tool's ``inputSchema``.

    Returns:
        list[TextContent]: A single text block carrying the human-readable result.
            Errors (unknown tool, not-found, invalid path, API failure) are returned
            as text content rather than raised, so the client always gets a message.

    Raises:
        KeyError: If a required argument (e.g. ``arguments["task"]``) is missing.

    Side effects:
        - Ensures/creates the SQLite schema (``_ensure_db``).
        - For ``hyperion_run``: inserts a row, seeds ``_PROGRESS``, and schedules a
          detached background task that executes the crew and updates the DB.
    """
    await _ensure_db()

    if name == "hyperion_run":
        # Short 8-char id keeps task references human-friendly; collision risk is
        # acceptable for the expected task volume.
        task_id = str(uuid.uuid4())[:8]
        request = arguments["task"]
        workflow = arguments.get("workflow")

        # Compile a plain-language orchestration prompt into an ad-hoc workflow
        # (req 4.1), unless an explicit workflow id was given. Persist it so the
        # run, HITL resume, and the trace UI can all reload it by id.
        workflow_prompt = arguments.get("workflow_prompt")
        if not workflow and workflow_prompt:
            from hyperion.agents.registry import load_all_agents
            from hyperion.crews.compiler import WorkflowCompileError, compile_workflow
            from hyperion.crews.workflows import save_workflow

            try:
                compiled = compile_workflow(workflow_prompt, load_all_agents())
                save_workflow(compiled)
                workflow = compiled.id
            except WorkflowCompileError as exc:
                return [TextContent(type="text", text=f"Could not build a workflow: {exc}")]

        async with aiosqlite.connect(_db_path()) as db:
            await db.execute(
                "INSERT INTO tasks (task_id, status, request) VALUES (?,?,?)",
                (task_id, "queued", request),
            )
            await db.commit()

        _PROGRESS[task_id] = []

        def _progress(line: str) -> None:
            """Append a progress line for this task to the in-memory buffer.

            Passed to ``run_crew`` as its ``progress_callback``. Synchronous and
            side-effect-only; surfaced later via ``hyperion_status``.
            """
            _PROGRESS.setdefault(task_id, []).append(line)

        async def _run():
            """Execute the crew in the background and persist its final outcome.

            Flips the task to ``running``, awaits ``run_crew``, then writes the
            resulting status/error/result_path back to the DB. Runs as a detached
            asyncio task so ``hyperion_run`` can return the task_id immediately.
            """
            async with aiosqlite.connect(_db_path()) as db:
                await db.execute(
                    "UPDATE tasks SET status='running', updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                    (task_id,),
                )
                await db.commit()
            result = await run_crew(
                task_id=task_id, request=request, progress_callback=_progress,
                workflow=workflow,
            )
            async with aiosqlite.connect(_db_path()) as db:
                await db.execute(
                    "UPDATE tasks SET status=?, error=?, result_path=?, updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                    (result["status"], result.get("error"), result.get("result_path"), task_id),
                )
                await db.commit()

        # Fire-and-forget: return immediately while the crew runs in the background.
        asyncio.create_task(_run())
        return [TextContent(type="text", text=f"Task submitted. task_id={task_id}")]

    elif name == "hyperion_status":
        task_id = arguments["task_id"]
        async with aiosqlite.connect(_db_path()) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT task_id, status, error, result_path FROM tasks WHERE task_id=?",
                (task_id,),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return [TextContent(type="text", text=f"Task {task_id!r} not found.")]
        progress = _PROGRESS.get(task_id, [])
        # Show only the last 20 progress lines to keep the response compact.
        text = (
            f"task_id: {row['task_id']}\n"
            f"status:  {row['status']}\n"
            f"error:   {row['error'] or 'none'}\n"
            f"result:  {row['result_path'] or 'pending'}\n"
            f"progress ({len(progress)} lines):\n"
            + "\n".join(f"  {l}" for l in progress[-20:])
        )
        return [TextContent(type="text", text=text)]

    elif name == "hyperion_artifact":
        task_id = arguments["task_id"]
        artifact_name = arguments.get("name", "result.md")
        path = (settings.tasks_dir / task_id / "artifacts" / artifact_name).resolve()
        base = (settings.tasks_dir / task_id).resolve()
        # Path-traversal guard: reject names like "../../etc/passwd" that resolve
        # outside this task's directory tree before reading anything from disk.
        if not path.is_relative_to(base):
            return [TextContent(type="text", text="Error: invalid path.")]
        if not path.exists():
            return [TextContent(type="text", text=f"Artifact '{artifact_name}' not found for task {task_id}.")]
        return [TextContent(type="text", text=path.read_text(encoding="utf-8"))]

    elif name == "hyperion_approve":
        task_id = arguments["task_id"]
        payload = {
            "action": arguments.get("action", "approve"),
            "chosen_option": arguments.get("chosen_option"),
            "edits": arguments.get("edits"),
        }
        return await _post_api(f"/tasks/{task_id}/approve", payload)

    elif name == "hyperion_feedback":
        task_id = arguments["task_id"]
        return await _post_api(
            f"/tasks/{task_id}/feedback", {"message": arguments["message"]}
        )

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _post_api(path: str, payload: dict) -> list[TextContent]:
    """POST to the Hyperion FastAPI service, which owns task orchestration.

    Used by ``hyperion_approve`` and ``hyperion_feedback`` to delegate HITL
    (plan approval, feedback/answer) operations to the FastAPI control plane,
    which holds the resume tokens and pause/resume state this process does not.

    Args:
        path: API path beginning with "/" (e.g. "/tasks/<id>/approve").
        payload: JSON-serializable request body.

    Returns:
        list[TextContent]: A text block reporting the returned task_id/status, or
            an error message. All failures (HTTP >= 400 and exceptions) are caught
            and returned as text rather than raised.

    Notes:
        - ``httpx`` is imported lazily to keep it out of the module import path.
        - Request timeout is fixed at 10 seconds.
    """
    import httpx

    url = settings.hyperion_api_url.rstrip("/") + path
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            return [TextContent(type="text", text=f"Error {resp.status_code}: {resp.text}")]
        data = resp.json()
        return [TextContent(
            type="text",
            text=f"task_id: {data.get('task_id')}\nstatus:  {data.get('status')}",
        )]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error contacting Hyperion API: {exc}")]


class _MCPApp:
    """Thin ASGI wrapper around StreamableHTTPSessionManager.handle_request.

    Adapts the MCP session manager into a plain ASGI callable so it can be mounted
    as a Starlette ``Route`` endpoint (Starlette expects an ASGI app, not a bound
    coroutine method).
    """

    def __init__(self, sm: StreamableHTTPSessionManager) -> None:
        """Store the session manager that will service each request.

        Args:
            sm: The streamable-HTTP session manager driving the MCP ``server``.
        """
        self._sm = sm

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entrypoint: forward the request to the session manager.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        await self._sm.handle_request(scope, receive, send)


def make_starlette_app() -> Starlette:
    """Build the Starlette ASGI app that serves the MCP endpoint at ``/mcp``.

    Wires a stateless ``StreamableHTTPSessionManager`` around the module-level
    ``server`` and exposes it via ``_MCPApp``. The session manager is started and
    stopped through the app ``lifespan`` so its background machinery runs for the
    lifetime of the server.

    Returns:
        Starlette: A configured ASGI application with the ``/mcp`` route and lifespan.
    """
    session_manager = StreamableHTTPSessionManager(
        app=server,
        stateless=True,  # no persistent session state needed between tool calls
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Run the session manager for the app's lifetime (startup→shutdown)."""
        async with session_manager.run():
            yield

    return Starlette(
        routes=[Route("/mcp", endpoint=_MCPApp(session_manager))],
        lifespan=lifespan,
    )


def main() -> None:
    """Run the MCP server with uvicorn (console/CLI entrypoint).

    Binds to 0.0.0.0:4101 so it is reachable from other containers in the
    Docker stack, not just localhost. ``uvicorn`` is imported lazily.
    """
    import uvicorn

    starlette_app = make_starlette_app()
    uvicorn.run(starlette_app, host="0.0.0.0", port=4101, log_level="info")


if __name__ == "__main__":
    main()
