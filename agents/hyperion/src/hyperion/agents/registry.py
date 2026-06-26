"""
Agent + tool registry — the data-driven substrate (Phase 0).

Agents are JSON records under ``config/agents/<id>.json`` (git-tracked, volume-mounted
so UI edits survive container rebuilds). Tools are a named registry; agent records
reference tools by name and the runner resolves them per-task.

Every later-phase capability plugs in here as a registry entry or a record edit —
never as a code edit to a hardcoded agent factory. See the implementation plan §4.1/§4.6.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from hyperion.agent_loop import ToolSpec
from hyperion.config import settings
from hyperion.feedback import AskUserTool, ReadHumanFeedbackTool
from hyperion.memory.context_store import ContextGetTool, ContextPutTool, RecallSimilarTasksTool
from hyperion.tools.notion import NotionWriteTool
from hyperion.tools.second_brain import SecondBrainTool
from hyperion.tools.web_search import WebSearchTool
from hyperion.tools.workspace import WorkspaceListTool, WorkspaceReadTool, WorkspaceWriteTool

# Valid agent-id format: lowercase slug (letters/digits to start, then letters/digits/_/-).
# Enforced everywhere an id is turned into a filesystem path to prevent path traversal
# and to keep record filenames portable.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


# ---------------------------------------------------------------------------
# Tool registry — name -> factory(task_id) -> tool instance
#
# Each factory returns a plain tool object exposing ``name``, ``description``,
# ``parameters`` (a JSON schema) and ``_run`` (the callable). ``build_tools``
# wraps each into a ``ToolSpec`` for the owned agent loop (see ``agent_loop``).
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable[[str], Any]] = {
    "workspace_read": lambda task_id: WorkspaceReadTool(task_id=task_id),
    "workspace_write": lambda task_id: WorkspaceWriteTool(task_id=task_id),
    "workspace_list": lambda task_id: WorkspaceListTool(task_id=task_id),
    "web_search": lambda task_id: WebSearchTool(),
    "second_brain": lambda task_id: SecondBrainTool(),
    "context_put": lambda task_id: ContextPutTool(task_id=task_id),
    "context_get": lambda task_id: ContextGetTool(task_id=task_id),
    "recall_similar_tasks": lambda task_id: RecallSimilarTasksTool(task_id=task_id),
    "read_human_feedback": lambda task_id: ReadHumanFeedbackTool(task_id=task_id),
    "ask_user": lambda task_id: AskUserTool(task_id=task_id),
    "notion_write": lambda task_id: NotionWriteTool(),
}


def register_tool(name: str, factory: Callable[[str], Any]) -> None:
    """Register a tool factory. Later phases call this to add new capabilities.

    Args:
        name: Registry key that agent records reference in their ``tools`` list.
        factory: Callable taking the current ``task_id`` and returning a fresh
            tool instance (exposing ``name``/``description``/``parameters``/``_run``).
            Per-task construction lets task-scoped tools (e.g. workspace/context
            tools) bind to the correct task.

    Side effects:
        Mutates the module-global ``TOOL_REGISTRY`` in place. Re-registering an
        existing name overwrites the previous factory.
    """
    TOOL_REGISTRY[name] = factory


def build_tools(names: list[str], task_id: str) -> list[ToolSpec]:
    """Resolve a list of tool names against the registry into ``ToolSpec`` descriptors.

    Args:
        names: Tool names (registry keys) requested by an agent record.
        task_id: The run/task identifier passed to each tool factory so that
            task-scoped tools bind to the right workspace and context store.

    Returns:
        One ``ToolSpec`` per name, in input order, each wrapping the tool's
        ``name``/``description``/``parameters`` and its ``_run`` callable.

    Raises:
        ValueError: If any name is not present in ``TOOL_REGISTRY``.
    """
    tools: list[ToolSpec] = []
    for name in names:
        factory = TOOL_REGISTRY.get(name)
        if factory is None:
            raise ValueError(f"Unknown tool {name!r} (not in TOOL_REGISTRY)")
        inst = factory(task_id)
        tools.append(ToolSpec(
            name=inst.name,
            description=inst.description,
            parameters=inst.parameters,
            fn=inst._run,
        ))
    return tools


# ---------------------------------------------------------------------------
# Agent record schema (the implementation plan §4.1)
# ---------------------------------------------------------------------------


class Thresholds(BaseModel):
    """Optional per-agent guardrails enforced by the runner/usage layer.

    Each ``None`` value means "no limit". Token caps bound a single activation's
    input/output; ``max_activations_per_day`` rate-limits how often the agent runs.
    """

    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_activations_per_day: Optional[int] = None


class AgentRecord(BaseModel):
    """Declarative definition of one agent, persisted as ``config/agents/<id>.json``.

    This is the central data-driven contract of the system: the runner builds an
    agent purely from these fields (role/goal/backstory + model/tool config),
    so new agents are added by writing a record, never by editing code.

    An agent is a pure *persona*: it carries no ordering or activation metadata.
    *When* and *in what order* an agent runs is decided entirely by the workflow
    DAG that references it (see ``hyperion.crews.workflows``). The lone exception is
    ``schedule_cron``, which lets the background scheduler fire an agent on a timer
    independently of any workflow.

    Notable fields:
        id: Slug-format unique identifier; also the JSON filename stem.
        role / goal / backstory: the persona prompt that defines the agent's behavior.
        model_alias / fallback_alias: LiteLLM role aliases (see ``MODEL_ALIASES``)
            or concrete model ids; fallback is used when the primary model fails.
        temperature / top_p / max_tokens / max_iter: LLM and agent-loop tuning.
        tools: Tool registry names resolved via ``build_tools`` at run time.
        schedule_cron: Optional 5-field cron expression; when set, the scheduler
            fires this agent as a standalone task on that timer.
        thresholds: Optional per-agent token/activation guardrails.
    """

    id: str
    name: str
    description: str = ""
    group: str = "core"
    active: bool = True
    role: str
    goal: str
    backstory: str
    model_alias: str = "worker"
    fallback_alias: Optional[str] = None
    temperature: float = 0.1
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_iter: int = 3
    tools: list[str] = Field(default_factory=list)
    schedule_cron: Optional[str] = None
    thresholds: Thresholds = Field(default_factory=Thresholds)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _agents_dir():
    """Return the directory holding agent JSON records (``<config_dir>/agents``).

    Resolved from ``settings.config_dir`` on each call so it tracks the volume-mounted
    config path; the directory may not yet exist (``save_agent`` creates it).
    """
    d = settings.config_dir / "agents"
    return d


def _record_path(agent_id: str):
    """Map an agent id to its on-disk JSON path, validating the id is a safe slug.

    Args:
        agent_id: Candidate agent identifier.

    Returns:
        ``Path`` to ``<agents_dir>/<agent_id>.json``.

    Raises:
        ValueError: If ``agent_id`` is not a valid slug (guards against path
            traversal and unportable filenames).
    """
    if not _SLUG_RE.match(agent_id):
        raise ValueError(f"Invalid agent id {agent_id!r} (must be a slug)")
    return _agents_dir() / f"{agent_id}.json"


def load_agent(agent_id: str) -> AgentRecord:
    """Load and parse a single agent record by id.

    Args:
        agent_id: Slug-format agent identifier.

    Returns:
        The parsed ``AgentRecord``.

    Raises:
        ValueError: If ``agent_id`` is not a valid slug.
        FileNotFoundError: If no record file exists for the id.
        pydantic.ValidationError: If the JSON does not match ``AgentRecord``.
    """
    path = _record_path(agent_id)
    if not path.exists():
        raise FileNotFoundError(f"No agent record for id {agent_id!r} at {path}")
    return AgentRecord.model_validate_json(path.read_text(encoding="utf-8"))


def load_all_agents() -> list[AgentRecord]:
    """All records, sorted by id.

    Returns:
        Every ``AgentRecord`` under the agents dir, ordered by ``id`` for
        determinism. Execution order is no longer a property of the agent set —
        it is defined per-run by the workflow DAG. Returns an empty list if the
        directory does not exist.

    Raises:
        pydantic.ValidationError: If any record file is malformed.
    """
    d = _agents_dir()
    if not d.exists():
        return []
    return [
        AgentRecord.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(d.glob("*.json"))
    ]


def save_agent(record: AgentRecord) -> None:
    """Persist an agent record to disk as pretty-printed, git-friendly JSON.

    Args:
        record: The agent record to write. Its ``id`` determines the filename.

    Side effects:
        Creates the agents directory if missing and writes/overwrites
        ``<id>.json``. The 2-space indent, ``ensure_ascii=False`` and trailing
        newline keep diffs clean for the git-tracked, volume-mounted config.

    Raises:
        ValueError: If ``record.id`` is not a valid slug.
    """
    d = _agents_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = _record_path(record.id)
    path.write_text(
        json.dumps(record.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def delete_agent(agent_id: str) -> None:
    """Delete an agent record file if it exists (no-op when already absent).

    Args:
        agent_id: Slug-format agent identifier.

    Side effects:
        Removes ``<id>.json`` from the agents directory.

    Raises:
        ValueError: If ``agent_id`` is not a valid slug.
    """
    path = _record_path(agent_id)
    if path.exists():
        path.unlink()


def validate_agent(record: AgentRecord) -> None:
    """Structural validation of a single agent record.

    Args:
        record: The agent record to check in isolation.

    Raises:
        ValueError: If the id is not a slug or the record references a tool absent
            from ``TOOL_REGISTRY``.
    """
    if not _SLUG_RE.match(record.id):
        raise ValueError(f"Agent id {record.id!r} must be a slug ([a-z0-9_-])")
    for tool_name in record.tools:
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Agent {record.id!r} references unknown tool {tool_name!r}")


# Recognized LiteLLM role aliases (multi-provider groups). A model_alias is valid
# if it is one of these or a concrete model id the proxy reports.
MODEL_ALIASES: tuple[str, ...] = ("smart", "worker", "cheap", "fast")
