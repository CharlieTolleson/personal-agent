"""
Workflow registry — free-form, per-agent DAG definitions.

A workflow is a named directed-acyclic graph of *nodes*; each node references an
agent (by id) and lists its upstream node ids. The runner topo-sorts the nodes and
executes them in dependency order, so a workflow can be any number of steps and an
agent can appear in more than one node (node id is distinct from agent id).

Workflows are JSON records under ``config/workflows/<id>.json`` (git-tracked,
volume-mounted like agents). They supersede the old fixed plan→work→synthesize
pipeline; the seeded ``claude-workflow`` reproduces it exactly so behavior is
unchanged when no workflow is chosen.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from hyperion.config import settings

# The role a node plays within its workflow. Drives the default task instructions
# (when a node has no explicit ``instruction``) and the HITL gate/revise flow:
# planning happens in "plan" nodes, the final report is written by "synthesize"
# nodes, everything else is "work". This used to live on the agent record as
# ``stage``; it now lives on the node so the same agent can play different roles in
# different workflows.
#
# "subworkflow" is special: instead of running an agent, the node runs another
# *workflow* (referenced by ``WorkflowNode.workflow``) as a single composable step,
# handing that child workflow's final report back to the parent like a work output.
# A subworkflow node sets ``workflow`` and leaves ``agent`` unset; every other kind
# sets ``agent`` and leaves ``workflow`` unset (enforced by ``validate_workflow``).
NodeKind = Literal["plan", "work", "synthesize", "subworkflow"]

# Identifier guard for workflow ids and node ids: lowercase alnum start, then
# alnum / underscore / hyphen. Used both as a filename-safety check (ids become
# JSON filenames in config/workflows/) and as a structural-validation rule.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class NodeWhen(BaseModel):
    """Optional conditional-firing rule for a node.

    When present, the node runs only if the planner-classified ``task_type`` of the
    run is listed in ``task_types`` (e.g. a developer node that fires only on
    ``code`` tasks). An empty/omitted ``when`` means the node always fires.
    """

    task_types: list[str] = Field(default_factory=list)


class NodePosition(BaseModel):
    """Canvas coordinates for the graphical workflow builder (UI-only metadata).

    Persisted purely so a hand-arranged layout in the React Flow workflow editor
    survives reloads. The runner ignores it entirely — execution order comes from
    the ``upstream`` DAG, never from geometry. Optional: nodes authored before the
    graphical editor (or via the API) have no position and are auto-laid-out by
    the UI on first open.
    """

    x: float
    y: float


class WorkflowNode(BaseModel):
    """A single executable step in a workflow DAG.

    A node usually binds an agent to a position in the graph. Because ``id`` is
    distinct from ``agent``, the same agent can appear in multiple nodes (e.g. a
    critic that reviews two different upstream branches).

    A node with ``kind == "subworkflow"`` is the exception: it runs another whole
    workflow (named by ``workflow``) as one composable step and leaves ``agent``
    unset. Exactly one of ``agent`` / ``workflow`` is set; ``validate_workflow``
    enforces this and that ``kind == "subworkflow"`` iff ``workflow`` is set.

    Attributes:
        id: Node slug, unique within the workflow. Used as the topo-sort key and
            as the target of other nodes' ``upstream`` references.
        agent: Id of the agent record this node runs (None for subworkflow nodes).
        workflow: Id of the child workflow this node runs (set only when
            ``kind == "subworkflow"``; None otherwise).
        kind: The role this node plays (plan / work / synthesize / subworkflow);
            drives the default task instructions and the HITL gate/revise flow.
        upstream: Ids of nodes that must complete before this node may run. An
            empty list means the node is a graph root with no dependencies.
        gate_before: When True, the runner pauses for human approval (HITL)
            before executing this node.
        instruction: Optional explicit task description that overrides the
            kind-derived default instruction for this node. For a subworkflow
            node it becomes the child run's request (parent request when unset).
        when: Optional conditional-firing rule; when set, the node runs only for
            the listed task types.
        position: Optional canvas coordinates for the graphical editor (UI-only;
            ignored by the runner). Persists hand-arranged layouts.
    """

    id: str                                   # node slug, unique within the workflow
    agent: Optional[str] = None               # agent record id this node runs (None for subworkflow)
    workflow: Optional[str] = None            # child workflow id (subworkflow nodes only)
    kind: NodeKind = "work"                   # role within the workflow (plan/work/synthesize/subworkflow)
    upstream: list[str] = Field(default_factory=list)  # node ids that must finish first
    gate_before: bool = False                 # pause for human approval before this node (HITL)
    instruction: Optional[str] = None         # overrides the kind-derived task description
    when: Optional[NodeWhen] = None           # conditional firing by task_type
    position: Optional[NodePosition] = None   # canvas coords for the graphical editor (UI-only)


class WorkflowRecord(BaseModel):
    """A complete, persistable workflow definition (one JSON file on disk).

    Attributes:
        id: Workflow slug; also the on-disk filename stem
            (``config/workflows/<id>.json``).
        name: Human-readable display name.
        description: Optional longer description shown in the UI.
        nodes: The DAG nodes. Validated/ordered by ``validate_workflow`` and
            ``topo_sort`` before execution.
    """

    id: str
    name: str
    description: str = ""
    nodes: list[WorkflowNode] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _workflows_dir():
    """Return the directory holding workflow JSON records.

    Resolved from ``settings.config_dir`` so it tracks the (volume-mounted)
    config root rather than being hard-coded.
    """
    return settings.config_dir / "workflows"


def _record_path(workflow_id: str):
    """Return the on-disk path for a workflow id.

    Args:
        workflow_id: The workflow slug.

    Returns:
        Path to ``<config>/workflows/<workflow_id>.json``.

    Raises:
        ValueError: If ``workflow_id`` is not a valid slug. This guards against
            path traversal / unsafe filenames since the id becomes a filename.
    """
    if not _SLUG_RE.match(workflow_id):
        raise ValueError(f"Invalid workflow id {workflow_id!r} (must be a slug)")
    return _workflows_dir() / f"{workflow_id}.json"


def load_workflow(workflow_id: str) -> WorkflowRecord:
    """Load and parse a single workflow record by id.

    Args:
        workflow_id: The workflow slug to load.

    Returns:
        The parsed ``WorkflowRecord``.

    Raises:
        ValueError: If ``workflow_id`` is not a valid slug.
        FileNotFoundError: If no record file exists for that id.
        pydantic.ValidationError: If the file's JSON does not match the schema.
    """
    path = _record_path(workflow_id)
    if not path.exists():
        raise FileNotFoundError(f"No workflow record for id {workflow_id!r} at {path}")
    return WorkflowRecord.model_validate_json(path.read_text(encoding="utf-8"))


def load_all_workflows() -> list[WorkflowRecord]:
    """Load every workflow record on disk, sorted by filename.

    Returns:
        A list of parsed ``WorkflowRecord`` objects (empty if the workflows
        directory does not exist).

    Raises:
        pydantic.ValidationError: If any record file fails schema validation.
    """
    d = _workflows_dir()
    if not d.exists():
        return []
    return [
        WorkflowRecord.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(d.glob("*.json"))
    ]


def save_workflow(record: WorkflowRecord) -> None:
    """Persist a workflow record to disk as pretty-printed JSON.

    Creates the workflows directory if needed and writes
    ``<config>/workflows/<record.id>.json`` (2-space indent, non-ASCII
    preserved, trailing newline for clean diffs).

    Args:
        record: The workflow to write.

    Raises:
        ValueError: If ``record.id`` is not a valid slug.

    Side effects:
        Creates the workflows directory and writes/overwrites the record file.
    """
    d = _workflows_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = _record_path(record.id)
    path.write_text(
        json.dumps(record.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def delete_workflow(workflow_id: str) -> None:
    """Delete a workflow record from disk if it exists.

    Args:
        workflow_id: The workflow slug to delete.

    Raises:
        ValueError: If ``workflow_id`` is not a valid slug.

    Side effects:
        Removes the record file. A no-op (no error) if the file is absent.
    """
    path = _record_path(workflow_id)
    if path.exists():
        path.unlink()


def get_default_workflow() -> WorkflowRecord:
    """The workflow used when a task does not name one.

    Resolution order: the persisted ``settings.default_workflow`` id, else the
    seeded ``claude-workflow``, else the first workflow on disk. Raises if the
    config dir has no workflows at all (a broken install)."""
    wanted = settings.default_workflow
    try:
        return load_workflow(wanted)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return load_workflow("claude-workflow")
    except (FileNotFoundError, ValueError):
        pass
    all_wf = load_all_workflows()
    if not all_wf:
        raise FileNotFoundError(
            f"No workflows found in {_workflows_dir()} (expected at least claude-workflow.json)"
        )
    return all_wf[0]


def resolve_workflow(workflow_id: Optional[str]) -> WorkflowRecord:
    """A named workflow, or the default when ``workflow_id`` is None/empty."""
    if workflow_id:
        return load_workflow(workflow_id)
    return get_default_workflow()


# ---------------------------------------------------------------------------
# Validation + topo-sort
# ---------------------------------------------------------------------------


def topo_sort(nodes: list[WorkflowNode]) -> list[WorkflowNode]:
    """Order nodes so every node follows its upstream deps. Tie-broken by id for
    determinism. Raises ValueError on a cycle or a dangling upstream reference."""
    # Kahn's algorithm. in-degree = number of upstream deps each node waits on.
    by_id = {n.id: n for n in nodes}
    indeg: dict[str, int] = {n.id: 0 for n in nodes}
    for n in nodes:
        for u in n.upstream:
            if u not in by_id:
                raise ValueError(f"Node {n.id!r} lists unknown upstream {u!r}")
            indeg[n.id] += 1

    # Roots (no deps) start ready; sorting here + below makes the order
    # deterministic when several nodes are simultaneously runnable.
    ready = sorted([nid for nid, d in indeg.items() if d == 0])
    out: list[str] = []
    while ready:
        nid = ready.pop(0)
        out.append(nid)
        # Releasing nid may unblock downstream nodes: decrement their in-degree
        # and enqueue any that have now had all upstreams satisfied.
        for other in nodes:
            if nid in other.upstream:
                indeg[other.id] -= 1
                if indeg[other.id] == 0:
                    ready.append(other.id)
        ready.sort()
    # Fewer emitted nodes than inputs => at least one cycle kept indeg > 0.
    if len(out) != len(nodes):
        remaining = sorted(set(by_id) - set(out))
        raise ValueError(f"Cycle detected in workflow upstream edges among {remaining}")
    return [by_id[nid] for nid in out]


def validate_workflow(
    record: WorkflowRecord,
    known_agent_ids: set[str],
    known_workflow_ids: Optional[set[str]] = None,
    resolve: Optional[Callable[[str], "WorkflowRecord"]] = None,
) -> None:
    """Structural validation of a workflow record.

    Checks: slug id, >=1 node, unique node ids, each node is exactly one of an
    agent node or a subworkflow node (``kind == "subworkflow"`` iff ``workflow``
    set, ``agent`` unset), every agent ref is known, upstream refs resolve within
    the workflow, and the node DAG is acyclic.

    Args:
        record: The workflow to validate.
        known_agent_ids: Ids of agents that may be referenced by agent nodes.
        known_workflow_ids: Ids of workflows that may be referenced by subworkflow
            nodes. When None, subworkflow-reference existence is not checked (only
            structure) — pass the real set to reject dangling ``workflow`` refs.
        resolve: Optional loader ``workflow_id -> WorkflowRecord`` used to walk
            subworkflow references for cross-workflow cycle detection. When None,
            cross-workflow cycle detection is skipped (single-workflow validation).

    Raises:
        ValueError: on any structural problem, an unknown agent/workflow
            reference, a dangling upstream ref, a node-level cycle, or a
            cross-workflow subworkflow cycle.
    """
    if not _SLUG_RE.match(record.id):
        raise ValueError(f"Workflow id {record.id!r} must be a slug ([a-z0-9_-])")
    if not record.nodes:
        raise ValueError("A workflow must have at least one node.")

    seen: set[str] = set()
    for node in record.nodes:
        if not _SLUG_RE.match(node.id):
            raise ValueError(f"Node id {node.id!r} must be a slug ([a-z0-9_-])")
        if node.id in seen:
            raise ValueError(f"Duplicate node id {node.id!r} in workflow {record.id!r}")
        seen.add(node.id)
        # Exactly-one-of: a subworkflow node runs a workflow; every other kind
        # runs an agent. Mixing the two (or setting neither) is a schema error.
        if node.kind == "subworkflow":
            if not node.workflow:
                raise ValueError(
                    f"Sub-workflow node {node.id!r} must set 'workflow'"
                )
            if node.agent:
                raise ValueError(
                    f"Sub-workflow node {node.id!r} must not also set 'agent'"
                )
            if known_workflow_ids is not None and node.workflow not in known_workflow_ids:
                raise ValueError(
                    f"Node {node.id!r} references unknown workflow {node.workflow!r}"
                )
        else:
            if node.workflow:
                raise ValueError(
                    f"Node {node.id!r} sets 'workflow' but kind is {node.kind!r} "
                    f"(use kind 'subworkflow' to run a workflow)"
                )
            if not node.agent:
                raise ValueError(f"Node {node.id!r} must set 'agent'")
            if node.agent not in known_agent_ids:
                raise ValueError(
                    f"Node {node.id!r} references unknown agent {node.agent!r}"
                )
    for node in record.nodes:
        for u in node.upstream:
            if u not in seen:
                raise ValueError(
                    f"Node {node.id!r} lists upstream {u!r} which is not a node in this workflow"
                )
    # Raises on a node-level cycle within this workflow's own DAG.
    topo_sort(record.nodes)
    # Raises on a cross-workflow cycle (A -> B -> A) reachable via subworkflow refs.
    if resolve is not None:
        _check_subworkflow_acyclic(record, resolve)


def _check_subworkflow_acyclic(
    record: WorkflowRecord, resolve: Callable[[str], "WorkflowRecord"]
) -> None:
    """Depth-first walk over subworkflow references; raise on any cycle through
    ``record``. ``record`` itself is used for its own id (it may be unsaved/edited),
    transitive references are loaded via ``resolve``. References that fail to load
    are skipped — their own validation is responsible for them, and a missing ref
    cannot extend a cycle."""

    def children(rec: WorkflowRecord) -> list[str]:
        return [n.workflow for n in rec.nodes if n.kind == "subworkflow" and n.workflow]

    def walk(wf_id: str, path: tuple[str, ...]) -> None:
        if wf_id in path:
            raise ValueError(
                "Sub-workflow cycle detected: " + " -> ".join((*path, wf_id))
            )
        if wf_id == record.id:
            rec: Optional[WorkflowRecord] = record
        else:
            try:
                rec = resolve(wf_id)
            except (FileNotFoundError, ValueError):
                return  # can't recurse into an unresolvable ref; not our cycle to find
        for child in children(rec):
            walk(child, (*path, wf_id))

    walk(record.id, ())
