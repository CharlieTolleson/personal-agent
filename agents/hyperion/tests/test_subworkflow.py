"""Sub-workflow node tests (workflows calling workflows).

A workflow node whose ``kind == "subworkflow"`` runs another whole workflow
(named by ``node.workflow``) as a single composable step, handing that child
workflow's final report back to the parent like a normal work output.

What is under test:
  - ``validate_workflow`` (in ``hyperion.crews.workflows``): the exactly-one-of
    schema rule (a subworkflow node sets ``workflow`` and no ``agent``; every
    other kind sets ``agent`` and no ``workflow``), dangling ``workflow`` refs,
    and cross-workflow cycle detection (A → B → A and the self-reference A → A).
  - ``runner._run_subworkflow`` / ``runner._execute_workflow`` dispatch: a parent
    workflow with a subworkflow node runs the child under a derived task id, copies
    the child's ``artifacts/result.md`` into the parent's ``notes/<node_id>.md``,
    and records the child run id under ``routing["subworkflows"]``.
  - The runtime depth cap (``settings.cap_subworkflow_depth``) backstops nesting.

Design / test-isolation notes:
  - The crew stages are mocked (``_run_stage`` / ``build_agent`` / the task
    builders) so these tests run with NO LLM calls and no agent execution; the
    fake ``_run_stage`` writes realistic ``notes/`` + ``artifacts/result.md`` so
    the empty-stage checks and the sub-workflow hand-off behave normally.
  - ``settings.tasks_dir`` is patched to ``tmp_path`` so runs read/write under a
    throwaway directory, and ``run_meta_tasks`` is mocked away (it would call an LLM).
  - Async tests use the ``anyio`` marker on the asyncio backend.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperion.crews import runner
from hyperion.crews.workflows import (
    WorkflowNode,
    WorkflowRecord,
    validate_workflow,
)


@pytest.fixture
def anyio_backend():
    """Run ``@pytest.mark.anyio`` tests on asyncio only (the runner is asyncio-based)."""
    return "asyncio"


# ---------------------------------------------------------------------------
# Schema validation: exactly-one-of (agent XOR workflow)
# ---------------------------------------------------------------------------


def test_subworkflow_node_requires_workflow_ref():
    """A subworkflow node with no ``workflow`` set fails validation."""
    rec = WorkflowRecord(
        id="wf", name="wf",
        nodes=[WorkflowNode(id="s", kind="subworkflow", upstream=[])],
    )
    with pytest.raises(ValueError, match="must set 'workflow'"):
        validate_workflow(rec, known_agent_ids=set(), known_workflow_ids={"child"})


def test_subworkflow_node_must_not_set_agent():
    """A subworkflow node may not also carry an agent (the two are exclusive)."""
    rec = WorkflowRecord(
        id="wf", name="wf",
        nodes=[WorkflowNode(id="s", kind="subworkflow", workflow="child",
                            agent="planner", upstream=[])],
    )
    with pytest.raises(ValueError, match="must not also set 'agent'"):
        validate_workflow(rec, {"planner"}, {"child"})


def test_agent_node_must_not_set_workflow():
    """A non-subworkflow node that sets ``workflow`` is rejected (wrong kind)."""
    rec = WorkflowRecord(
        id="wf", name="wf",
        nodes=[WorkflowNode(id="a", kind="work", agent="planner",
                            workflow="child", upstream=[])],
    )
    with pytest.raises(ValueError, match="use kind 'subworkflow'"):
        validate_workflow(rec, {"planner"}, {"child"})


def test_unknown_workflow_ref_rejected():
    """A subworkflow node referencing an unknown workflow id fails when the known
    set is supplied."""
    rec = WorkflowRecord(
        id="wf", name="wf",
        nodes=[WorkflowNode(id="s", kind="subworkflow", workflow="ghost", upstream=[])],
    )
    with pytest.raises(ValueError, match="unknown workflow 'ghost'"):
        validate_workflow(rec, set(), known_workflow_ids={"child"})


def test_unknown_workflow_ref_skipped_when_set_is_none():
    """With ``known_workflow_ids=None`` the existence check is skipped (structure
    only) — the node still validates as long as it is shaped correctly."""
    rec = WorkflowRecord(
        id="wf", name="wf",
        nodes=[WorkflowNode(id="s", kind="subworkflow", workflow="whatever", upstream=[])],
    )
    validate_workflow(rec, set(), known_workflow_ids=None)  # no raise


def test_valid_subworkflow_passes():
    """A well-formed parent (subworkflow node → agent node) validates cleanly."""
    rec = WorkflowRecord(
        id="parent", name="parent",
        nodes=[
            WorkflowNode(id="s", kind="subworkflow", workflow="child", upstream=[]),
            WorkflowNode(id="write", kind="synthesize", agent="synthesizer", upstream=["s"]),
        ],
    )
    validate_workflow(rec, {"synthesizer"}, {"child"})  # no raise


# ---------------------------------------------------------------------------
# Cross-workflow cycle detection
# ---------------------------------------------------------------------------


def test_self_referential_subworkflow_is_a_cycle():
    """A workflow whose subworkflow node points at itself is a 1-cycle."""
    rec = WorkflowRecord(
        id="loop", name="loop",
        nodes=[WorkflowNode(id="s", kind="subworkflow", workflow="loop", upstream=[])],
    )

    def resolve(wid):
        return rec

    with pytest.raises(ValueError, match="cycle detected"):
        validate_workflow(rec, set(), {"loop"}, resolve)


def test_mutual_subworkflow_cycle_detected():
    """A → B → A across two workflows is detected via the resolver."""
    a = WorkflowRecord(
        id="a", name="a",
        nodes=[WorkflowNode(id="s", kind="subworkflow", workflow="b", upstream=[])],
    )
    b = WorkflowRecord(
        id="b", name="b",
        nodes=[WorkflowNode(id="s", kind="subworkflow", workflow="a", upstream=[])],
    )
    registry = {"a": a, "b": b}

    with pytest.raises(ValueError, match="cycle detected"):
        validate_workflow(a, set(), {"a", "b"}, registry.__getitem__)


def test_acyclic_subworkflow_chain_passes():
    """A → B (B references no one) validates — a chain is fine, only cycles fail."""
    a = WorkflowRecord(
        id="a", name="a",
        nodes=[WorkflowNode(id="s", kind="subworkflow", workflow="b", upstream=[])],
    )
    b = WorkflowRecord(
        id="b", name="b",
        nodes=[WorkflowNode(id="w", kind="work", agent="researcher", upstream=[])],
    )
    registry = {"a": a, "b": b}
    validate_workflow(a, {"researcher"}, {"a", "b"}, registry.__getitem__)  # no raise


# ---------------------------------------------------------------------------
# Runner: end-to-end sub-workflow execution + hand-off
# ---------------------------------------------------------------------------


def _parent_workflow() -> WorkflowRecord:
    """Parent: a subworkflow node (``subrun`` → child) then a synthesize node."""
    return WorkflowRecord(
        id="parent-wf", name="parent",
        nodes=[
            WorkflowNode(id="subrun", kind="subworkflow", workflow="child-wf", upstream=[]),
            WorkflowNode(id="final", kind="synthesize", agent="synthesizer", upstream=["subrun"]),
        ],
    )


def _child_workflow() -> WorkflowRecord:
    """Child: a single work node, so the child run is trivially short."""
    return WorkflowRecord(
        id="child-wf", name="child",
        nodes=[WorkflowNode(id="dowork", kind="work", agent="researcher", upstream=[])],
    )


@contextlib.contextmanager
def _mock_subworkflow_crew(tasks_dir):
    """Patch the agent-run helpers, workflow resolver, and meta pipeline for LLM-free runs.

    The fake ``_run_stage`` writes ``notes/<stage>.md`` and ``artifacts/result.md``
    under each run's workspace so the empty-stage checks and the sub-workflow
    hand-off (which reads the child's ``result.md``) behave realistically.
    """
    from hyperion.config import settings

    async def _fake_stage(task_id, request, stage, agents, tasks, cb, deadline):
        base = settings.tasks_dir / task_id
        (base / "notes").mkdir(parents=True, exist_ok=True)
        (base / "artifacts").mkdir(parents=True, exist_ok=True)
        (base / "notes" / f"{stage}.md").write_text(f"notes {stage}", encoding="utf-8")
        (base / "artifacts" / "result.md").write_text(
            f"result from {stage} in {task_id}", encoding="utf-8"
        )

    registry = {"parent-wf": _parent_workflow(), "child-wf": _child_workflow()}

    with patch.object(settings, "tasks_dir", tasks_dir), \
         patch.object(runner, "build_agent", MagicMock()), \
         patch.object(runner, "load_agent", MagicMock()), \
         patch.object(runner, "discover_context", MagicMock(return_value=None)), \
         patch.object(runner, "_plan_task", MagicMock()), \
         patch.object(runner, "_work_task", MagicMock()), \
         patch.object(runner, "_synthesize_task", MagicMock()), \
         patch.object(runner, "_run_stage", new=_fake_stage), \
         patch("hyperion.crews.workflows.resolve_workflow",
               new=MagicMock(side_effect=lambda wid: registry[wid])), \
         patch("hyperion.server.meta_tasks.run_meta_tasks", new=AsyncMock()):
        yield


@pytest.mark.anyio
async def test_subworkflow_runs_child_and_hands_off_result(tmp_path):
    """A parent run executes its subworkflow node by running the child workflow
    under a derived task id, then copies the child's result into the parent's
    notes/ and records the child run id in routing."""
    with _mock_subworkflow_crew(tmp_path):
        result = await runner.run_task("parent1", "do the thing", workflow="parent-wf")

    assert result["status"] == "done"
    # The child ran under the derived id and its report was handed to the parent.
    handed = tmp_path / "parent1" / "notes" / "subrun.md"
    assert handed.exists()
    assert handed.read_text(encoding="utf-8") == "result from dowork in parent1__subrun"
    # Routing records the child run id for trace drill-down.
    assert result["routing"]["subworkflows"] == {"subrun": "parent1__subrun"}
    # The child got its own isolated workspace.
    assert (tmp_path / "parent1__subrun" / "artifacts" / "result.md").exists()


@pytest.mark.anyio
async def test_subworkflow_depth_cap_aborts(tmp_path):
    """``_run_subworkflow`` raises CapExceeded once nesting would exceed the cap,
    before even resolving the child (the depth guard is the outermost check)."""
    from hyperion.config import settings

    node = WorkflowNode(id="s", kind="subworkflow", workflow="child-wf", upstream=[])
    with patch.object(settings, "cap_subworkflow_depth", 2):
        with pytest.raises(runner.CapExceeded, match="depth cap"):
            # depth=2 → child would be depth 3 > cap 2.
            await runner._run_subworkflow(
                "t", node, "req", deadline=9e18, wall=900, caps={},
                progress_callback=None, depth=2,
            )
