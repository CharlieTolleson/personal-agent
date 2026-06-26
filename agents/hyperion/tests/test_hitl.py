"""HITL (human-in-the-loop) plan-gate tests (Phase 3).

This suite verifies Hyperion's human-in-the-loop control flow around the *plan*
stage of a crew run: the orchestrator can pause after planning and wait for a
human to approve, reject, or revise the plan before any work/synthesis runs.

What is under test (all in ``hyperion.crews.runner``):
  - ``gate(task_id, stage, hitl)`` — the truth table deciding *whether* a given
    stage pauses for human review under a given HITL mode (``off`` / ``plan`` /
    ``full``). Only the ``plan`` stage gates in ``plan`` mode.
  - ``run_task(...)`` — a full run that either pauses at the plan gate
    (``awaiting_approval``) or runs straight through (plan → research →
    synthesize) when HITL is ``off``.
  - ``resume_task(...)`` — the continuation after a pause, exercising the three
    human decisions: ``approve`` (run work + synth, no re-plan), ``reject``
    (fail, run nothing), and ``revise`` (re-plan and re-pause, or force through
    once the revision budget ``runner._MAX_REVISIONS`` is exhausted).

Design / test-isolation notes:
  - The crew stages are mocked (``_run_stage`` / ``build_agent`` and the task
    builders) via :func:`_mock_crew` so these tests exercise the pause/resume
    control flow with NO LLM calls and no agent execution.
  - ``plan.md`` is written to the workspace directly by :func:`_write_plan` to
    stand in for the (mocked) planner's output, so plan parsing /
    option-selection behave realistically.
  - ``settings.tasks_dir`` is patched to ``tmp_path`` in each async test so the
    runner reads/writes plans under a throwaway directory.
  - Async tests use the ``anyio`` marker backed by the asyncio backend (see the
    :func:`anyio_backend` fixture).
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperion.crews import runner
from hyperion.crews.plan_contract import parse_plan
from hyperion.crews.workflows import WorkflowNode, WorkflowRecord


def _default_workflow() -> WorkflowRecord:
    """The canonical 3-node research pipeline these tests are written against.

    Pinning the workflow here keeps the pause/resume tests deterministic and
    independent of whichever workflow happens to be configured as the ambient
    default (``settings.default_workflow``) or how its on-disk gates are set.
    """
    return WorkflowRecord(
        id="research-default",
        name="Research → Synthesize",
        nodes=[
            WorkflowNode(id="plan", agent="planner", kind="plan", upstream=[]),
            WorkflowNode(id="research", agent="researcher", kind="work", upstream=["plan"]),
            WorkflowNode(id="synthesize", agent="synthesizer", kind="synthesize",
                         upstream=["research"]),
        ],
    )


@pytest.fixture
def anyio_backend():
    """Force the ``anyio`` plugin to run ``@pytest.mark.anyio`` tests on asyncio only.

    Returns:
        str: The backend name ``"asyncio"`` (the runner is asyncio-based; this
        avoids also running each async test under trio).
    """
    return "asyncio"


@contextlib.contextmanager
def _mock_crew(stage_impl):
    """Patch out every agent-execution helper so control flow runs LLM-free.

    Replaces the agent builder, context discovery, and the three task builders
    with no-op mocks, and swaps the real ``_run_stage`` coroutine for a caller-
    supplied stand-in. This lets a test drive ``run_task`` / ``resume_task``
    through their full pause/resume logic without any real agents or LLM calls.

    Args:
        stage_impl: The (async) stand-in for ``runner._run_stage``. Pass an
            ``AsyncMock`` to assert on ``await_count``, or a plain coroutine
            function (e.g. one that records executed stage names) to observe
            ordering.

    Yields:
        The patched ``_run_stage`` object (the ``stage_impl`` as installed),
        useful for assertions such as ``stage.await_count``.

    Side effects:
        Temporarily patches attributes on the ``runner`` module for the
        duration of the ``with`` block; all patches are undone on exit.
    """
    with patch.object(runner, "build_agent", MagicMock()), \
         patch.object(runner, "discover_context", MagicMock(return_value=None)), \
         patch.object(runner, "load_agent", MagicMock()), \
         patch("hyperion.crews.workflows.resolve_workflow",
               new=MagicMock(return_value=_default_workflow())), \
         patch.object(runner, "_plan_task", MagicMock()), \
         patch.object(runner, "_work_task", MagicMock()), \
         patch.object(runner, "_synthesize_task", MagicMock()), \
         patch.object(runner, "_run_stage", new=stage_impl) as stage:
        yield stage


def _write_plan(base, task_id, *, task_type="research", with_options=True):
    """Write a minimal ``plan.md`` to stand in for the (mocked) planner output.

    Creates ``<base>/<task_id>/plan.md`` with YAML front-matter that the runner's
    plan parser understands, optionally including a two-option block (ids ``a``
    and ``b``, each with a single subtask ``s1``) so option-selection paths can
    be tested.

    Args:
        base: Base tasks directory (typically ``tmp_path``); the per-task
            subdirectory is created under it.
        task_id: Task identifier; used as the subdirectory name.
        task_type: Value for the plan's ``task_type`` front-matter field
            (drives which downstream stage runs, e.g. ``research``).
        with_options: When True, embed two selectable plan options; when False,
            write a plan with no options block.

    Side effects:
        Creates directories and writes ``plan.md`` (UTF-8) to disk.
    """
    d = base / task_id
    d.mkdir(parents=True, exist_ok=True)
    opts = ""
    if with_options:
        opts = (
            "options:\n"
            "  - id: a\n    summary: shallow pass\n    subtasks:\n      - id: s1\n        description: quick scan\n"
            "  - id: b\n    summary: deep dive\n    subtasks:\n      - id: s1\n        description: thorough\n"
        )
    (d / "plan.md").write_text(
        f"---\ntask_type: {task_type}\nkeywords: [demo]\n{opts}---\n\n# Plan\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# gate() truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage,hitl,expected",
    [
        ("plan", "off", False),
        ("plan", "plan", True),
        ("plan", "full", True),
        ("work", "full", False),       # only the plan stage gates
        ("synthesize", "plan", False),
    ],
)
def test_gate_truth_table(stage, hitl, expected):
    """gate() pauses only for the plan stage in plan/full mode; never otherwise."""
    assert runner.gate("t", stage, hitl) is expected


# ---------------------------------------------------------------------------
# run_task: pause vs. straight-through
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_task_pauses_at_plan_gate(tmp_path):
    """In hitl="plan" mode, run_task stops after planning with awaiting_approval
    and a pending payload, having run exactly the plan stage and nothing more."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path), \
         _mock_crew(AsyncMock()) as stage:
        _write_plan(tmp_path, "t1")
        result = await runner.run_task("t1", "do research", hitl="plan")

    assert result["status"] == "awaiting_approval"
    assert result["pending_stage"] == "research"
    assert result["pending_payload"]["revise_count"] == 0
    assert result["pending_payload"]["hitl"] == "plan"
    # Only the plan stage ran; work/synth are deferred until approval.
    assert stage.await_count == 1


@pytest.mark.anyio
async def test_run_task_straight_through_when_hitl_off(tmp_path):
    """With hitl="off", run_task completes end-to-end (plan→research→synthesize)
    and auto-selects the first plan option without ever pausing."""
    from hyperion.config import settings

    stages: list[str] = []

    async def _fake_stage(task_id, request, stage, agents, tasks, cb, deadline):
        """Stand-in for _run_stage that records each executed stage's name."""
        stages.append(stage)

    with patch.object(settings, "tasks_dir", tmp_path), _mock_crew(_fake_stage):
        _write_plan(tmp_path, "t2")
        result = await runner.run_task("t2", "do research", hitl="off")
        # No gate → first option auto-selected (read inside the tasks_dir patch).
        selected = parse_plan("t2").selected_option

    assert result["status"] == "done"
    assert stages == ["plan", "research", "synthesize"]
    assert selected == "a"


# ---------------------------------------------------------------------------
# resume_task: approve / reject / revise
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_approve_runs_work_and_synth(tmp_path):
    """resume_task(..., "approve") runs work + synthesize (no re-plan), records
    the human-chosen option ("b"), and finishes with status "done"."""
    from hyperion.config import settings

    stages: list[str] = []

    async def _fake_stage(task_id, request, stage, agents, tasks, cb, deadline):
        """Stand-in for _run_stage that records each executed stage's name."""
        stages.append(stage)

    with patch.object(settings, "tasks_dir", tmp_path), _mock_crew(_fake_stage):
        _write_plan(tmp_path, "t3")
        result = await runner.resume_task("t3", "do research", "approve", chosen_option="b")
        selected = parse_plan("t3").selected_option

    assert result["status"] == "done"
    assert stages == ["research", "synthesize"]   # no re-plan on approve
    assert selected == "b"


@pytest.mark.anyio
async def test_resume_reject_fails(tmp_path):
    """resume_task(..., "reject") fails the task with a "rejected" error and runs
    no further stages."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path), \
         _mock_crew(AsyncMock()) as stage:
        _write_plan(tmp_path, "t4")
        result = await runner.resume_task("t4", "do research", "reject")

    assert result["status"] == "failed"
    assert "rejected" in result["error"].lower()
    assert stage.await_count == 0   # nothing runs after a reject


@pytest.mark.anyio
async def test_resume_revise_repauses(tmp_path):
    """resume_task(..., "revise") re-runs the planner once, increments
    revise_count, and pauses again with awaiting_approval (budget not exhausted)."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path), \
         _mock_crew(AsyncMock()) as stage:
        _write_plan(tmp_path, "t5")
        result = await runner.resume_task(
            "t5", "do research", "revise", edits="add more depth", hitl="plan", revise_count=0
        )

    assert result["status"] == "awaiting_approval"
    assert result["pending_payload"]["revise_count"] == 1
    assert stage.await_count == 1   # the planner re-ran once


@pytest.mark.anyio
async def test_resume_revise_force_continues_after_budget(tmp_path):
    """A "revise" that exhausts the revision budget (revise_count = _MAX_REVISIONS
    - 1) re-plans once and is then forced straight through to completion instead
    of pausing again."""
    from hyperion.config import settings

    stages: list[str] = []

    async def _fake_stage(task_id, request, stage, agents, tasks, cb, deadline):
        """Stand-in for _run_stage that records each executed stage's name."""
        stages.append(stage)

    with patch.object(settings, "tasks_dir", tmp_path), _mock_crew(_fake_stage):
        _write_plan(tmp_path, "t6")
        # revise_count already at the cap-1 → this revise exhausts the budget.
        result = await runner.resume_task(
            "t6", "do research", "revise", edits="again", hitl="plan",
            revise_count=runner._MAX_REVISIONS - 1,
        )

    assert result["status"] == "done"
    assert stages == ["plan", "research", "synthesize"]   # re-plan, then forced through
