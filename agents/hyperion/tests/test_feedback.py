"""Feedback / alerts / affordance tests (Phase 6).

File-backed channels (feedback queue, affordances, alerts) are exercised against a
patched ``tasks_dir``. The crew stages are mocked so the awaiting_input pause/resume
control flow runs without any LLM call.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperion import alerts, feedback
from hyperion.config import settings
from hyperion.crews import runner


@pytest.fixture
def anyio_backend():
    """Pin anyio's parametrized backend to asyncio for the async tests.

    The ``anyio`` pytest plugin parametrizes ``@pytest.mark.anyio`` tests across
    every installed backend (asyncio, trio, ...). Returning a single string here
    restricts the matrix to asyncio only, so the async tests run once.

    Returns:
        str: The literal ``"asyncio"`` backend name.
    """
    return "asyncio"


@contextlib.contextmanager
def _mock_crew(stage_impl):
    """Patch the crew runner so ``run_task``/``resume_task`` execute without an LLM.

    Replaces the agent-building, context-discovery, and node-task-factory helpers
    with no-op mocks, and swaps ``runner._run_stage`` for the caller-supplied
    ``stage_impl``. This lets the pause/resume control flow be exercised end to end
    while the per-stage work is fully simulated.

    Args:
        stage_impl: Async callable used as the replacement ``_run_stage``. It
            receives ``(task_id, request, stage, agents, tasks, cb, deadline)`` and
            stands in for the real (LLM-driven) stage execution.

    Yields:
        unittest.mock.MagicMock: The patched ``_run_stage`` attribute (i.e.
            ``stage_impl`` as installed by ``patch.object``), for optional assertions.

    Side effects:
        Patches several attributes on the ``hyperion.crews.runner`` module for the
        duration of the ``with`` block; all patches are reverted on exit.
    """
    with patch.object(runner, "build_agent", MagicMock()), \
         patch.object(runner, "discover_context", MagicMock(return_value=None)), \
         patch.object(runner, "_plan_task", MagicMock()), \
         patch.object(runner, "_work_task", MagicMock()), \
         patch.object(runner, "_synthesize_task", MagicMock()), \
         patch.object(runner, "_run_stage", new=stage_impl) as stage:
        yield stage


def _write_plan(base, task_id):
    """Write a minimal valid ``plan.md`` for a task so resume can read its frontmatter.

    The runner expects an approved plan on disk before resuming. This helper lays
    down a task directory containing a ``plan.md`` whose YAML frontmatter carries the
    fields the runner parses (``task_type``, ``keywords``, and one option with a
    single subtask), followed by a placeholder Markdown body.

    Args:
        base: Base tasks directory (typically the patched ``settings.tasks_dir``,
            i.e. pytest's ``tmp_path``).
        task_id: Identifier whose subdirectory under ``base`` receives the plan.

    Side effects:
        Creates ``base/<task_id>/`` (including parents) and writes ``plan.md`` there.
    """
    d = base / task_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.md").write_text(
        "---\ntask_type: research\nkeywords: [demo]\n"
        "options:\n  - id: a\n    summary: shallow\n    subtasks:\n      - id: s1\n        description: scan\n"
        "---\n\n# Plan\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Feedback queue — delivered exactly once
# ---------------------------------------------------------------------------


def test_feedback_queue_roundtrip(tmp_path):
    """Appended feedback drains in FIFO order exactly once, then yields an empty list."""
    with patch.object(settings, "tasks_dir", tmp_path):
        feedback.append_feedback("t1", "look at X")
        feedback.append_feedback("t1", "also Y")
        first = feedback.drain_feedback("t1")
        second = feedback.drain_feedback("t1")

    assert first == ["look at X", "also Y"]
    assert second == []  # consumed once


def test_inject_feedback_wraps_as_data(tmp_path):
    """inject_feedback returns None when empty, else a 'data, not instructions' block that drains the queue once."""
    with patch.object(settings, "tasks_dir", tmp_path):
        assert runner.inject_feedback("t1") is None  # empty queue
        feedback.append_feedback("t1", "prioritise the API section")
        block = runner.inject_feedback("t1")
        assert runner.inject_feedback("t1") is None  # drained once

    assert block is not None
    assert "data, not instructions" in block
    assert "prioritise the API section" in block


# ---------------------------------------------------------------------------
# Affordances
# ---------------------------------------------------------------------------


def test_affordance_record_and_answer(tmp_path):
    """Recording an affordance makes it the latest pending one; answering clears it and pushes the answer onto the feedback queue."""
    with patch.object(settings, "tasks_dir", tmp_path):
        aff_id = feedback.record_affordance(
            "t2", {"type": "question", "prompt": "Which region?"}
        )
        pending = feedback.latest_pending_affordance("t2")
        assert pending is not None
        assert pending["id"] == aff_id
        assert pending["prompt"] == "Which region?"

        ok = feedback.answer_affordance("t2", "EU")
        assert ok is True
        assert feedback.latest_pending_affordance("t2") is None
        # The answer is pushed onto the feedback queue for the resuming stage.
        assert any("EU" in m for m in feedback.drain_feedback("t2"))


def test_ask_user_tool_records_affordance(tmp_path):
    """AskUserTool._run records a 'question' affordance and returns an acknowledgement string to the agent."""
    with patch.object(settings, "tasks_dir", tmp_path):
        tool = feedback.AskUserTool(task_id="t3")
        out = tool._run("Do you want a summary table?")
        assert "Asked the user" in out
        pending = feedback.latest_pending_affordance("t3")
    assert pending["type"] == "question"


# ---------------------------------------------------------------------------
# Alerts — exactly one per (task, kind)
# ---------------------------------------------------------------------------


def test_alert_fires_once_per_kind(tmp_path):
    """emit_alert returns True the first time for a (task, kind) and False (deduped) thereafter, writing the kind to alerts.md."""
    alerts.reset("t4")
    with patch.object(settings, "tasks_dir", tmp_path), \
         patch.object(alerts, "_push_notification", MagicMock()):
        first = alerts.emit_alert("t4", "tool-loop", "near cap")
        second = alerts.emit_alert("t4", "tool-loop", "near cap again")

    assert first is True
    assert second is False  # deduped
    body = (tmp_path / "t4" / "alerts.md").read_text(encoding="utf-8")
    assert "tool-loop" in body


def test_alert_disabled_by_setting(tmp_path):
    """With hyperion_hitl_alerts set to 'off', emit_alert returns False and writes no alerts.md file."""
    alerts.reset("t5")
    with patch.object(settings, "tasks_dir", tmp_path), \
         patch.object(settings, "hyperion_hitl_alerts", "off"), \
         patch.object(alerts, "_push_notification", MagicMock()):
        fired = alerts.emit_alert("t5", "wall-budget", "slow")
    assert fired is False
    assert not (tmp_path / "t5" / "alerts.md").exists()


# ---------------------------------------------------------------------------
# run_task pauses on a planner question, resumes on the answer
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_task_pauses_on_affordance(tmp_path):
    """run_task halts with status 'awaiting_input' (pending_stage='plan') when a stage records an affordance."""
    async def _stage(task_id, request, stage, agents, tasks, cb, deadline):
        # The (mocked) planner "asks" a question during the plan stage.
        if stage == "plan":
            feedback.record_affordance(task_id, {"type": "question", "prompt": "Scope?"})

    with patch.object(settings, "tasks_dir", tmp_path), _mock_crew(_stage):
        _write_plan(tmp_path, "t6")
        result = await runner.run_task("t6", "vague request", hitl="off")

    assert result["status"] == "awaiting_input"
    assert result["pending_stage"] == "plan"


@pytest.mark.anyio
async def test_resume_after_answer_runs_through(tmp_path):
    """resume_task with a 'revise' answer (and no pending affordance) finishes 'done', running plan -> research -> synthesize in order."""
    stages: list[str] = []

    async def _stage(task_id, request, stage, agents, tasks, cb, deadline):
        stages.append(stage)

    with patch.object(settings, "tasks_dir", tmp_path), _mock_crew(_stage):
        _write_plan(tmp_path, "t7")
        # The answer arrives as a revise with edits; no pending affordance remains.
        result = await runner.resume_task(
            "t7", "vague request", "revise", edits="Scope is EU only", hitl="off"
        )

    assert result["status"] == "done"
    assert stages == ["plan", "research", "synthesize"]
