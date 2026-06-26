"""Phase 1 — direct node-to-node context passing along DAG edges.

The runner threads each node's output text into its downstream nodes' task
descriptions (and seeds root nodes with the raw idea/request), instead of relying
solely on agents reading/writing workspace files. This is a pure runner concern.
See ``plan-phase1-context-passing.md``.

What is under test:
  - ``_upstream_context``: root nodes get the raw request; downstream nodes get
    each upstream's output labelled by node id; missing upstreams are skipped;
    oversized blocks are truncated with a marker.
  - ``_output_text``: normalizes CrewOutput (``.raw``), a sub-workflow child run
    dict (read its ``result_path`` file), and a bare string.
  - ``_node_task``: prepends the upstream-context header; a root ``plan`` node is
    NOT double-injected (``_plan_task`` already embeds the request).
  - End-to-end through ``_execute_workflow`` (LLM-free): the idea reaches every
    root node and each node's output flows into the next.

Test-isolation notes mirror ``test_subworkflow``: ``_run_stage`` is faked to
capture the task descriptions and return synthetic outputs, ``build_agent`` /
``load_agent`` / ``discover_context`` are mocked, ``settings.tasks_dir`` points at
``tmp_path``, and ``run_meta_tasks`` is mocked away.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperion.crews import runner
from hyperion.crews.workflows import WorkflowNode, WorkflowRecord


@pytest.fixture
def anyio_backend():
    """Run ``@pytest.mark.anyio`` tests on asyncio only (the runner is asyncio-based)."""
    return "asyncio"


# ---------------------------------------------------------------------------
# _upstream_context
# ---------------------------------------------------------------------------


def test_root_node_context_is_the_raw_request():
    """A node with no upstreams is seeded with the idea/request itself."""
    node = WorkflowNode(id="market-sizing", agent="researcher", kind="work", upstream=[])
    ctx = runner._upstream_context(node, {}, "BUILD A VET TRIAGE APP")
    assert "## Idea under evaluation" in ctx
    assert "BUILD A VET TRIAGE APP" in ctx


def test_downstream_node_gets_labelled_upstream_outputs():
    """A downstream node receives each upstream's output, labelled by node id and
    in DAG order."""
    node = WorkflowNode(
        id="syn", agent="synthesizer", kind="synthesize", upstream=["a", "b"]
    )
    outputs = {"a": "alpha findings", "b": "beta findings"}
    ctx = runner._upstream_context(node, outputs, "the idea")
    assert "## Input from upstream step `a`" in ctx
    assert "## Input from upstream step `b`" in ctx
    assert "alpha findings" in ctx and "beta findings" in ctx
    # The raw request is NOT injected into a non-root node.
    assert "## Idea under evaluation" not in ctx
    # DAG order preserved: a's block precedes b's.
    assert ctx.index("`a`") < ctx.index("`b`")


def test_missing_upstream_output_is_skipped():
    """An upstream with no captured output (skipped node / pre-resume) is omitted
    rather than emitting an empty block."""
    node = WorkflowNode(id="syn", agent="synthesizer", kind="synthesize",
                        upstream=["a", "b"])
    ctx = runner._upstream_context(node, {"a": "only alpha"}, "idea")
    assert "`a`" in ctx
    assert "`b`" not in ctx


def test_oversized_upstream_block_is_truncated():
    """A single upstream block over the cap is truncated with the marker so a
    runaway node can't blow the downstream context window."""
    big = "x" * (runner._MAX_UPSTREAM_BLOCK_CHARS + 5000)
    node = WorkflowNode(id="syn", agent="synthesizer", kind="synthesize",
                        upstream=["a"])
    ctx = runner._upstream_context(node, {"a": big}, "idea")
    assert "…[truncated]" in ctx
    # The body is capped near the limit, not the full 17k chars.
    assert len(ctx) < runner._MAX_UPSTREAM_BLOCK_CHARS + 500


# ---------------------------------------------------------------------------
# _output_text
# ---------------------------------------------------------------------------


def test_output_text_reads_crewoutput_raw():
    """A CrewOutput-like object is normalized via its ``.raw`` attribute."""
    assert runner._output_text(SimpleNamespace(raw="  hello  ")) == "hello"


def test_output_text_handles_bare_string():
    assert runner._output_text("  plain  ") == "plain"
    assert runner._output_text(None) == ""


def test_output_text_reads_subworkflow_result_file(tmp_path):
    """A sub-workflow child run dict is normalized by reading its result file."""
    rp = tmp_path / "result.md"
    rp.write_text("child report body", encoding="utf-8")
    result = {"status": "done", "result_path": str(rp)}
    assert runner._output_text(result) == "child report body"
    # A dict with no result_path contributes nothing.
    assert runner._output_text({"status": "done", "result_path": None}) == ""


# ---------------------------------------------------------------------------
# _node_task — header injection + plan double-injection guard
# ---------------------------------------------------------------------------


def _rec(agent_id: str = "researcher"):
    """Minimal stand-in for an AgentRecord (only ``.id`` is read by the builders)."""
    return SimpleNamespace(id=agent_id)


def test_instruction_node_gets_upstream_header():
    """An instruction node has the upstream context prepended ahead of its body."""
    node = WorkflowNode(id="syn", agent="synthesizer", kind="synthesize",
                        upstream=["a"], instruction="Synthesize the briefs.")
    up_ctx = runner._upstream_context(node, {"a": "alpha findings"}, "idea")
    task = runner._node_task(node, _rec("synthesizer"), None, "idea", None, None, up_ctx)
    assert "alpha findings" in task.description
    assert "Synthesize the briefs." in task.description
    # Header precedes the instruction body.
    assert task.description.index("alpha findings") < task.description.index("Synthesize")


def test_root_plan_node_is_not_double_injected():
    """A root ``plan`` node embeds the request once via ``_plan_task`` and must NOT
    also receive the root-request header."""
    node = WorkflowNode(id="plan", agent="planner", kind="plan", upstream=[])
    up_ctx = runner._upstream_context(node, {}, "THE REQUEST")
    task = runner._node_task(node, _rec("planner"), None, "THE REQUEST", None, None, up_ctx)
    # The request appears exactly once, and the root header label is absent.
    assert task.description.count("THE REQUEST") == 1
    assert "## Idea under evaluation" not in task.description


def test_root_work_node_gets_idea_header():
    """A root ``work`` node (no instruction) gets the idea via the header so it no
    longer runs blind — the latent-bug fix."""
    node = WorkflowNode(id="r1", agent="researcher", kind="work", upstream=[])
    up_ctx = runner._upstream_context(node, {}, "THE IDEA")
    task = runner._node_task(node, _rec("researcher"), None, "THE IDEA", None, None, up_ctx)
    assert "## Idea under evaluation" in task.description
    assert "THE IDEA" in task.description


# ---------------------------------------------------------------------------
# End-to-end through _execute_workflow (LLM-free)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mock_crew(tasks_dir, captured: dict):
    """Patch the runner so a workflow runs with no LLM/agent execution.

    The fake ``_run_stage`` records each node's task description into ``captured``
    and returns a synthetic CrewOutput whose ``.raw`` is ``OUTPUT::<node>`` so the
    next wave's upstream context contains a recognizable marker.
    """
    from hyperion.config import settings

    async def _fake_stage(task_id, request, stage, agents, tasks, cb, deadline):
        captured[stage] = tasks[0].description
        base = settings.tasks_dir / task_id / "artifacts"
        base.mkdir(parents=True, exist_ok=True)
        (base / "result.md").write_text(f"OUTPUT::{stage}", encoding="utf-8")
        return SimpleNamespace(raw=f"OUTPUT::{stage}")

    with patch.object(settings, "tasks_dir", tasks_dir), \
         patch.object(runner, "build_agent", MagicMock(return_value=None)), \
         patch.object(runner, "load_agent", MagicMock(side_effect=lambda a: _rec(a))), \
         patch.object(runner, "discover_context", MagicMock(return_value=None)), \
         patch.object(runner, "_run_stage", new=_fake_stage), \
         patch("hyperion.server.meta_tasks.run_meta_tasks", new=AsyncMock()):
        yield


def _diamond_workflow() -> WorkflowRecord:
    """Two parallel root researchers fanning into a synthesizer."""
    return WorkflowRecord(
        id="diamond", name="diamond",
        nodes=[
            WorkflowNode(id="r1", agent="researcher", kind="work", upstream=[],
                         instruction="Research angle one."),
            WorkflowNode(id="r2", agent="researcher", kind="work", upstream=[],
                         instruction="Research angle two."),
            WorkflowNode(id="syn", agent="synthesizer", kind="synthesize",
                         upstream=["r1", "r2"], instruction="Combine the briefs."),
        ],
    )


@pytest.mark.anyio
async def test_idea_flows_to_roots_and_outputs_thread_downstream(tmp_path):
    """End-to-end: the idea reaches every root node, and each root's output flows
    into the downstream synthesizer."""
    captured: dict = {}
    wf = _diamond_workflow()
    with _mock_crew(tmp_path, captured), \
         patch("hyperion.crews.workflows.resolve_workflow",
               new=MagicMock(return_value=wf)):
        result = await runner.run_task("t-diamond", "MY STARTUP IDEA", workflow="diamond")

    assert result["status"] == "done"
    # The raw idea reached both blind root researchers.
    assert "MY STARTUP IDEA" in captured["r1"]
    assert "MY STARTUP IDEA" in captured["r2"]
    # The synthesizer received both upstream outputs, labelled by node id.
    assert "OUTPUT::r1" in captured["syn"]
    assert "OUTPUT::r2" in captured["syn"]
    assert "`r1`" in captured["syn"] and "`r2`" in captured["syn"]
