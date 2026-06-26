"""Context-layer tests (Phase 4): blackboard + prioritize budget.

Purpose
-------
Unit tests for Hyperion's shared-context ("blackboard") layer and for the
token-budget trimming logic used when assembling context for an LLM call.

Scope / what is covered
------------------------
1. ``hyperion.memory.context_store`` — a per-task key/value blackboard that lets
   multiple agent stages in a workflow share facts. Tests cover put/get
   round-trips, whole-blackboard reads, missing-key/missing-task behavior, and
   the tool wrappers (``ContextPutTool`` / ``ContextGetTool``) that expose
   the store to agents.
2. ``hyperion.tools.reranker.prioritize`` — selects the highest-value candidate
   snippets that fit within a token budget. Tests cover normal trimming, the
   "always keep at least one" guarantee, and the empty-input case.

Key design / test-environment notes
------------------------------------
- ``context_store`` persists each task's blackboard under ``settings.tasks_dir``.
  Every test that touches it patches ``settings.tasks_dir`` to a pytest
  ``tmp_path`` so the tests are hermetic and never write to the real tasks dir.
- The reranker model (Infinity) is offline during tests. ``prioritize`` falls
  back to the original candidate order with uniform score 0.0, so these tests
  patch ``reranker.rerank`` to return that deterministic fallback rather than
  hitting the network.
- ``settings`` is imported lazily inside each test (after ``tasks_dir`` is
  patched) to keep the patch scoped to the test body.
"""

from __future__ import annotations

from unittest.mock import patch

from hyperion.memory import context_store
from hyperion.tools import reranker


def test_context_put_get_roundtrip(tmp_path):
    """Putting two keys then getting them back returns the values, the full
    blackboard dict, and None for an absent key."""
    from hyperion.config import settings

    # Redirect blackboard persistence to a throwaway dir for hermetic I/O.
    with patch.object(settings, "tasks_dir", tmp_path):
        context_store.context_put("t1", "headline", "42%")
        context_store.context_put("t1", "source", "https://example.com")
        assert context_store.context_get("t1", "headline") == "42%"
        # whole-blackboard read
        assert context_store.context_get("t1") == {
            "headline": "42%",
            "source": "https://example.com",
        }
        # missing key → None
        assert context_store.context_get("t1", "nope") is None


def test_context_get_missing_task_returns_empty(tmp_path):
    """Reading a task that has no blackboard yields an empty dict for a
    whole-board read and None for a single-key read (no error raised)."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path):
        assert context_store.context_get("ghost") == {}
        assert context_store.context_get("ghost", "k") is None


def test_synthesizer_reads_fact_it_did_not_produce(tmp_path):
    """Acceptance: a downstream stage reads a context fact written upstream."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path):
        # researcher writes
        context_store.ContextPutTool(task_id="t2")._run("finding", "GDP grew 3%")
        # synthesizer reads
        out = context_store.ContextGetTool(task_id="t2")._run("finding")
        assert "GDP grew 3%" in out


def test_prioritize_trims_to_token_budget():
    """With three ~100-token candidates and a 150-token budget, only the first
    fits, so prioritize returns just that one."""
    # Reranker is offline in tests → falls back to original order with score 0.0,
    # so prioritize keeps items in order until the budget is hit.
    candidates = ["x" * 400, "y" * 400, "z" * 400]  # ≈100 tokens each
    with patch.object(reranker, "rerank", return_value=[(0, 0.0), (1, 0.0), (2, 0.0)]):
        kept = reranker.prioritize("q", candidates, token_budget=150)
    # first (100) fits; second would push to 200 > 150 → dropped; always keep ≥1
    assert kept == [candidates[0]]


def test_prioritize_keeps_at_least_one_over_budget():
    """Even a single candidate that exceeds the budget is kept — prioritize
    never returns empty when there is at least one candidate."""
    big = ["a" * 10_000]  # far over budget
    with patch.object(reranker, "rerank", return_value=[(0, 0.0)]):
        kept = reranker.prioritize("q", big, token_budget=10)
    assert kept == big  # never drop the single best candidate


def test_prioritize_empty():
    """An empty candidate list returns an empty list (no rerank call needed)."""
    assert reranker.prioritize("q", []) == []
