"""
Unit tests for Hyperion tools.

Run: uv run pytest tests/test_tools.py -v

Purpose
-------
Fast, fully-offline unit tests for the building-block tools that Hyperion
agents call during a run. These exercise the small, deterministic pieces of
each tool's logic (path safety, sorting, HTML cleanup, output framing, loop
detection) without touching the network, Docker services, or the LiteLLM
proxy.

Role in the system
------------------
Hyperion agents (planner/researcher/developer/critic/synthesizer) invoke a
shared set of tools under ``hyperion.tools.*`` and use the loop-guard in
``hyperion.crews.default``. This suite is the regression net for those tools'
core behavior, so it must stay self-contained and cheap to run in CI.

Key design decisions / non-obvious context
------------------------------------------
- **Everything is mocked.** Network-backed tools (reranker, web search) have
  ``httpx`` patched so no real HTTP request is ever made; the SearXNG /
  Infinity / LiteLLM stack does not need to be running.
- **Imports are deferred into each test body**, not at module top level. This
  keeps import side effects (settings loading, optional heavy deps) scoped to
  the test that needs them and avoids import-time failures in unrelated tests.
- **``tmp_path`` + ``patch.object(settings, "tasks_dir", ...)``** is the
  standard pattern for workspace tests: it redirects the tool's on-disk root
  to a pytest-managed temp dir so reads/writes are isolated and auto-cleaned.
- Tests are grouped by the module under test via banner comments
  (workspace / reranker / web_search / crews.default).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# workspace.py
# ---------------------------------------------------------------------------


def test_workspace_write_read(tmp_path):
    """Writing then reading the same relative path round-trips the content.

    Confirms WorkspaceWriteTool persists a file under the task's sandboxed
    directory and WorkspaceReadTool reads it back; expects the written
    payload to be present in the read result.
    """
    from hyperion.config import settings
    from hyperion.tools.workspace import WorkspaceReadTool, WorkspaceWriteTool

    # Redirect the workspace root to a temp dir so the test is isolated and
    # auto-cleaned; both tools share the same task_id ("t1") to hit one sandbox.
    with patch.object(settings, "tasks_dir", tmp_path):
        writer = WorkspaceWriteTool(task_id="t1")
        reader = WorkspaceReadTool(task_id="t1")
        writer._run("notes/test.md", "hello world")
        assert "hello world" in reader._run("notes/test.md")


def test_workspace_traversal_rejected(tmp_path):
    """Path-traversal escapes are blocked by the workspace sandbox.

    A relative path containing ``..`` segments that would resolve outside the
    task directory must raise ValueError (matching "traversal") rather than
    writing to an arbitrary location like /etc/passwd.
    """
    from hyperion.config import settings
    from hyperion.tools.workspace import WorkspaceWriteTool

    with patch.object(settings, "tasks_dir", tmp_path):
        writer = WorkspaceWriteTool(task_id="t1")
        # "../../etc/passwd" attempts to break out of the sandbox; the tool
        # must reject it instead of following the traversal.
        with pytest.raises(ValueError, match="traversal"):
            writer._run("../../etc/passwd", "bad")


# ---------------------------------------------------------------------------
# reranker.py (offline — mock httpx)
# ---------------------------------------------------------------------------


def test_reranker_returns_sorted():
    """rerank() returns (index, score) pairs ordered by descending relevance.

    Given an Infinity-style reranker response with out-of-order indices, the
    highest-scoring document must come first and its score must be preserved.
    """
    from hyperion.tools.reranker import rerank

    # Simulated reranker payload: indices intentionally out of order so the
    # assertion proves the tool sorts by relevance_score, not input order.
    fake_response = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.7},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status = MagicMock()

    # Patch the reranker's HTTP call so no real request hits the Infinity service.
    with patch("httpx.post", return_value=mock_resp):
        ranked = rerank("query", ["a", "b", "c"], top_n=3)

    assert ranked[0][0] == 2  # highest score first
    assert ranked[0][1] == pytest.approx(0.9)


def test_reranker_graceful_fallback():
    """rerank() degrades gracefully when the reranker service is unreachable.

    On any HTTP exception, the tool must not raise; it returns the documents
    in their original order with zero scores so callers can still proceed.
    """
    from hyperion.tools.reranker import rerank

    # Simulate a network failure; the tool should swallow it and fall back.
    with patch("httpx.post", side_effect=Exception("network down")):
        ranked = rerank("query", ["a", "b", "c"])

    # Returns original order with 0 scores
    assert [r[0] for r in ranked] == [0, 1, 2]


# ---------------------------------------------------------------------------
# web_search.py (offline — mock httpx)
# ---------------------------------------------------------------------------


def test_web_search_strips_html():
    """_strip_html() removes tags and decodes HTML entities.

    Inline markup is dropped and entities like ``&amp;`` are decoded to their
    plain-text equivalents, yielding clean text for the agent to consume.
    """
    from hyperion.tools.web_search import _strip_html

    assert _strip_html("<b>hello</b> &amp; <i>world</i>") == "hello & world"


def test_web_search_returns_untrusted_prefix():
    """WebSearchTool frames results as untrusted and surfaces source URLs.

    Search output fed to an LLM is external/untrusted content, so the tool
    must label it (the word "untrusted" appears) and include the result URL
    for provenance. This is a prompt-injection safety guardrail.
    """
    from hyperion.tools.web_search import WebSearchTool

    # Minimal SearXNG-style result payload returned by the mocked httpx.get.
    fake = {
        "results": [
            {"title": "Test", "content": "result content", "url": "http://example.com"}
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake
    mock_resp.raise_for_status = MagicMock()

    # Patch httpx.get (the SearXNG query) and the rerank step so the test is
    # fully offline; rerank is stubbed to keep the single result at index 0.
    with patch("httpx.get", return_value=mock_resp):
        with patch("hyperion.tools.web_search.rerank", return_value=[(0, 1.0)]):
            tool = WebSearchTool()
            result = tool._run("test query")

    assert "untrusted" in result.lower()
    assert "example.com" in result


# ---------------------------------------------------------------------------
# crews/default.py — CapExceeded
# ---------------------------------------------------------------------------


def test_tool_call_tracker_detects_loop():
    """ToolCallTracker raises CapExceeded after cap identical calls in a row.

    Guards against an agent stuck repeating the same tool+args. With cap=3,
    the third consecutive identical call must raise CapExceeded.
    """
    from hyperion.crews.default import CapExceeded, ToolCallTracker

    tracker = ToolCallTracker(cap=3)
    # First two identical calls are allowed; the third trips the cap.
    tracker.check("web_search", {"q": "same"})
    tracker.check("web_search", {"q": "same"})
    with pytest.raises(CapExceeded):
        tracker.check("web_search", {"q": "same"})


def test_tool_call_tracker_resets_on_different_call():
    """A differing tool call resets the repeat counter, avoiding false trips.

    The cap counts only consecutive identical calls; once the args change the
    streak restarts, so subsequent repeated-but-fewer-than-cap calls must not
    raise.
    """
    from hyperion.crews.default import ToolCallTracker

    tracker = ToolCallTracker(cap=3)
    tracker.check("web_search", {"q": "same"})
    tracker.check("web_search", {"q": "same"})
    tracker.check("web_search", {"q": "different"})  # resets
    tracker.check("web_search", {"q": "different"})
    # No exception — count reset
