"""Reranker tool — thin client for the Infinity bge-reranker-v2-m3 service.

Purpose
-------
Provides cross-encoder reranking for Hyperion's retrieval pipeline. Given a query
and a list of candidate document strings, it asks the Infinity reranker service
(part of the shared ai-router Docker stack) to score how relevant each candidate
is to the query, then returns the candidates ordered best-first.

Role in the system
-------------------
Vector search (Qdrant) returns candidates by embedding similarity, which is fast
but coarse. This module adds a second, more precise ranking pass: the
``bge-reranker-v2-m3`` cross-encoder reads each (query, document) pair jointly and
produces a sharper relevance score. Agents/tools call :func:`prioritize` (or the
lower-level :func:`rerank`) to keep only the most relevant — and budget-fitting —
slice of retrieved context before feeding it to an LLM.

Key design decisions / non-obvious context
-------------------------------------------
- HTTP, not in-process: the model runs in the Infinity service (see
  ``settings.infinity_url``); this file is a thin synchronous ``httpx`` client.
  All LLM/model traffic goes through shared infra rather than loading models here.
- Fail-soft: if the reranker service is unreachable or errors, callers should
  still make progress. :func:`rerank` therefore degrades gracefully by returning
  the original ordering (with zero scores) instead of raising.
- Token-budget trimming: :func:`prioritize` enforces the deferred per-stage
  input-token cap (Phase 4) by dropping the lowest-ranked
  candidates rather than aborting when retrieval is too large to fit.
"""

from __future__ import annotations

import logging

import httpx

from hyperion.config import settings

logger = logging.getLogger(__name__)

# Cross-encoder model name expected by the Infinity reranker service. Must match a
# model loaded/served by that service; changing it here without changing the
# server config will cause the rerank request to fail (caught and degraded below).
_MODEL = "BAAI/bge-reranker-v2-m3"

# Max characters of each document actually sent to the cross-encoder for SCORING.
# The model runs on CPU here (no GPU passthrough on the Mac host), and its latency
# scales with (query, doc) sequence length. A full web_search batch (~17 docs of
# ~2000 chars) took ~17s — over the old 15s client timeout — so reranking silently
# degraded to original order on every large batch. bge-reranker-v2-m3 truncates to
# 512 tokens internally anyway, and the lead of a snippet carries the relevance
# signal, so we cap the SCORING text at ~512 chars (≈8s for the same batch). This is
# scoring-only: the returned indices still map to the caller's full-length documents.
_RANK_CHAR_CAP = 512

# Generous request timeout: even capped, a cold model or a CPU spike can push a large
# batch past a tight deadline. 30s leaves headroom; rerank is fail-soft past it.
_RERANK_TIMEOUT_S = 30.0

# In-process health counters. rerank() is deliberately fail-soft (falls back to the
# original order on any error), which means a chronic outage is otherwise INVISIBLE —
# results still come back, just unranked. These counters make that degradation
# observable via GET /metrics so "the reranker has been off for a week" can't hide.
# Best-effort: reset on process restart (the API is a single uvicorn process).
_STATS: dict[str, int] = {"calls": 0, "ok": 0, "degraded_timeout": 0, "degraded_error": 0}


def health_stats() -> dict[str, int]:
    """Snapshot of reranker call outcomes since process start (for GET /metrics).

    Keys: ``calls`` (total), ``ok`` (reranked successfully), ``degraded_timeout``
    (fell back because the service was too slow), ``degraded_error`` (fell back on
    any other error, e.g. connection refused / bad response). A rising
    ``degraded_*`` share means reranking is silently not happening.
    """
    return dict(_STATS)


def rerank(query: str, documents: list[str], top_n: int = 5) -> list[tuple[int, float]]:
    """
    Rerank documents against a query using the Infinity cross-encoder.

    Sends a single ``/rerank`` request to the Infinity service and returns the
    candidate indices ordered by descending relevance.

    Args:
        query: The search/query string to score documents against.
        documents: Candidate document strings, in their original order. The
            integer indices in the result refer to positions in this list.
        top_n: Maximum number of results to return (the highest-scored slice).
            Defaults to 5.

    Returns:
        A list of ``(original_index, relevance_score)`` tuples sorted descending
        by score and capped at ``top_n``. ``original_index`` is the position of
        the document in the input ``documents`` list, so callers can map results
        back to the originals. Returns an empty list when ``documents`` is empty.

    Raises:
        None. Network/HTTP/parsing failures are caught and handled by degrading
        gracefully: a warning is logged and the original ordering is returned
        with placeholder ``0.0`` scores (capped at ``top_n``). This fail-soft
        behavior lets the retrieval pipeline keep working when the reranker
        service is down.

    Side effects:
        Performs a synchronous HTTP POST to ``settings.infinity_url``
        (``_RERANK_TIMEOUT_S``) and may emit a warning log on failure. Each
        document is truncated to ``_RANK_CHAR_CAP`` chars for scoring only.
    """
    if not documents:
        return []
    # Cap the SCORING text per doc to bound CPU cross-encoder latency (see
    # _RANK_CHAR_CAP). Indices are preserved, so results still map to the caller's
    # full-length `documents`.
    ranking_docs = [d[:_RANK_CHAR_CAP] for d in documents]
    _STATS["calls"] += 1
    try:
        resp = httpx.post(
            f"{settings.infinity_url}/rerank",
            json={"model": _MODEL, "query": query, "documents": ranking_docs},
            timeout=_RERANK_TIMEOUT_S,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        # Infinity returns each result with its own "index" (position in the
        # input list) and "relevance_score"; re-sort defensively in case the
        # service does not already return them best-first, then take top_n.
        ranked = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
        _STATS["ok"] += 1
        return [(r["index"], r["relevance_score"]) for r in ranked[:top_n]]
    except Exception as exc:
        # Fail-soft: never propagate reranker outages to callers. Preserve the
        # original input order (first top_n items) so retrieval still proceeds.
        # Classify timeout (service too slow — tune cap/timeout/model) vs any other
        # error (down / bad response) so chronic slowness is distinguishable in the
        # logs and in the /metrics counters.
        if isinstance(exc, httpx.TimeoutException):
            _STATS["degraded_timeout"] += 1
            logger.warning(
                "Reranker TIMEOUT after %ss (%d docs) — returning original order. "
                "If frequent, the CPU cross-encoder is too slow for the batch; lower "
                "_RANK_CHAR_CAP / doc count, raise _RERANK_TIMEOUT_S, or use a lighter "
                "reranker/GPU.",
                _RERANK_TIMEOUT_S, len(documents),
            )
        else:
            _STATS["degraded_error"] += 1
            logger.warning("Reranker unavailable (%s) — returning original order", exc)
        return [(i, 0.0) for i in range(min(top_n, len(documents)))]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (≈4 chars/token) — good enough for budget trimming.

    Uses the common heuristic of ~4 characters per token to avoid a real
    tokenizer dependency on this hot path. Intended only for approximate budget
    accounting in :func:`prioritize`, not for exact accounting/billing.

    Args:
        text: The text whose token count to estimate.

    Returns:
        Estimated token count as an int, floored at 1 so non-empty inputs always
        cost at least one token (avoids zero-cost items slipping past budgets).
    """
    return max(1, len(text) // 4)


def prioritize(
    query: str,
    candidates: list[str],
    top_n: int | None = None,
    token_budget: int | None = None,
) -> list[str]:
    """Shared retrieval-prioritization primitive (Phase 4).

    Reranks ``candidates`` by relevance to ``query`` (highest first), then trims the
    lowest-scored items so the running token estimate fits ``token_budget``. This is
    how the deferred per-stage input-token cap is finally enforced: instead of
    aborting when raw retrieval is too large, we keep the most relevant slice that
    fits and log what was dropped.

    Returns the kept candidate strings in ranked order.

    Args:
        query: The query to rank ``candidates`` against.
        candidates: Candidate document strings to prioritize.
        top_n: Optional cap on how many candidates to consider after reranking.
            When ``None``, all candidates are considered.
        token_budget: Optional approximate token ceiling for the kept set. When
            set, candidates are added in ranked order until adding the next one
            would exceed the budget; lower-ranked overflow is dropped. When
            ``None``, no trimming occurs.

    Returns:
        The kept candidate strings (subset of ``candidates``) in best-first
        ranked order. Returns an empty list when ``candidates`` is empty.

    Raises:
        None directly. Delegates to :func:`rerank`, which is fail-soft (falls
        back to original order if the reranker service is unavailable).

    Side effects:
        Triggers a reranker HTTP call via :func:`rerank`, and emits an info log
        summarizing how many candidates were trimmed when ``token_budget`` causes
        any drops.
    """
    if not candidates:
        return []
    limit = top_n if top_n is not None else len(candidates)
    # Clamp top_n to the candidate count so we never request more than we have.
    ranked = rerank(query, candidates, top_n=min(limit, len(candidates)))

    kept: list[str] = []
    used = 0  # running estimated token total of kept items
    dropped = 0  # count of candidates skipped to stay within token_budget
    for idx, _score in ranked:
        text = candidates[idx]
        if token_budget is not None:
            cost = _estimate_tokens(text)
            # Skip this item if it would overflow the budget — but only once we
            # have kept at least one item ("and kept"), so the single most
            # relevant candidate is always returned even if it alone exceeds the
            # budget (better to return something than nothing).
            if used + cost > token_budget and kept:
                dropped += 1
                continue
            used += cost
        kept.append(text)
    if dropped:
        logger.info(
            "prioritize: trimmed %d/%d candidates to fit token_budget=%s (≈%d tokens kept)",
            dropped, len(candidates), token_budget, used,
        )
    return kept
