"""Second brain tool — semantic search over the ``second_brain`` Qdrant collection.

Role in the system
------------------
Exposes Charlie's personal Obsidian/Notion "second brain" (the PARA vault ingested
into Qdrant) as a CrewAI ``BaseTool`` so Hyperion agents (planner, researcher, etc.)
can pull in background knowledge, past notes, career goals, projects, and investments
while reasoning. It is one of the retrieval tools registered with the agents alongside
``web_search``, ``notion``, etc.

Pipeline
--------
1. Embed + vector-search the ``second_brain`` collection via the shared
   ``search_second_brain`` helper (over-fetches candidates).
2. Rerank candidates against the query with the Infinity reranker (``rerank``).
3. Format the top-``k`` results into a Markdown string, trimming output to a
   per-call token budget so a single call can't flood a stage's context window.

Key design decisions / non-obvious context
-------------------------------------------
- The shared ``qdrant_client.py`` (under ``agents/_shared/``) is loaded via
  ``importlib`` rather than a normal import. This deliberately sidesteps a name
  collision with the installed ``qdrant-client`` PyPI package, and tolerates two
  different filesystem layouts (local checkout vs. the Docker image build context).
- Retrieval output is capped by ``_SECOND_BRAIN_TOKEN_BUDGET`` at the source, so
  the cap is enforced regardless of which agent/stage invokes the tool.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load agents/_shared/qdrant_client.py via importlib to avoid name collision
# with the installed `qdrant-client` package.
# Local: parents[4] == agents/
# Docker: parents[4] == / (build context is agents/, so _shared/ lands at /app/_shared/)
_SHARED_PATH = Path(__file__).parents[4] / "_shared" / "qdrant_client.py"
if not _SHARED_PATH.exists():
    _SHARED_PATH = Path(__file__).parents[3] / "_shared" / "qdrant_client.py"
_spec = importlib.util.spec_from_file_location("hyperion_shared_qdrant", _SHARED_PATH)
_shared_qdrant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared_qdrant)
search_second_brain = _shared_qdrant.search_second_brain

from hyperion.config import settings
from hyperion.tools.reranker import rerank


# Per-call retrieval budget (tokens). Keeps a single tool call from dumping more
# context than a stage can absorb — the shared input cap, enforced at the source.
_SECOND_BRAIN_TOKEN_BUDGET = 6000


class SecondBrainTool:
    """Tool that performs reranked semantic search over the second brain.

    Registered with Hyperion agents so they can retrieve personal notes/knowledge
    on demand. The ``name`` and ``description`` fields are surfaced to the LLM as
    the tool's callable signature; the LLM passes a single natural-language query.

    Attributes:
        name: Tool identifier exposed to the agent/LLM ("search_second_brain").
        description: Natural-language guidance shown to the LLM on when/how to call.
        parameters: JSON schema for the tool's arguments.
        top_k: Number of results to keep after reranking (post-rerank cutoff).
    """

    name = "search_second_brain"
    description = (
        "Semantic search over the personal Notion second brain. "
        "Use for background knowledge, past notes, career goals, projects, and investments. "
        "Input: a natural-language query string."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."}
        },
        "required": ["query"],
    }

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def _run(self, query: str) -> str:
        """Search the second brain and return formatted, budget-trimmed results.

        Invoked by CrewAI when an agent calls this tool. Over-fetches vector
        candidates, reranks them, then renders the top results as Markdown.

        Args:
            query: Natural-language search string supplied by the agent/LLM.

        Returns:
            A Markdown string with a header and one section per result (title,
            relevance score, snippet, and the local Obsidian vault note path as the
            source — the vault is the source of truth; the Notion mirror URL is only
            used as a fallback when a note has no vault path). If no
            candidates clear the vector score threshold, returns a short
            "(No relevant notes found…)" placeholder string instead.

        Side effects:
            Issues a Qdrant vector query and a call to the reranker service; both
            are network/IO calls routed through the shared infrastructure.
        """
        # Fetch more candidates than needed, then rerank to top_k
        candidates = search_second_brain(
            query=query,
            limit=15,
            score_threshold=0.25,
        )
        if not candidates:
            return "(No relevant notes found in second brain.)"

        texts = [c["text"] for c in candidates]
        ranked = rerank(query, texts, top_n=self.top_k)

        lines = [f"## Second brain search results for: {query!r}\n"]
        used = 0
        for orig_idx, score in ranked:
            c = candidates[orig_idx]
            snippet = c["text"][:2000]  # cap per note
            cost = max(1, len(snippet) // 4)
            # Trim to the per-call token budget instead of dumping everything.
            if used and used + cost > _SECOND_BRAIN_TOKEN_BUDGET:
                lines.append(f"(…{len(ranked)} results trimmed to fit context budget)")
                break
            used += cost
            lines.append(f"### {c['title']} (relevance: {score:.3f})")
            lines.append(snippet)
            # Cite the local Obsidian vault note (source of truth); fall back to the
            # Notion mirror URL only when a note has no vault path recorded.
            source = c.get("file_path") or c.get("notion_url")
            if source:
                lines.append(f"Source: {source}")
            lines.append("")
        return "\n".join(lines)
