"""
qdrant_client.py — Shared Qdrant retrieval helper for ~/ai/agents.

Role in the system
------------------
This is the single supported way for any agent or tool under ~/ai/agents to
query the "second_brain" Qdrant collection. All agent code imports from here
rather than talking to Qdrant or the embedding endpoint directly, so the
retrieval strategy lives in one place.

The collection is populated by two ingestion scripts:
  - ~/secondbrain/ingest_obsidian.py  (Obsidian vault → Qdrant)
  - ~/secondbrain/ingest_notion.py    (Notion databases → Qdrant)

Both write the same payload shape (title, text, notion_url) into the same
collection, so a single search call covers all second-brain content.

Key design decisions
--------------------
- **LiteLLM proxy for embeddings**: embeddings are produced through the
  LiteLLM proxy (http://localhost:4000/v1), never calling the OpenAI API
  directly, per the repo-wide LLM call convention.
- **EMBED_MODEL must match ingestion**: the model used here must be identical
  to the one used during ingestion, or vectors will be incomparable.
- **Module-level singletons** (_qdrant, _oai): constructed once at import
  time and reused across all calls; cheap to construct, hold no per-query
  state.
- **score_threshold = 0.30**: intentionally permissive — cosine similarity on
  text-embedding-3-small can be low even for relevant passages. Raise this
  threshold (e.g. 0.40+) only if results are consistently noisy.

Configuration (all overridable via environment variables)
---------------------------------------------------------
  QDRANT_URL         — default: http://localhost:6333
  QDRANT_COLLECTION  — default: "second_brain"
  LITELLM_BASE_URL   — default: http://localhost:4000/v1
  LITELLM_MASTER_KEY — required for the LiteLLM proxy

Usage:
    from _shared.qdrant_client import search_second_brain
    results = search_second_brain("investment thesis for SaaS companies")
"""

import os
from openai import OpenAI
from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.models import Filter

# Configuration — environment variables override these defaults so the same
# code works both locally (against the Docker stack) and inside Docker.
QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION       = os.getenv("QDRANT_COLLECTION", "second_brain")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY      = os.getenv("LITELLM_MASTER_KEY", "")
# Must match the model used during ingestion; changing it invalidates all
# existing vectors in the collection.
EMBED_MODEL      = "text-embedding-3-small"

# Module-level singletons reused across all calls (no per-query state).
# _oai is pointed at the LiteLLM proxy, not a provider API directly.
_qdrant = _QdrantClient(url=QDRANT_URL)
_oai    = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_KEY)


def _embed(text: str) -> list[float]:
    """Embed a string via the LiteLLM proxy and return the dense vector.

    Args:
        text: The text to embed (typically a search query).

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        openai.OpenAIError: If the LiteLLM proxy request fails.
    """
    return _oai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def search_second_brain(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.30,
    filter: Filter | None = None,
) -> list[dict]:
    """Semantic search over the second_brain Qdrant collection.

    Embeds the query via LiteLLM, runs a cosine-similarity search against the
    Qdrant collection, and returns the top results.

    Args:
        query:           Natural-language search query.
        limit:           Maximum number of results to return (default 5).
        score_threshold: Minimum cosine similarity score (default 0.30).
                         Intentionally permissive — raise only if results are noisy.
        filter:          Optional Qdrant Filter to narrow results by payload fields
                         (e.g. restrict to a specific project or PARA category).

    Returns:
        List of dicts with keys:
            title (str)       — page/note title
            text (str)        — text preview of the matching chunk
            file_path (str)   — vault-relative path of the source Obsidian note
                                (the source of truth), else ""
            notion_url (str)  — Notion mirror URL if also synced to Notion, else ""
            score (float)     — cosine similarity, rounded to 3 decimal places

    Raises:
        openai.OpenAIError: If embedding the query fails.
    """
    vector = _embed(query)
    # qdrant-client >=1.7 removed Client.search(); use query_points().points.
    hits = _qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=filter,
        with_payload=True,
    ).points
    return [
        {
            "title":      h.payload.get("title", ""),
            "text":       h.payload.get("text", ""),
            "file_path":  h.payload.get("file_path", ""),
            "notion_url": h.payload.get("notion_url", ""),
            "score":      round(h.score, 3),
        }
        for h in hits
    ]


def format_context(results: list[dict]) -> str:
    """Format search_second_brain results into a context block for an LLM prompt.

    Produces a Markdown string with one section per result, including the title,
    full text snippet, relevance score, and source URL.

    Args:
        results: The output of search_second_brain().

    Returns:
        A Markdown context block string. Returns a placeholder if results is empty.
    """
    if not results:
        return "(No relevant notes found in second brain.)"
    lines = ["## Relevant notes from second brain\n"]
    for r in results:
        lines.append(f"### {r['title']} (relevance: {r['score']})")
        lines.append(r["text"])
        # Cite the local Obsidian vault note (source of truth); fall back to the
        # Notion mirror URL only if the note has no vault path recorded.
        source = r.get("file_path") or r.get("notion_url") or ""
        if source:
            lines.append(f"Source: {source}\n")
    return "\n".join(lines)
