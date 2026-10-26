"""Shared utilities package for the ``~/ai/agents`` workspace.

Purpose
-------
Marks ``agents/_shared`` as an importable Python package and serves as the
documented home for cross-agent helper modules. Code that lives here is meant
to be reused by every standalone agent project (Hyperion, the research agent,
the AI workflow agent, etc.) so that connection logic to shared infrastructure
is written once and imported everywhere.

Role in the system
------------------
The wider ``~/ai`` ecosystem runs a Docker Compose stack (see
``ai-router/``) exposing shared services: LiteLLM (LLM proxy, ``:4000``),
SearXNG, Open WebUI, and Langfuse. The helpers in this
package are the canonical clients agents use to reach the data-layer pieces of
that stack:

- ``web_search``: pointer to the canonical web search implementation.
- ``notion_client``: wraps Notion reads used by the vault mirror scripts.

The Qdrant helper (``qdrant_client``, ``search_second_brain()``) was removed on
2026-10-26 when Qdrant was retired; the vault is read with plain file search now.

Design notes / non-obvious context
----------------------------------
- This ``__init__`` is intentionally side-effect free: it does NOT eagerly
  import submodules. Each agent imports exactly the helper it needs (e.g.
  ``from agents._shared.notion_client import ...``), which keeps
  package import cheap and avoids pulling in optional/heavy dependencies (and
  their environment requirements) for agents that don't use them.
- Per workspace convention, all LLM calls must go through the LiteLLM proxy at
  ``http://localhost:4000/v1`` rather than provider SDKs directly; helpers here
  follow and reinforce that convention.
- The leading underscore in the package name (``_shared``) signals "internal
  shared infrastructure for the agents workspace," distinguishing it from
  deployable agent projects that sit alongside it.
"""
