"""Hyperion agent tool package.

This package groups the concrete tool implementations that Hyperion's agents
(planner / researcher / developer / critic / synthesizer) can invoke while
executing a task. Each sibling module exposes one capability:

- ``web_search``: live web lookups via the SearXNG meta-search instance in the
  ai-router stack.
- ``reranker``: relevance re-ranking of candidate passages via the Infinity
  reranker service.
- ``notion``: read access to the Notion workspace that mirrors the second brain.
- ``workspace``: scratch/file operations within the agent's working directory.

Design notes
------------
- This file is intentionally an (otherwise) empty package marker: it carries no
  re-exports so that importing ``hyperion.tools`` stays cheap and does not pull
  in every tool's (sometimes heavy) dependencies. Import the specific submodule
  you need instead.
- Per the repo-wide convention, any LLM/embedding calls made by these tools must
  route through the LiteLLM proxy at ``http://localhost:4000/v1`` rather than
  calling provider APIs directly.
"""
