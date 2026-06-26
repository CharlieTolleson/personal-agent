"""Hyperion agent memory subsystem.

This package groups the two complementary "memory" mechanisms that let a
multi-agent run reuse knowledge instead of re-deriving it on every stage:

* ``context_store`` — a *task-scoped* blackboard. Each run gets a
  ``tasks/{id}/context.json`` key/value file that any stage (planner,
  researcher, developer, critic, synthesizer) can read and write. This is the
  cross-stage channel for facts that are neither notes nor artifacts (e.g. the
  auto-discovered ``context_brief`` or recalled prior-task ids). Helpers are
  exposed both as plain functions for the runner and as tool wrappers
  (``ContextPutTool`` / ``ContextGetTool`` / ``RecallSimilarTasksTool``)
  granted to agents via the tool registry.

* ``episodic`` — *cross-task* long-term memory backed by the
  ``hyperion_memory`` Qdrant collection. After each completed run a single
  summary record is upserted (``store_episode``); the planner can semantically
  search prior runs before planning (``recall_similar_tasks``).

Design notes:
- This ``__init__`` is intentionally an empty package marker — it declares the
  ``hyperion.memory`` namespace without re-exporting submodule symbols, so
  callers import explicitly (e.g. ``from hyperion.memory.episodic import
  recall_similar_tasks``). Keeping it import-free also avoids pulling the heavy
  Qdrant/OpenAI clients in ``episodic`` at package-import time; those clients are
  constructed lazily inside the submodules only when memory is actually used.
- Per the workspace convention, all LLM/embedding calls in this package route
  through the LiteLLM proxy rather than provider SDKs directly.
"""
