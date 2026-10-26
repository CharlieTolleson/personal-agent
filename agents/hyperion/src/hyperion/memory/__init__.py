"""Hyperion agent memory subsystem.

This package holds the memory mechanism that lets a multi-agent run reuse
knowledge instead of re-deriving it on every stage:

* ``context_store`` — a *task-scoped* blackboard. Each run gets a
  ``tasks/{id}/context.json`` key/value file that any stage (planner,
  researcher, developer, critic, synthesizer) can read and write. This is the
  cross-stage channel for facts that are neither notes nor artifacts (e.g. the
  ``context_brief``). Helpers are exposed both as plain functions for the
  runner and as tool wrappers (``ContextPutTool`` / ``ContextGetTool``)
  granted to agents via the tool registry.

Cross-task episodic memory (the ``hyperion_memory`` Qdrant collection,
``store_episode`` / ``recall_similar_tasks``) was removed on 2026-10-26 when
Qdrant was retired: it had not been read in months. The last 42 records are
archived at ``~/archive/qdrant-2026-10-26/``.

Design notes:
- This ``__init__`` is intentionally an empty package marker — it declares the
  ``hyperion.memory`` namespace without re-exporting submodule symbols, so
  callers import explicitly (e.g. ``from hyperion.memory.context_store import
  context_put``).
"""
