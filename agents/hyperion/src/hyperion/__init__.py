"""Hyperion — self-hosted multi-agent AI system (top-level package marker).

This is the package root for ``hyperion``, the multi-agent orchestrator (owned
LiteLLM agent loop) that lives under ``agents/hyperion/`` in Charlie's ~/ai workspace.
Importing ``hyperion`` (or any submodule) executes this file; it currently does
no setup beyond declaring the package and its purpose.

Role in the system
------------------
Hyperion coordinates a set of specialized agents (planner, researcher,
developer, critic, synthesizer) over configurable workflow DAGs to complete
multi-step tasks. The package is organized into submodules such as:

- ``crews/`` — agent/workflow definitions and the workflow ``runner``.
- ``server/`` — FastAPI surface (``api``, MCP, webhooks, affordances) on :4100.
- ``llms`` — model wiring; all LLM calls route through the LiteLLM proxy
  (``http://localhost:4000/v1``) per the workspace-wide convention, never
  directly to provider APIs.
- memory/usage/feedback/scheduler/alerts/observability helpers backed by
  Qdrant (episodic memory) and Langfuse (tracing).

Design notes
-----------
- This module is intentionally minimal: keeping ``__init__`` side-effect-free
  means importing the package (e.g. for tooling, tests, or partial imports)
  never triggers network calls, config loading, or service connections. Any
  such initialization belongs in the relevant submodule, not here.
- No ``__all__`` or eager re-exports are defined, so consumers import the
  concrete submodule they need (e.g. ``from hyperion.server import api``).
"""
