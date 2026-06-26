"""Hyperion ``crews`` package: crew assembly and workflow execution.

This package is the orchestration core that turns a Hyperion task request into a
sequence of agent steps and runs them to completion. It sits between the
HTTP/MCP server layer (``hyperion.server``) and the individual agent definitions,
and is responsible for *how* agents are wired together and *in what order* they run.

Role in the system
-------------------
A request enters via the server, is shaped by the agent/tool/memory layers, and is
then handed to this package to build and execute a "crew" (a coordinated set of
agents working a task). The submodules divide that responsibility as follows:

- ``runner``        : The main execution engine. Drives a workflow DAG (or the
                      default linear pipeline) end to end, invoking agents,
                      threading context/memory between steps, recording usage,
                      and emitting trace/observability events.
- ``workflows``     : Loads and resolves the free-form workflow DAG definitions
                      (the JSON files under ``config/workflows/``) into an
                      executable step graph for the runner. ``resolve_workflow``
                      picks which workflow applies to a run (e.g. default vs.
                      research-critique).
- ``plan_contract`` : Defines the structured plan/contract schema that the planner
                      agent produces and that downstream steps consume.
- ``default``       : The built-in default crew/workflow used when no explicit
                      workflow is selected.

Design notes
------------
- This ``__init__`` is intentionally an empty package marker (carries no exports
  and runs no import-time side effects). Consumers import the concrete submodules
  directly, e.g. ``from hyperion.crews.runner import ...``. Keeping it empty avoids
  import cycles between the submodules and keeps package import cheap.
- All LLM access from this package routes through the LiteLLM proxy
  (http://localhost:4000/v1) per the workspace convention; agents/crews never call
  provider APIs directly.
"""
