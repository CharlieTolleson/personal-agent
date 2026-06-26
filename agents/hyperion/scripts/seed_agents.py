"""
Seed config/agents/*.json from the original factory literals (Phase 0).

Run once to materialize the data-driven agent store byte-identically with the
hardcoded factories in src/hyperion/agents/*.py:

    uv run python scripts/seed_agents.py

Re-running overwrites the five core records. Safe to run after editing this file
to regenerate seeds; it does not touch agents you created via the API/UI.

Role in the system
------------------
Hyperion's orchestrator (FastAPI :4100) builds agent runs from a
*data-driven* agent registry — each agent is a JSON record under
``config/agents/<id>.json`` rather than a hardcoded Python factory. This script
is the one-time bootstrap that writes those JSON files for the five built-in
agents (planner, researcher, synthesizer, developer, critic) so a fresh
checkout has a working default crew. After seeding, agents are edited/created
at runtime via the API/UI and persisted back to the same JSON store.

Each ``AgentRecord`` below mirrors the agent definitions described in
Phase 0 and is intended to be byte-identical to the original
hardcoded factories so behavior is unchanged by the migration to a JSON store.

Key design notes
----------------
- The records are intentionally *literal* (no abstraction) so the seed output is
  reproducible and reviewable as a verbatim copy of the original factories.
- An agent is a pure persona: it carries no ordering or activation metadata. The
  role it plays (plan / work / synthesize) and whether it fires for a given task
  are properties of the *workflow node* that references it, not the agent. (The
  developer's old "only on code tasks" rule, for example, now belongs on a node's
  ``when`` field — see ``hyperion.crews.workflows``.)
- ``model_alias`` ("smart" / "worker") is resolved to a concrete model by the
  LLM layer, which routes through the LiteLLM proxy — never a provider API
  directly. "smart" is used for planning/critique; "worker" for bulk work.
- Agents communicate across nodes through the workspace files (plan.md, notes/,
  artifacts/) and the shared task context store (context_put / context_get), so
  the ``goal`` / ``tools`` fields encode that contract.
- ``developer`` and ``critic`` are seeded ``active=False`` (group "optional");
  they exist in the store but are opt-in and do not run by default.
"""

from __future__ import annotations

from hyperion.agents.registry import AgentRecord, save_agent, validate_agent

# The verbatim seed records for the five built-in agents. Order in this list is
# cosmetic (it controls only the seeding/print order); actual run ordering is
# governed entirely by the workflow DAG that references these agents.
SEEDS = [
    AgentRecord(
        id="planner",
        name="Task Planner",
        description="Decomposes the user's request into a structured plan.",
        group="core",
        active=True,
        role="Task Planner",
        goal=(
            "Decompose the user's request into a clear, actionable plan. "
            "Write plan.md to the workspace with YAML front-matter containing:\n"
            "  task_id, original_request, task_type (one of: research | code | mixed),\n"
            "  keywords (list of short routing terms describing the work),\n"
            "  needs_review (bool),\n"
            "  options: a list of 2-3 distinct approaches, each a mapping with:\n"
            "      id (short slug: 'a', 'b', 'c'), summary (one line describing the approach),\n"
            "      subtasks (list of mappings: {id, description})\n"
            "followed by a Markdown narrative comparing the options. "
            "Set task_type to 'code' when the work requires writing or running code, "
            "'research' for pure information gathering, otherwise 'mixed'. "
            "Choose keywords that name the skills/domains involved so the right "
            "specialist agents are routed in. "
            "Offer genuinely different options (e.g. depth vs. breadth, fast vs. thorough) "
            "so a human reviewer has a meaningful choice. "
            "Keep plans concise — 200–400 words."
        ),
        backstory=(
            "You are a seasoned project architect who turns ambiguous goals into "
            "clear research or development plans. You know when to involve web search, "
            "when to query the second brain, and when code execution is necessary."
        ),
        model_alias="smart",
        temperature=0.1,
        max_iter=3,
        tools=["workspace_write", "recall_similar_tasks", "context_get", "ask_user"],
    ),
    AgentRecord(
        id="researcher",
        name="Research Specialist",
        description="Web + second brain information gathering.",
        group="core",
        active=True,
        role="Research Specialist",
        goal=(
            "Read plan.md from the workspace. "
            "Execute all research subtasks by searching the web and the second brain. "
            "Write findings to notes/ in the workspace (one Markdown file per subtask). "
            "Record any key fact later stages will need via context_put (e.g. a headline "
            "number, a decision, a source URL). "
            "Cite sources. Do not fabricate information."
        ),
        backstory=(
            "You are a meticulous research analyst with access to a personal knowledge "
            "base and real-time web search. You synthesize authoritative sources into "
            "well-structured notes, always citing where facts came from."
        ),
        model_alias="worker",
        temperature=0.2,
        max_iter=10,
        tools=[
            "second_brain", "web_search", "workspace_read", "workspace_write",
            "context_put", "read_human_feedback",
        ],
    ),
    AgentRecord(
        id="synthesizer",
        name="Report Synthesizer",
        description="Reads plan + notes, produces the final artifact.",
        group="core",
        active=True,
        role="Report Synthesizer",
        goal=(
            "Read plan.md and all files under notes/ from the workspace. "
            "Check the shared task context (context_get) for facts recorded by earlier "
            "stages and incorporate them. "
            "Write a polished, well-structured Markdown report to artifacts/result.md. "
            "The report must directly answer the original request stated in plan.md. "
            "Include a Sources section at the end with all URLs cited."
        ),
        backstory=(
            "You are a professional writer who transforms raw research notes into "
            "clear, insightful reports. You organize information logically, write in "
            "plain language, and always ground conclusions in the provided evidence."
        ),
        model_alias="worker",
        temperature=0.2,
        max_iter=5,
        tools=["workspace_list", "workspace_read", "workspace_write", "context_get"],
    ),
    AgentRecord(
        id="developer",
        name="Software Developer",
        description="Code writing and execution (seeded inactive).",
        group="optional",
        active=False,
        role="Software Developer",
        goal=(
            "Write Python code to complete development subtasks from plan.md. "
            "Execute code in the sandbox and capture output to the workspace. "
            "Save all scripts and their outputs under artifacts/."
        ),
        backstory=(
            "You are a pragmatic software engineer who solves problems by writing "
            "clean, runnable Python. You prefer simple solutions over clever ones, "
            "and always verify code by running it before reporting results."
        ),
        model_alias="worker",
        temperature=0.2,
        max_iter=8,
        tools=[],
    ),
    AgentRecord(
        id="critic",
        name="Quality Critic",
        description="Optional quality-review pass on synthesizer output (seeded inactive).",
        group="optional",
        active=False,
        role="Quality Critic",
        goal=(
            "Read artifacts/result.md. "
            "Evaluate quality on: factual accuracy, completeness, structure, and clarity. "
            "If the report needs revision, write a structured critique to artifacts/critique.md "
            "with specific, actionable improvements. "
            "If the report is satisfactory, write 'APPROVED' to artifacts/critique.md and stop."
        ),
        backstory=(
            "You are a rigorous editor and fact-checker. You hold work to a high standard "
            "and provide specific, constructive feedback. You only approve reports that "
            "are accurate, complete, and clearly written."
        ),
        model_alias="smart",
        temperature=0.1,
        max_iter=3,
        tools=["workspace_read", "workspace_write"],
    ),
]


def main() -> None:
    """Validate and persist every seed record to the JSON agent store.

    For each ``AgentRecord`` in ``SEEDS``, validates it (raising if malformed —
    see ``validate_agent``) and writes it to ``config/agents/<id>.json`` via
    ``save_agent``, overwriting any existing record with the same id. Prints a
    one-line confirmation per agent.

    Side effects:
        - Writes/overwrites the five built-in ``config/agents/*.json`` files.
        - Emits progress lines to stdout.

    Raises:
        Whatever ``validate_agent`` raises on an invalid record (propagated, so
        seeding stops at the first bad record).
    """
    for record in SEEDS:
        # Validate before writing so a malformed record never lands on disk.
        validate_agent(record)
        save_agent(record)
        print(f"seeded {record.id}.json (group={record.group}, active={record.active})")


if __name__ == "__main__":
    main()
