# personal-agent

A self-hosted AI workspace: a multi-provider LLM gateway, a multi-agent research orchestrator, a vector-search second brain, and a suite of Claude Code slash commands — all wired together with Docker Compose and running on a single laptop.

---

## What it is

This repo is a fully integrated personal AI operating system, built from scratch. The core components are:

| Component | What it does |
|---|---|
| **ai-router** | Docker Compose stack that wires together every backing service on a shared bridge network (`ai-net`). One `docker compose up -d` starts everything. |
| **Hyperion** | A multi-agent research orchestrator (owned LiteLLM agent loop + FastAPI). Accepts a natural-language task, decomposes it into a plan, researches via web + second brain, and produces a Markdown report. |
| **Hyperion UI** | React/Vite web console for managing agents, building workflows visually, monitoring live runs, and reconfiguring models without restarting the service. |
| **Second brain** | An Obsidian vault ingested into Qdrant, searchable by all agents and tools via semantic similarity + reranking. |
| **Skills** | Claude Code slash commands (`/hyperion`, `/research`) and Open WebUI tool plugins that surface Hyperion to any chat interface. |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     ai-net (Docker bridge)               │
│                                                          │
│  LiteLLM :4000  ←── every LLM call routes here          │
│      │                                                   │
│      ├─ Open WebUI :3000   (chat UI + tools)             │
│      ├─ Hyperion API :4100  (orchestrator)               │
│      ├─ Hyperion MCP :4101  (MCP server for Claude Code) │
│      └─ Hyperion UI  :4102  (React console)              │
│                                                          │
│  Qdrant :6333    (vector DB — second brain + memory)     │
│  Langfuse :3001  (LLM tracing + observability)           │
│  SearXNG :8888   (self-hosted federated web search)      │
│  Infinity :7997  (BAAI/bge-reranker-base)               │
│  Postgres        (LiteLLM virtual keys + Langfuse store) │
└──────────────────────────────────────────────────────────┘
```

**Key design decision:** every LLM call in the system — from Open WebUI, from Hyperion agents, from agent scripts — routes through the LiteLLM proxy. No component calls a provider API directly. This means all traffic is observable in Langfuse, all spending is controlled by virtual-key budgets, and swapping a model is a config file edit, not a code change.

---

## Hyperion — the orchestrator

Hyperion is the main project in this repo. It's a production-quality multi-agent system built on an owned LiteLLM tool-calling agent loop, with a number of architectural layers:

### Workflow engine
Tasks run as DAGs defined in JSON. The runner topo-sorts nodes (Kahn's algorithm), validates for cycles (including cross-workflow subworkflow cycles), and executes independent branches in parallel. Workflows are CRUD-managed via the API and editable visually in the React UI's workflow builder (React Flow).

```json
// config/workflows/research-critique.json
{
  "nodes": [
    { "id": "plan",     "kind": "plan",       "agent": "planner",     "upstream": [] },
    { "id": "research", "kind": "work",        "agent": "researcher",  "upstream": ["plan"] },
    { "id": "critique", "kind": "work",        "agent": "critic",      "upstream": ["plan"] },
    { "id": "synthesis","kind": "synthesize",  "agent": "synthesizer", "upstream": ["research","critique"] }
  ]
}
```

Sub-workflow nodes let you compose workflows: a node can invoke another whole workflow as a single step, with cross-workflow cycle detection at validation time.

### Human-in-the-loop
Any workflow node can set `gate_before: true` to pause for human approval before executing. Pause/resume state is persisted to SQLite (`aiosqlite`), so a paused task survives a server restart. The API exposes `/tasks/{id}/approve` and `/tasks/{id}/feedback` for the HITL flow.

### Memory
- **Episodic memory**: completed task summaries are written to a `hyperion_memory` Qdrant collection and injected as context on subsequent related tasks.
- **Second brain RAG**: the researcher agent queries the Qdrant `second_brain` collection (the Obsidian vault, text-embedding-3-small, 1536-dim) and reranks results through Infinity before passing them to the LLM.

### Model routing
Each agent role (`smart` / `worker` / `cheap` / `fast`) maps to a LiteLLM alias. Each alias is a prioritized provider list — Claude → Gemini 2.5 Pro → GPT-4o — with automatic fallback when a key is missing or a call fails. Role-to-model assignments are overridable at runtime via `PUT /config` without a restart, and persisted to `config/models.json` for the next boot.

Hyperion runs with a budget-capped LiteLLM virtual key ($10/day, 60 rpm), so runaway agent loops can't drain the account.

### Observability
Every LLM call is traced in Langfuse with session-level grouping by task ID. The API response includes a deep-link to the Langfuse trace so you can inspect any run end-to-end.

### Integrations
- **MCP server** (`:4101`): exposes Hyperion as a Model Context Protocol server, registered in `.mcp.json` so Claude Code can submit tasks natively.
- **Agent card** (`/.well-known/agent.json`): a discovery endpoint for external orchestrators.
- **Webhooks**: pass a `callback_url` on task submission; Hyperion POSTs the terminal result exactly once. Webhook hosts are validated against an SSRF allowlist (private/loopback only) before any request is sent.
- **Notion**: finished task artifacts can be saved to a Notion database via `POST /tasks/{id}/save-to-notion`.
- **Config portability**: `GET /config/export` → zip, `POST /config/import` → atomic validated import.
- **n8n**: drives Hyperion from an n8n Schedule node via the HTTP API + callback webhook, no Hyperion-side scheduler required.

### Test suite
```
tests/
├── test_api.py             # FastAPI endpoint coverage
├── test_agents_api.py      # agent CRUD
├── test_compiler.py        # workflow validation + topo-sort
├── test_subworkflow.py     # cross-workflow DAG + cycle detection
├── test_hitl.py            # human-in-the-loop pause/resume
├── test_context.py         # episodic memory injection
├── test_agent_loop.py      # owned LiteLLM tool-calling loop
└── ...
```

---

## Getting started

### Prerequisites
```bash
cd ~/ai/ai-router && docker compose up -d
```

Add whichever provider keys you have to `ai-router/.env` (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`). The system runs on any one of them.

### Start Hyperion
```bash
# Development
cd agents/hyperion
uv run hyperion-api      # :4100
uv run hyperion-mcp      # :4101 (second terminal)

# Or via Docker (full stack)
cd ai-router
docker compose -f docker-compose.yml -f ../agents/hyperion/docker-compose.override.yml up -d
```

### Submit a task
```bash
# CLI
curl -X POST http://localhost:4100/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Research the current state of multi-agent AI systems"}'

# Claude Code (via MCP)
/hyperion research the current state of multi-agent AI systems

# Open WebUI (via tool plugin)
# → Admin Panel → Tools → paste skills/tools/hyperion_delegate.py
```

### Ingest the second brain
The vault lives at `~/secondbrain` (outside this repo); its ingest runtime/venv stays here in `secondbrain-pipeline/`.
```bash
cd ~/secondbrain
source ~/ai/secondbrain-pipeline/.venv/bin/activate
python ingest_obsidian.py --incremental
```

---

## Stack

| Layer | Technology |
|---|---|
| Agent loop | Owned LiteLLM tool-calling loop (`agent_loop.py`) |
| LLM proxy | LiteLLM (digest-pinned Docker image) |
| API | FastAPI + aiosqlite |
| Vector DB | Qdrant (REST + gRPC) |
| Embeddings | OpenAI text-embedding-3-small |
| Reranker | BAAI/bge-reranker-base via Infinity |
| Web search | SearXNG (self-hosted, federated) |
| Observability | Langfuse |
| Frontend | React + Vite + TypeScript + Tailwind + React Flow |
| Package management | uv |
| Infrastructure | Docker Compose |
| MCP | Model Context Protocol (server + client) |

---

## Repository layout

```
ai-router/          Docker Compose stack definition + service configs
agents/
  hyperion/         Multi-agent orchestrator (the main project)
    src/hyperion/
      agents/       Planner, Researcher, Synthesizer, Developer, Critic
      crews/        Crew assembly, workflow runner, DAG compiler
      memory/       Episodic memory (Qdrant) + context injection
      server/       FastAPI (:4100), MCP server (:4101), webhooks
      tools/        Web search, second brain RAG, reranker, Notion
    config/
      agents/       Agent definitions (JSON, operator-editable)
      workflows/    Workflow DAGs (JSON, operator-editable)
    tests/          pytest suite
  hyperion-ui/      React/Vite web console (:4102)
  _shared/          Shared utilities (Qdrant client, Notion client, web search)
secondbrain-pipeline/  Vault ingest runtime (.venv, .env, state) → Qdrant; vault itself at ~/secondbrain
skills/
  commands/         Claude Code slash commands
  tools/            Open WebUI tool plugins
  hooks/            Claude Code event hooks
```
