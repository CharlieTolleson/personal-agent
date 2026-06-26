# Hyperion

Self-hosted multi-agent AI system. Decomposes natural-language tasks, researches via web + second brain, optionally executes code, and synthesizes polished Markdown reports.

**Stack:** LiteLLM proxy · Qdrant · SearXNG · Infinity reranker · Langfuse

## Services

| Service | Port | Purpose |
|---|---|---|
| Hyperion API | 4100 | FastAPI: submit tasks, poll status, stream progress |
| Hyperion MCP | 4101 | MCP server for Claude Code integration |
| SearXNG | 8888 | Self-hosted web search |
| Infinity | 7997 | Reranker (BAAI/bge-reranker-base) |
| Langfuse | 3001 | Tracing UI |

All services are on `ai-net` alongside the existing LiteLLM proxy (4000) and Qdrant (6333).

## Setup

### 1. Prerequisites

All infra services in `~/ai/ai-router/docker-compose.yml` must be running:

```bash
cd ~/ai/ai-router
docker compose up -d
```

**Provider keys are optional.** Add whichever you have to `~/ai/ai-router/.env` (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`). The role aliases list every provider in priority order and LiteLLM skips ones whose key is missing — so the system runs fully on Gemini + OpenAI alone. `ANTHROPIC_API_KEY` is intentionally absent here (Claude is used via the Max subscription, not pay-per-use API); see "Model routing" below for why this surfaces as failed Claude attempts in Langfuse and how to silence them.

### 2. Create Hyperion LiteLLM virtual key

```bash
export LITELLM_MASTER_KEY=<your key from ai-router/.env>
bash scripts/create_hyperion_key.sh
# Copy the output key to agents/hyperion/.env as LITELLM_HYPERION_KEY
```

### 3. Initialize Qdrant hyperion_memory collection

```bash
uv run python scripts/init_qdrant.py
```

### 4. Run locally (development)

```bash
# FastAPI service (port 4100)
uv run hyperion-api

# MCP server (port 4101) — in a second terminal
uv run hyperion-mcp
```

### 5. Run via Docker

```bash
cd ~/ai/ai-router
docker compose -f docker-compose.yml -f ../agents/hyperion/docker-compose.override.yml up -d hyperion hyperion-mcp
```

## Usage

### From Open WebUI

Paste `~/ai/skills/tools/hyperion_delegate.py` into **Admin Panel → Workspace → Tools**.
Then in any chat: `delegate_task: summarize the current state of my second brain re: career goals`

### From Claude Code

The MCP server is registered in `~/ai/.mcp.json`. Use the `/hyperion` skill:

```
/hyperion summarize my career goals from my second brain
/hyperion --critic write a detailed analysis of my investment portfolio
/hyperion --large research the latest developments in agentic AI systems
```

### Direct API

```bash
# Submit a task
curl -X POST http://localhost:4100/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Summarize my career goals from my second brain"}'

# Poll for status
curl http://localhost:4100/tasks/{task_id}

# Stream progress
curl http://localhost:4100/tasks/{task_id}/stream

# Get result
curl http://localhost:4100/tasks/{task_id}/artifacts/result.md
```

## Orchestration & integrations (Phase 9)

### Agent card

External orchestrators discover Hyperion's contract at a well-known path:

```bash
curl http://localhost:4100/.well-known/agent.json
```

It reports `schema_version`, the submit/status/stream/approve/feedback endpoints, and the
active plan/synthesize skills.

### Completion webhooks

Pass a `callback_url` when submitting; Hyperion POSTs the terminal result there **exactly
once** (on `done` or `failed`):

```bash
curl -X POST http://localhost:4100/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Research X", "callback_url": "http://host.docker.internal:5678/webhook/hyperion"}'
# → POST body: {"task_id", "status", "error", "result_path"}
```

**SSRF guard:** the callback host must resolve entirely to a private/loopback/link-local
address — public hosts are rejected with 422 before any request is made. Disable only on a
trusted network with `HYPERION_CALLBACK_SSRF_GUARD=off`.

### n8n / cron recipe (no new service)

Drive Hyperion from an existing n8n instance — no Hyperion-side scheduler needed:

1. **n8n Schedule (cron) node** → fires on your interval.
2. **HTTP Request node** → `POST http://hyperion:4100/tasks` (use the Docker service name on
   the shared network, or `host.docker.internal:4100` from a host-network n8n) with body
   `{"task": "...", "callback_url": "http://n8n:5678/webhook/<id>"}`.
3. **Webhook node** (the `callback_url`) → receives the result when the run finishes; branch
   on `status` and fan out to Slack/Notion/etc.

This keeps n8n as the scheduler/router and Hyperion as the worker. For Hyperion-native
schedules instead, set a `cron` trigger on an agent record (Phase 8 scheduler).

### Save a result to Notion

The Synthesizer's "save to Notion" follow-up (and the matching UI button) writes a finished
task's artifact to a Notion database. Set `NOTION_API_KEY` and `NOTION_DATABASE_ID` (compose
passes them through); absent either, the endpoint returns a clear "not configured" 502.

```bash
curl -X POST http://localhost:4100/tasks/{task_id}/save-to-notion -d '{}'
```

### Back up / move the agent store

```bash
curl -OJ http://localhost:4100/config/export                       # → hyperion-config.zip
curl -F "file=@hyperion-config.zip" http://localhost:4100/config/import
```

Import validates every record (DAG + at-least-one plan/synthesize) before writing — a
malformed archive is rejected atomically.

## Model routing & agent architecture

The crew is **fixed**: exactly three agents run in sequence — Planner → Researcher → Synthesizer. There is no dynamic agent spawning or task-dependent agent selection. Each agent has a capped ReAct iteration budget (one LLM call per iteration):

| Agent | LLM factory | Alias | `max_iter` | Effective model (no Anthropic key) |
|---|---|---|---|---|
| Planner | `planner_llm` | `smart` | 5 | Claude Opus → **Gemini 2.5 Pro** → GPT-4o |
| Researcher | `worker_llm` | `worker` | 10 | Claude Sonnet → **Gemini 2.5 Pro** → GPT-4o |
| Synthesizer | `worker_llm` | `worker` | 5 | Claude Sonnet → **Gemini 2.5 Pro** → GPT-4o |

(`developer` and `critic` agents exist in `agents/` but are not wired into the default crew yet.)

**How a model is chosen:** each agent calls a role factory in `llms.py`, which resolves to an alias (`smart`/`worker`/`cheap`/`fast`) defined in `ai-router/litellm_config.yaml`. Each alias is a multi-provider group listed in priority order; LiteLLM tries them top-down and falls back when a key is missing or a call fails. Override per-alias without code changes via env vars in `agents/hyperion/.env`: `MODEL_PLANNER`, `MODEL_WORKER`, `MODEL_CHEAP`.

**Why you see failed Claude calls in Langfuse:** the `smart`/`worker` aliases list Claude first, but `ANTHROPIC_API_KEY` is absent — so LiteLLM logs a failed Claude attempt, then falls back to Gemini. This is harmless. To silence it, reorder the alias entries in `litellm_config.yaml` so Gemini comes first, then `docker compose restart litellm`.

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run a quick smoke test
uv run python -c "
from hyperion.config import settings
print('LiteLLM:', settings.litellm_base_url)
print('Models: planner=%s worker=%s cheap=%s' % (settings.model_planner, settings.model_worker, settings.model_cheap))
"
```

## Project layout

```
src/hyperion/
├── config.py            env loading
├── llms.py              per-role LLM factories
├── observability.py     Langfuse handler
├── agents/              Planner, Researcher, Synthesizer, Developer, Critic
├── tools/               second_brain, web_search, reranker, workspace
├── crews/default.py     crew assembly + circuit breakers
├── memory/episodic.py   task history in Qdrant hyperion_memory
└── server/
    ├── api.py           FastAPI (port 4100)
    └── mcp.py           MCP server (port 4101)
```

## Phases

All 10 phases complete. (Phase summaries below; the living architecture
reference is `~/secondbrain/Projects/AgentArchitecture/architecture.md`.)

- **Phase 0** Contracts & substrate (LiteLLM, Langfuse, SearXNG, Infinity, Qdrant collection)
- **Phase 1** Unified stage-runner (tools, crew, FastAPI, MCP, OWUI tool, Claude Code surface)
- **Phase 2** Routing engine (per-role/per-alias model routing, fallbacks)
- **Phase 3** Pausable execution + human-in-the-loop plan gate
- **Phase 4** Context layer (episodic memory, second-brain RAG)
- **Phase 5** Agent CRUD + options API (registry-backed agents)
- **Phase 6** Feedback, alerts, affordances
- **Phase 7** Web UI (`hyperion-ui`, :4102)
- **Phase 8** Thresholds, metrics, monitoring, scheduler
- **Phase 9** Orchestration & polish (agent card, webhooks/SSRF, save-to-Notion, config export/import)
