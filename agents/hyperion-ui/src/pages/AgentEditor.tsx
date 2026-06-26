/**
 * AgentEditor — create / edit / duplicate / delete a single Hyperion agent.
 *
 * Role in the system:
 *   This is the Hyperion UI (React + Vite) page mounted at `/agents/new` and
 *   `/agents/:id`. It is the human-facing front end for the agent registry that
 *   the Hyperion orchestrator (FastAPI :4100) persists and uses to assemble its
 *   multi-agent pipeline. An agent is a pure *persona*: every field on this form
 *   maps 1:1 onto an `AgentRecord` (see ../api/client) which the backend stores
 *   and reads when building agent runs: identity/grouping, the prompt triple
 *   (role/goal/backstory), model routing (alias + optional fallback), tool
 *   grants, an optional `schedule_cron`, and per-run safety thresholds (token
 *   caps, activations/day circuit breaker). Ordering and activation are *not*
 *   here — they live on the workflow that references the agent.
 *
 * Data flow / design notes:
 *   - All server state is fetched and mutated through TanStack Query hooks in
 *     ../api/client (useAgent, useTools, useModels, useSaveAgent,
 *     useDeleteAgent, useDuplicateAgent). This component holds only a single
 *     local draft copy (`rec`) of the record being edited.
 *   - `isNew` is derived from the absence of a route `:id` param. It drives
 *     create-vs-update behaviour: the ID field is locked after creation, the
 *     save hook targets POST vs PUT, and Duplicate/Delete are hidden for new
 *     records.
 *   - The draft is seeded from `blankAgent()` and then overwritten once the
 *     existing record loads (see the useEffect), so the form is editable
 *     immediately even before the fetch resolves.
 *   - Model routing convention: an *alias* (smart/worker/cheap/fast) inherits
 *     the LiteLLM proxy's routing + provider fallback; a concrete model id pins
 *     one model exactly. See modelChoices for why aliases are listed first and
 *     deduped against the proxy's model list.
 *   - Guardrail banner: warns when editing a `core` agent, since its prompt
 *     shapes the default pipeline.
 */
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  useAgent,
  useDeleteAgent,
  useDuplicateAgent,
  useModels,
  useSaveAgent,
  useTools,
  type AgentRecord,
} from "../api/client";
import InfoTip from "../components/InfoTip";

/**
 * Build a fresh, fully-defaulted AgentRecord for the "new agent" form.
 *
 * Every field on the form is initialised here so the inputs are always
 * controlled (never undefined) from first render. Defaults mirror Hyperion's
 * conventions: a worker-model agent, low temperature, active, no schedule.
 *
 * @returns A blank AgentRecord with sensible defaults and an empty `id`/`name`.
 */
function blankAgent(): AgentRecord {
  return {
    id: "",
    name: "",
    description: "",
    group: "optional",
    active: true,
    role: "",
    goal: "",
    backstory: "",
    model_alias: "worker",
    fallback_alias: null,
    temperature: 0.2,
    top_p: null,
    max_tokens: null,
    max_iter: 5,
    tools: [],
    schedule_cron: null,
    thresholds: {
      max_input_tokens: null,
      max_output_tokens: null,
      max_activations_per_day: null,
    },
  };
}

/**
 * Page component for creating or editing one Hyperion agent.
 *
 * Reads the optional `:id` route param to decide between create (`isNew`) and
 * edit modes, loads the existing record plus the available tools/models, and
 * renders a sectioned form (Identity, Prompt, Model, Tools, Schedule, Thresholds)
 * bound to a local draft. On save/delete it navigates back to the dashboard
 * ("/"); on duplicate it navigates to the new clone's edit page.
 *
 * @returns The agent editor page element.
 */
export default function AgentEditor() {
  const { id } = useParams();
  // No route id => we're creating a new agent rather than editing an existing one.
  const isNew = !id;
  const nav = useNavigate();

  // Server state via TanStack Query: the record under edit, plus the option
  // lists (tools, models).
  const existing = useAgent(id);
  const { data: tools } = useTools();
  const { data: models } = useModels();

  // Mutations. `useSaveAgent(isNew)` selects create vs update semantics.
  const save = useSaveAgent(isNew);
  const del = useDeleteAgent();
  const dup = useDuplicateAgent();

  // The single editable draft. Starts blank, then is replaced by the fetched
  // record once it loads (below). All inputs are controlled against this.
  const [rec, setRec] = useState<AgentRecord>(blankAgent());

  // Seed the draft from the server once the existing record arrives. Runs again
  // if the fetched record reference changes (e.g. after a refetch).
  useEffect(() => {
    if (existing.data) setRec(existing.data);
  }, [existing.data]);

  /**
   * Immutably update a single top-level field of the draft record.
   *
   * Generic over the key so the value type is checked against AgentRecord[K].
   *
   * @param k Top-level AgentRecord field to set.
   * @param v New value for that field.
   */
  function set<K extends keyof AgentRecord>(k: K, v: AgentRecord[K]) {
    setRec((r) => ({ ...r, [k]: v }));
  }
  /**
   * Update a numeric threshold from a raw text input, coercing to number|null.
   *
   * An empty string maps to `null` ("inherit global / no limit"); any other
   * value is parsed with Number(). The threshold object is updated immutably.
   *
   * @param k Threshold field to set (max_input_tokens / max_output_tokens / max_activations_per_day).
   * @param v Raw input string ("" => null).
   */
  function setThreshold(k: keyof AgentRecord["thresholds"], v: string) {
    const n = v === "" ? null : Number(v);
    setRec((r) => ({ ...r, thresholds: { ...r.thresholds, [k]: n } }));
  }
  /**
   * Add or remove a tool grant by name (checkbox toggle).
   *
   * Removes the tool if already granted, otherwise appends it. Updates the
   * draft's `tools` array immutably.
   *
   * @param name Tool identifier to toggle.
   */
  function toggleTool(name: string) {
    setRec((r) => ({
      ...r,
      tools: r.tools.includes(name) ? r.tools.filter((t) => t !== name) : [...r.tools, name],
    }));
  }

  /**
   * Persist the draft (create or update per `isNew`) and return to the
   * dashboard on success. Errors surface via `save.isError` in the UI.
   */
  function onSave() {
    save.mutate(rec, { onSuccess: () => nav("/") });
  }
  /**
   * Delete the current agent after a confirm() prompt, then return to the
   * dashboard. No-op when creating (no `id`).
   */
  function onDelete() {
    if (!id) return;
    if (!confirm(`Delete agent "${rec.name}"?`)) return;
    del.mutate(id, { onSuccess: () => nav("/") });
  }
  /**
   * Duplicate the current agent and navigate to the new clone's edit page.
   * No-op when creating (no `id`).
   */
  function onDuplicate() {
    if (!id) return;
    dup.mutate(id, { onSuccess: (clone) => nav(`/agents/${clone.id}`) });
  }

  // Show a spinner only while editing (not creating) and the fetch is in flight,
  // so the form isn't rendered against the still-blank default record.
  if (!isNew && existing.isLoading) return <p className="text-slate-400">Loading…</p>;

  // Fall back to the canonical alias set if the proxy hasn't reported aliases yet.
  const aliasList = models?.aliases ?? ["smart", "worker", "cheap", "fast"];
  // Whether this agent belongs to the default-pipeline "core" group; gates the
  // edit-carefully warning banner.
  const editingCore = rec.group === "core";

  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="mb-4 text-lg font-semibold">{isNew ? "New agent" : `Edit: ${rec.name}`}</h2>

      {/* Guardrail banner: for existing core agents, whose prompt shapes the
          default pipeline. */}
      {editingCore && !isNew && (
        <div className="card mb-4 border-amber-500/40 bg-amber-500/10 text-sm text-amber-200">
          <p>This is a <b>core</b> agent — its prompt shapes the default pipeline. Edit carefully.</p>
        </div>
      )}

      <div className="space-y-4">
        {/* Identity */}
        <div className="card grid grid-cols-2 gap-3">
          <div>
            <label className="label">
              ID (slug)
              <InfoTip text="Unique identifier used in routing and config files. Lowercase letters, numbers, hyphens and underscores only. Fixed once created." />
            </label>
            <input
              className="input"
              value={rec.id}
              disabled={!isNew}
              onChange={(e) => set("id", e.target.value)}
            />
          </div>
          <div>
            <label className="label">
              Name
              <InfoTip text="Human-readable display name shown on the dashboard and in monitoring." />
            </label>
            <input className="input" value={rec.name} onChange={(e) => set("name", e.target.value)} />
          </div>
          <div className="col-span-2">
            <label className="label">
              Description
              <InfoTip text="Short summary of what this agent does. Shown on its dashboard card." />
            </label>
            <input
              className="input"
              value={rec.description}
              onChange={(e) => set("description", e.target.value)}
            />
          </div>
          <div>
            <label className="label">
              Group
              <InfoTip text="Organizational label used to group and filter agents on the dashboard (e.g. core, optional). Agents in the 'core' group form the default pipeline." />
            </label>
            <input className="input" value={rec.group} onChange={(e) => set("group", e.target.value)} />
          </div>
          <label className="col-span-2 flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={rec.active}
              onChange={(e) => set("active", e.target.checked)}
            />
            Active
          </label>
        </div>

        {/* Prompt */}
        <div className="card space-y-3">
          <div>
            <label className="label">
              Role
              <InfoTip text="The agent's job title / persona, e.g. 'Senior Research Analyst'. Sets the perspective the model adopts when reasoning." />
            </label>
            <input className="input" value={rec.role} onChange={(e) => set("role", e.target.value)} />
          </div>
          <div>
            <label className="label">
              Goal
              <InfoTip text="The objective this agent pursues each run — its core instruction. Be specific; this is the primary driver of the model's behavior." />
            </label>
            <textarea
              className="input min-h-[120px] resize-y"
              value={rec.goal}
              onChange={(e) => set("goal", e.target.value)}
            />
          </div>
          <div>
            <label className="label">
              Backstory
              <InfoTip text="Background and personality that shape how the agent approaches its goal — expertise, tone, and any standing constraints." />
            </label>
            <textarea
              className="input min-h-[80px] resize-y"
              value={rec.backstory}
              onChange={(e) => set("backstory", e.target.value)}
            />
          </div>
        </div>

        {/* Model — grouped <select> (aliases first, then concrete model ids).
            Using <select> rather than <input list> avoids the browser-native
            datalist filtering that shows only the option matching the current
            value (e.g. "worker" → only "worker" in the dropdown). The select
            always shows every option regardless of the current value.
            When an alias is selected, a fallback-chain hint renders below the
            control so operators can see what's behind smart/worker/cheap/fast. */}
        <div className="card grid grid-cols-2 gap-3">
          <div>
            <label className="label">
              Model
              <InfoTip text="Which model powers this agent. Pick a role alias (smart/worker/cheap/fast) to inherit the global routing with automatic provider fallback, or choose a concrete model id (e.g. gemini-2.5-pro) to pin one exactly." />
            </label>
            <select
              className="input"
              value={rec.model_alias}
              onChange={(e) => set("model_alias", e.target.value)}
            >
              <optgroup label="Aliases (recommended — provider fallback included)">
                {aliasList.map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </optgroup>
              {(models?.models ?? []).length > 0 && (
                <optgroup label="Concrete model IDs (pins one provider)">
                  {(models?.models ?? []).map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </optgroup>
              )}
            </select>
            {models?.alias_details?.[rec.model_alias] && (
              <div className="mt-1 text-xs text-slate-500">
                {rec.model_alias} → {models.alias_details[rec.model_alias].join(" → ")}
              </div>
            )}
          </div>
          <div>
            <label className="label">
              Fallback model
              <InfoTip text="Optional. If the primary model's call fails, the agent retries once against this model. Pick an alias or a concrete model id. Leave blank for no per-agent fallback (alias routing still applies its own proxy-level fallbacks)." />
            </label>
            <select
              className="input"
              value={rec.fallback_alias ?? ""}
              onChange={(e) => set("fallback_alias", e.target.value || null)}
            >
              <option value="">(none)</option>
              <optgroup label="Aliases">
                {aliasList.map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </optgroup>
              {(models?.models ?? []).length > 0 && (
                <optgroup label="Concrete model IDs">
                  {(models?.models ?? []).map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </optgroup>
              )}
            </select>
            {rec.fallback_alias && models?.alias_details?.[rec.fallback_alias] && (
              <div className="mt-1 text-xs text-slate-500">
                {rec.fallback_alias} → {models.alias_details[rec.fallback_alias].join(" → ")}
              </div>
            )}
          </div>
          <div>
            <label className="label">
              Temperature
              <InfoTip text="Controls randomness of the model's output (0–2). Lower (0–0.3) is focused and deterministic — best for research and analysis; higher (0.7–1.0) is more creative and varied. Hyperion defaults to 0.1–0.2." />
            </label>
            <input
              className="input"
              type="number"
              step="0.1"
              value={rec.temperature}
              onChange={(e) => set("temperature", Number(e.target.value))}
            />
          </div>
          <div>
            <label className="label">
              Max iter
              <InfoTip text="Maximum reasoning/tool-use loops this agent may take in a single run (one LLM call per loop). Higher allows more thorough work but costs more tokens and time." />
            </label>
            <input
              className="input"
              type="number"
              value={rec.max_iter}
              onChange={(e) => set("max_iter", Number(e.target.value))}
            />
          </div>
        </div>

        {/* Tools */}
        <div className="card">
          <label className="label">
            Tools
            <InfoTip text="External capabilities this agent may call during a run — e.g. web search, second-brain retrieval, code execution. Hover each to see what it does." />
          </label>
          <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
            {(tools ?? []).map((t) => (
              <label key={t.name} className="flex items-center gap-2 text-sm" title={t.description}>
                <input
                  type="checkbox"
                  checked={rec.tools.includes(t.name)}
                  onChange={() => toggleTool(t.name)}
                />
                {t.name}
              </label>
            ))}
          </div>
        </div>

        {/* Schedule */}
        <div className="card">
          <label className="label">
            Run on schedule (cron)
            <InfoTip text="Optional. A 5-field cron expression (e.g. '*/5 * * * *') that fires this agent as a standalone task on a timer, independent of any workflow. Leave blank for no schedule — the agent then only runs when a workflow uses it." />
          </label>
          <input
            className="input"
            placeholder="(none — runs only via workflows)"
            value={rec.schedule_cron ?? ""}
            onChange={(e) => set("schedule_cron", e.target.value || null)}
          />
        </div>

        {/* Thresholds */}
        <div className="card grid grid-cols-3 gap-3">
          <div>
            <label className="label">
              Max input tokens
              <InfoTip text="Per-run cap on tokens sent to the model by this agent. The run aborts (circuit breaker) if exceeded. Leave blank to inherit the global cap / no limit." />
            </label>
            <input
              className="input"
              type="number"
              value={rec.thresholds.max_input_tokens ?? ""}
              onChange={(e) => setThreshold("max_input_tokens", e.target.value)}
            />
          </div>
          <div>
            <label className="label">
              Max output tokens
              <InfoTip text="Per-run cap on tokens this agent may generate. The run aborts if exceeded. Leave blank to inherit the global cap / no limit." />
            </label>
            <input
              className="input"
              type="number"
              value={rec.thresholds.max_output_tokens ?? ""}
              onChange={(e) => setThreshold("max_output_tokens", e.target.value)}
            />
          </div>
          <div>
            <label className="label">
              Max activations/day
              <InfoTip text="Cap on how many times this agent may run in a 24-hour window — useful for scheduled/triggered agents. Leave blank for unlimited." />
            </label>
            <input
              className="input"
              type="number"
              value={rec.thresholds.max_activations_per_day ?? ""}
              onChange={(e) => setThreshold("max_activations_per_day", e.target.value)}
            />
          </div>
        </div>

        {save.isError && (
          <div className="card border-rose-500/40 bg-rose-500/10 text-sm text-rose-200">
            {(save.error as Error).message}
          </div>
        )}

        <div className="flex items-center gap-2">
          <button className="btn btn-primary" onClick={onSave} disabled={save.isPending}>
            {save.isPending ? "Saving…" : "Save"}
          </button>
          <button className="btn" onClick={() => nav("/")}>
            Cancel
          </button>
          {!isNew && (
            <>
              <button className="btn" onClick={onDuplicate} disabled={dup.isPending}>
                Duplicate
              </button>
              <button className="btn btn-danger ml-auto" onClick={onDelete} disabled={del.isPending}>
                Delete
              </button>
            </>
          )}
        </div>
        {del.isError && <p className="text-sm text-rose-300">{(del.error as Error).message}</p>}
      </div>
    </div>
  );
}
