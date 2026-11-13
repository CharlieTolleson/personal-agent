/**
 * Settings.tsx — Hyperion UI "Settings" page.
 *
 * Role in the system:
 *   Rendered as a route in the Hyperion web console (Vite/React app on :4102, see
 *   `agents/hyperion-ui`). This page is the operator-facing control panel for global
 *   Hyperion configuration, talking to the Hyperion FastAPI backend (:4100) via the
 *   typed hooks in `../api/client`. All data flows through TanStack Query, so reads are
 *   cached and writes invalidate the relevant query keys to keep the UI in sync.
 *
 * What it lets the operator do:
 *   1. Roles         — the editable list of logical model slots (planner / worker /
 *      cheap built-ins + any custom roles). Each role maps to an alias or concrete
 *      model. Saved as a batch via `useUpdateRoles`.
 *   2. Aliases       — define/edit/delete model aliases and their ordered provider
 *      fallback chains. New aliases are written through to LiteLLM so they actually
 *      route (see `tools/litellm_admin`); a per-alias status badge shows that state.
 *      Built-in aliases (smart/worker/cheap/fast) come from litellm_config.yaml and
 *      can be re-pointed but not deleted.
 *   3. Default workflow — choose which workflow DAG runs when a request doesn't pick
 *      one. The actual DAGs are authored on the separate "Workflows" tab.
 *   4. Caps          — read-only display of token/cost caps.
 *   5. Agent store   — export/import the full agent config as a zip.
 *   6. Providers     — read-only badges showing whether each provider has an API key.
 *
 * Key design decisions / non-obvious context:
 *   - Model pickers use a grouped `<select>` (aliases vs concrete ids), NOT
 *     `<input list=datalist>`: the native datalist filters options to the current
 *     value, which makes it look like only one model exists (same fix as AgentEditor).
 *   - Roles edit a local `roleDraft` array and commit the whole list at once; aliases
 *     edit a local `aliasRows` array and commit one alias at a time (the backend upserts
 *     per name and reconciles only that alias's proxy deployments).
 *
 * @module pages/Settings
 */
import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  downloadConfigExport,
  importConfig,
  useAliases,
  useConfig,
  useDeleteAlias,
  useModels,
  useSaveAlias,
  useUpdateConfig,
  useUpdateRoles,
  useWorkflows,
  type AliasRoutingStatus,
  type Role,
} from "../api/client";
import { useToast } from "../components/Toast";

/** Built-in role names that cannot be removed (they feed hard-coded LLM factories). */
const BUILTIN_ROLE_NAMES = ["planner", "worker", "cheap"];

/** Tailwind classes for each alias routing-status badge. */
const STATUS_STYLE: Record<AliasRoutingStatus, string> = {
  applied: "border-emerald-500/40 text-emerald-300",
  builtin: "border-sky-500/40 text-sky-300",
  partial: "border-amber-500/40 text-amber-300",
  pending: "border-edge text-slate-400",
  unknown: "border-edge text-slate-500",
  deleted: "border-edge text-slate-500",
  error: "border-rose-500/40 text-rose-300",
};

/**
 * A grouped model `<select>`: aliases first (optionally), then concrete model ids.
 * Used for role targets (aliases + concrete) and alias chain entries (concrete only).
 */
function ModelSelect({
  value,
  onChange,
  aliases,
  models,
  includeAliases,
}: {
  value: string;
  onChange: (v: string) => void;
  aliases: string[];
  models: string[];
  includeAliases: boolean;
}) {
  return (
    <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
      {/* An empty placeholder so a not-yet-chosen value doesn't silently bind to the first option. */}
      {!value && <option value="" />}
      {includeAliases && aliases.length > 0 && (
        <optgroup label="Aliases (provider fallback included)">
          {aliases.map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </optgroup>
      )}
      {models.length > 0 && (
        <optgroup label="Concrete model IDs (pins one provider)">
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </optgroup>
      )}
    </select>
  );
}

/** Local editable shape for one alias row. `isNew` rows aren't persisted yet. */
interface AliasRow {
  name: string;
  models: string[];
  isBuiltin: boolean;
  isNew: boolean;
}

/**
 * Settings page component (default route export). Sources everything from the API hooks
 * and holds local drafts for the roles list and alias rows until the operator saves.
 */
export default function Settings() {
  const { data: config } = useConfig();
  const { data: models } = useModels();
  const { data: aliasInfo } = useAliases();
  const { data: workflows } = useWorkflows();
  const update = useUpdateConfig();
  const updateRoles = useUpdateRoles();
  const saveAlias = useSaveAlias();
  const deleteAlias = useDeleteAlias();
  const toast = useToast();
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [importing, setImporting] = useState(false);

  const aliasNames = models?.aliases ?? [];
  const concreteModels = models?.models ?? [];

  // ----- Roles draft (whole-list edit, committed by "Save roles") -----------
  const [roleDraft, setRoleDraft] = useState<Role[]>([]);
  useEffect(() => {
    if (models?.roles) setRoleDraft(models.roles.map((r) => ({ ...r })));
  }, [models?.roles]);

  function setRole(idx: number, patch: Partial<Role>) {
    setRoleDraft((rs) => rs.map((r, i) => (i === idx ? { ...r, ...patch } : r)));
  }
  function addRole() {
    setRoleDraft((rs) => [...rs, { name: "", note: "", model: "" }]);
  }
  function removeRole(idx: number) {
    setRoleDraft((rs) => rs.filter((_, i) => i !== idx));
  }
  function saveRoles() {
    updateRoles.mutate(roleDraft, {
      onSuccess: () => toast.push("Roles saved", "success"),
      onError: (e) => toast.push((e as Error).message, "error"),
    });
  }

  // ----- Alias rows draft (per-alias save / delete) -------------------------
  const [aliasRows, setAliasRows] = useState<AliasRow[]>([]);
  useEffect(() => {
    if (!aliasInfo) return;
    const builtins = new Set(aliasInfo.builtins);
    setAliasRows(
      Object.entries(aliasInfo.aliases).map(([name, ms]) => ({
        name,
        models: [...ms],
        isBuiltin: builtins.has(name),
        isNew: false,
      })),
    );
  }, [aliasInfo]);

  function setAliasRow(idx: number, patch: Partial<AliasRow>) {
    setAliasRows((rows) => rows.map((r, i) => (i === idx ? { ...r, ...patch } : r)));
  }
  function setAliasModel(rowIdx: number, modelIdx: number, value: string) {
    setAliasRows((rows) =>
      rows.map((r, i) =>
        i === rowIdx ? { ...r, models: r.models.map((m, j) => (j === modelIdx ? value : m)) } : r,
      ),
    );
  }
  function moveAliasModel(rowIdx: number, modelIdx: number, dir: -1 | 1) {
    setAliasRows((rows) =>
      rows.map((r, i) => {
        if (i !== rowIdx) return r;
        const j = modelIdx + dir;
        if (j < 0 || j >= r.models.length) return r;
        const ms = [...r.models];
        [ms[modelIdx], ms[j]] = [ms[j], ms[modelIdx]];
        return { ...r, models: ms };
      }),
    );
  }
  function addAliasModel(rowIdx: number) {
    setAliasRows((rows) =>
      rows.map((r, i) => (i === rowIdx ? { ...r, models: [...r.models, ""] } : r)),
    );
  }
  function removeAliasModel(rowIdx: number, modelIdx: number) {
    setAliasRows((rows) =>
      rows.map((r, i) =>
        i === rowIdx ? { ...r, models: r.models.filter((_, j) => j !== modelIdx) } : r,
      ),
    );
  }
  function addAlias() {
    setAliasRows((rows) => [...rows, { name: "", models: [""], isBuiltin: false, isNew: true }]);
  }
  function saveAliasRow(row: AliasRow) {
    const cleaned = row.models.filter((m) => m);
    if (!row.name || cleaned.length === 0) {
      toast.push("Alias needs a name and at least one model", "error");
      return;
    }
    saveAlias.mutate(
      { name: row.name, models: cleaned },
      {
        onSuccess: (res) => toast.push(`Alias ${res.name} saved (${res.status.status})`, "success"),
        onError: (e) => toast.push((e as Error).message, "error"),
      },
    );
  }
  function deleteAliasRow(row: AliasRow, idx: number) {
    if (row.isNew) {
      setAliasRows((rows) => rows.filter((_, i) => i !== idx));
      return;
    }
    deleteAlias.mutate(row.name, {
      onSuccess: () => toast.push(`Alias ${row.name} deleted`, "success"),
      onError: (e) => toast.push((e as Error).message, "error"),
    });
  }

  /**
   * Upload an agent-config zip to the backend and reconcile the UI on success.
   * @param file - The `.zip` config archive chosen by the operator.
   */
  async function onImportFile(file: File) {
    setImporting(true);
    try {
      const res = await importConfig(file);
      const wf = res.workflows?.length ? ` + ${res.workflows.length} workflow(s)` : "";
      toast.push(`Imported ${res.count} agent(s)${wf}: ${res.imported.join(", ")}`, "success");
      qc.invalidateQueries({ queryKey: ["agents"] });
      qc.invalidateQueries({ queryKey: ["groups"] });
      qc.invalidateQueries({ queryKey: ["workflows"] });
    } catch (e) {
      toast.push((e as Error).message, "error");
    } finally {
      setImporting(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <h2 className="mb-4 text-lg font-semibold">Settings</h2>

      {/* ── Roles ──────────────────────────────────────────────────────────── */}
      <div className="card mb-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Roles</h3>
          <button className="btn btn-sm" onClick={addRole}>+ Add role</button>
        </div>
        <p className="text-xs text-slate-500">
          Logical model slots the orchestrator selects by intent. Each maps to an alias or a
          concrete model. The built-in roles (planner / worker / cheap) can be re-pointed but
          not removed.
        </p>
        {roleDraft.map((r, idx) => {
          const isBuiltin = BUILTIN_ROLE_NAMES.includes(r.name);
          const chain = models?.alias_details?.[r.model];
          return (
            <div key={idx} className="grid grid-cols-[1fr_1fr_auto] items-start gap-2">
              <div>
                <input
                  className="input"
                  placeholder="role name"
                  value={r.name}
                  disabled={isBuiltin}
                  onChange={(e) => setRole(idx, { name: e.target.value })}
                />
                <input
                  className="input mt-1 text-xs"
                  placeholder="note (what it's used for)"
                  value={r.note}
                  onChange={(e) => setRole(idx, { note: e.target.value })}
                />
              </div>
              <div>
                <ModelSelect
                  value={r.model}
                  onChange={(v) => setRole(idx, { model: v })}
                  aliases={aliasNames}
                  models={concreteModels}
                  includeAliases
                />
                {chain && (
                  <div className="mt-1 text-xs text-slate-500">{r.model} → {chain.join(" → ")}</div>
                )}
              </div>
              <button
                className="btn btn-sm"
                disabled={isBuiltin}
                title={isBuiltin ? "Built-in role" : "Remove role"}
                onClick={() => removeRole(idx)}
              >
                ✕
              </button>
            </div>
          );
        })}
        <div className="flex items-center gap-3">
          <button className="btn btn-primary" onClick={saveRoles} disabled={updateRoles.isPending}>
            {updateRoles.isPending ? "Saving…" : "Save roles"}
          </button>
        </div>
      </div>

      {/* ── Aliases ────────────────────────────────────────────────────────── */}
      <div className="card mb-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Aliases</h3>
          <button className="btn btn-sm" onClick={addAlias}>+ New alias</button>
        </div>
        <p className="text-xs text-slate-500">
          A model alias is a multi-provider group with an ordered fallback chain. New aliases
          are registered with the LiteLLM proxy so they route across providers; built-in aliases
          (smart / worker / cheap / fast) are defined in litellm_config.yaml and can't be deleted.
        </p>
        {aliasRows.map((row, idx) => {
          const status = aliasInfo?.status?.[row.name]?.status as AliasRoutingStatus | undefined;
          return (
            <div key={idx} className="rounded border border-edge p-3 space-y-2">
              <div className="flex items-center gap-2">
                <input
                  className="input max-w-[12rem]"
                  placeholder="alias name"
                  value={row.name}
                  disabled={row.isBuiltin || !row.isNew}
                  onChange={(e) => setAliasRow(idx, { name: e.target.value })}
                />
                {status && (
                  <span className={`pill ${STATUS_STYLE[status]}`}>{status}</span>
                )}
                <div className="ml-auto flex gap-2">
                  <button
                    className="btn btn-sm"
                    onClick={() => saveAliasRow(row)}
                    disabled={saveAlias.isPending}
                  >
                    Save
                  </button>
                  <button
                    className="btn btn-sm"
                    disabled={row.isBuiltin || deleteAlias.isPending}
                    title={row.isBuiltin ? "Built-in alias" : "Delete alias"}
                    onClick={() => deleteAliasRow(row, idx)}
                  >
                    Delete
                  </button>
                </div>
              </div>
              {/* Ordered fallback chain — concrete model ids, reorderable. */}
              <div className="space-y-1">
                {row.models.map((m, j) => (
                  <div key={j} className="flex items-center gap-1">
                    <span className="w-4 text-right text-xs text-slate-500">{j + 1}.</span>
                    <ModelSelect
                      value={m}
                      onChange={(v) => setAliasModel(idx, j, v)}
                      aliases={aliasNames}
                      models={concreteModels}
                      includeAliases={false}
                    />
                    <button className="btn btn-sm" title="Move up" disabled={j === 0}
                      onClick={() => moveAliasModel(idx, j, -1)}>↑</button>
                    <button className="btn btn-sm" title="Move down" disabled={j === row.models.length - 1}
                      onClick={() => moveAliasModel(idx, j, 1)}>↓</button>
                    <button className="btn btn-sm" title="Remove" disabled={row.models.length === 1}
                      onClick={() => removeAliasModel(idx, j)}>✕</button>
                  </div>
                ))}
                <button className="btn btn-sm" onClick={() => addAliasModel(idx)}>+ Add model</button>
              </div>
              {aliasInfo?.status?.[row.name]?.detail && (
                <div className="text-xs text-amber-300">{aliasInfo.status[row.name].detail}</div>
              )}
            </div>
          );
        })}
      </div>

      {config && (
        <>
          <div className="card mb-4">
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Default workflow
            </h3>
            <p className="mb-3 text-xs text-slate-500">
              The workflow (DAG) used when a run doesn't pick one. Manage workflows on the
              Workflows tab.
            </p>
            <select
              className="input max-w-sm"
              value={config.default_workflow}
              disabled={update.isPending}
              onChange={(e) =>
                update.mutate(
                  { default_workflow: e.target.value },
                  {
                    onSuccess: () => toast.push(`Default workflow → ${e.target.value}`, "success"),
                    onError: (err) => toast.push((err as Error).message, "error"),
                  },
                )
              }
            >
              {(workflows ?? []).map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name || w.id}
                </option>
              ))}
            </select>
          </div>

          <div className="card mb-4">
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Caps
            </h3>
            <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
              {Object.entries(config.caps).map(([k, v]) => (
                <div key={k}>
                  <div className="text-xs text-slate-500">{k}</div>
                  <div className="font-mono">{v.toLocaleString()}</div>
                </div>
              ))}
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Caps are set via env / per-request; per-agent overrides live in the agent editor.
            </p>
          </div>

          <div className="card mb-4">
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Agent store
            </h3>
            <p className="mb-3 text-xs text-slate-500">
              Export every agent record as a zip to back up or move between machines. Import
              validates the whole set (DAG + at-least-one plan/synthesize) before writing.
            </p>
            <div className="flex items-center gap-3">
              <button
                className="btn"
                onClick={() =>
                  downloadConfigExport().catch((e: Error) =>
                    toast.push(`Export failed: ${e.message}`, "error"),
                  )
                }
              >
                Export config
              </button>
              <button
                className="btn"
                disabled={importing}
                onClick={() => fileRef.current?.click()}
              >
                {importing ? "Importing…" : "Import config"}
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".zip"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onImportFile(f);
                }}
              />
            </div>
          </div>

          <div className="card">
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Providers
            </h3>
            <div className="flex flex-wrap gap-2">
              {Object.entries(config.providers).map(([name, p]) => (
                <span
                  key={name}
                  className={`pill ${p.key_present ? "border-emerald-500/40 text-emerald-300" : "border-edge text-slate-500"}`}
                >
                  {name}: {p.key_present ? "ok" : "no key"}
                </span>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
