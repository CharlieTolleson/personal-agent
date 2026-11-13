/**
 * Hyperion UI — API client and TanStack Query hooks.
 *
 * This module is the single typed boundary between the Hyperion React/Vite web
 * console (port 4102) and the Hyperion FastAPI backend (default port 4100). It
 * defines:
 *   1. TypeScript interfaces that mirror the backend record shapes
 *      (see the Python side: `hyperion/server/api.py` and `registry.py`).
 *   2. A small `fetch` wrapper (`req`) that handles JSON encoding, error
 *      extraction, and the 204 No-Content case.
 *   3. A collection of TanStack Query hooks (`useQuery`/`useMutation`) — one per
 *      backend endpoint — that components consume directly. Mutations invalidate
 *      the relevant query keys on success so the UI re-fetches fresh data.
 *
 * Design notes:
 * - The backend base URL is resolved once at module load from the
 *   `VITE_HYPERION_API` build-time env var, falling back to localhost:4100.
 * - All network access goes through `req` so error handling stays uniform; the
 *   one exception is the config import/export pair, which deals with binary
 *   multipart data (zip) and therefore bypasses the JSON helper.
 * - Query keys follow a `[resource, ...params]` convention; mutation
 *   `onSuccess` handlers invalidate by the matching key prefix to trigger
 *   refetches. Some list queries poll via `refetchInterval` for live updates.
 *
 * This file contains NO React components — only hooks and plain helpers. Hooks
 * must be called from within React components / other hooks per the rules of
 * hooks.
 */
import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from "@tanstack/react-query";
import { authHeaders, noteAuthFailure } from "./auth";

/**
 * Base URL for every Hyperion API request.
 *
 * Resolved once at module load: prefer the `VITE_HYPERION_API` build-time env
 * var (trailing slash stripped so paths can be concatenated cleanly), otherwise
 * fall back to the local dev backend at http://localhost:4100.
 */
export const API_BASE =
  (import.meta.env.VITE_HYPERION_API as string | undefined)?.replace(/\/$/, "") ||
  "http://localhost:4100";

// ---------------------------------------------------------------------------
// Types — mirror the Hyperion backend records (registry.py / api.py).
// ---------------------------------------------------------------------------

/**
 * The role a node plays within a workflow: planning, working, final synthesis, or
 * running a whole other workflow as one step (`subworkflow`). A `subworkflow` node
 * sets `workflow` (a child workflow id) instead of `agent`.
 */
export type NodeKind = "plan" | "work" | "synthesize" | "subworkflow";

/** Per-agent usage caps; `null` means "inherit the global cap / no override". */
export interface Thresholds {
  max_input_tokens: number | null;
  max_output_tokens: number | null;
  max_activations_per_day: number | null;
}

/**
 * Full editable configuration of a single Hyperion agent, as stored in the
 * backend registry. An agent is a pure persona — prompt + model + tools; *when*
 * and *in what order* it runs is decided by the workflow that references it. The
 * lone scheduling hook is `schedule_cron`. Mirrors `AgentRecord` in `registry.py`.
 */
export interface AgentRecord {
  id: string;
  name: string;
  description: string;
  group: string;
  active: boolean;
  role: string;
  goal: string;
  backstory: string;
  model_alias: string;
  fallback_alias: string | null;
  temperature: number;
  top_p: number | null;
  max_tokens: number | null;
  max_iter: number;
  tools: string[];
  schedule_cron: string | null;
  thresholds: Thresholds;
}

/** A tool that agents can be granted, as surfaced by the backend tool registry. */
export interface ToolInfo {
  name: string;
  description: string;
}

/**
 * Available model aliases/IDs plus the currently selected models for each of the
 * three role slots (planner / worker / cheap) the backend resolves at runtime.
 * `alias_details` maps each alias name to its ordered fallback chain of concrete
 * models (e.g. `smart → ["claude-opus-4-6 (anthropic)", ...]`), so the UI can
 * show what's behind an alias without the operator opening litellm_config.yaml.
 */
export interface ModelsInfo {
  aliases: string[];
  models: string[];
  current: { planner: string; worker: string; cheap: string };
  alias_details: Record<string, string[]>;
  /** Operator-editable role list (planner/worker/cheap built-ins + any custom roles). */
  roles: Role[];
  /** Raw `alias name -> ordered concrete model ids` map (the editable form of `alias_details`). */
  aliases_detail: Record<string, string[]>;
}

/**
 * A logical model slot the orchestrator selects by intent. The three built-in roles
 * (planner/worker/cheap) feed the hard-coded LLM factory functions; operators may add
 * more. `model` is an alias name or a concrete model id.
 */
export interface Role {
  name: string;
  note: string;
  model: string;
}

/** Live routing state of an alias against the LiteLLM proxy (see /aliases). */
export type AliasRoutingStatus =
  | "builtin"
  | "applied"
  | "partial"
  | "pending"
  | "unknown"
  | "deleted"
  | "error";

/**
 * Response of GET /aliases: each alias's ordered model chain, which names are built-in
 * (defined in litellm_config.yaml and not deletable), and per-alias routing status.
 */
export interface AliasesResponse {
  aliases: Record<string, string[]>;
  builtins: string[];
  status: Record<string, { status: AliasRoutingStatus; detail?: string }>;
}

/** A single selectable option in a "choice"-type affordance. */
export interface AffordanceOption {
  id: string;
  label: string;
  description: string;
}

/** A single input field in a "form"-type affordance. */
export interface AffordanceField {
  id: string;
  label: string;
  type: "text" | "number" | "boolean" | "select";
  options: string[];
  required: boolean;
}

/**
 * A human-in-the-loop interaction the backend requests mid-run: a choice, form,
 * free-text question, or confirmation. Rendered by the UI when a task enters an
 * `awaiting_input` / `awaiting_approval` state; the user's response is sent back
 * via `useApproveTask`. `options` is used for "choice", `fields` for "form".
 */
export interface Affordance {
  type: "choice" | "form" | "question" | "confirm";
  prompt: string;
  options: AffordanceOption[];
  fields: AffordanceField[];
  agent_id: string | null;
  stage: string | null;
}

/**
 * Outcome of the backend's routing pass for a task: which agents were selected,
 * which were skipped (with reasons), and the resulting execution DAG mapping
 * each agent id to its list of upstream dependency ids.
 */
export interface RoutingResult {
  workflow: string | null;
  selected_agents: string[];
  skipped: { id: string; reason: string }[];
  dag: Record<string, string[]>;
}

/** Lifecycle state of a submitted task. */
export type TaskStatus =
  | "queued"
  | "running"
  | "awaiting_approval"
  | "awaiting_input"
  | "done"
  | "failed"
  | "cancelled";

/**
 * Live state of a single task as returned by submit / poll / approve endpoints.
 * `progress_lines` streams human-readable status; when the task is paused for
 * HITL, `pending_stage` and `pending_affordance` describe what input is needed.
 */
export interface TaskResponse {
  task_id: string;
  status: TaskStatus;
  error: string | null;
  result_path: string | null;
  progress_lines: string[];
  routing: RoutingResult | null;
  pending_stage: string | null;
  pending_affordance: Affordance | null;
}

/**
 * System-wide configuration snapshot: available models, provider key/health
 * status, usage caps, the default workflow id, and the LiteLLM proxy URL all
 * calls are routed through.
 * `alias_fallback_order` maps each alias to its ordered fallback chain so the
 * Settings page can show what's behind smart/worker/cheap/fast.
 */
export interface ConfigResponse {
  models: Record<string, { alias: string; note: string; env_var: string | null }>;
  roles: Role[];
  aliases: Record<string, string[]>;
  providers: Record<string, { key_present: boolean; status: string }>;
  caps: Record<string, number>;
  default_workflow: string;
  litellm_url: string;
  alias_fallback_order: Record<string, string[]>;
}

/** Optional conditional-firing rule for a node: run only for these task types. */
export interface NodeWhen {
  task_types: string[];
}

/**
 * Canvas coordinates for the graphical workflow builder (UI-only metadata).
 * Persisted so a hand-arranged layout survives reloads; ignored by the runner,
 * which orders nodes by the upstream DAG. Optional — nodes without a position are
 * auto-laid-out by the editor on first open.
 */
export interface NodePosition {
  x: number;
  y: number;
}

/**
 * One node in a workflow DAG. Usually an agent instance playing a `kind` role
 * (plan / work / synthesize); a node with `kind === "subworkflow"` instead runs
 * a whole child workflow (named by `workflow`) as one composable step and leaves
 * `agent` null. Exactly one of `agent` / `workflow` is set. Each node carries
 * explicit upstream dependencies, an optional HITL gate before it runs
 * (`gate_before`), an optional per-node instruction override (for a subworkflow
 * node this becomes the child run's request), an optional `when` conditional-firing
 * rule, and an optional canvas `position` for the graphical editor.
 */
export interface WorkflowNode {
  id: string;
  agent: string | null;
  workflow?: string | null;
  kind: NodeKind;
  upstream: string[];
  gate_before: boolean;
  instruction: string | null;
  when: NodeWhen | null;
  position?: NodePosition | null;
}

/** A named, reusable agent DAG selectable when submitting a task. */
export interface WorkflowRecord {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
}

// ---------------------------------------------------------------------------
// Fetch helper
// ---------------------------------------------------------------------------

/**
 * Core fetch wrapper for JSON endpoints. Prepends {@link API_BASE}, sets the
 * JSON content-type and bearer-token headers, and normalizes error handling.
 * A 401 clears the stored token and reopens the token form (`api/auth.ts`).
 *
 * On a non-2xx response it throws an `Error` whose message is `"<status>:
 * <detail>"`, preferring the backend's `detail` field (FastAPI convention) and
 * falling back to the raw body or HTTP status text. A 204 No-Content response
 * resolves to `undefined` (cast to `T`).
 *
 * @typeParam T - Expected shape of the parsed JSON response body.
 * @param path - API path beginning with "/" (appended to {@link API_BASE}).
 * @param init - Optional `fetch` options (method, body, etc.); spread over the
 *   defaults so callers can override or add headers.
 * @returns The parsed response body typed as `T`.
 * @throws Error if the response status is not ok (2xx).
 */
async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(API_BASE + path, {
    ...init,
    // Merged after `init` so a caller's headers can't drop the bearer token.
    headers: { "content-type": "application/json", ...init?.headers, ...authHeaders() },
  });
  noteAuthFailure(resp);
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* keep statusText */
    }
    throw new Error(`${resp.status}: ${detail}`);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------

/**
 * Query hook: fetch the full list of configured agents.
 * @returns TanStack Query result with `data: AgentRecord[]`.
 */
export function useAgents() {
  return useQuery({ queryKey: ["agents"], queryFn: () => req<AgentRecord[]>("/agents") });
}

/**
 * Query hook: fetch a single agent by id. Disabled until `id` is defined.
 * @param id - Agent id, or undefined to keep the query idle.
 * @returns TanStack Query result with `data: AgentRecord`.
 */
export function useAgent(id: string | undefined) {
  return useQuery({
    queryKey: ["agent", id],
    queryFn: () => req<AgentRecord>(`/agents/${id}`),
    enabled: !!id,
  });
}

/**
 * Mutation hook: create or update an agent. Invalidates the agents list on
 * success so the table refreshes.
 * @param isNew - When true, POSTs to /agents (create); otherwise PUTs to
 *   /agents/{id} (update). Determines both the URL and HTTP method.
 * @returns Mutation whose `mutate(rec: AgentRecord)` persists the record.
 */
export function useSaveAgent(isNew: boolean) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (rec: AgentRecord) =>
      req<AgentRecord>(isNew ? "/agents" : `/agents/${rec.id}`, {
        method: isNew ? "POST" : "PUT",
        body: JSON.stringify(rec),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agents"] }),
  });
}

/**
 * Mutation hook: delete an agent by id, invalidating the agents list.
 * @returns Mutation whose `mutate(id: string)` deletes the agent.
 */
export function useDeleteAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => req(`/agents/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agents"] }),
  });
}

/**
 * Mutation hook: duplicate an existing agent server-side, invalidating the list.
 * @returns Mutation whose `mutate(id: string)` returns the new AgentRecord.
 */
export function useDuplicateAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      req<AgentRecord>(`/agents/${id}/duplicate`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agents"] }),
  });
}

/**
 * Query hook: fetch the list of agent group names (used for grouping/filtering).
 * @returns TanStack Query result with `data: string[]`.
 */
export function useGroups() {
  return useQuery({ queryKey: ["groups"], queryFn: () => req<string[]>("/groups") });
}

/**
 * Query hook: fetch the catalog of tools agents can be granted.
 * @returns TanStack Query result with `data: ToolInfo[]`.
 */
export function useTools() {
  return useQuery({ queryKey: ["tools"], queryFn: () => req<ToolInfo[]>("/tools") });
}

/**
 * Query hook: fetch available models and the current per-role selections.
 * @returns TanStack Query result with `data: ModelsInfo`.
 */
export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: () => req<ModelsInfo>("/models") });
}

/**
 * Query hook: fetch the system configuration snapshot.
 * @returns TanStack Query result with `data: ConfigResponse`.
 */
export function useConfig() {
  return useQuery({ queryKey: ["config"], queryFn: () => req<ConfigResponse>("/config") });
}

/**
 * Mutation hook: update one or more global config settings (per-role model
 * selections and/or the default workflow). On success it invalidates both the
 * `config` and `models` queries, since changing a model selection affects both.
 * @returns Mutation whose `mutate(body)` accepts a partial set of the keys
 *   `model_planner` | `model_worker` | `model_cheap` | `default_workflow`.
 */
export function useUpdateConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (
      body: Partial<
        Record<"model_planner" | "model_worker" | "model_cheap" | "default_workflow", string>
      >,
    ) => req<ConfigResponse>("/config", { method: "PUT", body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["config"] });
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

/** Invalidate every query whose data depends on the role/alias registry. */
function invalidateRegistry(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: ["config"] });
  qc.invalidateQueries({ queryKey: ["models"] });
  qc.invalidateQueries({ queryKey: ["roles"] });
  qc.invalidateQueries({ queryKey: ["aliases"] });
}

/**
 * Mutation hook: replace the full roles list (add / rename / remove / re-point).
 * Invalidates the registry-dependent queries on success.
 * @returns Mutation whose `mutate(roles: Role[])` persists the new list.
 */
export function useUpdateRoles() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (roles: Role[]) =>
      req<{ roles: Role[] }>("/roles", { method: "PUT", body: JSON.stringify({ roles }) }),
    onSuccess: () => invalidateRegistry(qc),
  });
}

/**
 * Query hook: fetch alias chains plus each alias's live proxy routing status.
 * @returns TanStack Query result with `data: AliasesResponse`.
 */
export function useAliases() {
  return useQuery({ queryKey: ["aliases"], queryFn: () => req<AliasesResponse>("/aliases") });
}

/**
 * Mutation hook: create or replace one alias's ordered model chain, writing through
 * to LiteLLM. Invalidates the registry-dependent queries on success.
 * @returns Mutation whose `mutate({name, models})` upserts the alias.
 */
export function useSaveAlias() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, models }: { name: string; models: string[] }) =>
      req<{ name: string; models: string[]; status: { status: AliasRoutingStatus; detail?: string } }>(
        `/aliases/${name}`,
        { method: "PUT", body: JSON.stringify({ models }) },
      ),
    onSuccess: () => invalidateRegistry(qc),
  });
}

/**
 * Mutation hook: delete a user-defined alias (built-ins / referenced aliases are
 * refused by the backend). Invalidates the registry-dependent queries on success.
 * @returns Mutation whose `mutate(name: string)` deletes the alias.
 */
export function useDeleteAlias() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => req(`/aliases/${name}`, { method: "DELETE" }),
    onSuccess: () => invalidateRegistry(qc),
  });
}

// ---------------------------------------------------------------------------
// Workflows — named DAGs of agent nodes (run picker + editor)
// ---------------------------------------------------------------------------

/**
 * Query hook: fetch all saved workflow DAGs (for the run picker and editor).
 * @returns TanStack Query result with `data: WorkflowRecord[]`.
 */
export function useWorkflows() {
  return useQuery({
    queryKey: ["workflows"],
    queryFn: () => req<WorkflowRecord[]>("/workflows"),
  });
}

/**
 * Query hook: fetch a single workflow by id. Disabled until `id` is defined.
 * @param id - Workflow id, or undefined to keep the query idle.
 * @returns TanStack Query result with `data: WorkflowRecord`.
 */
export function useWorkflow(id: string | undefined) {
  return useQuery({
    queryKey: ["workflow", id],
    queryFn: () => req<WorkflowRecord>(`/workflows/${id}`),
    enabled: !!id,
  });
}

/**
 * Mutation hook: create or update a workflow, invalidating the workflows list.
 * @param isNew - When true, POSTs to /workflows (create); otherwise PUTs to
 *   /workflows/{id} (update). Determines both the URL and HTTP method.
 * @returns Mutation whose `mutate(rec: WorkflowRecord)` persists the record.
 */
export function useSaveWorkflow(isNew: boolean) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (rec: WorkflowRecord) =>
      req<WorkflowRecord>(isNew ? "/workflows" : `/workflows/${rec.id}`, {
        method: isNew ? "POST" : "PUT",
        body: JSON.stringify(rec),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["workflows"] }),
  });
}

/**
 * Mutation hook: delete a workflow by id, invalidating the workflows list.
 * @returns Mutation whose `mutate(id: string)` deletes the workflow.
 */
export function useDeleteWorkflow() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => req(`/workflows/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["workflows"] }),
  });
}

/**
 * Mutation hook: duplicate a workflow server-side, invalidating the list.
 * @returns Mutation whose `mutate(id: string)` returns the new WorkflowRecord.
 */
export function useDuplicateWorkflow() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      req<WorkflowRecord>(`/workflows/${id}/duplicate`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["workflows"] }),
  });
}

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

/**
 * Body for submitting a new task. `hitl` selects the human-in-the-loop level
 * (off / approve-plan-only / full gating); `workflow` optionally pins a specific
 * workflow DAG (null/omitted lets the backend auto-route).
 */
export interface SubmitTaskBody {
  task: string;
  hitl?: "off" | "plan" | "full";
  workflow?: string | null;
  // Plain-language description of how agents should collaborate; the server
  // compiles it into an ad-hoc workflow DAG. Ignored when `workflow` is set.
  workflow_prompt?: string;
}

/**
 * Mutation hook: submit a new task for execution.
 * Note: no query invalidation here — callers typically poll the returned
 * `task_id` via {@link useTask}.
 * @returns Mutation whose `mutate(body: SubmitTaskBody)` returns a TaskResponse.
 */
export function useSubmitTask() {
  return useMutation({
    mutationFn: (body: SubmitTaskBody) =>
      req<TaskResponse>("/tasks", { method: "POST", body: JSON.stringify(body) }),
  });
}

/**
 * Query hook: fetch a single task's live state. Disabled until `id` is defined.
 * @param id - Task id, or undefined to keep the query idle.
 * @param opts - Extra query options (spread last), e.g. a `refetchInterval` for
 *   polling a running task until it completes.
 * @returns TanStack Query result with `data: TaskResponse`.
 */
export function useTask(
  id: string | undefined,
  opts?: Partial<UseQueryOptions<TaskResponse>>,
) {
  return useQuery({
    queryKey: ["task", id],
    queryFn: () => req<TaskResponse>(`/tasks/${id}`),
    enabled: !!id,
    ...opts,
  });
}

/**
 * Body for responding to a task's pending HITL affordance. `action` is the
 * decision; `chosen_option` carries a selected choice option id, and `edits`
 * carries free-text revisions/form input, depending on the affordance type.
 */
export interface ApproveBody {
  action: "approve" | "revise" | "reject";
  chosen_option?: string;
  edits?: string;
}

/**
 * Mutation hook: respond to a task's pending HITL gate. Invalidates that task's
 * query on success so the UI picks up the resumed state.
 * @param id - The task id being approved/revised/rejected.
 * @returns Mutation whose `mutate(body: ApproveBody)` returns a TaskResponse.
 */
export function useApproveTask(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ApproveBody) =>
      req<TaskResponse>(`/tasks/${id}/approve`, { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["task", id] }),
  });
}

/**
 * Mutation hook: send free-text feedback for a task (e.g. a follow-up message),
 * invalidating that task's query on success.
 * @param id - The task id to attach feedback to.
 * @returns Mutation whose `mutate(message: string)` returns a TaskResponse.
 */
export function useFeedbackTask(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (message: string) =>
      req<TaskResponse>(`/tasks/${id}/feedback`, {
        method: "POST",
        body: JSON.stringify({ message }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["task", id] }),
  });
}

/**
 * Mutation hook: cancel a non-terminal run. The backend marks it `cancelled`
 * immediately; invalidates that task's query on success so the UI reflects the
 * stop without waiting for the next poll.
 * @param id - The task id to stop.
 * @returns Mutation whose `mutate()` returns the updated TaskResponse.
 */
export function useStopTask(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => req<TaskResponse>(`/tasks/${id}/stop`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["task", id] }),
  });
}

/** Result of exporting a task to Notion: the created page's URL and id. */
export interface NotionPageResult {
  url: string | null;
  id: string | null;
}

/**
 * Mutation hook: export a completed task's result to a new Notion page.
 * @param id - The task id whose result is exported.
 * @returns Mutation whose `mutate(title?: string)` returns a NotionPageResult;
 *   `title` is optional and defaults to null (backend chooses a title).
 */
export function useSaveToNotion(id: string) {
  return useMutation({
    mutationFn: (title?: string) =>
      req<NotionPageResult>(`/tasks/${id}/save-to-notion`, {
        method: "POST",
        body: JSON.stringify({ title: title ?? null }),
      }),
  });
}

// ---------------------------------------------------------------------------
// Config export / import (Phase 9) — binary zip, handled outside the JSON helper
// ---------------------------------------------------------------------------

/**
 * Download the config export zip. Fetched (not a plain `<a href>` link) because a
 * navigation cannot carry the bearer token; the zip is handed to the browser as a
 * blob download instead.
 * @throws Error if the export response is not ok (2xx).
 */
export async function downloadConfigExport(): Promise<void> {
  const resp = await fetch(`${API_BASE}/config/export`, { headers: authHeaders() });
  noteAuthFailure(resp);
  if (!resp.ok) throw new Error(`${resp.status}: ${resp.statusText}`);
  const blob = await resp.blob();
  // Keep the server's filename when it sends one.
  const match = /filename="?([^";]+)"?/.exec(resp.headers.get("content-disposition") ?? "");
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = match?.[1] ?? "hyperion-config.zip";
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Fetch a task artifact as text (e.g. `result.md`), with the bearer token.
 * @param taskId - The Hyperion task id.
 * @param name - Artifact file name under the task directory.
 * @returns The artifact body.
 * @throws Error `HTTP <status>` on a non-2xx response.
 */
export async function fetchArtifactText(taskId: string, name: string): Promise<string> {
  const resp = await fetch(`${API_BASE}/tasks/${taskId}/artifacts/${name}`, {
    headers: authHeaders(),
  });
  noteAuthFailure(resp);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.text();
}

/**
 * Upload a previously-exported config zip to restore agents/workflows.
 *
 * Bypasses the JSON `req` helper because it sends multipart/form-data (binary
 * file upload); error handling mirrors `req` (throws `"<status>: <detail>"`).
 * Note: the browser sets the multipart boundary header automatically when a
 * FormData body is used, so no content-type header is set here.
 *
 * @param file - The config zip File selected by the user.
 * @returns Summary of what was imported: restored ids, count, and workflow ids.
 * @throws Error if the upload response is not ok (2xx).
 */
export async function importConfig(
  file: File,
): Promise<{ imported: string[]; count: number; workflows: string[] }> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE}/config/import`, {
    method: "POST",
    body: form,
    headers: authHeaders(),
  });
  noteAuthFailure(resp);
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      detail = (await resp.json()).detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json();
}

// ---------------------------------------------------------------------------
// Monitoring — run history, per-agent metrics, thresholds (Phase 8)
// ---------------------------------------------------------------------------

/** A single row in the paginated run-history list. */
export interface TaskListItem {
  task_id: string;
  status: TaskStatus;
  request: string;
  error: string | null;
  created_at: string;
  updated_at: string;
  hitl: string | null;
  langfuse_url: string | null;
}

/** A page of run history: the items plus pagination metadata. */
export interface TasksPage {
  total: number;
  limit: number;
  offset: number;
  items: TaskListItem[];
}

/**
 * Query hook: fetch a page of run history. Polls every 5s so the list stays live
 * as tasks progress.
 * @param limit - Page size (default 50).
 * @param offset - Row offset for pagination (default 0).
 * @returns TanStack Query result with `data: TasksPage`.
 */
export function useTasks(limit = 50, offset = 0) {
  return useQuery({
    queryKey: ["tasks", limit, offset],
    queryFn: () => req<TasksPage>(`/tasks?limit=${limit}&offset=${offset}`),
    refetchInterval: 5000,
  });
}

/** Aggregated usage/health metrics for one agent over the tracked window. */
export interface AgentMetric {
  id: string;
  name: string;
  group: string;
  active: boolean;
  activations: number;
  errors: number;
  error_rate: number;
  tokens: { input: number; output: number };
  thresholds: Thresholds;
}

/** System-wide metrics: task totals/status breakdown, caps, and per-agent rows. */
export interface MetricsResponse {
  tasks_total: number;
  by_status: Record<string, number>;
  caps: Record<string, number>;
  agents: AgentMetric[];
}

/**
 * Query hook: fetch the metrics dashboard data. Polls every 5s for live updates.
 * @returns TanStack Query result with `data: MetricsResponse`.
 */
export function useMetrics() {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: () => req<MetricsResponse>("/metrics"),
    refetchInterval: 5000,
  });
}

/** Current threshold config: global caps plus per-agent overrides keyed by id. */
export interface ThresholdsResponse {
  global: Record<string, number>;
  agents: Record<string, Thresholds>;
}

/**
 * Query hook: fetch the current global and per-agent thresholds.
 * @returns TanStack Query result with `data: ThresholdsResponse`.
 */
export function useThresholds() {
  return useQuery({
    queryKey: ["thresholds"],
    queryFn: () => req<ThresholdsResponse>("/thresholds"),
  });
}

/**
 * Body for updating thresholds. Top-level `cap_*` fields set global caps; the
 * optional `agents` map applies partial per-agent overrides keyed by agent id.
 * All fields are optional — only the provided keys are changed.
 */
export interface ThresholdUpdateBody {
  cap_input_tokens?: number;
  cap_output_tokens?: number;
  cap_tool_loop?: number;
  cap_wall_seconds?: number;
  agents?: Record<string, Partial<Thresholds>>;
}

/**
 * Mutation hook: update global and/or per-agent thresholds. On success it
 * invalidates both `thresholds` and `metrics` (the metrics view shows caps).
 * @returns Mutation whose `mutate(body: ThresholdUpdateBody)` returns the
 *   updated ThresholdsResponse.
 */
export function useUpdateThresholds() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ThresholdUpdateBody) =>
      req<ThresholdsResponse>("/thresholds", { method: "PUT", body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["thresholds"] });
      qc.invalidateQueries({ queryKey: ["metrics"] });
    },
  });
}

// ---------------------------------------------------------------------------
// Trace Flow — per-task LLM call graph (reuses RoutingResult above)
// ---------------------------------------------------------------------------

/**
 * A single LLM call recorded during a task run, used to render the per-task
 * trace flow / call graph. Includes token and cost accounting, truncated
 * prompt/response previews, tools invoked, and timing.
 */
export interface TraceEvent {
  agent_role: string; // the agent id, e.g. "planner", "researcher", "meta/title"
  node_id: string | null; // workflow node this call ran under (null for meta-prompt calls)
  prompt_type: "user-facing" | "meta-prompt";
  model: string | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  prompt_preview: string | null;
  response_preview: string | null;
  tools_used: string[];
  started_at: string;
  duration_ms: number | null;
}

/**
 * Trace data for one task: the original request, status, routing DAG, and the
 * ordered list of LLM call events used to draw the trace flow graph.
 */
export interface TraceResponse {
  task_id: string;
  request: string;
  status: string;
  routing: RoutingResult | null;
  events: TraceEvent[];
}

/**
 * Query hook: fetch the LLM call trace for a task. Disabled until `taskId` is
 * defined.
 * @param taskId - Task id, or undefined to keep the query idle.
 * @returns TanStack Query result with `data: TraceResponse`.
 */
export function useTraceEvents(taskId: string | undefined) {
  return useQuery({
    queryKey: ["trace", taskId],
    queryFn: () => req<TraceResponse>(`/tasks/${taskId}/trace`),
    enabled: !!taskId,
  });
}
