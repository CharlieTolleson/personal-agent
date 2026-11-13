/**
 * RunDetail.tsx — Hyperion UI page for inspecting a single multi-agent run.
 *
 * Route: `/runs/:id` (see the app router). Renders the full lifecycle of one
 * Hyperion task fetched from the Hyperion API (http://localhost:4100 by default,
 * via `API_BASE`):
 *   - status pill + a link to the trace flow view
 *   - error banner (if the run failed)
 *   - any pending "affordance" (an approval/input gate the run is blocked on)
 *   - a free-text "steer" box while the run is actively running
 *   - routing breakdown (which agent/model handled each stage)
 *   - live progress log lines
 *   - the final rendered `result.md` artifact + a "Save to Notion" action
 *
 * Live-update design: `useTask` polls the API every 1500ms while the run is
 * non-terminal and stops polling once the status is terminal (done/failed).
 * Terminal statuses are centralized in the `TERMINAL` constant so the polling
 * predicate and any future callers agree on what "finished" means.
 *
 * This file is purely presentational/orchestration — all data fetching and
 * mutations live in the `../api/client` React Query hooks; this component just
 * wires them to UI and handles local input state.
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  fetchArtifactText,
  useApproveTask,
  useFeedbackTask,
  useSaveToNotion,
  useStopTask,
  useTask,
  type TaskStatus,
} from "../api/client";
import AffordanceView from "../components/Affordance";
import Routing from "../components/Routing";
import { useToast } from "../components/Toast";

/**
 * Statuses that mean the run has finished and will not change further.
 * Used to halt polling (see `useTask` refetchInterval below).
 */
const TERMINAL: TaskStatus[] = ["done", "failed", "cancelled"];

/**
 * Maps a task status to Tailwind border/text utility classes for the status
 * pill. Unknown statuses fall back to a neutral `border-edge` (handled at the
 * call site), so missing keys here are non-fatal.
 */
const statusColor: Record<string, string> = {
  running: "border-sky-500/40 text-sky-200",
  queued: "border-edge text-slate-300",
  awaiting_approval: "border-amber-500/40 text-amber-200",
  awaiting_input: "border-amber-500/40 text-amber-200",
  done: "border-emerald-500/40 text-emerald-200",
  failed: "border-rose-500/40 text-rose-200",
  cancelled: "border-slate-500/40 text-slate-300",
};

/**
 * Fetches and renders the run's final `result.md` artifact as preformatted text.
 *
 * Loads the artifact directly from the API (not via a React Query hook) because
 * it is fetched once when the run is done and does not need polling/caching.
 *
 * @param props.taskId - The Hyperion task id whose result.md to fetch.
 * @returns A loading line, an error line (e.g. artifact missing / non-200), or
 *          the rendered markdown text in a scrollable `<pre>`.
 */
function Result({ taskId }: { taskId: string }) {
  // `text === null` means "not loaded yet"; `err` holds the failure message.
  const [text, setText] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    // Reject non-2xx responses so the .catch path renders the error line
    // (a 404 here just means the artifact was never produced).
    fetchArtifactText(taskId, "result.md")
      .then(setText)
      .catch((e) => setErr(e.message));
  }, [taskId]);

  if (err) return <p className="text-sm text-slate-500">No result.md ({err})</p>;
  if (text === null) return <p className="text-sm text-slate-500">Loading result…</p>;
  return (
    <pre className="card max-h-[500px] overflow-auto whitespace-pre-wrap text-sm text-slate-200">
      {text}
    </pre>
  );
}

/**
 * Top-level page component for `/runs/:id`.
 *
 * Reads the task id from the route params, polls the task while it is live, and
 * composes the status header, error banner, affordance gate, steer box, routing
 * panel, progress log, and final result into a single column.
 *
 * @returns The run detail view, or a loading placeholder until the first fetch.
 */
export default function RunDetail() {
  const { id } = useParams();
  // Non-null assertion: this component is only mounted on the `/runs/:id` route,
  // so `id` is always present.
  const taskId = id!;

  const { data: task } = useTask(taskId, {
    // Poll every 1500ms while running; stop once the status is terminal to
    // avoid hammering the API for finished runs.
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s && TERMINAL.includes(s) ? false : 1500;
    },
  });

  const approve = useApproveTask(taskId);
  const feedback = useFeedbackTask(taskId);
  const stop = useStopTask(taskId);
  // Disable interactive controls while either mutation is in flight.
  const busy = approve.isPending || feedback.isPending;

  if (!task) return <p className="text-slate-400">Loading task…</p>;

  // A run can be stopped while it has not reached a terminal state.
  const stoppable = !TERMINAL.includes(task.status);

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Run</h2>
          <div className="font-mono text-xs text-slate-500">{task.task_id}</div>
        </div>
        <div className="flex items-center gap-2">
          <Link to={`/runs/${taskId}/trace`} className="btn text-sm">
            View Trace
          </Link>
          {stoppable && (
            <button
              className="btn text-sm border-rose-500/40 text-rose-200"
              disabled={stop.isPending}
              onClick={() => stop.mutate()}
            >
              {stop.isPending ? "Stopping…" : "Stop"}
            </button>
          )}
          <span className={`pill ${statusColor[task.status] ?? "border-edge"}`}>{task.status}</span>
        </div>
      </div>

      {task.error && (
        <div className="card border-rose-500/40 bg-rose-500/10 text-sm text-rose-200">
          {task.error}
        </div>
      )}

      {task.pending_affordance && (
        <AffordanceView
          affordance={task.pending_affordance}
          busy={busy}
          onApprove={(body) => approve.mutate(body)}
          onFeedback={(msg) => feedback.mutate(msg)}
        />
      )}

      {task.status === "running" && (
        <div className="card">
          <label className="label">Steer this run (drained between stages)</label>
          <SteerBox onSend={(m) => feedback.mutate(m)} busy={busy} />
        </div>
      )}

      {task.routing && <Routing routing={task.routing} />}

      <div className="card">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Progress
        </h3>
        <pre className="max-h-[300px] overflow-auto whitespace-pre-wrap text-xs text-slate-400">
          {task.progress_lines.join("\n") || "(no progress yet)"}
        </pre>
      </div>

      {task.status === "done" && (
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Result
            </h3>
            <SaveToNotionButton taskId={taskId} />
          </div>
          <Result taskId={taskId} />
        </div>
      )}
    </div>
  );
}

/**
 * Button that persists the finished run's result to Notion and surfaces the
 * outcome via a toast (success shows the created page URL when available).
 *
 * @param props.taskId - The Hyperion task id whose result to save.
 * @returns A button that is disabled while the save mutation is pending.
 */
function SaveToNotionButton({ taskId }: { taskId: string }) {
  const save = useSaveToNotion(taskId);
  const toast = useToast();
  return (
    <button
      className="btn"
      disabled={save.isPending}
      onClick={() =>
        save.mutate(undefined, {
          onSuccess: (r) =>
            toast.push(r.url ? `Saved to Notion → ${r.url}` : "Saved to Notion", "success"),
          onError: (e) => toast.push((e as Error).message, "error"),
        })
      }
    >
      {save.isPending ? "Saving…" : "Save to Notion"}
    </button>
  );
}

/**
 * Free-text input for steering an in-progress run. The message is delivered via
 * the same feedback mutation used for affordances; the backend drains queued
 * steer messages between stages (hence the surrounding label copy).
 *
 * Clears the input after sending. Send is disabled while a mutation is in
 * flight or when the trimmed message is empty.
 *
 * @param props.onSend - Callback invoked with the message text on send.
 * @param props.busy - Whether a mutation is in flight (disables the button).
 * @returns An input + send button row.
 */
function SteerBox({ onSend, busy }: { onSend: (m: string) => void; busy: boolean }) {
  const [msg, setMsg] = useState("");
  return (
    <div className="flex gap-2">
      <input
        className="input"
        placeholder="e.g. focus on the cost section"
        value={msg}
        onChange={(e) => setMsg(e.target.value)}
      />
      <button
        className="btn btn-primary"
        disabled={busy || !msg.trim()}
        onClick={() => {
          onSend(msg);
          setMsg("");
        }}
      >
        Send
      </button>
    </div>
  );
}
