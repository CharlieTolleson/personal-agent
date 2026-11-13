/**
 * TokenGate — asks for the Hyperion API token before rendering the app.
 *
 * Wraps the whole UI (see `main.tsx`). When no token is stored, or the API has
 * just answered 401 (`AUTH_REQUIRED_EVENT`), it shows a single paste field instead
 * of the app. Saving stores the token (`api/auth.ts`) and invalidates every query so
 * the pages refetch with the new header.
 *
 * The form shows a command that copies the token from `~/ai/ai-router/.env` to the
 * clipboard without printing it, so the value never lands in terminal scrollback.
 */
import { useEffect, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { AUTH_REQUIRED_EVENT, getToken, setToken } from "../api/auth";

/** Shell command that copies the token to the macOS clipboard without echoing it. */
const COPY_COMMAND =
  "grep '^HYPERION_API_TOKEN=' ~/ai/ai-router/.env | cut -d= -f2- | tr -d '\\n\"' | pbcopy";

/**
 * Render children only once an API token is stored.
 * @param props.children - The application tree to show when authenticated.
 * @returns The token form, or the children.
 */
export default function TokenGate({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [hasToken, setHasToken] = useState<boolean>(() => !!getToken());
  const [draft, setDraft] = useState("");

  useEffect(() => {
    // A 401 anywhere in the app clears the token and lands here.
    const onAuthRequired = () => setHasToken(false);
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  }, []);

  if (hasToken) return <>{children}</>;

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <form
        className="card w-full max-w-lg"
        onSubmit={(e) => {
          e.preventDefault();
          if (!draft.trim()) return;
          setToken(draft);
          setDraft("");
          setHasToken(true);
          void queryClient.invalidateQueries();
        }}
      >
        <h1 className="mb-2 text-lg font-semibold">Hyperion API token</h1>
        <p className="mb-3 text-sm text-slate-400">
          The Hyperion API needs a token. Copy it to your clipboard with this command, then
          paste it below. It is kept in this browser only.
        </p>
        <pre className="mb-3 overflow-x-auto whitespace-pre-wrap break-all rounded bg-slate-900 p-2 text-xs text-slate-300">
          {COPY_COMMAND}
        </pre>
        <input
          type="password"
          autoComplete="off"
          className="input mb-3 w-full"
          placeholder="Paste token"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
        />
        <button type="submit" className="btn">
          Save
        </button>
      </form>
    </div>
  );
}
