/**
 * main.tsx — Application entry point / bootstrap for the Hyperion UI.
 *
 * Purpose:
 *   This is the root module loaded by Vite (referenced from index.html). It mounts
 *   the React application into the DOM and wires up the global, app-wide providers
 *   that every page and component depends on.
 *
 * Role in the system:
 *   The Hyperion UI is the React + TypeScript + Vite web console (served on :4102)
 *   for the Hyperion multi-agent orchestrator (FastAPI on :4100). This file is the
 *   composition root: it establishes the provider hierarchy and hands off all actual
 *   routing/rendering to <App /> (see ./App.tsx).
 *
 * Provider hierarchy (outermost -> innermost), order is intentional:
 *   1. React.StrictMode      — dev-only checks for unsafe lifecycles/side effects.
 *   2. QueryClientProvider   — TanStack Query cache for server-state (API calls to
 *                              the Hyperion backend). Must wrap anything using hooks
 *                              like useQuery/useMutation.
 *   3. BrowserRouter         — react-router-dom history/routing context. Must wrap
 *                              <App /> so route components and <Link>s resolve.
 *   4. ToastProvider         — app-wide toast/notification context (see ./components/Toast).
 *   5. <TokenGate>           — asks for the Hyperion API token (stored per browser)
 *                              before anything calls the API; needs QueryClient.
 *   6. <App />               — the actual route tree and UI.
 *
 * Key design decisions:
 *   - A single shared QueryClient is created once at module scope (not per-render) so
 *     the query cache persists for the lifetime of the page.
 *   - refetchOnWindowFocus is disabled to avoid surprise refetches (and extra backend
 *     load) every time the user tabs back to the window; retry is capped at 1 so a
 *     failing request surfaces an error quickly instead of retrying repeatedly.
 */
import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ToastProvider } from "./components/Toast";
import TokenGate from "./components/TokenGate";
import "./index.css";

/**
 * Shared TanStack Query client for the entire app.
 *
 * Created once at module scope so the in-memory query cache lives for the page's
 * lifetime and is shared across every component via QueryClientProvider below.
 *
 * Default query options:
 *   - refetchOnWindowFocus: false — do not auto-refetch when the window regains focus.
 *   - retry: 1 — retry a failed query at most once before reporting the error.
 */
const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
});

// Mount the React tree into the #root element defined in index.html.
// The non-null assertion (!) is safe because index.html always provides #root;
// it tells TypeScript getElementById will not return null here.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <ToastProvider>
          <TokenGate>
            <App />
          </TokenGate>
        </ToastProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
