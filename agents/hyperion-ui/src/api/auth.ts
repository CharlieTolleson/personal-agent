/**
 * Hyperion UI — bearer-token storage for the Hyperion API.
 *
 * The Hyperion API (:4100) refuses every request without
 * `Authorization: Bearer <HYPERION_API_TOKEN>` (see `agents/hyperion/src/hyperion/server/auth.py`).
 * This UI is a static bundle served by nginx on :4102 and calls the API straight from
 * the browser, so the token cannot be baked into the bundle (anyone who could load
 * :4102 would then hold it) and nginx must not proxy with it (that would turn :4102
 * into an open door). Instead the operator pastes the token once; it is kept in this
 * origin's localStorage and attached to every request by `authHeaders()`.
 *
 * Design notes:
 * - Storage access is wrapped in try/catch: localStorage can throw (private mode,
 *   blocked site data), and the UI must then simply ask again.
 * - The token is never put in a URL (no query-string auth), so it never reaches
 *   nginx or Hyperion access logs or browser history.
 * - A 401 from the API clears the stored token and fires `AUTH_REQUIRED_EVENT`,
 *   which `TokenGate` listens for to show the paste form again (rotation, typo).
 */

/** localStorage key holding the token for this origin. */
const STORAGE_KEY = "hyperion.apiToken";

/** Window event fired when the API rejects the stored token (or none is stored). */
export const AUTH_REQUIRED_EVENT = "hyperion:auth-required";

/**
 * Read the stored API token.
 * @returns The token, or null when none is stored or storage is unavailable.
 */
export function getToken(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

/**
 * Store the API token for later requests.
 * @param token - The token the operator pasted (surrounding whitespace removed).
 */
export function setToken(token: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, token.trim());
  } catch {
    /* storage blocked: the gate will ask again on the next 401 */
  }
}

/** Forget the stored token (used after a 401). */
export function clearToken(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* nothing to clear */
  }
}

/**
 * Headers that authenticate a request to the Hyperion API.
 * @returns `{ Authorization: "Bearer <token>" }`, or `{}` when no token is stored
 *   (the request then gets a 401, which reopens the token form).
 */
export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * React to an API response: on 401, drop the stored token and ask for a new one.
 *
 * Only clears when the rejected request carried the token that is still stored, so
 * a late 401 for a request sent with an old token can't wipe a freshly saved one.
 *
 * @param resp - Any response from the Hyperion API.
 * @param sentHeaders - The auth headers that request was sent with (`authHeaders()`).
 */
export function noteAuthFailure(resp: Response, sentHeaders: Record<string, string>): void {
  if (resp.status !== 401) return;
  const sent = sentHeaders.Authorization ?? null;
  const current = authHeaders().Authorization ?? null;
  if (sent !== current) return;
  clearToken();
  window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
}
