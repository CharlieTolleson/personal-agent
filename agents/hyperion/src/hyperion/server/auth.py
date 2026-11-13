"""
Bearer-token gate for every Hyperion HTTP surface (API on :4100, MCP on :4101).

Why this exists
---------------
Hyperion listens on localhost, and every coding agent on this Mac can reach localhost
ports from its sandbox (measured 2026-11-12). Before this module, any local process
could start paid runs, export or import config, save to Notion, and overwrite source
files under ``/admin/files`` with no credential. This middleware closes that: every
HTTP and WebSocket request must carry ``Authorization: Bearer <HYPERION_API_TOKEN>``.

Design decisions
----------------
- **One shared token, from the environment.** ``HYPERION_API_TOKEN`` lives in
  ``~/ai/ai-router/.env`` (gitignored, the same store as the LiteLLM master key) and
  reaches the containers through compose ``environment:`` interpolation. It is read
  per request so tests can set it, and so a fresh value takes effect on the next
  container recreate without code changes.
- **Fail closed.** No token configured, or one shorter than ``MIN_TOKEN_LENGTH``,
  means every request gets 503; the entrypoints also refuse to start (``require_token``).
- **Constant-time compare.** Both sides are hashed with SHA-256 and compared with
  ``hmac.compare_digest``, so neither the content nor the length of the expected
  token leaks through timing.
- **Header only.** No query-string or cookie fallback: a token in a URL lands in
  access logs, browser history, and ``ps``-visible command lines.
- **Pure ASGI, not BaseHTTPMiddleware.** The API streams SSE and the MCP endpoint
  streams responses; a pure ASGI wrapper does not buffer or break those.
- **CORS stays outermost on the API.** The browser UI's preflight ``OPTIONS`` cannot
  carry ``Authorization``; Starlette's ``CORSMiddleware`` answers a valid preflight
  itself without calling inner apps, so it must wrap this middleware (in FastAPI the
  last ``add_middleware`` call is outermost). A preflight reveals nothing, and a
  non-preflight ``OPTIONS`` still reaches this gate and gets 401.
- **No route is exempt,** including docs, ``/openapi.json``, and liveness paths.
- **Nothing secret is logged.** Refusals log the method and path only (never headers),
  and the error body is a fixed string.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Name of the environment variable holding the shared token.
TOKEN_ENV = "HYPERION_API_TOKEN"

# 32 characters is the floor for a random token (`openssl rand -hex 32` gives 64).
# Shorter values are treated as unset so a placeholder or typo can't open the gate.
MIN_TOKEN_LENGTH = 32


class TokenNotConfigured(RuntimeError):
    """Raised at startup when ``HYPERION_API_TOKEN`` is missing or too short."""


def _expected_token() -> str | None:
    """Return the configured token, or None when it is missing or too short.

    Returns:
        The token string from ``HYPERION_API_TOKEN`` with surrounding whitespace
        stripped, or None if it is unset, shorter than ``MIN_TOKEN_LENGTH``, or a
        ``change-me`` placeholder.
    """
    value = os.environ.get(TOKEN_ENV, "").strip()
    # A copied .env.example placeholder ("change-me...") is long enough to pass
    # the length floor, so reject it by prefix too.
    if len(value) < MIN_TOKEN_LENGTH or value.lower().startswith("change-me"):
        return None
    return value


def require_token() -> None:
    """Refuse to start a server without a usable token.

    Called from each entrypoint's ``main`` so a missing secret shows up as a crash
    at boot (visible in ``docker compose ps``) instead of a server that answers 503
    forever. The error names the variable, never a value.

    Raises:
        TokenNotConfigured: If ``HYPERION_API_TOKEN`` is unset or too short.
    """
    if _expected_token() is None:
        raise TokenNotConfigured(
            f"{TOKEN_ENV} is unset or shorter than {MIN_TOKEN_LENGTH} characters; "
            "Hyperion will not start without it (see agents/hyperion/README.md, Auth)."
        )


def _digest(value: bytes) -> bytes:
    """Hash a value so the constant-time compare sees equal-length inputs.

    Args:
        value: Raw bytes to hash.

    Returns:
        The 32-byte SHA-256 digest.
    """
    return hashlib.sha256(value).digest()


def is_authorized(authorization: bytes | None) -> bool:
    """Check an ``Authorization`` header value against the configured token.

    Args:
        authorization: The raw header value, or None when the header is absent.

    Returns:
        True only when a token is configured and the header is exactly
        ``Bearer <token>`` (scheme case-insensitive). False otherwise, including
        when no token is configured (fail closed).
    """
    expected = _expected_token()
    if expected is None or not authorization:
        return False
    scheme, _, presented = authorization.partition(b" ")
    if scheme.lower() != b"bearer" or not presented:
        return False
    return hmac.compare_digest(_digest(presented.strip()), _digest(expected.encode()))


def _header(scope: Scope, name: bytes) -> bytes | None:
    """Return the first value of a request header from an ASGI scope.

    Args:
        scope: ASGI connection scope (``headers`` is a list of lowercase byte pairs).
        name: Lowercase header name.

    Returns:
        The header value, or None if absent.
    """
    for key, value in scope.get("headers") or ():
        if key == name:
            return value
    return None


class BearerTokenMiddleware:
    """ASGI middleware that refuses any HTTP or WebSocket request without the token.

    Lifespan events pass through untouched so startup and shutdown still run.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap an ASGI app.

        Args:
            app: The inner ASGI application to protect.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Gate one ASGI connection.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        if _expected_token() is None:
            # Misconfiguration, not a caller error: 503 so it reads as "server not
            # ready", and never fall through to the app.
            logger.error("auth: %s not configured; refusing %s", TOKEN_ENV, scope.get("path"))
            await self._refuse(scope, send, 503, "auth not configured")
            return

        if not is_authorized(_header(scope, b"authorization")):
            logger.warning("auth: refused %s %s", scope.get("method", "WS"), scope.get("path"))
            await self._refuse(scope, send, 401, "unauthorized")
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(scope: Scope, send: Send, status: int, detail: str) -> None:
        """Send a fixed JSON refusal (HTTP) or close the socket (WebSocket).

        Args:
            scope: ASGI connection scope.
            send: ASGI send callable.
            status: HTTP status code to return.
            detail: Fixed, non-sensitive message for the body.
        """
        if scope["type"] == "websocket":
            # 1008 = policy violation. Closing before accept rejects the handshake.
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": detail}).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if status == 401:
            headers.append((b"www-authenticate", b"Bearer"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
