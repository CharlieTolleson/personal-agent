"""
webhooks.py — outbound task-completion callbacks with an SSRF guard (Phase 9).

A task may carry a ``callback_url``; on terminal completion we POST the result
there exactly once. Because the URL is caller-supplied and the API can reach
internal services (LiteLLM, Langfuse, the Docker network), every URL is
checked against an SSRF allowlist first: the host must resolve *entirely* to
private, loopback, or link-local addresses. A host that resolves to any public
address — or fails to resolve — is rejected before any connection is attempted.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

from hyperion.config import settings

logger = logging.getLogger(__name__)


class UnsafeCallbackURL(ValueError):
    """Raised when a ``callback_url`` fails the SSRF allowlist check.

    Subclasses :class:`ValueError` so existing call sites that catch broad value
    errors still behave, while callers that want to distinguish guard rejections
    (e.g. :func:`fire_callback`) can catch this specific type.
    """


def _is_private_ip(ip: str) -> bool:
    """Return True when ``ip`` is a non-routable (safe-to-call) address.

    "Safe" here means the address is private (RFC 1918 / ULA), loopback, or
    link-local — i.e. it cannot reach the public internet and is therefore
    eligible under the SSRF allowlist.

    Args:
        ip: A textual IPv4 or IPv6 address (e.g. ``"127.0.0.1"`` or ``"::1"``).

    Returns:
        bool: True if the address is private, loopback, or link-local; False if
        it is a globally routable (public) address.

    Raises:
        ValueError: If ``ip`` is not a valid IP address literal (propagated from
            :func:`ipaddress.ip_address`).
    """
    addr = ipaddress.ip_address(ip)
    return addr.is_private or addr.is_loopback or addr.is_link_local


def validate_callback_url(url: str) -> None:
    """Raise UnsafeCallbackURL unless every resolved address is private/loopback.

    Resolving the host (not just string-matching it) is what defeats DNS-rebinding
    and decimal/hex IP encodings — we check the addresses the OS would actually dial.

    Args:
        url: The caller-supplied callback URL to vet before any connection.

    Returns:
        None: Returns normally (no value) when the URL is considered safe, or
        when the SSRF guard is explicitly disabled via configuration.

    Raises:
        UnsafeCallbackURL: If the guard is enabled and any of the following hold:
            the scheme is not http/https; the URL has no host; the host fails to
            resolve; the host resolves to zero addresses; or *any* resolved
            address is publicly routable. Rejection happens before connecting.

    Notes:
        Controlled by ``settings.hyperion_callback_ssrf_guard``; the value ``"off"``
        (case-insensitive) bypasses all checks. Absent/empty config defaults to
        guard ON (fail-safe). Requiring *every* resolved address to be private
        (not just one) prevents a host that mixes public and private records from
        slipping through.
    """
    # Default to guard ON when unset/empty; only the literal "off" disables it.
    if (settings.hyperion_callback_ssrf_guard or "on").lower() == "off":
        return

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeCallbackURL(f"callback_url scheme must be http/https, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeCallbackURL("callback_url has no host")

    try:
        # Resolve via the OS resolver so we vet the addresses that would actually
        # be dialed; default to the scheme's standard port when none is given.
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeCallbackURL(f"callback_url host {host!r} did not resolve: {exc}") from exc

    # getaddrinfo entries are 5-tuples; index [4][0] is the sockaddr's IP string.
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise UnsafeCallbackURL(f"callback_url host {host!r} resolved to no addresses")
    for ip in addresses:
        if not _is_private_ip(ip):
            raise UnsafeCallbackURL(
                f"callback_url host {host!r} resolves to public address {ip} — refusing (SSRF guard)"
            )


async def fire_callback(url: str, payload: dict) -> bool:
    """POST ``payload`` to ``url`` once. Best-effort: never raises into the caller.
    Returns True on a 2xx response, False otherwise (including a guard rejection).

    Intended to be called on terminal task completion to deliver the result to a
    caller-supplied callback. Designed to be swallow-all so that a failed or
    malicious callback can never disrupt task bookkeeping.

    Args:
        url: The destination callback URL (vetted by :func:`validate_callback_url`
            before any request is made).
        payload: JSON-serializable body to POST (sent as the request's ``json``).

    Returns:
        bool: True if the SSRF check passed and the server replied with a 2xx
        status; False on guard rejection, non-2xx response, timeout, or any
        network/transport error.

    Notes:
        - Uses a fixed 10-second timeout for connect+read+write.
        - Sends exactly one POST attempt; there is no retry.
        - All exceptions are caught and logged at WARNING; none propagate.
    """
    try:
        validate_callback_url(url)
    except UnsafeCallbackURL as exc:
        logger.warning("Skipping callback: %s", exc)
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
        ok = 200 <= resp.status_code < 300
        if not ok:
            logger.warning("Callback to %s returned HTTP %s", url, resp.status_code)
        return ok
    except Exception as exc:  # network failure must never break task bookkeeping
        logger.warning("Callback to %s failed: %s", url, exc)
        return False
