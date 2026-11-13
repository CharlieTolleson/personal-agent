#!/usr/bin/env bash
# =============================================================================
# mcp_auth_headers.sh: Claude Code `headersHelper` for the Hyperion MCP server
# =============================================================================
#
# PURPOSE
#   Hyperion's MCP endpoint (:4101) refuses any request without
#   "Authorization: Bearer $HYPERION_API_TOKEN" (src/hyperion/server/auth.py).
#   Claude Code runs this script (configured as `headersHelper` in ~/ai/.mcp.json)
#   on each connection and sends the JSON object it prints as request headers.
#
# WHERE THE TOKEN COMES FROM
#   HYPERION_API_TOKEN in ~/ai/ai-router/.env (override the file with
#   HYPERION_ENV_FILE). That gitignored file is the single store; the containers
#   get the same value through docker compose. If HYPERION_API_TOKEN is already
#   in the environment, that wins.
#
# OUTPUT
#   stdout: {"Authorization": "Bearer <token>"} (the headersHelper contract; this
#   is the only place the value is written, and only to Claude Code's pipe).
#   On a missing token it prints nothing to stdout, names the variable on stderr,
#   and exits 1, so the connection fails closed with a clear reason.
#
# USAGE
#   scripts/mcp_auth_headers.sh            # do not run it by hand in a shared
#                                          # terminal: it prints the secret
# =============================================================================
set -euo pipefail

ENV_FILE="${HYPERION_ENV_FILE:-$HOME/ai/ai-router/.env}"

# python3 parses the dotenv line (handles optional quotes) and emits JSON safely;
# the token goes through the environment, never through argv (visible in `ps`).
HYPERION_ENV_FILE="$ENV_FILE" exec python3 - <<'PY'
import json, os, sys

token = os.environ.get("HYPERION_API_TOKEN", "").strip()
if not token:
    path = os.environ["HYPERION_ENV_FILE"]
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if line.startswith("HYPERION_API_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
if len(token) < 32:
    sys.stderr.write("mcp_auth_headers: HYPERION_API_TOKEN not found or too short\n")
    sys.exit(1)
sys.stdout.write(json.dumps({"Authorization": f"Bearer {token}"}))
PY
