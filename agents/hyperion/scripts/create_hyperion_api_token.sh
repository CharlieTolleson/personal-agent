#!/usr/bin/env bash
# =============================================================================
# create_hyperion_api_token.sh: mint the Hyperion bearer token without showing it
# =============================================================================
#
# PURPOSE
#   Writes a fresh random HYPERION_API_TOKEN (64 hex chars from the OS CSPRNG)
#   into ~/ai/ai-router/.env, the gitignored file every Hyperion caller reads it
#   from (see src/hyperion/server/auth.py). The value is never printed, logged, or
#   passed on a command line.
#
# USAGE
#   scripts/create_hyperion_api_token.sh            # add one if none is set
#   scripts/create_hyperion_api_token.sh --rotate   # replace the existing one
#   HYPERION_ENV_FILE=/path/.env scripts/...         # use another env file
#
# AFTER RUNNING (the new value takes effect only on recreate, not restart)
#   cd ~/ai/ai-router && docker compose up -d hyperion hyperion-mcp open-webui
#   The Hyperion UI (:4102) asks for the token again; Claude Code picks it up on
#   its next MCP connection.
# =============================================================================
set -euo pipefail

ENV_FILE="${HYPERION_ENV_FILE:-$HOME/ai/ai-router/.env}"
ROTATE="${1:-}"

HYPERION_ENV_FILE="$ENV_FILE" ROTATE="$ROTATE" exec python3 - <<'PY'
import os, secrets, sys

path = os.environ["HYPERION_ENV_FILE"]
rotate = os.environ.get("ROTATE") == "--rotate"
with open(path, encoding="utf-8") as fh:
    lines = fh.read().splitlines()

idx = [i for i, l in enumerate(lines) if l.strip().startswith("HYPERION_API_TOKEN=")]
existing = idx and len(lines[idx[-1]].split("=", 1)[1].strip().strip('"\'')) >= 32
if existing and not rotate:
    print(f"HYPERION_API_TOKEN already set in {path}; pass --rotate to replace it.")
    sys.exit(0)

new_line = "HYPERION_API_TOKEN=" + secrets.token_hex(32)
if idx:
    for i in idx:
        lines[i] = None
    lines[idx[0]] = new_line
    lines = [l for l in lines if l is not None]
else:
    lines += ["", "# Hyperion API + MCP bearer token (agents/hyperion/src/hyperion/server/auth.py)", new_line]

# Write via a temp file with 0600 perms, then atomically replace.
tmp = path + ".tmp"
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
os.replace(tmp, path)
print(f"{'Rotated' if existing else 'Wrote'} HYPERION_API_TOKEN in {path} (value not shown).")
PY
