"""
Tests for the two token scripts in ``agents/hyperion/scripts``.

* ``mcp_auth_headers.sh`` (Claude Code ``headersHelper``): prints exactly one JSON
  header object from the env file, prefers the file over a stale environment value,
  and on a missing/short token exits 1 with nothing on stdout (fails closed).
* ``create_hyperion_api_token.sh``: writes a 64-hex token without printing it,
  keeps the other lines, leaves the file at 0600, leaves no temp file behind, and
  only replaces an existing token with ``--rotate``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

from auth_helpers import TEST_TOKEN

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script: str, env_file: Path, *args: str, env: dict | None = None):
    """Run a script against a temp env file and return the CompletedProcess."""
    full_env = {"PATH": os.environ["PATH"], "HOME": str(env_file.parent),
                "HYPERION_ENV_FILE": str(env_file), **(env or {})}
    return subprocess.run([str(SCRIPTS / script), *args], capture_output=True,
                          text=True, env=full_env, timeout=30)


def test_headers_helper_prints_bearer_json(tmp_path) -> None:
    """The helper emits {"Authorization": "Bearer <token>"} read from the env file."""
    f = tmp_path / ".env"
    f.write_text(f"OTHER=1\nHYPERION_API_TOKEN=\"{TEST_TOKEN}\"\n")
    out = _run("mcp_auth_headers.sh", f)
    assert out.returncode == 0
    assert json.loads(out.stdout) == {"Authorization": f"Bearer {TEST_TOKEN}"}
    assert out.stderr == ""


def test_headers_helper_prefers_file_over_stale_env(tmp_path) -> None:
    """After rotation, an old exported value in the environment does not win."""
    f = tmp_path / ".env"
    f.write_text(f"HYPERION_API_TOKEN={TEST_TOKEN}\n")
    out = _run("mcp_auth_headers.sh", f, env={"HYPERION_API_TOKEN": "old-" + "a" * 40})
    assert json.loads(out.stdout)["Authorization"] == f"Bearer {TEST_TOKEN}"


def test_headers_helper_fails_closed(tmp_path) -> None:
    """Missing, short, or absent-file tokens: exit 1, empty stdout, no value on stderr."""
    f = tmp_path / ".env"
    for content in ("OTHER=1\n", "HYPERION_API_TOKEN=short-value\n", None):
        if content is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(content)
        out = _run("mcp_auth_headers.sh", f)
        assert out.returncode == 1
        assert out.stdout == ""
        assert "short-value" not in out.stderr


def test_mint_script_writes_secret_quietly(tmp_path) -> None:
    """Minting adds a 64-hex token, prints no token, keeps other lines, sets 0600,
    and leaves no temp file in the directory."""
    f = tmp_path / ".env"
    f.write_text("KEEP_ME=1\n")
    f.chmod(0o644)
    out = _run("create_hyperion_api_token.sh", f)
    assert out.returncode == 0, out.stderr
    text = f.read_text()
    token = re.search(r"^HYPERION_API_TOKEN=([0-9a-f]{64})$", text, re.M).group(1)
    assert "KEEP_ME=1" in text
    assert token not in out.stdout + out.stderr
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


def test_mint_script_rotates_only_when_asked(tmp_path) -> None:
    """Without --rotate an existing token is kept (file still tightened to 0600);
    with --rotate it is replaced by a different one, with a single line remaining."""
    f = tmp_path / ".env"
    f.write_text(f"HYPERION_API_TOKEN={TEST_TOKEN}\n")
    f.chmod(0o644)
    _run("create_hyperion_api_token.sh", f)
    assert f.read_text() == f"HYPERION_API_TOKEN={TEST_TOKEN}\n"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    out = _run("create_hyperion_api_token.sh", f, "--rotate")
    assert out.returncode == 0
    lines = [l for l in f.read_text().splitlines() if l.startswith("HYPERION_API_TOKEN=")]
    assert len(lines) == 1 and TEST_TOKEN not in lines[0]
