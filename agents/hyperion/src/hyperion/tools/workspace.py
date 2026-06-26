"""Workspace tools — per-task scratchpad confined to ``tasks/{task_id}/``.

Role in the system
------------------
Hyperion agents (planner/researcher/developer/critic/synthesizer) often need a
place to stash intermediate artifacts — draft files, notes, generated code —
while a task runs. This module exposes three tool classes that
give agents read/write/list access to a *sandboxed* directory unique to the
current task: ``settings.tasks_dir / <task_id>``.

Security model
--------------
The central concern is preventing an LLM-driven agent from escaping its task
sandbox and touching arbitrary host files. All path resolution goes through the
``_safe_path`` helper, which:
  * rejects malformed/traversal-laden ``task_id`` values, and
  * verifies (via ``Path.is_relative_to`` on resolved, symlink-expanded paths)
    that the final target stays inside the task's base directory.
A naïve ``startswith`` check is deliberately avoided because it would let a
sibling directory like ``"abc-evil"`` masquerade as being inside ``"abc"``.

Each tool instance is bound to a single ``task_id`` at construction time (a
required Pydantic field), so the sandbox boundary is fixed per agent run and is
not something the LLM can override through tool input.

Usage
-----
``workspace_tools(task_id)`` is the factory that produces the trio of tools to
hand to a crew/agent for a given task.
"""

from __future__ import annotations

from pathlib import Path

from hyperion.config import settings


def _safe_path(task_id: str, rel_path: str) -> Path:
    """Resolve rel_path inside the task workspace; reject traversal attempts.

    Uses Path.is_relative_to (Python 3.9+) which correctly handles sibling-dir
    escapes — a string startswith check would let "abc-evil" match "abc".

    Args:
        task_id: Identifier of the current task; selects the sandbox directory
            under ``settings.tasks_dir``. Must not be empty and must not contain
            ``/`` or ``..`` (which could redirect the base outside the tasks dir).
        rel_path: Path relative to the task workspace requested by the caller.
            May contain subdirectories; traversal that escapes the sandbox is
            rejected rather than honored.

    Returns:
        An absolute, fully-resolved ``Path`` guaranteed to live inside the
        task's workspace directory.

    Raises:
        ValueError: If ``task_id`` is invalid, or if the resolved ``rel_path``
            would land outside the task's base directory (path traversal).
    """
    # Guard the task_id itself first: a "/" or ".." here could move ``base``
    # out of settings.tasks_dir before the per-path traversal check even runs.
    if not task_id or "/" in task_id or ".." in task_id:
        raise ValueError(f"Invalid task_id: {task_id!r}")
    # resolve() collapses ".." segments and follows symlinks so the containment
    # check below compares real on-disk locations, not lexical strings.
    base = (settings.tasks_dir / task_id).resolve()
    full = (base / rel_path).resolve()
    if not full.is_relative_to(base):
        raise ValueError(f"Path traversal rejected: {rel_path!r}")
    return full


class WorkspaceReadTool:
    """Tool that reads a single file from the bound task's workspace.

    The instance is pinned to one ``task_id``, so the agent can only ever read
    within that task's sandbox.

    Attributes:
        name: Tool name surfaced to the LLM/agent ("workspace_read").
        description: Natural-language usage hint shown to the agent.
        parameters: JSON schema for the tool's arguments.
        task_id: The task whose workspace this tool reads from.
    """

    name = "workspace_read"
    description = (
        "Read a file from the current task's workspace. "
        "Input: relative file path within the task workspace."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path within the workspace."}
        },
        "required": ["path"],
    }

    def __init__(self, task_id: str):
        self.task_id = task_id

    def _run(self, path: str) -> str:
        """Read and return the UTF-8 contents of ``path`` in the workspace.

        Args:
            path: Relative path to the file within the task workspace.

        Returns:
            The file's text contents, or a human-readable ``(File not found: ...)``
            sentinel string when the file does not exist (never raises for a
            missing file, so the agent can react gracefully).

        Raises:
            ValueError: If ``path`` escapes the task sandbox (via ``_safe_path``).
        """
        target = _safe_path(self.task_id, path)
        if not target.exists():
            return f"(File not found: {path})"
        return target.read_text(encoding="utf-8")


class WorkspaceWriteTool:
    """Tool that writes a file into the bound task's workspace.

    Writes are confined to the task sandbox and create any missing parent
    directories. Existing files are overwritten (truncated).

    Attributes:
        name: Tool name surfaced to the LLM/agent ("workspace_write").
        description: Natural-language usage hint shown to the agent.
        parameters: JSON schema for the tool's arguments.
        task_id: The task whose workspace this tool writes to.
    """

    name = "workspace_write"
    description = (
        "Write content to a file in the current task's workspace. "
        "Input: a 'path' (relative file path) and 'content' (string to write)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path within the workspace."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, task_id: str):
        self.task_id = task_id

    def _run(self, path: str, content: str = "") -> str:
        """Write ``content`` to ``path`` within the task workspace.

        Args:
            path: Relative destination path within the task workspace.
            content: Text to write (UTF-8). Defaults to empty string, which
                creates/truncates the file to zero length.

        Returns:
            A confirmation string reporting how many characters were written.

        Raises:
            ValueError: If ``path`` escapes the task sandbox (via ``_safe_path``).

        Side effects:
            Creates parent directories as needed and overwrites any existing
            file at the target path.
        """
        target = _safe_path(self.task_id, path)
        # Ensure the directory tree exists so writes to nested paths succeed.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"


class WorkspaceListTool:
    """Tool that lists every file in the bound task's workspace.

    Attributes:
        name: Tool name surfaced to the LLM/agent ("workspace_list").
        description: Natural-language usage hint shown to the agent.
        parameters: JSON schema for the tool's arguments (none needed).
        task_id: The task whose workspace this tool enumerates.
    """

    name = "workspace_list"
    description = "List all files in the current task's workspace. No input needed."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, task_id: str):
        self.task_id = task_id

    def _run(self, _: str = "") -> str:
        """Recursively list the workspace's files as newline-joined relative paths.

        Args:
            _: Unused; present only to satisfy the tool's ``_run`` signature
                (this tool takes no meaningful input).

        Returns:
            A newline-separated, alphabetically sorted list of file paths
            relative to the task workspace, or ``(Workspace is empty.)`` when the
            directory is absent or contains no files.

        Note:
            Directories are excluded (``p.is_file()``); only leaf files appear.
            The base path is used directly (not via ``_safe_path``) because the
            ``task_id`` is fixed at construction and no caller-supplied path is
            involved here.
        """
        base = settings.tasks_dir / self.task_id
        if not base.exists():
            return "(Workspace is empty.)"
        # rglob("*") walks the whole tree; filter to files and report paths
        # relative to the sandbox root for compactness.
        files = sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())
        return "\n".join(files) if files else "(Workspace is empty.)"


def workspace_tools(task_id: str) -> list:
    """Build the trio of workspace tools bound to a single task.

    Args:
        task_id: Identifier of the task whose sandbox the returned tools operate
            within. All three tools are pinned to this id.

    Returns:
        A list of tool instances — ``WorkspaceReadTool``,
        ``WorkspaceWriteTool``, and ``WorkspaceListTool`` — ready to attach to an
        agent for the given task.
    """
    return [
        WorkspaceReadTool(task_id=task_id),
        WorkspaceWriteTool(task_id=task_id),
        WorkspaceListTool(task_id=task_id),
    ]
