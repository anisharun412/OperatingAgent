"""The bridge that lets the agent use the MCP tools.

The agent doesn't ship any file tools of its own anymore. Instead it borrows the
whole fleet - read/write files, git, the terminal, search - from the gateway MCP
server that already composes them. This module is the adapter between the two:

  * `MCPToolProvider` opens one in-memory connection to the gateway, asks it for
    its tool list, and hands back one `MCPTool` per tool. It closes the same
    connection when the run is over.
  * `MCPTool` is a normal `Tool` as far as the rest of the agent is concerned. It
    carries the gateway tool's name, description and argument schema, and when the
    model calls it, it forwards the call over the connection and turns whatever
    comes back into a plain `ToolResult`.

Nothing here needs the network: the gateway runs *inside this process* over
FastMCP's in-memory transport, so there's no subprocess and no socket. That's why
`fastmcp` and the gateway are only imported when a connection is actually opened -
the agent still imports and runs fine on a machine that hasn't installed the MCP
extra.

Two honest touches for safety. Each tool is tagged with permission flags inferred
from its name (`_infer_permissions`), so the existing policy asks before anything
that writes, deletes, or runs a shell command - it never has to understand MCP to
do its job. And every failure the gateway reports comes back as a failed
`ToolResult`, never an exception, so a broken tool can't end a run.

Those flags are also where the shell tool gets marked `SANDBOX`, which is how it
ends up running inside a container instead of in this process (see
`tools/sandbox.py`). When the configured sandbox cannot be created, it degrades
to running the command on the host in the session workspace (see the pool's
`fallback` flag): set `AGENT_SANDBOX_FALLBACK=error` to get the failed result
back instead of a host fallback.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .base import (
    ExecutionMode,
    Tool,
    ToolDefinition,
    ToolPermissions,
    ToolResult,
)

#: The one tool here that is a shell command, and so the one that can be moved
#: into a container.
TERMINAL_TOOL = "terminal_run_command"

#: Added to the terminal tool's description, because where a command runs changes
#: what it can do and the model should know before it writes one.
_SANDBOX_NOTE = (
    "\n\nThis usually runs in a locked container: the project folder is the working "
    "directory (mounted at /workspace), there is no network, and nothing outside the "
    "project folder and /tmp is writable. Use paths relative to the project folder - "
    "an absolute path from the user's machine won't exist in there. If Docker or "
    "the sandbox image is unavailable, the command runs directly on the user's "
    "machine in the project folder instead (no isolation), so it needs the real "
    "absolute paths on that machine."
)


# One MCP tool, wearing the agent's Tool coat
class MCPTool(Tool):
    """A single tool exposed by the gateway, made to look like any other Tool.

    It never runs anything itself: `execute` forwards the call over the shared
    connection and translates the answer. The connection is owned by the
    `MCPToolProvider` that made this tool, so many tools share one link.
    """

    def __init__(self, client: Any, spec: Any, client_resolver: Any = None) -> None:
        self._client = client
        self._client_resolver = client_resolver
        name = _spec_field(spec, "name") or ""
        description = _spec_field(spec, "description") or ""
        schema = (
            _spec_field(spec, "inputSchema")
            or _spec_field(spec, "input_schema")
            or {"type": "object", "properties": {}}
        )
        # namespace is left empty so full_name is exactly the gateway's tool name
        # (e.g. "filesystem_read_file") - the same name the model calls and the
        # registry looks up.
        self._definition = ToolDefinition(
            name=name,
            description=description + (_SANDBOX_NOTE if name == TERMINAL_TOOL else ""),
            input_schema=schema,
            permissions=_infer_permissions(name),
            namespace="",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def preview(self, arguments: dict) -> str:
        """A readable one-liner for the permission prompt (the real call, not a guess)."""
        name = self._definition.name
        args = arguments or {}
        if name == "terminal_run_command":
            return f"Run a shell command: {args.get('command', '')}".rstrip()
        if name.startswith("filesystem_"):
            verb = name[len("filesystem_"):].replace("_", " ")
            path = args.get("path") or args.get("source") or args.get("destination")
            return f"{verb}: {path}" if path else verb
        return f"{name}({_short_args(args)})"

    async def execute(self, arguments: dict, context: Any) -> ToolResult:
        """Forward the call to the gateway and turn the reply into a ToolResult."""
        try:
            client = (
                self._client_resolver(context)
                if self._client_resolver is not None
                else self._client
            )
            if client is None:
                return ToolResult(
                    False,
                    error=(
                        f"MCP workspace is not attached for "
                        f"{self._definition.name!r}"
                    ),
                )
            raw = await client.call_tool(self._definition.name, dict(arguments or {}))
        except Exception as exc:  # a tool error surfaces here on some versions
            return ToolResult(
                False,
                error=f"MCP tool {self._definition.name!r} failed: {type(exc).__name__}: {exc}",
            )
        return _to_tool_result(raw)

    def sandbox_command(self, arguments: dict) -> str | None:
        """The shell command this call would run, so it can run in a container.

        Only the terminal tool has one. Everything else here answers None and keeps
        going through the gateway in this process: the file and git tools aren't
        commands, and the file server's own path fence is a better check than
        re-implementing them as shell calls (see `tools/sandbox.py` for that
        decision and what it costs).
        """
        if self._definition.name != TERMINAL_TOOL:
            return None
        command = (arguments or {}).get("command") or ""
        return str(command).strip() or None


# The connection to the gateway
class MCPToolProvider:
    """Opens in-memory gateway links and lends out workspace-aware tools.

    Each workspace gets its own gateway link. Call `connect()` when a workspace
    is first used (it returns tools to register), then `close()` at shutdown.
    Both `fastmcp` and the gateway are imported lazily, so importing this module
    never requires the MCP extra to be installed.
    """

    def __init__(self, gateway_factory: Any = None) -> None:
        # gateway_factory lets a test pass a tiny fake gateway; production leaves
        # it None and we build the real one.
        self._gateway_factory = gateway_factory
        self._client: Any = None
        self._clients: dict[str, Any] = {}
        self._connect_lock = asyncio.Lock()
        self._shutdown = False

    async def connect(self, root: str | None = None) -> list:
        """Open the connection and return one MCPTool per gateway tool.

        `root`, when given, is the one folder the tools are confined to. It is
        passed to `build_gateway` by argument, which builds the file, git and
        terminal sub-servers pinned to it - so the folder never has to be written
        into a shared `*_SERVER_ROOT` environment variable, and two providers in
        one process can serve two different folders without clobbering each other.
        """
        resolved = str(Path(root).expanduser().resolve()) if root else str(Path.cwd().resolve())
        async with self._connect_lock:
            if self._shutdown:
                raise RuntimeError("MCP tool provider is shut down")
            if resolved in self._clients:
                return []

            Client = _import_client()
            gateway = self._build_gateway(resolved)
            client = Client(gateway)
            # Hold the async connection open for the whole run; close() ends it.
            try:
                await client.__aenter__()
                specs = await client.list_tools()
            except Exception:
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    pass
                try:
                    closer = getattr(client, "close", None)
                    if callable(closer):
                        await closer()  # type: ignore[operator]
                except Exception:
                    pass
                raise

            self._clients[resolved] = client
            self._client = client
            return [
                MCPTool(client, spec, client_resolver=self._client_for_context)
                for spec in specs
            ]

    async def close(self) -> None:
        """End the connection. Safe to call more than once."""
        async with self._connect_lock:
            self._shutdown = True
            clients, self._clients = self._clients, {}
            self._client = None
            for client in clients.values():
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    # Best-effort teardown: if it's already closed there's nothing to do.
                    pass

    def _client_for_context(self, context: Any) -> Any | None:
        """Resolve the MCP connection from the active native session."""
        session = getattr(context, "session", None)
        working_directory = getattr(session, "working_directory", None)
        if working_directory:
            try:
                root = str(Path(working_directory).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                return None
            # A session with an explicit workspace must never silently borrow a
            # client attached to another workspace.
            return self._clients.get(root)
        # A direct MCPTool test or a single-root runtime can use the only client
        # even when no session context is available.
        if len(self._clients) == 1:
            return next(iter(self._clients.values()))
        return None

    def _build_gateway(self, root: str | None = None) -> Any:
        if self._gateway_factory is not None:
            return self._gateway_factory()
        try:
            from gateway_server.server import build_gateway
        except ImportError as exc:
            raise RuntimeError(
                "The MCP gateway isn't importable, so the agent can't reach its "
                "tools. From the repo root run `uv sync --all-packages`."
            ) from exc
        return build_gateway(root=root)


def _import_client() -> Any:
    """Import FastMCP's Client lazily, with a friendly message if it's missing."""
    try:
        from fastmcp import Client
    except ImportError as exc:
        raise RuntimeError(
            "The 'fastmcp' package isn't installed, so the agent can't reach the "
            "MCP tools. From the repo root run `uv sync --all-packages`."
        ) from exc
    return Client


# Permissions: infer honest flags from the tool's name
# Filesystem tools that only look.
_FS_READS = {
    "read_file",
    "list_directory",
    "exists",
    "metadata",
    "search_files",
    "watch_directory",
}
# Filesystem tools that change the user's files.
_FS_WRITES = {
    "write_file",
    "delete_file",
    "delete_directory",
    "create_directory",
    "move_file",
    "copy_file",
    "rename_file",
}
_SEARCH_READS = {"search_documents", "list_indices"}
# Multi-character git verbs that clearly change state; kept unambiguous on purpose
# so a read tool ("git_status", "diff") is never mistaken for a mutation.
_GIT_WRITE_HINTS = (
    "commit",
    "checkout",
    "reset",
    "merge",
    "stash",
    "revert",
    "restore",
    "apply",
)
_GIT_NETWORK_HINTS = ("push", "pull", "fetch", "clone")


def _infer_permissions(name: str) -> ToolPermissions:
    """Read the tool's name and set the flags the policy relies on.

    The categories match the gateway's namespaces (`filesystem_`, `git_`,
    `terminal_`, `search_`). When in doubt the result is "not read-only", which
    makes the policy stop and ask rather than assume a tool is safe.

    `reaches_paths` is set on every tool whose reach comes from an argument - a
    path, a repository, a working directory. It is set even on the read-only ones,
    because "this only reads" says nothing about *where* it reads; that is what
    `WorkspacePolicy` checks.
    """
    category, _, rest = name.partition("_")

    if category == "filesystem":
        if rest in _FS_READS:
            return ToolPermissions(read_only=True, reaches_paths=True)
        # Known writes, and anything unrecognised, are treated as able to change files.
        return ToolPermissions(destructive=True, reaches_paths=True)

    if category == "git":
        if any(hint in rest for hint in _GIT_NETWORK_HINTS):
            return ToolPermissions(destructive=True, needs_network=True, reaches_paths=True)
        if any(hint in rest for hint in _GIT_WRITE_HINTS):
            return ToolPermissions(destructive=True, reaches_paths=True)
        # diff, git_log, git_status, list_branches: read-only, but they still take
        # a repository argument, so where they look is still worth checking.
        return ToolPermissions(read_only=True, reaches_paths=True)

    if category == "terminal":
        if rest == "list_processes":
            return ToolPermissions(read_only=True)
        # run_command is a real shell. Marked SANDBOX so the manager routes it into
        # the session's container. An unavailable configured sandbox is a failed
        # tool result, never an implicit host-process fallback.
        return ToolPermissions(
            destructive=True,
            reaches_paths=True,
            execution_mode=ExecutionMode.SANDBOX,
        )

    if category == "search":
        if rest in _SEARCH_READS:
            return ToolPermissions(read_only=True)
        # index_documents writes to an in-memory index: not destructive to the
        # user's system, but it does read files off disk, so its reach matters.
        return ToolPermissions(reaches_paths=True)

    if category == "gateway":
        return ToolPermissions(read_only=True)  # gateway_health is a liveness probe

    # An unknown server: don't assume it's safe.
    return ToolPermissions()


# Shape helpers: MCP replies -> ToolResult, tolerant of version differences
def _spec_field(spec: Any, name: str, default: Any = None) -> Any:
    """Read a field off a tool spec, whether it's an object or a dict."""
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _to_tool_result(raw: Any) -> ToolResult:
    """Turn whatever `call_tool` returned into a ToolResult.

    Newer FastMCP returns a result object with `.content`, `.data`,
    `.structured_content` and `.is_error`; very old versions returned a bare list
    of content blocks. Both are handled.
    """
    is_error = bool(_spec_field(raw, "is_error", False))
    text = _extract_output(raw)
    if is_error:
        return ToolResult(False, error=text or "the tool reported an error")
    return ToolResult(True, output=text)


def _extract_output(raw: Any) -> str:
    """Pull readable text out of a tool reply, preferring the human-facing content."""
    content = _spec_field(raw, "content")
    if content is None and isinstance(raw, (list, tuple)):
        content = raw  # very old shape: the reply *is* the block list

    text = _text_from_blocks(content)
    if text:
        return text

    structured = _spec_field(raw, "structured_content")
    if structured is not None:
        return _as_text(structured)

    data = _spec_field(raw, "data")
    if data is not None:
        return _as_text(data)

    # Nothing structured and no blocks: fall back to a string of the whole reply,
    # unless it was the (already-empty) block list.
    if content is None and not isinstance(raw, (list, tuple)):
        return _as_text(raw)
    return ""


def _text_from_blocks(blocks: Any) -> str:
    """Join the text of any text content blocks; note non-text ones by type."""
    if not blocks:
        return ""
    parts: list = []
    for block in blocks:
        text = _spec_field(block, "text")
        if text is not None:
            parts.append(str(text))
        else:
            block_type = _spec_field(block, "type") or "non-text"
            parts.append(f"[{block_type} content]")
    return "\n".join(part for part in parts if part)


def _as_text(value: Any) -> str:
    """A string for the model to read: pass strings through, JSON the rest."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _short_args(arguments: dict, limit: int = 80) -> str:
    """A compact, truncated rendering of arguments for a preview line."""
    try:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(arguments)
    return text if len(text) <= limit else text[:limit] + "..."
