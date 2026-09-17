import logging
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

from common.config import SandboxConfig, ToolPermissionConfig
from common.interfaces import IMCPClient
from common.tools import ToolCallRequest, ToolCallResult, ToolInfo
from sandbox import DEFAULT_IMAGE, ContainerPool, HostCommandRunner

log = logging.getLogger(__name__)


class ToolRegistry:
    """
    Maps agent tool requests to MCP tools.

    The ExecutorNode should depend on this abstraction,
    not directly on individual MCP servers.
    """

    _CATEGORY_FIELDS: ClassVar[dict[str, str]] = {
        "filesystem": "file_system",
        "terminal": "terminal",
        "git": "git",
        "search": "search",
        "knowledge": "knowledge",
        "memory": "memory",
    }
    _DIRECT_TOOL_CATEGORIES: ClassVar[dict[str, str]] = {
        "read_file": "filesystem",
        "write_file": "filesystem",
        "delete_file": "filesystem",
        "copy_file": "filesystem",
        "move_file": "filesystem",
        "rename_file": "filesystem",
        "list_directory": "filesystem",
        "create_directory": "filesystem",
        "delete_directory": "filesystem",
        "exists": "filesystem",
        "metadata": "filesystem",
        "search_files": "filesystem",
        "watch_directory": "filesystem",
        "run_command": "terminal",
        "list_processes": "terminal",
        "git_status": "git",
        "list_branches": "git",
        "git_log": "git",
        "diff": "git",
        "index_documents": "search",
        "search_documents": "search",
        "list_indices": "search",
    }
    _PATH_KEYS = ("path", "directory", "source", "destination", "root")
    # A few MCP servers use these names instead of the conventional ``path``
    # field.  They are kept as an explicit policy because MCP tool schemas are
    # optional and an unknown filesystem-capable tool must fail closed.
    _PATH_FIELD_HINTS = frozenset(
        {
            "cwd",
            "file",
            "filename",
            "filepath",
            "file_path",
            "working_directory",
            "workdir",
            "directory_path",
        }
    )
    _TOOL_PATH_FIELDS: ClassVar[dict[str, frozenset[str]]] = {}

    def __init__(
        self,
        mcp_adapter: IMCPClient,
        permissions: ToolPermissionConfig | None = None,
        sandbox: SandboxConfig | None = None,
        tool_path_fields: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        self._mcp = mcp_adapter
        self._permissions = permissions or ToolPermissionConfig()
        self._sandbox = sandbox or SandboxConfig()
        self._sandbox_pool: ContainerPool | None = None
        self._workspace_clients: dict[str, IMCPClient] = {}
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        self._tool_path_fields = {
            name: frozenset(field.lower() for field in fields)
            for name, fields in (tool_path_fields or {}).items()
        }
        self._tools_loaded = False

    async def list_tools(self) -> list[ToolInfo]:
        tools = await self._mcp.list_tools()
        self._tool_schemas = {
            tool.name: tool.schema.input_schema
            for tool in tools
            if isinstance(tool.schema.input_schema, dict)
        }
        self._tools_loaded = True
        return [tool for tool in tools if self._is_allowed(tool.name)]

    async def call(
        self,
        request: ToolCallRequest,
        *,
        workspace: str | None = None,
    ) -> ToolCallResult:
        if not self._is_allowed(request.tool_name):
            category = self._category(request.tool_name) or "unknown"
            return ToolCallResult(
                success=False,
                output=None,
                error=f"tool category '{category}' is disabled by configuration",
            )
        if self._sandbox.enabled and not self._tools_loaded:
            # Tool schemas are the authoritative source for non-standard path
            # fields on uncategorized MCP tools.  Discover them lazily so a
            # direct call is protected even when list_tools was not called by
            # the planner first.
            try:
                await self.list_tools()
            except Exception as exc:  # noqa: BLE001 - external MCP discovery boundary
                # A failed discovery must not make a safe, already-known call
                # unusable; the explicit policy and conservative field hints
                # still apply below.
                log.warning(
                    "could not load schema for MCP tool %s before validation: %s",
                    request.tool_name,
                    exc,
                )
        # Validate the same expanded path representation that is sent to an
        # MCP client.  This prevents values such as ``~/.ssh/id_rsa`` from
        # being checked as a workspace-relative path first.
        try:
            forwarded_request = self._request_for_workspace(request, workspace)
        except (OSError, RuntimeError, ValueError) as exc:
            return ToolCallResult(
                success=False,
                output=None,
                error=f"sandbox workspace is invalid: {exc}",
            )
        sandbox_error = self._sandbox_error(forwarded_request, workspace)
        if sandbox_error:
            return ToolCallResult(success=False, output=None, error=sandbox_error)
        sandbox_result = await self._call_in_sandbox(request, workspace)
        if sandbox_result is not None:
            return sandbox_result
        client = self._client_for_workspace(workspace)
        return await client.call_tool(forwarded_request)

    async def call_by_name(
        self,
        tool_name: str,
        arguments: dict,
        *,
        workspace: str | None = None,
    ) -> ToolCallResult:

        request = ToolCallRequest(
            tool_name=tool_name,
            arguments=arguments,
        )

        return await self.call(request, workspace=workspace)

    async def aclose(self) -> None:
        """Close resources owned by the underlying MCP client."""
        if self._sandbox_pool is not None:
            await self._sandbox_pool.stop_all()
        for client in self._workspace_clients.values():
            close_workspace_client = getattr(client, "aclose", None)
            if close_workspace_client is not None:
                await close_workspace_client()
        self._workspace_clients.clear()
        close = getattr(self._mcp, "aclose", None)
        if close is not None:
            await close()

    def _client_for_workspace(self, workspace: str | None) -> IMCPClient:
        if not workspace:
            return self._mcp
        root = str(Path(workspace).expanduser().resolve())
        existing = self._workspace_clients.get(root)
        if existing is not None:
            return existing
        factory = getattr(self._mcp, "for_workspace", None)
        if not callable(factory):
            return self._mcp
        # ``for_workspace`` is an optional extension on IMCPClient.  Keep the
        # base protocol small and validate the dynamic result at this boundary.
        client = cast(IMCPClient, factory(root))
        self._workspace_clients[root] = client
        return client

    @classmethod
    def _category(cls, tool_name: str) -> str | None:
        prefix, separator, _rest = tool_name.partition("_")
        if separator and prefix in cls._CATEGORY_FIELDS:
            return prefix
        return cls._DIRECT_TOOL_CATEGORIES.get(tool_name)

    def _is_allowed(self, tool_name: str) -> bool:
        category = self._category(tool_name)
        if category is None:
            return True
        return bool(getattr(self._permissions, self._CATEGORY_FIELDS[category]))

    def _sandbox_error(
        self, request: ToolCallRequest, workspace: str | None = None
    ) -> str | None:
        if not self._sandbox.enabled:
            return None
        category = self._category(request.tool_name)
        path_fields = self._path_fields_for_tool(request.tool_name)
        has_path_arguments = any(
            key.lower() in path_fields
            for key, _value in self._walk_arguments(request.arguments)
        )
        server_isolated = bool(
            getattr(self._mcp, "workspace_isolated", False)
            or getattr(self._mcp, "server_side_workspace_isolated", False)
        )
        # Unknown MCP tools are not trusted merely because they have no mapped
        # category. Their path arguments still cross the same workspace boundary
        # unless the adapter explicitly guarantees server-side isolation.
        unclassified_paths = category is None and has_path_arguments and not server_isolated
        if category is None and not unclassified_paths:
            return None
        # Mapped tools preserve their existing category rules: an explicitly
        # selected workspace is checked client-side, while terminal commands
        # still need a sandbox workspace to mount.
        configured_workspace = self._sandbox.workspace != Path("./workspace")
        if not unclassified_paths and workspace is None and (
            (category != "terminal" and not configured_workspace)
            or (category == "terminal" and not request.arguments.get("command"))
        ):
            return None

        try:
            root = (
                Path(workspace).expanduser().resolve()
                if workspace
                else self._sandbox.workspace.expanduser().resolve()
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return f"sandbox workspace is invalid: {exc}"
        if not root.is_dir():
            return f"sandbox workspace does not exist: {root}"
        for key, value in self._walk_arguments(request.arguments):
            if key.lower() not in path_fields:
                continue
            # ``expanduser`` must happen before deciding whether a path is
            # relative.  Otherwise ``~`` is incorrectly treated as a child of
            # the workspace and can pass the boundary check.
            candidate = Path(value).expanduser()
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (root / candidate).resolve()
            )
            if not resolved.is_relative_to(root):
                return f"filesystem path escapes configured workspace: {value}"
        return None

    @classmethod
    def _schema_path_fields(cls, schema: dict[str, Any]) -> set[str]:
        fields: set[str] = set()
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return fields
        for name, definition in properties.items():
            lowered = str(name).lower()
            if lowered in cls._PATH_KEYS or lowered in cls._PATH_FIELD_HINTS:
                fields.add(lowered)
            if isinstance(definition, dict):
                fmt = str(definition.get("format", "")).lower()
                description = str(definition.get("description", "")).lower()
                if fmt in {"path", "file-path", "filepath"} or "path" in description:
                    fields.add(lowered)
                items = definition.get("items")
                if isinstance(items, dict):
                    if str(items.get("format", "")).lower() in {
                        "path",
                        "file-path",
                        "filepath",
                    }:
                        fields.add(lowered)
                    # Object fields nested under array items (e.g. files[].target)
                    # are path-bearing too; the argument walker already descends
                    # into lists, so discovering the names is sufficient.
                    fields.update(cls._schema_path_fields(items))
                fields.update(cls._schema_path_fields(definition))
        return fields

    def _path_fields_for_tool(self, tool_name: str) -> frozenset[str]:
        fields = {field.lower() for field in self._PATH_KEYS}
        fields.update(field.lower() for field in self._PATH_FIELD_HINTS)
        policy_fields = set(self._TOOL_PATH_FIELDS.get(tool_name, ()))
        policy_fields.update(self._tool_path_fields.get(tool_name, ()))
        fields.update(
            field.lower() for field in policy_fields
        )
        schema = self._tool_schemas.get(tool_name)
        if schema is not None:
            fields.update(self._schema_path_fields(schema))
        return frozenset(fields)

    def _request_for_workspace(
        self, request: ToolCallRequest, workspace: str | None
    ) -> ToolCallRequest:
        """Make relative path arguments unambiguous for the MCP subprocess."""
        if not workspace:
            return request
        root = Path(workspace).expanduser().resolve()
        return ToolCallRequest(
            tool_name=request.tool_name,
            arguments=self._resolve_path_arguments(
                request.arguments,
                root,
                path_fields=self._path_fields_for_tool(request.tool_name),
            ),
        )

    @classmethod
    def _resolve_path_arguments(
        cls,
        value: Any,
        root: Path,
        key: str = "",
        path_fields: frozenset[str] | None = None,
    ) -> Any:
        fields = path_fields or frozenset(field.lower() for field in cls._PATH_KEYS)
        if isinstance(value, dict):
            return {
                child_key: cls._resolve_path_arguments(
                    child, root, str(child_key), path_fields=fields
                )
                for child_key, child in value.items()
            }
        if isinstance(value, list):
            return [
                cls._resolve_path_arguments(child, root, key, path_fields=fields)
                for child in value
            ]
        if isinstance(value, str) and key.lower() in fields:
            candidate = Path(value).expanduser()
            return str(candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve())
        return value

    async def _call_in_sandbox(
        self, request: ToolCallRequest, workspace: str | None = None
    ) -> ToolCallResult | None:
        if not self._sandbox.enabled or self._category(request.tool_name) != "terminal":
            return None
        command = request.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        if self._sandbox_pool is None:
            self._sandbox_pool = ContainerPool(
                image=self._sandbox.image or DEFAULT_IMAGE,
                network=False,
                fallback=getattr(self._sandbox, "fallback", True),
            )
        try:
            root = (
                Path(workspace).expanduser().resolve()
                if workspace
                else self._sandbox.workspace.expanduser().resolve()
            )
            runner = await self._sandbox_pool.get("langgraph", str(root))
        except (OSError, RuntimeError, ValueError) as exc:
            return ToolCallResult(
                success=False,
                output=None,
                error=f"sandbox workspace is invalid: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - optional sandbox boundary
            return ToolCallResult(
                success=False,
                output=None,
                error=f"sandbox unavailable: {exc}",
            )
        if runner is None:
            reason = self._sandbox_pool.reason or f"could not mount workspace {root}"
            if getattr(self._sandbox_pool, "fallback", False) and root.is_dir():
                # Degraded but usable: run the same command on the host, in the
                # session workspace, mirroring the native track's host fallback.
                log.warning("sandbox host fallback: Docker unavailable (%s)", reason)
                note = " [host fallback: ran on the host - Docker sandbox unavailable]"
                result = await HostCommandRunner(str(root)).run(command, timeout=300)
                output = result.combined()
                if result.timed_out:
                    return ToolCallResult(success=False, output=None, error="command timed out in sandbox")
                if result.exit_code != 0:
                    return ToolCallResult(success=False, output=output, error=f"command failed in sandbox (exit {result.exit_code})")
                return ToolCallResult(success=True, output=f"{output}{note}" if output else note)
            return ToolCallResult(
                success=False,
                output=None,
                error=f"sandbox unavailable: {reason}",
            )
        try:
            result = await runner.run(command, timeout=300)
        except Exception as exc:  # noqa: BLE001 - optional sandbox boundary
            return ToolCallResult(success=False, output=None, error=f"sandbox execution failed: {exc}")
        output = result.combined()
        if result.timed_out:
            return ToolCallResult(success=False, output=None, error="command timed out in sandbox")
        if result.exit_code != 0:
            return ToolCallResult(success=False, output=output, error=f"command failed in sandbox (exit {result.exit_code})")
        return ToolCallResult(success=True, output=output or "(no output)")

    @classmethod
    def _walk_arguments(cls, value: Any, key: str = ""):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                yield from cls._walk_arguments(child_value, str(child_key))
        elif isinstance(value, list):
            for child in value:
                yield from cls._walk_arguments(child, key)
        elif isinstance(value, str):
            yield key, value
