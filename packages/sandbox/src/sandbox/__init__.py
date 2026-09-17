"""Small Docker-backed command sandbox shared by both agent tracks.

The workspace is mounted at /workspace in every container. The default image is
the project-owned build from infra/sandbox-images.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_IMAGE = "operating-agent-sandbox:py312"
DEFAULT_MEMORY = "512m"
DEFAULT_CPUS = "1.0"
#: Kill stray processes and cap how many a workload may fork. A workload that
#: forks faster than this cannot fork-bomb the host; execs beyond the cap
#: fail loudly inside the container instead.
DEFAULT_PIDS_LIMIT = "256"
#: Writable scratch inside an otherwise read-only root filesystem. ``noexec``
#: keeps it from becoming a staging ground for dropped binaries.
DEFAULT_TMPFS = ("/tmp:rw,noexec,nosuid,size=64m",)
log = logging.getLogger(__name__)


def _threaded_subprocess_required() -> bool:
    """Return whether the active event loop cannot spawn subprocesses.

    Windows' selector loop is required by psycopg, but it deliberately does not
    implement asyncio subprocess transports.  Docker commands are still safe
    to run from a worker thread using ``subprocess.run`` in that configuration.
    """
    if sys.platform != "win32":
        return False
    try:
        return isinstance(asyncio.get_running_loop(), asyncio.SelectorEventLoop)
    except RuntimeError:
        return False


@dataclass(slots=True)
class CommandOutput:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def combined(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


class ContainerRunner:
    def __init__(
        self,
        container_id: str,
        image: str,
        on_timeout_destroy: Any = None,
    ) -> None:
        self.container_id = container_id
        self.image = image
        # Called (awaited) when a command times out so the pool can destroy
        # the container. Set by ContainerPool; None keeps the old behaviour
        # of only stopping the host-side client (tests use this to observe).
        self._on_timeout_destroy = on_timeout_destroy

    async def run(self, command: str | list[str], timeout: float) -> CommandOutput:
        args = ["docker", "exec", self.container_id]
        if isinstance(command, str):
            args.extend(["sh", "-lc", command])
        else:
            args.extend(str(part) for part in command)
        if _threaded_subprocess_required():
            try:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                if self._on_timeout_destroy is not None:
                    try:
                        await self._on_timeout_destroy()
                    except Exception as exc:  # noqa: BLE001 - destroy is best effort
                        log.debug("could not destroy timed-out container: %s", exc)
                return CommandOutput(-1, timed_out=True)
            return CommandOutput(
                completed.returncode or 0,
                completed.stdout.decode(errors="replace"),
                completed.stderr.decode(errors="replace"),
            )
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            # Killing the host-side `docker exec` client does NOT stop the
            # workload inside the container — the shell and everything it
            # spawned would keep running, orphaned, indefinitely. Destroy the
            # session container instead: PID-namespace teardown guarantees the
            # whole process tree dies, children included. The pool recreates
            # the container on the next command; the workspace bind-mount
            # survives, so only in-container scratch state is lost.
            process.kill()
            try:
                await asyncio.wait_for(process.communicate(), 10)
            except Exception:  # noqa: BLE001 - reaping is best effort
                log.debug("could not reap timed-out docker exec client")
            if self._on_timeout_destroy is not None:
                try:
                    await self._on_timeout_destroy()
                except Exception as exc:  # noqa: BLE001 - destroy is best effort
                    log.debug("could not destroy timed-out container: %s", exc)
            return CommandOutput(-1, timed_out=True)
        return CommandOutput(
            process.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )


class HostCommandRunner:
    """Run a sandbox command on the host because the container path is down.

    This is the degradation path, never the primary one: it executes the same
    command the container would have run (``tool.sandbox_command``) directly on
    the host, confined to the session's workspace directory, with the user's own
    privileges and no network or filesystem isolation. It is only selected when
    ``ContainerPool`` is built with ``fallback=True`` (the API default) and the
    session runner cannot be created; callers surface that fact in the result
    and the status line.
    """

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace

    async def run(self, command: str | list[str], timeout: float) -> CommandOutput:
        if isinstance(command, str):
            args: list[str] = (
                ["cmd", "/c", command] if sys.platform == "win32" else ["sh", "-lc", command]
            )
        else:
            args = [str(part) for part in command]
        cwd = self.workspace
        if _threaded_subprocess_required():
            try:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    args,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return CommandOutput(-1, stderr=str(exc), timed_out=isinstance(exc, subprocess.TimeoutExpired))
            return CommandOutput(
                completed.returncode or 0,
                completed.stdout.decode(errors="replace"),
                completed.stderr.decode(errors="replace"),
            )
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            return CommandOutput(-1, stderr=str(exc), timed_out=False)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            try:
                process.kill()
                await asyncio.wait_for(process.communicate(), 10)
            except Exception:  # noqa: BLE001 - reaping is best effort
                log.debug("could not reap timed-out host command")
            return CommandOutput(-1, timed_out=True)
        return CommandOutput(
            process.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )


class ContainerPool:
    """Create one disposable container per logical session.

    Hardening flags are constructor policy so deployments can tighten or
    relax them without touching call sites. The defaults assume an
    untrusted workload: no capabilities, no new privileges, capped PIDs,
    a read-only root filesystem with a small noexec ``/tmp``, and no
    network. ``user`` is empty by default — the image runs as root because
    the bind-mounted workspace must stay writable on hosts with different
    UIDs; pass an explicit ``--user`` value when the image bakes in a
    matching non-root user.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        memory: str = DEFAULT_MEMORY,
        cpus: str = DEFAULT_CPUS,
        network: bool = False,
        pids_limit: str = DEFAULT_PIDS_LIMIT,
        readonly: bool = True,
        tmpfs: tuple[str, ...] = DEFAULT_TMPFS,
        cap_drop: tuple[str, ...] = ("ALL",),
        no_new_privileges: bool = True,
        user: str = "",
        fallback: bool = True,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.pids_limit = pids_limit
        self.readonly = readonly
        self.tmpfs = tuple(tmpfs)
        self.cap_drop = tuple(cap_drop)
        self.no_new_privileges = no_new_privileges
        self.user = user
        #: When True (the default), a command is run on the host if the session
        #: container cannot be created because Docker or its image is missing.
        #: Set False to fail closed instead (``AGENT_SANDBOX_FALLBACK=error``).
        self.fallback = fallback
        self.reason = ""
        self._runners: dict[str, ContainerRunner] = {}
        self._runner_meta: dict[str, tuple[str, str]] = {}
        self._lock = asyncio.Lock()
        self._available: bool | None = None

    async def available(self) -> bool:
        if shutil.which("docker") is None:
            self.reason = "Docker CLI is not installed"
            self._available = False
            return False
        if _threaded_subprocess_required():
            try:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.reason = str(exc) or "Docker daemon is unavailable"
                self._available = False
                return False
            if completed.returncode != 0:
                self.reason = completed.stderr.decode(errors="replace").strip() or "Docker daemon is unavailable"
                self._available = False
                return False
            self.reason = ""
            self._available = True
            return True
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "info", "--format", "{{.ServerVersion}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        except (OSError, TimeoutError) as exc:
            if process is not None:
                try:
                    process.kill()
                    await asyncio.wait_for(process.communicate(), 10)
                except (OSError, TimeoutError, ProcessLookupError):
                    pass
            self.reason = str(exc) or "Docker daemon is unavailable"
            self._available = False
            return False
        if process.returncode != 0:
            self.reason = stderr.decode(errors="replace").strip() or "Docker daemon is unavailable"
            self._available = False
            return False
        self.reason = ""
        self._available = True
        return True

    async def probe(self) -> bool:
        """Check whether Docker can host sandbox containers.

        This is the startup-facing name used by the native CLI. A probe never
        raises for an unavailable Docker installation; callers can report the
        failure or fail closed without attempting host execution.
        """
        if not await self.available():
            return False
        if _threaded_subprocess_required():
            try:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "image", "inspect", self.image],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.reason = str(exc) or "could not inspect sandbox image"
                self._available = False
                return False
            if completed.returncode != 0:
                self.reason = completed.stderr.decode(errors="replace").strip() or f"sandbox image {self.image!r} is not available"
                self._available = False
                return False
            return True
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "image", "inspect", self.image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        except (OSError, TimeoutError) as exc:
            if process is not None:
                try:
                    process.kill()
                    await asyncio.wait_for(process.communicate(), 10)
                except (OSError, TimeoutError, ProcessLookupError):
                    pass
            self.reason = str(exc) or "could not inspect sandbox image"
            self._available = False
            return False
        if process.returncode != 0:
            self.reason = stderr.decode(errors="replace").strip() or (
                f"sandbox image {self.image!r} is not available"
            )
            self._available = False
            return False
        return True

    def status_line(self) -> str:
        """Return a concise, user-facing description of the current mode."""
        if self._available is True:
            return f"sandbox: on - Docker container ({self.image})"
        if self.degraded:
            return f"sandbox: degraded - {self.reason}; terminal commands run on the host"
        if self.reason:
            return f"sandbox: off - {self.reason}"
        return "sandbox: off - Docker availability has not been checked"

    @property
    def degraded(self) -> bool:
        """True when the container cannot be reached and host fallback is on.

        Callers use this to label where a command actually ran, instead of
        claiming an unavailable container.
        """
        return bool(self.fallback) and self._available is False and bool(self.reason)

    async def get(self, session_id: str, workspace: str) -> ContainerRunner | None:
        try:
            root = Path(workspace).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            # A bad workspace argument says nothing about Docker: report it
            # without touching availability, like the branch below.
            self.reason = f"invalid workspace: {exc}"
            return None
        if not root.is_dir():
            self.reason = f"workspace does not exist: {root}"
            return None

        key = f"{session_id}:{root}"
        async with self._lock:
            existing = self._runners.get(key)
            if existing is not None:
                return existing
            # ``available()`` only probes the Docker daemon. A successful daemon
            # probe must not overwrite a previous image-readiness failure before
            # docker run is attempted; probe() checks both conditions together.
            if not await self.probe():
                return None
            name = f"operating-agent-{uuid4().hex[:12]}"
            args = self._run_args(name, root)
            if _threaded_subprocess_required():
                try:
                    completed = await asyncio.to_thread(
                        subprocess.run,
                        args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=30,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    await self._stop(name)
                    self.reason = str(exc) or "could not start Docker container"
                    return None
                if completed.returncode != 0:
                    self.reason = completed.stderr.decode(errors="replace").strip() or "could not start Docker container"
                    return None
                container_id = completed.stdout.decode(errors="replace").strip()
                if not container_id:
                    self.reason = "Docker returned an empty container id"
                    await self._stop(name)
                    return None
                runner = ContainerRunner(
                    container_id,
                    self.image,
                    on_timeout_destroy=lambda: self._destroy_runner(key, container_id),
                )
                self._runners[key] = runner
                self._runner_meta[key] = (session_id, str(root))
                log.info("created sandbox container=%s session=%s workspace=%s", container_id, session_id, root)
                return runner
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
            except (OSError, TimeoutError) as exc:
                # A half-created container would linger nameless-but-running;
                # best-effort kill it before tearing it down by name.
                if process is not None:
                    try:
                        process.kill()
                        await asyncio.wait_for(process.communicate(), 10)
                    except (OSError, TimeoutError, ProcessLookupError):
                        pass
                await self._stop(name)
                self.reason = str(exc) or "could not start Docker container"
                return None
            if process.returncode != 0:
                self.reason = stderr.decode(errors="replace").strip() or "could not start Docker container"
                return None
            container_id = stdout.decode(errors="replace").strip()
            if not container_id:
                self.reason = "Docker returned an empty container id"
                await self._stop(name)
                return None
            runner = ContainerRunner(
                container_id,
                self.image,
                on_timeout_destroy=lambda: self._destroy_runner(key, container_id),
            )
            self._runners[key] = runner
            self._runner_meta[key] = (session_id, str(root))
            log.info("created sandbox container=%s session=%s workspace=%s", container_id, session_id, root)
            return runner

    def list_containers(self) -> list[dict[str, str]]:
        """Return the live containers owned by this pool for UI/API inspection."""
        rows: list[dict[str, str]] = []
        for key, runner in self._runners.items():
            session_id, workspace = self._runner_meta.get(
                key, key.split(":", 1) if ":" in key else (key, "")
            )
            rows.append(
                {
                    "session_id": session_id,
                    "workspace": workspace,
                    "container_id": runner.container_id,
                    "image": runner.image,
                    "status": "running",
                }
            )
        return rows

    async def destroy_session(self, session_id: str) -> int:
        """Stop all containers owned by this pool for one agent session."""
        async with self._lock:
            matches = [
                (key, runner)
                for key, runner in self._runners.items()
                if self._runner_meta.get(key, (key.split(":", 1)[0], ""))[0] == session_id
            ]
            for key, _runner in matches:
                self._runners.pop(key, None)
                self._runner_meta.pop(key, None)
        await asyncio.gather(
            *(self._stop(runner.container_id) for _key, runner in matches),
            return_exceptions=True,
        )
        return len(matches)

    def _run_args(self, name: str, root: Path) -> list[str]:
        """The ``docker run`` argv for one session container. Pure function of
        policy, so security tests can assert on it without Docker."""
        args = [
            "docker", "run", "-d", "--rm", "--init", "--name", name,
            "--workdir", "/workspace",
            "--memory", self.memory, "--cpus", self.cpus,
            "--pids-limit", self.pids_limit,
        ]
        for cap in self.cap_drop:
            args.extend(["--cap-drop", cap])
        if self.no_new_privileges:
            args.extend(["--security-opt", "no-new-privileges"])
        if self.readonly:
            # The workspace stays writable through its bind mount; everything
            # else in the root filesystem is read-only.
            args.append("--read-only")
            for spec in self.tmpfs:
                args.extend(["--tmpfs", spec])
        if self.user:
            args.extend(["--user", self.user])
        args.extend(["--mount", f"type=bind,source={root},target=/workspace"])
        if not self.network:
            args.extend(["--network", "none"])
        args.extend([self.image, "sleep", "infinity"])
        return args

    async def _destroy_runner(self, key: str, container_id: str) -> None:
        """Evict a runner and destroy its container (timeout path).

        Eviction is conditional on the id still matching: a recreated runner
        for the same key must never be removed by a stale timeout.
        """
        async with self._lock:
            current = self._runners.get(key)
            if current is None or current.container_id != container_id:
                return
            del self._runners[key]
            self._runner_meta.pop(key, None)
        await self._stop(container_id)

    async def run(
        self,
        tool: object,
        arguments: dict,
        context: object,
        timeout: float,
    ) -> object | None:
        """Run a native SANDBOX tool in the container for its session workspace."""
        command_builder = getattr(tool, "sandbox_command", None)
        if not callable(command_builder):
            return None
        command = command_builder(arguments)
        if isinstance(command, str):
            sandbox_command: str | list[str] = command
        elif isinstance(command, list) and all(
            isinstance(part, str) for part in command
        ):
            sandbox_command = command
        else:
            return None
        try:
            from agent_native.tools.base import ToolResult
        except ImportError:
            # Without the agent_native result type this pool cannot answer;
            # treat that as "not implemented" so the manager runs natively.
            return None
        session = getattr(context, "session", None)
        session_id = str(getattr(session, "id", "native"))
        workspace = str(getattr(session, "working_directory", ".") or ".")
        runner = await self.get(session_id, workspace)
        if runner is None:
            reason = self.reason or "Docker sandbox is unavailable"
            if self.fallback and workspace and Path(workspace).expanduser().is_dir():
                # Degraded but usable: run the same command on the host, in the
                # session's workspace. Fallback must never apply to a missing
                # workspace - that is a bad argument, not a missing Docker.
                host_result = await HostCommandRunner(
                    str(Path(workspace).expanduser().resolve())
                ).run(sandbox_command, timeout=timeout)
                log.warning("sandbox host fallback: Docker unavailable (%s)", reason)
                note = " [host fallback: ran on the host - Docker sandbox unavailable]"
                output = host_result.combined()
                if host_result.timed_out:
                    return ToolResult(False, error="command timed out in sandbox")
                if host_result.exit_code != 0:
                    return ToolResult(
                        False,
                        output=output,
                        error=f"command failed in sandbox (exit {host_result.exit_code})",
                    )
                return ToolResult(True, output=f"{output}{note}" if output else note)
            return ToolResult(False, error=f"sandbox unavailable: {reason}")
        result = await runner.run(sandbox_command, timeout=timeout)
        output = result.combined()
        if result.timed_out:
            return ToolResult(False, error="command timed out in sandbox")
        if result.exit_code != 0:
            return ToolResult(
                False,
                output=output,
                error=f"command failed in sandbox (exit {result.exit_code})",
            )
        return ToolResult(True, output=output or "(no output)")

    async def stop_all(self) -> None:
        async with self._lock:
            runners, self._runners = self._runners, {}
            self._runner_meta.clear()
        await asyncio.gather(
            *(self._stop(runner.container_id) for runner in runners.values()),
            return_exceptions=True,
        )

    async def close(self) -> None:
        """Release all containers owned by this pool."""
        await self.stop_all()

    async def _stop(self, container_id: str) -> None:
        if _threaded_subprocess_required():
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "rm", "-f", container_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.debug("could not remove sandbox container=%s: %s", container_id, exc)
            return
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.communicate(), 10)
        except (OSError, TimeoutError) as exc:
            log.debug("could not remove sandbox container=%s: %s", container_id, exc)


ContainerSandbox = ContainerPool


def main() -> None:
    print("sandbox package: Docker-backed command isolation")


__all__ = [
    "DEFAULT_CPUS",
    "DEFAULT_IMAGE",
    "DEFAULT_MEMORY",
    "DEFAULT_PIDS_LIMIT",
    "DEFAULT_TMPFS",
    "CommandOutput",
    "ContainerPool",
    "ContainerRunner",
    "ContainerSandbox",
    "HostCommandRunner",
]
