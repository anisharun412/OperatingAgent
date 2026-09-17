"""API settings and the per-track ``AgentConfig`` builder.

``ApiSettings`` is a frozen, env-sourced snapshot (mirroring the ``from_env``
pattern in ``observability.settings``). ``build_agent_config`` assembles an
``AgentConfig`` for a track using only dataclass construction — no model is
built and no network is touched — so it is safe to call from unit tests.
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from common.config import (
    AgentConfig,
    BehaviourConfig,
    CheckpointConfig,
    ExecutionConfig,
    LLMConfig,
    MetadataConfig,
    PromptConfig,
    SandboxConfig,
    ToolPermissionConfig,
    TracingConfig,
)
from common.enums import AgentTrack, RiskLevel
from observability import LangfuseSettings

DEFAULT_CORS_ORIGINS = (
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
)
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "testserver")


def _default_data_dir() -> Path:
    """Return the per-user application data directory for this platform."""
    if os.name == "nt":
        root = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
        if root:
            return Path(root) / "OperatingAgent"
    elif os.name == "posix":
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "OperatingAgent"
        root = os.getenv("XDG_DATA_HOME")
        if root:
            return Path(root) / "OperatingAgent"
    return Path.home() / ".operating-agent"


DEFAULT_DATA_DIR = _default_data_dir()
DEFAULT_SQLITE_DATABASE_PATH = DEFAULT_DATA_DIR / "operating-agent.db"


def _default_prompt_dir() -> Path:
    """Resolve LangGraph prompts via package resources when installed."""
    try:
        from importlib.resources import files

        pkg_prompts = files("agent_langgraph") / "prompts"
        # files() returns a Traversable; check existence without requiring as_file
        if pkg_prompts.is_dir():
            return Path(str(pkg_prompts))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return Path(__file__).resolve().parents[3] / "agent-langgraph" / "prompts"
    return Path(__file__).resolve().parents[3] / "agent-langgraph" / "prompts"


DEFAULT_PROMPT_DIR = _default_prompt_dir()


def _split_csv(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    return values or default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    return None if raw in (None, "") else int(raw)


@dataclass(slots=True, frozen=True)
class ApiSettings:
    """Resolved API configuration for one process."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    database_url: str | None = field(default=None, repr=False)
    repository_backend: str = "memory"
    # Production-safe default: an unavailable configured Postgres store should
    # fail startup unless a desktop fallback is explicitly selected.
    repository_fallback: str = "error"
    repository_connect_timeout_seconds: float = 5.0
    sqlite_database_path: str = str(DEFAULT_SQLITE_DATABASE_PATH)

    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS

    default_track: AgentTrack = AgentTrack.LANGGRAPH
    approval_threshold: RiskLevel = RiskLevel.REVIEW

    llm_provider: str = "ollama"
    llm_model: str = "llama3.1"
    llm_base_url: str | None = None
    llm_timeout_seconds: int = 60
    llm_temperature: float = 0.0
    llm_max_tokens: int | None = None

    llm_max_retries: int = 7
    llm_top_p: float = 1.0
    llm_input_price_per_million: float = 0.0
    llm_output_price_per_million: float = 0.0

    execution_max_iterations: int = 20
    execution_timeout_seconds: int = 300
    execution_retry_attempts: int = 2
    execution_max_replans: int = 3
    execution_stream: bool = True
    execution_enable_checkpoints: bool = True
    execution_enable_interrupts: bool = True

    sandbox_enabled: bool = True
    sandbox_workspace: str = "./workspace"
    sandbox_image: str = ""
    #: "host" runs commands on the host when Docker is unavailable; "error"
    #: keeps the fail-closed behavior.
    sandbox_fallback: str = "host"

    permission_file_system: bool = True
    permission_terminal: bool = True
    permission_git: bool = True
    permission_search: bool = True
    permission_knowledge: bool = True
    permission_memory: bool = True

    checkpoint_backend: str = "auto"
    checkpoint_namespace: str = "default"

    require_verification: bool = False
    require_human_approval: bool = True

    prompt_dir: str = str(DEFAULT_PROMPT_DIR)
    planner_prompt: str | None = None
    verifier_prompt: str | None = None
    responder_prompt: str | None = None

    mcp_gateway_command: str = sys.executable
    mcp_gateway_args: tuple[str, ...] = ("-m", "gateway_server")
    tracing_enabled: bool = False

    @classmethod
    def from_env(cls) -> ApiSettings:
        database_url = os.getenv("DATABASE_URL") or None
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        provider_prefix = provider.upper()
        model = (
            os.getenv("LLM_MODEL")
            or os.getenv(f"{provider_prefix}_MODEL")
            or "llama3.1"
        )
        base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv(f"{provider_prefix}_BASE_URL")
            or None
        )
        backend = os.getenv(
            "API_REPOSITORY_BACKEND", "postgres" if database_url else "memory"
        ).lower()
        data_dir = Path(os.getenv("OPERATING_AGENT_DATA_DIR") or DEFAULT_DATA_DIR)
        sqlite_database_path = os.getenv("SQLITE_DATABASE_PATH") or str(
            data_dir / "operating-agent.db"
        )
        return cls(
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("API_LOG_LEVEL", "info"),
            database_url=database_url,
            repository_backend=backend,
            repository_fallback=os.getenv("API_REPOSITORY_FALLBACK", "error").strip().lower(),
            repository_connect_timeout_seconds=_env_float(
                "API_REPOSITORY_CONNECT_TIMEOUT_SECONDS", 5.0
            ),
            sqlite_database_path=sqlite_database_path,
            cors_origins=_split_csv(
                os.getenv("API_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)),
                DEFAULT_CORS_ORIGINS,
            ),
            allowed_hosts=_split_csv(
                os.getenv("API_ALLOWED_HOSTS", ",".join(DEFAULT_ALLOWED_HOSTS)),
                DEFAULT_ALLOWED_HOSTS,
            ),
            default_track=AgentTrack(os.getenv("API_DEFAULT_TRACK", "langgraph")),
            approval_threshold=RiskLevel(os.getenv("API_APPROVAL_THRESHOLD", "review")),
            llm_provider=provider,
            llm_model=model,
            llm_base_url=base_url,
            llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 60),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
            llm_max_tokens=_env_optional_int("LLM_MAX_TOKENS"),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 7),
            llm_top_p=_env_float("LLM_TOP_P", 1.0),
            llm_input_price_per_million=_env_float("LLM_INPUT_PRICE_PER_MILLION", 0.0),
            llm_output_price_per_million=_env_float("LLM_OUTPUT_PRICE_PER_MILLION", 0.0),
            execution_max_iterations=_env_int("AGENT_MAX_ITERATIONS", 20),
            execution_timeout_seconds=_env_int(
                "AGENT_EXECUTION_TIMEOUT_SECONDS", 300
            ),
            execution_retry_attempts=_env_int("AGENT_RETRY_ATTEMPTS", 2),
            execution_max_replans=_env_int("AGENT_MAX_REPLANS", 3),
            execution_stream=_env_bool("AGENT_STREAM", True),
            execution_enable_checkpoints=_env_bool(
                "AGENT_ENABLE_CHECKPOINTS", True
            ),
            execution_enable_interrupts=_env_bool("AGENT_ENABLE_INTERRUPTS", True),
            sandbox_enabled=_env_bool("AGENT_SANDBOX_ENABLED", True),
            sandbox_workspace=os.getenv("AGENT_WORKSPACE", "./workspace"),
            sandbox_image=os.getenv("AGENT_SANDBOX_IMAGE", ""),
            sandbox_fallback=(
                "error"
                if (os.getenv("AGENT_SANDBOX_FALLBACK") or "host").strip().lower()
                in {"error", "off", "false", "no", "0"}
                else "host"
            ),
            permission_file_system=_env_bool("AGENT_PERMISSION_FILE_SYSTEM", True),
            permission_terminal=_env_bool("AGENT_PERMISSION_TERMINAL", True),
            permission_git=_env_bool("AGENT_PERMISSION_GIT", True),
            permission_search=_env_bool("AGENT_PERMISSION_SEARCH", True),
            permission_knowledge=_env_bool("AGENT_PERMISSION_KNOWLEDGE", True),
            permission_memory=_env_bool("AGENT_PERMISSION_MEMORY", True),
            checkpoint_backend=os.getenv("AGENT_CHECKPOINT_BACKEND", "auto").lower(),
            checkpoint_namespace=os.getenv("AGENT_CHECKPOINT_NAMESPACE", "default"),
            require_verification=_env_bool("AGENT_REQUIRE_VERIFICATION", False),
            require_human_approval=_env_bool("AGENT_REQUIRE_HUMAN_APPROVAL", True),
            prompt_dir=os.getenv("AGENT_PROMPT_DIR") or str(DEFAULT_PROMPT_DIR),
            planner_prompt=os.getenv("AGENT_PLANNER_PROMPT") or None,
            verifier_prompt=os.getenv("AGENT_VERIFIER_PROMPT") or None,
            responder_prompt=os.getenv("AGENT_RESPONDER_PROMPT") or None,
            mcp_gateway_command=os.getenv("MCP_GATEWAY_COMMAND") or sys.executable,
            mcp_gateway_args=tuple(
                shlex.split(os.getenv("MCP_GATEWAY_ARGS", "-m gateway_server"))
            ),
            tracing_enabled=LangfuseSettings.from_env().enabled,
        )

    def build_agent_config(self, track: AgentTrack | None = None) -> AgentConfig:
        """Assemble an ``AgentConfig`` for ``track``.

        The api key is read from ``{PROVIDER}_API_KEY`` at call time and defaults
        to an empty string (hermetic). The ``auto`` checkpoint backend selects
        Postgres when a ``DATABASE_URL`` is configured, SQLite when the
        repository backend is SQLite, and memory otherwise. Tracing follows the
        shared Langfuse enablement rule (both keys present).
        """
        provider = self.llm_provider
        api_key = os.getenv(f"{provider.upper()}_API_KEY", "")
        prompts = Path(self.prompt_dir)
        checkpoint_backend = self.checkpoint_backend
        if checkpoint_backend == "auto":
            if self.repository_backend in {
                "sqlite",
                "file",
                "file-based",
                "file_based",
            }:
                checkpoint_backend = "sqlite"
            elif self.database_url:
                checkpoint_backend = "postgres"
            else:
                checkpoint_backend = "memory"
        checkpoint_connection = self.database_url
        if checkpoint_backend == "sqlite":
            checkpoint_connection = self.sqlite_database_path
        resolved_track = track or self.default_track
        return AgentConfig(
            llm=LLMConfig(
                provider=provider,
                model=self.llm_model,
                api_key=api_key,
                timeout_seconds=self.llm_timeout_seconds,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                max_retries=self.llm_max_retries,
                top_p=self.llm_top_p,
                input_price_per_million=self.llm_input_price_per_million,
                output_price_per_million=self.llm_output_price_per_million,
                base_url=self.llm_base_url,
            ),
            execution=ExecutionConfig(
                max_iterations=self.execution_max_iterations,
                timeout_seconds=self.execution_timeout_seconds,
                retry_attempts=self.execution_retry_attempts,
                max_replans=self.execution_max_replans,
                stream=self.execution_stream,
                enable_checkpoints=self.execution_enable_checkpoints,
                enable_interrupts=self.execution_enable_interrupts,
            ),
            sandbox=SandboxConfig(
                enabled=self.sandbox_enabled,
                workspace=Path(self.sandbox_workspace),
                image=self.sandbox_image,
                fallback=(self.sandbox_fallback == "host"),
            ),
            permissions=ToolPermissionConfig(
                file_system=self.permission_file_system,
                terminal=self.permission_terminal,
                git=self.permission_git,
                search=self.permission_search,
                knowledge=self.permission_knowledge,
                memory=self.permission_memory,
            ),
            checkpoint=CheckpointConfig(
                backend=checkpoint_backend,
                connection_string=checkpoint_connection,
                namespace=self.checkpoint_namespace,
            ),
            tracing=TracingConfig(enabled=self.tracing_enabled),
            behaviour=BehaviourConfig(
                require_verification=self.require_verification,
                require_human_approval=self.require_human_approval,
                risk_threshold=self.approval_threshold.value,
            ),
            prompts=PromptConfig(
                planner_prompt=(
                    Path(self.planner_prompt)
                    if self.planner_prompt
                    else prompts / "planner.txt"
                ),
                verifier_prompt=(
                    Path(self.verifier_prompt)
                    if self.verifier_prompt
                    else prompts / "verifier.txt"
                ),
                responder_prompt=(
                    Path(self.responder_prompt)
                    if self.responder_prompt
                    else prompts / "responder.txt"
                ),
            ),
            metadata=MetadataConfig(tags={"track": resolved_track.value}),
        )
