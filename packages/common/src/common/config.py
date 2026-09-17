from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============================================================
# LLM
# ============================================================

@dataclass(slots=True, frozen=True)
class LLMConfig:
    provider: str
    model: str
    # repr=False excludes the secret from the dataclass-generated __repr__.
    # Field access is unchanged; only the string representation omits it.
    api_key: str = field(repr=False)

    timeout_seconds: int = 60
    temperature: float = 0.0

    max_tokens: int | None = None

    # Provider-client retries with exponential backoff on transient errors
    # (429 rate limits, 5xx, connection resets). The native track retries in
    # its loop; the LangGraph track relies on this passthrough. Seven retries
    # accumulate roughly 30s of backoff, enough to outlast a typical
    # tokens-per-minute window on a free-tier key.
    max_retries: int = 7

    top_p: float = 1.0

    base_url: str | None = None

    # Optional per-million USD pricing for computing run cost from provider-
    # reported token usage. Zero (unset) means cost is unknown, not free: the
    # orchestrator reports None rather than a fabricated $0.
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("llm.provider must not be empty")
        if not self.model.strip():
            raise ValueError("llm.model must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("llm.timeout_seconds must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("llm.max_tokens must be positive when set")
        if self.max_retries < 0:
            raise ValueError("llm.max_retries must not be negative")
        if not 0 <= self.temperature <= 2:
            raise ValueError("llm.temperature must be between 0 and 2")
        if not 0 < self.top_p <= 1:
            raise ValueError("llm.top_p must be greater than 0 and at most 1")
        if self.input_price_per_million < 0 or self.output_price_per_million < 0:
            raise ValueError("llm prices must not be negative")


# ============================================================
# Execution
# ============================================================

@dataclass(slots=True, frozen=True)
class ExecutionConfig:

    max_iterations: int = 20

    timeout_seconds: int = 300

    retry_attempts: int = 2

    # Whole-run replan budget for a plan/replan architecture: how many times
    # a failed step may be handed back to the planner before the run gives
    # up and reports honestly. This is the graph counterpart of the loop
    # architecture's per-run turn budget (max_iterations); without it the
    # two recovery budgets cannot be compared on equal terms.
    max_replans: int = 3

    stream: bool = True

    enable_checkpoints: bool = True

    enable_interrupts: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("execution.max_iterations must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("execution.timeout_seconds must be positive")
        if self.retry_attempts < 0:
            raise ValueError("execution.retry_attempts must not be negative")
        if self.max_replans < 1:
            raise ValueError("execution.max_replans must be positive")


# ============================================================
# Sandbox
# ============================================================

@dataclass(slots=True, frozen=True)
class SandboxConfig:

    enabled: bool = True

    workspace: Path = Path("./workspace")

    image: str = ""

    #: True (default) runs terminal commands on the host when Docker is
    #: unavailable instead of failing closed. Mirrors the pool's fallback flag
    #: and ``AGENT_SANDBOX_FALLBACK`` in the API settings.
    fallback: bool = True


# ============================================================
# Tool Permissions
# ============================================================

@dataclass(slots=True, frozen=True)
class ToolPermissionConfig:

    file_system: bool = True

    terminal: bool = True

    git: bool = True

    search: bool = True

    knowledge: bool = True

    memory: bool = True


# ============================================================
# Checkpoint
# ============================================================

@dataclass(slots=True, frozen=True)
class CheckpointConfig:

    backend: str = "postgres"

    # A DSN carries the DB password; keep it out of __repr__ (see LLMConfig).
    connection_string: str | None = field(default=None, repr=False)

    namespace: str = "default"

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("checkpoint.backend must not be empty")
        if not self.namespace.strip():
            raise ValueError("checkpoint.namespace must not be empty")


# ============================================================
# Tracing
# ============================================================

@dataclass(slots=True, frozen=True)
class TracingConfig:

    enabled: bool = True


# ============================================================
# Agent Behaviour
# ============================================================

@dataclass(slots=True, frozen=True)
class BehaviourConfig:

    # Off by default: the verifier then only checks the tool result for an
    # error (a failed or empty step) and routes a tool error to the planner to
    # modify the plan. Set True to also run the LLM semantic check that judges
    # whether a step's output achieved its intent.
    require_verification: bool = False

    require_human_approval: bool = True

    risk_threshold: str = "review"

    def __post_init__(self) -> None:
        if self.risk_threshold not in {"safe", "review", "blocked"}:
            raise ValueError(
                "behaviour.risk_threshold must be 'safe', 'review', or 'blocked'"
            )


# ============================================================
# Prompt
# ============================================================

@dataclass(slots=True, frozen=True)
class PromptConfig:

    planner_prompt: Path

    verifier_prompt: Path

    responder_prompt: Path


# ============================================================
# Metadata
# ============================================================

@dataclass(slots=True, frozen=True)
class MetadataConfig:

    tags: dict[str, str] = field(default_factory=dict)

    custom: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Root Agent Config
# ============================================================

@dataclass(slots=True, frozen=True)
class AgentConfig:

    llm: LLMConfig

    execution: ExecutionConfig

    sandbox: SandboxConfig

    permissions: ToolPermissionConfig

    checkpoint: CheckpointConfig

    tracing: TracingConfig

    behaviour: BehaviourConfig

    prompts: PromptConfig

    metadata: MetadataConfig = field(default_factory=MetadataConfig)
