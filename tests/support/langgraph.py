"""Reusable stubs and builders for the agent-langgraph track.

Importable directly (``from tests.support.langgraph import StubModel``) so both
fixtures and tests that need to build bespoke variants share one definition.
The nodes take their dependencies from ``Runtime[AgentContext]``, which is the
seam these stubs plug into — no test needs an LLM, an MCP gateway, or Langfuse
credentials. The real ``langgraph.runtime.Runtime`` is used (not a hand-rolled
stand-in) so a change in how nodes read the runtime surfaces in these tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_langgraph.graph.state import AgentPlan, AgentState, PlanStep
from agent_langgraph.runtime.context import AgentContext
from agent_langgraph.tracing.tracer import Tracer
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
from common.risk import RiskClassifier
from common.tools import ToolCallResult, ToolInfo, ToolSchema
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

# Config


def build_agent_config(
    *,
    provider: str = "stub",
    llm_timeout_seconds: int = 60,
    llm_temperature: float = 0.0,
    llm_max_tokens: int | None = None,
    llm_top_p: float = 1.0,
    max_iterations: int = 20,
    timeout_seconds: int = 10,
    retry_attempts: int = 1,
    stream: bool = True,
    checkpoint_backend: str = "memory",
    checkpoint_namespace: str = "default",
    connection_string: str | None = None,
    enable_checkpoints: bool = True,
    enable_interrupts: bool = True,
    tracing_enabled: bool = False,
    require_verification: bool = True,
    require_human_approval: bool = False,
    risk_threshold: str = "review",
    prompt_dir: str | Path = "prompts",
    sandbox: SandboxConfig | None = None,
    permissions: ToolPermissionConfig | None = None,
    metadata: MetadataConfig | None = None,
) -> AgentConfig:
    """Build a fully-populated ``AgentConfig`` for tests.

    Defaults are the hermetic ones: an in-memory checkpointer, tracing off, and
    no human gate. Tests that care about a knob pass it explicitly, which keeps
    the intent of each test visible at its call site.
    """
    prompts = Path(prompt_dir)
    return AgentConfig(
        llm=LLMConfig(
            provider=provider,
            model="stub-model",
            api_key="test-key",
            timeout_seconds=llm_timeout_seconds,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            top_p=llm_top_p,
        ),
        execution=ExecutionConfig(
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            stream=stream,
            enable_checkpoints=enable_checkpoints,
            enable_interrupts=enable_interrupts,
        ),
        sandbox=sandbox or SandboxConfig(),
        permissions=permissions or ToolPermissionConfig(),
        checkpoint=CheckpointConfig(
            backend=checkpoint_backend,
            connection_string=connection_string,
            namespace=checkpoint_namespace,
        ),
        tracing=TracingConfig(enabled=tracing_enabled),
        behaviour=BehaviourConfig(
            require_verification=require_verification,
            require_human_approval=require_human_approval,
            risk_threshold=risk_threshold,
        ),
        prompts=PromptConfig(
            planner_prompt=prompts / "planner.txt",
            verifier_prompt=prompts / "verifier.txt",
            responder_prompt=prompts / "responder.txt",
        ),
        metadata=metadata or MetadataConfig(),
    )


# Stub dependencies


DEFAULT_PLAN = AgentPlan(
    summary="echo the goal",
    reasoning="one tool step plus one reasoning step is enough",
    steps=[
        PlanStep(id=1, description="echo hello", tool_name="echo_tool",
                 arguments={"text": "hello"}),
        PlanStep(id=2, description="summarise the result", tool_name=None),
    ],
)


class StubStructured:
    """The object ``with_structured_output`` returns; records what it was asked."""

    def __init__(self, result: Any, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.invocations: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.invocations.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.result


class StubModel:
    """A chat model stand-in that dispatches structured output by schema.

    The planner asks for ``AgentPlan``; the verifier asks for its private
    ``_Verdict``. Dispatching on the schema name means one stub serves both
    without the test having to know which node is running.
    """

    def __init__(
        self,
        *,
        plan: AgentPlan | None = None,
        verdict_success: bool = True,
        verdict_reason: str = "output matches intent",
        answer: str = "Echoed 'hello' and summarised the result.",
        structured_error: Exception | None = None,
        invoke_error: Exception | None = None,
        answers: list[str] | None = None,
    ) -> None:
        self.plan = plan if plan is not None else DEFAULT_PLAN
        self.verdict_success = verdict_success
        self.verdict_reason = verdict_reason
        self.answer = answer
        self.structured_error = structured_error
        self.invoke_error = invoke_error
        #: Sequential replies for the unstructured ``ainvoke``; consumed one per
        #: call, falling back to ``answer`` once empty (lets planner-recovery
        #: tests script schema-echo-then-plan sequences).
        self.answers = list(answers) if answers else []

        #: ``(schema, method)`` for every ``with_structured_output`` call.
        self.structured_calls: list[tuple[Any, Any]] = []
        #: Message lists passed to the unstructured ``ainvoke`` (the responder).
        self.invocations: list[list[Any]] = []
        #: Every ``StubStructured`` handed out, so tests can read its messages.
        self.structured_handles: list[StubStructured] = []

    def with_structured_output(self, schema: Any, method: str | None = None, **_: Any) -> StubStructured:
        self.structured_calls.append((schema, method))
        if getattr(schema, "__name__", "") == "AgentPlan":
            result: Any = self.plan
        else:
            result = schema(success=self.verdict_success, reason=self.verdict_reason)
        handle = StubStructured(result, self.structured_error)
        self.structured_handles.append(handle)
        return handle

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        if self.invoke_error is not None:
            raise self.invoke_error
        if self.answers:
            return AIMessage(content=self.answers.pop(0))
        return AIMessage(content=self.answer)


class StubModelProvider:
    """``ModelProvider`` stand-in returning a shared ``StubModel``."""

    def __init__(self, model: StubModel | None = None) -> None:
        self.model = model if model is not None else StubModel()

    def get_model(self) -> StubModel:
        return self.model


class StubPromptManager:
    """``PromptManager`` stand-in; each prompt can be made to raise."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def _prompt(self, name: str) -> str:
        self.calls.append(name)
        if self.error is not None:
            raise self.error
        return f"You are the {name}."

    def planner(self) -> str:
        return self._prompt("planner")

    def verifier(self) -> str:
        return self._prompt("verifier")

    def responder(self) -> str:
        return self._prompt("responder")


class StubToolRegistry:
    """``ToolRegistry`` stand-in recording invocations and replaying results."""

    def __init__(
        self,
        *,
        results: dict[str, ToolCallResult] | None = None,
        default: ToolCallResult | None = None,
        tools: list[ToolInfo] | None = None,
        list_error: Exception | None = None,
        call_error: Exception | None = None,
    ) -> None:
        self.results = results or {}
        self.default = default
        self.tools = tools if tools is not None else []
        self.list_error = list_error
        self.call_error = call_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolInfo]:
        if self.list_error is not None:
            raise self.list_error
        return self.tools

    async def call_by_name(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.calls.append((tool_name, dict(arguments)))
        if self.call_error is not None:
            raise self.call_error
        if tool_name in self.results:
            return self.results[tool_name]
        if self.default is not None:
            return self.default
        return ToolCallResult(
            success=True, output=f"echo:{arguments.get('text')}", error=None
        )


def make_tool_info(name: str, description: str = "", risk_level: str = "safe") -> ToolInfo:
    """A minimal ``ToolInfo`` for planner-hint and registry tests."""
    return ToolInfo(
        name=name,
        description=description or f"{name} description",
        schema=ToolSchema(input_schema={}, output_schema={}),
        risk_level=risk_level,
    )


# Context / runtime


def build_context(
    config: AgentConfig,
    *,
    model: StubModel | None = None,
    model_provider: Any = None,
    tool_registry: Any = None,
    prompt_manager: Any = None,
    risk_classifier: Any = None,
    tracer: Any = None,
    approval_handler: Any = None,
    auto_approve_all: bool = False,
    task_id: str = "test-task",
) -> AgentContext:
    """Assemble an ``AgentContext`` from stubs, defaulting every slot.

    ``model_provider`` overrides ``model`` and takes any duck-typed provider —
    that is the seam the live tiers use to plug in the real ``ModelProvider``
    (``AgentContext`` is frozen, so it has to be supplied at construction).
    """
    return AgentContext(
        model_provider=(
            model_provider if model_provider is not None else StubModelProvider(model)
        ),
        tool_registry=tool_registry if tool_registry is not None else StubToolRegistry(),
        risk_classifier=risk_classifier if risk_classifier is not None else RiskClassifier(),
        prompt_manager=prompt_manager if prompt_manager is not None else StubPromptManager(),
        # Tracing is disabled (no credentials), so this is a real Tracer running
        # its no-op path — the same object the nodes see in production.
        tracer=tracer if tracer is not None else Tracer(config.tracing),
        config=config,
        approval_handler=approval_handler,
        auto_approve_all=auto_approve_all,
        task_id=task_id,
    )


def build_runtime(context: AgentContext) -> Runtime[AgentContext]:
    """Wrap a context in a real ``Runtime`` the way LangGraph invokes nodes."""
    return Runtime(context=context)


# State helpers


def make_plan(*steps: PlanStep, summary: str = "plan summary",
              reasoning: str = "plan reasoning",
              requires_remediation: bool = False) -> AgentPlan:
    """Build an ``AgentPlan`` from the given steps."""
    return AgentPlan(
        summary=summary,
        reasoning=reasoning,
        steps=list(steps),
        requires_remediation=requires_remediation,
    )


def make_step(step_id: int = 1, **overrides: Any) -> PlanStep:
    """Build a ``PlanStep``, defaulting to a single tool-backed step."""
    fields: dict[str, Any] = {
        "id": step_id,
        "description": f"step {step_id}",
        "tool_name": "echo_tool",
        "arguments": {"text": "hello"},
    }
    fields.update(overrides)
    return PlanStep(**fields)


def make_state(**overrides: Any) -> AgentState:
    """Build an ``AgentState`` with the keys every node expects present.

    Nodes read most fields with ``.get()`` but ``should_execute`` indexes
    ``plan`` and ``current_step`` directly, so both are always populated.
    """
    state: dict[str, Any] = {
        "messages": [],
        "goal": "Echo hello",
        "plan": make_plan(make_step()),
        "current_step": 0,
        "workflow_phase": None,
        "findings": [],
        "verification_success": None,
        "verification_reason": None,
        "retry_count": 0,
        "last_error": None,
        "status": None,
    }
    state.update(overrides)
    return state  # type: ignore[return-value]
