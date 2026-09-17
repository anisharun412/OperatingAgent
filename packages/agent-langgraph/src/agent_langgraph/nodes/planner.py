from __future__ import annotations

import json
import logging
from typing import Any

from agent_langgraph.graph.state import AgentPlan, AgentState, Finding
from agent_langgraph.runtime.context import AgentContext
from common.enums import TaskStatus, WorkflowPhase
from common.events import AgentEvent
from common.exceptions import PlanningException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

log = logging.getLogger(__name__)


async def _emit_event(ctx: AgentContext, event: AgentEvent) -> None:
    """Deliver a best-effort observability event to the service sink.

    Reasoning/plan events feed the UI's Thought-process view and the durable
    history, but they must never turn a good plan into a failed run: a sink
    failure is logged loudly and swallowed here. (Authoritative persistence
    failures are surfaced by the service layer's own ``on_event`` wrapper.)
    """
    if ctx.event_sink is None:
        return
    try:
        outcome = ctx.event_sink(event)
        if outcome is not None and hasattr(outcome, "__await__"):
            await outcome
    except Exception as exc:  # noqa: BLE001 - observability is not load-bearing
        log.warning("planner event sink raised: %s", exc)

#: Per-finding detail cap when building the remediation prompt, so a large
#: investigation cannot blow the model's context window.
_MAX_FINDING_DETAIL = 1_500


def _format_findings(findings: list[Finding]) -> str:
    """Render accumulated findings for the planner prompt."""
    lines = []
    for n, finding in enumerate(findings, start=1):
        detail = finding.detail
        if len(detail) > _MAX_FINDING_DETAIL:
            detail = f"{detail[:_MAX_FINDING_DETAIL]}... [truncated]"
        source = f" (via {finding.source_tool})" if finding.source_tool else ""
        lines.append(f"{n}. {finding.description}{source}\n   {detail}")
    return "\n".join(lines)


def _current_task_findings(findings: list[Finding] | None, task_id: str) -> list[Finding]:
    return [
        finding
        for finding in findings or []
        if getattr(finding, "task_id", "") == task_id
    ]


def _phase_instruction(phase: WorkflowPhase, findings: list[Finding]) -> str:
    """Phase-specific planning instruction appended to the system prompt.

    This is what makes a single planner serve both phases: the same goal yields
    a read-only investigation plan first, then a remediation plan built from
    what that investigation actually found.
    """
    if phase is WorkflowPhase.REMEDIATE:
        if not findings:
            # Defensive: the phase transition skips remediation when nothing was
            # found, so this should not be reachable.
            return (
                "\n\nCurrent phase: REMEDIATE, but no findings were recorded. "
                "Produce the smallest plan that addresses the goal directly."
            )
        return (
            "\n\nCurrent phase: REMEDIATE."
            "\nThe investigation phase recorded the findings below. Produce a plan "
            "that acts on them — the fixes, changes, or follow-up work they call "
            "for. Reference the specific findings your steps address, and do not "
            "create a step for a finding that needs no action. Do not re-run the "
            "investigation."
            f"\n\nFindings from investigation:\n{_format_findings(findings)}"
        )

    return (
        "\n\nCurrent phase: INVESTIGATE."
        "\nIf the goal can be satisfied outright, plan it directly and set "
        "requires_remediation to false."
        "\nIf the goal instead asks you to find something and then act on it "
        "(e.g. 'find and fix bugs', 'check for issues and resolve them'), plan "
        "ONLY the investigation now and set requires_remediation to true: list "
        "and read files, search, inspect status, run tests or checks. In that "
        "case do NOT plan any step that writes, edits, deletes, moves or "
        "otherwise modifies state — the follow-up plan is built separately once "
        "the findings are in. Each investigation step should produce an "
        "observation."
    )


async def _available_tools_hint(runtime: Runtime[AgentContext]) -> str:
    """Describe the tools the planner may reference, defensively.

    The planner produces structured output, so tools are *described* (not
    bound) — that lets it fill in ``PlanStep.tool_name`` with real names. If
    the tool registry / MCP adapter isn't wired yet, planning still proceeds
    without a hint rather than crashing.
    """
    try:
        tools = await runtime.context.tool_registry.list_tools()
    except Exception as exc:  # noqa: BLE001 - pluggable tool registry boundary
        log.warning("could not list tools for planning: %s", exc)
        return ""

    if not tools:
        return ""

    lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    return f"\n\nAvailable tools (reference these by name in tool_name):\n{lines}"


#: Plain-text recovery attempts after a structured plan parse fails.
_PLAN_TEXT_RETRIES = 2
#: How much of a failing completion to quote back in the corrective prompt.
_MAX_FAILING_SAMPLE = 600


def _truncate(text: str, limit: int = _MAX_FAILING_SAMPLE) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def _plan_shape_hint() -> str:
    """The exact shape a plan response must have (for corrective prompts)."""
    return (
        "Reply with ONLY a single JSON object shaped EXACTLY like this example, "
        "with real values for the goal — no prose, no markdown fences, and do "
        "NOT return the JSON Schema definition: "
        '{"summary": "...", "reasoning": "...", '
        '"steps": [{"id": 1, "description": "...", "tool_name": "...", '
        '"arguments": {...}}], "requires_remediation": false}.'
    )


def _extract_text(response: Any) -> str:
    """Pull text out of a chat completion, whatever its content shape."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    return text.strip()


def _looks_like_schema(payload: Any, raw_text: str) -> bool:
    """Detect the witnessed failure: the model echoed the schema itself."""
    if "Structured planner output" in raw_text:
        return True
    if isinstance(payload, list):
        return len(payload) == 1 and _looks_like_schema(payload[0], "")
    return (
        isinstance(payload, dict)
        and payload.get("type") == "object"
        and isinstance(payload.get("properties"), dict)
    )


async def _recover_plan(
    raw_model: Any,
    messages: list,
    first_error: Exception,
    *,
    attempts: int = _PLAN_TEXT_RETRIES,
) -> AgentPlan:
    """Rebuild a plan in plain text after structured parsing failed.

    Providers in ``json_schema`` mode occasionally echo the prompt's schema
    verbatim (or otherwise unparsable text). Crashing the run then is worse
    than one more model round-trip: ask the model — without the structured
    wrapper — to produce the plan object, quoting its own failing output back
    as a correction. Raises ``PlanningException`` only when every attempt fails.
    """
    transcript: list = [
        *messages,
        HumanMessage(
            content=(
                f"Your previous response was not a valid plan "
                f"({type(first_error).__name__}: {first_error}). "
                + _plan_shape_hint()
            )
        ),
    ]
    last_error: Exception | None = None
    for _attempt in range(1, attempts + 1):
        try:
            response = await raw_model.ainvoke(transcript)
        except Exception as exc:  # noqa: BLE001 - retried with a correction below
            last_error = exc
            transcript = [
                *transcript,
                HumanMessage(content=f"The model call failed ({exc}). {_plan_shape_hint()}"),
            ]
            continue
        text = _strip_json_fences(_extract_text(response))
        try:
            payload = json.loads(text)
        except ValueError as exc:
            last_error = exc
            transcript = [
                *transcript,
                HumanMessage(
                    content=(
                        f"That was not valid JSON ({exc}). {_plan_shape_hint()} "
                        f"Your failing response was: {_truncate(text)}"
                    )
                ),
            ]
            continue
        if _looks_like_schema(payload, text):
            last_error = ValueError("model returned the JSON Schema instead of a plan")
            transcript = [
                *transcript,
                HumanMessage(
                    content=(
                        "You returned the JSON Schema definition instead of an "
                        "actual plan instance. Produce the plan object itself. "
                        + _plan_shape_hint()
                    )
                ),
            ]
            continue
        try:
            return AgentPlan.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - corrected explicitly below
            last_error = exc
            transcript = [
                *transcript,
                HumanMessage(
                    content=(
                        f"That response failed plan validation: {exc}. "
                        + _plan_shape_hint()
                        + f" Your failing response was: {_truncate(text)}"
                    )
                ),
            ]
    raise PlanningException(
        f"planner recovery failed after {attempts} attempt(s): {last_error}"
    )


async def planner_function(
    goal: str,
    messages: list,
    runtime: Runtime[AgentContext],
    phase: WorkflowPhase = WorkflowPhase.INVESTIGATE,
    findings: list[Finding] | None = None,
) -> AgentPlan:
    """
    Generate a structured plan for ``goal`` using the configured provider.

    Args:
        goal (str): The goal for which to generate a plan.
        messages (list): Prior conversation messages.
        runtime (Runtime[AgentContext]): Runtime carrying the shared deps.
        phase (WorkflowPhase): Which phase to plan for.
        findings (list[Finding] | None): Observations from earlier phases.

    Returns:
        AgentPlan: The validated structured plan.
    """
    ctx = runtime.context

    model = ctx.model_provider.get_model().with_structured_output(
        AgentPlan, method="json_schema"
    )

    active_findings = _current_task_findings(findings, ctx.task_id)
    system_prompt = (
        ctx.prompt_manager.planner()
        + _phase_instruction(phase, active_findings)
        + await _available_tools_hint(runtime)
    )
    full_messages = [
        SystemMessage(content=system_prompt),
        *messages,
        HumanMessage(content=f"Generate a plan for the goal: {goal}"),
    ]

    try:
        plan = await model.ainvoke(full_messages)
    except Exception as exc:  # noqa: BLE001 - every model failure funnels into recovery below
        # A model that echoes the schema (or otherwise unparsable text) must not
        # end the run: one plain-text corrective round-trip recovers it, and the
        # original error stays in the final message for diagnosis.
        provider = ctx.config.llm.provider
        log.warning(
            "structured plan parse failed with provider %s (%s: %s); retrying with a corrective prompt",
            provider, type(exc).__name__, exc,
        )
        try:
            plan = await _recover_plan(ctx.model_provider.get_model(), full_messages, exc)
        except Exception as recovery_exc:
            log.error("planning failed with provider %s: %s", provider, recovery_exc)
            raise PlanningException(
                f"planner failed with provider {provider}: structured parse failed ({exc}); "
                f"recovery failed ({recovery_exc})"
            ) from recovery_exc

    validated_plan = plan if isinstance(plan, AgentPlan) else AgentPlan.model_validate(plan)
    validated_plan = _remove_synthesis_steps(validated_plan)

    log.info(
        "plan generated with provider %s for phase %s",
        ctx.config.llm.provider, phase.value,
    )
    # Surface the plan's rationale to the UI (Thought process) and the durable
    # history, mirroring the native track's ``reasoning_delta`` stream. The
    # ``plan_created`` record carries the full structure for Activity/history.
    if validated_plan.reasoning and validated_plan.reasoning.strip():
        await _emit_event(
            ctx,
            AgentEvent(
                type="reasoning_delta",
                payload={"text": validated_plan.reasoning.strip()},
            ),
        )
    await _emit_event(
        ctx,
        AgentEvent(
            type="plan_created",
            payload={
                "summary": validated_plan.summary,
                "reasoning": validated_plan.reasoning,
                "phase": phase.value,
                "requires_remediation": validated_plan.requires_remediation,
                "steps": [
                    {
                        "id": step.id,
                        "description": step.description,
                        "tool_name": step.tool_name,
                        "arguments": step.arguments,
                    }
                    for step in validated_plan.steps
                ],
            },
        ),
    )
    return validated_plan


def _remove_synthesis_steps(plan: AgentPlan) -> AgentPlan:
    """Keep planning executable and leave synthesis to the responder.

    The planner occasionally returns a final no-tool step such as "suggest a
    commit message". The executor cannot perform that step; treating its
    description as output makes an instruction look like an observation in the
    transcript. The responder already has the goal and tool results, so it is
    the correct place to produce the requested prose or artifact.
    """
    executable = [step for step in plan.steps if step.tool_name]
    dropped = len(plan.steps) - len(executable)
    if dropped:
        log.warning(
            "planner returned %d non-executable synthesis step(s); "
            "delegating synthesis to responder",
            dropped,
        )
        plan = plan.model_copy(update={"steps": executable})
    return plan


async def PlannerNode(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """
    Plan the work for the current phase and reset the step pointer.

    Runs once per phase: first for the investigation, then again after the phase
    transition for the remediation, using the findings it accumulated. Neither
    ``findings`` nor ``retry_count`` is touched here — findings must survive the
    replan, and the retry budget is shared across the whole run.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict: State delta with the generated plan.
    """
    goal = state.get("goal")
    if not goal:
        raise PlanningException("planner invoked without a goal")

    # First entry establishes the phase; later entries come from the transition.
    phase = state.get("workflow_phase") or WorkflowPhase.INVESTIGATE
    findings = _current_task_findings(state.get("findings", []), runtime.context.task_id)

    plan = await planner_function(
        goal,
        state.get("messages", []),
        runtime,
        phase=phase,
        findings=findings,
    )

    return {
        "plan": plan,
        "current_step": 0,
        "workflow_phase": phase,
        "status": TaskStatus.PLANNING,
    }
