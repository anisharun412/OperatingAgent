"""Tests for the planner node (``agent_langgraph.nodes.planner``)."""

from __future__ import annotations

import json

import pytest
from agent_langgraph.graph.state import AgentPlan, Finding, PlanStep
from agent_langgraph.nodes.planner import PlannerNode, planner_function
from common.enums import TaskStatus, WorkflowPhase
from common.exceptions import PlanningException
from langchain_core.messages import SystemMessage

from tests.support.langgraph import (
    StubModel,
    StubToolRegistry,
    build_context,
    build_runtime,
    make_state,
    make_tool_info,
)


async def test_planner_node_returns_plan_and_resets_pointer(agent_config, stub_model) -> None:
    runtime = build_runtime(build_context(agent_config, model=stub_model))
    delta = await PlannerNode(make_state(), runtime)

    assert isinstance(delta["plan"], AgentPlan)
    assert delta["current_step"] == 0
    assert delta["status"] is TaskStatus.PLANNING


async def test_planner_node_raises_without_goal(agent_config, stub_model) -> None:
    runtime = build_runtime(build_context(agent_config, model=stub_model))
    with pytest.raises(PlanningException):
        await PlannerNode(make_state(goal=""), runtime)


async def test_planner_uses_structured_output_with_json_schema(agent_config, stub_model) -> None:
    runtime = build_runtime(build_context(agent_config, model=stub_model))
    await planner_function("goal", [], runtime)

    schema, method = stub_model.structured_calls[0]
    assert schema is AgentPlan
    assert method == "json_schema"


async def test_planner_prompt_includes_available_tools_hint(agent_config) -> None:
    model = StubModel()
    registry = StubToolRegistry(tools=[make_tool_info("echo_tool", "echoes text")])
    runtime = build_runtime(build_context(agent_config, model=model, tool_registry=registry))

    await planner_function("goal", [], runtime)

    system_message = model.structured_handles[0].invocations[0][0]
    assert isinstance(system_message, SystemMessage)
    assert "You are the planner." in system_message.content
    assert "echo_tool: echoes text" in system_message.content


async def test_planner_prompt_has_no_hint_when_no_tools(agent_config) -> None:
    model = StubModel()
    runtime = build_runtime(
        build_context(agent_config, model=model, tool_registry=StubToolRegistry(tools=[]))
    )
    await planner_function("goal", [], runtime)

    system_message = model.structured_handles[0].invocations[0][0]
    assert system_message.content.startswith("You are the planner.")
    assert "Available tools" not in system_message.content


async def test_planner_tolerates_tool_listing_failure(agent_config) -> None:
    """If the registry can't be reached, planning proceeds without a hint."""
    model = StubModel()
    registry = StubToolRegistry(list_error=RuntimeError("gateway down"))
    runtime = build_runtime(build_context(agent_config, model=model, tool_registry=registry))

    plan = await planner_function("goal", [], runtime)
    assert isinstance(plan, AgentPlan)
    system_message = model.structured_handles[0].invocations[0][0]
    assert system_message.content.startswith("You are the planner.")
    assert "Available tools" not in system_message.content


async def test_planner_wraps_model_error_in_planning_exception(agent_config) -> None:
    model = StubModel(structured_error=RuntimeError("provider exploded"))
    runtime = build_runtime(build_context(agent_config, model=model))

    with pytest.raises(PlanningException) as excinfo:
        await planner_function("goal", [], runtime)
    assert "provider exploded" in str(excinfo.value)


async def test_planner_recovers_when_model_echoes_schema(agent_config) -> None:
    """The witnessed Groq failure: json_schema mode returns the schema itself.

    The run must recover via the corrective prompt instead of crashing with
    ``Failed to parse AgentPlan``.
    """
    schema_echo = json.dumps(
        [
            {
                "description": "Structured planner output.",
                "properties": {"summary": {"type": "string"}},
                "type": "object",
            }
        ]
    )
    good_plan = json.dumps(
        {
            "summary": "scaffold the project",
            "reasoning": "the project folder does not exist yet",
            "steps": [
                {
                    "id": 1,
                    "description": "scaffold a vite project",
                    "tool_name": "run_command",
                    "arguments": {"command": "npm create vite@latest test"},
                }
            ],
            "requires_remediation": False,
        }
    )
    model = StubModel(
        structured_error=RuntimeError("Failed to parse AgentPlan"),
        answers=[schema_echo, good_plan],
    )
    runtime = build_runtime(build_context(agent_config, model=model))

    plan = await planner_function("create a vite project", [], runtime)

    assert isinstance(plan, AgentPlan)
    assert plan.summary == "scaffold the project"
    assert [step.tool_name for step in plan.steps] == ["run_command"]
    # The schema echo was quoted back to the model as a correction.
    second_prompt = model.invocations[1][-1].content
    assert "instead of an actual plan instance" in second_prompt


async def test_planner_recovery_exhaustion_reports_original_error(agent_config) -> None:
    """Recovery that never converges still surfaces the structured error."""
    model = StubModel(
        structured_error=RuntimeError("provider exploded"),
        answers=["not json", "still not json"],
    )
    runtime = build_runtime(build_context(agent_config, model=model))

    with pytest.raises(PlanningException) as excinfo:
        await planner_function("goal", [], runtime)
    message = str(excinfo.value)
    assert "provider exploded" in message
    assert "recovery failed" in message


async def test_planner_forwards_prior_messages(agent_config) -> None:
    """Prior conversation messages sit between the system prompt and the new
    human 'generate a plan' turn."""
    from langchain_core.messages import HumanMessage

    model = StubModel()
    runtime = build_runtime(build_context(agent_config, model=model))
    prior = [HumanMessage(content="earlier turn")]

    await planner_function("my goal", prior, runtime)
    messages = model.structured_handles[0].invocations[0]
    assert messages[1] is prior[0]
    assert "my goal" in messages[-1].content


async def test_planner_drops_no_tool_synthesis_steps(agent_config) -> None:
    """Final prose/artifacts belong to the responder, not the executor."""
    model = StubModel(
        plan=AgentPlan(
            summary="inspect and summarise",
            reasoning="the model added a synthesis step",
            steps=[
                PlanStep(id=1, description="inspect status", tool_name="git_status"),
                PlanStep(id=2, description="suggest a commit message"),
            ],
        )
    )
    runtime = build_runtime(build_context(agent_config, model=model))

    plan = await planner_function("check status and give a commit message", [], runtime)

    assert [step.tool_name for step in plan.steps] == ["git_status"]


async def test_planner_only_includes_findings_for_current_task(agent_config) -> None:
    model = StubModel()
    runtime = build_runtime(build_context(agent_config, model=model, task_id="current-task"))
    state = make_state(
        workflow_phase=WorkflowPhase.REMEDIATE,
        findings=[
            Finding(task_id="previous-task", step_id=1, description="old finding", detail="old detail"),
            Finding(task_id="current-task", step_id=2, description="current finding", detail="current detail"),
        ],
    )

    await PlannerNode(state, runtime)

    system_message = model.structured_handles[0].invocations[0][0]
    assert "current detail" in system_message.content
    assert "old detail" not in system_message.content
