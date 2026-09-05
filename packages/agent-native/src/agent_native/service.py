"""The front door.

`AgentRuntime` wires every piece together once - database, event bus, model and
tool registries, the policy stack, the permission manager, the loop. Build one
and you have a working agent. `AgentService` is the thin surface a UI or an HTTP
layer calls: start a session, send a message, answer a permission prompt, watch
the events. Nothing above this layer needs to know how any of it is built.

The HTTP server, the real MCP wrapping and Postgres all sit on top of this same
service in later passes; keeping the service small is what lets them.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from .config import AgentConfig, PromptBuilder, discover_skills, skill_listing
from .context import ContextManager
from .conversation import Session, Usage, _new_id, system_message, user_message
from .database import Database, MemoryDatabase
from .events import EventBus, EventType
from .hooks import HookContext, HookManager, HookPoint
from .loop import (
    AgentLoop,
    Cancellation,
    Limits,
    RetryCoordinator,
    RunContext,
    RunResult,
    RunStatus,
)
from .memory import MemoryStore, read_project_instructions
from .models.base import ModelRegistry
from .monitoring import Monitoring
from .permissions import (
    PermissionDuration,
    PermissionManager,
    PermissionStore,
    PlanModePolicy,
    PolicyChain,
    RulePolicy,
    SessionPolicy,
    WorkspacePolicy,
)
from .redaction import Redactor
from .tools.base import ToolRegistry
from .tools.builtins import default_tools
from .tools.manager import ToolManager
from .tools.memory_tools import format_memories, memory_tools
from .tools.plan_tool import PlanTool
from .tools.skill_tool import InvokeSkillTool
from .tools.subagent import DelegateTool, FanOutTool, is_helper_run


class AgentRuntime:
    """Everything wired together. The one object that owns the moving parts."""

    def __init__(
        self,
        database: Database | None = None,
        model_registry: ModelRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        policy: Any = None,
        agents: list | None = None,
        monitoring: Monitoring | None = None,
        sandbox: Any = None,
    ) -> None:
        self.database = database or MemoryDatabase()
        # One redactor for the whole runtime, installed into every sink that stores
        # or ships text - the event bus, the trace exporter, the memory store - so a
        # secret (the Groq key dumped in tool output, say) can't reach an event row,
        # a trace file, the collector, a memory, or a log that prints from them. The
        # conversation itself is left exact; see redaction.py for why.
        self.redactor = Redactor()
        self.events = EventBus(self.database, redactor=self.redactor)
        self.models = model_registry or ModelRegistry()
        self.tools = tool_registry or ToolRegistry()

        # PlanModePolicy is inert unless a run sets `Limits.plan_mode`, so it costs
        # an ordinary run nothing; when a run is in plan mode it denies every
        # mutating call, and denial wins in the chain - the hard half of the gate.
        self.policy = policy or PolicyChain(
            [RulePolicy(), WorkspacePolicy(), SessionPolicy(), PlanModePolicy()]
        )
        self.permission_store = PermissionStore(self.database)
        self.permissions = PermissionManager(self.permission_store, self.events)
        # Where shell commands run. None means "in this process, like everything
        # else" - the caller decides, because starting containers is the caller's
        # cost to accept (see tools/sandbox.py).
        self.sandbox = sandbox
        self.tool_manager = ToolManager(
            self.tools, self.policy, self.permissions, sandbox=sandbox
        )

        self.context = ContextManager()
        # Recording is on by default - a run you can't inspect afterwards is the
        # thing this project is trying not to build. Writing traces to disk is
        # still opt-in: pass Monitoring(trace_dir=...) to get files.
        self.monitoring = monitoring or Monitoring()
        # A caller-supplied Monitoring might not know about redaction; give it the
        # runtime's redactor unless it already has one, so traces are masked either way.
        if getattr(self.monitoring, "redactor", None) is None:
            self.monitoring.redactor = self.redactor
        self.prompt = PromptBuilder()
        # One backoff coordinator for the whole runtime, shared into the loop and
        # every helper loop it spawns, so a rate limit that hits several at once is
        # backed off as one instead of retried N-ways (see loop.RetryCoordinator).
        self.retry_coordinator = RetryCoordinator()

        # Notes that outlive a run. The two tools are registered here rather than
        # borrowed from MCP because a note is the agent's own bookkeeping, and
        # because they need the same store the prompt is seeded from.
        self.memory = MemoryStore(self.database, redactor=self.redactor)
        for tool in memory_tools(self.memory):
            self.tools.register(tool)
        # A checklist the model keeps for itself. Nothing reads it but the model.
        self.tools.register(PlanTool())
        # Loads a named skill's full instructions on demand - the second half of
        # progressive disclosure (the catalogue is seeded into the prompt below).
        # Read-only and stateless: it re-reads the skill folder on each call.
        self.tools.register(InvokeSkillTool())

        # Lifecycle hooks the user registers on the runtime. Shared into the loop
        # and, through the runtime, into every helper loop, so a hook fires wherever
        # a tool runs or a run stops. Empty by default, and an empty manager is a
        # no-op at every point - registering nothing leaves behaviour unchanged.
        self.hooks = HookManager()

        self.loop = AgentLoop(
            self.models,
            self.tools,
            self.tool_manager,
            self.context,
            self.events,
            self.database,
            self.monitoring,
            retry_coordinator=self.retry_coordinator,
            hooks=self.hooks,
        )

        agents = agents or [AgentConfig()]
        self.agents: dict = {a.name: a for a in agents}

        # Registered last, because they read `self.agents` to list the helpers they
        # can name, and hold the runtime so they can build a loop of their own.
        # `delegate` runs one helper on one job; `fan_out` runs one across many at
        # once. Both hand a helper a tool list with both of these removed, so no
        # helper delegates or fans out in turn.
        self.tools.register(DelegateTool(self))
        self.tools.register(FanOutTool(self))

    def register_default_tools(self) -> None:
        """Register any built-in tools. None ship now - the agent's tools come from
        the MCP gateway (see tools/mcp_bridge.py), so this registers nothing; it's
        kept only for API compatibility."""
        for tool in default_tools():
            self.tools.register(tool)

    def config_for(self, agent: str) -> AgentConfig:
        return self.agents.get(agent) or AgentConfig(name=agent)

    def reconfigure_models(
        self,
        *,
        provider: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        timeout_seconds: int = 60,
    ) -> list[str]:
        """Replace provider/model registrations for future runs.

        Existing loops keep their resolved provider and model objects. New runs
        resolve the updated registry without rebuilding the database, tools, or
        event bus.
        """
        provider = provider.strip().lower()
        model = model.strip()
        if not provider or not model:
            raise ValueError("provider and model are required")

        if provider == "groq":
            from .models.groq_model import Groq

            self.models.register_provider("groq", Groq(base_url=base_url))
        elif provider == "ollama":
            from .models.ollama_model import Ollama

            self.models.register_provider("ollama", Ollama(host=base_url or "http://localhost:11434"))
        else:
            raise ValueError("native provider must be 'groq' or 'ollama'")

        from .models.base import Model

        self.models.register_model(model, Model(provider=provider, model_id=model))
        for config in self.agents.values():
            config.model = model
            config.temperature = temperature
            config.top_p = top_p
            config.max_output_tokens = max_tokens
            config.timeout_seconds = timeout_seconds
        return self.models.list_model_names()


class AgentService:
    """The small set of things the outside world asks the agent to do."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def create_session(
        self,
        agent: str = "build",
        title: str = "",
        working_directory: str = ".",
        session_id: str | None = None,
    ) -> Session:
        """Start a new session and seed it with the agent's system prompt.

        This is where the two kinds of carried-over knowledge get picked up: the
        `AGENT.md` the user wrote in the working folder, and the handful of notes
        that most recently proved useful. Both are read once, here, so a long run
        never re-reads them mid-conversation and quietly changes its own
        instructions.
        """
        config = self.runtime.config_for(agent)
        session_kwargs: dict[str, Any] = {
            "agent": agent,
            "title": title,
            "working_directory": working_directory,
        }
        if session_id is not None:
            session_kwargs["id"] = session_id
        session = Session(**session_kwargs)
        await self.runtime.database.create_session(session)

        tool_names = [
            t.definition.full_name for t in self.runtime.tools.get_available_tools(config)
        ]
        # Session id is deliberately not passed: a fresh session should start with
        # what was learned in the *earlier* ones, which is the whole feature.
        remembered = await self.runtime.memory.recent()
        prompt = self.runtime.prompt.build(
            config,
            session,
            tool_names,
            project_instructions=read_project_instructions(working_directory),
            remembered=format_memories(remembered) if remembered else "",
            skills=skill_listing(discover_skills(working_directory)),
        )
        await self.runtime.database.save_message(system_message(prompt, session.id))
        return session

    async def send_message(
        self,
        session_id: str,
        text: str,
        limits: Limits | None = None,
        cancellation: Cancellation | None = None,
        media: list | None = None,
    ):
        """Add the user's message and run the agent until it stops."""
        session = await self.runtime.database.get_session(session_id)
        if session is None:
            raise KeyError(f"No such session: {session_id!r}")
        config = self.runtime.config_for(session.agent)

        # Mint the run before writing its first event.  The canonical Postgres
        # schema anchors every event to an agent_run, including MESSAGE_ADDED.
        run_id = "run_" + uuid.uuid4().hex[:8]
        user_msg = user_message(session_id, text, media=media)
        await self.runtime.database.save_message(user_msg)
        await self.runtime.events.emit(
            session_id,
            EventType.MESSAGE_ADDED,
            {"id": user_msg.id, "role": "user"},
            run_id,
        )

        # The run id is minted here, before the loop, so the prompt-submitted hook
        # can name the run the prompt kicks off. This point is observe-only - a hook
        # here can log or annotate, but only a pre-tool hook may veto - so its return
        # is deliberately ignored. With nothing registered, this is skipped entirely.
        if self.runtime.hooks.has(HookPoint.PROMPT_SUBMITTED):
            await self.runtime.hooks.dispatch(
                HookContext(
                    point=HookPoint.PROMPT_SUBMITTED,
                    session_id=session_id,
                    run_id=run_id,
                    text=text,
                )
            )

        conversation = await self.runtime.database.load_conversation(session_id)
        context = RunContext(
            session=session,
            run_id=run_id,
            config=config,
            limits=limits or Limits(max_turns=config.max_turns),
            cancellation=cancellation or Cancellation(),
        )
        return await self.runtime.loop.run(conversation, context)

    async def resume_run(
        self,
        session_id: str,
        limits: Limits | None = None,
        cancellation: Cancellation | None = None,
    ):
        """Reattach to a session and carry its last run to completion.

        A run can stop before it finishes for reasons that have nothing to do with
        the model: the process was killed, the machine rebooted, a deploy cycled
        the server. Because every step is written to the event log and every message
        to the database as it happens, the run doesn't have to start over - it is
        picked up from exactly where the log ends. This is the run-level counterpart
        to `subscribe`, which already replays the *events* from a cursor; here it's
        the work itself that continues.

        Two things make that safe. If the log's last event is RUN_FINISHED the run
        already completed: its receipt is rebuilt from that event and returned with
        no model call, so resuming a finished run is a no-op that still hands back
        the result. Otherwise the stored conversation is replayed into the loop,
        which finishes any tool call that was in flight and re-runs nothing that
        already has a saved result (see `AgentLoop._prepare_resume`) - the
        no-duplicate-side-effects guarantee. Reusing the interrupted run's id means
        the resumed run updates that run's record and keeps emitting under the same
        run rather than forking a second one.
        """
        session = await self.runtime.database.get_session(session_id)
        if session is None:
            raise KeyError(f"No such session: {session_id!r}")

        events = [e for e in await self.runtime.database.load_events(session_id, 0) if not is_helper_run(e.run_id)]
        if events and events[-1].type == EventType.RUN_FINISHED:
            # Already finished - rebuild the receipt and don't touch the model.
            return _run_result_from_event(events[-1])

        config = self.runtime.config_for(session.agent)
        conversation = await self.runtime.database.load_conversation(session_id)
        context = RunContext(
            session=session,
            # Continue the interrupted run's id when the log has one, so its record
            # is updated in place and its events stay under one run; only invent a
            # fresh id if nothing run-scoped was ever logged.
            run_id=_latest_run_id(events) or ("run_" + uuid.uuid4().hex[:8]),
            config=config,
            limits=limits or Limits(max_turns=config.max_turns),
            cancellation=cancellation or Cancellation(),
        )
        return await self.runtime.loop.run(conversation, context)

    async def fork_session(self, session_id: str, title: str = "") -> Session:
        """Branch a conversation into a new session to try an alternative.

        The message history is copied verbatim - the system prompt included - so the
        fork begins knowing everything the original knew, down to the same seeded
        instructions. The event log is *not* copied: the fork gets its own stream
        numbered from one, so diverging one session never renumbers or disturbs the
        other. Each copied message gets a fresh id under the new session; the parts
        list is shallow-copied so appends to one history can't reach into the other.
        """
        source = await self.runtime.database.get_session(session_id)
        if source is None:
            raise KeyError(f"No such session: {session_id!r}")

        fork = Session(
            agent=source.agent,
            title=title or (f"{source.title} (fork)" if source.title else "fork"),
            working_directory=source.working_directory,
        )
        await self.runtime.database.create_session(fork)

        conversation = await self.runtime.database.load_conversation(session_id)
        for message in conversation.messages:
            await self.runtime.database.save_message(
                replace(message, id=_new_id(), session_id=fork.id, parts=list(message.parts))
            )
        return fork

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and everything under it; return whether it existed.

        A thin pass-through to the store, which removes the messages, events, runs,
        and the session's own grants and notes in one shot. False means there was no
        such session, which the caller turns into a not-found rather than a silent ok.
        """
        return await self.runtime.database.delete_session(session_id)

    async def resolve_permission(
        self,
        call_id: str,
        allowed: bool,
        duration: PermissionDuration = PermissionDuration.ONCE,
        scope: str = "",
    ) -> None:
        """Answer a permission prompt the agent is waiting on.

        `scope` narrows a remembered yes to a folder ("notes"), so approving for
        the session doesn't have to mean approving everywhere.
        """
        await self.runtime.permissions.resolve(call_id, allowed, duration, scope)

    def pending_permissions(self) -> list:
        """Permission prompts currently waiting on the user."""
        return self.runtime.permissions.pending()

    async def subscribe(self, session_id: str, from_sequence: int = 0):
        """Stream a session's events, catching up from `from_sequence` first."""
        async for event in self.runtime.events.subscribe(session_id, from_sequence):
            yield event

    async def get_conversation(self, session_id: str):
        """The full stored conversation for a session."""
        return await self.runtime.database.load_conversation(session_id)


# ---------------------------------------------------------------------------
# Rebuilding a run's receipt from what was logged, for resume
# ---------------------------------------------------------------------------
def _latest_run_id(events: list) -> str:
    """The id of the most recent run any event belongs to.

    The events from before the first turn - the user message going in - carry no
    run id, so this walks back to the last one that does. Reusing that id is what
    lets a resumed run update the interrupted run's record instead of forking a
    second one.
    """
    for event in reversed(events):
        if event.run_id:
            return event.run_id
    return ""


def _run_result_from_event(event: Any) -> RunResult:
    """Rebuild a run's receipt from the RUN_FINISHED event that recorded it.

    Used when a resume finds the run already finished: the caller gets the same
    RunResult it would have got the first time, without a model call. The numbers
    are the receipt fields the loop wrote (see `AgentLoop._finish`); redaction only
    ever masks secret-shaped text, so they come back intact.
    """
    data = event.data or {}
    usage = Usage(
        input_tokens=int(data.get("input_tokens", 0) or 0),
        output_tokens=int(data.get("output_tokens", 0) or 0),
        cached_tokens=int(data.get("cached_tokens", 0) or 0),
        reasoning_tokens=int(data.get("reasoning_tokens", 0) or 0),
    )
    return RunResult(
        run_id=event.run_id or data.get("run_id", ""),
        status=_status_from_value(data.get("status", RunStatus.FINISHED.value)),
        turns=int(data.get("turns", 0) or 0),
        final_text=data.get("final_text", "") or "",
        usage=usage,
        error=data.get("error", "") or "",
        duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
        cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
        model=data.get("model", "") or "",
        retries=int(data.get("retries", 0) or 0),
        fallbacks=int(data.get("fallbacks", 0) or 0),
        stop_reason=data.get("stop_reason", "") or "",
        trace_id=data.get("trace_id", "") or "",
    )


def _status_from_value(value: str) -> RunStatus:
    """Map a stored status string back to a RunStatus, defaulting to FINISHED."""
    for status in RunStatus:
        if status.value == value:
            return status
    return RunStatus.FINISHED
