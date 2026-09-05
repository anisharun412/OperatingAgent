"""The agent loop: think, act, observe, repeat.

This is the whole engine. One run is:

  1. render the conversation and ask the model to think (run_turn)
  2. the model answers with text, or with tool calls, or both
  3. if it asked for tools, run them and add each result back (run_tools)
  4. go back to step 1 so the model can read the results

It stops when the model gives an answer with no tool calls (FINISHED), when it
hits a limit on turns or time (LIMIT_REACHED), when the caller cancels (CANCELLED),
or when something throws (ERROR). The one rule that makes it robust: a failed tool
call is not an error - it's a result. It goes back into the conversation like any
other, and the model gets to react to it. Only an unexpected exception ends a run.

Every step emits a numbered event on the bus, so a UI can follow along live and a
dropped client can catch up later.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .conversation import (
    Reasoning,
    Role,
    ToolCall,
    ToolCallStatus,
    Usage,
    assistant_message,
    tool_result_message,
)
from .events import EventType
from .hooks import HookContext, HookManager, HookPoint
from .models.base import StreamType, mark_cacheable_prefix, stable_prefix_fingerprint
from .monitoring import Monitoring
from .tools.base import ToolResult, native_schema

#: How many tools may run at once. A model that asks for thirty reads shouldn't
#: open thirty file handles and thirty MCP calls in the same instant. This is the
#: default; a run can raise or lower it through `Limits.max_parallel_tools`.
MAX_PARALLEL_TOOLS = 8

#: How long to wait before the first retry. Each later one waits twice as long,
#: so three retries pause for about 1s, 2s and 4s. Short enough that a rate limit
#: clears without the user thinking the agent hung. The default behind
#: `Limits.retry_first_delay_seconds`.
_RETRY_FIRST_DELAY_SECONDS = 1.0

#: What a tool call that never ran because the user stopped the agent says for
#: itself. It goes in the conversation like any other result, so the next turn can
#: see that the work was asked for and didn't happen.
_STOPPED_MESSAGE = "Not run: you stopped the agent before this call started."


# ---------------------------------------------------------------------------
# The knobs and the outcome
# ---------------------------------------------------------------------------
@dataclass
class Limits:
    """How far a single run may go, and the shape of how it gets there.

    Four of these are *ceilings*: they stop a run cleanly - a partial result and a
    named reason, never an error. `max_turns` and `wall_clock_seconds` bound how
    long it runs; `max_cost_usd` and `max_total_tokens` bound what it spends,
    checked between turns against the usage the provider has reported so far. A
    ceiling of 0 means "no ceiling" for that dimension, so leaving one unset never
    stops a run. Because the check falls between turns, a budget ceiling can't
    un-spend a call already made or price the next one before it runs: the run
    stops *before* the next call and hands back what it had, which is the honest
    thing a budget can promise.

    The other three don't stop a run, they tune how it behaves. `max_retries` keeps
    a run alive: how many extra times a model call may be re-sent after a failure
    that looks temporary, waiting `retry_first_delay_seconds` before the first
    re-send and twice as long each time after. `max_parallel_tools` caps how many
    tool calls run at once within a turn. `helper_max_turns` is the ceiling on a
    delegated helper's own run; 0 means "leave it to the delegate tool's default".

    `reasoning_effort` is the thinking-budget knob for this run: "" means "use the
    agent's default (`AgentConfig.reasoning_effort`), or the provider's if that's
    empty too", and "low"/"medium"/"high" ask a reasoning model to think more or
    less. It doesn't stop or bound a run - it steers how the model spends a turn -
    but it lives here because it's a per-run choice, and set here it overrides the
    agent's default. A model with no thinking mode ignores it.

    `plan_mode` turns this run into a read-only investigation: the model is shown
    only the tools that can't change anything, and any mutating call it makes anyway
    is refused, so it can look and propose but not act. It's a per-run mode, not an
    agent trait - you enter it for a task, get a plan, and "approve" by making the
    next run without it (full tools again). Off by default, so an ordinary run is
    untouched. See `_tool_schemas` for the soft half and `PlanModePolicy` for the
    hard half of the gate.

    They all live here because they're per-run knobs: two runs of the same agent
    can be given different envelopes without touching the agent. The field defaults
    fall back to the module constants above, so leaving them unset changes nothing.
    """

    max_turns: int = 10
    wall_clock_seconds: float = 0.0  # 0 means no time limit
    max_cost_usd: float = 0.0  # 0 means no cost ceiling (USD, priced on the run's model)
    max_total_tokens: int = 0  # 0 means no token ceiling (input + output, summed)
    max_retries: int = 3
    max_parallel_tools: int = MAX_PARALLEL_TOOLS
    retry_first_delay_seconds: float = _RETRY_FIRST_DELAY_SECONDS
    helper_max_turns: int = 0  # 0 = use the delegate tool's own default
    reasoning_effort: str = ""  # "" = fall back to the agent's default, then the provider's
    plan_mode: bool = False  # True = read-only investigation; mutating tools refused


class Cancellation:
    """A flag the caller flips to ask a run to stop at the next safe point."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class RetryCoordinator:
    """Staggers retry backoff across everything that shares one provider.

    Within a single run only one model call is ever in flight - turns are
    sequential - so here this does nothing. It earns its place the moment several
    runs share a provider: two HTTP sessions on the same server, or a run and the
    helpers it spawned. When the provider rate-limits all of them at once, each
    would otherwise sleep the same second and then retry in the same instant,
    reproducing the 429 N ways and never letting the limit clear. Serialising the
    *sleeps* through one lock means the retries fan out across the delay instead of
    landing together - the cheapest honest form of "don't retry N-ways at once".

    It is deliberately not a rate limiter: it adds no delay of its own and never
    blocks a first attempt, only the waiting between retries. One coordinator is
    built per runtime and shared into every loop, including helper loops, so the
    whole tree backs off as one.
    """

    def __init__(self) -> None:
        # Safe to build outside a running loop on 3.10+: the lock binds to the
        # loop lazily on first use, which is always inside `AgentLoop.run`.
        self._lock = asyncio.Lock()

    async def backoff(self, delay: float) -> None:
        """Sleep `delay`, but never at the same time as another backoff."""
        async with self._lock:
            await asyncio.sleep(max(0.0, delay))


class RunStatus(str, Enum):
    """How a run ended."""

    RUNNING = "running"
    FINISHED = "finished"            # the model gave a final answer
    LIMIT_REACHED = "limit_reached"  # ran out of turns or time
    CANCELLED = "cancelled"          # the caller stopped it
    ERROR = "error"                  # something unexpected threw


@dataclass
class RunContext:
    """Everything one run carries around. Tools are handed this."""

    session: Any
    run_id: str
    config: Any
    limits: Limits
    cancellation: Cancellation
    turn: int = 0
    started_at: float = field(default_factory=time.monotonic)
    retries: int = 0  # model calls re-sent after a temporary failure
    fallbacks: int = 0  # times the run failed over to an alternate model
    # The front of the last request, and whether it has ever moved. Only used to
    # explain a prompt cache that never hits - see `_check_prefix`.
    prefix_fingerprint: str = ""
    prefix_changed: bool = False


@dataclass
class RunResult:
    """The summary a caller gets back when a run ends.

    The receipt fields (duration, cost, tokens via `usage`, retries) say how long
    it took, what those tokens cost on the model that was actually used, and how
    many model calls had to be re-sent. Without them there is no way to tell whether
    a change made the agent cheaper or just made it feel faster.

    `stop_reason` names *which* ceiling ended a `LIMIT_REACHED` run - "max_turns",
    "wall_clock", "max_cost" or "max_tokens" - and is empty for every other ending.
    The status says a limit was hit; this says which one, so a caller doesn't have
    to guess whether to widen the budget or the turn cap.

    `fallbacks` counts how many times the run had to leave the model it started on
    for an alternate - zero for the ordinary single-model run. Paired with `model`
    (the one it *ended* on) it says whether the primary carried the whole run.
    """

    run_id: str
    status: RunStatus
    turns: int = 0
    final_text: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    model: str = ""
    retries: int = 0
    fallbacks: int = 0
    stop_reason: str = ""
    trace_id: str = ""


@dataclass
class RunRecord:
    """The durable record of a run, saved to the database."""

    run_id: str
    session_id: str
    status: str
    final_text: str = ""
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    cached_tokens: int = 0
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    model: str = ""
    retries: int = 0
    reasoning_tokens: int = 0
    trace_id: str = ""


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
class AgentLoop:
    """Runs the think/act/observe cycle for one conversation."""

    def __init__(
        self,
        models: Any,
        tools: Any,
        tool_manager: Any,
        context_manager: Any,
        event_bus: Any,
        database: Any,
        monitoring: Any = None,
        retry_coordinator: RetryCoordinator | None = None,
        hooks: HookManager | None = None,
    ) -> None:
        self._models = models
        self._tools = tools
        self._tool_manager = tool_manager
        self._context = context_manager
        self._bus = event_bus
        self._db = database
        # Always have something to open spans on. A caller who passes nothing gets
        # in-memory recording and no files, which is safe anywhere.
        self._monitoring = monitoring if monitoring is not None else Monitoring()
        # Shared so retries across sibling loops (a run and its helpers, or several
        # runs on one server) don't all back off in the same instant. A loop built
        # on its own still gets a private one, so it works standalone in a test.
        self._retry_coordinator = retry_coordinator or RetryCoordinator()
        # Lifecycle hooks. An empty manager is a no-op at every point: the call
        # sites guard on `has(...)`, so a loop built without hooks behaves exactly
        # as it did before hooks existed. A runtime passes its shared manager here
        # (and to its helpers) so a hook the user registered fires everywhere.
        self._hooks = hooks if hooks is not None else HookManager()

    async def run(self, conversation: Any, context: RunContext) -> RunResult:
        """Drive the conversation to a stopping point and return the outcome."""
        session = context.session
        run_id = context.run_id
        limits = context.limits
        cancel = context.cancellation

        status = RunStatus.RUNNING
        error = ""
        # Provider-reported usage, summed across turns. The receipt and its cost
        # are built from this alone - never from the local estimator, which is a
        # ~4-chars-per-token guess kept only for the compaction budget.
        total = Usage()
        # The last turn's real prompt-token count, fed back into the compaction
        # decision so it runs on the provider's truth rather than that estimate.
        # Zero until the first usage report arrives.
        last_input_tokens = 0
        final_text = ""
        # Which ceiling ended the run, when one did. Stays "" for a normal finish,
        # a cancel or an error - only a LIMIT_REACHED run names its reason.
        stop_reason = ""
        turn = 0
        # Bound before the try, because model resolution happens *inside* it: if
        # the configured name is wrong, these still have to be safe to read on the
        # way out. `model` tracks the *currently active* model, which a fallover can
        # advance mid-run; `total_cost` is accumulated per turn against whichever
        # model actually served that turn, so a run that switches models is priced
        # honestly. For a single-model run this equals pricing the whole total at
        # the one model, exactly as before, because `cost_of` is linear.
        model = None
        total_cost = 0.0

        # The root Langfuse v4 observation represents this complete agent turn.
        # Keep the user request on it so evaluations and the tracing table have a
        # useful root input instead of only opaque child generations.
        goal = ""
        for message in reversed(getattr(conversation, "messages", [])):
            if getattr(message, "role", None) is Role.USER:
                goal = message.text()
                break
        run_attributes = {
            "input": goal,
            "session_id": session.id,
            "trace_name": f"native-run:{session.agent}",
            "tags": ["agent-native", f"agent:{context.config.name}"],
        }
        environment = os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
        release = os.getenv("LANGFUSE_RELEASE")
        if environment:
            run_attributes["environment"] = environment
        if release:
            run_attributes["version"] = release

        try:
            with self._monitoring.run_span(run_id, **run_attributes) as run_trace:
                # Inside the try: a bad model/provider name becomes a clean ERROR
                # result, not an exception that escapes with no record. The primary
                # (config.model) is resolved strictly - a bad name raises here, as
                # it always has - and any registered fallbacks are appended after
                # it, with a broken fallback skipped rather than made fatal.
                candidates = self._resolve_candidates(context.config)
                active = 0
                model, provider = candidates[active]

                # Resume preamble. On a fresh message the conversation ends with the
                # user's turn, so this does nothing and the loop runs as usual. On a
                # reattach it finishes any tool call that was in flight - never one
                # that already has a saved result, so no side effect happens twice -
                # and, if the model had already given its final answer before the
                # interruption, hands it back here so the model isn't asked again.
                finished_on_resume = await self._prepare_resume(conversation, context)
                if finished_on_resume is not None:
                    status = finished_on_resume.status
                    final_text = finished_on_resume.final_text

                while finished_on_resume is None:
                    if cancel.cancelled:
                        status = RunStatus.CANCELLED
                        break
                    if turn >= limits.max_turns:
                        status = RunStatus.LIMIT_REACHED
                        stop_reason = "max_turns"
                        break
                    if limits.wall_clock_seconds and (
                        time.monotonic() - context.started_at
                    ) > limits.wall_clock_seconds:
                        status = RunStatus.LIMIT_REACHED
                        stop_reason = "wall_clock"
                        break
                    # Budget ceilings, checked between turns on the usage reported
                    # so far. The check is here, before the next call, because a
                    # ceiling can't un-spend a call already made or price one not
                    # yet made - so it stops the run and returns what it had.
                    # `total_cost` is accumulated per turn against the model that
                    # served it, so this stays honest even after a fallover.
                    if limits.max_cost_usd and total_cost >= limits.max_cost_usd:
                        status = RunStatus.LIMIT_REACHED
                        stop_reason = "max_cost"
                        break
                    if limits.max_total_tokens and (
                        total.input_tokens + total.output_tokens
                    ) >= limits.max_total_tokens:
                        status = RunStatus.LIMIT_REACHED
                        stop_reason = "max_tokens"
                        break

                    turn += 1
                    context.turn = turn
                    with self._monitoring.turn_span(turn) as turn_trace:
                        await self._bus.emit(
                            session.id, EventType.TURN_STARTED, {"turn": turn}, run_id
                        )

                        # Keep the conversation inside the active model's window first.
                        if self._context.needs_compaction(
                            conversation, model, last_input_tokens
                        ):
                            await self._compact(conversation, model, provider)

                        # run_turn may fail over to a later candidate when the
                        # active model keeps failing; it returns the index that
                        # actually answered, so the run switches to that model for
                        # the next window check, the pricing below, and the receipt.
                        assistant_msg, _finish, active = await self.run_turn(
                            conversation, candidates, active, context
                        )
                        model, provider = candidates[active]
                        conversation.add(assistant_msg)
                        await self._db.save_message(assistant_msg)
                        _add_usage(total, assistant_msg.usage)
                        # Price this turn against the model that served it, now, so a
                        # run that changed models is summed across both rather than
                        # priced at whichever one happened to be active at the end.
                        if assistant_msg.usage:
                            turn_cost = model.cost_of(
                                assistant_msg.usage.input_tokens,
                                assistant_msg.usage.output_tokens,
                            )
                            total_cost += turn_cost
                            turn_trace.set(
                                model=model.model_id,
                                provider=model.provider,
                                output=assistant_msg.text(),
                                input_tokens=assistant_msg.usage.input_tokens,
                                output_tokens=assistant_msg.usage.output_tokens,
                                total_tokens=(
                                    assistant_msg.usage.input_tokens
                                    + assistant_msg.usage.output_tokens
                                ),
                                cached_tokens=assistant_msg.usage.cached_tokens,
                                cost=turn_cost,
                            )
                        # Remember this turn's real prompt size for the next
                        # compaction check; ignore a turn the provider didn't
                        # report (it leaves the last good number in place).
                        if assistant_msg.usage and assistant_msg.usage.input_tokens:
                            last_input_tokens = assistant_msg.usage.input_tokens
                        await self._bus.emit(
                            session.id,
                            EventType.MESSAGE_ADDED,
                            {
                                "id": assistant_msg.id,
                                "role": "assistant",
                                "has_tool_calls": assistant_msg.has_tool_calls(),
                            },
                            run_id,
                        )

                        # A stop that arrived while the reply was streaming. The
                        # partial answer is kept and saved above - it's what the
                        # user saw - but nothing is run on the way out: a stop
                        # that still edited a file would not be a stop.
                        if cancel.cancelled:
                            final_text = assistant_msg.text()
                            status = RunStatus.CANCELLED
                            break

                        if assistant_msg.has_tool_calls():
                            await self.run_tools(assistant_msg, conversation, context)
                            continue

                        final_text = assistant_msg.text()
                        status = RunStatus.FINISHED
                        break

                run_trace.set(
                    output=final_text,
                    status=status.value,
                    turns=turn,
                    # Only interesting when it's True: the prompt cache can't hit
                    # on a request whose front keeps moving.
                    prefix_changed=context.prefix_changed,
                    # Names the ceiling on a LIMIT_REACHED run; "" otherwise, so the
                    # trace explains a stop instead of leaving it to be inferred.
                    stop_reason=stop_reason,
                )
        except Exception as exc:  # a thrown error ends the run, but is reported cleanly
            status = RunStatus.ERROR
            error = f"{type(exc).__name__}: {exc}"
            await self._bus.emit(session.id, EventType.ERROR, {"error": error}, run_id)

        trace_ids = getattr(self._monitoring, "langfuse_trace_ids", None)
        trace_id = trace_ids.get(run_id, "") if isinstance(trace_ids, Mapping) else ""
        result = RunResult(
            run_id=run_id,
            status=status,
            turns=turn,
            final_text=final_text,
            usage=total,
            error=error,
            duration_seconds=round(time.monotonic() - context.started_at, 3),
            # Accumulated per turn against the model that served each one, so a run
            # that failed over is priced across every model it used rather than
            # pricing the whole total at whichever one was active last. A run that
            # died before it resolved a model never accumulated anything, so this is
            # 0.0 - the same honest zero an unpriced model reports.
            cost_usd=round(total_cost, 6),
            # The model the run *ended* on: the one and only model for an ordinary
            # run, the fallback for one that failed over. `fallbacks` and the
            # MODEL_FALLBACK events carry the rest of the story.
            model=model.model_id if model is not None else context.config.model,
            retries=context.retries,
            fallbacks=context.fallbacks,
            stop_reason=stop_reason,
            trace_id=trace_id,
        )
        try:
            await self._finish(session, result)
            await self._run_stop(session, run_id, result)
        finally:
            # The API process is long-lived, so waiting for shutdown would keep
            # completed native traces buffered. Flush after each run just like the
            # LangGraph track; Monitoring makes this best-effort and optional for
            # custom monitoring implementations.
            flush = getattr(self._monitoring, "flush", None)
            if callable(flush):
                flush()
        return result

    async def _run_stop(self, session: Any, run_id: str, result: RunResult) -> None:
        """Fire the stop hook once a run reaches its stopping point, whatever the outcome.

        A top-level conversation fires RUN_STOP; a subagent - its run id carries the
        helper separator - fires SUBAGENT_STOP, so a hook can tell "the whole thing
        finished" from "one helper finished", the same distinction the event stream
        draws with `is_helper_run`. The loop is the one place every run, parent or
        helper, actually stops (a helper never reaches the service), so the stop
        points live here rather than in the service. With neither hook registered
        nothing is imported and nothing runs, so an ordinary run is untouched.
        """
        if not (self._hooks.has(HookPoint.RUN_STOP) or self._hooks.has(HookPoint.SUBAGENT_STOP)):
            return
        from .tools.subagent import (
            is_helper_run,  # lazy: loop <-> subagent import cycle
        )

        point = HookPoint.SUBAGENT_STOP if is_helper_run(run_id) else HookPoint.RUN_STOP
        await self._hooks.dispatch(
            HookContext(
                point=point,
                session_id=session.id,
                run_id=run_id,
                text=result.final_text,
                status=result.status.value,
            )
        )

    # -- keeping the conversation inside the window -------------------------
    def _check_prefix(self, wire: list, tool_schemas: list, context: RunContext) -> None:
        """Notice if the unchanging front of the request changed. Costs a hash.

        A provider only serves a cached prefix when the request starts with the
        same bytes as the last one, so a system prompt that moves - a date, a
        re-ordered tool list, a summary merged into the wrong place - turns the
        discount off silently. Recording it on the run means the answer to "why is
        `cached_tokens` always zero" is in the trace instead of being guessed at.
        """
        fingerprint = stable_prefix_fingerprint(wire, tool_schemas)
        if context.prefix_fingerprint and context.prefix_fingerprint != fingerprint:
            context.prefix_changed = True
        context.prefix_fingerprint = fingerprint

    async def _compact(self, conversation: Any, model: Any, provider: Any) -> None:
        """Fold the older messages away, letting the model write the summary.

        The model-written summary is preferred and the template is the fallback,
        which `compact_with_model` handles by itself. The `getattr` is for a
        context manager that predates it - a caller can pass its own, and it
        shouldn't have to grow a method to keep working.
        """
        write = getattr(self._context, "compact_with_model", None)
        if write is None:
            self._context.compact(conversation, model)
            return
        await write(conversation, model, provider)

    # -- which tools the model may see this turn ----------------------------
    def _tool_schemas(self, context: RunContext) -> list:
        """The tools to offer the model this turn, in native-tool-calling shape.

        Normally every tool the agent's `allowed_tools` permits. In plan mode the
        list is narrowed to the read-only ones: the model is shown only tools that
        cannot change anything, so it investigates and proposes rather than acting,
        and never has a mutating tool to reach for in the first place. This is the
        *soft* half of the plan-mode gate - guiding by what's visible.
        `PlanModePolicy` is the *hard* half: a mutating call the model makes anyway
        (a name it hallucinated, a tool it saw in an earlier non-plan turn) is
        refused at authorization, so plan mode is a guarantee and not just a hint.
        """
        tools = self._tools.get_available_tools(context.config)
        if getattr(context.limits, "plan_mode", False):
            tools = [t for t in tools if t.definition.permissions.read_only]
        return [native_schema(t.definition) for t in tools]

    # -- the models this run may use, primary first -------------------------
    def _resolve_candidates(self, config: Any) -> list:
        """The ordered `(model, provider)` pairs this run may use to answer a turn.

        The first is the primary (`config.model`), resolved exactly as a run always
        has: its lookup is *allowed to raise* here, so a main model that isn't
        registered - or whose provider isn't - surfaces as the clean ERROR it always
        did, rather than being quietly papered over by a fallback the user never
        meant to run first. Each entry in `config.fallback_models` is then appended
        in order, but best-effort: a fallback name that isn't registered, or whose
        provider isn't, is *skipped* rather than raised, because a broken fallback
        must never turn a healthy primary run into an error - listing an optional
        second provider has to be safe. A name that repeats the primary or an
        earlier fallback is dropped, so the chain never wastes an attempt re-trying
        the same dead model under a second name.
        """
        primary = self._models.get(config.model)
        candidates = [(primary, self._models.get_provider(primary))]
        seen = {config.model}
        for name in getattr(config, "fallback_models", None) or []:
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                model = self._models.get(name)
                provider = self._models.get_provider(model)
            except Exception:
                continue
            candidates.append((model, provider))
        return candidates

    # -- one turn: ask the model, fail over if the active one is exhausted ---
    async def run_turn(
        self, conversation: Any, candidates: list, active: int, context: RunContext
    ):
        """Stream one model reply, failing over to an alternate model when needed.

        Two layers of resilience compose here, cheapest first. The inner one
        (`_attempt_turn_with_retries`) re-sends the *same* request to the *same*
        model when it fails in a way a moment's wait might cure - a rate limit, a
        timeout, a dropped connection. Only when that is exhausted does the outer
        loop here reach for the next model in the run's fallback chain: a different
        model, and usually a different provider, so a provider that is down or is
        rate-limiting the whole account doesn't end the run when another could
        finish it.

        Both layers obey the same one rule, and for the same reason: an attempt is
        only safe to abandon while it has produced nothing. Once a word has streamed
        to the user, neither a retry nor a fallover can start over without printing
        it twice - so a failure past that point is surfaced, not papered over. That
        is why `progress` is created once here and shared through every attempt of
        the turn, retries and fallovers alike.

        The fallover is *sticky*: this returns the index that actually answered, and
        the run stays on it. A provider that has just failed shouldn't be paid a full
        retry-and-timeout cycle at the top of every remaining turn only to fail the
        same way; once the run has moved on, it stays moved on.
        """
        # Shared across every attempt of this turn - retries and fallovers both -
        # because once anything has reached the user no later attempt is safe.
        progress = {"streamed": False}
        while True:
            model, provider = candidates[active]
            try:
                assistant_msg, finish_reason = await self._attempt_turn_with_retries(
                    conversation, model, provider, context, progress
                )
                return assistant_msg, finish_reason, active
            except Exception as exc:
                # Fall over only when a different model could plausibly do better:
                # the failure looks temporary, nothing has streamed (so a fresh
                # start is safe), the run wasn't cancelled, and there is another
                # model left to try. Otherwise this is the same raise the retry
                # layer would have done - the run's outer handler makes it an ERROR.
                if (
                    progress["streamed"]
                    or active + 1 >= len(candidates)
                    or context.cancellation.cancelled
                    or not _is_temporary(exc)
                ):
                    raise
                next_model, _next_provider = candidates[active + 1]
                await self._bus.emit(
                    context.session.id,
                    EventType.MODEL_FALLBACK,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "from_model": model.model_id,
                        "to_model": next_model.model_id,
                        "attempt": context.fallbacks + 1,
                    },
                    context.run_id,
                )
                active += 1
                context.fallbacks += 1

    async def _attempt_turn_with_retries(
        self, conversation: Any, model: Any, provider: Any, context: RunContext, progress: dict
    ):
        """Stream one reply from one model, re-sending the request if it fails temporarily.

        A model call fails for two very different reasons. Some failures are the
        provider having a bad moment - a rate limit, a timeout, a dropped
        connection - and the very same request a second later just works. Others
        are the request itself being wrong - a bad key, a model name that doesn't
        exist - and re-sending only wastes the user's time before showing them the
        same message. So the first kind is retried with a growing pause and the
        second kind fails at once. Anything unrecognised is treated as the second
        kind: failing fast on a surprise is better than a silent triple-charge.

        One more rule, and it's the reason this wrapper exists instead of a plain
        `for attempt in ...`: a retry is only safe while the failed attempt has
        produced nothing. Once a word of the answer has been streamed to the
        screen, starting over would print it twice, so a failure after that point
        is reported rather than papered over. `progress` is owned by the caller
        (`run_turn`) and shared across models, so a model that streamed nothing here
        can still be failed over, but one that streamed cannot.

        A failure this layer can't cure is re-raised, not swallowed: it's `run_turn`
        that decides whether another model is worth trying.
        """
        attempt = 0
        while True:
            try:
                return await self._stream_turn(conversation, model, provider, context, progress)
            except Exception as exc:
                attempt += 1
                if (
                    progress["streamed"]
                    or attempt > max(0, context.limits.max_retries)
                    or context.cancellation.cancelled
                    or not _is_temporary(exc)
                ):
                    raise
                context.retries += 1
                delay = context.limits.retry_first_delay_seconds * (2 ** (attempt - 1))
                await self._bus.emit(
                    context.session.id,
                    EventType.ERROR,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "temporary": True,
                        "attempt": attempt,
                        "retrying_in_seconds": round(delay, 2),
                    },
                    context.run_id,
                )
                # Through the coordinator, not a bare sleep: if sibling loops are
                # backing off too, their sleeps are serialised so the retries don't
                # all fire at once and re-trip the same rate limit.
                await self._retry_coordinator.backoff(delay)

    async def _stream_turn(
        self,
        conversation: Any,
        model: Any,
        provider: Any,
        context: RunContext,
        progress: dict | None = None,
    ):
        """One attempt at a turn: stream the reply and build it into a Message.

        Cancellation is checked here, between chunks, and not only between turns.
        A turn is where all the waiting happens - a long answer streams for many
        seconds - so a stop that's only noticed at the turn boundary arrives after
        the thing the user wanted stopped has already finished. The useful version
        is steering: the correction lands on the next action rather than the next
        conversation.

        Stopping mid-stream **keeps the text and drops the tool calls.** Text is
        whole as far as it got and the user has already read it on screen. A tool
        call is not: its arguments arrive as a stream of JSON, and half of a JSON
        object is not a smaller request, it's a broken one. Worse, an assistant
        message carrying tool calls is a promise - the wire format requires a
        result for every call - so keeping a half-formed call would leave the
        conversation permanently unable to be sent back to a model.
        """
        session = context.session
        run_id = context.run_id
        progress = progress if progress is not None else {"streamed": False}

        wire = conversation.render(model.tool_format)
        tool_schemas = self._tool_schemas(context)
        # In plan mode, tell the model in the request (never in a stored message)
        # that it's confined to investigating and proposing. The narrowed tool list
        # already stops it reaching for a mutating tool; this stops it being
        # surprised by that, and asks it to write the plan down instead.
        if context.limits.plan_mode:
            wire = _with_plan_mode_banner(wire)
        # The front of this request is the same as last turn's, or it isn't and
        # nothing was going to be cached. Check, then mark (a no-op for every
        # provider here, which is the honest state of it - see models/base.py).
        self._check_prefix(wire, tool_schemas, context)
        wire = mark_cacheable_prefix(wire, model.cache_marker)

        text_parts: list = []
        reasoning_parts: list = []
        fragments: dict = {}  # index -> {id, name, arguments}
        order: list = []      # indices in the order they first appeared
        usage = Usage()
        finish_reason = "stop"
        cancelled = False

        optional_kwargs = {
            "top_p": getattr(context.config, "top_p", 1.0),
            "max_tokens": getattr(context.config, "max_output_tokens", None),
            "timeout_seconds": getattr(context.config, "timeout_seconds", 60),
            **_effort_kwargs(context),
        }
        try:
            parameters = inspect.signature(provider.stream).parameters
            if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
                optional_kwargs = {
                    key: value for key, value in optional_kwargs.items() if key in parameters
                }
        except (TypeError, ValueError):
            pass
        stream = provider.stream(
            wire,
            tool_schemas,
            model,
            context.config.temperature,
            **optional_kwargs,
        )
        try:
            async for event in stream:
                if context.cancellation.cancelled:
                    cancelled = True
                    finish_reason = "cancelled"
                    break
                if event.type == StreamType.TEXT:
                    chunk = event.data.get("text", "")
                    if chunk:
                        text_parts.append(chunk)
                        progress["streamed"] = True  # past here, a retry would duplicate
                        await self._bus.emit(session.id, EventType.ASSISTANT_DELTA, {"text": chunk}, run_id)
                elif event.type == StreamType.REASONING:
                    chunk = event.data.get("text", "")
                    if chunk:
                        reasoning_parts.append(chunk)
                        progress["streamed"] = True
                        await self._bus.emit(session.id, EventType.REASONING_DELTA, {"text": chunk}, run_id)
                elif event.type == StreamType.TOOL_CALL:
                    _merge_fragment(fragments, order, event.data)
                elif event.type == StreamType.USAGE:
                    usage.input_tokens = event.data.get("input_tokens", usage.input_tokens)
                    usage.output_tokens = event.data.get("output_tokens", usage.output_tokens)
                    usage.cached_tokens = event.data.get("cached_tokens", usage.cached_tokens)
                    usage.reasoning_tokens = event.data.get(
                        "reasoning_tokens", usage.reasoning_tokens
                    )
                elif event.type == StreamType.DONE:
                    finish_reason = event.data.get("finish_reason", finish_reason)
        finally:
            # Leaving a stream open holds the HTTP response open with it. On the
            # normal path this is already exhausted and closing does nothing; on
            # the cancelled path it's the whole point.
            await _close_stream(stream)

        # Dropped, not kept: see the docstring. The text that did arrive stays.
        tool_calls = [] if cancelled else _build_tool_calls(fragments, order)
        assistant_msg = assistant_message(
            session_id=session.id,
            text="".join(text_parts),
            tool_calls=tool_calls or None,
            model=model.model_id,
            usage=usage,
        )
        if reasoning_parts:
            # Hidden from the wire (render ignores Reasoning); kept for the record.
            assistant_msg.parts.insert(0, Reasoning("".join(reasoning_parts)))
        return assistant_msg, finish_reason

    # -- run every tool the model asked for, add each result back -----------
    async def run_tools(self, assistant_msg: Any, conversation: Any, context: RunContext) -> None:
        """Run the turn's tools and append their results in the order they were asked for.

        Three rules, in the order they matter:

        **Permission is asked for one call at a time, up front.** Three tools all
        prompting at once would be unreadable on a terminal, and a user answering
        three interleaved questions will approve the wrong one. So every call is
        authorized in the order the model made it, before any of them runs.

        **Only reads run together.** A burst of file reads is the case worth
        parallelising and it's also the safe one. Two writes to the same file at
        the same time is a race, and a read that overtakes the write before it
        would see the old contents - so anything that isn't read-only runs on its
        own, and the reads around it keep their place in the queue. That's why the
        calls are grouped into consecutive runs rather than split into "all reads"
        and "the rest": grouping preserves what the model meant by the order.

        **Results go back in the model's order, and are written down as each group
        finishes.** Whatever finishes first, each result is paired with its own
        call, so the conversation reads the same way every time - which is what
        makes a run reproducible. Saving per group rather than all at the very end
        is also what makes a run *resumable*: see `_run_calls`.
        """
        await self._run_calls(assistant_msg.tool_calls(), conversation, context)

    async def _run_calls(self, calls: list, conversation: Any, context: RunContext) -> None:
        """Authorize, run and persist a set of tool calls, in the model's order.

        Persisting each call's result the moment its group finishes - rather than
        all of them together at the very end - is what lets a resumed run avoid
        doing a side effect twice. A crash leaves every call that ran with its
        result already saved, so resume skips exactly those and re-runs only the
        rest. The two kinds of call make this safe from both ends: a write runs
        alone and is durable before the next call starts, and the only calls that
        ever share a group are read-only, where running one a second time on resume
        changes nothing.

        It takes a plain list of calls, not the assistant message, because resume
        hands it only the calls that never got a result - the same machinery,
        pointed at the unfinished tail.
        """
        if not calls:
            return

        # Phase one: permission, in order, one at a time.
        refusals: dict = {}
        for call in calls:
            refusals[call.id] = await self._tool_manager.authorize(call, context)

        # Phase two: run each group, then write its results down before the next
        # group starts. Consecutive read-only calls go together; anything else runs
        # alone. A refusal costs nothing to "run", so it rides along. A stop is
        # checked between groups too: a turn can ask for a dozen calls, so "stopped"
        # has to mean the rest don't happen - but each one still gets a result
        # saying why, because the wire format needs a result for every call.
        limit = asyncio.Semaphore(max(1, context.limits.max_parallel_tools))
        for group in self._group_calls(calls, refusals):
            if context.cancellation.cancelled:
                for call in group:
                    call.status = ToolCallStatus.ERROR
                    await self._persist_tool_result(
                        call, ToolResult(False, error=_STOPPED_MESSAGE), conversation, context
                    )
                continue
            if len(group) == 1:
                call = group[0]
                result = await self._run_one_call(call, refusals[call.id], context, None)
                await self._persist_tool_result(call, result, conversation, context)
            else:
                done = await asyncio.gather(
                    *(self._run_one_call(c, refusals[c.id], context, limit) for c in group)
                )
                for call, result in zip(group, done):
                    await self._persist_tool_result(call, result, conversation, context)

    async def _persist_tool_result(
        self, call: Any, result: Any, conversation: Any, context: RunContext
    ) -> None:
        """Add one finished tool call to the conversation, save it, and announce it.

        Called the instant a call's group finishes, so the result is durable before
        anything else runs - which is what a resume relies on to know the call is
        done. Results are added in the model's order because the groups, and the
        calls within them, are walked in that order.
        """
        finished = ToolCall(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            status=call.status,
            output=result.output,
            error=result.error,
        )
        tool_msg = tool_result_message(context.session.id, finished)
        conversation.add(tool_msg)
        await self._db.save_message(tool_msg)
        await self._bus.emit(
            context.session.id,
            EventType.MESSAGE_ADDED,
            {"id": tool_msg.id, "role": "tool", "call_id": call.id},
            context.run_id,
        )

    # -- picking a resumed run back up where it stopped ---------------------
    async def _prepare_resume(self, conversation: Any, context: RunContext):
        """Settle any unfinished business before the loop starts, on a resume.

        A run reaches the loop in one of a few states, and only the last two are a
        resume at all - the first two are what every fresh `send_message` looks
        like, so the common path falls straight through and costs one look at the
        last message:

          * ends with a **user or system** message → nothing to settle, run normally;
          * ends with an **assistant message that has no tool calls** → the model
            had already given its final answer before the process died. Returning a
            FINISHED result hands that answer back so `run` skips the loop and just
            re-emits the finish, rather than paying for a turn to regenerate it;
          * ends with an **assistant message whose tool calls aren't all answered**,
            or with **some but not all** of those results → the crash landed
            mid-tool-batch. Run only the calls that never got a result and let the
            loop carry on; the ones already saved are left exactly as they were,
            which is the no-duplicate-side-effects guarantee;
          * ends with **every call answered** → fall through; the loop's next turn
            is the model reading those results, same as if it had never stopped.

        Returning a RunResult means "don't enter the loop"; returning None means
        "carry on". Only the finished-answer case returns a result.
        """
        messages = conversation.messages
        if not messages:
            return None
        last = messages[-1]
        if last.role in (Role.USER, Role.SYSTEM):
            return None
        if last.role == Role.ASSISTANT and not last.tool_calls():
            # The final answer was saved but the run never got to record itself.
            # Hand it back so `run` falls through to `_finish` without a model call.
            return RunResult(
                run_id=context.run_id,
                status=RunStatus.FINISHED,
                turns=0,
                final_text=last.text(),
            )

        assistant = self._last_open_assistant(messages)
        if assistant is not None:
            answered = _answered_call_ids(messages)
            pending = [c for c in assistant.tool_calls() if c.id not in answered]
            if pending:
                await self._run_calls(pending, conversation, context)
        return None

    def _last_open_assistant(self, messages: list):
        """The assistant message whose tool calls the tail is still answering.

        Walk back from the end: the first assistant-with-tool-calls we meet is the
        one in flight, unless we cross a later user message first - which would mean
        those calls belong to a turn that already completed and a new turn has since
        begun, so there's nothing open to finish.
        """
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT and msg.tool_calls():
                return msg
            if msg.role == Role.USER:
                return None
        return None

    async def _run_one_call(
        self,
        call: Any,
        refusal: Any,
        context: RunContext,
        limit: asyncio.Semaphore | None,
    ) -> Any:
        """One tool call, from its start event to its finish event."""
        session = context.session
        run_id = context.run_id

        tool = self._tools.find(call.name)
        preview = tool.preview(call.arguments) if tool is not None else call.name
        # Whether this tool only reads - handed to the tool-point hooks so one that
        # cares only about mutations (the auto-checkpointer) can skip a read cheaply.
        read_only = bool(tool is not None and tool.definition.permissions.read_only)
        await self._bus.emit(
            session.id,
            EventType.TOOL_STARTED,
            {"call_id": call.id, "name": call.name, "arguments": call.arguments, "preview": preview},
            run_id,
        )
        call.status = ToolCallStatus.RUNNING

        if refusal is not None:
            result = refusal  # never authorized, so never run
        else:
            result = await self._pre_tool_veto(call, context, read_only)  # a hook may block it
            if result is None:
                with self._monitoring.tool_span(
                    call.name,
                    input={"name": call.name, "arguments": call.arguments},
                ) as tool_trace:
                    if limit is None:
                        result = await self._tool_manager.run_authorized(call, context)
                    else:
                        async with limit:
                            result = await self._tool_manager.run_authorized(call, context)
                    tool_trace.set(
                        success=result.success,
                        output=result.output,
                        error=result.error,
                        truncated=result.truncated,
                    )
                await self._post_tool(call, context, result, read_only)  # observe what it returned

        call.status = ToolCallStatus.SUCCESS if result.success else ToolCallStatus.ERROR
        call.output = result.output
        call.error = result.error

        await self._bus.emit(
            session.id,
            EventType.TOOL_FINISHED,
            {
                "call_id": call.id,
                "name": call.name,
                "success": result.success,
                "output": (result.output or "")[:500],
                "error": result.error,
                "truncated": result.truncated,
            },
            run_id,
        )
        return result

    # -- lifecycle hooks around a tool call ---------------------------------
    async def _pre_tool_veto(
        self, call: Any, context: RunContext, read_only: bool
    ) -> ToolResult | None:
        """Run the PRE_TOOL hooks; return a refusal result if one vetoed, else None.

        With no PRE_TOOL hook registered this returns None without building a
        context or awaiting anything, so an ordinary run is untouched. A veto is
        turned into an ordinary failed ToolResult - the model reads it like any
        other tool failure and can change course, exactly as it does for a
        permission refusal. It is a hard stop, not a prompt.
        """
        if not self._hooks.has(HookPoint.PRE_TOOL):
            return None
        outcome = await self._hooks.dispatch(
            HookContext(
                point=HookPoint.PRE_TOOL,
                session_id=context.session.id,
                run_id=context.run_id,
                tool_name=call.name,
                arguments=call.arguments,
                read_only=read_only,
                working_directory=getattr(context.session, "working_directory", "") or "",
            )
        )
        if outcome is not None and outcome.block:
            reason = outcome.reason or "a pre-tool hook blocked this call"
            return ToolResult(success=False, error=f"Blocked before running: {reason}")
        return None

    async def _post_tool(
        self, call: Any, context: RunContext, result: Any, read_only: bool
    ) -> None:
        """Run the POST_TOOL hooks with the result. Observe-only: the return is ignored.

        Fires only for a call that actually ran - not one refused by permission or
        vetoed by a pre-tool hook - so "after it returns" means what it says. With
        no POST_TOOL hook registered it does nothing at all.
        """
        if not self._hooks.has(HookPoint.POST_TOOL):
            return
        await self._hooks.dispatch(
            HookContext(
                point=HookPoint.POST_TOOL,
                session_id=context.session.id,
                run_id=context.run_id,
                tool_name=call.name,
                arguments=call.arguments,
                result=result,
                read_only=read_only,
                working_directory=getattr(context.session, "working_directory", "") or "",
            )
        )

    def _group_calls(self, calls: list, refusals: dict) -> list:
        """Split the calls into consecutive runs that may go at the same time.

        A group of more than one is all read-only (or refused, which runs nothing
        at all). Everything else ends up in a group of its own.
        """
        groups: list = []
        for call in calls:
            together = refusals[call.id] is not None or self._is_parallel_safe(call.name)
            if together and groups and self._group_is_parallel(groups[-1], refusals):
                groups[-1].append(call)
            else:
                groups.append([call])
        return groups

    def _group_is_parallel(self, group: list, refusals: dict) -> bool:
        return all(
            refusals[c.id] is not None or self._is_parallel_safe(c.name) for c in group
        )

    def _is_parallel_safe(self, name: str) -> bool:
        """Read-only tools can run alongside each other; nothing else can.

        An unknown tool is not safe. It can't run anyway, but defaulting the
        unknown case to "safe" is the kind of assumption that ages badly.
        """
        tool = self._tools.find(name)
        return bool(tool is not None and tool.definition.permissions.read_only)

    # -- record the run and announce it's over ------------------------------
    async def _finish(self, session: Any, result: RunResult) -> None:
        record = RunRecord(
            run_id=result.run_id,
            session_id=session.id,
            status=result.status.value,
            final_text=result.final_text,
            trace_id=result.trace_id,
            turns=result.turns,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            error=result.error,
            cached_tokens=result.usage.cached_tokens,
            duration_seconds=result.duration_seconds,
            cost_usd=result.cost_usd,
            model=result.model,
            retries=result.retries,
            reasoning_tokens=result.usage.reasoning_tokens,
        )
        await self._db.save_run(record)
        await self._bus.emit(
            session.id,
            EventType.RUN_FINISHED,
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "turns": result.turns,
                "final_text": result.final_text,
                "error": result.error,
                # The same receipt the caller gets, so a UI that only watches the
                # event stream doesn't have to be handed the RunResult too.
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cached_tokens": result.usage.cached_tokens,
                # A breakdown of output_tokens, not an addition - see Usage. Carried
                # so a stream watcher can show how much of the reply was thinking.
                "reasoning_tokens": result.usage.reasoning_tokens,
                "duration_seconds": result.duration_seconds,
                "cost_usd": result.cost_usd,
                "model": result.model,
                "retries": result.retries,
                # How many times the run left the model it started on. Zero for an
                # ordinary run; carried so a stream watcher sees a fallover too.
                "fallbacks": result.fallbacks,
                # Empty unless a ceiling stopped the run, in which case it names
                # which one - so a stream watcher can report the reason too.
                "stop_reason": result.stop_reason,
                "trace_id": result.trace_id,
            },
            result.run_id,
        )


# ---------------------------------------------------------------------------
# Assembling streamed tool-call fragments
# ---------------------------------------------------------------------------
def _merge_fragment(fragments: dict, order: list, data: dict) -> None:
    """Fold one streamed tool-call fragment into the call it belongs to.

    Providers send a tool call in pieces keyed by `index`: the id and name arrive
    once, the arguments dribble in as string chunks. A provider that sends one
    whole fragment and one that sends many both end up here.
    """
    index = data.get("index", 0)
    if index not in fragments:
        fragments[index] = {"id": "", "name": "", "arguments": ""}
        order.append(index)
    frag = fragments[index]
    if data.get("id"):
        frag["id"] = data["id"]
    if data.get("name"):
        frag["name"] = data["name"]
    if data.get("arguments"):
        frag["arguments"] += data["arguments"]


async def _close_stream(stream: Any) -> None:
    """Shut a model stream down, whether or not it has anything to shut down.

    A provider's `stream` is an async generator, and one left half-read holds the
    HTTP response open behind it. Closing it is only interesting when a turn was
    abandoned part way; on the normal path it's already finished and this does
    nothing. Anything raised on the way out is swallowed on purpose - the reply is
    already built, and failing to hang up is not a reason to lose it.
    """
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:
        pass


def _build_tool_calls(fragments: dict, order: list) -> list:
    """Turn assembled fragments into ToolCall parts, in the order they appeared."""
    calls = []
    for index in order:
        frag = fragments[index]
        calls.append(
            ToolCall(
                id=frag["id"] or ("call_" + uuid.uuid4().hex[:8]),
                name=frag["name"],
                arguments=_parse_args(frag["arguments"]),
                status=ToolCallStatus.PENDING,
            )
        )
    return calls


def _parse_args(raw: str) -> dict:
    """Parse the streamed argument string into a dict; empty or broken -> {}."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _add_usage(total: Usage, usage: Usage | None) -> None:
    if usage is None:
        return
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    total.cached_tokens += usage.cached_tokens
    total.reasoning_tokens += usage.reasoning_tokens


def _effort_kwargs(context: RunContext) -> dict:
    """The thinking-budget to hand this turn's `provider.stream`, if any.

    The per-run `Limits.reasoning_effort` wins; the agent's own
    `AgentConfig.reasoning_effort` is the fallback; empty means "no preference",
    and then nothing is passed at all. Returning a dict to splat, rather than
    always passing the keyword, is deliberate: a provider whose `stream` predates
    the knob (every scripted test double) is then called with exactly the old
    signature, so adding the feature breaks nothing that didn't ask for it.
    """
    effort = context.limits.reasoning_effort or context.config.reasoning_effort
    return {"reasoning_effort": effort} if effort else {}


#: What the model is told, in the request only, when a run is in plan mode. It
#: rides on the rendered wire and is never stored, so it shapes this run without
#: leaving a trace a later (approved) run would replay.
_PLAN_MODE_BANNER = (
    "You are in PLAN MODE. Only read-only tools are available to you right now: "
    "investigate the task and work out a concrete approach, but do not try to "
    "change anything. Writing files, running commands, and every other mutating "
    "action are blocked until the user has reviewed your plan and approved it. Use "
    "the plan tool to record the steps you would take, then give the user that plan "
    "as your final answer rather than carrying it out."
)


def _with_plan_mode_banner(wire: list) -> list:
    """Return the wire with the plan-mode banner folded into its system message.

    Folded into the leading system message when there is one - rather than added
    as a new message - so the message count and the assistant/tool-result pairing
    the wire format requires are left exactly as they were. The input list and its
    dicts are never mutated (a fresh head dict is built), because `render` hands
    back a fresh wire each turn and the caller relies on that staying true.
    """
    if wire and wire[0].get("role") == "system":
        head = dict(wire[0])
        head["content"] = f"{head.get('content', '')}\n\n{_PLAN_MODE_BANNER}".strip()
        return [head, *wire[1:]]
    return [{"role": "system", "content": _PLAN_MODE_BANNER}, *wire]


def _answered_call_ids(messages: list) -> set:
    """The ids of every tool call that already has a saved result.

    A resume runs only the calls that aren't in this set, which is what keeps a
    side effect from happening twice: a result on disk means the call ran, so it
    is left alone no matter where the crash fell.
    """
    answered: set = set()
    for msg in messages:
        if msg.role == Role.TOOL:
            for call in msg.tool_calls():
                answered.add(call.id)
    return answered


# ---------------------------------------------------------------------------
# Telling a bad moment apart from a bad request
# ---------------------------------------------------------------------------
#: Phrases that mean "try again": the provider is busy, slow, or briefly gone.
_TEMPORARY_HINTS = (
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "connect",
    "temporarily unavailable",
    "overloaded",
    "capacity",
    "500",
    "502",
    "503",
    "504",
    "internal server error",
    "bad gateway",
    "service unavailable",
)

#: Phrases that mean "don't bother": the request itself is wrong, and the second
#: attempt will fail exactly the same way. Checked first, so a message that
#: happens to contain both words is treated as permanent.
_PERMANENT_HINTS = (
    "api key",
    "api_key",
    "unauthorized",
    "authentication",
    "401",
    "403",
    "404",
    "invalid_request",
    "no model registered",
    "no provider registered",
    "does not exist",
    "is not installed",
    "context_length",
    "context length",
    "too large",
)


def _is_temporary(exc: BaseException) -> bool:
    """Is this the kind of failure that a second attempt might survive?

    There is no portable exception type for "the provider is busy" - each SDK
    raises its own - so this reads the message. Two deliberate biases: permanent
    phrases are checked first, and a message matching *nothing* is treated as
    permanent. An unknown error is far more likely to be a real bug than a blip,
    and retrying a real bug three times just makes it slower to find.
    """
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(hint in text for hint in _PERMANENT_HINTS):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any(hint in text for hint in _TEMPORARY_HINTS)
