"""In-memory ``TaskRepository`` — the default, fully hermetic backend.

Plain dicts, no I/O. It is the store the unit suite runs against and the
sensible default for local dev where a Postgres instance is overkill. It keeps
the same task/run/event shape as the Postgres store so switching backends
changes nothing above the repository seam.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from common.agent import AgentRunResult, AgentTask
from common.approvals import ApprovalRecord, ApprovalRequest
from common.config import AgentConfig
from common.enums import RunStatus, TaskStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord

from ..errors import TaskNotFound, ThreadNotFound
from .base import RunSummary, ThreadRecord


@dataclass(slots=True)
class _Run:
    id: str
    task_id: str
    status: RunStatus
    order: int
    output: str | None = None
    last_error: str | None = None
    events: list[tuple[int, str, dict]] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    trace_refs: list[dict] = field(default_factory=list)
    approvals: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class _Thread:
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class InMemoryTaskRepository:
    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._threads: dict[str, _Thread] = {}
        self._task_status: dict[str, TaskStatus] = {}
        self._runs: dict[str, _Run] = {}
        self._approvals: dict[str, ApprovalRecord] = {}
        self._order = itertools.count()
        self._tools: dict[tuple[str, str], tuple[str, dict]] = {}

    async def save_task(self, task: AgentTask) -> None:
        self._tasks[task.id] = task
        # AgentTask.created_at is naive (datetime.utcnow); thread timestamps
        # are tz-aware, so normalize before storing or comparing.
        created_at = task.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        thread = self._threads.get(task.thread_id)
        if thread is None:
            self._threads[task.thread_id] = _Thread(
                id=task.thread_id,
                title=task.metadata.get("title"),
                created_at=created_at,
                updated_at=created_at,
            )
        else:
            thread.updated_at = max(thread.updated_at, created_at)
        self._task_status.setdefault(task.id, TaskStatus.PLANNING)

    async def create_thread(self, thread_id: str, title: str | None = None) -> ThreadRecord:
        now = datetime.now(UTC)
        thread = self._threads.get(thread_id)
        if thread is None:
            thread = _Thread(thread_id, title, now, now)
            self._threads[thread_id] = thread
        return ThreadRecord(thread.id, thread.title, 0, thread.created_at, thread.updated_at)

    async def delete_thread(self, thread_id: str) -> bool:
        if thread_id not in self._threads:
            return False
        task_ids = {task.id for task in self._tasks.values() if task.thread_id == thread_id}
        for task_id in task_ids:
            del self._tasks[task_id]
            self._task_status.pop(task_id, None)
        for run_id in [run.id for run in self._runs.values() if run.task_id in task_ids]:
            del self._runs[run_id]
        for approval_id in [
            approval_id
            for approval_id, record in self._approvals.items()
            if record.request.task_id in task_ids
        ]:
            del self._approvals[approval_id]
        del self._threads[thread_id]
        return True

    async def get_task(self, task_id: str) -> AgentTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFound(task_id) from None

    async def list_threads(self, *, limit: int, offset: int) -> list[ThreadRecord]:
        threads = sorted(
            self._threads.values(),
            key=lambda thread: (thread.updated_at, thread.id),
            reverse=True,
        )
        selected = threads[offset : offset + limit]
        return [
            ThreadRecord(
                id=thread.id,
                title=thread.title,
                task_count=sum(
                    task.thread_id == thread.id for task in self._tasks.values()
                ),
                created_at=thread.created_at,
                updated_at=thread.updated_at,
            )
            for thread in selected
        ]

    async def list_tasks_by_thread(
        self, thread_id: str, *, limit: int, offset: int
    ) -> list[tuple[AgentTask, RunStatus | None]]:
        if thread_id not in self._threads:
            raise ThreadNotFound(thread_id)
        tasks = sorted(
            (task for task in self._tasks.values() if task.thread_id == thread_id),
            key=lambda task: (task.created_at, task.id),
            reverse=True,
        )
        return [
            (task, self._latest_run_status(task.id))
            for task in tasks[offset : offset + limit]
        ]

    async def create_run(
        self, task_id: str, config: AgentConfig, metadata: dict | None = None
    ) -> str:
        run_id = str(uuid4())
        self._runs[run_id] = _Run(
            id=run_id,
            task_id=task_id,
            status=RunStatus.CREATED,
            order=next(self._order),
            metadata=metadata or {},
        )
        return run_id

    async def get_latest_run_id(self, task_id: str) -> str | None:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return None
        return max(runs, key=lambda r: r.order).id

    async def get_latest_run_metadata(self, task_id: str) -> dict:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return {}
        return dict(max(runs, key=lambda r: r.order).metadata)

    async def get_latest_run(self, task_id: str) -> RunSummary | None:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return None
        run = max(runs, key=lambda r: r.order)
        return RunSummary(
            run_id=run.id,
            status=run.status,
            output=run.output,
            error=run.last_error,
            metadata=dict(run.metadata),
        )

    async def mark_run_running(self, run_id: str) -> None:
        self._runs[run_id].status = RunStatus.RUNNING

    async def append_event(
        self, run_id: str, event: AgentEvent, sequence_number: int
    ) -> None:
        self._runs[run_id].events.append((sequence_number, event.type, event.payload))

    async def save_llm_call(self, run_id: str, record: LLMCallRecord) -> None:
        self._runs[run_id].llm_calls.append(record)

    async def save_tool_call(self, run_id: str, record: ToolCallRecord) -> None:
        tool_id = await self.upsert_tool(
            record.server_name,
            record.base_url,
            {
                "name": record.tool_name,
                "description": record.description,
                "input_schema": record.input_schema,
            },
        )
        from dataclasses import replace
        self._runs[run_id].tool_calls.append(replace(record, tool_id=tool_id))

    async def save_phase(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].phases.append(value)
        return value["id"]

    async def close_phase(self, run_id: str, payload: dict) -> None:
        for phase in self._runs[run_id].phases:
            if phase["id"] == payload["phase_id"]:
                phase.update(payload)
                return

    async def save_plan(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].plans.append(value)
        return value["id"]

    async def save_finding(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].findings.append(value)
        return value["id"]

    async def save_verification(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].verifications.append(value)
        return value["id"]

    async def save_trace_ref(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4())}
        self._runs[run_id].trace_refs.append(value)
        return value["id"]

    async def save_approval(self, run_id: str, payload: dict) -> str:
        value = {**payload, "id": payload.get("id") or str(uuid4()), "status": "pending"}
        self._runs[run_id].approvals.append(value)
        return value["id"]

    async def upsert_tool(
        self, server_name: str, base_url: str | None, tool_spec: dict
    ) -> str:
        key = (server_name, str(tool_spec["name"]))
        existing = self._tools.get(key)
        tool_id = existing[0] if existing else str(uuid4())
        self._tools[key] = (tool_id, {**tool_spec, "base_url": base_url})
        return tool_id

    async def finalize_run(self, run_id: str, result: AgentRunResult) -> None:
        run = self._runs[run_id]
        run.status = result.status
        run.output = result.output
        run.last_error = result.metadata.get("error")
        run.metadata.update(result.metadata)

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        self._task_status[task_id] = status

    async def get_latest_run_status(self, task_id: str) -> RunStatus | None:
        return self._latest_run_status(task_id)

    async def list_events(
        self, task_id: str, *, latest_run_only: bool = True
    ) -> list[AgentEvent]:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if latest_run_only and runs:
            runs = [max(runs, key=lambda r: r.order)]
        events: list[AgentEvent] = []
        for run in sorted(runs, key=lambda r: r.order):
            events.extend(
                AgentEvent(type=event_type, payload=dict(payload))
                for _sequence, event_type, payload in sorted(
                    run.events, key=lambda event: event[0]
                )
            )
        return events

    async def list_thread_events(self, thread_id: str) -> list[tuple[str, AgentEvent]]:
        task_ids = {task.id for task in self._tasks.values() if task.thread_id == thread_id}
        runs = sorted(
            (run for run in self._runs.values() if run.task_id in task_ids),
            key=lambda run: run.order,
        )
        return [
            (run.task_id, AgentEvent(type=event_type, payload=dict(payload)))
            for run in runs
            for _sequence, event_type, payload in sorted(
                run.events, key=lambda event: event[0]
            )
        ]

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        self._approvals.setdefault(request.id, ApprovalRecord(request=request))

    async def resolve_approval(
        self,
        request_or_payload: str | dict,
        approved: bool | None = None,
        note: str | None = None,
    ) -> None:
        if isinstance(request_or_payload, dict):
            payload = request_or_payload
            for run in self._runs.values():
                for approval in run.approvals:
                    if approval["id"] == payload["approval_id"]:
                        approval.update(payload)
                        approval["status"] = (
                            "approved" if payload["approved"] else "denied"
                        )
                        break
            request_id = str(payload.get("approval_id", ""))
            approved = bool(payload.get("approved"))
            note = payload.get("note")
        else:
            request_id = request_or_payload
        record = self._approvals.get(request_id)
        if record is not None and approved is not None:
            self._approvals[request_id] = ApprovalRecord(
                request=record.request, approved=approved, note=note
            )

    async def get_approval_state(self, request_id: str) -> ApprovalRecord | None:
        return self._approvals.get(request_id)

    async def list_pending_approvals(self) -> list[ApprovalRequest]:
        return [
            record.request
            for record in self._approvals.values()
            if record.approved is None
        ]

    def _latest_run_status(self, task_id: str) -> RunStatus | None:
        runs = [r for r in self._runs.values() if r.task_id == task_id]
        if not runs:
            return None
        return max(runs, key=lambda r: r.order).status

    # -- test/introspection helpers (not part of the Protocol) --------------

    def events_for(self, run_id: str) -> list[tuple[int, str, dict]]:
        return list(self._runs[run_id].events)

    def llm_calls_for(self, run_id: str) -> list[LLMCallRecord]:
        return list(self._runs[run_id].llm_calls)

    def tool_calls_for(self, run_id: str) -> list[ToolCallRecord]:
        return list(self._runs[run_id].tool_calls)

    def task_status(self, task_id: str) -> TaskStatus | None:
        return self._task_status.get(task_id)
