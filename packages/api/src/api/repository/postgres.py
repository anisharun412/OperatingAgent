"""Postgres-backed ``TaskRepository`` over the 5-table run spine.

Writes ``actors -> agent_threads -> agent_tasks -> config_snapshots ->
agent_runs -> agent_events`` using a psycopg 3 async connection pool. The pool
runs in autocommit mode; the multi-row writes (``save_task``, ``create_run``)
are wrapped in an explicit ``conn.transaction()`` so they land atomically.

The run spine and durable state/action events are persisted here. LangGraph's
checkpointer owns graph snapshots separately, while Langfuse remains the source
of truth for model observations and evaluation metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.agent import AgentRunResult, AgentTask
from common.approvals import ApprovalRecord, ApprovalRequest
from common.config import AgentConfig
from common.enums import AgentTrack, RunStatus, TaskStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord
from psycopg.types.json import Jsonb

from ..errors import TaskNotFound, ThreadNotFound
from ..serialization import config_content_hash, config_to_snapshot
from . import _sql
from .base import RunSummary, ThreadRecord

if TYPE_CHECKING:  # avoid importing psycopg_pool at module import time
    from psycopg_pool import AsyncConnectionPool

#: Stable identity for the actor that owns API-created threads.
_API_ACTOR_EXTERNAL_ID = "system:api"
_API_ACTOR_DISPLAY_NAME = "API service"


async def _fetch_scalar(cur) -> object:
    row = await cur.fetchone()
    if row is None:
        raise RuntimeError("database statement did not return the expected row")
    return row[0]


class PostgresTaskRepository:
    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    async def create_thread(self, thread_id: str, title: str | None = None) -> ThreadRecord:
        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute(_sql.UPSERT_ACTOR, (_API_ACTOR_EXTERNAL_ID, _API_ACTOR_DISPLAY_NAME))
            actor = await cur.fetchone()
            if actor is None:
                raise RuntimeError("actor upsert returned no row")
            await cur.execute(_sql.UPSERT_THREAD, (thread_id, actor[0], title))
        records = await self.list_threads(limit=500, offset=0)
        for record in records:
            if record.id == thread_id:
                return record
        raise RuntimeError("thread creation did not return a record")

    async def delete_thread(self, thread_id: str) -> bool:
        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute(_sql.DELETE_THREAD_EVENTS, (thread_id,))
            await cur.execute(_sql.DELETE_THREAD_RUNS, (thread_id,))
            await cur.execute(_sql.DELETE_THREAD_TASKS, (thread_id,))
            await cur.execute(_sql.DELETE_THREAD, (thread_id, _API_ACTOR_EXTERNAL_ID))
            return cur.rowcount > 0

    async def save_task(self, task: AgentTask) -> None:
        title = task.metadata.get("title")
        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute(
                _sql.UPSERT_ACTOR,
                (_API_ACTOR_EXTERNAL_ID, _API_ACTOR_DISPLAY_NAME),
            )
            actor_row = await cur.fetchone()
            if actor_row is None:
                raise RuntimeError("actor upsert returned no row")
            actor_id = actor_row[0]
            await cur.execute(_sql.UPSERT_THREAD, (task.thread_id, actor_id, title))
            await cur.execute(
                _sql.INSERT_TASK,
                (
                    task.id,
                    task.thread_id,
                    task.goal,
                    task.track.value,
                    TaskStatus.PLANNING.value,
                    Jsonb(task.metadata),
                ),
            )

    async def get_task(self, task_id: str) -> AgentTask:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.SELECT_TASK, (task_id,))
            row = await cur.fetchone()
        if row is None:
            raise TaskNotFound(task_id)
        id_, thread_id, goal, track, metadata, created_at = row
        return AgentTask(
            id=str(id_),
            goal=goal,
            thread_id=thread_id,
            track=AgentTrack(track),
            metadata=metadata or {},
            created_at=created_at,
        )

    async def list_threads(self, *, limit: int, offset: int) -> list[ThreadRecord]:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.SELECT_THREADS,
                (_API_ACTOR_EXTERNAL_ID, limit, offset),
            )
            rows = await cur.fetchall()
        return [
            ThreadRecord(
                id=thread_id,
                title=title,
                task_count=task_count,
                created_at=created_at,
                updated_at=updated_at,
            )
            for thread_id, title, task_count, created_at, updated_at in rows
        ]

    async def list_tasks_by_thread(
        self, thread_id: str, *, limit: int, offset: int
    ) -> list[tuple[AgentTask, RunStatus | None]]:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.SELECT_THREAD_EXISTS,
                (thread_id, _API_ACTOR_EXTERNAL_ID),
            )
            if await cur.fetchone() is None:
                raise ThreadNotFound(thread_id)
            await cur.execute(_sql.SELECT_TASKS_BY_THREAD, (thread_id, limit, offset))
            rows = await cur.fetchall()
        return [
            (
                AgentTask(
                    id=str(task_id),
                    goal=goal,
                    thread_id=row_thread_id,
                    track=AgentTrack(track),
                    metadata=metadata or {},
                    created_at=created_at,
                ),
                RunStatus(run_status) if run_status is not None else None,
            )
            for (
                task_id,
                row_thread_id,
                goal,
                track,
                metadata,
                created_at,
                run_status,
            ) in rows
        ]

    async def create_run(
        self, task_id: str, config: AgentConfig, metadata: dict | None = None
    ) -> str:
        snapshot = config_to_snapshot(config)
        content_hash = config_content_hash(snapshot)
        async with (
            self._pool.connection() as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute(
                _sql.UPSERT_CONFIG_SNAPSHOT,
                (
                    content_hash,
                    Jsonb(snapshot["llm_config"]),
                    Jsonb(snapshot["execution_config"]),
                    Jsonb(snapshot["sandbox_config"]),
                    Jsonb(snapshot["permissions_config"]),
                    Jsonb(snapshot["checkpoint_config"]),
                    Jsonb(snapshot["tracing_config"]),
                    Jsonb(snapshot["behaviour_config"]),
                    Jsonb(snapshot["prompts_config"]),
                ),
            )
            snapshot_row = await cur.fetchone()
            if snapshot_row is None:
                raise RuntimeError("config snapshot upsert returned no row")
            snapshot_id = snapshot_row[0]
            await cur.execute(
                _sql.INSERT_RUN,
                (
                    task_id,
                    task_id,
                    snapshot_id,
                    RunStatus.CREATED.value,
                    Jsonb(metadata or {}),
                ),
            )
            run_row = await cur.fetchone()
            if run_row is None:
                raise RuntimeError("run insert returned no row")
            run_id = run_row[0]
        return str(run_id)

    async def get_latest_run_id(self, task_id: str) -> str | None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.SELECT_LATEST_RUN_ID, (task_id,))
            row = await cur.fetchone()
        return str(row[0]) if row is not None else None

    async def get_latest_run_metadata(self, task_id: str) -> dict:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.SELECT_LATEST_RUN_METADATA, (task_id,))
            row = await cur.fetchone()
        return dict(row[0] or {}) if row is not None else {}

    async def get_latest_run(self, task_id: str) -> RunSummary | None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.SELECT_LATEST_RUN, (task_id,))
            row = await cur.fetchone()
        if row is None:
            return None
        run_id, status, output, error, metadata = row
        return RunSummary(
            run_id=str(run_id),
            status=RunStatus(status),
            output=output,
            error=error,
            metadata=dict(metadata or {}),
        )

    async def mark_run_running(self, run_id: str) -> None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.MARK_RUN_RUNNING, (run_id,))

    async def append_event(
        self, run_id: str, event, sequence_number: int
    ) -> None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.INSERT_EVENT,
                (run_id, sequence_number, event.type, Jsonb(event.payload)),
            )

    async def save_llm_call(self, run_id: str, record: LLMCallRecord) -> None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.INSERT_LLM_CALL,
                (
                    run_id,
                    record.node_name,
                    record.provider,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.cost,
                    record.error,
                    record.started_at,
                    record.finished_at,
                ),
            )

    async def upsert_tool(
        self, server_name: str, base_url: str | None, tool_spec: dict
    ) -> str:
        async with self._pool.connection() as conn, conn.transaction():
            async with conn.cursor() as cur:
                tool_id = await self._upsert_tool(
                    cur, server_name, base_url, tool_spec
                )
                return str(tool_id)

    async def _upsert_tool(
        self, cur, server_name: str, base_url: str | None, tool_spec: dict
    ):
        await cur.execute(_sql.UPSERT_MCP_SERVER, (server_name, base_url))
        server_id = await _fetch_scalar(cur)
        await cur.execute(
            _sql.UPSERT_TOOL,
            (
                server_id,
                tool_spec["name"],
                tool_spec.get("description"),
                Jsonb(tool_spec.get("input_schema")),
            ),
        )
        return await _fetch_scalar(cur)

    async def save_tool_call(self, run_id: str, record: ToolCallRecord) -> None:
        async with self._pool.connection() as conn, conn.transaction():
            async with conn.cursor() as cur:
                tool_id = await self._upsert_tool(
                    cur,
                    record.server_name,
                    record.base_url,
                    {
                        "name": record.tool_name,
                        "description": record.description,
                        "input_schema": record.input_schema,
                    },
                )
                await cur.execute(
                    _sql.INSERT_TOOL_CALL,
                    (
                        run_id, tool_id, Jsonb(record.arguments), record.success,
                        Jsonb(record.output), record.error, record.risk_level,
                        record.risk_reason, record.attempt, record.started_at,
                        record.finished_at,
                    ),
                )

    async def save_phase(self, run_id: str, payload: dict) -> str:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.INSERT_PHASE,
                (
                    payload.get("id"),
                    run_id,
                    payload["sequence"],
                    payload["phase"],
                    payload.get("entry_reason"),
                    payload.get("entered_at"),
                ),
            )
            return str(await _fetch_scalar(cur))

    async def close_phase(self, run_id: str, payload: dict) -> None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.CLOSE_PHASE,
                (payload.get("exited_at"), run_id, payload["phase_id"]),
            )

    async def save_plan(self, run_id: str, payload: dict) -> str:
        async with self._pool.connection() as conn:
            async with conn.transaction(), conn.cursor() as cur:
                await cur.execute(
                    _sql.INSERT_PLAN,
                    (
                        payload.get("id"),
                        run_id,
                        payload["phase_id"],
                        payload["revision"],
                        payload.get("summary"),
                        payload.get("reasoning"),
                        bool(payload.get("requires_remediation", False)),
                    ),
                )
                plan_id = await _fetch_scalar(cur)
                for index, step in enumerate(payload.get("steps", [])):
                    await cur.execute(
                        _sql.INSERT_PLAN_STEP,
                        (
                            step.get("id"),
                            plan_id,
                            run_id,
                            step.get("step_number", index),
                            step.get("description"),
                            step.get("tool_id"),
                            Jsonb(step.get("arguments")),
                            step.get("status"),
                            step.get("output"),
                        ),
                    )
                return str(plan_id)

    async def save_finding(self, run_id: str, payload: dict) -> str:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.INSERT_FINDING,
                (
                    payload.get("id"),
                    run_id,
                    payload.get("phase_id"),
                    payload.get("plan_step_id"),
                    payload.get("description"),
                    payload.get("detail"),
                    payload.get("source_tool_id"),
                ),
            )
            return str(await _fetch_scalar(cur))

    async def save_verification(self, run_id: str, payload: dict) -> str:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.INSERT_VERIFICATION,
                (
                    payload.get("id"),
                    run_id,
                    payload["plan_step_id"],
                    payload.get("tool_call_id"),
                    payload.get("attempt", 1),
                    payload["result"],
                    payload.get("reason"),
                    bool(payload.get("deterministic", False)),
                    Jsonb(payload.get("evidence")),
                ),
            )
            return str(await _fetch_scalar(cur))

    async def save_trace_ref(self, run_id: str, payload: dict) -> str:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.INSERT_TRACE_REF,
                (
                    payload.get("id"),
                    run_id,
                    payload.get("provider", "langfuse"),
                    payload["trace_id"],
                    Jsonb(payload.get("metadata") or {}),
                ),
            )
            return str(await _fetch_scalar(cur))

    async def save_approval(self, run_id: str, payload: dict) -> str:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.INSERT_APPROVAL,
                (
                    payload.get("id"),
                    run_id,
                    payload["plan_step_id"],
                    payload.get("reason"),
                    payload.get("expires_at"),
                ),
            )
            return str(await _fetch_scalar(cur))

    async def _resolve_approval_record(self, payload: dict) -> None:
        async with self._pool.connection() as conn:
            async with conn.transaction(), conn.cursor() as cur:
                external_id = payload.get(
                    "resolved_by_external_id", "system:api-approval"
                )
                await cur.execute(
                    _sql.RESOLVE_APPROVAL_ACTOR,
                    (
                        external_id,
                        payload.get("resolved_by_display_name", "API approver"),
                    ),
                )
                actor_id = await _fetch_scalar(cur)
                await cur.execute(
                    _sql.RESOLVE_APPROVAL,
                    (
                        "approved" if payload["approved"] else "denied",
                        actor_id,
                        payload.get("note"),
                        payload.get("resolved_at"),
                        payload.get("tool_call_id"),
                        payload["approval_id"],
                    ),
                )

    async def finalize_run(self, run_id: str, result: AgentRunResult) -> None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.FINALIZE_RUN,
                (
                    result.status.value,
                    result.output,
                    result.metadata.get("error"),
                    Jsonb(result.metadata),
                    run_id,
                ),
            )

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.UPDATE_TASK_STATUS, (status.value, task_id))

    async def get_latest_run_status(self, task_id: str) -> RunStatus | None:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.SELECT_LATEST_RUN_STATUS, (task_id,))
            row = await cur.fetchone()
        if row is None:
            return None
        return RunStatus(row[0])

    async def list_events(
        self, task_id: str, *, latest_run_only: bool = True
    ) -> list[AgentEvent]:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.SELECT_TASK_EVENTS,
                (task_id, latest_run_only, task_id),
            )
            rows = await cur.fetchall()
        return [AgentEvent(type=event_type, payload=payload or {}) for event_type, payload in rows]

    async def list_thread_events(self, thread_id: str) -> list[tuple[str, AgentEvent]]:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.SELECT_THREAD_EVENTS, (thread_id,))
            rows = await cur.fetchall()
        return [
            (str(task_id), AgentEvent(type=event_type, payload=payload or {}))
            for task_id, event_type, payload in rows
        ]

    @staticmethod
    def _approval_from_payload(payload: dict) -> ApprovalRequest:
        from common.enums import RiskLevel

        risk = payload.get("risk_level")
        return ApprovalRequest(
            id=str(payload["request_id"]),
            task_id=str(payload["task_id"]),
            tool_name=str(payload["tool_name"]),
            arguments=payload.get("arguments") or {},
            risk_level=RiskLevel(risk) if risk else None,
            description=payload.get("description"),
        )

    @classmethod
    def _approval_record(cls, event_type: str, payload: dict) -> ApprovalRecord:
        request = cls._approval_from_payload(payload)
        if event_type == "approval_resolved":
            return ApprovalRecord(
                request=request,
                approved=bool(payload.get("approved")),
                note=payload.get("note"),
            )
        return ApprovalRecord(request=request)

    async def _insert_approval_event(
        self, request_id: str, task_id: str, event_type: str, payload: dict
    ) -> None:
        run_id = await self.get_latest_run_id(task_id)
        if run_id is None:
            return
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                _sql.INSERT_APPROVAL_EVENT,
                (
                    run_id,
                    run_id,
                    run_id,
                    event_type,
                    Jsonb(payload),
                ),
            )

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        await self._insert_approval_event(
            request.id,
            request.task_id,
            "approval_requested",
            {
                "request_id": request.id,
                "task_id": request.task_id,
                "tool_name": request.tool_name,
                "arguments": request.arguments,
                "risk_level": request.risk_level.value if request.risk_level else None,
                "description": request.description,
            },
        )

    async def resolve_approval(
        self,
        request_or_payload: str | dict,
        approved: bool | None = None,
        note: str | None = None,
    ) -> None:
        if isinstance(request_or_payload, dict):
            await self._resolve_approval_record(request_or_payload)
            return
        request_id = request_or_payload
        current = await self.get_approval_state(request_id)
        if current is None:
            return
        payload = {
            "request_id": request_id,
            "task_id": current.request.task_id,
            "tool_name": current.request.tool_name,
            "arguments": current.request.arguments,
            "risk_level": (
                current.request.risk_level.value
                if current.request.risk_level
                else None
            ),
            "description": current.request.description,
            "approved": approved,
            "note": note,
        }
        await self._insert_approval_event(
            request_id, current.request.task_id, "approval_resolved", payload
        )

    async def _approval_states(self) -> dict[str, ApprovalRecord]:
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(_sql.SELECT_APPROVAL_STATES)
            rows = await cur.fetchall()
        return {
            str(payload["request_id"]): self._approval_record(event_type, payload)
            for event_type, payload in rows
        }

    async def get_approval_state(self, request_id: str) -> ApprovalRecord | None:
        return (await self._approval_states()).get(request_id)

    async def list_pending_approvals(self) -> list[ApprovalRequest]:
        return [
            record.request
            for record in (await self._approval_states()).values()
            if record.approved is None
        ]
