"""SQLite-backed task repository for desktop deployments.

The API repository contract is deliberately backend-neutral.  SQLite keeps the
desktop path durable without requiring Docker or a running PostgreSQL server.
The state is serialized as one application-owned blob because the in-memory
repository already defines the complete, tested object model (including enums
and nested event payloads).  SQLite still provides atomic replacement, locking,
and crash-safe journaling for that blob.

This backend is intended for a single desktop API process.  PostgreSQL remains
the recommended multi-process/server backend.
"""

from __future__ import annotations

import asyncio
import itertools
import pickle
import sqlite3
from pathlib import Path

from common.agent import AgentRunResult, AgentTask
from common.approvals import ApprovalRequest
from common.config import AgentConfig
from common.enums import TaskStatus
from common.events import AgentEvent, LLMCallRecord, ToolCallRecord

from .memory import InMemoryTaskRepository

_TABLE = "api_repository_state"
_STATE_FIELDS = (
    "_tasks",
    "_threads",
    "_task_status",
    "_runs",
    "_approvals",
    "_tools",
)


class SQLiteTaskRepository(InMemoryTaskRepository):
    """Durable ``TaskRepository`` using a local SQLite database file."""

    def __init__(self, database_path: str | Path) -> None:
        super().__init__()
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), state BLOB NOT NULL)"
        )
        self._connection.commit()
        self._write_lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError("SQLite repository is closed")
        row = connection.execute(
            f"SELECT state FROM {_TABLE} WHERE id = 1"
        ).fetchone()
        if row is None:
            return
        try:
            state = pickle.loads(row[0])
        except (
            EOFError,
            pickle.PickleError,
            TypeError,
            ValueError,
            ImportError,
            AttributeError,
        ) as exc:
            raise RuntimeError(
                f"SQLite repository state at {self.database_path} is invalid"
            ) from exc
        if not isinstance(state, dict):
            raise TypeError(
                f"SQLite repository state at {self.database_path} has an invalid shape"
            )
        for field in _STATE_FIELDS:
            value = state.get(field)
            if isinstance(value, dict):
                setattr(self, field, value)
        max_order = max(
            (getattr(run, "order", -1) for run in self._runs.values()),
            default=-1,
        )
        self._order = itertools.count(max_order + 1)

    def _write_sync(self) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError("SQLite repository is closed")
        state = {field: getattr(self, field) for field in _STATE_FIELDS}
        payload = sqlite3.Binary(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        connection.execute(
            f"INSERT INTO {_TABLE} (id, state) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET state = excluded.state",
            (payload,),
        )
        connection.commit()

    async def _persist_locked(self) -> None:
        await asyncio.to_thread(self._write_sync)

    async def close(self) -> None:
        """Flush and close the SQLite connection."""
        async with self._write_lock:
            if self._connection is not None:
                await self._persist_locked()
                connection = self._connection
                if connection is None:
                    return
                self._connection = None  # type: ignore[assignment]
                await asyncio.to_thread(connection.close)

    async def save_task(self, task: AgentTask) -> None:
        async with self._write_lock:
            await super().save_task(task)
            await self._persist_locked()

    async def create_thread(self, thread_id: str, title: str | None = None):  # type: ignore[override]
        async with self._write_lock:
            record = await super().create_thread(thread_id, title)
            await self._persist_locked()
            return record

    async def delete_thread(self, thread_id: str) -> bool:
        async with self._write_lock:
            deleted = await super().delete_thread(thread_id)
            if deleted:
                await self._persist_locked()
            return deleted

    async def create_run(
        self, task_id: str, config: AgentConfig, metadata: dict | None = None
    ) -> str:
        async with self._write_lock:
            run_id = await InMemoryTaskRepository.create_run(
                self, task_id, config, metadata
            )
            await self._persist_locked()
            return run_id

    async def mark_run_running(self, run_id: str) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.mark_run_running(self, run_id)
            await self._persist_locked()

    async def append_event(
        self, run_id: str, event: AgentEvent, sequence_number: int
    ) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.append_event(
                self, run_id, event, sequence_number
            )
            await self._persist_locked()

    async def save_llm_call(self, run_id: str, record: LLMCallRecord) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.save_llm_call(self, run_id, record)
            await self._persist_locked()

    async def save_tool_call(self, run_id: str, record: ToolCallRecord) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.save_tool_call(self, run_id, record)
            await self._persist_locked()

    async def save_phase(self, run_id: str, payload: dict) -> str:
        async with self._write_lock:
            value = await InMemoryTaskRepository.save_phase(self, run_id, payload)
            await self._persist_locked()
            return value

    async def close_phase(self, run_id: str, payload: dict) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.close_phase(self, run_id, payload)
            await self._persist_locked()

    async def save_plan(self, run_id: str, payload: dict) -> str:
        async with self._write_lock:
            value = await InMemoryTaskRepository.save_plan(self, run_id, payload)
            await self._persist_locked()
            return value

    async def save_finding(self, run_id: str, payload: dict) -> str:
        async with self._write_lock:
            value = await InMemoryTaskRepository.save_finding(self, run_id, payload)
            await self._persist_locked()
            return value

    async def save_verification(self, run_id: str, payload: dict) -> str:
        async with self._write_lock:
            value = await InMemoryTaskRepository.save_verification(
                self, run_id, payload
            )
            await self._persist_locked()
            return value

    async def save_trace_ref(self, run_id: str, payload: dict) -> str:
        async with self._write_lock:
            value = await InMemoryTaskRepository.save_trace_ref(self, run_id, payload)
            await self._persist_locked()
            return value

    async def save_approval(self, run_id: str, payload: dict) -> str:
        async with self._write_lock:
            value = await InMemoryTaskRepository.save_approval(self, run_id, payload)
            await self._persist_locked()
            return value

    async def finalize_run(self, run_id: str, result: AgentRunResult) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.finalize_run(self, run_id, result)
            await self._persist_locked()

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.update_task_status(self, task_id, status)
            await self._persist_locked()

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.save_approval_request(self, request)
            await self._persist_locked()

    async def resolve_approval(
        self,
        request_or_payload: str | dict,
        approved: bool | None = None,
        note: str | None = None,
    ) -> None:
        async with self._write_lock:
            await InMemoryTaskRepository.resolve_approval(
                self, request_or_payload, approved, note
            )
            await self._persist_locked()
