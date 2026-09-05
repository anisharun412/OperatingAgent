"""The same store, but on Postgres, so a run survives the process that made it.

`MemoryDatabase` is perfect until you close the terminal. This is the version
that remembers: same sixteen methods, same promises, different place to put them.
Nothing above `Database` changes - the loop, the event bus and the permission
store cannot tell which one they were handed, which is the whole point of the
interface being small.

Three things are worth knowing before reading the code.

**Messages are stored as JSON, not as tables.** A message is a list of parts
(text, reasoning, a tool call, a compaction summary), and shredding that into
normalised rows would mean a migration every time a new kind of part is invented.
The parts go in a JSONB column and come back through the same factories the rest
of the code uses. The one column that isn't JSON is `ordinal`, because reading a
conversation back in the right order matters more than anything else here.

**The event counter lives in the database.** `next_sequence` is a single
`UPDATE ... RETURNING`, so two runs asking at the same instant get two different
numbers. Doing it in Python would be correct until the day a second process opens
the same database, and then it would be quietly, unfixably wrong.

**asyncpg is imported lazily.** The package still imports on a machine that never
installed it; you only need it if you actually ask for Postgres.

**A dropped connection is retried, not fatal.** Every query goes through `_run`,
which acquires a connection from the pool, runs the query, and - if the
connection was lost (the server bounced, a network blip, the pool handed back a
stale socket) - waits and tries again with a fresh one, a few times, before
giving up. This mirrors the loop's own retry-on-transient-failure: a database
hiccup should cost a pause, not the run.

**The schema is infrastructure-owned.** `apply_schema` is intentionally a
verification step: Docker/deployment applies `infra/docker/postgres/schema.sql`
and the numbered migrations; the application requires both `001_base` and
`002_native_conversation` instead of attempting DDL at runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from .conversation import (
    Compaction,
    Conversation,
    Media,
    Message,
    Reasoning,
    Role,
    Session,
    Text,
    ToolCall,
    ToolCallStatus,
    Usage,
)
from .database import Database
from .events import Event
from .memory import Memory
from .permissions import PermissionDuration, PermissionGrant

LOGGER = logging.getLogger(__name__)

BASELINE_VERSION = 1
REQUIRED_MIGRATIONS = ("001_base", "002_native_conversation")
REQUIRED_MIGRATION = REQUIRED_MIGRATIONS[-1]
_ID_NAMESPACE = uuid.UUID("d18041c6-9239-4ac3-8adb-baa561ae988d")


class PostgresDatabase(Database):
    """The Database interface, backed by Postgres through asyncpg.

    Build it with a connection string and `await connect()` before use, or use
    `await PostgresDatabase.open(dsn)` which does both and applies the schema.
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 1,
        max_size: int = 10,
        *,
        max_retries: int = 3,
        retry_first_delay: float = 0.5,
    ) -> None:
        self.dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None
        #: How many times a lost connection is retried before the error is raised,
        #: and the first backoff in seconds (it doubles each attempt). The defaults
        #: echo the loop's retry-on-transient philosophy; a caller that would
        #: rather a database hiccup fail fast can pass ``max_retries=0``.
        self._max_retries = max(0, max_retries)
        self._retry_first_delay = max(0.0, retry_first_delay)
        #: Built once by ``_transient_errors`` then cached - see there for why it
        #: can't just be a module constant.
        self._transient: tuple[type[BaseException], ...] | None = None

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    async def open(cls, dsn: str, apply_schema: bool = True) -> PostgresDatabase:
        """Connect and verify the infrastructure-owned canonical schema."""
        db = cls(dsn)
        await db.connect()
        if apply_schema:
            await db.apply_schema()
        return db

    async def connect(self) -> None:
        if self._pool is not None:
            return
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "Postgres support needs asyncpg. Install it with "
                "`uv sync --all-packages --extra postgres`."
            ) from exc
        self._pool = await asyncpg.create_pool(
            self.dsn, min_size=self._min_size, max_size=self._max_size
        )

    async def apply_schema(self) -> None:
        """Verify the canonical schema; applications never execute DDL."""
        missing = []
        for migration_id in REQUIRED_MIGRATIONS:
            present = await self._fetchval(
                "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE id = $1)",
                migration_id,
            )
            if not present:
                missing.append(migration_id)
        if missing:
            raise RuntimeError(
                f"Postgres schema migration(s) {', '.join(missing)} are missing. "
                "Apply infra/docker/postgres/schema.sql before starting agent-native."
            )

    async def schema_version(self) -> int:
        """The highest schema version applied (0 on a database that has none yet)."""

        async def operation(conn: Any) -> int:
            value = await conn.fetchval("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
            return int(value or 0)

        return await self._run(operation)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # -- retry-aware execution --------------------------------------------
    def _acquire(self):
        if self._pool is None:
            raise RuntimeError("Not connected. Call `await database.connect()` first.")
        return self._pool.acquire()

    def _transient_errors(self) -> tuple[type[BaseException], ...]:
        """Exception types worth retrying: a lost or unreachable connection.

        Built once and cached. The builtin socket errors always apply; asyncpg's
        own connection-loss classes are added when it's installed - it's an
        optional dependency whose exception tree has shifted between versions, so
        importing defensively and reading the classes by name beats hard-coding
        one that might not exist. A programming error - bad SQL, wrong argument
        count, a constraint violation - is deliberately *not* here: retrying it
        only spends the backoff on its way to the same failure.
        """
        if self._transient is not None:
            return self._transient
        errors: list[type[BaseException]] = [
            ConnectionError,
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
        ]
        try:
            import asyncpg
        except ImportError:  # pragma: no cover - depends on install
            pass
        else:
            for attr in (
                "PostgresConnectionError",
                "InterfaceError",
                "ConnectionDoesNotExistError",
            ):
                exc_type = getattr(asyncpg, attr, None)
                if isinstance(exc_type, type):
                    errors.append(exc_type)
        self._transient = tuple(dict.fromkeys(errors))
        return self._transient

    async def _run(self, operation: Any) -> Any:
        """Run ``operation(conn)`` on a pooled connection, retrying a lost one.

        ``operation`` is a callable taking a connection and returning an awaitable,
        so it can be re-run against a *fresh* connection - the only thing that
        helps, since a dropped connection can't be resumed. A transient failure
        waits and tries again with exponential backoff (the store's echo of the
        loop's retry-on-transient rule); anything else, and the final attempt, is
        raised unchanged. When the retry works it's invisible: callers see
        ordinary asyncpg results and ordinary asyncpg errors.
        """
        delay = self._retry_first_delay
        last: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._acquire() as conn:
                    return await operation(conn)
            except self._transient_errors() as exc:
                last = exc
                if attempt >= self._max_retries:
                    break
                LOGGER.warning(
                    "database connection failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                delay *= 2
        assert last is not None  # only reached after at least one caught failure
        raise last

    async def _execute(self, sql: str, *args: Any) -> Any:
        return await self._run(lambda conn: conn.execute(sql, *args))

    async def _fetch(self, sql: str, *args: Any) -> Any:
        return await self._run(lambda conn: conn.fetch(sql, *args))

    async def _fetchrow(self, sql: str, *args: Any) -> Any:
        return await self._run(lambda conn: conn.fetchrow(sql, *args))

    async def _fetchval(self, sql: str, *args: Any) -> Any:
        return await self._run(lambda conn: conn.fetchval(sql, *args))

    # -- sessions ----------------------------------------------------------
    async def create_session(self, session: Session) -> None:
        async def operation(conn: Any) -> None:
            async with conn.transaction():
                actor_id = _native_actor_id()
                await conn.execute(
                    "INSERT INTO actors (id, kind, external_id, display_name) "
                    "VALUES ($1, 'agent', 'agent-native', 'agent-native') "
                    "ON CONFLICT (id) DO NOTHING",
                    actor_id,
                )
                await conn.execute(
                    """
                    INSERT INTO agent_threads (id, owner_actor_id, title, metadata)
                    VALUES ($1, $2, $3, $4::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        metadata = EXCLUDED.metadata
                    """,
                    session.id,
                    actor_id,
                    session.title or None,
                    json.dumps(_session_metadata(session)),
                )

        await self._run(operation)

    async def get_session(self, session_id: str) -> Session | None:
        row = await self._fetchrow(
            "SELECT id, title, metadata FROM agent_threads WHERE id = $1",
            session_id,
        )
        if row is None:
            return None
        return _row_to_session(row)

    async def delete_session(self, session_id: str) -> bool:
        async def operation(conn: Any) -> bool:
            status = await conn.execute("DELETE FROM agent_threads WHERE id = $1", session_id)
            try:
                return int(str(status).split()[-1]) > 0
            except (ValueError, IndexError):  # pragma: no cover - defensive
                return False

        return await self._run(operation)

    async def list_sessions(self, working_directory: str = "", limit: int = 0) -> list:
        """Sessions newest first. `created_at DESC` orders; `id` breaks a tie.

        The tie-breaker echoes `list_runs`: two sessions can share a `created_at`
        to the microsecond, and a history view that reorders them between calls is
        one you can't diff.
        """
        sql = "SELECT id, title, metadata FROM agent_threads "
        args: list = []
        if working_directory:
            sql += "WHERE metadata->>'working_directory' = $1 "
            args.append(working_directory)
        sql += "ORDER BY created_at DESC, id DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"  # int-cast, so no value reaches SQL unchecked
        rows = await self._fetch(sql, *args)
        return [_row_to_session(row) for row in rows]

    async def save_message(self, message: Message) -> None:
        try:
            await self._execute(
                """
                INSERT INTO conversation_messages (id, thread_id, role, parts, model, usage, created_at, native_message_id)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7, $8)
                ON CONFLICT (id) DO NOTHING
                """,
                _uuid_for("message", message.id),
                message.session_id,
                message.role.value,
                json.dumps([_part_to_json(p) for p in message.parts]),
                message.model,
                json.dumps(_usage_to_json(message.usage)) if message.usage else None,
                message.created_at,
                message.id,
            )
        except Exception as exc:
            if "native_message_id" not in str(exc):
                raise
            await self._execute(
                """
                INSERT INTO conversation_messages (id, thread_id, role, parts, model, usage, created_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7)
                ON CONFLICT (id) DO NOTHING
                """,
                _uuid_for("message", message.id),
                message.session_id,
                message.role.value,
                json.dumps([_part_to_json(p) for p in message.parts]),
                message.model,
                json.dumps(_usage_to_json(message.usage)) if message.usage else None,
                message.created_at,
            )

    async def load_conversation(self, session_id: str) -> Conversation:
        try:
            rows = await self._fetch(
                "SELECT COALESCE(native_message_id, id::text) AS id, thread_id AS session_id, role, parts, model, usage, created_at "
                "FROM conversation_messages WHERE thread_id = $1 ORDER BY ordinal",
                session_id,
            )
        except Exception as exc:
            if "native_message_id" not in str(exc):
                raise
            rows = await self._fetch(
                "SELECT id::text AS id, thread_id AS session_id, role, parts, model, usage, created_at "
                "FROM conversation_messages WHERE thread_id = $1 ORDER BY ordinal",
                session_id,
            )
        return Conversation([_row_to_message(row) for row in rows])

    # -- events ------------------------------------------------------------
    async def next_sequence(self, session_id: str) -> int:
        async def operation(conn: Any) -> int:
            value = await conn.fetchval(
                """
                INSERT INTO native_event_sequences (thread_id, last_sequence)
                VALUES ($1, 1)
                ON CONFLICT (thread_id) DO UPDATE
                    SET last_sequence = native_event_sequences.last_sequence + 1
                RETURNING last_sequence
                """,
                session_id,
            )
            return int(value or 1)

        return await self._run(operation)

    async def save_event(self, event: Event) -> None:
        async def operation(conn: Any) -> None:
            async with conn.transaction():
                run_uuid = await _ensure_run(conn, event.session_id, event.run_id or "session")
                await conn.execute(
                    """
                    INSERT INTO agent_events (id, run_id, sequence_number, event_type, payload, created_at)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                    ON CONFLICT (run_id, sequence_number) DO NOTHING
                    """,
                    _uuid_for("event", f"{event.session_id}:{event.sequence}"),
                    run_uuid,
                    event.sequence,
                    event.type,
                    json.dumps({**event.data, "native_run_id": event.run_id}, default=str),
                    event.time,
                )
                await _persist_event_details(conn, run_uuid, event)
        await self._run(operation)

    async def load_events(self, session_id: str, after_sequence: int = 0) -> list:
        rows = await self._fetch(
            """
            SELECT at.thread_id AS session_id, ae.sequence_number AS sequence,
                   ae.event_type AS type, COALESCE(NULLIF(ae.payload->>'native_run_id', ''), ar.metadata->>'native_run_id', '') AS run_id,
                   ae.payload AS data, ae.created_at AS time
            FROM agent_events ae
            JOIN agent_runs ar ON ar.id = ae.run_id
            JOIN agent_tasks at ON at.id = ar.task_id
            WHERE at.thread_id = $1 AND ae.sequence_number > $2
            ORDER BY ae.sequence_number
            """,
            session_id,
            after_sequence,
        )
        return [
            Event(
                sequence=row["sequence"],
                type=row["type"],
                session_id=row["session_id"],
                run_id=row["run_id"],
                data=_load_json(row["data"], {}),
                time=row["time"],
            )
            for row in rows
        ]

    # -- runs --------------------------------------------------------------
    async def save_run(self, run: Any) -> None:
        async def operation(conn: Any) -> None:
            async with conn.transaction():
                native_run_id = str(getattr(run, "run_id", "") or "")
                session_id = str(getattr(run, "session_id", "") or "")
                run_uuid = await _ensure_run(conn, session_id, native_run_id)
                status = _canonical_run_status(str(getattr(run, "status", "")))
                receipt = _run_metadata(run)
                duration = float(getattr(run, "duration_seconds", 0.0) or 0.0)
                await conn.execute(
                    """
                    UPDATE agent_runs SET status=$2::run_status, output=$3,
                        last_error=$4, retry_count=$5, metadata=$6::jsonb,
                        started_at=now() - ($7 * interval '1 second'),
                        finished_at=now()
                    WHERE id=$1
                    """,
                    run_uuid,
                    status,
                    str(getattr(run, "final_text", "") or "") or None,
                    str(getattr(run, "error", "") or "") or None,
                    int(getattr(run, "retries", 0) or 0),
                    json.dumps(receipt),
                    duration,
                )
                task_status = _canonical_task_status(str(getattr(run, "status", "")))
                await conn.execute(
                    "UPDATE agent_tasks SET status=$2::task_status WHERE id=(SELECT task_id FROM agent_runs WHERE id=$1)",
                    run_uuid,
                    task_status,
                )
                await conn.execute(
                    "DELETE FROM llm_calls WHERE run_id=$1 AND node_name IN ('native_receipt','native_turn')",
                    run_uuid,
                )
                turns = max(1, int(getattr(run, "turns", 0) or 0))
                input_tokens = int(getattr(run, "input_tokens", 0) or 0)
                output_tokens = int(getattr(run, "output_tokens", 0) or 0)
                cost = float(getattr(run, "cost_usd", 0.0) or 0.0)
                for turn in range(turns):
                    await conn.execute(
                        """
                        INSERT INTO llm_calls
                            (id, run_id, node_name, model, prompt_tokens, completion_tokens,
                             cost, error, started_at, finished_at)
                        VALUES ($1,$2,'native_turn',$3,$4,$5,$6,$7,
                                now() - ($8 * interval '1 second'),now())
                        """,
                        _uuid_for("llm-call", f"{native_run_id}:{turn}"),
                        run_uuid,
                        str(getattr(run, "model", "") or "") or None,
                        _share(input_tokens, turns, turn),
                        _share(output_tokens, turns, turn),
                        cost / turns,
                        str(getattr(run, "error", "") or "") or None,
                        duration,
                    )
        await self._run(operation)

    async def get_run(self, run_id: str) -> Any:
        """One run's receipt, rebuilt into a `RunRecord`, or None if there isn't one."""
        row = await self._fetchrow(
            """
             SELECT ar.metadata, at.thread_id AS session_id, ar.status, ar.output, ar.last_error,
                   ar.retry_count, ar.duration_ms, lc.model, lc.prompt_tokens,
                   lc.completion_tokens, lc.cost
            FROM agent_runs ar
            JOIN agent_tasks at ON at.id=ar.task_id
            LEFT JOIN LATERAL (
                SELECT max(model) AS model, sum(prompt_tokens) AS prompt_tokens,
                       sum(completion_tokens) AS completion_tokens, sum(cost) AS cost
                FROM llm_calls WHERE run_id=ar.id AND node_name='native_turn'
            ) lc ON true
            WHERE ar.metadata->>'native_run_id' = $1
            """,
            run_id,
        )
        return _row_to_run(row) if row is not None else None

    async def list_runs(self, session_id: str = "", limit: int = 0) -> list:
        """Receipts newest first. `created_at DESC` is the order; `run_id` breaks a tie.

        The tie-breaker matters for the same reason `messages.ordinal` does: two
        runs can share a `created_at` to the microsecond, and a report that lists
        them in a shuffling order every time is not a report you can diff.
        """
        sql = """
             SELECT ar.metadata, at.thread_id AS session_id, ar.status, ar.output, ar.last_error,
                   ar.retry_count, ar.duration_ms, lc.model, lc.prompt_tokens,
                   lc.completion_tokens, lc.cost
            FROM agent_runs ar
            JOIN agent_tasks at ON at.id=ar.task_id
            LEFT JOIN LATERAL (
                SELECT max(model) AS model, sum(prompt_tokens) AS prompt_tokens,
                       sum(completion_tokens) AS completion_tokens, sum(cost) AS cost
                FROM llm_calls WHERE run_id=ar.id AND node_name='native_turn'
            ) lc ON true
        """
        args: list = []
        if session_id:
            sql += "WHERE at.thread_id = $1 "
            args.append(session_id)
        sql += "ORDER BY ar.started_at DESC NULLS LAST, ar.id DESC"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"  # int-cast, so no value reaches SQL unchecked
        rows = await self._fetch(sql, *args)
        return [_row_to_run(row) for row in rows]

    # -- permissions -------------------------------------------------------
    async def save_permission(self, grant: Any) -> None:
        await self._execute(
            "INSERT INTO native_permission_grants "
            "(tool_pattern, duration, thread_id, argument_pattern) "
            "VALUES ($1, $2, $3, $4)",
            getattr(grant, "tool_pattern", ""),
            _duration_value(getattr(grant, "duration", PermissionDuration.ONCE)),
            getattr(grant, "session_id", "") or None,
            getattr(grant, "argument_pattern", "") or "",
        )

    async def load_permissions(self, session_id: str) -> list:
        """Grants for this session, plus the global ones (empty session_id)."""
        rows = await self._fetch(
            "SELECT tool_pattern, duration, thread_id AS session_id, argument_pattern "
            "FROM native_permission_grants "
            "WHERE (thread_id = $1 OR thread_id IS NULL) AND consumed_at IS NULL",
            session_id,
        )
        return [
            PermissionGrant(
                tool_pattern=row["tool_pattern"],
                duration=PermissionDuration(row["duration"]),
                session_id=row["session_id"] or "",
                argument_pattern=row["argument_pattern"] or "",
            )
            for row in rows
        ]

    # -- memories ----------------------------------------------------------
    async def save_memory(self, memory: Memory) -> None:
        """Store a note. Re-saving the same id updates the text rather than duplicating."""
        await self._execute(
            """
            INSERT INTO memory_items
                (id, scope, thread_id, memory_type, content, metadata, created_at, last_accessed_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                memory_type     = EXCLUDED.memory_type,
                content         = EXCLUDED.content,
                last_accessed_at = EXCLUDED.last_accessed_at
            """,
            _uuid_for("memory", memory.id),
            "thread" if memory.session_id else "global",
            memory.session_id or None,
            memory.kind,
            memory.text,
            json.dumps({"native_memory_id": memory.id}),
            memory.created_at,
            memory.last_used_at,
        )

    async def load_memories(self, session_id: str = "") -> list:
        """This session's notes plus the unscoped ones. Scoring happens in Python."""
        rows = await self._fetch(
            "SELECT id, scope, thread_id, memory_type, content, metadata, "
            "created_at, last_accessed_at FROM memory_items "
            "WHERE scope = 'global' OR $1 = '' OR thread_id = $1 "
            "ORDER BY last_accessed_at DESC",
            session_id or "",
        )
        return [
            Memory(
                id=_load_json(row["metadata"], {}).get("native_memory_id", str(row["id"])),
                session_id=row["thread_id"] or "",
                kind=row["memory_type"],
                text=row["content"],
                created_at=row["created_at"],
                last_used_at=row["last_accessed_at"],
            )
            for row in rows
        ]

    async def touch_memory(self, memory_id: str, when: datetime) -> None:
        await self._execute(
            "UPDATE memory_items SET last_accessed_at = $2 "
            "WHERE metadata->>'native_memory_id' = $1",
            memory_id,
            when,
        )


# ---------------------------------------------------------------------------
# Turning messages into JSON and back
# ---------------------------------------------------------------------------
def _part_to_json(part: Any) -> dict:
    """One message part as a plain dict. `kind` is what tells them apart on the way back."""
    if isinstance(part, Text):
        return {"kind": "text", "text": part.text}
    if isinstance(part, Reasoning):
        return {"kind": "reasoning", "text": part.text, "hidden": part.hidden}
    if isinstance(part, ToolCall):
        return {
            "kind": "tool_call",
            "id": part.id,
            "name": part.name,
            "arguments": part.arguments,
            "status": part.status.value,
            "output": part.output,
            "error": part.error,
        }
    if isinstance(part, Compaction):
        return {
            "kind": "compaction",
            "summary": part.summary,
            "old_messages": part.old_messages,
            "tokens_before": part.tokens_before,
            "tokens_after": part.tokens_after,
        }
    if isinstance(part, Media):
        return {
            "kind": "media",
            "data": part.data,
            "mime_type": part.mime_type,
            "detail": part.detail,
        }
    # An unknown part is stored as its text rather than dropped: a conversation
    # that loses a message is worse than one that loses some formatting.
    return {"kind": "text", "text": str(getattr(part, "text", ""))}


def _part_from_json(data: dict) -> Any:
    kind = data.get("kind", "text")
    if kind == "reasoning":
        return Reasoning(data.get("text", ""), hidden=data.get("hidden", True))
    if kind == "tool_call":
        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments") or {},
            status=ToolCallStatus(data.get("status", "pending")),
            output=data.get("output", "") or "",
            error=data.get("error", "") or "",
        )
    if kind == "compaction":
        return Compaction(
            summary=data.get("summary", ""),
            old_messages=data.get("old_messages") or [],
            tokens_before=data.get("tokens_before", 0),
            tokens_after=data.get("tokens_after", 0),
        )
    if kind == "media":
        return Media(
            data=data.get("data", ""),
            mime_type=data.get("mime_type", "application/octet-stream"),
            detail=data.get("detail", ""),
        )
    return Text(data.get("text", ""))


def _usage_to_json(usage: Usage | None) -> dict | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_tokens": usage.cached_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _row_to_message(row: Any) -> Message:
    usage_data = _load_json(row["usage"], None)
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=Role(row["role"]),
        parts=[_part_from_json(p) for p in _load_json(row["parts"], [])],
        model=row["model"],
        usage=Usage(**usage_data) if usage_data else None,
        created_at=row["created_at"] or datetime.now(UTC),
    )


def _row_to_run(row: Any) -> Any:
    """A `runs` row back into a `RunRecord`.

    `RunRecord` is imported here rather than at module top so `postgres.py` keeps
    importing on a machine that never pulls in the loop, matching the lazy-asyncpg
    rule above. `created_at` is read only for ordering, so it has no field here.
    """
    from .loop import RunRecord

    metadata = _load_json(row["metadata"], {})
    return RunRecord(
        run_id=metadata.get("native_run_id", ""),
        session_id=row["session_id"],
        status=metadata.get("native_status", row["status"]),
        final_text=row["output"] or "",
        turns=int(metadata.get("turns", 0) or 0),
        input_tokens=int(row["prompt_tokens"] or 0),
        output_tokens=int(row["completion_tokens"] or 0),
        error=row["last_error"] or "",
        cached_tokens=int(metadata.get("cached_tokens", 0) or 0),
        duration_seconds=float(row["duration_ms"] or 0) / 1000.0,
        cost_usd=float(row["cost"] or 0.0),
        model=row["model"] or "",
        retries=row["retry_count"],
        reasoning_tokens=int(metadata.get("reasoning_tokens", 0) or 0),
        trace_id=metadata.get("trace_id", "") or "",
    )


def _load_json(value: Any, default: Any) -> Any:
    """asyncpg hands back JSONB as a string unless a codec is set. Accept both."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _duration_value(duration: Any) -> str:
    return duration.value if isinstance(duration, PermissionDuration) else str(duration)


def _uuid_for(kind: str, value: str) -> uuid.UUID:
    return uuid.uuid5(_ID_NAMESPACE, f"{kind}:{value}")


def _native_actor_id() -> uuid.UUID:
    return _uuid_for("actor", "agent-native")


def _session_metadata(session: Session) -> dict:
    return {
        "agent": session.agent,
        "working_directory": session.working_directory,
        "revision": session.revision,
    }


def _row_to_session(row: Any) -> Session:
    metadata = _load_json(row["metadata"], {})
    return Session(
        id=row["id"],
        agent=metadata.get("agent", "build"),
        title=row["title"] or "",
        working_directory=metadata.get("working_directory", "."),
        revision=int(metadata.get("revision", 0) or 0),
    )


def _canonical_run_status(status: str) -> str:
    return {
        "finished": "completed",
        "completed": "completed",
        "error": "failed",
        "failed": "failed",
        "cancelled": "interrupted",
        "limit_reached": "interrupted",
        "interrupted": "interrupted",
        "running": "running",
    }.get(status, "created")


def _canonical_task_status(status: str) -> str:
    return {
        "completed": "completed",
        "failed": "failed",
        "interrupted": "interrupted",
        "running": "executing",
    }.get(_canonical_run_status(status), "planning")


def _run_metadata(run: Any) -> dict:
    return {
        "native_run_id": str(getattr(run, "run_id", "") or ""),
        "native_status": str(getattr(run, "status", "") or ""),
        "turns": int(getattr(run, "turns", 0) or 0),
        "cached_tokens": int(getattr(run, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(run, "reasoning_tokens", 0) or 0),
        "trace_id": str(getattr(run, "trace_id", "") or ""),
    }


def _share(total: int, count: int, index: int) -> int:
    """Split an integer exactly; the final row receives the remainder."""
    base = total // count
    return total - base * (count - 1) if index == count - 1 else base


async def _persist_event_details(conn: Any, run_id: uuid.UUID, event: Event) -> None:
    if event.type not in ("tool_started", "tool_finished"):
        return
    name = str(event.data.get("name", "") or "unknown_tool")
    call_id = str(event.data.get("call_id", "") or f"event-{event.sequence}")
    server_id = _uuid_for("mcp-server", "agent-native-runtime")
    tool_id = _uuid_for("tool", f"agent-native-runtime:{name}")
    tool_call_id = _uuid_for("tool-call", f"{run_id}:{call_id}")
    await conn.execute(
        "INSERT INTO mcp_servers (id, name, enabled) "
        "VALUES ($1, 'agent-native-runtime', true) ON CONFLICT (id) DO NOTHING",
        server_id,
    )
    await conn.execute(
        """
        INSERT INTO tools (id, server_id, name, last_seen_at)
        VALUES ($1,$2,$3,$4)
        ON CONFLICT (server_id, name) DO UPDATE SET last_seen_at=EXCLUDED.last_seen_at
        """,
        tool_id,
        server_id,
        name,
        event.time,
    )
    if event.type == "tool_started":
        await conn.execute(
            """
            INSERT INTO tool_calls (id, run_id, tool_id, attempt, arguments, started_at)
            VALUES ($1,$2,$3,1,$4::jsonb,$5)
            ON CONFLICT (id) DO UPDATE SET
                arguments=EXCLUDED.arguments, started_at=EXCLUDED.started_at
            """,
            tool_call_id,
            run_id,
            tool_id,
            json.dumps(event.data.get("arguments") or {}),
            event.time,
        )
        return
    await conn.execute(
        """
        INSERT INTO tool_calls
            (id, run_id, tool_id, attempt, success, output, error, started_at, finished_at)
        VALUES ($1,$2,$3,1,$4,$5::jsonb,$6,$7,$7)
        ON CONFLICT (id) DO UPDATE SET
            success=EXCLUDED.success, output=EXCLUDED.output,
            error=EXCLUDED.error, finished_at=EXCLUDED.finished_at
        """,
        tool_call_id,
        run_id,
        tool_id,
        bool(event.data.get("success", False)),
        json.dumps(event.data.get("output")),
        str(event.data.get("error", "") or "") or None,
        event.time,
    )


async def _ensure_run(conn: Any, session_id: str, native_run_id: str) -> uuid.UUID:
    actor_id = _native_actor_id()
    await conn.execute(
        "INSERT INTO actors (id, kind, external_id, display_name) "
        "VALUES ($1, 'agent', 'agent-native', 'agent-native') ON CONFLICT (id) DO NOTHING",
        actor_id,
    )
    await conn.execute(
        "INSERT INTO agent_threads (id, owner_actor_id, metadata) "
        "VALUES ($1, $2, '{}'::jsonb) ON CONFLICT (id) DO NOTHING",
        session_id,
        actor_id,
    )
    config_json = json.dumps({"source": "agent-native"}, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    config_id = _uuid_for("config", config_hash)
    await conn.execute(
        "INSERT INTO config_snapshots (id, content_hash, behaviour_config) "
        "VALUES ($1, $2, $3::jsonb) ON CONFLICT (content_hash) DO NOTHING",
        config_id,
        config_hash,
        config_json,
    )
    task_id = _uuid_for("task", f"{session_id}:{native_run_id}")
    run_id = _uuid_for("run", f"{session_id}:{native_run_id}")
    await conn.execute(
        """
        INSERT INTO agent_tasks (id, thread_id, goal, track, status, metadata)
        VALUES ($1,$2,$3,'native','executing',$4::jsonb)
        ON CONFLICT (id) DO NOTHING
        """,
        task_id,
        session_id,
        f"Native run {native_run_id}",
        json.dumps({"native_run_id": native_run_id}),
    )
    await conn.execute(
        """
        INSERT INTO agent_runs (id, task_id, config_snapshot_id, status, metadata, started_at)
        VALUES ($1,$2,$3,'running',$4::jsonb,now())
        ON CONFLICT (id) DO NOTHING
        """,
        run_id,
        task_id,
        config_id,
        json.dumps({"native_run_id": native_run_id, "native_status": "running"}),
    )
    return run_id
