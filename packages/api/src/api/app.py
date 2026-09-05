"""Application factory: wiring, lifespan, CORS and error handling.

``create_app`` builds the ``FastAPI`` instance and includes the routers; the
lifespan constructs the runtime collaborators (repository, orchestrators,
broker, approvals, task service) and tears them down cleanly on shutdown —
cancelling in-flight runs, closing every stream, flushing tracing and closing
the connection pool.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from observability import flush, init_tracing, shutdown
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import ApiSettings
from .environment import load_environment
from .errors import register_exception_handlers
from .langgraph_settings import router as langgraph_settings_router
from .orchestration.factory import build_orchestrators
from .repository.factory import build_repository
from .repository.memory import InMemoryTaskRepository
from .repository.sqlite import SQLiteTaskRepository
from .routers import approvals, health, stream, tasks, threads
from .security import SecurityHeadersMiddleware
from .services.approval_gateway import ApprovalGateway
from .services.event_broker import EventBroker
from .services.task_service import TaskService

# Native-track imports are optional at import-time so the Task API still boots
# even if agent-native is not installed; the lifespan will degrade gracefully.
try:
    from .native import settings as native_settings
    from .native.routers import events as native_events
    from .native.routers import health as native_health
    from .native.routers import messages as native_messages
    from .native.routers import permissions as native_permissions
    from .native.routers import runs as native_runs
    from .native.routers import sessions as native_sessions

    _NATIVE_ROUTERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NATIVE_ROUTERS_AVAILABLE = False
    native_events = native_health = native_messages = native_permissions = native_runs = native_sessions = native_settings = None  # type: ignore

log = logging.getLogger(__name__)

def create_app(settings: ApiSettings | None = None) -> FastAPI:
    if settings is None:
        load_environment()
    resolved_settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal resolved_settings

        def use_fallback(backend: str) -> None:
            """Switch every application consumer to the selected local backend."""
            nonlocal resolved_settings
            resolved_settings = replace(
                resolved_settings,
                database_url=None,
                repository_backend=backend,
                checkpoint_backend=backend,
            )

        def build_fallback_repository():
            fallback = resolved_settings.repository_fallback
            if fallback in {"sqlite", "file", "file-based", "file_based"}:
                use_fallback("sqlite")
                return SQLiteTaskRepository(resolved_settings.sqlite_database_path)
            if fallback in {"memory", "inmemory", "in_memory"}:
                use_fallback("memory")
                return InMemoryTaskRepository()
            raise RuntimeError(
                "Postgres is unavailable and API_REPOSITORY_FALLBACK is not an "
                "explicit supported fallback (sqlite or memory)"
            )

        tracing_client = init_tracing()  # idempotent; no-op without credentials
        log.info(
            "Langfuse tracing %s for API worker",
            "enabled" if tracing_client is not None else "disabled",
        )

        try:
            repository, pool = build_repository(resolved_settings)
        except Exception as exc:
            if (
                resolved_settings.repository_backend == "postgres"
                and resolved_settings.repository_fallback
                in {
                    "memory",
                    "inmemory",
                    "in_memory",
                    "sqlite",
                    "file",
                    "file-based",
                    "file_based",
                }
            ):
                log.warning(
                    "Postgres repository could not be initialized; using %s repository: %s",
                    resolved_settings.repository_fallback,
                    exc,
                )
                repository, pool = build_fallback_repository(), None
            else:
                raise
        if pool is not None:
            # Verify one usable connection before serving requests. Desktop
            # deployments may opt into SQLite or memory explicitly.
            try:
                await asyncio.wait_for(
                    pool.open(wait=True),
                    timeout=resolved_settings.repository_connect_timeout_seconds,
                )
            except Exception as exc:
                try:
                    await pool.close(timeout=1.0)
                except Exception as close_exc:  # noqa: BLE001 - cleanup is best effort
                    log.debug("Postgres pool close after startup failure failed: %s", close_exc)
                if resolved_settings.repository_fallback not in {
                    "memory",
                    "inmemory",
                    "in_memory",
                    "sqlite",
                    "file",
                    "file-based",
                    "file_based",
                }:
                    raise
                log.warning(
                    "Postgres repository unavailable; using %s repository: %s",
                    resolved_settings.repository_fallback,
                    exc,
                )
                repository = build_fallback_repository()
                pool = None

        approval_gateway = ApprovalGateway(
            threshold=resolved_settings.approval_threshold,
            repository=repository,
        )
        await approval_gateway.restore()
        broker = EventBroker()
        background: set[asyncio.Task] = set()

        # Native-track runtime — parallel, isolated from Task repository
        native_runtime = None
        native_service = None
        native_pool = None
        native_background: set[asyncio.Task] = set()
        native_cancels: dict = {}
        try:
            from .native.runtime import (
                build_native_database,
                build_native_sandbox,
                wire_native_models,
            )

            try:
                native_db, native_pool = build_native_database(resolved_settings)
                # For postgres, open the native pool explicitly
                if native_pool is not None and hasattr(native_pool, "connect"):
                    # PostgresDatabase owns asyncpg pool
                    try:
                        await asyncio.wait_for(
                            native_pool.connect(),
                            timeout=resolved_settings.repository_connect_timeout_seconds,
                        )
                    except Exception as exc:
                        # Only fall back in explicit desktop/dev mode; otherwise
                        # preserve the configured failure behavior.
                        fallback = getattr(
                            resolved_settings, "repository_fallback", "memory"
                        ).lower()
                        if fallback in (
                            "memory",
                            "inmemory",
                            "in_memory",
                            "sqlite",
                            "file",
                            "file-based",
                            "file_based",
                        ):
                            log.warning(
                                "Native Postgres connect failed, falling back to %s: %s",
                                fallback,
                                exc,
                            )
                            try:
                                await native_pool.close()
                            except Exception as close_exc:  # noqa: BLE001 - cleanup is best effort
                                log.debug("Native Postgres pool close after startup failure failed: %s", close_exc)
                            if fallback in ("sqlite", "file", "file-based", "file_based"):
                                from agent_native.sqlite import SQLiteDatabase

                                native_db = SQLiteDatabase(
                                    resolved_settings.sqlite_database_path
                                )
                            else:
                                from agent_native.database import MemoryDatabase

                                native_db = MemoryDatabase()
                            native_pool = None
                        else:
                            log.warning("Native Postgres connect failed, marking native runtime unavailable: %s", exc)
                            raise
                # Enforce schema only if the DB reports missing migrations
                if native_pool is not None:
                    apply_schema = getattr(native_db, "apply_schema", None)
                    if callable(apply_schema):
                        try:
                            await apply_schema()  # type: ignore[operator]  # pyright: ignore[reportGeneralTypeIssues]
                        except Exception as exc:  # noqa: BLE001 - schema validation is a startup degradation boundary
                            log.warning("Native schema check failed (continuing): %s", exc)

                from agent_native.config import AgentConfig as NativeAgentConfig
                from agent_native.service import AgentRuntime, AgentService

                native_config = NativeAgentConfig(
                    name="build",
                    model=resolved_settings.llm_model,
                    max_turns=resolved_settings.execution_max_iterations,
                    temperature=resolved_settings.llm_temperature,
                )
                native_runtime = AgentRuntime(
                    database=native_db,
                    agents=[native_config],
                    sandbox=build_native_sandbox(resolved_settings),
                )
                wire_native_models(native_runtime, settings=resolved_settings)
                native_service = AgentService(native_runtime)
                log.info(
                    "Native runtime ready: db=%s agents=%s models=%s",
                    type(native_db).__name__,
                    list(getattr(native_runtime, "agents", {}).keys()),
                    list(getattr(getattr(native_runtime, "models", None), "_models", {}).keys()) if hasattr(getattr(native_runtime, "models", None), "_models") else [],
                )
            except Exception as exc:  # noqa: BLE001 - native track is optional
                log.warning("Native runtime not available: %s", exc)
                native_runtime = None
                native_service = None
        except ImportError as exc:
            log.debug("Native package not importable: %s", exc)

        app.state.native_runtime = native_runtime
        app.state.native_service = native_service
        app.state.native_background = native_background
        app.state.native_cancels = native_cancels
        app.state.native_pool = native_pool

        # Build both tracks after native startup so the shared /tasks endpoint
        # receives the real AgentService-backed native adapter.
        orchestrators = build_orchestrators(
            resolved_settings,
            approval_handler=approval_gateway,
            native_service=native_service,
        )
        app.state.settings = resolved_settings
        app.state.repository = repository
        app.state.broker = broker
        app.state.approvals = approval_gateway
        app.state.background = background
        app.state.task_service = TaskService(
            orchestrators=orchestrators,
            repository=repository,
            broker=broker,
            approvals=approval_gateway,
            settings=resolved_settings,
            background=background,
        )

        try:
            yield
        finally:
            for run_task in list(background):
                run_task.cancel()
            if background:
                await asyncio.gather(*background, return_exceptions=True)
            await broker.aclose_all()
            await asyncio.gather(
                *(orchestrator.aclose() for orchestrator in orchestrators.values()),
                return_exceptions=True,
            )
            # Native teardown — cancel in-flight sends, close events, MCP, DB
            for t in list(native_background):
                t.cancel()
            if native_background:
                await asyncio.gather(*native_background, return_exceptions=True)
            if native_runtime is not None:
                # Close any MCP providers attached lazily
                for prov in getattr(native_runtime, "_mcp_providers", []) or []:
                    try:
                        await prov.close()
                    except Exception as exc:  # noqa: BLE001 - cleanup must not mask shutdown
                        log.debug("Native MCP provider close failed: %s", exc)
                # Close native event bus and DB
                try:
                    await native_runtime.events.close()
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask shutdown
                    log.debug("Native event bus close failed: %s", exc)
                try:
                    await native_runtime.database.close()
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask shutdown
                    log.debug("Native database close failed: %s", exc)
                try:
                    sandbox = getattr(native_runtime, "sandbox", None)
                    close_sandbox = getattr(sandbox, "close", None)
                    if callable(close_sandbox):
                        close_result = close_sandbox()
                        if inspect.isawaitable(close_result):
                            await close_result
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask shutdown
                    log.debug("Native sandbox close failed: %s", exc)
                # Flush monitoring traces if enabled
                try:
                    for _ in native_runtime.monitoring.shutdown():
                        pass
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask shutdown
                    log.debug("Native monitoring shutdown failed: %s", exc)
            elif native_pool is not None:
                try:
                    await native_pool.close()
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask shutdown
                    log.debug("Native pool close failed: %s", exc)
            flush()
            shutdown()
            if pool is not None:
                await pool.close()
            else:
                close_repository = getattr(repository, "close", None)
                if callable(close_repository):
                    close_result = close_repository()
                    if inspect.isawaitable(close_result):
                        await close_result

    app = FastAPI(title="OperatingAgent API", version="0.1.0", lifespan=lifespan)
    # Available before the lifespan runs so get_settings works under test.
    app.state.settings = resolved_settings

    # allow_credentials with a wildcard origin is rejected by browsers, so only
    # enable credentials when the origins are explicitly enumerated.
    wildcard = "*" in resolved_settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(resolved_settings.allowed_hosts),
    )
    app.add_middleware(SecurityHeadersMiddleware)

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(tasks.router)
    app.include_router(threads.router)
    app.include_router(stream.router)
    app.include_router(approvals.router)
    app.include_router(langgraph_settings_router)
    # Native-track routes — mounted separately so existing paths are untouched
    if _NATIVE_ROUTERS_AVAILABLE:
        assert native_health is not None
        assert native_sessions is not None
        assert native_messages is not None
        assert native_events is not None
        assert native_permissions is not None
        assert native_runs is not None
        app.include_router(native_health.router)
        app.include_router(native_sessions.router)
        app.include_router(native_settings.router)
        app.include_router(native_messages.router)
        app.include_router(native_events.router)
        app.include_router(native_permissions.router)
        app.include_router(native_runs.router)
    return app
