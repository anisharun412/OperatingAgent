# api

The HTTP / WebSocket entry point for OperatingAgent. It turns a submitted goal
into a background agent run, persists the run to the configured Postgres or
SQLite system-of-record (or an in-memory store in dev/CI), and streams execution events back to clients
over Server-Sent Events and WebSocket.

It implements the `packages/api` collaborators from
`docs/architecture/class-diagram.mermaid` — `TaskService`, `ApprovalGateway`,
`TaskRepository`, `TaskRouter` (the `/tasks` routes) and `TaskStreamSocket`
(the streaming routes) — and dispatches to an `IAgentOrchestrator` (the
LangGraph and `agent_native` tracks) without modifying `common`, `agent-langgraph`
or `observability`.

## Run it

```bash
uv sync
docker compose --env-file .env -f infra/docker/docker-compose.yml up -d postgres
uv run api                       # loads .env, then starts on 127.0.0.1:8000
```

For the desktop path, PostgreSQL and Docker are optional. Set
`API_REPOSITORY_BACKEND=sqlite` in `.env` to use the durable local SQLite file,
then start the Tauri shell from `apps/desktop` with `npm run tauri -- dev`.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/tasks \
  -H 'content-type: application/json' \
  -d '{"goal":"say hi","track":"native","workspace":"."}'
curl http://127.0.0.1:8000/threads
curl http://127.0.0.1:8000/threads/<thread-id>/tasks
curl http://127.0.0.1:8000/threads/<thread-id>/tasks/<task-id>
curl -X POST http://127.0.0.1:8000/threads/<thread-id>/tasks \
  -H 'content-type: application/json' \
  -d '{"goal":"follow up"}'
curl -N http://127.0.0.1:8000/tasks/<id>/events      # SSE stream
```

The **native** track responds with no LLM credentials configured. The
**langgraph** track boots whenever its model integration imports cleanly; a run
degrades to a `FAILED` result (never a crash) when no live model/gateway is
reachable.

The LangGraph track launches `gateway_server` itself over FastMCP stdio. Do not
start a separate HTTP MCP process for normal API use.

## Configuration (environment)

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_HOST` | `127.0.0.1` | bind host |
| `API_PORT` | `8000` | bind port |
| `API_LOG_LEVEL` | `info` | uvicorn log level |
| `DATABASE_URL` | _unset_ | Postgres DSN; when set the repository defaults to `postgres` |
| `API_REPOSITORY_BACKEND` | `postgres` if `DATABASE_URL` else `memory` | `memory` \| `sqlite` \| `postgres` |
| `API_REPOSITORY_FALLBACK` | `error` (`sqlite` in `.env.example`) | Fallback when Postgres cannot start: `sqlite`, `memory`, or `error` |
| `API_REPOSITORY_CONNECT_TIMEOUT_SECONDS` | `5` | Seconds to wait for Postgres before applying the fallback |
| `OPERATING_AGENT_DATA_DIR` | platform application-data directory | Parent directory for the desktop SQLite file |
| `SQLITE_DATABASE_PATH` | `<data dir>/operating-agent.db` | Explicit SQLite file used by API, native, and LangGraph |
| `API_CORS_ORIGINS` | Tauri production origins plus `localhost:1420` | comma-separated frontend origins allowed to call the API |
| `API_ALLOWED_HOSTS` | `127.0.0.1,localhost,testserver` | accepted HTTP `Host` values; prevents DNS rebinding against the local service |
| `API_DEFAULT_TRACK` | `langgraph` | `native` \| `langgraph` used when a request omits `track` |
| `API_APPROVAL_THRESHOLD` | `review` | lowest risk level that requires a human gate |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_BASE_URL` | `ollama` / `llama3.1` / _unset_ | model for the langgraph track |
| `LLM_TIMEOUT_SECONDS` / `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TOP_P` | `60` / `0` / _unset_ / `1` | model generation controls |
| `MCP_GATEWAY_COMMAND` / `MCP_GATEWAY_ARGS` | current Python / `-m gateway_server` | stdio command the LangGraph track launches for tools |
| `AGENT_PROMPT_DIR` | packaged prompts | directory holding `planner.txt` / `verifier.txt` / `responder.txt` |
| `AGENT_PLANNER_PROMPT` / `AGENT_VERIFIER_PROMPT` / `AGENT_RESPONDER_PROMPT` | _unset_ | exact prompt paths overriding `AGENT_PROMPT_DIR` |
| `AGENT_MAX_ITERATIONS` / `AGENT_EXECUTION_TIMEOUT_SECONDS` / `AGENT_RETRY_ATTEMPTS` | `20` / `300` / `2` | graph and tool execution limits |
| `AGENT_STREAM` / `AGENT_ENABLE_CHECKPOINTS` / `AGENT_ENABLE_INTERRUPTS` | `true` / `true` / `true` | execution mode and resumability controls |
| `AGENT_REQUIRE_VERIFICATION` / `AGENT_REQUIRE_HUMAN_APPROVAL` | `false` / `true` | semantic verification and approval gates |
| `AGENT_CHECKPOINT_BACKEND` / `AGENT_CHECKPOINT_NAMESPACE` | `auto` / `default` | checkpoint storage and namespace |
| `AGENT_SANDBOX_ENABLED` / `AGENT_WORKSPACE` | `true` / `./workspace` | filesystem tool confinement |
| `AGENT_SANDBOX_IMAGE` | `operating-agent-sandbox:py312` | Docker image used for terminal tools |
| `AGENT_SANDBOX_FALLBACK` | `host` | `host` runs terminal commands on the host when Docker is unavailable; `error` fails closed instead |
| `AGENT_PERMISSION_*` | `true` | category switches for filesystem, terminal, git, search, knowledge, and memory tools |

Langfuse tracing follows the shared `observability` package: it is enabled only
when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present.

## Enable the Docker sandbox

The sandbox is enabled by default for both agent tracks, but Docker Desktop and
the project image must be available to the API process. From the repository
root, run:

```powershell
docker desktop start
docker build -t operating-agent-sandbox:py312 .\infra\sandbox-images
New-Item -ItemType Directory -Force .\workspace
```

In `.env`, keep these values (or point them at an existing directory and a
custom image):

```dotenv
AGENT_SANDBOX_ENABLED=true
AGENT_WORKSPACE=./workspace
AGENT_SANDBOX_IMAGE=operating-agent-sandbox:py312
```

Restart `uv run api` after changing environment values. The desktop Settings
panel polls `GET /native/sandbox`; it should show the image and `sandbox: on`.
Terminal commands are then executed in a disposable container with the chosen
workspace mounted at `/workspace`, no network, a read-only root filesystem,
and the configured resource limits. If Docker or the image is unavailable, the
pool degrades to running the command on the host in the session workspace
(`sandbox: degraded ...` in the status panel) so the agent keeps working; set
`AGENT_SANDBOX_FALLBACK=error` to restore the fail-closed behavior instead.

## HTTP security

The API is designed as a loopback service launched by the Tauri application, so
it does not require an application API key. Keep `API_HOST=127.0.0.1`; do not bind
to `0.0.0.0` unless the desktop-only trust model is replaced with authentication.

Browser access is restricted to `API_CORS_ORIGINS`, and requests with an
untrusted `Host` header are rejected through `API_ALLOWED_HOSTS`. Responses also
carry no-store, MIME-sniffing, framing, referrer, and permissions-policy headers.
Set the exact Tauri/Vite origin in `API_CORS_ORIGINS` rather than using `*`.

## Threads

`GET /threads` returns conversation summaries ordered by most recent activity.
`POST /tasks` creates a task in a newly generated thread; the response contains
the generated `thread_id`. `POST /threads/{thread_id}/tasks` appends a task to an
existing thread and does not accept a thread id in its body.
`GET /threads/{thread_id}/tasks` returns that thread's tasks with their latest
run status, newest first. Both endpoints accept `limit` (1-500, default 100) and
`offset` (default 0) query parameters.
`GET /threads/{thread_id}/tasks/{task_id}` is the canonical task lookup and
validates that the task belongs to that thread. The unscoped task lookup is not
available.

## Persistence

The local PostgreSQL service is defined in `infra/docker/docker-compose.yml` and
initializes `infra/docker/postgres/schema.sql` on its first startup. Copy
`.env.example` to `.env`, start the service, and the API will select PostgreSQL
through `DATABASE_URL` and `API_REPOSITORY_BACKEND=postgres`.

`memory` remains available for hermetic use. With
`API_REPOSITORY_FALLBACK=sqlite`, an unavailable Postgres/Docker service is logged
and the API continues with a durable SQLite repository. Set the fallback to
`memory` for an ephemeral run or `error` when durable Postgres persistence is
required. `postgres` writes the run spine —
`actors → agent_threads → agent_tasks → agent_runs → agent_events`, plus the
content-addressed `config_snapshots` row each run points at. State, message,
tool-action, and approval lifecycle events use the existing `agent_events` JSON
payload, so no parallel conversation-memory table is required. Secrets
(`llm.api_key`, `checkpoint.connection_string`) are redacted from the config
snapshot **and** from its content hash before anything is written.

LangGraph uses the same `DATABASE_URL` for Postgres checkpoints, or the configured
`SQLITE_DATABASE_PATH` for the SQLite checkpointer. Its saver owns the checkpoint
table migrations and creates them when it first opens; do not add those tables to
the application schema.

## Windows note

The psycopg async pool requires a Selector event loop on Windows. The `api`
console launcher passes that loop factory directly to Uvicorn, so `uv run api`
works with the PostgreSQL repository instead of selecting the incompatible
Proactor loop.
## Runtime persistence boundaries

- `POST /threads/{thread_id}/tasks/{id}/resume` resumes the latest LangGraph checkpoint. Pass
  `resume_value` for a pending LangGraph interrupt and optionally `checkpoint_id`
  to target a specific snapshot.
- A task submitted through an existing thread is treated as a new conversational
  turn. The prior transcript remains in the checkpoint and the new goal is
  appended as a human message while active plan/verdict state resets.
- Persisted events are hydrated into SSE/WebSocket replay after an API restart.
- Langfuse remains authoritative for LLM generations, token accounting, and
  evaluation. Local events retain workflow, tool, and approval actions for
  recovery and audit.
- External tools should treat deterministic `call_id` values in
  `tool_started` / `tool_finished` events as idempotency keys for non-repeatable
  side effects.

## Native track

The native implementation has its own session-oriented API under `/native` and
uses the same PostgreSQL deployment plus numbered native migrations. The shared
`/tasks` API remains available for both registered tracks; native-specific
conversation, permissions, and run-history endpoints use `AgentService`.

Native runs expose `trace_id` on `RunResponse` and in the terminal event when
Langfuse is enabled. Use that id to find the run in Cloud Langfuse; the SDK's
`flush()` confirms delivery, while Cloud ingestion and UI search can lag by
roughly 15-30 seconds.
