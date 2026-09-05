"""SQL for the Postgres run spine, kept out of the repository logic.

Each statement targets exactly the columns this change writes and leans on the
schema's server-side defaults for everything else (``gen_random_uuid()`` ids,
``now()`` timestamps, the generated ``duration_ms``). Enum-typed columns take a
text value with an explicit ``::enum`` cast; jsonb columns take a value wrapped
in ``psycopg.types.json.Jsonb``.
"""

from __future__ import annotations

# One shared service identity owns API-created threads. external_id is UNIQUE,
# so DO UPDATE (a no-op touch) lets RETURNING yield the id on conflict too.
UPSERT_ACTOR = """
INSERT INTO actors (kind, external_id, display_name)
VALUES ('system', %s, %s)
ON CONFLICT (external_id) DO UPDATE SET display_name = EXCLUDED.display_name
RETURNING id
"""

# thread id is TEXT and equals the LangGraph thread_id we generate.
UPSERT_THREAD = """
INSERT INTO agent_threads (id, owner_actor_id, title)
VALUES (%s, %s, %s)
ON CONFLICT (id) DO UPDATE SET updated_at = now()
RETURNING id
"""

INSERT_TASK = """
INSERT INTO agent_tasks (id, thread_id, goal, track, status, metadata)
VALUES (%s, %s, %s, %s::agent_track, %s::task_status, %s)
ON CONFLICT (id) DO UPDATE SET goal = EXCLUDED.goal
RETURNING id
"""

# Content-addressed: the no-op DO UPDATE makes RETURNING fire on conflict, so a
# repeated config reuses its existing snapshot row.
UPSERT_CONFIG_SNAPSHOT = """
INSERT INTO config_snapshots (
    content_hash, llm_config, execution_config, sandbox_config,
    permissions_config, checkpoint_config, tracing_config,
    behaviour_config, prompts_config
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (content_hash) DO UPDATE SET content_hash = EXCLUDED.content_hash
RETURNING id
"""

# attempt is derived so a retried task (a second run) gets attempt 2 and does
# not collide with UNIQUE (task_id, attempt).
INSERT_RUN = """
INSERT INTO agent_runs (task_id, attempt, config_snapshot_id, status, metadata)
VALUES (
    %s,
    (SELECT COALESCE(MAX(attempt), 0) + 1 FROM agent_runs WHERE task_id = %s),
    %s,
    %s::run_status,
    %s
)
RETURNING id
"""

MARK_RUN_RUNNING = """
UPDATE agent_runs
SET status = 'running'::run_status, started_at = now()
WHERE id = %s
"""

INSERT_EVENT = """
INSERT INTO agent_events (run_id, sequence_number, event_type, payload)
VALUES (%s, %s, %s, %s)
ON CONFLICT (run_id, sequence_number) DO NOTHING
"""

FINALIZE_RUN = """
UPDATE agent_runs
SET status = %s::run_status,
    output = %s,
    last_error = %s,
    metadata = metadata || %s,
    finished_at = now()
WHERE id = %s
"""

UPDATE_TASK_STATUS = """
UPDATE agent_tasks SET status = %s::task_status WHERE id = %s
"""

SELECT_TASK = """
SELECT id, thread_id, goal, track, metadata, created_at
FROM agent_tasks
WHERE id = %s
"""

SELECT_THREADS = """
SELECT
    thread.id,
    thread.title,
    COUNT(task.id),
    thread.created_at,
    thread.updated_at
FROM agent_threads AS thread
JOIN actors AS owner ON owner.id = thread.owner_actor_id
LEFT JOIN agent_tasks AS task ON task.thread_id = thread.id
WHERE owner.external_id = %s
GROUP BY thread.id
ORDER BY thread.updated_at DESC, thread.id DESC
LIMIT %s OFFSET %s
"""

SELECT_THREAD_EXISTS = """
SELECT 1
FROM agent_threads AS thread
JOIN actors AS owner ON owner.id = thread.owner_actor_id
WHERE thread.id = %s AND owner.external_id = %s
"""

SELECT_TASKS_BY_THREAD = """
SELECT
    task.id,
    task.thread_id,
    task.goal,
    task.track,
    task.metadata,
    task.created_at,
    latest_run.status
FROM agent_tasks AS task
LEFT JOIN LATERAL (
    SELECT run.status
    FROM agent_runs AS run
    WHERE run.task_id = task.id
    ORDER BY run.attempt DESC
    LIMIT 1
) AS latest_run ON true
WHERE task.thread_id = %s
ORDER BY task.created_at DESC, task.id DESC
LIMIT %s OFFSET %s
"""

SELECT_LATEST_RUN_STATUS = """
SELECT status FROM agent_runs
WHERE task_id = %s
ORDER BY attempt DESC
LIMIT 1
"""

UPSERT_MCP_SERVER = """
INSERT INTO mcp_servers (name, base_url)
VALUES (%s, %s)
ON CONFLICT (name) DO UPDATE SET base_url = EXCLUDED.base_url, enabled = true
RETURNING id
"""

UPSERT_TOOL = """
INSERT INTO tools (server_id, name, description, input_schema)
VALUES (%s, %s, %s, %s)
ON CONFLICT (server_id, name) DO UPDATE SET
    description = EXCLUDED.description,
    input_schema = EXCLUDED.input_schema,
    last_seen_at = now()
RETURNING id
"""

INSERT_LLM_CALL = """
INSERT INTO llm_calls (
    run_id, node_name, provider, model, prompt_tokens, completion_tokens,
    cost, error, started_at, finished_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

INSERT_TOOL_CALL = """
INSERT INTO tool_calls (
    run_id, tool_id, arguments, success, output, error, risk_level,
    risk_reason, attempt, started_at, finished_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s::risk_level, %s, %s, %s, %s)
"""

INSERT_PHASE = """
INSERT INTO run_phases (id, run_id, sequence, phase, entry_reason, entered_at)
VALUES (COALESCE(%s::uuid, gen_random_uuid()), %s, %s, %s::workflow_phase, %s, COALESCE(%s, now()))
RETURNING id
"""

CLOSE_PHASE = """
UPDATE run_phases SET exited_at = COALESCE(%s, now())
WHERE run_id = %s AND id = %s
"""

INSERT_PLAN = """
INSERT INTO plans (id, run_id, phase_id, revision, summary, reasoning, requires_remediation)
VALUES (COALESCE(%s::uuid, gen_random_uuid()), %s, %s, %s, %s, %s, %s)
RETURNING id
"""

INSERT_PLAN_STEP = """
INSERT INTO plan_steps
    (id, plan_id, run_id, step_number, description, tool_id, arguments, status, output)
VALUES (COALESCE(%s::uuid, gen_random_uuid()), %s, %s, %s, %s, %s, %s,
        COALESCE(%s, 'created')::run_status, %s)
"""

INSERT_FINDING = """
INSERT INTO run_findings
    (id, run_id, phase_id, plan_step_id, description, detail, source_tool_id)
VALUES (COALESCE(%s::uuid, gen_random_uuid()), %s, %s, %s, %s, %s, %s)
RETURNING id
"""

INSERT_VERIFICATION = """
INSERT INTO verification_results
    (id, run_id, plan_step_id, tool_call_id, attempt, result, reason, deterministic, evidence)
VALUES (COALESCE(%s::uuid, gen_random_uuid()), %s, %s, %s, %s,
        %s::verification_verdict, %s, %s, %s)
RETURNING id
"""

INSERT_TRACE_REF = """
INSERT INTO trace_refs (id, run_id, provider, trace_id, metadata)
VALUES (COALESCE(%s::uuid, gen_random_uuid()), %s, %s, %s, %s)
ON CONFLICT (provider, trace_id) DO UPDATE SET metadata = EXCLUDED.metadata
RETURNING id
"""

INSERT_APPROVAL = """
INSERT INTO approval_requests (id, run_id, plan_step_id, reason, expires_at)
VALUES (COALESCE(%s::uuid, gen_random_uuid()), %s, %s, %s, %s)
RETURNING id
"""

RESOLVE_APPROVAL_ACTOR = """
INSERT INTO actors (kind, external_id, display_name)
VALUES ('human', %s, %s)
ON CONFLICT (external_id) DO UPDATE SET display_name = EXCLUDED.display_name
RETURNING id
"""

RESOLVE_APPROVAL = """
UPDATE approval_requests SET
    status = %s, resolved_by_actor_id = %s, decision_note = %s,
    resolved_at = COALESCE(%s, now()), tool_call_id = COALESCE(%s, tool_call_id)
WHERE id = %s
"""

SELECT_LATEST_RUN_ID = """
SELECT id FROM agent_runs
WHERE task_id = %s
ORDER BY attempt DESC
LIMIT 1
"""

SELECT_LATEST_RUN_METADATA = """
SELECT metadata FROM agent_runs
WHERE task_id = %s
ORDER BY attempt DESC
LIMIT 1
"""

SELECT_TASK_EVENTS = """
SELECT event.event_type, event.payload
FROM agent_events AS event
JOIN agent_runs AS run ON run.id = event.run_id
WHERE run.task_id = %s
  AND (%s = false OR run.id = (
      SELECT latest.id FROM agent_runs AS latest
      WHERE latest.task_id = %s
      ORDER BY latest.attempt DESC
      LIMIT 1
  ))
ORDER BY run.attempt ASC, event.sequence_number ASC, event.id ASC
"""

SELECT_THREAD_EVENTS = """
SELECT task.id, event.event_type, event.payload
FROM agent_tasks AS task
JOIN agent_runs AS run ON run.task_id = task.id
JOIN agent_events AS event ON event.run_id = run.id
WHERE task.thread_id = %s
ORDER BY task.created_at ASC, run.attempt ASC,
         event.sequence_number ASC, event.id ASC
"""

SELECT_APPROVAL_STATES = """
WITH ranked AS (
    SELECT event.event_type, event.payload,
           ROW_NUMBER() OVER (
               PARTITION BY event.payload->>'request_id'
               ORDER BY event.created_at DESC, event.id DESC
           ) AS rank
    FROM agent_events AS event
    WHERE event.event_type IN ('approval_requested', 'approval_resolved')
)
SELECT event_type, payload
FROM ranked
WHERE rank = 1
"""

INSERT_APPROVAL_EVENT = """
WITH lock AS (
    SELECT pg_advisory_xact_lock(hashtext(%s::text))
), next_sequence AS (
    SELECT COALESCE(
        (SELECT MAX(sequence_number) FROM agent_events WHERE run_id = %s),
        -1
    ) + 1 AS sequence_number
    FROM lock
)
INSERT INTO agent_events (run_id, sequence_number, event_type, payload)
SELECT %s, next_sequence.sequence_number, %s, %s
FROM next_sequence
"""

SELECT_LATEST_RUN = """
SELECT id, status, output, last_error, metadata
FROM agent_runs
WHERE task_id = %s
ORDER BY attempt DESC
LIMIT 1
"""

# Thread deletion, leaf tables first. The thread row itself is scoped to the
# API-owned actor so one caller can never delete another owner's thread.
DELETE_THREAD_EVENTS = """
DELETE FROM agent_events AS event
USING agent_runs AS run, agent_tasks AS task
WHERE event.run_id = run.id
  AND run.task_id = task.id
  AND task.thread_id = %s
"""

DELETE_THREAD_RUNS = """
DELETE FROM agent_runs AS run
USING agent_tasks AS task
WHERE run.task_id = task.id
  AND task.thread_id = %s
"""

DELETE_THREAD_TASKS = """
DELETE FROM agent_tasks
WHERE thread_id = %s
"""

DELETE_THREAD = """
DELETE FROM agent_threads AS thread
USING actors AS owner
WHERE thread.id = %s
  AND thread.owner_actor_id = owner.id
  AND owner.external_id = %s
"""
