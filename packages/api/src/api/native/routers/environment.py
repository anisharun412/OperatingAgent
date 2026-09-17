"""Native environment — which env vars drive the agent.

An explicit allowlist, read live from ``os.environ`` on every call. Secrets
(API keys, database URLs) are never returned by value — only whether they are
set — so this stays safe to render in the desktop Analytics tab.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

router = APIRouter(prefix="/native", tags=["native-environment"])

#: Shown with their effective values.
_VISIBLE_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_TOP_P",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT_SECONDS",
    "AGENT_WORKSPACE",
    "AGENT_SANDBOX_ENABLED",
    "AGENT_SANDBOX_IMAGE",
    "AGENT_SANDBOX_FALLBACK",
    "AGENT_MAX_ITERATIONS",
    "AGENT_RETRY_ATTEMPTS",
    "AGENT_STREAM",
    "AGENT_ENABLE_INTERRUPTS",
    "AGENT_CHECKPOINT_BACKEND",
    "AGENT_CHECKPOINT_NAMESPACE",
    "AGENT_PROMPT_DIR",
    "AGENT_PLANNER_PROMPT",
    "AGENT_VERIFIER_PROMPT",
    "AGENT_RESPONDER_PROMPT",
    "AGENT_PERMISSION_FILE_SYSTEM",
    "AGENT_PERMISSION_TERMINAL",
    "AGENT_PERMISSION_GIT",
    "AGENT_PERMISSION_SEARCH",
    "AGENT_PERMISSION_KNOWLEDGE",
    "AGENT_PERMISSION_MEMORY",
    "API_DEFAULT_TRACK",
    "API_APPROVAL_THRESHOLD",
    "API_REPOSITORY_BACKEND",
    "API_REPOSITORY_FALLBACK",
    "OPERATING_AGENT_DATA_DIR",
    "OPERATING_AGENT_WORKSPACE",
    "SQLITE_DATABASE_PATH",
    "MCP_GATEWAY_COMMAND",
)

#: Presence only — values never leave the process.
_SECRET_VARS = (
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DATABASE_URL",
    "LLM_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "MCP_GATEWAY_ARGS",
)


@router.get("/environment")
async def environment() -> dict[str, object]:
    variables: list[dict[str, object]] = []
    for name in _VISIBLE_VARS:
        value = os.environ.get(name, "").strip()
        variables.append({"name": name, "value": value or None, "secret": False, "set": bool(value)})
    variables.extend(
        {"name": name, "value": None, "secret": True, "set": bool(os.environ.get(name, "").strip())}
        for name in _SECRET_VARS
    )
    return {"variables": variables}
