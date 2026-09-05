"""Live settings bridge for the LangGraph orchestrator."""

from __future__ import annotations

import os
from typing import Any

from common.config import AgentConfig, LLMConfig
from fastapi import APIRouter, HTTPException, Query, Request

from .settings import (
    RuntimeLLMSettings,
    clean_base_url,
    default_model,
    normalize_base_url,
    normalize_model,
    normalize_provider,
    provider_models,
    resolve_model,
)

router = APIRouter(prefix="/settings/langgraph", tags=["langgraph-settings"])

_LANGGRAPH_PROVIDERS = ("ollama", "groq", "openai", "anthropic")


def _orchestrator(request: Request) -> Any:
    service = getattr(request.app.state, "task_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="task service not initialized")
    orchestrator = service.orchestrators.get("langgraph") if hasattr(service, "orchestrators") else None
    if orchestrator is None:
        orchestrator = getattr(service, "_orchestrators", {}).get("langgraph")
    if orchestrator is None or not hasattr(orchestrator, "reconfigure"):
        raise HTTPException(status_code=503, detail="langgraph orchestrator is not initialized")
    return orchestrator


@router.get("")
async def get_langgraph_settings(request: Request) -> dict[str, Any]:
    agent = _orchestrator(request)
    config = agent.config
    models = await provider_models(config.llm.provider, config.llm.base_url)
    return {
        "track": "langgraph",
        "provider": config.llm.provider,
        "model": config.llm.model,
        "base_url": config.llm.base_url,
        "temperature": config.llm.temperature,
        "top_p": config.llm.top_p,
        "max_tokens": config.llm.max_tokens,
        "timeout_seconds": config.llm.timeout_seconds,
        "models": models,
        "default_model": default_model(config.llm.provider, models or None),
        "applies_to": "new runs",
    }


@router.get("/models")
async def list_langgraph_models(
    provider: str = Query(default="ollama"),
    base_url: str | None = Query(default=None),
) -> dict[str, Any]:
    provider = normalize_provider(provider)
    base_url = normalize_base_url(base_url)
    models = await provider_models(provider, base_url)
    return {
        "track": "langgraph",
        "provider": provider,
        "models": models,
        "default_model": default_model(provider, models or None),
    }


@router.patch("")
async def update_langgraph_settings(body: RuntimeLLMSettings, request: Request) -> dict[str, Any]:
    agent = _orchestrator(request)
    old = agent.config
    provider = normalize_provider(body.provider)
    if provider not in _LANGGRAPH_PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported langgraph provider {provider!r}; expected one of {', '.join(_LANGGRAPH_PROVIDERS)}",
        )
    base_url = clean_base_url(provider, normalize_base_url(body.base_url))
    downloaded = await provider_models(provider, base_url) if provider == "ollama" else []
    model = resolve_model(provider, normalize_model(body.model), downloaded or None)
    if not model:
        raise HTTPException(status_code=422, detail=f"no default model for provider {provider!r}; set model explicitly")
    try:
        if provider == "ollama":
            if not downloaded:
                raise ValueError(
                    f"Ollama is not reachable at {base_url or 'http://localhost:11434'} or has no models installed"
                )
            if model not in downloaded:
                raise ValueError(f"Ollama model {model!r} is not installed; available: {downloaded}")
        elif provider == "groq":
            if not os.getenv("GROQ_API_KEY", "").strip():
                raise ValueError("GROQ_API_KEY is not configured for the API process")
        config = AgentConfig(
            llm=LLMConfig(
                provider=provider,
                model=model,
                api_key=old.llm.api_key,
                timeout_seconds=body.timeout_seconds,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                top_p=body.top_p,
                base_url=base_url,
            ),
            execution=old.execution,
            sandbox=old.sandbox,
            permissions=old.permissions,
            checkpoint=old.checkpoint,
            tracing=old.tracing,
            behaviour=old.behaviour,
            prompts=old.prompts,
            metadata=old.metadata,
        )
        await agent.reconfigure(config)
    except (ValueError, NotImplementedError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    models = await provider_models(provider, base_url)
    return {
        "track": "langgraph",
        "provider": provider,
        "model": model,
        "models": models,
        "default_model": default_model(provider, models or None),
        "applies_to": "new runs",
    }
