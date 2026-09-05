"""Live settings for the native AgentRuntime."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..settings import (
    RuntimeLLMSettings,
    clean_base_url,
    default_model,
    normalize_base_url,
    normalize_model,
    normalize_provider,
    provider_models,
    resolve_model,
)
from .dependencies import get_native_runtime

router = APIRouter(prefix="/native/settings", tags=["native-settings"])
NativeRuntimeDep = Annotated[Any, Depends(get_native_runtime)]

_NATIVE_PROVIDERS = ("groq", "ollama")


@router.get("")
async def get_native_settings(runtime: NativeRuntimeDep) -> dict[str, Any]:
    agents = list(getattr(runtime, "agents", {}).values())
    config = agents[0] if agents else None
    provider = ""
    if config:
        try:
            provider = runtime.models.get(config.model).provider
        except (KeyError, AttributeError):
            provider = ""
    provider = normalize_provider(provider)
    downloaded = await provider_models(provider, None) if provider == "ollama" else []
    models = list(runtime.models.list_model_names())
    for name in downloaded:
        if name not in models:
            models.append(name)
    return {
        "track": "native",
        "model": getattr(config, "model", "") if config else "",
        "provider": provider,
        "models": models,
        "default_model": default_model(provider, downloaded or None),
        "temperature": getattr(config, "temperature", 0.0),
        "top_p": getattr(config, "top_p", 1.0),
        "max_tokens": getattr(config, "max_output_tokens", None),
        "timeout_seconds": getattr(config, "timeout_seconds", 60),
    }


@router.get("/models")
async def list_native_models(
    provider: str = Query(default="ollama"),
    base_url: str | None = Query(default=None),
) -> dict[str, Any]:
    provider = normalize_provider(provider)
    base_url = normalize_base_url(base_url)
    models = await provider_models(provider, base_url)
    return {
        "track": "native",
        "provider": provider,
        "models": models,
        "default_model": default_model(provider, models or None),
    }


@router.patch("")
async def update_native_settings(
    body: RuntimeLLMSettings,
    runtime: NativeRuntimeDep,
) -> dict[str, Any]:
    provider = normalize_provider(body.provider)
    if provider not in _NATIVE_PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported native provider {provider!r}; expected one of {', '.join(_NATIVE_PROVIDERS)}",
        )
    base_url = clean_base_url(provider, normalize_base_url(body.base_url))
    downloaded = await provider_models(provider, base_url) if provider == "ollama" else []
    model = resolve_model(provider, normalize_model(body.model), downloaded or None)
    if not model:
        raise HTTPException(status_code=422, detail=f"no default model for provider {provider!r}; set model explicitly")
    if provider == "ollama":
        if not downloaded:
            raise HTTPException(
                status_code=422,
                detail=f"Ollama is not reachable at {base_url or 'http://localhost:11434'} or has no models installed",
            )
        if model not in downloaded:
            raise HTTPException(
                status_code=422,
                detail=f"Ollama model {model!r} is not installed; available: {downloaded}",
            )
    try:
        models = runtime.reconfigure_models(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            timeout_seconds=body.timeout_seconds,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for name in downloaded:
        if name not in models:
            models.append(name)
    return {
        "track": "native",
        "provider": provider,
        "model": model,
        "models": models,
        "default_model": default_model(provider, downloaded or None),
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_tokens,
        "timeout_seconds": body.timeout_seconds,
        "applies_to": "new runs",
    }
