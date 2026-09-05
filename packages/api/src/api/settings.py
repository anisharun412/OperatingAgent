"""Shared API models and helpers for runtime settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RuntimeLLMSettings(BaseModel):
    provider: str = Field(default="ollama", min_length=1)
    model: str = ""
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: int = Field(default=60, gt=0)


_DEFAULT_MODELS: dict[str, str] = {
    "ollama": "qwen3.5:0.8b",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}

_KNOWN_MODELS: dict[str, list[str]] = {
    "groq": ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
}

_OLLAMA_HOSTS = ("http://localhost:11434", "http://127.0.0.1:11434")


def normalize_provider(value: object) -> str:
    return str(value or "ollama").strip().lower() or "ollama"


def normalize_model(value: object) -> str:
    return str(value or "").strip()


def normalize_base_url(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def normalize_ollama_host(value: object) -> str:
    """Return the Ollama server root (never the ``/api`` path)."""
    text = str(value or "").strip().rstrip("/") or "http://localhost:11434"
    if text.endswith("/api"):
        text = text[: -len("/api")].rstrip("/") or "http://localhost:11434"
    return text


def clean_base_url(provider: str, base_url: str | None) -> str | None:
    """Drop an Ollama URL that was carried over to a cloud provider.

    This is the main cause of Groq's confusing ``404 page not found``: the
    Groq client was pointed at the local Ollama server.
    """
    if base_url and provider != "ollama":
        for host in _OLLAMA_HOSTS:
            if base_url.startswith(host):
                return None
    return base_url


async def ollama_models(base_url: str | None) -> list[str]:
    import httpx

    host = normalize_ollama_host(base_url)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{host}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return []
    models: list[str] = []
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        name = item.get("name") if isinstance(item, dict) else None
        if name and str(name) not in models:
            models.append(str(name))
    return models


async def provider_models(provider: str, base_url: str | None) -> list[str]:
    provider = normalize_provider(provider)
    if provider == "ollama":
        return await ollama_models(base_url)
    return list(_KNOWN_MODELS.get(provider, []))


def default_model(provider: str, downloaded: list[str] | None = None) -> str:
    provider = normalize_provider(provider)
    if provider == "ollama" and downloaded:
        return downloaded[0]
    return _DEFAULT_MODELS.get(provider, "")


def resolve_model(provider: str, model: str, downloaded: list[str] | None = None) -> str:
    """Return the explicit model, or the provider default when empty."""
    model = normalize_model(model)
    if model:
        return model
    if provider == "ollama" and downloaded:
        return downloaded[0]
    return _DEFAULT_MODELS.get(provider, "")
