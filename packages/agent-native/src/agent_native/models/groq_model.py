"""Groq provider adapter."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable
from typing import Any

from .base import Model, StreamEvent, StreamType, require_vision_support

GROQ_MODELS: dict[str, Model] = {
    "llama-3.3-70b": Model(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        context_size=131_072,
        max_output=8_192,
    ),
    "gpt-oss-20b": Model(
        provider="groq",
        model_id="openai/gpt-oss-20b",
        context_size=131_072,
        max_output=8_192,
    ),
    "gpt-oss-120b": Model(
        provider="groq",
        model_id="openai/gpt-oss-120b",
        context_size=131_072,
        max_output=8_192,
    ),
}


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class Groq:
    """Lazy wrapper around ``groq.AsyncGroq``."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = base_url
        self._client: Any = None

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            from groq import AsyncGroq

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncGroq(**kwargs)
        return self._client

    async def stream(
        self,
        messages: list,
        tools: list,
        model: Model,
        temperature: float = 0.0,
        **kwargs: Any,
    ):
        require_vision_support(messages, model)
        request: dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        request["top_p"] = kwargs.get("top_p", 1.0)
        if kwargs.get("max_tokens") is not None:
            request["max_tokens"] = kwargs["max_tokens"]
        # ``stream_options`` is part of OpenAI's streaming API, but the Groq
        # SDK versions in the wild do not all expose it.  Passing it to a
        # generated SDK method that lacks the keyword fails before any request
        # is sent.  Opt in only when the installed client advertises support;
        # usage chunks are still consumed when a server provides them.
        create = self._get_client().chat.completions.create
        try:
            parameters: Iterable[inspect.Parameter] = (
                inspect.signature(create).parameters.values()
            )
        except (TypeError, ValueError):
            parameters = ()
        if any(
            parameter.name == "stream_options"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            request["stream_options"] = {"include_usage": True}
        if tools:
            request["tools"] = tools
        effort = kwargs.get("reasoning_effort")
        if effort:
            request["reasoning_effort"] = effort

        response = await create(**request)
        finish_reason = "stop"
        async for chunk in response:
            choices = _value(chunk, "choices", []) or []
            for choice in choices:
                delta = _value(choice, "delta", {}) or {}
                text = _value(delta, "content", "")
                if text:
                    yield StreamEvent(StreamType.TEXT, {"text": text})
                reasoning = _value(delta, "reasoning", None)
                if reasoning is None:
                    reasoning = _value(delta, "reasoning_content", "")
                if reasoning:
                    yield StreamEvent(StreamType.REASONING, {"text": reasoning})
                for call in _value(delta, "tool_calls", []) or []:
                    function = _value(call, "function", {}) or {}
                    yield StreamEvent(
                        StreamType.TOOL_CALL,
                        {
                            "index": _value(call, "index", 0),
                            "id": _value(call, "id", None),
                            "name": _value(function, "name", None),
                            "arguments": _value(function, "arguments", "") or "",
                        },
                    )
                reason = _value(choice, "finish_reason", None)
                if reason:
                    finish_reason = reason

            usage = _value(chunk, "usage", None)
            if usage is not None:
                details = _value(usage, "prompt_tokens_details", {}) or {}
                yield StreamEvent(
                    StreamType.USAGE,
                    {
                        "input_tokens": _value(usage, "prompt_tokens", 0) or 0,
                        "output_tokens": _value(usage, "completion_tokens", 0) or 0,
                        "cached_tokens": _value(details, "cached_tokens", 0) or 0,
                    },
                )
        yield StreamEvent(StreamType.DONE, {"finish_reason": finish_reason})

    def count_tokens(self, messages: list) -> int:
        from .base import rough_token_count

        return rough_token_count(messages)
