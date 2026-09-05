"""Ollama provider adapter and OpenAI-to-Ollama message translation."""

from __future__ import annotations

import os
from typing import Any

from .base import Model, StreamEvent, StreamType, require_vision_support


def _data_url_to_base64(value: str) -> str:
    if not value:
        return ""
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _to_ollama_messages(messages: list) -> list:
    translated: list = []
    for original in messages:
        if not isinstance(original, dict) or not isinstance(original.get("content"), list):
            translated.append(dict(original) if isinstance(original, dict) else original)
            continue
        message = {key: value for key, value in original.items() if key != "content"}
        text_parts: list[str] = []
        images: list[str] = []
        for part in original["content"]:
            if not isinstance(part, dict):
                text_parts.append(str(part))
                continue
            if part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    text_parts.append(str(text))
            elif part.get("type") == "image_url":
                image = part.get("image_url") or {}
                url = image.get("url", "") if isinstance(image, dict) else image
                encoded = _data_url_to_base64(str(url))
                if encoded:
                    images.append(encoded)
            else:
                text = part.get("text") or part.get("content")
                if text:
                    text_parts.append(str(text))
        message["content"] = "\n".join(text_parts)
        if images:
            message["images"] = images
        translated.append(message)
    return translated


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class Ollama:
    """Lazy wrapper around the official Ollama async client."""

    def __init__(self, host: str | None = None) -> None:
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from ollama import AsyncClient

            self._client = AsyncClient(host=self.host)
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
            "messages": _to_ollama_messages(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 1.0),
                **({"num_predict": kwargs["max_tokens"]} if kwargs.get("max_tokens") is not None else {}),
            },
        }
        if tools:
            request["tools"] = tools
        response = await self._get_client().chat(**request)
        finish_reason = "stop"
        index = 0
        async for chunk in response:
            message = _value(chunk, "message", {}) or {}
            text = _value(message, "content", "") or ""
            if text:
                yield StreamEvent(StreamType.TEXT, {"text": text})
            reasoning = _value(message, "thinking", "") or ""
            if reasoning:
                yield StreamEvent(StreamType.REASONING, {"text": reasoning})
            for call in _value(message, "tool_calls", []) or []:
                function = _value(call, "function", {}) or {}
                arguments = _value(function, "arguments", {})
                if not isinstance(arguments, str):
                    import json

                    arguments = json.dumps(arguments, ensure_ascii=False)
                yield StreamEvent(
                    StreamType.TOOL_CALL,
                    {
                        "index": index,
                        "id": _value(call, "id", None) or f"ollama_call_{index}",
                        "name": _value(function, "name", ""),
                        "arguments": arguments,
                    },
                )
                index += 1
            input_tokens = _value(chunk, "prompt_eval_count", None)
            output_tokens = _value(chunk, "eval_count", None)
            if input_tokens is not None or output_tokens is not None:
                yield StreamEvent(
                    StreamType.USAGE,
                    {
                        "input_tokens": input_tokens or 0,
                        "output_tokens": output_tokens or 0,
                    },
                )
            if _value(chunk, "done", False):
                finish_reason = _value(chunk, "done_reason", "stop") or "stop"
        yield StreamEvent(StreamType.DONE, {"finish_reason": finish_reason})

    def count_tokens(self, messages: list) -> int:
        from .base import rough_token_count

        return rough_token_count(_to_ollama_messages(messages))
