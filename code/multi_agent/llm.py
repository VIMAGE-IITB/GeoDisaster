"""LLM client.

One ``llm_call`` function. Three backends behind ``MODEL_BACKEND``:

* ``stub``    — returns canned dict-keyed responses (used by tests).
* ``hf``      — HF Inference Providers via OpenAI-compatible router.
* ``ollama``  — local Ollama via OpenAI-compatible endpoint.

When a pydantic ``schema`` is given, we parse + validate. On validation error
we retry **once** with the error appended to the message. After that we raise
``LLMValidationError`` and the caller treats it as a CONTRACT_FAILURE.

No JSON repair logic, no normalization. The schema is the validator.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

# Lazy openai import — only used for hf/ollama backends so stub-only tests
# don't pay the cost.
_openai_client_cache: dict[str, Any] = {}


class LLMValidationError(RuntimeError):
    """Raised when the LLM cannot produce schema-conforming JSON within budget."""


class LLMTransportError(RuntimeError):
    """Raised on network / auth failures contacting the backend."""


@dataclass
class LLMResponse:
    """Container returned by ``llm_call`` when no schema is given."""

    raw: str


# ---------------------------------------------------------------------------
# JSON extraction (lightweight, NO repair)
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)


def _strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text, count=1)
        if text.endswith("```"):
            text = text[: -3].rstrip()
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` substring. Raises if none found.

    This is structural — we walk braces. We do NOT attempt to fix unbalanced
    JSON or close it for the model. If it's broken we want the validator to
    raise so the model retries.
    """
    text = _strip_markdown_fences(text)
    start = text.find("{")
    if start < 0:
        raise LLMValidationError(f"no JSON object found in response: {text!r}")
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise LLMValidationError(f"unbalanced JSON object in response: {text!r}")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _backend() -> str:
    return os.environ.get("MODEL_BACKEND", "stub").strip().lower()


def _stub_call(messages: list[dict[str, Any]]) -> str:
    """Look up a canned response by stub key.

    The stub dictionary is set per-test via :func:`set_stub_responses`. Each
    call pops the next response in queue order. If the queue is empty the stub
    raises so tests fail loudly rather than fall back to a default.
    """
    if not _STUB_QUEUE:
        raise LLMTransportError("stub queue empty — set_stub_responses() before calling llm_call")
    return _STUB_QUEUE.pop(0)


def _hf_call(messages: list[dict[str, Any]]) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise LLMTransportError("HF_TOKEN env var is required for the hf backend")
    client = _openai_client_cache.get("hf")
    if client is None:
        from openai import OpenAI  # imported lazily

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
        _openai_client_cache["hf"] = client
    model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=int(os.environ.get("HF_MAX_TOKENS", "1024")),
        )
    except Exception as exc:  # pragma: no cover — network-dependent
        raise LLMTransportError(f"HF Inference call failed: {exc}") from exc
    return resp.choices[0].message.content or ""


def _ollama_call(messages: list[dict[str, Any]]) -> str:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    client = _openai_client_cache.get("ollama")
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=os.environ.get("OLLAMA_API_KEY", "ollama"))
        _openai_client_cache["ollama"] = client
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=int(os.environ.get("OLLAMA_MAX_TOKENS", "1024")),
        )
    except Exception as exc:  # pragma: no cover
        raise LLMTransportError(f"Ollama call failed: {exc}") from exc
    return resp.choices[0].message.content or ""


_BACKEND_CALLS = {
    "stub": _stub_call,
    "hf": _hf_call,
    "ollama": _ollama_call,
}


# ---------------------------------------------------------------------------
# Stub queue (test-controlled)
# ---------------------------------------------------------------------------


_STUB_QUEUE: list[str] = []


def set_stub_responses(responses: list[str]) -> None:
    """Replace the stub queue. Each ``llm_call`` pops one entry."""
    _STUB_QUEUE.clear()
    _STUB_QUEUE.extend(responses)


def stub_queue_len() -> int:
    return len(_STUB_QUEUE)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def llm_call(
    *,
    system: str,
    user: str,
    schema: type[BaseModel] | None = None,
    max_retries: int = 1,
    extra_messages: list[dict[str, Any]] | None = None,
) -> Any:
    """Send one chat message; if ``schema`` given, parse + validate.

    Returns:
        - schema instance if ``schema`` was provided
        - ``LLMResponse(raw=...)`` otherwise

    Raises:
        ``LLMValidationError`` after exhausting retries.
        ``LLMTransportError`` on network/auth failure (no retry).
    """
    base_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    if extra_messages:
        base_messages.extend(extra_messages)
    base_messages.append({"role": "user", "content": user})

    backend_call = _BACKEND_CALLS[_backend()]

    if schema is None:
        raw = backend_call(base_messages)
        return LLMResponse(raw=raw)

    last_error: str = ""
    for attempt in range(max_retries + 1):
        msgs = list(base_messages)
        if last_error and attempt > 0:
            # Inject an error feedback message so the model knows what to fix.
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply did not match the required schema. "
                        f"Validation error:\n{last_error}\n"
                        "Reply again with ONE JSON object matching the schema. No prose."
                    ),
                }
            )
        raw = backend_call(msgs)
        try:
            json_str = _extract_first_json_object(raw)
            obj = json.loads(json_str)
            return schema.model_validate(obj)
        except (LLMValidationError, json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            continue
    raise LLMValidationError(
        f"LLM failed schema validation after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
