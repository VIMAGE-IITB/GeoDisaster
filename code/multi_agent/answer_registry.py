"""Runtime-owned references to schema-valid terminal answers.

The registry separates the model-facing transport contract from the answer
contract used by the evaluator.  Agents emit the stable token
``ANSWER_JSON:answer_1``; the runtime resolves that token to a JSON value that
was projected only from successful worker evidence and validated against the
current question's declared answer schema.

This module never reads a gold answer, GT chain, or teacher trajectory.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, MutableMapping

from .answer_contract import validate_explicit_answer


ANSWER_REF = "ANSWER_JSON:answer_1"


@dataclass(frozen=True)
class AnswerResolution:
    ok: bool
    answer: Any | None = None
    error: str = ""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def register_answer(
    state: MutableMapping[str, Any],
    case: dict[str, Any],
    answer: Any,
    *,
    produced_by_step: str,
    produced_by_tool: str,
) -> str:
    """Validate and publish the current authoritative answer projection.

    ``answer_1`` is a stable slot, not a worker-artifact handle.  Later
    successful worker evidence may refresh the slot with a newer complete
    projection; incomplete or invalid projections never overwrite it.
    """

    validation = validate_explicit_answer(case, answer)
    if not validation.ok:
        raise ValueError("cannot register an answer that fails the declared schema")
    canonical = _canonical_json(validation.answer)
    registry = state.setdefault("answer_registry", {})
    registry[ANSWER_REF] = {
        "answer": deepcopy(validation.answer),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "produced_by_step": produced_by_step,
        "produced_by_tool": produced_by_tool,
    }
    return ANSWER_REF


def resolve_answer_ref(state: MutableMapping[str, Any], answer_ref: str) -> AnswerResolution:
    """Resolve one exact runtime reference, with no fallback or coercion."""

    ref = str(answer_ref or "").strip()
    if ref != ANSWER_REF:
        return AnswerResolution(False, error=f"unsupported answer reference: {ref!r}")
    record = (state.get("answer_registry") or {}).get(ref)
    if not isinstance(record, dict) or "answer" not in record:
        return AnswerResolution(False, error=f"answer reference is not ready: {ref}")
    return AnswerResolution(True, answer=deepcopy(record["answer"]))


def answer_registry_cards(state: MutableMapping[str, Any]) -> list[str]:
    """Compact model-visible cards; answer payloads are deliberately hidden."""

    out: list[str] = []
    for ref, record in sorted((state.get("answer_registry") or {}).items()):
        if not isinstance(record, dict) or "answer" not in record:
            continue
        out.append(
            f"  {ref} [schema-valid; sha256={str(record.get('sha256') or '')[:12]}]"
        )
    return out


def answer_sha256(state: MutableMapping[str, Any], answer_ref: str = ANSWER_REF) -> str:
    record = (state.get("answer_registry") or {}).get(answer_ref)
    return str(record.get("sha256") or "") if isinstance(record, dict) else ""
