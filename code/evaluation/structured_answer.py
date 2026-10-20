"""Versioned, structure-aware answer scoring for GeoDisaster.

The benchmark's answers are JSON values.  Correctness therefore requires the
same fields, types, null states, strings, and array order as the reference;
numeric values alone are never sufficient. Numeric leaves use the published
absolute tolerance below unless a caller supplies an override; relative
tolerance is disabled so allowed error does not grow with value magnitude.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any


SCORING_POLICY_VERSION = "geodisaster_structured_answer_v2"
DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 0.0


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _is_strict_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_strict_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_strict_json_value(item)
            for key, item in value.items()
        )
    return False


def _json_value(value: Any) -> tuple[bool, Any]:
    if not isinstance(value, str):
        return _is_strict_json_value(value), value
    text = value.strip()
    if not text:
        return False, None
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text, parse_constant=_reject_nonstandard_constant)
        return _is_strict_json_value(parsed), parsed
    except (json.JSONDecodeError, ValueError):
        return False, text


def _equal(expected: Any, predicted: Any, *, abs_tol: float, rel_tol: float) -> bool:
    if expected is None or isinstance(expected, bool):
        return type(predicted) is type(expected) and predicted == expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(predicted, (int, float)) or isinstance(predicted, bool):
            return False
        a, b = float(expected), float(predicted)
        return math.isfinite(a) and math.isfinite(b) and math.isclose(
            a, b, rel_tol=rel_tol, abs_tol=abs_tol
        )
    if isinstance(expected, str):
        return isinstance(predicted, str) and predicted == expected
    if isinstance(expected, list):
        return (
            isinstance(predicted, list)
            and len(expected) == len(predicted)
            and all(
                _equal(a, b, abs_tol=abs_tol, rel_tol=rel_tol)
                for a, b in zip(expected, predicted)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(predicted, dict)
            and set(expected) == set(predicted)
            and all(
                _equal(expected[key], predicted[key], abs_tol=abs_tol, rel_tol=rel_tol)
                for key in expected
            )
        )
    return type(predicted) is type(expected) and predicted == expected


def _token_f1(gold: str, pred: str) -> float:
    tokenize = lambda text: re.findall(r"[a-z0-9]+", str(text or "").lower())
    gold_tokens, pred_tokens = tokenize(gold), tokenize(pred)
    if not gold_tokens or not pred_tokens:
        return 0.0
    overlap = sum((Counter(gold_tokens) & Counter(pred_tokens)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(
    gold: Any,
    pred: Any,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, Any]:
    gold_ok, gold_value = _json_value(gold)
    pred_ok, pred_value = _json_value(pred)
    gold_text = json.dumps(gold_value, sort_keys=True, ensure_ascii=False) if gold_ok else str(gold)
    pred_text = json.dumps(pred_value, sort_keys=True, ensure_ascii=False) if pred_ok else str(pred)
    structured_match = bool(gold_ok and pred_ok and _equal(
        gold_value, pred_value, abs_tol=abs_tol, rel_tol=rel_tol
    ))
    exact = bool(gold_ok and pred_ok and gold_value == pred_value)
    token_f1 = _token_f1(gold_text, pred_text)
    seq_ratio = SequenceMatcher(None, gold_text.lower(), pred_text.lower()).ratio() if pred_text else 0.0
    answer_similarity = max(token_f1, seq_ratio)
    return {
        "scoring_policy": SCORING_POLICY_VERSION,
        # Backward-compatible field name, now with actual accuracy semantics.
        "answer_acc": 1.0 if structured_match else 0.0,
        "answer_similarity": answer_similarity,
        "exact_match": exact,
        "structured_match": structured_match,
        "numeric_match": structured_match,
        # Retained for result-file compatibility; it is deliberately never a
        # correctness route under the active structured policy.
        "numeric_subset_match": False,
        "correct": structured_match,
        "valid_json": pred_ok,
        "token_f1": token_f1,
        "sequence_ratio": seq_ratio,
        "absolute_tolerance": abs_tol,
        "relative_tolerance": rel_tol,
    }


def structured_diff(
    expected: Any,
    predicted: Any,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return deterministic JSON-pointer differences under the score policy."""

    expected_ok, expected_value = _json_value(expected)
    predicted_ok, predicted_value = _json_value(predicted)
    if not expected_ok:
        return [{"pointer": "", "kind": "invalid_expected_json"}]
    if not predicted_ok:
        return [
            {
                "pointer": "",
                "kind": "invalid_predicted_json",
                "predicted": str(predicted),
            }
        ]

    rows: list[dict[str, Any]] = []

    def add(pointer: str, kind: str, exp: Any = None, pred: Any = None) -> None:
        if len(rows) >= limit:
            return
        row = {"pointer": pointer or "", "kind": kind}
        if kind not in {"missing_key", "extra_key"} or exp is not None:
            row["expected"] = exp
        if kind not in {"missing_key"}:
            row["predicted"] = pred
        rows.append(row)

    def token(value: Any) -> str:
        return str(value).replace("~", "~0").replace("/", "~1")

    def walk(exp: Any, pred: Any, pointer: str) -> None:
        if len(rows) >= limit:
            return
        if exp is None or isinstance(exp, bool):
            if type(pred) is not type(exp) or pred != exp:
                add(pointer, "value_mismatch", exp, pred)
            return
        if isinstance(exp, (int, float)) and not isinstance(exp, bool):
            if not isinstance(pred, (int, float)) or isinstance(pred, bool):
                add(pointer, "type_mismatch", exp, pred)
            elif not (
                math.isfinite(float(exp))
                and math.isfinite(float(pred))
                and math.isclose(float(exp), float(pred), rel_tol=rel_tol, abs_tol=abs_tol)
            ):
                add(pointer, "numeric_mismatch", exp, pred)
            return
        if isinstance(exp, str):
            if not isinstance(pred, str):
                add(pointer, "type_mismatch", exp, pred)
            elif pred != exp:
                add(pointer, "value_mismatch", exp, pred)
            return
        if isinstance(exp, list):
            if not isinstance(pred, list):
                add(pointer, "type_mismatch", exp, pred)
                return
            if len(exp) != len(pred):
                add(pointer, "array_length_mismatch", len(exp), len(pred))
            for index, (left, right) in enumerate(zip(exp, pred)):
                walk(left, right, f"{pointer}/{index}")
            return
        if isinstance(exp, dict):
            if not isinstance(pred, dict):
                add(pointer, "type_mismatch", exp, pred)
                return
            for key in sorted(set(exp) - set(pred)):
                add(f"{pointer}/{token(key)}", "missing_key", exp[key], None)
            for key in sorted(set(pred) - set(exp)):
                add(f"{pointer}/{token(key)}", "extra_key", None, pred[key])
            for key in sorted(set(exp) & set(pred)):
                walk(exp[key], pred[key], f"{pointer}/{token(key)}")
            return
        if type(pred) is not type(exp) or pred != exp:
            add(pointer, "value_mismatch", exp, pred)

    walk(expected_value, predicted_value, "")
    if len(rows) >= limit:
        rows.append({"pointer": "", "kind": "diff_truncated", "limit": limit})
    return rows

