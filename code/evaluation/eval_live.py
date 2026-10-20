"""Shared live-evaluation metrics and compatibility helpers.

The canonical GeoDisaster benchmark entry point is :mod:`eval_multi_agent`.
It imports the sequence metrics and optional model/controller helpers retained
here. Agent-facing GeoDisaster tool execution is owned by
``multi_agent.tool_gateway`` and always invokes real worker computation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.structured_answer import score_answer as _score_answer

try:  # only required for the OpenAI-compatible model backend
    from openai import OpenAI
except Exception:  # noqa: BLE001
    OpenAI = None  # type: ignore[assignment]

try:  # only needed for the optional live tool-controller eval path
    from tool_server.tool_workers.tool_manager.base_manager import ToolManager
except Exception:  # noqa: BLE001
    ToolManager = None  # type: ignore

_LIVE_CONTEXT: dict[str, Any] = {"state": None}
_SCRIPTED_CASES_CACHE: dict[str, dict[int, dict[str, Any]]] = {}


def _maybe_init_accelerate() -> None:
    try:
        import importlib

        accelerate_mod = importlib.import_module("accelerate")
        accelerator_cls = getattr(accelerate_mod, "Accelerator", None)
        if accelerator_cls is not None:
            accelerator_cls()
    except Exception:
        pass


def _extract_question_text(value: str) -> str:
    text = str(value or "").strip()
    if "\n\nQuestion:" in text:
        return text.split("\n\nQuestion:", 1)[1].strip()
    marker = "Question:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "")
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(raw[start:], start=start):
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
                try:
                    return json.loads(raw[start : idx + 1])
                except Exception:
                    return None
    return None


def _extract_gold_label(item: dict[str, Any]) -> str:
    for turn in reversed(item.get("conversation", []) or []):
        if str(turn.get("from", "")).lower() not in {"gpt", "assistant"}:
            continue
        payload = _extract_first_json_object(str(turn.get("value", "")))
        if not isinstance(payload, dict):
            continue
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            continue
        action = actions[0]
        if not isinstance(action, dict) or action.get("name") != "Terminate":
            continue
        arguments = action.get("arguments") or {}
        if isinstance(arguments, dict):
            answer = str(arguments.get("answer", "") or arguments.get("ans", "")).strip()
            if answer:
                return answer
    return ""


_TOOL_ARG_SPEC: dict[str, dict[str, set[str]]] = {
    "GetAreaBoundary": {"required": {"area"}, "optional": {"buffer_m"}},
    "AddPoisLayer": {"required": {"gpkg", "query", "layer_name"}, "optional": set()},
    "ComputeDistance": {"required": {"gpkg", "src_layer", "tar_layer"}, "optional": {"top"}},
    "DisplayOnMap": {"required": {"gpkg", "layers"}, "optional": set()},
    "GetBboxFromGeotiff": {"required": {"geotiff"}, "optional": set()},
    "DisplayOnGeotiff": {"required": {"gpkg", "layers", "geotiff"}, "optional": set()},
    "Calculator": {"required": {"expression"}, "optional": set()},
    "Solver": {"required": {"command"}, "optional": set()},
    "Plot": {"required": {"command"}, "optional": set()},
    "GoogleSearch": {"required": {"query"}, "optional": {"k"}},
    "ObjectDetection": {"required": {"image"}, "optional": set()},
    "TextToBbox": {"required": {"image", "text"}, "optional": {"top1"}},
    "CountGivenObject": {"required": {"image", "text"}, "optional": set()},
    "ImageDescription": {"required": {"image"}, "optional": set()},
    "RegionAttributeDescription": {"required": {"image", "attribute"}, "optional": {"bbox"}},
    "ChangeDetection": {"required": {"text", "pre_image", "post_image"}, "optional": set()},
    "SegmentObjectPixels": {"required": {"image", "text"}, "optional": {"flag"}},
    "OCR": {"required": {"image"}, "optional": set()},
    "DrawBox": {"required": {"image", "bbox", "annotation"}, "optional": set()},
    "AddText": {"required": {"image", "text", "position"}, "optional": {"color"}},
    "AddIndexLayer": {"required": {"gpkg", "index_type", "layer_name"}, "optional": {"year", "month"}},
    "ComputeIndexChange": {"required": {"gpkg", "index_type", "layer1_name", "layer2_name", "diff_layer_name"}, "optional": set()},
    "ShowIndexLayer": {"required": {"gpkg", "index_type", "layer_name"}, "optional": set()},
}

# The legacy EO tools above remain available for the legacy evaluator. The
# finalized GeoDisaster tools derive their validation contract from the same
# schema used by workers and prompts.
try:
    from tool_server.tool_workers.online_workers.final_dataset_tool_specs import (
        agent_tool_spec,
        validate_agent_args as _validate_geodisaster_agent_args,
    )

    from multi_agent.final_dataset_profile import FINAL_DATASET_TOOLS

    for _tool_name in FINAL_DATASET_TOOLS:
        _spec = agent_tool_spec(_tool_name)
        _required = set(_spec["required"])
        _TOOL_ARG_SPEC[_tool_name] = {
            "required": _required,
            "optional": set(_spec["properties"]) - _required,
        }
except Exception:
    _validate_geodisaster_agent_args = None
    pass


def _summary_is_concrete(summary: str) -> bool:
    text = str(summary or "").strip().lower()
    if not text:
        return False
    if re.search(r"\d", text):
        return True
    concrete_tokens = (
        "gpkg_",
        "image_",
        "tif_",
        "layer ",
        "distance",
        "count",
        "bbox",
        "pixel",
        "ndbi",
        "ndvi",
        "route",
        "saved",
        "created",
        "annotated",
        "rendered",
        "failed",
        "cannot",
        "missing",
        "closest",
        "visible",
    )
    return any(token in text for token in concrete_tokens)


def _validate_arg_names(tool_name: str, args: dict[str, Any]) -> bool:
    spec = _TOOL_ARG_SPEC.get(tool_name)
    if spec is None:
        return True
    keys = set((args or {}).keys())
    required = spec["required"]
    allowed = required | spec["optional"]
    return required.issubset(keys) and keys.issubset(allowed)


def _geodisaster_arg_contract_ok(tool_name: str, args: dict[str, Any]) -> bool:
    if _validate_geodisaster_agent_args is None or tool_name not in globals().get("FINAL_DATASET_TOOLS", ()):
        return _validate_arg_names(tool_name, args)
    return not bool(_validate_geodisaster_agent_args(tool_name, dict(args or {})))


def _categorize_tool_issue(text: str, failure_category: str = "") -> str:
    if failure_category in {
        "agent_invalid_args",
        "agent_semantic_conflict",
        "invalid_enum",
        "missing_required_argument",
    }:
        return "arg_value_error"
    if failure_category in {"unknown_tool", "forbidden_tool"}:
        return "forbidden_tool"
    if failure_category in {"session_error", "wrong_handle_type", "missing_dependency"}:
        return "missing_dependency"
    if failure_category in {"transport_error", "configuration_error", "worker_error"}:
        return failure_category
    lower = str(text or "").lower()
    if not lower:
        return "unknown"
    if "timed out" in lower or "read timeout" in lower:
        return "worker_timeout"
    if "incorrect padding" in lower:
        return "downstream_render_error"
    if "invalid flag type" in lower or "invalid bbox format" in lower:
        return "arg_value_error"
    if "not found" in lower and "geopackage" in lower:
        return "missing_gpkg"
    if "missing layers" in lower or "could not be opened" in lower:
        return "missing_layer"
    if "no crs" in lower or "cannot load raster" in lower:
        return "data_or_tool_input_mismatch"
    if "is not allowed to call" in lower:
        return "forbidden_tool"
    if "requires" in lower or "missing" in lower:
        return "arg_value_error"
    return "tool_error"


def _normalize_query(query: Any) -> str:
    if not isinstance(query, dict):
        return str(query or "").strip().lower()
    parts = []
    for key, value in sorted((str(k).strip().lower(), str(v).strip().lower()) for k, v in query.items()):
        parts.append(f"{key}={value}")
    return "|".join(parts)


def _normalize_bbox(value: Any) -> str:
    nums = [str(int(float(n))) for n in re.findall(r"[-+]?\d*\.?\d+", str(value or ""))]
    return ",".join(nums[:4])


def _semantic_tool_signature(tool_name: str, args: dict[str, Any]) -> str:
    args = dict(args or {})
    if tool_name == "GetAreaBoundary":
        area = str(args.get("area", "")).strip().lower()
        raw_buffer = args.get("buffer_m")
        try:
            buffer_value = int(raw_buffer) if raw_buffer is not None else 0
        except Exception:
            buffer_value = str(raw_buffer).strip().lower()
        return f"{tool_name}|area={area}|buffer_m={buffer_value}"
    if tool_name == "AddPoisLayer":
        return (
            f"{tool_name}|gpkg={str(args.get('gpkg', '')).strip().lower()}"
            f"|query={_normalize_query(args.get('query'))}"
        )
    if tool_name == "AddIndexLayer":
        return (
            f"{tool_name}|gpkg={str(args.get('gpkg', '')).strip().lower()}"
            f"|index_type={str(args.get('index_type', '')).strip().lower()}"
            f"|year={str(args.get('year', '')).strip().lower()}"
            f"|month={str(args.get('month', '')).strip().lower()}"
        )
    if tool_name == "ComputeIndexChange":
        return (
            f"{tool_name}|gpkg={str(args.get('gpkg', '')).strip().lower()}"
            f"|index_type={str(args.get('index_type', '')).strip().lower()}"
        )
    if tool_name == "ComputeDistance":
        return (
            f"{tool_name}|gpkg={str(args.get('gpkg', '')).strip().lower()}"
            f"|src={str(args.get('src_layer', '')).strip().lower()}"
            f"|tar={str(args.get('tar_layer', '')).strip().lower()}"
        )
    if tool_name == "GetBboxFromGeotiff":
        return f"{tool_name}|geotiff={str(args.get('geotiff', '')).strip().lower()}"
    if tool_name == "TextToBbox":
        return (
            f"{tool_name}|image={str(args.get('image', '')).strip().lower()}"
            f"|text={str(args.get('text', '')).strip().lower()}"
        )
    if tool_name == "SegmentObjectPixels":
        return (
            f"{tool_name}|image={str(args.get('image', '')).strip().lower()}"
            f"|text={str(args.get('text', '')).strip().lower()}"
            f"|flag={str(args.get('flag', '')).strip().lower()}"
        )
    if tool_name == "CountGivenObject":
        return (
            f"{tool_name}|image={str(args.get('image', '')).strip().lower()}"
            f"|text={str(args.get('text', '')).strip().lower()}"
        )
    if tool_name == "RegionAttributeDescription":
        return (
            f"{tool_name}|image={str(args.get('image', '')).strip().lower()}"
            f"|attribute={str(args.get('attribute', '')).strip().lower()}"
            f"|bbox={_normalize_bbox(args.get('bbox'))}"
        )
    if tool_name == "ChangeDetection":
        return (
            f"{tool_name}|text={str(args.get('text', '')).strip().lower()}"
            f"|pre={str(args.get('pre_image', '')).strip().lower()}"
            f"|post={str(args.get('post_image', '')).strip().lower()}"
        )
    return f"{tool_name}|{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"


def _default_scripted_db_path() -> Path | None:
    env_path = str(os.environ.get("MAS_SCRIPTED_TOOL_DATA_PATH", "")).strip()
    if env_path:
        path = Path(env_path).resolve()
        return path if path.exists() else None
    candidates = [
        Path("data/scripted_test_full.json").resolve(),
        Path("data/scripted_train_full.json").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_scripted_case_map() -> dict[int, dict[str, Any]]:
    db_path = _default_scripted_db_path()
    if db_path is None:
        return {}
    cache_key = str(db_path)
    cached = _SCRIPTED_CASES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    payload = json.loads(db_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        _SCRIPTED_CASES_CACHE[cache_key] = {}
        return {}
    case_map = {int(case.get("case_id", idx)): case for idx, case in enumerate(cases) if isinstance(case, dict)}
    _SCRIPTED_CASES_CACHE[cache_key] = case_map
    return case_map


def _expected_case_for_record(record: dict[str, Any]) -> dict[str, Any] | None:
    # GeoDisaster evaluation rows carry the public canonical chain directly.
    # Prefer it over the historical OpenEarth scripted-case database so online
    # sequence metrics are meaningful for the 2,921-record benchmark.
    chain = record.get("gt_chain")
    if isinstance(chain, list):
        return {
            "turns": [
                {"tool": call.get("tool"), "args": dict(call.get("args") or {})}
                for call in chain
                if isinstance(call, dict) and call.get("tool") != "TaskComplete"
            ]
        }
    case_map = _load_scripted_case_map()
    if not case_map:
        return None
    return case_map.get(int(record.get("dataset_index", -1)))


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _arg_value_ok(
    tool_name: str,
    args: dict[str, Any],
    ok: bool,
    text: str,
    failure_category: str = "",
) -> bool:
    if not _geodisaster_arg_contract_ok(tool_name, args):
        return False
    if failure_category in {
        "agent_invalid_args",
        "agent_semantic_conflict",
        "invalid_enum",
        "missing_required_argument",
        "wrong_handle_type",
        "missing_dependency",
        "forbidden_tool",
    }:
        return False
    if tool_name == "SegmentObjectPixels" and "flag" in (args or {}) and not isinstance(args.get("flag"), bool):
        return False
    if tool_name in {"RegionAttributeDescription", "DrawBox"} and "bbox" in (args or {}):
        bbox = args.get("bbox")
        if bbox is not None and not re.match(r"^\(?\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\)?$", str(bbox)):
            return False
    if ok:
        return True
    return _categorize_tool_issue(text, failure_category) not in {
        "arg_value_error",
        "missing_gpkg",
        "missing_layer",
        "forbidden_tool",
    }


def _turn_agent_metrics_template() -> dict[str, Any]:
    return {
        "specialist_step_events": 0,
        "specialist_valid_turns": 0,
        "specialist_invalid_turns": 0,
        "tool_calls": 0,
        "tool_successes": 0,
        "arg_name_valid": 0,
        "arg_value_valid": 0,
        "summary_events": 0,
        "summary_concrete": 0,
        "task_complete": 0,
        "contract_failure": 0,
        "loop_failures": 0,
        "auto_downgrades": 0,
        "wrong_tool_calls": 0,
        "orc_invalid_events": 0,
        "orc_total_events": 0,
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _finalize_agent_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    out["TSR"] = _safe_div(raw["tool_successes"], raw["tool_calls"])
    out["ArgN"] = _safe_div(raw["arg_name_valid"], raw["tool_calls"])
    out["ArgV"] = _safe_div(raw["arg_value_valid"], raw["tool_calls"])
    out["Summ"] = _safe_div(raw["summary_concrete"], raw["summary_events"])
    out["Inst"] = _safe_div(raw["specialist_valid_turns"], raw["specialist_step_events"])
    out["loop_failure_rate"] = _safe_div(raw["loop_failures"], raw["summary_events"])
    out["contract_failure_rate"] = _safe_div(raw["contract_failure"], raw["summary_events"])
    out["wrong_tool_rate"] = _safe_div(raw["wrong_tool_calls"], raw["tool_calls"])
    out["orc_valid_rate"] = _safe_div(raw["orc_total_events"] - raw["orc_invalid_events"], raw["orc_total_events"])
    return out


def _first_divergence_index(expected: list[str], actual: list[str]) -> int | None:
    limit = min(len(expected), len(actual))
    for idx in range(limit):
        if expected[idx] != actual[idx]:
            return idx + 1
    if len(expected) != len(actual):
        return limit + 1
    return None


def _coverage_completion_index(expected: list[str], actual: list[str]) -> int | None:
    if not expected:
        return None
    exp_idx = 0
    for act_idx, signature in enumerate(actual):
        if exp_idx < len(expected) and signature == expected[exp_idx]:
            exp_idx += 1
            if exp_idx == len(expected):
                return act_idx
    return None


def _sequence_metrics_for_record(
    record: dict[str, Any],
    *,
    actual_tool_events: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    orc_delegate_events: list[dict[str, Any]],
    specialist_complete_events: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_case = _expected_case_for_record(record)
    expected_turns = list((expected_case or {}).get("turns") or [])
    expected_signatures = [_semantic_tool_signature(str(turn.get("tool", "")), dict(turn.get("args") or {})) for turn in expected_turns]
    actual_signatures = [str(row.get("semantic_signature", "")) for row in tool_rows]

    expected_counter = Counter(expected_signatures)
    actual_counter = Counter(actual_signatures)
    overcall_count = 0
    for signature, actual_count in actual_counter.items():
        overcall_count += max(actual_count - expected_counter.get(signature, 0), 0)
    redundant_call_count = sum(max(count - 1, 0) for count in actual_counter.values())

    first_divergence = _first_divergence_index(expected_signatures, actual_signatures) if expected_signatures else None
    completion_index = _coverage_completion_index(expected_signatures, actual_signatures) if expected_signatures else None
    stop_after_sufficient_evidence = None
    if completion_index is not None:
        stop_after_sufficient_evidence = completion_index == len(actual_signatures) - 1

    actual_vra_tools = [row for row in tool_rows if row.get("agent") == "VRA"]
    expected_vra_tools = [turn for turn in expected_turns if str(turn.get("tool", "")) in {
        "TextToBbox",
        "SegmentObjectPixels",
        "CountGivenObject",
        "RegionAttributeDescription",
        "ChangeDetection",
        "GetBboxFromGeotiff",
        "OCR",
        "ImageDescription",
        "ObjectDetection",
        "DrawBox",
    }]
    vra_wrong_first_tool = 0
    if actual_vra_tools and expected_vra_tools:
        vra_wrong_first_tool = int(actual_vra_tools[0]["tool"] != str(expected_vra_tools[0].get("tool", "")))

    ga_bbox_detours = sum(
        1
        for row in tool_rows
        if row.get("agent") == "GA" and row.get("tool") in {"GetBboxFromGeotiff", "TextToBbox"}
    )
    ga_repeated_boundary_poi_overcalls = 0
    ga_semantic_counts = Counter(
        row["semantic_signature"]
        for row in tool_rows
        if row.get("agent") == "GA" and row.get("tool") in {"GetAreaBoundary", "AddPoisLayer"}
    )
    ga_repeated_boundary_poi_overcalls = sum(max(count - 1, 0) for count in ga_semantic_counts.values())
    vra_text_box_counts = Counter(
        row["semantic_signature"]
        for row in tool_rows
        if row.get("agent") == "VRA" and row.get("tool") == "TextToBbox"
    )
    vra_repeated_text_boxing = sum(max(count - 1, 0) for count in vra_text_box_counts.values())

    delegate_step_ids = [str(event.get("step_id", "")) for event in orc_delegate_events if event.get("step_id")]
    plan_steps = list((record.get("plan") or {}).get("steps") or [])
    ordered_plan_ids = [str(step.get("id", "")) for step in plan_steps if isinstance(step, dict) and step.get("id")]
    seen_step_ids: set[str] = set()
    orc_wrong_step_delegation = 0
    expected_pos = 0
    for step_id in delegate_step_ids:
        while expected_pos < len(ordered_plan_ids) and ordered_plan_ids[expected_pos] in seen_step_ids:
            expected_pos += 1
        expected_step_id = ordered_plan_ids[expected_pos] if expected_pos < len(ordered_plan_ids) else ""
        if expected_step_id and step_id != expected_step_id:
            orc_wrong_step_delegation += 1
        seen_step_ids.add(step_id)

    tool_failures = sum(1 for row in tool_rows if not row.get("ok"))
    arg_grounding_failures = sum(
        1
        for row in tool_rows
        if not row.get("arg_name_valid") or not row.get("arg_value_valid")
    )
    constraint_violations = sum(
        1
        for row in tool_rows
        if row.get("issue_type") in {"forbidden_tool", "arg_value_error"}
    )
    state_update_failures = sum(
        1
        for row in tool_rows
        if row.get("issue_type") in {"missing_gpkg", "missing_layer"}
    )
    inter_agent_inconsistency = 0
    signature_agents: dict[str, set[str]] = defaultdict(set)
    for row in tool_rows:
        signature_agents[str(row.get("semantic_signature", ""))].add(str(row.get("agent", "")))
    inter_agent_inconsistency = sum(1 for agents in signature_agents.values() if len({a for a in agents if a}) > 1)

    premature_termination = 0
    if record.get("status") != "DONE" and expected_signatures:
        if completion_index is None:
            premature_termination = 1
    elif specialist_complete_events and expected_signatures:
        if completion_index is None:
            premature_termination = 1

    final_answer = str(record.get("final_answer", "") or "")
    gold_label = str(record.get("label", "") or "")
    strict_metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    strict_correct = bool(strict_metrics.get("correct") or strict_metrics.get("structured_match"))
    pa_weak_numeric_synthesis = 0
    if (
        not strict_correct
        and any(event.get("agent") == "PA" for event in specialist_complete_events)
        and _extract_numbers(gold_label)
    ):
        gold_nums = _extract_numbers(gold_label)
        pred_nums = _extract_numbers(final_answer)
        if not pred_nums or not all(
            any(abs(g - p) <= 1e-6 for p in pred_nums) for g in gold_nums
        ):
            pa_weak_numeric_synthesis = 1

    sequence_metrics = {
        "expected_tool_calls": len(expected_signatures),
        "actual_tool_calls": len(actual_signatures),
        "first_tool_accuracy": (
            actual_signatures[0] == expected_signatures[0] if expected_signatures and actual_signatures else None
        ),
        "first_divergence_index": first_divergence,
        "overcall_count": overcall_count,
        "redundant_call_count": redundant_call_count,
        "stop_after_sufficient_evidence": stop_after_sufficient_evidence,
    }

    failure_buckets = {
        "planning_failure": int(bool(record.get("abort_reason")) and "plan" in str(record.get("abort_reason", "")).lower())
        or int(any(event.get("phase") == "plan" and (event.get("payload") or {}).get("error") for event in record.get("trace") or []))
        or int(not ordered_plan_ids),
        "tool_selection_failure": int(any(not row.get("ok") and row.get("issue_type") not in {"arg_value_error", "missing_gpkg", "missing_layer"} for row in tool_rows))
        or int(first_divergence == 1 if first_divergence is not None else 0),
        "argument_grounding_failure": int(arg_grounding_failures > 0),
        "state_update_failure": int(state_update_failures > 0),
        "constraint_violation_failure": int(constraint_violations > 0),
        "premature_termination": int(premature_termination > 0),
        "inter_agent_inconsistency": int(inter_agent_inconsistency > 0),
        "redundant_or_degenerate_loops": int(redundant_call_count > 0 or any("hit max inner turns" in str(e.get("summary", "")).lower() for e in specialist_complete_events)),
        "tool_failure": int(tool_failures > 0),
        "orc_plan_invalidity": int(any((event.get("payload") or {}).get("error") for event in record.get("trace") or [] if event.get("role") == "ORC")),
        "orc_wrong_step_delegation": int(orc_wrong_step_delegation > 0),
        "ga_bbox_detours": ga_bbox_detours,
        "ga_repeated_boundary_poi_overcalls": ga_repeated_boundary_poi_overcalls,
        "vra_wrong_first_tool": vra_wrong_first_tool,
        "vra_repeated_text_boxing": vra_repeated_text_boxing,
        "pa_weak_numeric_synthesis": pa_weak_numeric_synthesis,
    }
    return sequence_metrics, failure_buckets


def _aggregate_sequence_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_tool_calls = [float((row.get("sequence_metrics") or {}).get("expected_tool_calls", 0)) for row in results]
    actual_tool_calls = [float((row.get("sequence_metrics") or {}).get("actual_tool_calls", 0)) for row in results]
    first_tool_values = [
        1.0 if metrics.get("first_tool_accuracy") else 0.0
        for row in results
        if isinstance((metrics := row.get("sequence_metrics") or {}), dict) and metrics.get("first_tool_accuracy") is not None
    ]
    first_divergence_values = [
        float(metrics["first_divergence_index"])
        for row in results
        if isinstance((metrics := row.get("sequence_metrics") or {}), dict) and metrics.get("first_divergence_index") is not None
    ]
    overcall_values = [float((row.get("sequence_metrics") or {}).get("overcall_count", 0)) for row in results]
    redundant_values = [float((row.get("sequence_metrics") or {}).get("redundant_call_count", 0)) for row in results]
    stop_after_values = [
        1.0 if metrics.get("stop_after_sufficient_evidence") else 0.0
        for row in results
        if isinstance((metrics := row.get("sequence_metrics") or {}), dict) and metrics.get("stop_after_sufficient_evidence") is not None
    ]
    return {
        "expected_tool_calls_avg": _safe_mean(expected_tool_calls),
        "actual_tool_calls_avg": _safe_mean(actual_tool_calls),
        "first_tool_accuracy": _safe_mean(first_tool_values),
        "first_tool_accuracy_n": len(first_tool_values),
        "first_divergence_index_avg": _safe_mean(first_divergence_values),
        "first_divergence_index_n": len(first_divergence_values),
        "overcall_count_avg": _safe_mean(overcall_values),
        "redundant_call_count_avg": _safe_mean(redundant_values),
        "stop_after_sufficient_evidence_rate": _safe_mean(stop_after_values),
        "stop_after_sufficient_evidence_n": len(stop_after_values),
    }


def _aggregate_failure_buckets(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals: defaultdict[str, float] = defaultdict(float)
    for row in results:
        for key, value in (row.get("failure_buckets") or {}).items():
            if isinstance(value, (int, float, bool)):
                totals[key] += float(value)
    out: dict[str, Any] = {}
    n = max(len(results), 1)
    for key, total in sorted(totals.items()):
        out[key] = int(total) if abs(total - round(total)) < 1e-9 else float(total)
        out[f"{key}_rate"] = float(total / n)
    return out


def _analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    trace = record.get("trace") or []
    steps = record.get("steps") or []
    step_to_agent = {step["id"]: step["agent"] for step in steps if isinstance(step, dict) and step.get("id")}
    step_to_subtask = {step["id"]: step.get("subtask", "") for step in steps if isinstance(step, dict) and step.get("id")}
    agent_metrics: dict[str, dict[str, Any]] = defaultdict(_turn_agent_metrics_template)
    conversation_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    failure_tags: list[dict[str, Any]] = []
    actual_tool_events: list[dict[str, Any]] = []
    orc_delegate_events: list[dict[str, Any]] = []
    specialist_complete_events: list[dict[str, Any]] = []

    for idx, event in enumerate(trace):
        role = event.get("role")
        phase = event.get("phase")
        step_id = event.get("step_id")
        payload = event.get("payload") or {}
        agent = role if role in {"GA", "PA", "VRA", "ORC"} else step_to_agent.get(step_id, "UNKNOWN")

        if role == "ORC":
            agent_metrics["ORC"]["orc_total_events"] += 1
            if phase == "delegate":
                orc_delegate_events.append(
                    {
                        "step_id": step_id,
                        "payload": payload,
                        "trace_index": idx,
                    }
                )
            if isinstance(payload, dict) and payload.get("error"):
                agent_metrics["ORC"]["orc_invalid_events"] += 1
                failure_tags.append(
                    {
                        "dataset_index": record["dataset_index"],
                        "question": record["question"],
                        "agent": "ORC",
                        "kind": "orc_invalid",
                        "phase": phase,
                        "detail": payload.get("error"),
                    }
                )

        if role in {"GA", "PA", "VRA"} and phase == "step":
            agent_metrics[role]["specialist_step_events"] += 1
            row = {
                "dataset_index": record["dataset_index"],
                "idx": record["idx"],
                "question": record["question"],
                "step_id": step_id,
                "agent": role,
                "phase": phase,
                "turn": event.get("turn"),
                "subtask": step_to_subtask.get(step_id, ""),
                "user_message": payload.get("user_message", ""),
                "assistant_turn": payload.get("turn"),
                "error": payload.get("error", ""),
                "trace_index": idx,
            }
            conversation_rows.append(row)
            if payload.get("turn") is not None:
                agent_metrics[role]["specialist_valid_turns"] += 1
            if payload.get("error"):
                agent_metrics[role]["specialist_invalid_turns"] += 1
                failure_tags.append(
                    {
                        "dataset_index": record["dataset_index"],
                        "question": record["question"],
                        "agent": role,
                        "kind": "schema_invalid",
                        "step_id": step_id,
                        "detail": payload.get("error"),
                    }
                )

        if phase == "tool_call":
            tool_name = payload.get("tool", "")
            args = payload.get("args") or {}
            ok = bool(payload.get("ok"))
            text = str(payload.get("text", ""))
            produced_fields = ((payload.get("produced") or {}).get("fields") or {})
            failure_category = str(produced_fields.get("failure_category") or "")
            metric_agent = step_to_agent.get(step_id, role if role in {"GA", "PA", "VRA"} else "UNKNOWN")
            semantic_signature = _semantic_tool_signature(str(tool_name), dict(args))
            arg_name_valid = _validate_arg_names(tool_name, args)
            arg_value_valid = _arg_value_ok(tool_name, args, ok, text, failure_category)
            issue_type = "" if ok else _categorize_tool_issue(text, failure_category)
            agent_metrics[metric_agent]["tool_calls"] += 1
            agent_metrics[metric_agent]["tool_successes"] += int(ok)
            agent_metrics[metric_agent]["arg_name_valid"] += int(arg_name_valid)
            agent_metrics[metric_agent]["arg_value_valid"] += int(arg_value_valid)
            if "is not allowed to call" in text:
                agent_metrics[metric_agent]["wrong_tool_calls"] += 1
            tool_row = {
                "dataset_index": record["dataset_index"],
                "idx": record["idx"],
                "question": record["question"],
                "step_id": step_id,
                "subtask": step_to_subtask.get(step_id, ""),
                "agent": metric_agent,
                "turn": event.get("turn"),
                "tool": tool_name,
                "args": args,
                "ok": ok,
                "text": text,
                "issue_type": issue_type,
                "failure_category": failure_category,
                "produced": payload.get("produced", {}),
                "trace_index": idx,
                "semantic_signature": semantic_signature,
                "arg_name_valid": arg_name_valid,
                "arg_value_valid": arg_value_valid,
            }
            tool_rows.append(tool_row)
            actual_tool_events.append(tool_row)
            if not ok:
                failure_tags.append(
                    {
                        "dataset_index": record["dataset_index"],
                        "question": record["question"],
                        "agent": metric_agent,
                        "kind": "tool_error",
                        "step_id": step_id,
                        "tool": tool_name,
                        "detail": text,
                        "issue_type": issue_type,
                    }
                )

        if role in {"GA", "PA", "VRA"} and phase == "complete":
            summary = str(payload.get("summary", ""))
            status = str(payload.get("status", ""))
            specialist_complete_events.append(
                {
                    "agent": role,
                    "step_id": step_id,
                    "status": status,
                    "summary": summary,
                    "trace_index": idx,
                }
            )
            agent_metrics[role]["summary_events"] += 1
            agent_metrics[role]["summary_concrete"] += int(_summary_is_concrete(summary))
            agent_metrics[role]["task_complete"] += int(status == "TASK_COMPLETE")
            agent_metrics[role]["contract_failure"] += int(status == "CONTRACT_FAILURE")
            agent_metrics[role]["loop_failures"] += int("hit max inner turns" in summary)
            agent_metrics[role]["auto_downgrades"] += int("Auto-downgrade:" in summary)
            trajectory_rows.append(
                {
                    "dataset_index": record["dataset_index"],
                    "idx": record["idx"],
                    "question": record["question"],
                    "step_id": step_id,
                    "agent": role,
                    "subtask": step_to_subtask.get(step_id, ""),
                    "status": status,
                    "summary": summary,
                    "trace_index": idx,
                }
            )

    overall = _turn_agent_metrics_template()
    for metrics in agent_metrics.values():
        for key, value in metrics.items():
            overall[key] += value

    sequence_metrics, failure_buckets = _sequence_metrics_for_record(
        record,
        actual_tool_events=actual_tool_events,
        tool_rows=tool_rows,
        orc_delegate_events=orc_delegate_events,
        specialist_complete_events=specialist_complete_events,
    )

    return {
        "agent_raw_metrics": {agent: dict(metrics) for agent, metrics in sorted(agent_metrics.items())},
        "overall_raw_metrics": dict(overall),
        "agent_metrics": {agent: _finalize_agent_metrics(metrics) for agent, metrics in sorted(agent_metrics.items())},
        "overall_metrics": _finalize_agent_metrics(overall),
        "sequence_metrics": sequence_metrics,
        "failure_buckets": failure_buckets,
        "alignment_failure_tags": failure_tags,
        "datasets": {
            "agent_conversations": conversation_rows,
            "tool_calls": tool_rows,
            "trajectories": trajectory_rows,
        },
    }


def _build_samples(
    *,
    dataset_path: Path,
    image_dir_path: Path,
    num_samples: int,
    thinkgeo_only: bool,
) -> list[dict[str, Any]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("records"), list):
        raw = raw["records"]
    samples: list[dict[str, Any]] = []
    for dataset_index, item in enumerate(raw):
        conversation = item.get("conversation", []) or []
        question_source = item.get("question", "") or (conversation[0].get("value", "") if conversation else "")
        if not question_source:
            continue
        images = item.get("images", []) or []
        if thinkgeo_only and not any(Path(str(image)).name.startswith("TG") for image in images):
            continue
        resolved_images: list[str] = []
        for image in images:
            image_path = Path(str(image))
            if not image_path.is_absolute():
                image_path = image_dir_path / image_path.name
            resolved_images.append(str(image_path.resolve()))
        samples.append(
            {
                "dataset_index": dataset_index,
                "idx": item.get("idx", dataset_index),
                "question": _extract_question_text(question_source),
                "label": item.get("expected_final_answer", item.get("label") or item.get("final_answer") or _extract_gold_label(item)),
                "images": resolved_images,
                "scenario_case_id": dataset_index,
            }
        )
        if num_samples and len(samples) >= num_samples:
            break
    return samples


def _existing_conversation_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for path in output_dir.glob("conversation_*.json"):
        match = re.match(r"conversation_(\d+)\.json$", path.name)
        if not match:
            continue
        try:
            sample_no = int(match.group(1))
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append((sample_no, payload))
    return [payload for _, payload in sorted(records, key=lambda item: item[0])]


def _install_openai_backend(*, base_url: str, model: str, api_key: str) -> None:
    from multi_agent import llm as llm_mod

    if OpenAI is None:
        raise RuntimeError("openai package is required for the OpenAI-compatible backend")
    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    def _openai_call(messages: list[dict[str, Any]]) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "1024")),
        )
        return response.choices[0].message.content or ""

    llm_mod._BACKEND_CALLS["openai"] = _openai_call


def _install_local_qwen_backend(*, pretrained: str, max_new_tokens: int, adapter_root: str = "") -> None:
    from multi_agent import llm as llm_mod
    from multi_agent.prompts import SYSTEM_PROMPTS
    from evaluation.local_model_backend import build_local_chat_backend

    adapter_names_by_system = {
        SYSTEM_PROMPTS["ORC_PLAN"]: "ORC",
        SYSTEM_PROMPTS["ORC_DECIDE"]: "ORC",
        SYSTEM_PROMPTS["GA"]: "GA",
        SYSTEM_PROMPTS["PA"]: "PA",
        SYSTEM_PROMPTS["VRA"]: "VRA",
    }

    llm_mod._BACKEND_CALLS["local_qwen"] = build_local_chat_backend(
        pretrained=pretrained,
        max_new_tokens=max_new_tokens,
        adapter_root=adapter_root,
        system_to_adapter=adapter_names_by_system,
    )


def _resolve_artifact_handle(state: dict[str, Any], value: Any) -> Any:
    if not isinstance(value, str):
        return value
    artifacts = state.get("artifacts", {})
    record = artifacts.get(value)
    if record is None and value.startswith("geotiff_"):
        record = artifacts.get(f"tif_{value.split('_', 1)[1]}")
    if record is None and value.startswith("img_"):
        record = artifacts.get(f"image_{value.split('_', 1)[1]}")
    if record is None and value.startswith("image_"):
        record = artifacts.get(f"img_{value.split('_', 1)[1]}")
    if not isinstance(record, dict):
        return value
    return record.get("path", value)


def _resolve_tool_args(state: dict[str, Any], tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    args = dict(tool_args or {})
    for key in ("gpkg", "image", "geotiff", "pre_image", "post_image"):
        if key in args:
            args[key] = _resolve_artifact_handle(state, args[key])
    if tool_name == "AddText" and "color" not in args:
        args["color"] = "green"
    return args


def _artifact_handle_for_value(state: dict[str, Any], value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    artifacts = state.get("artifacts", {})
    if value in artifacts:
        return value
    if value.startswith("geotiff_") and f"tif_{value.split('_', 1)[1]}" in artifacts:
        return f"tif_{value.split('_', 1)[1]}"
    if value.startswith("img_") and f"image_{value.split('_', 1)[1]}" in artifacts:
        return f"image_{value.split('_', 1)[1]}"
    if value.startswith("image_") and f"img_{value.split('_', 1)[1]}" in artifacts:
        return f"img_{value.split('_', 1)[1]}"
    for handle, record in artifacts.items():
        if isinstance(record, dict) and str(record.get("path", "")) == value:
            return str(handle)
    return ""


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric leaves without corrupting scientific notation.

    Evaluation records normally contain JSON answers, so parse those
    structurally first. The regex fallback is reserved for diagnostic prose.
    """

    def leaves(value: Any) -> list[float]:
        if isinstance(value, bool) or value is None:
            return []
        if isinstance(value, (int, float)):
            number = float(value)
            return [number] if math.isfinite(number) else []
        if isinstance(value, dict):
            return [number for child in value.values() for number in leaves(child)]
        if isinstance(value, list):
            return [number for child in value for number in leaves(child)]
        return []

    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if parsed is not None and not isinstance(parsed, str):
        return leaves(parsed)
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return [float(match) for match in re.findall(pattern, raw)]


def _parse_compute_distance_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*([^,\n]+?)\s*,\s*([^,\n]+?),\s*distance=([-+]?\d*\.?\d+)\s*m"
        r"(?:,\s*travel_time=([-+]?\d*\.?\d+)\s*s)?\s*$",
        flags=re.I | re.M,
    )
    for match in pattern.finditer(str(text or "")):
        src = match.group(1).strip()
        tar = match.group(2).strip()
        distance_m = round(float(match.group(3)), 2)
        row: dict[str, Any] = {
            "src": src,
            "tar": tar,
            "pair": f"{src} -> {tar}",
            "distance_m": distance_m,
        }
        if match.group(4):
            row["travel_time_s"] = round(float(match.group(4)), 2)
        rows.append(row)
    return rows


def _infer_fields(tool_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    text = str(raw.get("text", "")).strip()
    fields: dict[str, Any] = {}
    if tool_name == "AddPoisLayer":
        match = re.search(r"Saved\s+(\d+)\s+POIs\s+to\s+layer\s+'([^']+)'", text)
        if match:
            fields[f"{match.group(2)}_count"] = int(match.group(1))
    elif tool_name == "ComputeDistance":
        rows = _parse_compute_distance_rows(text)
        if rows:
            distances = [float(row["distance_m"]) for row in rows]
            fields["distance_rows"] = rows
            fields["distances_m"] = distances
            fields["distance_pair_count"] = len(rows)
            closest = min(rows, key=lambda row: float(row["distance_m"]))
            farthest = max(rows, key=lambda row: float(row["distance_m"]))
            fields["pair"] = closest["pair"]
            fields["distance_m"] = closest["distance_m"]
            fields["min_pair"] = closest["pair"]
            fields["min_distance_m"] = closest["distance_m"]
            fields["max_pair"] = farthest["pair"]
            fields["max_distance_m"] = farthest["distance_m"]
            fields["avg_distance_m"] = round(sum(distances) / len(distances), 2)
            if any("travel_time_s" in row for row in rows):
                travel_times = [float(row["travel_time_s"]) for row in rows if "travel_time_s" in row]
                fields["travel_times_s"] = travel_times
                best_time = closest.get("travel_time_s")
                if best_time is not None:
                    fields["travel_time_s"] = best_time
    elif tool_name in {"Calculator", "CountGivenObject"}:
        numbers = _extract_numbers(text)
        if numbers:
            value = numbers[0]
            fields["value"] = int(value) if value.is_integer() else value
            fields["count"] = int(value) if value.is_integer() else value
    elif tool_name == "TextToBbox":
        bboxes = raw.get("bboxes")
        if isinstance(bboxes, list):
            fields["count"] = len(bboxes)
            fields["bboxes"] = bboxes
    elif tool_name == "ObjectDetection":
        bboxes = raw.get("bboxes")
        if isinstance(bboxes, list):
            fields["count"] = len(bboxes)
            fields["bboxes"] = bboxes
    elif tool_name == "SegmentObjectPixels":
        text_value = str(raw.get("text", "")).strip()
        try:
            literal = json.loads(text_value)
        except Exception:
            literal = None
        if isinstance(literal, list):
            fields["pixel_counts"] = literal
            fields["pixel_count"] = int(sum(float(v) for v in literal))
        else:
            numbers = _extract_numbers(text_value)
            if numbers:
                value = numbers[0]
                fields["pixel_count"] = int(value) if value.is_integer() else value
    elif tool_name == "OCR":
        fields["ocr_text"] = text
    elif tool_name in {"ImageDescription", "RegionAttributeDescription", "ChangeDetection", "GoogleSearch", "Solver"}:
        if text:
            fields["result_text"] = text
    elif tool_name == "GetBboxFromGeotiff":
        nums = _extract_numbers(text)
        if len(nums) >= 4:
            fields["value"] = nums[:4]
    return fields


def _infer_layers(tool_name: str, resolved_args: dict[str, Any], state: dict[str, Any], tool_args: dict[str, Any]) -> list[tuple[str, str]]:
    if tool_name == "GetAreaBoundary":
        gpkg = str(resolved_args.get("gpkg", "__autogen__"))
        return [(gpkg, "area_boundary")]
    if tool_name in {"AddPoisLayer", "AddIndexLayer"}:
        layer_name = str(resolved_args.get("layer_name", "")).strip()
        gpkg = _artifact_handle_for_value(state, tool_args.get("gpkg")) or str(resolved_args.get("gpkg", "__autogen__"))
        return [(gpkg, layer_name)] if layer_name else []
    if tool_name == "ComputeDistance":
        src = str(resolved_args.get("src_layer", "")).strip()
        tar = str(resolved_args.get("tar_layer", "")).strip()
        gpkg = _artifact_handle_for_value(state, tool_args.get("gpkg")) or str(resolved_args.get("gpkg", "__autogen__"))
        return [(gpkg, f"{src}_to_{tar}_distances")] if src and tar else []
    if tool_name == "ComputeIndexChange":
        diff = str(resolved_args.get("diff_layer_name", "")).strip()
        gpkg = _artifact_handle_for_value(state, tool_args.get("gpkg")) or str(resolved_args.get("gpkg", "__autogen__"))
        return [(gpkg, diff)] if diff else []
    return []


def _live_call_tool_factory(tool_manager: ToolManager, schemas_mod: Any):
    def _live_call_tool(tool_name: str, tool_args: dict[str, Any]):
        state = _LIVE_CONTEXT.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("live tool call received no active state")
        resolved_args = _resolve_tool_args(state, tool_name, tool_args)
        call_args = dict(resolved_args)
        scenario_session_id = str(state.get("scenario_session_id", "")).strip()
        if scenario_session_id:
            call_args["__scenario_session_id"] = scenario_session_id
            call_args["__scenario_case_id"] = int(state.get("scenario_case_id", -1))
            call_args["__scenario_original_args"] = dict(tool_args or {})
        raw = tool_manager.call_tool(tool_name, call_args)
        error_code = raw.get("error_code", 1)
        ok = int(error_code) == 0 if error_code is not None else False
        produced = schemas_mod.Produced()

        if ok:
            raw_produced = raw.get("produced")
            used_explicit_produced = False
            if isinstance(raw_produced, dict):
                used_explicit_produced = True
                for art in raw_produced.get("artifacts", []) or []:
                    if not isinstance(art, dict) or not art.get("path") or not art.get("kind"):
                        continue
                    produced.artifacts.append(
                        schemas_mod.ArtifactRecord(
                            handle=str(art.get("handle", "__autogen__")),
                            kind=str(art["kind"]),
                            path=str(art["path"]),
                            note=str(art.get("note", f"{tool_name} output")),
                        )
                    )
                for lay in raw_produced.get("layers", []) or []:
                    if not isinstance(lay, dict) or not lay.get("name"):
                        continue
                    produced.layers.append(
                        schemas_mod.LayerRecord(
                            gpkg=str(lay.get("gpkg", "__autogen__")),
                            name=str(lay["name"]),
                        )
                    )
                for field_name, value in (raw_produced.get("fields", {}) or {}).items():
                    produced.fields[str(field_name)] = value
            if isinstance(raw.get("gpkg"), str) and raw.get("gpkg"):
                produced.artifacts.append(
                    schemas_mod.ArtifactRecord(
                        handle="__autogen__",
                        kind="gpkg",
                        path=str(raw["gpkg"]),
                        note=f"{tool_name} output",
                    )
                )
            if isinstance(raw.get("image"), str) and raw.get("image"):
                produced.artifacts.append(
                    schemas_mod.ArtifactRecord(
                        handle="__autogen__",
                        kind="img",
                        path=str(raw["image"]),
                        note=f"{tool_name} output",
                    )
                )
            if isinstance(raw.get("geotiff"), str) and raw.get("geotiff"):
                produced.artifacts.append(
                    schemas_mod.ArtifactRecord(
                        handle="__autogen__",
                        kind="tif",
                        path=str(raw["geotiff"]),
                        note=f"{tool_name} output",
                    )
                )

            if not used_explicit_produced:
                gpkg_ref = "__autogen__" if any(art.kind == "gpkg" for art in produced.artifacts) else str(resolved_args.get("gpkg", ""))
                for _, layer_name in _infer_layers(tool_name, resolved_args, state, tool_args):
                    if layer_name:
                        layer_gpkg = _artifact_handle_for_value(state, tool_args.get("gpkg")) or gpkg_ref
                        produced.layers.append(schemas_mod.LayerRecord(gpkg=layer_gpkg, name=layer_name))

                for field_name, value in _infer_fields(tool_name, raw).items():
                    produced.fields[field_name] = value

        return schemas_mod.ToolResult(ok=ok, text=str(raw.get("text", "")), produced=produced)

    return _live_call_tool


def _episode_for_sample(sample: dict[str, Any], state_mod: Any):
    state = state_mod.new_episode(question=sample["question"], images=list(sample["images"]))
    state["scenario_case_id"] = int(sample.get("scenario_case_id", sample["dataset_index"]))
    state["scenario_session_id"] = f"scenario-{int(time.time() * 1000000)}-{sample['dataset_index']}"
    for image_path in sample["images"]:
        suffix = Path(image_path).suffix.lower()
        kind = "tif" if suffix in {".tif", ".tiff"} else "img"
        note = "input scene" if kind == "tif" else "input image"
        state["counters"][kind] = state["counters"].get(kind, 0) + 1
        handle_prefix = "image" if kind == "img" else kind
        handle = f"{handle_prefix}_{state['counters'][kind]}"
        state["artifacts"][handle] = {
            "kind": kind,
            "path": image_path,
            "produced_by_step": "",
            "note": note,
        }
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GeoDisaster multi-agent pipeline with live tools and a live model backend.")
    parser.add_argument("--dataset_path", default="data/openearth_agent/test.json")
    parser.add_argument("--image_dir_path", default="data/openearth_agent/test")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--thinkgeo_only", action="store_true")
    parser.add_argument("--controller_addr", default=os.environ.get("MAS_CONTROLLER_ADDR", "http://127.0.0.1:20001"))
    parser.add_argument("--model_backend", default=os.environ.get("MODEL_BACKEND", "local_qwen"))
    parser.add_argument("--model_path", default=os.environ.get("GEODISASTER_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--openai_base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai_model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAS_MAX_NEW_TOKENS", "1024")))
    parser.add_argument("--recursion_limit", type=int, default=120)
    parser.add_argument("--adapter_root", default=os.environ.get("MAS_ADAPTER_ROOT", ""))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.model_backend == "openai" and not args.openai_base_url:
        raise SystemExit("MODEL_BACKEND=openai requires --openai_base_url or OPENAI_BASE_URL")

    _maybe_init_accelerate()

    os.environ["MODEL_BACKEND"] = args.model_backend
    if args.model_backend == "openai":
        _install_openai_backend(
            base_url=args.openai_base_url,
            model=args.openai_model,
            api_key=args.openai_api_key,
        )
    elif args.model_backend == "local_qwen":
        _install_local_qwen_backend(
            pretrained=args.model_path,
            max_new_tokens=args.max_new_tokens,
            adapter_root=args.adapter_root,
        )
    else:
        raise SystemExit(f"Unsupported model_backend: {args.model_backend}")

    from multi_agent import agents as agents_mod
    from multi_agent import tool_gateway as tool_gateway_mod
    from multi_agent import prompts as prompts_mod
    from multi_agent import schemas as schemas_mod
    from multi_agent import state as state_mod

    tool_manager = ToolManager(args.controller_addr)
    live_call_tool = _live_call_tool_factory(tool_manager, schemas_mod)

    original_specialist_node = agents_mod.specialist_node

    def _specialist_node_with_context(state):
        previous = _LIVE_CONTEXT.get("state")
        _LIVE_CONTEXT["state"] = state
        try:
            return original_specialist_node(state)
        finally:
            _LIVE_CONTEXT["state"] = previous

    agents_mod.call_tool = live_call_tool
    agents_mod.is_tool_allowed = tool_gateway_mod.is_tool_allowed
    agents_mod.specialist_node = _specialist_node_with_context
    prompts_mod.AGENT_TOOLS = tool_gateway_mod.AGENT_TOOLS

    from multi_agent.graph import build_graph

    dataset_path = Path(args.dataset_path).resolve()
    image_dir_path = Path(args.image_dir_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        Path("runs") / f"live_{'thinkgeo' if args.thinkgeo_only else 'geodisaster'}_{time.strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _build_samples(
        dataset_path=dataset_path,
        image_dir_path=image_dir_path,
        num_samples=args.num_samples,
        thinkgeo_only=args.thinkgeo_only,
    )
    if not samples:
        raise SystemExit("No samples selected.")

    graph = build_graph()
    results: list[dict[str, Any]] = []
    agent_conversations_path = output_dir / "agent_conversations.jsonl"
    tool_calls_path = output_dir / "tool_calls.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    failure_tags_path = output_dir / "alignment_failure_tags.jsonl"
    existing_records: list[dict[str, Any]] = []
    if args.resume:
        existing_records = _existing_conversation_records(output_dir)
        results.extend(existing_records)
    else:
        for export_path in (agent_conversations_path, tool_calls_path, trajectories_path, failure_tags_path):
            if export_path.exists():
                export_path.unlink()

    start_index = len(existing_records)
    remaining_samples = samples[start_index:]
    for sample_no, sample in enumerate(remaining_samples, start=start_index + 1):
        state = _episode_for_sample(sample, state_mod)
        started = time.time()
        try:
            final_state = graph.invoke(state, {"recursion_limit": args.recursion_limit})
        except Exception as exc:
            final_state = state
            final_state["status"] = "ABORTED"
            final_state["abort_reason"] = f"runtime_exception: {exc}"
            final_state["final_answer"] = ""
        wall_time_s = round(time.time() - started, 3)
        metrics = _score_answer(sample["label"], final_state.get("final_answer") or "")
        record = {
            "dataset_index": sample["dataset_index"],
            "idx": sample["idx"],
            "question": sample["question"],
            "label": sample["label"],
            "images": sample["images"],
            "status": final_state.get("status"),
            "abort_reason": final_state.get("abort_reason"),
            "final_answer": final_state.get("final_answer"),
            "wall_time_s": wall_time_s,
            "plan": final_state.get("plan").model_dump() if final_state.get("plan") is not None else None,
            "steps": final_state.get("steps"),
            "shared_state": {
                "artifacts": final_state.get("artifacts"),
                "layers": final_state.get("layers"),
                "fields": {
                    name: {"value": value["value"], "produced_by_step": value["produced_by_step"]}
                    for name, value in (final_state.get("fields") or {}).items()
                },
            },
            "trace": final_state.get("history"),
            "metrics": metrics,
        }
        analysis = _analyze_record(record)
        record["alignment_metrics"] = analysis["overall_metrics"]
        record["alignment_raw_metrics"] = analysis["overall_raw_metrics"]
        record["agent_metrics"] = analysis["agent_metrics"]
        record["agent_raw_metrics"] = analysis["agent_raw_metrics"]
        record["sequence_metrics"] = analysis["sequence_metrics"]
        record["failure_buckets"] = analysis["failure_buckets"]
        results.append(record)
        (output_dir / f"conversation_{sample_no}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        for row in analysis["datasets"]["agent_conversations"]:
            with agent_conversations_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        for row in analysis["datasets"]["tool_calls"]:
            with tool_calls_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        for row in analysis["datasets"]["trajectories"]:
            with trajectories_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        for row in analysis["alignment_failure_tags"]:
            with failure_tags_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(
            f"[{sample_no}/{len(samples)}] dataset_index={sample['dataset_index']} "
            f"status={record['status']} answer_acc={metrics['answer_acc']:.3f} "
            f"answer={record['final_answer']!r}"
        )

    aggregate_agent_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in results:
        for agent, metrics in (row.get("agent_raw_metrics") or {}).items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    aggregate_agent_counts[agent][key] += float(value)

    per_agent_summary: dict[str, Any] = {}
    for agent, totals in sorted(aggregate_agent_counts.items()):
        raw = _turn_agent_metrics_template()
        for key in raw:
            raw[key] = totals.get(key, 0.0)
        per_agent_summary[agent] = _finalize_agent_metrics(raw)

    aggregate_overall_raw = _turn_agent_metrics_template()
    for row in results:
        metrics = row.get("alignment_raw_metrics") or {}
        for key in aggregate_overall_raw:
            aggregate_overall_raw[key] += float(metrics.get(key, 0.0))
    alignment_summary = _finalize_agent_metrics(aggregate_overall_raw)
    sequence_summary = _aggregate_sequence_metrics(results)
    failure_bucket_summary = _aggregate_failure_buckets(results)

    summary = {
        "n_samples": len(results),
        "dataset_path": str(dataset_path),
        "image_dir_path": str(image_dir_path),
        "thinkgeo_only": args.thinkgeo_only,
        "controller_addr": args.controller_addr,
        "model_backend": args.model_backend,
        "model_path": args.model_path if args.model_backend == "local_qwen" else "",
        "adapter_root": args.adapter_root if args.model_backend == "local_qwen" else "",
        "openai_base_url": args.openai_base_url if args.model_backend == "openai" else "",
        "openai_model": args.openai_model if args.model_backend == "openai" else "",
        "task_complete_rate": sum(1 for row in results if row["status"] == "DONE") / len(results),
        "answer_acc_avg": sum(row["metrics"]["answer_acc"] for row in results) / len(results),
        "token_f1_avg": sum(row["metrics"]["token_f1"] for row in results) / len(results),
        "sequence_ratio_avg": sum(row["metrics"]["sequence_ratio"] for row in results) / len(results),
        "exact_match_rate": sum(1 for row in results if row["metrics"]["exact_match"]) / len(results),
        "numeric_match_rate": sum(1 for row in results if row["metrics"]["numeric_match"]) / len(results),
        "numeric_subset_match_rate": sum(1 for row in results if row["metrics"]["numeric_subset_match"]) / len(results),
        "correct_rate": sum(1 for row in results if row["metrics"]["correct"]) / len(results),
        "alignment_metrics": alignment_summary,
        "sequence_metrics": sequence_summary,
        "failure_buckets": failure_bucket_summary,
        "per_agent_metrics": per_agent_summary,
        "exports": {
            "agent_conversations_jsonl": str(agent_conversations_path),
            "tool_calls_jsonl": str(tool_calls_path),
            "trajectories_jsonl": str(trajectories_path),
            "alignment_failure_tags_jsonl": str(failure_tags_path),
        },
        "results": [
            {
                "dataset_index": row["dataset_index"],
                "idx": row["idx"],
                "status": row["status"],
                "abort_reason": row["abort_reason"],
                "final_answer": row["final_answer"],
                "metrics": row["metrics"],
                "alignment_metrics": row["alignment_metrics"],
                "sequence_metrics": row.get("sequence_metrics", {}),
                "failure_buckets": row.get("failure_buckets", {}),
                "agent_metrics": row["agent_metrics"],
                "wall_time_s": row["wall_time_s"],
            }
            for row in results
        ],
        "outside_files_used": [
            "tool_server/tool_workers/tool_manager/base_manager.py",
            str(dataset_path),
            str(image_dir_path),
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Wrote summary -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
