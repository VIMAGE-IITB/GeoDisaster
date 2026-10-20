from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent  # code/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from tool_server.tool_workers.online_workers.final_dataset_tool_specs import TOOL_SPECS

# Per-tool (required, optional) argument signature derived from the live tool specs.
ARG_SPEC: dict[str, tuple[set[str], set[str]]] = {
    name: (
        set(spec.get("required", [])),
        set(spec.get("properties", {})) - set(spec.get("required", [])) - {"session_id"},
    )
    for name, spec in TOOL_SPECS.items()
}
from evaluation.eval_live import _score_answer


IMAGE_PRODUCERS = {"DisplayOnMap", "DisplayOnGeotiff", "ShowIndexLayer", "DrawBox", "AddText", "Plot"}
GENERIC_SUBTASK_PATTERNS = (
    "execute required",
    "execute vra",
    "compute derived values",
    "synthesize final",
    "use the available evidence",
)


class MetricState:
    def __init__(self, images: list[str] | None = None) -> None:
        self.artifact_counters: Counter[str] = Counter()
        self.artifact_aliases: dict[str, str] = {}
        self.layer_semantics: dict[str, str] = {}
        self.fields: set[str] = set()
        for image in images or []:
            suffix = Path(str(image)).suffix.lower()
            kind = "tif" if suffix in {".tif", ".tiff"} else "image"
            self.artifact_counters[kind] += 1
            handle = f"{kind}_{self.artifact_counters[kind]}"
            self._register_artifact_aliases(handle, [str(image), Path(str(image)).name])
            if kind == "image":
                self.artifact_aliases[f"img_{self.artifact_counters[kind]}"] = handle

    def _register_artifact_aliases(self, handle: str, aliases: list[str]) -> None:
        self.artifact_aliases[_norm_ref(handle)] = handle
        for alias in aliases:
            if alias:
                self.artifact_aliases[_norm_ref(alias)] = handle

    def add_produced(self, tool: str, args: dict[str, Any], produced: dict[str, Any]) -> None:
        if tool == "AddPoisLayer":
            layer = str(args.get("layer_name", "")).strip()
            if layer:
                self.layer_semantics[layer] = _tag_signature(args.get("query"))
        elif tool == "AddIndexLayer":
            layer = str(args.get("layer_name", "")).strip()
            if layer:
                self.layer_semantics[layer] = _signature_for_call(tool, args, self)
        elif tool == "ComputeIndexChange":
            layer = str(args.get("diff_layer_name", "")).strip()
            if layer:
                self.layer_semantics[layer] = _signature_for_call(tool, args, self)
        elif tool == "ComputeDistance":
            src = str(args.get("src_layer", "")).strip()
            tar = str(args.get("tar_layer", "")).strip()
            if src and tar:
                self.layer_semantics[f"{src}_to_{tar}_distances"] = _signature_for_call(tool, args, self)

        for layer in produced.get("layers") or []:
            name = str((layer or {}).get("name", "")).strip()
            if name and name not in self.layer_semantics:
                self.layer_semantics[name] = _norm_text(name)
        for field_name in (produced.get("fields") or {}):
            self.fields.add(str(field_name))

        for artifact in produced.get("artifacts") or []:
            kind = str((artifact or {}).get("kind", "img")).strip()
            path = str((artifact or {}).get("path", "")).strip()
            if kind == "gpkg":
                self.artifact_counters["gpkg"] = max(self.artifact_counters["gpkg"], 1)
                self._register_artifact_aliases("gpkg_1", [path, Path(path).name])
            elif kind == "tif":
                self.artifact_counters["tif"] += 1
                self._register_artifact_aliases(f"tif_{self.artifact_counters['tif']}", [path, Path(path).name])
            else:
                self.artifact_counters["image"] += 1
                self._register_artifact_aliases(f"image_{self.artifact_counters['image']}", [path, Path(path).name])

        if tool in IMAGE_PRODUCERS and not produced.get("artifacts"):
            self.artifact_counters["image"] += 1
            self._register_artifact_aliases(f"image_{self.artifact_counters['image']}", _output_aliases_from_text(""))

    def resolve_ref(self, value: Any) -> str:
        text = _norm_ref(value)
        return self.artifact_aliases.get(text, text)

    def layer_semantic(self, name: Any) -> str:
        layer = str(name or "").strip()
        return self.layer_semantics.get(layer, _norm_text(layer))


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\\n", " ").strip()).lower()


def _norm_ref(value: Any) -> str:
    text = _norm_text(value).rstrip(".")
    match = re.fullmatch(r"img_(\d+)", text)
    if match:
        return f"image_{match.group(1)}"
    match = re.fullmatch(r"geotiff_(\d+)", text)
    if match:
        return f"tif_{match.group(1)}"
    match = re.fullmatch(r"tiff_(\d+)", text)
    if match:
        return f"tif_{match.group(1)}"
    return text


def _norm_gpkg(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "gpkg_1"
    if re.fullmatch(r"gpkg_\d+", text):
        return text
    if "gpkg" in text.lower() or text.lower().endswith(".gpkg"):
        match = re.search(r"gpkg[_-]?(\d+)", text, flags=re.I)
        return f"gpkg_{match.group(1)}" if match else "gpkg_1"
    return text


def _tag_signature(query: Any) -> str:
    if not isinstance(query, dict) or len(query) != 1:
        return f"tag={_norm_text(query)}"
    key, value = next(iter(query.items()))
    return f"tag={_norm_text(key)}={_norm_text(value)}"


def _numbers_signature(value: Any) -> str:
    nums = re.findall(r"[-+]?\d*\.?\d+", str(value or ""))
    return ",".join(nums)


def _normalize_expr(expr: Any) -> str:
    text = str(expr or "").strip().lower().replace(" ", "")
    text = text.replace("math.sqrt", "sqrt")
    return text


def _signature_for_call(tool: str, args: dict[str, Any], state: MetricState) -> str:
    args = dict(args or {})
    if tool == "GetAreaBoundary":
        buffer_value = args.get("buffer_m")
        buffer_part = "" if buffer_value in {None, ""} else f"|buffer_m={buffer_value}"
        return f"GetAreaBoundary|area={_norm_text(args.get('area'))}{buffer_part}"
    if tool == "GetBboxFromGeotiff":
        return f"GetBboxFromGeotiff|geotiff={state.resolve_ref(args.get('geotiff'))}"
    if tool == "AddPoisLayer":
        return f"AddPoisLayer|gpkg={_norm_gpkg(args.get('gpkg'))}|{_tag_signature(args.get('query'))}"
    if tool == "ComputeDistance":
        return (
            f"ComputeDistance|gpkg={_norm_gpkg(args.get('gpkg'))}"
            f"|src={state.layer_semantic(args.get('src_layer'))}|tar={state.layer_semantic(args.get('tar_layer'))}"
        )
    if tool in {"DisplayOnMap", "DisplayOnGeotiff"}:
        layers = sorted(state.layer_semantic(layer) for layer in args.get("layers") or [])
        geo = f"|geotiff={state.resolve_ref(args.get('geotiff'))}" if tool == "DisplayOnGeotiff" else ""
        return f"{tool}|gpkg={_norm_gpkg(args.get('gpkg'))}|layers={','.join(layers)}{geo}"
    if tool == "AddIndexLayer":
        return (
            f"AddIndexLayer|gpkg={_norm_gpkg(args.get('gpkg'))}|index={_norm_text(args.get('index_type'))}"
            f"|year={args.get('year', '')}|month={args.get('month', '')}"
        )
    if tool == "ComputeIndexChange":
        return (
            f"ComputeIndexChange|gpkg={_norm_gpkg(args.get('gpkg'))}|index={_norm_text(args.get('index_type'))}"
            f"|l1={state.layer_semantic(args.get('layer1_name'))}|l2={state.layer_semantic(args.get('layer2_name'))}"
        )
    if tool == "ShowIndexLayer":
        return f"ShowIndexLayer|gpkg={_norm_gpkg(args.get('gpkg'))}|index={_norm_text(args.get('index_type'))}|layer={state.layer_semantic(args.get('layer_name'))}"
    if tool in {"TextToBbox", "CountGivenObject", "SegmentObjectPixels"}:
        flag = f"|flag={str(args.get('flag')).lower()}" if "flag" in args else ""
        return f"{tool}|image={state.resolve_ref(args.get('image'))}|text={_norm_text(args.get('text'))}{flag}"
    if tool == "RegionAttributeDescription":
        bbox = f"|bbox={_numbers_signature(args.get('bbox'))}" if args.get("bbox") is not None else ""
        return f"{tool}|image={state.resolve_ref(args.get('image'))}|attribute={_norm_text(args.get('attribute'))}{bbox}"
    if tool == "ChangeDetection":
        return (
            f"ChangeDetection|text={_norm_text(args.get('text'))}"
            f"|pre={state.resolve_ref(args.get('pre_image'))}|post={state.resolve_ref(args.get('post_image'))}"
        )
    if tool == "DrawBox":
        return f"DrawBox|image={state.resolve_ref(args.get('image'))}|bbox={_numbers_signature(args.get('bbox'))}|annotation_nums={_numbers_signature(args.get('annotation'))}"
    if tool == "AddText":
        return f"AddText|image={state.resolve_ref(args.get('image'))}|text_nums={_numbers_signature(args.get('text'))}|position={_norm_text(args.get('position'))}"
    if tool == "Calculator":
        return f"Calculator|expr={_normalize_expr(args.get('expression'))}"
    if tool == "Solver":
        return f"Solver|nums={_numbers_signature(args.get('command'))}"
    if tool == "Plot":
        return f"Plot|nums={_numbers_signature(args.get('command'))}"
    if tool == "GoogleSearch":
        return f"GoogleSearch|query={_norm_text(args.get('query'))}"
    if tool in {"ObjectDetection", "ImageDescription", "OCR"}:
        return f"{tool}|image={state.resolve_ref(args.get('image'))}"
    return f"{tool}|{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"


def _output_aliases_from_text(text: str) -> list[str]:
    aliases = []
    raw = str(text or "")
    for match in re.findall(r"(?:saved to|->)\s+([^\s,;]+)", raw, flags=re.I):
        aliases.append(match.replace("\\n", "").strip().strip("\"'").rstrip("."))
    stripped = raw.replace("\\n", "").strip().strip("\"'").rstrip(".")
    if re.search(r"\.(png|jpg|jpeg|tif|tiff)$", stripped, flags=re.I):
        aliases.append(stripped)
    return [alias for alias in aliases if alias]


def _load_teacher_dir(path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    for file_path in sorted(path.glob("multiagent_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        records[int(file_path.stem.split("_")[-1])] = json.loads(file_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"No multiagent_*.json files found in {path}")
    return records


def _load_eval_records(path: Path) -> dict[int, dict[str, Any]]:
    out = {}
    for file_path in sorted(path.glob("conversation_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        sample_no = int(file_path.stem.split("_")[-1]) - 1
        out[sample_no] = json.loads(file_path.read_text(encoding="utf-8"))
    if not out:
        raise ValueError(f"No conversation_*.json files found in {path}")
    return out


def _teacher_tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for execution in record.get("executions") or []:
        agent = str(execution.get("agent", ""))
        step_id = str(execution.get("step_id", ""))
        for turn in execution.get("turns") or []:
            action = turn.get("action") or {}
            tool = str(action.get("tool", ""))
            if not tool or tool == "TaskComplete":
                continue
            calls.append(
                {
                    "agent": agent,
                    "step_id": step_id,
                    "tool": tool,
                    "args": dict(action.get("args") or {}),
                    "ok": True,
                    "produced": dict((turn.get("tool_output") or {}).get("produced") or {}),
                }
            )
    return calls


def _actual_tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    step_agents = {str(step.get("id")): str(step.get("agent")) for step in (record.get("steps") or [])}
    step_agents.update({str(step.get("id")): str(step.get("agent")) for step in ((record.get("plan") or {}).get("steps") or [])})
    calls = []
    for event in record.get("trace") or []:
        if event.get("role") != "tool" or event.get("phase") != "tool_call":
            continue
        payload = event.get("payload") or {}
        step_id = str(event.get("step_id") or "")
        calls.append(
            {
                "agent": step_agents.get(step_id, ""),
                "step_id": step_id,
                "turn": event.get("turn"),
                "tool": str(payload.get("tool", "")),
                "args": dict(payload.get("args") or {}),
                "ok": bool(payload.get("ok", False)),
                "text": str(payload.get("text", "")),
                "produced": dict(payload.get("produced") or {}),
            }
        )
    return calls


def _signature_sequence(calls: list[dict[str, Any]], images: list[str]) -> list[str]:
    state = MetricState(images)
    signatures = []
    for call in calls:
        tool = str(call.get("tool", ""))
        args = dict(call.get("args") or {})
        signatures.append(_signature_for_call(tool, args, state))
        if call.get("ok", True):
            state.add_produced(tool, args, dict(call.get("produced") or {}))
    return signatures


def _arg_names_valid(tool: str, args: dict[str, Any]) -> bool:
    if tool not in ARG_SPEC:
        return False
    required, optional = ARG_SPEC[tool]
    keys = set(args)
    return required.issubset(keys) and not (keys - required - optional)


def _handle_arg_values(args: dict[str, Any]) -> list[str]:
    values = []
    for key in ("gpkg", "image", "geotiff", "pre_image", "post_image"):
        value = args.get(key)
        if isinstance(value, str):
            values.append(value)
    return values


def _is_canonical_ref(value: str) -> bool:
    return bool(re.fullmatch(r"(gpkg|image|tif)_\d+", value))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _greedy_coverage(expected: list[str], actual: list[str]) -> tuple[int, int | None]:
    pos = 0
    completion_idx = None
    for idx, sig in enumerate(actual):
        if pos < len(expected) and sig == expected[pos]:
            pos += 1
            if pos == len(expected):
                completion_idx = idx
                break
    return pos, completion_idx


def _counter_prf(expected: list[str], actual: list[str]) -> dict[str, float]:
    exp = Counter(expected)
    act = Counter(actual)
    overlap = sum(min(count, act.get(sig, 0)) for sig, count in exp.items())
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"unordered_precision": precision, "unordered_recall": recall, "unordered_f1": f1}


def _first_divergence(expected: list[str], actual: list[str]) -> int | None:
    limit = min(len(expected), len(actual))
    for idx in range(limit):
        if expected[idx] != actual[idx]:
            return idx + 1
    if len(expected) != len(actual):
        return limit + 1
    return None


def _plan_metrics(gt: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    gt_steps = list((gt.get("plan") or {}).get("steps") or [])
    actual_steps = list((actual.get("plan") or {}).get("steps") or [])
    gt_roles = [str(step.get("agent", "")) for step in gt_steps]
    actual_roles = [str(step.get("agent", "")) for step in actual_steps]
    generic_subtasks = sum(
        1 for step in actual_steps if any(pattern in _norm_text(step.get("subtask", "")) for pattern in GENERIC_SUBTASK_PATTERNS)
    )
    vague_needs = sum(
        1
        for step in actual_steps
        for need in (step.get("needs") or [])
        if str(need) in {"layer source", "layer target"} or str(need).startswith(("step", "previous"))
    )
    return {
        "gt_step_count": len(gt_steps),
        "actual_step_count": len(actual_steps),
        "step_count_match": len(gt_steps) == len(actual_steps),
        "role_sequence_match": gt_roles == actual_roles,
        "role_lcs_ratio": _lcs_len(gt_roles, actual_roles) / len(gt_roles) if gt_roles else 0.0,
        "generic_subtask_count": generic_subtasks,
        "vague_need_count": vague_needs,
        "plan_valid_shape": bool(actual_steps and all(step.get("id") and step.get("agent") and step.get("subtask") for step in actual_steps)),
    }


def _complete_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in record.get("trace") or []
        if event.get("role") in {"GA", "PA", "VRA"} and event.get("phase") == "complete"
    ]


def evaluate_case(case_id: int, gt: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    images = list(gt.get("images") or actual.get("images") or [])
    expected_calls = _teacher_tool_calls(gt)
    actual_calls = _actual_tool_calls(actual)
    expected_sigs = _signature_sequence(expected_calls, images)
    actual_sigs = _signature_sequence(actual_calls, images)
    lcs = _lcs_len(expected_sigs, actual_sigs)
    greedy, completion_idx = _greedy_coverage(expected_sigs, actual_sigs)
    prf = _counter_prf(expected_sigs, actual_sigs)
    first_div = _first_divergence(expected_sigs, actual_sigs)

    expected_counter = Counter(expected_sigs)
    actual_counter = Counter(actual_sigs)
    overcalls = sum(max(count - expected_counter.get(sig, 0), 0) for sig, count in actual_counter.items())
    missing = sum(max(count - actual_counter.get(sig, 0), 0) for sig, count in expected_counter.items())
    redundant = sum(max(count - 1, 0) for count in actual_counter.values())

    arg_valid = [_arg_names_valid(call["tool"], dict(call.get("args") or {})) for call in actual_calls]
    handle_values = [value for call in actual_calls for value in _handle_arg_values(dict(call.get("args") or {}))]
    complete_events = _complete_events(actual)
    answer_metrics = _score_answer(
        gt.get("expected_final_answer", gt.get("final_answer", actual.get("label", ""))),
        actual.get("final_answer", ""),
    )

    return {
        "case_id": case_id,
        "idx": actual.get("idx", gt.get("idx")),
        "question": gt.get("question", actual.get("question", "")),
        "status": actual.get("status"),
        "plan": _plan_metrics(gt, actual),
        "toolchain": {
            "expected_tool_count": len(expected_sigs),
            "actual_tool_count": len(actual_sigs),
            "exact_sequence_match": expected_sigs == actual_sigs,
            "first_tool_match": bool(expected_sigs and actual_sigs and expected_sigs[0] == actual_sigs[0]),
            "first_divergence_index": first_div,
            "ordered_lcs": lcs,
            "ordered_lcs_recall": lcs / len(expected_sigs) if expected_sigs else 0.0,
            "greedy_ordered_coverage": greedy,
            "greedy_ordered_coverage_rate": greedy / len(expected_sigs) if expected_sigs else 0.0,
            "completion_call_index": completion_idx,
            "stop_after_completion": completion_idx is not None and completion_idx == len(actual_sigs) - 1,
            "overcall_count": overcalls,
            "missing_call_count": missing,
            "redundant_call_count": redundant,
            **prf,
        },
        "contracts": {
            "tool_success_rate": sum(1 for call in actual_calls if call.get("ok")) / len(actual_calls) if actual_calls else 0.0,
            "arg_name_valid_rate": sum(arg_valid) / len(arg_valid) if arg_valid else 0.0,
            "handle_ref_count": len(handle_values),
            "canonical_handle_rate": sum(1 for value in handle_values if _is_canonical_ref(value)) / len(handle_values) if handle_values else 1.0,
            "raw_handle_count": sum(1 for value in handle_values if not _is_canonical_ref(value)),
            "task_complete_count": sum(1 for event in complete_events if (event.get("payload") or {}).get("status") == "TASK_COMPLETE"),
            "contract_failure_count": sum(1 for event in complete_events if (event.get("payload") or {}).get("status") == "CONTRACT_FAILURE"),
            "max_turn_failure": any("max inner turns" in _norm_text((event.get("payload") or {}).get("summary", "")) for event in complete_events),
        },
        "answer": answer_metrics,
        "expected_signatures": expected_sigs,
        "actual_signatures": actual_sigs,
    }


def _mean(rows: list[dict[str, Any]], path: str) -> float:
    values = []
    keys = path.split(".")
    for row in rows:
        value: Any = row
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, (int, float, bool)):
            values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def _rate(rows: list[dict[str, Any]], path: str) -> float:
    return _mean(rows, path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_cases": len(rows),
        "plan": {
            "step_count_match_rate": _rate(rows, "plan.step_count_match"),
            "role_sequence_match_rate": _rate(rows, "plan.role_sequence_match"),
            "role_lcs_ratio_avg": _mean(rows, "plan.role_lcs_ratio"),
            "generic_subtask_avg": _mean(rows, "plan.generic_subtask_count"),
            "vague_need_avg": _mean(rows, "plan.vague_need_count"),
            "plan_valid_shape_rate": _rate(rows, "plan.plan_valid_shape"),
        },
        "toolchain": {
            "exact_sequence_match_rate": _rate(rows, "toolchain.exact_sequence_match"),
            "first_tool_match_rate": _rate(rows, "toolchain.first_tool_match"),
            "ordered_lcs_recall_avg": _mean(rows, "toolchain.ordered_lcs_recall"),
            "greedy_ordered_coverage_rate_avg": _mean(rows, "toolchain.greedy_ordered_coverage_rate"),
            "unordered_precision_avg": _mean(rows, "toolchain.unordered_precision"),
            "unordered_recall_avg": _mean(rows, "toolchain.unordered_recall"),
            "unordered_f1_avg": _mean(rows, "toolchain.unordered_f1"),
            "expected_tool_count_avg": _mean(rows, "toolchain.expected_tool_count"),
            "actual_tool_count_avg": _mean(rows, "toolchain.actual_tool_count"),
            "overcall_count_avg": _mean(rows, "toolchain.overcall_count"),
            "missing_call_count_avg": _mean(rows, "toolchain.missing_call_count"),
            "redundant_call_count_avg": _mean(rows, "toolchain.redundant_call_count"),
            "stop_after_completion_rate": _rate(rows, "toolchain.stop_after_completion"),
        },
        "contracts": {
            "tool_success_rate_avg": _mean(rows, "contracts.tool_success_rate"),
            "arg_name_valid_rate_avg": _mean(rows, "contracts.arg_name_valid_rate"),
            "canonical_handle_rate_avg": _mean(rows, "contracts.canonical_handle_rate"),
            "raw_handle_count_avg": _mean(rows, "contracts.raw_handle_count"),
            "task_complete_count_avg": _mean(rows, "contracts.task_complete_count"),
            "contract_failure_count_avg": _mean(rows, "contracts.contract_failure_count"),
            "max_turn_failure_rate": _rate(rows, "contracts.max_turn_failure"),
        },
        "answer": {
            "answer_acc_avg": _mean(rows, "answer.answer_acc"),
            "correct_rate": _rate(rows, "answer.correct"),
            "exact_match_rate": _rate(rows, "answer.exact_match"),
            "numeric_subset_match_rate": _rate(rows, "answer.numeric_subset_match"),
            "token_f1_avg": _mean(rows, "answer.token_f1"),
        },
    }


def _flatten_summary(summary: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in summary.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_summary(value, new_key))
        elif isinstance(value, (int, float, bool)):
            out[new_key] = float(value)
    return out


def compare_summaries(current: dict[str, Any], baseline_path: Path | None) -> dict[str, Any]:
    if baseline_path is None:
        return {}
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = baseline_payload.get("summary", baseline_payload)
    cur_flat = _flatten_summary(current)
    base_flat = _flatten_summary(baseline)
    deltas = {}
    for key in sorted(cur_flat):
        if key in base_flat:
            deltas[key] = {"baseline": base_flat[key], "current": cur_flat[key], "delta": cur_flat[key] - base_flat[key]}
    return {"baseline_path": str(baseline_path), "deltas": deltas}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model eval conversations against structured GT teacher toolchains.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--output_dir", default="", help="Defaults to eval_dir/gt_toolchain_metrics")
    parser.add_argument("--baseline_summary", default="", help="Optional prior gt_metrics_summary.json for delta reporting.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    teacher_dir = Path(args.teacher_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else eval_dir / "gt_toolchain_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_records = _load_teacher_dir(teacher_dir)
    eval_records = _load_eval_records(eval_dir)
    rows = []
    missing_cases = []
    for case_id, gt in sorted(gt_records.items()):
        actual = eval_records.get(case_id)
        if actual is None:
            missing_cases.append(case_id)
            continue
        rows.append(evaluate_case(case_id, gt, actual))

    summary = summarize(rows)
    report = {
        "eval_dir": str(eval_dir),
        "teacher_dir": str(teacher_dir),
        "n_gt_cases": len(gt_records),
        "n_eval_cases": len(eval_records),
        "missing_case_ids": missing_cases,
        "summary": summary,
        "comparison": compare_summaries(summary, Path(args.baseline_summary).resolve() if args.baseline_summary else None),
    }

    (output_dir / "gt_metrics_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "gt_metrics_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
