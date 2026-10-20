from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from .rl_common import iter_jsonl, stable_hash, write_json, write_jsonl
except ImportError:  # pragma: no cover - direct script execution
    from rl_common import iter_jsonl, stable_hash, write_json, write_jsonl


ARTIFACT_ARG_KEYS = {"gpkg", "image", "geotiff", "pre_image", "post_image"}
HANDLE_RE = re.compile(r"^(?:gpkg|image|img|tif|geotiff)_\d+$", re.IGNORECASE)
RAW_ARTIFACT_RE = re.compile(r"\.(?:gpkg|geojson|tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


@dataclass(frozen=True)
class RewardWeights:
    correct: float = 1.0
    answer_acc: float = 0.7
    task_complete: float = 0.3
    loop_abort: float = -0.5
    invalid_orc_decision: float = -0.2
    redundant_tool_call: float = -0.1
    handle_violation: float = -0.3


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    correct: int
    answer_acc: float
    task_complete: int
    any_loop_abort: int
    n_invalid_orc_decisions: int
    n_redundant_tool_calls: int
    handle_discipline_violation: int


def episode_reward(record: Mapping[str, Any], weights: RewardWeights | None = None) -> RewardBreakdown:
    """Compute the scalar RL reward from one eval conversation record.

    The function accepts the `conversation_*.json` shape emitted by
    `evaluation/eval_live.py` after `recompute_eval_metrics.py` has run. It
    also works on partially populated records by falling back to trace scans.
    """
    weights = weights or RewardWeights()
    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    alignment_raw = record.get("alignment_raw_metrics") if isinstance(record.get("alignment_raw_metrics"), Mapping) else {}
    sequence_metrics = record.get("sequence_metrics") if isinstance(record.get("sequence_metrics"), Mapping) else {}

    # Strict structure-aware correctness is the only correctness route.
    # ``numeric_subset_match`` is retained in old logs for compatibility but
    # must never grant reward to a structurally wrong answer.
    correct = int(bool(metrics.get("correct") or metrics.get("structured_match")))
    answer_acc = _clamp01(_float_metric(metrics, "answer_acc", "answer_accuracy", "ans", "score"))
    task_complete = int(str(record.get("status", "")) == "DONE")
    any_loop_abort = int(_has_loop_abort(record, alignment_raw))
    n_invalid_orc = _invalid_orc_decisions(record, alignment_raw)
    n_redundant = _redundant_successful_tool_calls(record, sequence_metrics)
    handle_violation = int(_has_handle_discipline_violation(record))

    reward = (
        weights.correct * correct
        + weights.answer_acc * answer_acc
        + weights.task_complete * task_complete
        + weights.loop_abort * any_loop_abort
        + weights.invalid_orc_decision * n_invalid_orc
        + weights.redundant_tool_call * n_redundant
        + weights.handle_violation * handle_violation
    )
    return RewardBreakdown(
        reward=float(reward),
        correct=correct,
        answer_acc=float(answer_acc),
        task_complete=task_complete,
        any_loop_abort=any_loop_abort,
        n_invalid_orc_decisions=int(n_invalid_orc),
        n_redundant_tool_calls=int(n_redundant),
        handle_discipline_violation=handle_violation,
    )


def reward_from_record(record: Mapping[str, Any]) -> float:
    return episode_reward(record).reward


def _trace(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("trace")
    return list(trace) if isinstance(trace, list) else []


def _float_metric(metrics: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _has_loop_abort(record: Mapping[str, Any], alignment_raw: Mapping[str, Any]) -> bool:
    if float(alignment_raw.get("loop_failures", 0) or 0) > 0:
        return True
    abort_reason = str(record.get("abort_reason", "") or "").lower()
    if "max inner turns" in abort_reason or "loop" in abort_reason:
        return True
    for event in _trace(record):
        payload = event.get("payload") or {}
        text = " ".join(str(payload.get(key, "")) for key in ("summary", "error", "text"))
        if "hit max inner turns" in text.lower():
            return True
    return False


def _invalid_orc_decisions(record: Mapping[str, Any], alignment_raw: Mapping[str, Any]) -> int:
    raw_count = int(float(alignment_raw.get("orc_invalid_events", 0) or 0))
    if raw_count:
        return raw_count
    count = 0
    for event in _trace(record):
        if event.get("role") == "ORC" and (event.get("payload") or {}).get("error"):
            count += 1
    return count


def _canonical_args(args: Any) -> str:
    return json.dumps(args or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _redundant_successful_tool_calls(record: Mapping[str, Any], sequence_metrics: Mapping[str, Any]) -> int:
    count = 0
    seen_successes: set[tuple[str, str, str]] = set()
    for event in _trace(record):
        if event.get("phase") != "tool_call":
            continue
        payload = event.get("payload") or {}
        tool = str(payload.get("tool", ""))
        key = (str(event.get("step_id", "")), tool, _canonical_args(payload.get("args")))
        if key in seen_successes:
            count += 1
        if payload.get("ok") is True:
            seen_successes.add(key)
    if count:
        return count
    return int(float(sequence_metrics.get("redundant_call_count", 0) or 0))


def _artifact_records(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    shared = record.get("shared_state") if isinstance(record.get("shared_state"), Mapping) else {}
    artifacts = shared.get("artifacts") if isinstance(shared.get("artifacts"), Mapping) else {}
    return {str(key): dict(value) for key, value in artifacts.items() if isinstance(value, Mapping)}


def _has_handle_discipline_violation(record: Mapping[str, Any]) -> bool:
    artifacts = _artifact_records(record)
    path_to_handle = {
        str(artifact.get("path", "")): handle
        for handle, artifact in artifacts.items()
        if isinstance(artifact, Mapping) and artifact.get("path")
    }
    basename_to_handle = {Path(path).name: handle for path, handle in path_to_handle.items()}

    for event in _trace(record):
        if event.get("phase") != "tool_call":
            continue
        args = (event.get("payload") or {}).get("args") or {}
        if not isinstance(args, Mapping):
            continue
        for key in ARTIFACT_ARG_KEYS:
            value = args.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            if HANDLE_RE.match(value):
                continue
            basename = Path(value).name
            if value in path_to_handle or basename in basename_to_handle:
                return True
            if RAW_ARTIFACT_RE.search(value) and artifacts:
                return True
    return False


def _load_records(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_dir():
        rows: list[dict[str, Any]] = []
        for path in sorted(input_path.glob("conversation_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        return rows
    if input_path.suffix == ".jsonl":
        return list(iter_jsonl(input_path))
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return list(payload["results"])
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported reward input: {input_path}")


def _self_test() -> None:
    record = {
        "status": "DONE",
        "metrics": {"correct": True, "answer_acc": 1.0},
        "shared_state": {"artifacts": {"gpkg_1": {"kind": "gpkg", "path": "fix_aoi.gpkg"}}},
        "trace": [
            {"role": "ORC", "phase": "delegate", "payload": {}, "step_id": "s1"},
            {
                "role": "tool",
                "phase": "tool_call",
                "step_id": "s1",
                "payload": {"tool": "DisplayOnMap", "args": {"gpkg": "fix_aoi.gpkg", "layers": ["a"]}, "ok": True},
            },
            {
                "role": "tool",
                "phase": "tool_call",
                "step_id": "s1",
                "payload": {"tool": "DisplayOnMap", "args": {"gpkg": "fix_aoi.gpkg", "layers": ["a"]}, "ok": True},
            },
        ],
    }
    breakdown = episode_reward(record)
    assert breakdown.correct == 1
    assert breakdown.task_complete == 1
    assert breakdown.n_redundant_tool_calls == 1
    assert breakdown.handle_discipline_violation == 1
    assert breakdown.answer_acc == 1.0
    assert abs(breakdown.reward - 1.6) < 1e-9, breakdown
    print(json.dumps({"self_test": "ok", **asdict(breakdown)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute episode-level RL rewards for evaluation records.")
    parser.add_argument("--input", default="", help="Eval dir, conversation JSON, summary JSON, or JSONL file.")
    parser.add_argument("--output_json", default="", help="Optional aggregate reward summary path.")
    parser.add_argument("--output_jsonl", default="", help="Optional per-record reward JSONL path.")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.input:
        raise SystemExit("--input is required unless --self_test is used")

    records = _load_records(Path(args.input).resolve())
    rows = []
    for idx, record in enumerate(records):
        breakdown = episode_reward(record)
        rows.append(
            {
                "row_id": idx,
                "dataset_index": record.get("dataset_index"),
                "idx": record.get("idx"),
                "question": record.get("question", ""),
                "reward_key": stable_hash({"dataset_index": record.get("dataset_index"), "idx": record.get("idx")}),
                **asdict(breakdown),
            }
        )
    rewards = [float(row["reward"]) for row in rows]
    summary = {
        "n_records": len(rows),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "correct_rate": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0,
        "answer_acc_mean": sum(row["answer_acc"] for row in rows) / len(rows) if rows else 0.0,
        "task_complete_rate": sum(row["task_complete"] for row in rows) / len(rows) if rows else 0.0,
        "loop_abort_rate": sum(row["any_loop_abort"] for row in rows) / len(rows) if rows else 0.0,
        "handle_violation_rate": sum(row["handle_discipline_violation"] for row in rows) / len(rows) if rows else 0.0,
    }
    if args.output_jsonl:
        write_jsonl(Path(args.output_jsonl).resolve(), rows)
    if args.output_json:
        write_json(Path(args.output_json).resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
