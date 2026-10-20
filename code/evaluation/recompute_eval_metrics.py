from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_live import (
    _aggregate_failure_buckets,
    _aggregate_sequence_metrics,
    _analyze_record,
    _finalize_agent_metrics,
    _turn_agent_metrics_template,
)
from evaluation.eval_multi_agent import _summarize_online, logical_question_type


def _load_records(output_dir: Path) -> list[tuple[int, Path, dict[str, Any]]]:
    rows: list[tuple[int, Path, dict[str, Any]]] = []
    for path in sorted(output_dir.glob("conversation_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        sample_no = int(path.stem.split("_")[-1])
        rows.append((sample_no, path, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _resolve_scripted_tool_data_path(output_dir: Path, explicit_path: str) -> str:
    if explicit_path.strip():
        return str(Path(explicit_path).resolve())
    env_path = os.environ.get("MAS_SCRIPTED_TOOL_DATA_PATH", "").strip()
    if env_path:
        return str(Path(env_path).resolve())

    config_path = output_dir.parent / "experiment_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
        scripted_db = str(config.get("scripted_eval_db", "")).strip()
        if scripted_db:
            return str(Path(scripted_db).resolve())
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill alignment, sequence, and failure metrics for an eval output dir.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--scripted_tool_data_path",
        default="",
        help="Scripted DB used for expected-tool inline metrics. Defaults to env or ../experiment_config.json.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    scripted_tool_data_path = _resolve_scripted_tool_data_path(output_dir, args.scripted_tool_data_path)
    if scripted_tool_data_path:
        os.environ["MAS_SCRIPTED_TOOL_DATA_PATH"] = scripted_tool_data_path

    records = _load_records(output_dir)
    if not records:
        raise SystemExit(f"No conversation_*.json files found in {output_dir}")

    agent_conversations_path = output_dir / "agent_conversations.jsonl"
    tool_calls_path = output_dir / "tool_calls.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    failure_tags_path = output_dir / "alignment_failure_tags.jsonl"
    updated_records: list[dict[str, Any]] = []
    agent_conversations: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    failure_tags: list[dict[str, Any]] = []
    for _, path, record in records:
        record["logical_question_type"] = logical_question_type(record)
        analysis = _analyze_record(record)
        record["alignment_metrics"] = analysis["overall_metrics"]
        record["alignment_raw_metrics"] = analysis["overall_raw_metrics"]
        record["agent_metrics"] = analysis["agent_metrics"]
        record["agent_raw_metrics"] = analysis["agent_raw_metrics"]
        record["sequence_metrics"] = analysis["sequence_metrics"]
        record["failure_buckets"] = analysis["failure_buckets"]
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        updated_records.append(record)

        agent_conversations.extend(analysis["datasets"]["agent_conversations"])
        tool_calls.extend(analysis["datasets"]["tool_calls"])
        trajectories.extend(analysis["datasets"]["trajectories"])
        failure_tags.extend(analysis["alignment_failure_tags"])

    _write_jsonl(agent_conversations_path, agent_conversations)
    _write_jsonl(tool_calls_path, tool_calls)
    _write_jsonl(trajectories_path, trajectories)
    _write_jsonl(failure_tags_path, failure_tags)

    aggregate_agent_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in updated_records:
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
    for row in updated_records:
        metrics = row.get("alignment_raw_metrics") or {}
        for key in aggregate_overall_raw:
            aggregate_overall_raw[key] += float(metrics.get(key, 0.0))
    alignment_summary = _finalize_agent_metrics(aggregate_overall_raw)
    sequence_summary = _aggregate_sequence_metrics(updated_records)
    failure_bucket_summary = _aggregate_failure_buckets(updated_records)

    old_summary_path = output_dir / "summary.json"
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8")) if old_summary_path.exists() else {}
    summary = {
        **old_summary,
        **_summarize_online(updated_records),
        "alignment_metrics": alignment_summary,
        "sequence_metrics": sequence_summary,
        "failure_buckets": failure_bucket_summary,
        "per_agent_metrics": per_agent_summary,
        "scripted_tool_data_path": scripted_tool_data_path,
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
                "abort_reason": row.get("abort_reason"),
                "final_answer": row.get("final_answer"),
                "metrics": row.get("metrics", {}),
                "alignment_metrics": row.get("alignment_metrics", {}),
                "sequence_metrics": row.get("sequence_metrics", {}),
                "failure_buckets": row.get("failure_buckets", {}),
                "agent_metrics": row.get("agent_metrics", {}),
                "wall_time_s": row.get("wall_time_s"),
            }
            for row in updated_records
        ],
    }
    old_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Updated metrics for {len(updated_records)} conversations in {output_dir}")


if __name__ == "__main__":
    main()
