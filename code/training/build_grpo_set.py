#!/usr/bin/env python3
"""Build a diverse GeoDisaster GRPO prompt set.

The selector is deterministic and intentionally simple:
- every source/question-type bucket gets exactly the requested coverage first;
- within each bucket, prefer longer, multi-agent, multi-tool cases;
- output preserves a stable shuffled order so GRPO does not see one domain in
  a single contiguous block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.environ.get("GEODISASTER_DATASET_ROOT", str(ROOT.parent / "geodisaster-agentic-2921"))
).expanduser()
DEFAULT_INPUT = DATASET_ROOT / "data" / "multi_agent" / "train_conversations.jsonl"
DEFAULT_OUTPUT = ROOT / "runs" / "grpo_data" / "train.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def tool_call_count(row: dict[str, Any]) -> int:
    count = 0
    for exe in row.get("executions") or []:
        for turn in exe.get("turns") or []:
            action = turn.get("action") or {}
            if action.get("tool") and action.get("tool") != "TaskComplete":
                count += 1
    return count


def row_id(row: dict[str, Any]) -> str:
    """Stable public/internal row identifier."""

    return str(row.get("dataset_id") or row.get("id") or row.get("sample_id") or "")


def row_source(row: dict[str, Any]) -> str:
    """Source bucket for both internal and public GeoDisaster schemas.

    Public multi-agent records use ids such as
    ``multi_hazard/wildfire/000002/A``; keeping the hazard subfamily preserves
    the 43 source/question-type buckets used by the GRPO selector.
    """

    rid = row_id(row)
    parts = rid.split("/")
    if len(parts) >= 2 and parts[0] == "multi_hazard":
        return "/".join(parts[:2])
    if row.get("source"):
        return str(row["source"])
    return parts[0] if parts and parts[0] else "unknown"


def bucket_key(row: dict[str, Any]) -> str:
    return f"{row_source(row)}::{row.get('question_type')}"


def agent_count(row: dict[str, Any]) -> int:
    return len({(exe or {}).get("agent") for exe in row.get("executions") or [] if (exe or {}).get("agent")})


def trajectory_signature(row: dict[str, Any]) -> str:
    """Identify equivalent learning problems without relying on record IDs.

    The canonical split builder marks proven complete-trajectory duplicates
    with inverse-multiplicity ``training_weight``. Unit-weight records remain
    distinct even if their question wording and answer happen to match.
    """

    if float(row.get("training_weight", 1.0)) == 1.0:
        return f"case:{row_id(row)}"

    payload = {
        "bucket": bucket_key(row),
        "question": row.get("raw_question") or row.get("question"),
        "plan": row.get("plan"),
        "gt_chain": row.get("gt_chain"),
        "answer": row.get("resolved_final_answer") or row.get("expected_final_answer"),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def complexity_score(row: dict[str, Any]) -> tuple[int, int, int, str]:
    """Higher sort tuple means stronger GRPO candidate."""

    handles = row.get("available_initial_handles") or []
    return (
        tool_call_count(row),
        agent_count(row),
        len(handles),
        row_id(row),
    )


def select_rows(
    rows: list[dict[str, Any]],
    *,
    target_size: int,
    min_per_bucket: int,
    max_per_bucket: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[bucket_key(row)].append(row)

    for key, bucket_rows in list(buckets.items()):
        unique_rows: dict[str, dict[str, Any]] = {}
        for row in sorted(bucket_rows, key=row_id):
            unique_rows.setdefault(trajectory_signature(row), row)
        buckets[key] = list(unique_rows.values())
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        bucket_rows.sort(key=complexity_score, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_signatures: set[str] = set()

    for key in sorted(buckets):
        for row in buckets[key][: min(min_per_bucket, len(buckets[key]))]:
            dataset_id = row_id(row)
            signature = trajectory_signature(row)
            if dataset_id and dataset_id not in selected_ids and signature not in selected_signatures:
                selected.append(row)
                selected_ids.add(dataset_id)
                selected_signatures.add(signature)

    bucket_positions: dict[str, int] = {
        key: min(min_per_bucket, len(bucket_rows))
        for key, bucket_rows in buckets.items()
    }
    selected_per_bucket: dict[str, int] = defaultdict(int)
    for row in selected:
        selected_per_bucket[bucket_key(row)] += 1

    # Balanced hard-case fill: one case per bucket per round. This avoids a
    # single abundant bucket dominating the RL objective while still taking the
    # most complex remaining rows inside each bucket first.
    made_progress = True
    while len(selected) < target_size and made_progress:
        made_progress = False
        for key in sorted(buckets):
            if len(selected) >= target_size:
                break
            if selected_per_bucket[key] >= max_per_bucket:
                continue
            pos = bucket_positions[key]
            bucket_rows = buckets[key]
            while pos < len(bucket_rows) and (
                row_id(bucket_rows[pos]) in selected_ids
                or trajectory_signature(bucket_rows[pos]) in selected_signatures
            ):
                pos += 1
            bucket_positions[key] = pos
            if pos >= len(bucket_rows):
                continue
            row = bucket_rows[pos]
            selected.append(row)
            selected_ids.add(row_id(row))
            selected_signatures.add(trajectory_signature(row))
            selected_per_bucket[key] += 1
            bucket_positions[key] += 1
            made_progress = True

    # If a strict cap still leaves us short because many buckets are tiny, relax
    # the cap only after balanced coverage is exhausted.
    if len(selected) < target_size:
        remaining = [
            row
            for row in rows
            if row_id(row) not in selected_ids
            and trajectory_signature(row) not in selected_signatures
        ]
        rng.shuffle(remaining)
        remaining.sort(key=complexity_score, reverse=True)
        for row in remaining:
            if len(selected) >= target_size:
                break
            if trajectory_signature(row) in selected_signatures:
                continue
            selected.append(row)
            selected_ids.add(row_id(row))
            selected_signatures.add(trajectory_signature(row))

    rng.shuffle(selected)
    return selected[:target_size]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    by_agent: dict[str, int] = defaultdict(int)
    by_tool: dict[str, int] = defaultdict(int)
    tool_lengths = []
    agent_lengths = []
    for row in rows:
        by_bucket[bucket_key(row)] += 1
        by_source[row_source(row)] += 1
        tool_lengths.append(tool_call_count(row))
        agent_lengths.append(agent_count(row))
        for exe in row.get("executions") or []:
            agent = str(exe.get("agent") or "")
            if agent:
                by_agent[agent] += 1
            for turn in exe.get("turns") or []:
                action = turn.get("action") or {}
                tool = action.get("tool")
                if tool and tool != "TaskComplete":
                    by_tool[str(tool)] += 1
    return {
        "records": len(rows),
        "unique_trajectory_signatures": len({trajectory_signature(row) for row in rows}),
        "source_counts": dict(sorted(by_source.items())),
        "agent_step_counts": dict(sorted(by_agent.items())),
        "tool_counts": dict(sorted(by_tool.items())),
        "question_type_buckets": len(by_bucket),
        "bucket_counts": dict(sorted(by_bucket.items())),
        "bucket_count_min": min(by_bucket.values()) if by_bucket else 0,
        "bucket_count_max": max(by_bucket.values()) if by_bucket else 0,
        "mean_tool_calls": round(sum(tool_lengths) / len(tool_lengths), 4) if tool_lengths else 0.0,
        "min_tool_calls": min(tool_lengths) if tool_lengths else 0,
        "max_tool_calls": max(tool_lengths) if tool_lengths else 0,
        "mean_agents": round(sum(agent_lengths) / len(agent_lengths), 4) if agent_lengths else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target_size", type=int, default=258)
    parser.add_argument("--min_per_bucket", type=int, default=6)
    parser.add_argument("--max_per_bucket", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl.resolve())
    selected = select_rows(
        rows,
        target_size=args.target_size,
        min_per_bucket=args.min_per_bucket,
        max_per_bucket=args.max_per_bucket,
        seed=args.seed,
    )
    write_jsonl(args.output_jsonl.resolve(), selected)
    summary = summarize(selected)
    summary.update(
        {
            "input_records": len(rows),
            "input_unique_trajectory_signatures": len({trajectory_signature(row) for row in rows}),
            "input_jsonl": str(args.input_jsonl.resolve()),
            "output_jsonl": str(args.output_jsonl.resolve()),
            "target_size": args.target_size,
            "min_per_bucket": args.min_per_bucket,
            "max_per_bucket": args.max_per_bucket,
            "seed": args.seed,
        }
    )
    write_json(args.output_jsonl.resolve().with_suffix(".summary.json"), summary)
    if summary["question_type_buckets"] and summary["bucket_count_min"] != summary["bucket_count_max"]:
        raise SystemExit(
            "GRPO selection is not evenly balanced: "
            f"min={summary['bucket_count_min']} max={summary['bucket_count_max']}"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
