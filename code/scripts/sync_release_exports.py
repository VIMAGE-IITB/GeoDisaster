#!/usr/bin/env python3
"""Synchronize derived agent exports with the canonical 2,921-record JSONL."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("MAS_AGENT_PROFILE", "geodisaster")
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from scripts.build_final_dataset_multiagentic import (  # noqa: E402
    build_interleaved_gt_flow,
    build_orchestration_flow,
    derive_answer_from_executions,
    summarize,
    tool_output_from_trace,
    write_ownership_docs,
    write_role_sft,
)
from multi_agent.answer_registry import ANSWER_REF  # noqa: E402
from evaluation.structured_answer import score_answer  # noqa: E402
from single_agent.prompts import SINGLE_AGENT_SYSTEM  # noqa: E402
from training.build_single_agent_sft import convert_record, explode_case_to_sft_rows  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_private_runtime_metadata(value: Any) -> Any:
    """Drop non-portable worker bookkeeping from public derived exports."""

    if isinstance(value, dict):
        value.pop("sample_dir", None)
        for key, child in list(value.items()):
            if key.endswith("_payload_keys") and isinstance(child, list):
                child[:] = [item for item in child if item != "sample_dir"]
            value[key] = remove_private_runtime_metadata(child)
        return value
    elif isinstance(value, list):
        value[:] = [remove_private_runtime_metadata(child) for child in value]
        return value
    elif isinstance(value, str):
        # Older teacher summaries listed raw worker keys even after the value
        # itself was removed. Keep the analytical field list, not the private
        # runtime-directory key.
        if "fields:" in value:
            return value.replace("fields: sample_dir, ", "fields: ").replace(
                ", sample_dir", ""
            )
        return value
    return value


def synchronize_calls(record: dict[str, Any], canonical_chain: list[dict[str, Any]]) -> None:
    expected = [call for call in canonical_chain if call.get("tool") != "TaskComplete"]
    actual_turns = []
    for execution in record.get("executions") or []:
        for turn in execution.get("turns") or []:
            action = turn.get("action") or {}
            if action.get("tool") and action.get("tool") != "TaskComplete":
                actual_turns.append(turn)
    if [turn["action"]["tool"] for turn in actual_turns] != [call["tool"] for call in expected]:
        raise ValueError(f"tool sequence drift for {record.get('dataset_id')}")
    for turn, call in zip(actual_turns, expected):
        turn["action"]["args"] = dict(call.get("args") or {})


def refresh_executions_from_replay(
    record: dict[str, Any], replay: dict[str, Any]
) -> None:
    """Replace stored tool observations with outputs from one successful live replay.

    The agent actions and their order remain unchanged. Replay rows are accepted only
    when their tool sequence matches the teacher execution exactly, which prevents a
    result from being attached to the wrong action.
    """

    if replay.get("status") != "passed":
        raise ValueError(
            f"cannot refresh {record.get('dataset_id')} from failed replay"
        )
    turns = [
        turn
        for execution in record.get("executions") or []
        for turn in execution.get("turns") or []
        if (turn.get("action") or {}).get("tool") != "TaskComplete"
    ]
    trace = [
        row
        for row in replay.get("trace") or []
        if row.get("tool") != "TaskComplete"
    ]
    expected_tools = [str((turn.get("action") or {}).get("tool") or "") for turn in turns]
    replay_tools = [str(row.get("tool") or "") for row in trace]
    if expected_tools != replay_tools:
        raise ValueError(
            f"replay tool sequence drift for {record.get('dataset_id')}: "
            f"teacher={expected_tools}, replay={replay_tools}"
        )
    for turn, trace_row in zip(turns, trace):
        turn["tool_output"] = copy.deepcopy(tool_output_from_trace(trace_row))


def synchronize_multi_record(record: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    remove_private_runtime_metadata(record)
    record["expected_final_answer"] = canonical["expected_final_answer"]
    record["gt_chain"] = canonical["gt_chain"]
    record["initial_artifacts"] = canonical["initial_artifacts"]
    record["available_initial_handles"] = canonical["available_initial_handles"]
    synchronize_calls(record, canonical["gt_chain"])
    resolved, resolved_sha256, provenance = derive_answer_from_executions(
        record, record.get("executions") or []
    )
    if not score_answer(canonical["expected_final_answer"], resolved)["correct"]:
        raise ValueError(
            f"worker-derived answer differs from canonical answer for {record.get('dataset_id')}"
        )
    record["schema_version"] = "geodisaster_multiagentic_sft_v3"
    record["final_answer_ref"] = ANSWER_REF
    record["resolved_final_answer"] = resolved
    record["resolved_answer_sha256"] = resolved_sha256
    record["resolved_answer_provenance"] = provenance
    record.pop("final_answer", None)
    record.pop("answer_package", None)
    record["answer_support"] = {
        "score": 1.0,
        "basis": "schema_valid_runtime_projection",
        "resolved_answer_sha256": resolved_sha256,
        "diagnostic_only": True,
        "note": (
            "The successful worker payload projected to a schema-valid terminal answer. "
            "Strict structured terminal-answer scoring determines correctness."
        ),
    }
    record["orchestration_flow"] = build_orchestration_flow(record)
    record["interleaved_gt_flow"] = build_interleaved_gt_flow(record)
    record["multiagentic_gt_chain"] = record["interleaved_gt_flow"]
    return record


def rebuild_single_agent(dataset_root: Path, train: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    out = dataset_root / "data" / "single_agent"
    train_cases = [convert_record(row, source="geodisaster") for row in train]
    test_cases = [convert_record(row, source="geodisaster_test") for row in test]
    train_turns = [item for case in train_cases for item in explode_case_to_sft_rows(case)]
    test_turns = [item for case in test_cases for item in explode_case_to_sft_rows(case)]
    write_jsonl(out / "train_conversations.jsonl", train_cases)
    write_jsonl(out / "test_reference_conversations.jsonl", test_cases)
    write_jsonl(out / "train_turns.jsonl", train_turns)
    write_jsonl(out / "test_turns.jsonl", test_turns)
    (out / "single_agent_system_prompt.md").write_text(
        "# Single-Agent System Prompt\n\n```text\n" + SINGLE_AGENT_SYSTEM + "\n```\n",
        encoding="utf-8",
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "geodisaster_single_agent_release_v2"
    manifest["terminal_answer_protocol"] = {
        "action_field": "answer_ref",
        "reference": ANSWER_REF,
        "resolution": "runtime answer registry populated from schema-valid worker evidence",
    }
    manifest["counts"] = {
        "train_cases": len(train_cases),
        "train_sft_turns": len(train_turns),
        "test_cases": len(test_cases),
        "test_sft_turns": len(test_turns),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--replay-results",
        type=Path,
        help="Live replay JSONL used to refresh strictly mismatching derived observations.",
    )
    parser.add_argument(
        "--refresh-strict-mismatches",
        action="store_true",
        help="Refresh only records that fail the current structured-answer policy.",
    )
    args = parser.parse_args()
    if args.refresh_strict_mismatches and args.replay_results is None:
        parser.error("--refresh-strict-mismatches requires --replay-results")
    root = args.dataset_root.resolve()
    core_path = root / "data/core/geodisaster_2921_source_with_splits.jsonl"
    canonical_rows = read_jsonl(core_path)
    canonical = {row["id"]: row for row in canonical_rows}
    replay_by_id: dict[str, dict[str, Any]] = {}
    if args.replay_results is not None:
        replay_by_id = {
            str(row["dataset_id"]): row
            for row in read_jsonl(args.replay_results.resolve())
        }
    refreshed_ids: list[str] = []
    multi_dir = root / "data/multi_agent"
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split, filename in (("train", "train_conversations.jsonl"), ("test", "test_conversations.jsonl")):
        path = multi_dir / filename
        records = read_jsonl(path)
        for record in records:
            record_id = str(record.get("dataset_id") or record["id"])
            canonical_record = canonical[record_id]
            if args.refresh_strict_mismatches and not score_answer(
                canonical_record["expected_final_answer"],
                record.get("resolved_final_answer"),
            )["correct"]:
                replay = replay_by_id.get(record_id)
                if replay is None:
                    raise ValueError(f"missing replay result for {record_id}")
                refresh_executions_from_replay(record, replay)
                refreshed_ids.append(record_id)
            synchronize_multi_record(record, canonical_record)
        write_jsonl(path, records)
        split_records[split] = records
        write_role_sft(
            records,
            multi_dir / f"sft_{split}",
            "train" if split == "train" else "eval",
        )

    write_ownership_docs(multi_dir / "tool_ownership_and_prompt_context.md")
    manifest_path = multi_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "geodisaster_multi_agent_release_v3"
    manifest["terminal_answer_protocol"] = {
        "action_field": "answer_ref",
        "reference": ANSWER_REF,
        "resolution": "runtime answer registry populated from schema-valid worker evidence",
    }
    manifest["canonical_sha256"] = sha256(core_path)
    for split, records in split_records.items():
        manifest["splits"][split]["records"] = len(records)
        manifest["splits"][split]["summary"] = summarize(records)
        role_manifest = json.loads(
            (multi_dir / f"sft_{split}" / "manifest.json").read_text(encoding="utf-8")
        )
        for role, entry in role_manifest["roles"].items():
            entry["path"] = f"data/multi_agent/sft_{split}/{entry['path']}"
        manifest["splits"][split]["sft"] = role_manifest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rebuild_single_agent(root, split_records["train"], split_records["test"])
    print(
        json.dumps(
            {
                "canonical_records": len(canonical_rows),
                "train": len(split_records["train"]),
                "test": len(split_records["test"]),
                "refreshed_records": len(refreshed_ids),
                "refreshed_record_ids": refreshed_ids,
                "terminal_answer_contract": "reference_only_to_runtime_resolved_structured_json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
