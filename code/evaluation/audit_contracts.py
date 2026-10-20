#!/usr/bin/env python3
"""Audit code/data/tool-contract synchronization for the public release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from evaluation.structured_answer import (
    DEFAULT_ABS_TOL,
    DEFAULT_REL_TOL,
    SCORING_POLICY_VERSION,
    score_answer,
)
from multi_agent.final_dataset_profile import (
    FINAL_DATASET_AGENT_TOOLS,
    FINAL_DATASET_TOOLS,
)
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import (
    TOOL_SPECS,
    agent_tool_spec,
    api_map,
    validate_agent_args,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
ANSWER_REF = "ANSWER_JSON:answer_1"


def _answer_sha256(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def logical_type(record: dict[str, Any]) -> str:
    parts = str(record["id"]).split("/")
    if parts[0] == "multi_hazard":
        return f"multi_hazard/{parts[1]}/{record['question_type']}"
    return f"{parts[0]}/{record['question_type']}"


def _assistant_payloads(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError as exc:
            errors.append(f"assistant message is not JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append("assistant message is not a JSON object")
            continue
        payloads.append(payload)
    return payloads, errors


def _audit_training_rows(
    path: Path,
    canonical: dict[str, dict[str, Any]],
    *,
    expected_split: str,
    expected_role: str | None = None,
    require_terminal_answer: bool = False,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows = 0
    assistant_json_errors = 0
    unknown_ids = 0
    split_errors = 0
    role_errors = 0
    tool_contract_errors = 0
    ownership_errors = 0
    terminal_answer_errors = 0
    for row in read_jsonl(path):
        rows += 1
        record_id = str(row.get("dataset_id") or row.get("id") or "")
        source = canonical.get(record_id)
        if source is None:
            unknown_ids += 1
            continue
        row_split = str(row.get("final_split") or source.get("split") or "")
        if row_split != expected_split or str(source.get("split")) != expected_split:
            split_errors += 1
        if expected_role and str(row.get("role") or "").upper() != expected_role:
            role_errors += 1
        payloads, json_errors = _assistant_payloads(row)
        assistant_json_errors += len(json_errors)
        for payload in payloads:
            action = payload.get("action", payload)
            if not isinstance(action, dict):
                continue
            tool = str(action.get("tool") or "")
            if tool in {"Terminate", "FinalAnswer"} and action.get("answer_ref") != ANSWER_REF:
                terminal_answer_errors += 1
            if tool not in TOOL_SPECS:
                continue
            if validate_agent_args(tool, action.get("args") or {}):
                tool_contract_errors += 1
            if expected_role and FINAL_DATASET_AGENT_TOOLS.get(expected_role) is not None:
                if tool not in FINAL_DATASET_AGENT_TOOLS[expected_role]:
                    ownership_errors += 1
        if require_terminal_answer:
            last_action = (payloads[-1].get("action") or {}) if payloads else {}
            if (
                last_action.get("tool") != "FinalAnswer"
                or last_action.get("answer_ref") != ANSWER_REF
                or row.get("resolved_answer_sha256")
                != (expected_hashes or {}).get(
                    record_id, _answer_sha256(source["expected_final_answer"])
                )
            ):
                terminal_answer_errors += 1
    return {
        "path": str(path),
        "rows": rows,
        "assistant_json_errors": assistant_json_errors,
        "unknown_ids": unknown_ids,
        "split_errors": split_errors,
        "role_errors": role_errors,
        "tool_contract_errors": tool_contract_errors,
        "ownership_errors": ownership_errors,
        "terminal_answer_errors": terminal_answer_errors,
        "ok": not any(
            (
                assistant_json_errors,
                unknown_ids,
                split_errors,
                role_errors,
                tool_contract_errors,
                ownership_errors,
                terminal_answer_errors,
            )
        ),
    }


def audit(dataset_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    registry_dir = CODE_ROOT / "tool_registry"
    registry = json.loads((registry_dir / "agent_tool_registry.json").read_text(encoding="utf-8"))
    api = json.loads((registry_dir / "tool_api_map.json").read_text(encoding="utf-8"))
    ownership = json.loads((registry_dir / "tool_ownership.json").read_text(encoding="utf-8"))
    templates = json.loads((registry_dir / "tool_templates_43.json").read_text(encoding="utf-8"))["templates"]
    tool_sets = {
        "worker_specs": set(TOOL_SPECS),
        "profile": set(FINAL_DATASET_TOOLS),
        "registry": {row["name"] for row in registry["tools"]},
        "api_map": {row["tool"] for row in api["tools"]},
        "ownership": set(ownership["tool_to_agent"]),
    }
    if len({frozenset(value) for value in tool_sets.values()}) != 1:
        failures.append(f"tool-name sets differ: {tool_sets}")
    if ownership["agent_tools"] != FINAL_DATASET_AGENT_TOOLS:
        failures.append("tool ownership differs from runtime profile")
    if api["tools"] != api_map():
        failures.append("tool_api_map.json differs from executable TOOL_SPECS")

    registry_by_name = {row["name"]: row for row in registry["tools"]}
    for name in TOOL_SPECS:
        visible = agent_tool_spec(name)
        if set(registry_by_name[name]["args"]) != set(visible["properties"]):
            failures.append(f"registry argument drift: {name}")
        worker_path = CODE_ROOT / "tool_server/tool_workers/online_workers" / f"{name}_worker.py"
        tree = ast.parse(worker_path.read_text(encoding="utf-8"), filename=str(worker_path))
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        if f"process_{name}" not in functions:
            failures.append(f"missing process function: {name}")

    template_errors = 0
    for template in templates:
        for call in template["eo_style_chain"]:
            if call["tool"] == "TaskComplete":
                continue
            errors = validate_agent_args(call["tool"], call.get("args") or {})
            if errors:
                template_errors += 1
                failures.append(f"template {template['question_type']} {call['tool']}: {errors}")

    core_path = dataset_root / "data/core/geodisaster_2921_source_with_splits.jsonl"
    scoring_policy_path = dataset_root / "data/core/scoring_policy_43.json"
    scoring_policy = json.loads(scoring_policy_path.read_text(encoding="utf-8"))
    declared_numeric = scoring_policy.get("numeric") or {}
    scoring_policy_matches = (
        scoring_policy.get("schema_version") == SCORING_POLICY_VERSION
        and declared_numeric.get("absolute_tolerance") == DEFAULT_ABS_TOL
        and declared_numeric.get("relative_tolerance") == DEFAULT_REL_TOL
    )
    if not scoring_policy_matches:
        failures.append(
            "dataset scoring policy differs from executable evaluator: "
            f"declared={scoring_policy}, executable={{'schema_version': "
            f"{SCORING_POLICY_VERSION!r}, 'absolute_tolerance': {DEFAULT_ABS_TOL}, "
            f"'relative_tolerance': {DEFAULT_REL_TOL}}}"
        )
    core = read_jsonl(core_path)
    schemas = json.loads(
        (dataset_root / "data/core/answer_schemas_43.json").read_text(encoding="utf-8")
    )["schemas"]
    chain_errors = 0
    answer_schema_errors = 0
    for record in core:
        for call in record["gt_chain"]:
            if call["tool"] == "TaskComplete":
                continue
            errors = validate_agent_args(call["tool"], call.get("args") or {})
            if errors:
                chain_errors += 1
                if chain_errors <= 20:
                    failures.append(f"core chain {record['id']} {call['tool']}: {errors}")
        try:
            Draft202012Validator(schemas[logical_type(record)]).validate(
                record["expected_final_answer"]
            )
        except Exception as exc:
            answer_schema_errors += 1
            if answer_schema_errors <= 20:
                failures.append(f"answer schema {record['id']}: {exc}")

    canonical = {record["id"]: record for record in core}
    derived: dict[str, Any] = {}
    multi_resolved_hashes: dict[str, str] = {}
    multi_sources: dict[str, str] = {}
    for family, train_name, test_name in (
        ("multi_agent", "train_conversations.jsonl", "test_conversations.jsonl"),
        ("single_agent", "train_conversations.jsonl", "test_reference_conversations.jsonl"),
    ):
        rows = read_jsonl(dataset_root / f"data/{family}/{train_name}") + read_jsonl(
            dataset_root / f"data/{family}/{test_name}"
        )
        ids = [str(row.get("dataset_id") or row.get("id")) for row in rows]
        answer_mismatch = 0
        chain_mismatch = 0
        source_mismatch = 0
        for row, record_id in zip(rows, ids):
            source = canonical[record_id]
            if family == "multi_agent":
                prediction = row.get("resolved_final_answer")
                chain = row.get("gt_chain")
                ref_ok = row.get("final_answer_ref") == ANSWER_REF
                hash_ok = row.get("resolved_answer_sha256") == _answer_sha256(prediction)
                multi_resolved_hashes[record_id] = str(row.get("resolved_answer_sha256") or "")
                multi_sources[record_id] = str(row.get("source") or "")
            else:
                action = json.loads(row["messages"][-1]["content"])["action"]
                prediction = source["expected_final_answer"]
                chain = None
                ref_ok = action.get("answer_ref") == ANSWER_REF
                hash_ok = row.get("resolved_answer_sha256") == multi_resolved_hashes.get(record_id)
                if str(row.get("source") or "") != multi_sources.get(record_id):
                    source_mismatch += 1
            if (
                not ref_ok
                or not hash_ok
                or not score_answer(source["expected_final_answer"], prediction)["correct"]
            ):
                answer_mismatch += 1
            if chain is not None and chain != source["gt_chain"]:
                chain_mismatch += 1
        derived[family] = {
            "records": len(rows),
            "unique_ids": len(set(ids)),
            "id_set_matches": set(ids) == set(canonical),
            "terminal_answer_mismatch": answer_mismatch,
            "canonical_chain_mismatch": chain_mismatch,
            "source_family_mismatch": source_mismatch,
        }
        if derived[family]["records"] != len(core) or not derived[family]["id_set_matches"]:
            failures.append(f"{family} record/id synchronization failed")
        if answer_mismatch:
            failures.append(f"{family} has {answer_mismatch} terminal answer mismatches")
        if chain_mismatch:
            failures.append(f"{family} has {chain_mismatch} canonical chain mismatches")
        if source_mismatch:
            failures.append(f"{family} has {source_mismatch} source-family mismatches")

    training_exports: dict[str, Any] = {}
    for split, folder, suffix in (
        ("train", "sft_train", "train"),
        ("test", "sft_test", "eval"),
    ):
        for role in ("ORC", "GA", "PA", "VRA"):
            key = f"multi_agent/{folder}/{role.lower()}_{suffix}.jsonl"
            result = _audit_training_rows(
                dataset_root / "data" / key,
                canonical,
                expected_split=split,
                expected_role=role,
            )
            training_exports[key] = result
            if not result["ok"]:
                failures.append(f"training export contract failed: {key}: {result}")

    for split, name in (
        ("train", "train_conversations.jsonl"),
        ("test", "test_reference_conversations.jsonl"),
    ):
        key = f"single_agent/{name}"
        result = _audit_training_rows(
            dataset_root / "data" / key,
            canonical,
            expected_split=split,
            require_terminal_answer=True,
            expected_hashes=multi_resolved_hashes,
        )
        training_exports[key] = result
        if not result["ok"]:
            failures.append(f"training export contract failed: {key}: {result}")

    for split, name in (("train", "train_turns.jsonl"), ("test", "test_turns.jsonl")):
        key = f"single_agent/{name}"
        result = _audit_training_rows(
            dataset_root / "data" / key,
            canonical,
            expected_split=split,
        )
        training_exports[key] = result
        if not result["ok"]:
            failures.append(f"training export contract failed: {key}: {result}")

    return {
        "ok": not failures,
        "tools": len(TOOL_SPECS),
        "templates": len(templates),
        "template_count_sum": sum(int(row["count_in_final_dataset"]) for row in templates),
        "core_records": len(core),
        "logical_question_types": len({logical_type(record) for record in core}),
        "core_chain_contract_errors": chain_errors,
        "expected_answer_schema_errors": answer_schema_errors,
        "scoring_policy": {
            "matches_executable": scoring_policy_matches,
            "schema_version": SCORING_POLICY_VERSION,
            "absolute_tolerance": DEFAULT_ABS_TOL,
            "relative_tolerance": DEFAULT_REL_TOL,
        },
        "derived": derived,
        "training_exports": training_exports,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset_root.resolve())
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
