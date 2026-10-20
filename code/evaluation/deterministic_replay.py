"""Execute prescribed GeoDisaster chains through the agent-facing dispatcher."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.structured_answer import score_answer
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import validate_agent_args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def logical_question_type(record: dict[str, Any]) -> str:
    record_id = str(record.get("dataset_id") or record.get("id") or "")
    parts = record_id.split("/")
    qtype = str(record.get("question_type") or (parts[-1] if parts else "unknown"))
    if parts and parts[0] == "multi_hazard" and len(parts) >= 2:
        return f"multi_hazard/{parts[1]}/{qtype}"
    family = parts[0] if parts and parts[0] else str(record.get("source") or "unknown")
    return f"{family}/{qtype}"


def _category(text: str, *, contract_error: bool = False) -> str:
    lower = str(text or "").lower()
    if contract_error or "argument contract error" in lower:
        return "agent_invalid_args"
    if "transport_error" in lower or "registered worker" in lower or "connection" in lower:
        return "transport_error"
    if "unknown handle" in lower or "requires an opened" in lower or "missing session" in lower:
        return "session_error"
    if "worker_error" in lower or "failed:" in lower:
        return "worker_error"
    return "execution_error"


def execute_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    from multi_agent.answer_contract import project_from_state
    from multi_agent.tool_gateway import (
        call_tool,
        clear_final_dataset_context,
        set_final_dataset_context,
    )

    session_id = f"det_{index:05d}_{int(time.time() * 1000)}"
    set_final_dataset_context(record, session_id=session_id)
    calls: list[dict[str, Any]] = []
    state_fields: dict[str, Any] = {}
    failure_category = ""
    try:
        for ordinal, call in enumerate(record.get("gt_chain") or [], start=1):
            tool = str(call.get("tool") or "")
            if tool == "TaskComplete":
                continue
            args = dict(call.get("args") or {})
            contract_errors = validate_agent_args(tool, args)
            if contract_errors:
                calls.append(
                    {
                        "ordinal": ordinal,
                        "tool": tool,
                        "args": args,
                        "ok": False,
                        "contract_errors": contract_errors,
                    }
                )
                failure_category = "contract_error"
                break
            result = call_tool(tool, args)
            if result.ok:
                state_fields.update(result.produced.fields)
            calls.append(
                {
                    "ordinal": ordinal,
                    "tool": tool,
                    "args": args,
                    "ok": bool(result.ok),
                    "text": str(result.text),
                }
            )
            if not result.ok:
                failure_category = _category(result.text)
                break
    finally:
        clear_final_dataset_context()

    expected = record.get("expected_final_answer")
    execution_ok = bool(calls) and all(call["ok"] for call in calls)
    projection = project_from_state(record, state_fields) if execution_ok else None
    terminal_present = bool(projection and projection.ok)
    terminal = projection.answer if projection and projection.ok else None
    answer = score_answer(expected, terminal) if terminal_present else {
        "correct": False,
        "answer_acc": 0.0,
        "structured_match": False,
        "valid_json": False,
        "scoring_policy": "terminal_answer_missing",
    }
    if execution_ok and not answer["correct"]:
        failure_category = "answer_schema_or_value_error"
    if execution_ok and answer["correct"]:
        failure_category = "success"
    return {
        "dataset_id": record.get("dataset_id") or record.get("id"),
        "logical_question_type": logical_question_type(record),
        "session_id": session_id,
        "tool_backend": os.environ.get("GEODISASTER_TOOL_BACKEND", "inprocess"),
        "execution_ok": execution_ok,
        "terminal_present": terminal_present,
        "answer": answer,
        "success": bool(execution_ok and answer["correct"]),
        "failure_category": failure_category or "execution_error",
        "calls": calls,
        "projected_final_answer": terminal,
        "answer_contract_errors": list(projection.errors) if projection and not projection.ok else [],
    }


def run_deterministic_replay(
    input_jsonl: Path,
    output_dir: Path,
    *,
    limit: int = 0,
    one_per_type: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(input_jsonl)
    if one_per_type:
        selected: dict[str, dict[str, Any]] = {}
        for record in records:
            selected.setdefault(logical_question_type(record), record)
        records = [selected[key] for key in sorted(selected)]
    if limit:
        records = records[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GEODISASTER_SESSION_DIR", str(output_dir / "tool_sessions"))
    os.environ.setdefault("GEODISASTER_ARTIFACT_DIR", str(output_dir / "tool_artifacts"))
    try:
        from tool_server.tool_workers.online_workers import final_dataset_worker_state as worker_state

        worker_state.SESSION_DIR = Path(os.environ["GEODISASTER_SESSION_DIR"])
        worker_state.ARTIFACT_DIR = Path(os.environ["GEODISASTER_ARTIFACT_DIR"])
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        result = execute_record(record, index)
        rows.append(result)
        print(
            f"[{index}/{len(records)}] {'PASS' if result['success'] else 'FAIL'} "
            f"{result['logical_question_type']} {result['failure_category']}",
            flush=True,
        )

    cases_path = output_dir / "deterministic_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    categories = Counter(row["failure_category"] for row in rows)
    summary = {
        "mode": "deterministic_replay",
        "tool_backend": os.environ.get("GEODISASTER_TOOL_BACKEND", "inprocess"),
        "records": len(rows),
        "logical_question_types": len({row["logical_question_type"] for row in rows}),
        "execution_pass": sum(row["execution_ok"] for row in rows),
        "terminal_answer_pass": sum(bool(row["answer"].get("correct")) for row in rows),
        "end_to_end_pass": sum(row["success"] for row in rows),
        "failure_categories": dict(sorted(categories.items())),
        "cases": str(cases_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
