#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from single_agent.prompts import (  # noqa: E402
    SINGLE_AGENT_SYSTEM,
    build_initial_user_message,
    build_runtime_user_message,
    compact_json,
)
from single_agent.runtime import (  # noqa: E402
    absorb_tool_result,
    build_single_plan_from_record,
    clear_geodisaster_context,
    flatten_teacher_tool_turns,
    install_geodisaster_context,
    new_single_state,
    render_memory,
)
from multi_agent.schemas import Produced, ToolResult  # noqa: E402
from multi_agent.answer_registry import (  # noqa: E402
    ANSWER_REF,
    answer_sha256,
    register_answer,
    resolve_answer_ref,
)
from evaluation.structured_answer import score_answer  # noqa: E402


DATASET_ROOT = Path(
    os.environ.get("GEODISASTER_DATASET_ROOT", str(ROOT.parent / "geodisaster-agentic-2921"))
).expanduser()
DEFAULT_GD_TRAIN = DATASET_ROOT / "data" / "multi_agent" / "train_conversations.jsonl"
DEFAULT_GD_TEST = DATASET_ROOT / "data" / "multi_agent" / "test_conversations.jsonl"
DEFAULT_EO = ROOT / "data/optional_eo_teacher_train.json"
DEFAULT_OUT = ROOT / "runs" / "single_agent_sft"


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def load_eo(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("records") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"unexpected EO teacher format in {path}")
    return rows[:limit] if limit else rows


def tool_result_from_teacher(turn: dict[str, Any]) -> ToolResult | None:
    raw = turn.get("tool_output")
    if not isinstance(raw, dict):
        return None
    produced = raw.get("produced") or {}
    # Some teacher files omit artifact handles, which is valid for pydantic if we
    # add the runtime placeholder expected by absorb_tool_result.
    artifacts = []
    for art in produced.get("artifacts") or []:
        item = dict(art)
        item.setdefault("handle", "__autogen__")
        artifacts.append(item)
    layers = [dict(layer) for layer in produced.get("layers") or []]
    fields = dict(produced.get("fields") or {})
    return ToolResult(
        ok=bool(raw.get("ok", True)),
        text=str(raw.get("text") or raw.get("observation") or ""),
        produced=Produced(artifacts=artifacts, layers=layers, fields=fields),
    )


def convert_record(record: dict[str, Any], *, source: str) -> dict[str, Any]:
    state = new_single_state(record)
    install_geodisaster_context(record, session_id=f"single_sft_{record.get('idx', 'case')}")
    plan_action = build_single_plan_from_record(record)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SINGLE_AGENT_SYSTEM},
        {"role": "user", "content": build_initial_user_message(record)},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "thought": "Plan the evidence path first so tool calls stay grounded in visible handles and requested outputs.",
                    "action": plan_action,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    state["plan"] = plan_action
    last_observation: dict[str, Any] | None = None
    for turn_no, turn in enumerate(flatten_teacher_tool_turns(record), start=1):
        messages.append(
            {
                "role": "user",
                "content": build_runtime_user_message(
                    record,
                    plan=plan_action,
                    memory_text=render_memory(state),
                    last_observation=last_observation,
                    turn_no=turn_no,
                ),
            }
        )
        action = dict(turn.get("action") or {})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "thought": str(turn.get("thought") or f"Use {action.get('tool')} to produce the next needed evidence."),
                        "action": action,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        result = tool_result_from_teacher(turn)
        if result is not None and isinstance(action, dict):
            last_observation = absorb_tool_result(state, str(action.get("tool")), dict(action.get("args") or {}), result)
        else:
            last_observation = {"tool": action.get("tool"), "status": "UNKNOWN", "summary": "teacher turn had no tool output"}

    resolution = resolve_answer_ref(state, ANSWER_REF)
    if (
        record.get("resolved_final_answer") is not None
        and (
            not resolution.ok
            or not score_answer(record["resolved_final_answer"], resolution.answer)["correct"]
        )
    ):
        register_answer(
            state,
            record,
            record["resolved_final_answer"],
            produced_by_step="single_sft_export",
            produced_by_tool="validated_teacher_projection_cache",
        )
        resolution = resolve_answer_ref(state, ANSWER_REF)
    if not resolution.ok:
        clear_geodisaster_context()
        raise ValueError(
            f"teacher tools did not produce a schema-valid answer reference for "
            f"{record.get('dataset_id') or record.get('id')}: {resolution.error}"
        )
    expected = record.get("expected_final_answer")
    if expected is not None and not score_answer(expected, resolution.answer)["correct"]:
        clear_geodisaster_context()
        raise ValueError(
            f"worker-derived answer differs from canonical expected answer for "
            f"{record.get('dataset_id') or record.get('id')}"
        )
    messages.append(
        {
            "role": "user",
            "content": build_runtime_user_message(
                record,
                plan=plan_action,
                memory_text=render_memory(state),
                last_observation=last_observation,
                turn_no=len(messages),
            ),
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "thought": "The necessary evidence is now in shared memory; answer from the tool results without inventing new values.",
                    "action": {"tool": "FinalAnswer", "answer_ref": ANSWER_REF},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    )
    dataset_id = record.get("dataset_id") or record.get("id") or record.get("case_id") or record.get("idx")
    # Link the single-agent target to the already audited resolved payload in
    # the source conversation. A fresh hash of a numerically equivalent live
    # projection can differ because strict scoring permits small tolerances.
    resolved_sha256 = str(record.get("resolved_answer_sha256") or answer_sha256(state))
    clear_geodisaster_context()
    return {
        "messages": messages,
        # Preserve the domain source used by worker dispatch. ``source`` here
        # must never be replaced by an export/split label such as
        # ``geodisaster_test``.
        "source": record.get("source") or source,
        "export_source": source,
        "idx": record.get("idx"),
        "id": dataset_id,
        "dataset_id": dataset_id,
        "sample_id": record.get("sample_id"),
        "question_type": record.get("question_type") or record.get("category"),
        "final_split": record.get("final_split"),
        "tool_sequence": [t.get("action", {}).get("tool") for t in flatten_teacher_tool_turns(record)],
        "training_weight": float(record.get("training_weight", 1.0)),
        "training_weight_contract": record.get("training_weight_contract"),
        "final_answer_ref": ANSWER_REF,
        "resolved_answer_sha256": resolved_sha256,
    }


def explode_case_to_sft_rows(case_row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = case_row["messages"]
    if not messages or messages[0].get("role") != "system":
        raise ValueError("single-agent case row must start with a system message")
    system = messages[0]
    rows: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if idx == 0 or messages[idx - 1].get("role") != "user":
            continue
        action_name = ""
        try:
            payload = json.loads(message.get("content") or "{}")
            action_name = str((payload.get("action") or {}).get("tool") or "")
        except Exception:
            action_name = ""
        rows.append(
            {
                "messages": [system, messages[idx - 1], message],
                "source": case_row.get("source", ""),
                "idx": case_row.get("idx"),
                "id": case_row.get("id"),
                "dataset_id": case_row.get("dataset_id"),
                "sample_id": case_row.get("sample_id"),
                "question_type": case_row.get("question_type"),
                "final_split": case_row.get("final_split"),
                "case_id": case_row.get("dataset_id") or case_row.get("idx"),
                "kind": "single_agent_turn",
                "assistant_action": action_name,
                "assistant_turn_index": len(rows),
                "training_weight": float(case_row.get("training_weight", 1.0)),
                "training_weight_contract": case_row.get("training_weight_contract"),
                "final_answer_ref": case_row.get("final_answer_ref"),
                "resolved_answer_sha256": case_row.get("resolved_answer_sha256"),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview(path: Path, rows: list[dict[str, Any]], n: int = 43) -> None:
    lines = ["# Single-Agent Toolchain Preview", ""]
    for i, row in enumerate(rows[:n], start=1):
        messages = row["messages"]
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        question = first_user.split("REGISTERED INPUTS", 1)[0].replace("QUESTION", "").strip()
        tools = [t for t in row.get("tool_sequence") or [] if t]
        lines.extend(
            [
                f"## {i:03d} {row.get('dataset_id')}",
                "",
                question[:1000],
                "",
                f"Tools: {' -> '.join(tools)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geodisaster_train", type=Path, default=DEFAULT_GD_TRAIN)
    parser.add_argument("--geodisaster_test", type=Path, default=DEFAULT_GD_TEST)
    parser.add_argument("--eo_teacher", type=Path, default=DEFAULT_EO)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit_geodisaster", type=int, default=0)
    parser.add_argument("--limit_eo", type=int, default=0)
    args = parser.parse_args()

    gd_train_cases = [convert_record(r, source="geodisaster") for r in load_jsonl(args.geodisaster_train, args.limit_geodisaster)]
    gd_test_cases = [convert_record(r, source="geodisaster_test") for r in load_jsonl(args.geodisaster_test, 0)]
    eo_cases = [convert_record(r, source="eo_5k") for r in load_eo(args.eo_teacher, args.limit_eo)]
    gd_train = [row for case in gd_train_cases for row in explode_case_to_sft_rows(case)]
    gd_test = [row for case in gd_test_cases for row in explode_case_to_sft_rows(case)]
    eo_rows = [row for case in eo_cases for row in explode_case_to_sft_rows(case)]
    combined = eo_rows + gd_train

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "geodisaster_single_agent_train.jsonl", gd_train)
    write_jsonl(out / "geodisaster_single_agent_train_conversations.jsonl", gd_train_cases)
    write_jsonl(out / "geodisaster_single_agent_test_reference.jsonl", gd_test_cases)
    write_jsonl(out / "geodisaster_single_agent_test_sft_turns.jsonl", gd_test)
    write_jsonl(out / "combined_eo5k_geodisaster_single_agent_train.jsonl", combined)
    write_jsonl(out / "eo5k_single_agent_train_conversations.jsonl", eo_cases)
    (out / "single_agent_system_prompt.md").write_text(
        "# Single-Agent System Prompt\n\n```text\n" + SINGLE_AGENT_SYSTEM + "\n```\n",
        encoding="utf-8",
    )
    write_preview(out / "preview_43_single_agent_toolchains.md", gd_train_cases, n=43)
    (out / "preview_43_single_agent_train.json").write_text(
        json.dumps(gd_train_cases[:43], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "geodisaster_train_source": args.geodisaster_train.name,
        "geodisaster_test_source": args.geodisaster_test.name,
        "eo_teacher_source": args.eo_teacher.name,
        "outputs": {
            "geodisaster_train": "geodisaster_single_agent_train.jsonl",
            "geodisaster_train_conversations": "geodisaster_single_agent_train_conversations.jsonl",
            "geodisaster_test_reference": "geodisaster_single_agent_test_reference.jsonl",
            "geodisaster_test_sft_turns": "geodisaster_single_agent_test_sft_turns.jsonl",
            "combined_train": "combined_eo5k_geodisaster_single_agent_train.jsonl",
            "eo_conversations": "eo5k_single_agent_train_conversations.jsonl",
            "single_agent_system_prompt": "single_agent_system_prompt.md",
            "preview_43_json": "preview_43_single_agent_train.json",
            "preview_43_md": "preview_43_single_agent_toolchains.md",
        },
        "counts": {
            "geodisaster_train_cases": len(gd_train_cases),
            "geodisaster_train_sft_turns": len(gd_train),
            "geodisaster_test_cases": len(gd_test_cases),
            "geodisaster_test_sft_turns": len(gd_test),
            "eo_cases": len(eo_cases),
            "eo_sft_turns": len(eo_rows),
            "combined": len(combined),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
