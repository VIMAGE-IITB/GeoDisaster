from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from multi_agent.answer_contract import project_from_state
from multi_agent.answer_registry import answer_registry_cards, register_answer, resolve_answer_ref
from multi_agent.schemas import ArtifactRecord, LayerRecord, ToolResult
from multi_agent.tool_gateway import (
    call_tool as call_geodisaster_tool,
    clear_final_dataset_context,
    current_sanitized_case,
    set_final_dataset_context,
)

from .prompts import compact_artifact_line, compact_json


def normalize_geodisaster_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return dict(args or {})


def new_single_state(record: dict[str, Any]) -> dict[str, Any]:
    question = record.get("question")
    if isinstance(question, dict):
        question_text = str(question.get("text") or record.get("raw_question") or "")
    else:
        question_text = str(question or record.get("raw_question") or "")
    state = {
        "question": question_text,
        "artifacts": {},
        "layers": {},
        "fields": {},
        "answer_registry": {},
        "field_history": {},
        "history": [],
        "counters": {"gpkg": 0, "img": 0, "tif": 0},
        "plan": None,
        "final_answer": None,
        "status": "RUNNING",
    }
    artifacts = record.get("initial_artifacts") if isinstance(record.get("initial_artifacts"), dict) else {}
    for handle, card in artifacts.items():
        state["artifacts"][handle] = {
            "kind": card.get("type") or card.get("kind") or "artifact",
            "path": _first_path(card) or handle,
            "note": _artifact_note(handle, card),
            "produced_by": "input",
        }
    for idx, path in enumerate(record.get("images") or [], start=1):
        state["artifacts"][f"image_{idx}"] = {
            "kind": "img",
            "path": str(path),
            "note": "input image",
            "produced_by": "input",
        }
    return state


def _first_path(card: dict[str, Any]) -> str:
    if isinstance(card.get("question_path"), dict) and card["question_path"].get("path"):
        return str(card["question_path"]["path"])
    paths = card.get("paths") if isinstance(card.get("paths"), dict) else {}
    for entries in paths.values():
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and entry.get("path"):
                    return str(entry["path"])
    return ""


def _artifact_note(handle: str, card: dict[str, Any]) -> str:
    return f"{handle}: {card.get('description') or card.get('role') or card.get('type') or 'input'}"


def install_geodisaster_context(record: dict[str, Any], session_id: str | None = None) -> str:
    return set_final_dataset_context(record, session_id=session_id)


def clear_geodisaster_context() -> None:
    clear_final_dataset_context()


def call_single_tool(name: str, args: dict[str, Any]) -> ToolResult:
    # Route unknown names through the same gateway as known tools. The gateway
    # returns the canonical ``unknown_tool`` category, keeping single- and
    # multi-agent observations and evaluator telemetry consistent.
    return call_geodisaster_tool(name, normalize_geodisaster_args(name, args))


def absorb_tool_result(state: dict[str, Any], tool_name: str, args: dict[str, Any], result: ToolResult) -> dict[str, Any]:
    produced = result.produced
    # Prompt field order must be reproducible.  Wall-clock timestamps made
    # same-tool fields exchange order across otherwise identical export runs.
    # A completed-turn ordinal preserves recency while giving fields from the
    # same observation a stable lexical tie-break in _preferred_field_order.
    turn_ordinal = len(state.get("history") or []) + 1
    handle_map: dict[str, str] = {}
    out_artifacts: list[dict[str, Any]] = []
    for art in produced.artifacts:
        handle = _register_artifact(state, art)
        handle_map[art.handle] = handle
        out_artifacts.append({"handle": handle, "kind": art.kind, "path": art.path, "note": art.note})

    out_layers: list[dict[str, Any]] = []
    for lay in produced.layers:
        gpkg = handle_map.get(lay.gpkg, lay.gpkg)
        state["layers"].setdefault(gpkg, [])
        if lay.name not in state["layers"][gpkg]:
            state["layers"][gpkg].append(lay.name)
        out_layers.append({"gpkg": gpkg, "name": lay.name})

    out_fields: dict[str, Any] = {}
    for key, value in produced.fields.items():
        history = state.setdefault("field_history", {}).setdefault(key, [])
        history.append(value)
        state["fields"][key] = {"value": value, "tool": tool_name, "ts": turn_ordinal}
        out_fields[key] = value
        if len(history) > 1:
            state["fields"][f"{key}_history"] = {
                "value": history[-12:],
                "tool": tool_name,
                "ts": turn_ordinal,
            }

    case = current_sanitized_case()
    if result.ok and case:
        projection = project_from_state(case, state["fields"])
        if projection.ok:
            register_answer(
                state,
                case,
                projection.answer,
                produced_by_step=f"turn_{turn_ordinal}",
                produced_by_tool=tool_name,
            )

    payload = {
        "tool": tool_name,
        "args": args,
        "ok": bool(result.ok),
        "text": str(result.text),
        "produced": {"artifacts": out_artifacts, "layers": out_layers, "fields": out_fields},
    }
    state["history"].append({"role": "tool", "payload": payload})
    return payload


def _register_artifact(state: dict[str, Any], art: ArtifactRecord) -> str:
    kind = art.kind
    for handle, record in state["artifacts"].items():
        if record.get("kind") == kind and record.get("path") == art.path:
            return handle
    state["counters"][kind] = int(state["counters"].get(kind, 0)) + 1
    prefix = "image" if kind == "img" else kind
    handle = f"{prefix}_{state['counters'][kind]}"
    state["artifacts"][handle] = {
        "kind": kind,
        "path": art.path,
        "note": art.note,
        "produced_by": "tool",
    }
    return handle


def render_memory(state: dict[str, Any], *, max_fields: int = 16) -> str:
    lines: list[str] = []
    if state.get("artifacts"):
        lines.append("Artifacts:")
        for handle, card in sorted(state["artifacts"].items()):
            lines.append(compact_artifact_line(handle, card))
    if state.get("layers"):
        lines.append("Layers:")
        for gpkg, names in sorted(state["layers"].items()):
            lines.append(f"- {gpkg}: {', '.join(map(str, names[:12]))}")
    fields = state.get("fields") or {}
    if fields:
        lines.append("Fields:")
        preferred = _preferred_field_order(fields)
        for key in preferred[:max_fields]:
            value = fields[key]["value"] if isinstance(fields[key], dict) and "value" in fields[key] else fields[key]
            lines.append(f"- {key} = {compact_json(value, max_chars=320)}")
        if len(preferred) > max_fields:
            lines.append(f"- ... {len(preferred) - max_fields} older fields hidden")
    lines.append("Available answer references:")
    cards = answer_registry_cards(state)
    lines.extend(cards or ["  (none; continue gathering evidence)"])
    return "\n".join(lines)


def _preferred_field_order(fields: dict[str, Any]) -> list[str]:
    keys = list(fields.keys())
    priority = []
    for key in keys:
        if key == "latest_answer_ref" or key.endswith(("_answer_ref", "_final_answer")):
            continue
        if key.endswith(("_payload", "_payload_keys", "_history")):
            continue
        score = 0
        if key.startswith("latest_"):
            score += 100
        if key.endswith("_summary") or key.endswith("_evidence") or key.endswith("_payload"):
            score += 40
        try:
            ts = float(fields[key].get("ts", 0)) if isinstance(fields[key], dict) else 0.0
        except Exception:
            ts = 0.0
        priority.append((-score, -ts, key))
    priority.sort()
    return [key for _, _, key in priority]


def flatten_teacher_tool_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for execution in record.get("executions") or []:
        for turn in execution.get("turns") or []:
            action = turn.get("action") or {}
            if action.get("tool") == "TaskComplete":
                continue
            turns.append(turn)
    if turns:
        return turns
    for call in record.get("gt_chain") or []:
        if call.get("tool") == "TaskComplete":
            continue
        turns.append({"thought": f"Use {call.get('tool')} for the next required evidence.", "action": call})
    return turns


def build_single_plan_from_record(record: dict[str, Any]) -> dict[str, Any]:
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    steps: list[dict[str, Any]] = []
    for idx, step in enumerate(plan.get("steps") or [], start=1):
        objective = str(step.get("subtask") or step.get("objective") or "").strip()
        objective = re.sub(r"^(GA|PA|VRA)\s+contract\s+for\s+", "", objective, flags=re.I)
        objective = re.sub(r"^(GA|PA|VRA)\s*:\s*", "", objective, flags=re.I)
        if not objective:
            objective = "Produce the next evidence needed for the answer."
        steps.append({"id": f"s{idx}", "objective": objective, "needs": list(step.get("needs") or [])})
    if not steps:
        turns = flatten_teacher_tool_turns(record)
        steps = [{"id": "s1", "objective": "Use the declared tools to gather evidence and answer the question.", "needs": []}]
        if turns:
            steps[0]["objective"] = "Gather the required evidence, compute any requested values, and answer from tool results."
    return {
        "tool": "Plan",
        "goal": str(plan.get("goal") or "Solve the question using the available tools and visible handles."),
        "steps": steps[:10],
    }
