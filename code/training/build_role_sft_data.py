from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multi_agent.prompts import SYSTEM_PROMPTS, build_orc_decide_message, build_orc_plan_message, build_specialist_message
from multi_agent.schemas import Plan
from multi_agent.answer_contract import project_from_state
from multi_agent.answer_registry import ANSWER_REF, register_answer, resolve_answer_ref
from multi_agent.state import (
    add_layer,
    append_event,
    assign_artifact_handle,
    find_step,
    install_plan,
    mark_step,
    new_episode,
    open_contract_window,
    set_field,
)


ROLE_ORDER = ["ORC", "GA", "PA", "VRA"]


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _last_specialist_report(state: dict[str, Any]) -> dict[str, Any] | None:
    for event in reversed(state["history"]):
        if event["role"] in {"GA", "PA", "VRA"} and event["phase"] == "complete":
            return {
                "agent": event["role"],
                "step_id": event["step_id"],
                "status": event["payload"].get("status"),
                "summary": event["payload"].get("summary"),
            }
    return None


def _register_input_images(state: dict[str, Any], images: list[str]) -> None:
    for image in images:
        image_name = Path(str(image)).name
        suffix = image_name.lower()
        kind = "tif" if suffix.endswith((".tif", ".tiff")) else "img"
        assign_artifact_handle(state, kind=kind, path=str(image), note="input image", produced_by_step="")


def _absorb_explicit_produced(state: dict[str, Any], step_id: str, produced: dict[str, Any]) -> dict[str, Any]:
    handle_map: dict[str, str] = {}
    out_artifacts: list[dict[str, Any]] = []
    for art in produced.get("artifacts", []) or []:
        kind = "gpkg" if str(art.get("kind", "")) == "gpkg" else str(art.get("kind", "img"))
        path = str(art.get("path", ""))
        note = str(art.get("note", ""))
        handle = assign_artifact_handle(state, kind=kind, path=path, note=note, produced_by_step=step_id)
        placeholder = str(art.get("handle") or art.get("gpkg") or "__autogen__")
        handle_map[placeholder] = handle
        out_artifacts.append({"handle": handle, "kind": kind, "path": path, "note": note})

    out_layers: list[dict[str, Any]] = []
    for lay in produced.get("layers", []) or []:
        gpkg = str(lay.get("gpkg", ""))
        gpkg = handle_map.get(gpkg, gpkg)
        name = str(lay.get("name", "")).strip()
        if gpkg in state["artifacts"] and name:
            add_layer(state, gpkg, name, produced_by_step=step_id)
        out_layers.append({"gpkg": gpkg, "name": name})

    out_fields: dict[str, Any] = {}
    for fname, value in (produced.get("fields") or {}).items():
        set_field(state, str(fname), value, produced_by_step=step_id)
        out_fields[str(fname)] = value

    return {"artifacts": out_artifacts, "layers": out_layers, "fields": out_fields}


def _tool_payload(case_id: int, step_id: str, tool_name: str, args: dict[str, Any], tool_output: dict[str, Any], out_root: Path) -> dict[str, Any]:
    ok = bool(tool_output.get("ok", True))
    text = str(tool_output.get("text", ""))
    produced = json.loads(json.dumps(tool_output.get("produced") or {}))
    if tool_name == "GetAreaBoundary":
        gpkg_dir = out_root / "replay_artifacts" / "scripted_gpks"
        gpkg_path = gpkg_dir / f"case_{case_id}_gpkg_1.gpkg"
        if not produced:
            produced = {
                "artifacts": [{"kind": "gpkg", "path": str(gpkg_path.resolve()), "note": "scenario boundary"}],
                "layers": [{"gpkg": "__autogen__", "name": "area_boundary"}],
                "fields": {},
            }
        else:
            artifacts = list(produced.get("artifacts") or [])
            if not artifacts:
                artifacts.append({"kind": "gpkg", "path": str(gpkg_path.resolve()), "note": "scenario boundary"})
            else:
                for artifact in artifacts:
                    if artifact.get("kind") == "gpkg":
                        artifact["path"] = str(gpkg_path.resolve())
            produced["artifacts"] = artifacts
    produced_dump = _absorb_explicit_produced(state=_STATE, step_id=step_id, produced=produced)
    if ok:
        projection = project_from_state(_RECORD, _STATE["fields"])
        if projection.ok:
            register_answer(
                _STATE,
                _RECORD,
                projection.answer,
                produced_by_step=step_id,
                produced_by_tool=tool_name,
            )
    return {
        "tool": tool_name,
        "args": args,
        "ok": ok,
        "text": text,
        "produced": produced_dump,
    }


_STATE: dict[str, Any]
_RECORD: dict[str, Any]


def _build_rows_for_case(record: dict[str, Any], out_root: Path) -> dict[str, list[dict[str, Any]]]:
    global _RECORD, _STATE

    question = str(record.get("question", "")).strip()
    images = list(record.get("images") or [])
    idx = int(record.get("idx", 0))
    _STATE = new_episode(question=question, images=images)
    _RECORD = record
    _register_input_images(_STATE, images)

    role_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}

    plan_payload = dict(record.get("plan") or {})
    plan = Plan.model_validate(plan_payload)
    plan_user = build_orc_plan_message(_STATE)
    role_rows["ORC"].append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["ORC_PLAN"]},
                {"role": "user", "content": plan_user},
                {"role": "assistant", "content": _json_text(plan_payload)},
            ],
            "role": "ORC",
            "case_id": idx,
            "step_id": "plan",
            "kind": "plan",
        }
    )
    install_plan(_STATE, plan)
    append_event(_STATE, role="ORC", phase="plan", payload={"plan": plan_payload, "user_message": plan_user})

    executions = list(record.get("executions") or [])
    for exec_index, execution in enumerate(executions):
        step_id = str(execution.get("step_id", ""))
        step = find_step(_STATE, step_id)
        if step is None:
            continue

        decide_user = build_orc_decide_message(_STATE, _last_specialist_report(_STATE))
        call_agent_payload = {"tool": "CallAgent", "step_id": step_id}
        role_rows["ORC"].append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS["ORC_DECIDE"]},
                    {"role": "user", "content": decide_user},
                    {"role": "assistant", "content": _json_text(call_agent_payload)},
                ],
                "role": "ORC",
                "case_id": idx,
                "step_id": step_id,
                "kind": "decide_call_agent",
            }
        )
        append_event(_STATE, role="ORC", phase="delegate", step_id=step_id, payload={"decision": call_agent_payload, "user_message": decide_user})
        _STATE["current_step_id"] = step_id
        mark_step(_STATE, step_id, "running")
        open_contract_window(_STATE, step_id)
        contract_window_index = _STATE["contract_window"][step_id]

        completion_status = "TASK_COMPLETE"
        completion_summary = ""
        role = str(execution.get("agent", step["agent"]))
        for turn_no, turn in enumerate(execution.get("turns") or [], start=1):
            user_msg = build_specialist_message(_STATE, step, turn_no, contract_window_index)
            action = dict(turn.get("action") or {})
            thought = str(turn.get("thought", "")).strip() or "Follow the tool contract."
            if action.get("tool") == "TaskComplete":
                target_payload = {
                    "thought": thought,
                    "action": {
                        "tool": "TaskComplete",
                        "status": str(action.get("status", "TASK_COMPLETE")),
                        "summary": str(action.get("summary", "")),
                    },
                }
                completion_status = str(action.get("status", "TASK_COMPLETE"))
                completion_summary = str(action.get("summary", ""))
            else:
                target_payload = {
                    "thought": thought,
                    "action": {
                        "tool": str(action.get("tool", "")),
                        "args": dict(action.get("args") or {}),
                    },
                }
            role_rows[role].append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPTS[role]},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": _json_text(target_payload)},
                    ],
                    "role": role,
                    "case_id": idx,
                    "step_id": step_id,
                    "kind": "specialist_turn",
                    "turn_no": turn_no,
                }
            )
            append_event(_STATE, role=role, phase="step", step_id=step_id, turn=turn_no, payload={"user_message": user_msg, "turn": target_payload})
            if action.get("tool") == "TaskComplete":
                continue
            tool_payload = _tool_payload(
                case_id=idx,
                step_id=step_id,
                tool_name=str(action.get("tool", "")),
                args=dict(action.get("args") or {}),
                tool_output=dict(turn.get("tool_output") or {}),
                out_root=out_root,
            )
            append_event(_STATE, role="tool", phase="tool_call", step_id=step_id, turn=turn_no, payload=tool_payload)

        mark_step(_STATE, step_id, "done" if completion_status == "TASK_COMPLETE" else "failed", completion_summary)
        _STATE["current_step_id"] = None
        append_event(_STATE, role=role, phase="complete", step_id=step_id, payload={"status": completion_status, "summary": completion_summary})

        if exec_index == len(executions) - 1:
            final_user = build_orc_decide_message(_STATE, _last_specialist_report(_STATE))
            resolution = resolve_answer_ref(_STATE, ANSWER_REF)
            if not resolution.ok:
                raise ValueError(
                    f"teacher tools did not produce an answer reference for case {idx}: "
                    f"{resolution.error}"
                )
            terminate_payload = {"tool": "Terminate", "answer_ref": ANSWER_REF}
            role_rows["ORC"].append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPTS["ORC_DECIDE"]},
                        {"role": "user", "content": final_user},
                        {"role": "assistant", "content": _json_text(terminate_payload)},
                    ],
                    "role": "ORC",
                    "case_id": idx,
                    "step_id": "terminate",
                    "kind": "decide_terminate",
                }
            )

    return role_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build role-wise v3 SFT datasets from structured teacher traces.")
    parser.add_argument("--teacher_dir", required=True, help="Directory containing multiagent_*.json files.")
    parser.add_argument("--output_dir", required=True, help="Output directory for role datasets.")
    parser.add_argument("--limit", type=int, default=0, help="Optional teacher case limit.")
    args = parser.parse_args()

    teacher_dir = Path(args.teacher_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(teacher_dir.glob("multiagent_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    if args.limit:
        files = files[: args.limit]

    role_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        case_rows = _build_rows_for_case(record, output_dir)
        for role in ROLE_ORDER:
            role_rows[role].extend(case_rows[role])

    manifest: dict[str, Any] = {
        "teacher_dir": str(teacher_dir),
        "n_cases": len(files),
        "roles": {},
    }
    for role, rows in role_rows.items():
        out_path = output_dir / f"{role.lower()}_train.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["roles"][role] = {"path": str(out_path), "rows": len(rows)}

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
