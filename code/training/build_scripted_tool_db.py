from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PASS_THROUGH_TOOLS = {"GoogleSearch", "Plot", "Solver", "Calculator"}


def _answer_text(value: Any) -> str:
    if isinstance(value, (dict, list, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value or "").strip()


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "")
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(raw[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : idx + 1])
                except Exception:
                    return None
    return None


def _extract_question_text(value: str) -> str:
    text = str(value or "").strip()
    if "\n\nQuestion:" in text:
        return text.split("\n\nQuestion:", 1)[1].strip()
    marker = "Question:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def _extract_gold_label(record: dict[str, Any]) -> str:
    if record.get("resolved_final_answer") is not None:
        return _answer_text(record["resolved_final_answer"])
    if record.get("expected_final_answer") is not None:
        return _answer_text(record["expected_final_answer"])
    if record.get("final_answer"):
        return str(record.get("final_answer", "")).strip()
    for turn in reversed(record.get("conversation", []) or []):
        if str(turn.get("from", "")).lower() not in {"gpt", "assistant"}:
            continue
        payload = _extract_first_json_object(str(turn.get("value", "")))
        if not isinstance(payload, dict):
            continue
        actions = payload.get("actions") or []
        if not isinstance(actions, list) or not actions:
            continue
        action = actions[0]
        if str(action.get("name", "")) != "Terminate":
            continue
        arguments = action.get("arguments") or {}
        if isinstance(arguments, dict):
            answer = str(arguments.get("answer", "") or arguments.get("ans", "")).strip()
            if answer:
                return answer
    return ""


def _ensure_placeholder_gpkg(data_root: Path, case_id: int) -> str:
    out_dir = data_root / "replay_artifacts" / "scripted_gpks"
    path = out_dir / f"case_{case_id}_gpkg_1.gpkg"
    return str(path.resolve())


def _first_int(text: str) -> int | None:
    import re

    match = re.search(r"(\d+)", str(text or ""))
    return int(match.group(1)) if match else None


def _parse_first_distance(text: str) -> tuple[str, float | None, float | None]:
    import re

    match = re.search(
        r"\n([^,\n]+?)\s*,\s*([^,\n]+?),\s*distance=([-+]?\d*\.?\d+)\s*m(?:,\s*travel_time=([-+]?\d*\.?\d+)\s*s)?",
        "\n" + str(text or ""),
    )
    if not match:
        return "", None, None
    pair = f"{match.group(1).strip()} -> {match.group(2).strip()}"
    distance_m = round(float(match.group(3)), 2)
    travel_time_s = round(float(match.group(4)), 2) if match.group(4) else None
    return pair, distance_m, travel_time_s


def _response_from_observation(
    *,
    case_id: int,
    data_root: Path,
    tool_name: str,
    args: dict[str, Any],
    observation_text: str,
) -> dict[str, Any]:
    response: dict[str, Any] = {"text": observation_text, "error_code": 0}
    if tool_name == "GetAreaBoundary":
        gpkg_path = _ensure_placeholder_gpkg(data_root, case_id)
        response["gpkg"] = gpkg_path
        response["produced"] = {
            "artifacts": [{"kind": "gpkg", "path": gpkg_path, "note": "scenario boundary"}],
            "layers": [{"gpkg": "__autogen__", "name": "area_boundary"}],
            "fields": {},
        }
        return response
    if tool_name == "AddPoisLayer":
        layer_name = str(args.get("layer_name", "")).strip()
        count = _first_int(observation_text)
        response["produced"] = {
            "artifacts": [],
            "layers": [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": layer_name}] if layer_name else [],
            "fields": {f"{layer_name}_count": count} if layer_name and count is not None else {},
        }
        return response
    if tool_name == "ComputeDistance":
        src = str(args.get("src_layer", "")).strip()
        tar = str(args.get("tar_layer", "")).strip()
        pair, distance_m, travel_time_s = _parse_first_distance(observation_text)
        fields: dict[str, Any] = {}
        if pair:
            fields["pair"] = pair
        if distance_m is not None:
            fields["distance_m"] = distance_m
        if travel_time_s is not None:
            fields["travel_time_s"] = travel_time_s
        response["produced"] = {
            "artifacts": [],
            "layers": [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": f"{src}_to_{tar}_distances"}] if src and tar else [],
            "fields": fields,
        }
        return response
    return response


def _response_from_explicit_output(
    *,
    case_id: int,
    data_root: Path,
    tool_name: str,
    tool_output: dict[str, Any],
) -> dict[str, Any]:
    ok = bool(tool_output.get("ok", True))
    response: dict[str, Any] = {
        "text": str(tool_output.get("text", "")),
        "error_code": 0 if ok else 1,
    }
    produced = tool_output.get("produced")
    if isinstance(produced, dict):
        response["produced"] = json.loads(json.dumps(produced))
    if tool_name == "GetAreaBoundary":
        gpkg_path = _ensure_placeholder_gpkg(data_root, case_id)
        response["gpkg"] = gpkg_path
        if "produced" not in response:
            response["produced"] = {
                "artifacts": [{"kind": "gpkg", "path": gpkg_path, "note": "scenario boundary"}],
                "layers": [{"gpkg": "__autogen__", "name": "area_boundary"}],
                "fields": {},
            }
        else:
            artifacts = list(response["produced"].get("artifacts") or [])
            if not artifacts:
                artifacts.append({"kind": "gpkg", "path": gpkg_path, "note": "scenario boundary"})
            else:
                for artifact in artifacts:
                    if artifact.get("kind") == "gpkg":
                        artifact["path"] = gpkg_path
            response["produced"]["artifacts"] = artifacts
    return response


def _find_next_observation(conversation_tail: list[dict[str, Any]]) -> str:
    for item in conversation_tail:
        if str(item.get("from", "")).lower() not in {"human", "user"}:
            continue
        value = str(item.get("value", ""))
        if "OBSERVATION:" in value:
            text = value.split("OBSERVATION:", 1)[1]
            text = text.split("Please summarize the model outputs and answer my first question.", 1)[0]
            return text.strip()
    return ""


def _build_case_from_conversation(case_id: int, record: dict[str, Any], data_root: Path) -> dict[str, Any]:
    conversation = list(record.get("conversation") or [])
    turns: list[dict[str, Any]] = []
    for idx, item in enumerate(conversation):
        if str(item.get("from", "")).lower() not in {"gpt", "assistant"}:
            continue
        payload = _extract_first_json_object(str(item.get("value", "")))
        if not isinstance(payload, dict):
            continue
        actions = payload.get("actions") or []
        if not isinstance(actions, list) or not actions:
            continue
        action = actions[0]
        tool_name = str(action.get("name", "")).strip()
        if not tool_name or tool_name in {"Terminate", *PASS_THROUGH_TOOLS}:
            continue
        args = dict(action.get("arguments") or {})
        obs = _find_next_observation(conversation[idx + 1 :])
        if not obs:
            continue
        turns.append(
            {
                "tool": tool_name,
                "args": args,
                "response": _response_from_observation(
                    case_id=case_id,
                    data_root=data_root,
                    tool_name=tool_name,
                    args=args,
                    observation_text=obs,
                ),
            }
        )
    question_source = record.get("question", "") or (conversation[0].get("value", "") if conversation else "")
    return {
        "case_id": case_id,
        "idx": int(record.get("idx", case_id)),
        "question": _extract_question_text(question_source),
        "label": _extract_gold_label(record),
        "images": list(record.get("images") or []),
        "turns": turns,
    }


def _build_case_from_execution(case_id: int, record: dict[str, Any], data_root: Path) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    for execution in record.get("executions") or []:
        for item in execution.get("turns") or []:
            action = dict(item.get("action") or {})
            tool_name = str(action.get("tool", "")).strip()
            if not tool_name or tool_name in {"TaskComplete", *PASS_THROUGH_TOOLS}:
                continue
            turns.append(
                {
                    "tool": tool_name,
                    "args": dict(action.get("args") or {}),
                    "response": _response_from_explicit_output(
                        case_id=case_id,
                        data_root=data_root,
                        tool_name=tool_name,
                        tool_output=dict(item.get("tool_output") or {}),
                    ),
                }
            )
    return {
        "case_id": case_id,
        "idx": int(record.get("idx", case_id)),
        "question": _extract_question_text(str(record.get("question", "")).strip()),
        "label": _extract_gold_label(record),
        "images": list(record.get("images") or []),
        "turns": turns,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        records = []
        for file_path in sorted(path.glob("multiagent_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
            records.append(json.loads(file_path.read_text(encoding="utf-8")))
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported source schema: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile scripted tool cases for fast v3 evaluation.")
    parser.add_argument("--source", required=True, help="Source file or directory.")
    parser.add_argument("--output", required=True, help="Compiled scenario JSON path.")
    parser.add_argument("--dataset-output", default="", help="Optional eval dataset JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit.")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()
    dataset_output_path = Path(args.dataset_output).resolve() if args.dataset_output else None
    records = _load_records(source_path)
    if args.limit:
        records = records[: args.limit]

    data_root = output_path.parent
    cases: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    for case_id, record in enumerate(records):
        if isinstance(record, dict) and isinstance(record.get("executions"), list):
            case = _build_case_from_execution(case_id, record, data_root)
        else:
            case = _build_case_from_conversation(case_id, record, data_root)
        cases.append(case)
        eval_records.append(
            {
                "idx": case["idx"],
                "question": case["question"],
                "label": case["label"],
                "images": case["images"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "meta": {
                    "source": str(source_path),
                    "n_cases": len(cases),
                },
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if dataset_output_path is not None:
        dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_output_path.write_text(
            json.dumps({"meta": {"source": str(source_path)}, "records": eval_records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {len(cases)} scripted cases to {output_path}")
    if dataset_output_path is not None:
        print(f"Wrote eval dataset records to {dataset_output_path}")


if __name__ == "__main__":
    main()
