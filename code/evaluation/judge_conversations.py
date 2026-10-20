#!/usr/bin/env python3
"""Build and score GeoDisaster LLM-as-judge requests from conversation logs.

The script is intentionally run-agnostic:

  - it accepts one or more conversation JSON files, JSONL files, or directories;
  - it builds compact case payloads for an LLM judge;
  - in dry-run mode it writes requests only;
  - in openai mode it calls an OpenAI-compatible chat-completions endpoint;
  - it aggregates Table-3/Table-4 style metrics plus per-agent breakdowns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.structured_answer import score_answer, structured_diff  # noqa: E402


DEFAULT_PROMPT = REPO_ROOT / "docs" / "geodisaster_llm_as_judge_prompt.md"
DEFAULT_OUT = ROOT / "runs" / "geodisaster_conversation_judge"

TABLE3_KEYS = ["Inst", "Tool", "ArgN", "ArgV", "Summ"]
TASK_KEYS = ["TSR", "Ans", "StrictCorrect", "CSR", "ESR", "PlanAcc", "DelegAcc"]
TABLE4_KEYS = ["Per", "Op", "Logic", "GIS", "AnyOr", "SameO", "Uni", "Ans", "Gen"]
ALL_SCORE_KEYS = list(dict.fromkeys(TABLE3_KEYS + TASK_KEYS + TABLE4_KEYS))
# The prompt applies the strict paper-table rule that failed emitted tool calls
# cannot preserve perfect ArgN/ArgV credit. This script aggregates the judge's
# already-scored JSON values; deterministic trace-cap audits can be kept beside
# any released evaluation bundle alongside the corresponding denominators.
FAILURE_TAGS = [
    "PlanErr",
    "AgentErr",
    "ToolErr",
    "ArgErr",
    "FormatErr",
    "TermErr",
    "LoopErr",
    "ReplanErr",
    "SynthErr",
    "ConstraintErr",
    "ToolExecErr",
    "AbortErr",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def extract_marked(text: str, name: str) -> str:
    start = f"<!-- {name}_START -->"
    end = f"<!-- {name}_END -->"
    if start not in text or end not in text:
        raise ValueError(f"Prompt file is missing markers {start} / {end}")
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def load_prompt(prompt_md: Path) -> tuple[str, str]:
    text = read_text(prompt_md)
    return extract_marked(text, "SYSTEM_PROMPT"), extract_marked(text, "USER_PROMPT_TEMPLATE")


def clip_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def compact(value: Any, *, max_text: int, max_items: int = 24, depth: int = 0) -> Any:
    if depth > 4:
        return clip_text(value, max_text)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= max_items:
                out["__truncated_keys__"] = len(value) - max_items
                break
            if str(key) in {"user_message"}:
                out[str(key) + "_excerpt"] = clip_text(val, max_text)
            else:
                out[str(key)] = compact(val, max_text=max_text, max_items=max_items, depth=depth + 1)
        return out
    if isinstance(value, list):
        items = [
            compact(item, max_text=max_text, max_items=max_items, depth=depth + 1)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append({"__truncated_items__": len(value) - max_items})
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return clip_text(value, max_text)
        return value
    return clip_text(value, max_text)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
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
                    obj = json.loads(raw[start : idx + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def conversation_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"conversation_(\d+)\.json$", path.name)
    idx = int(match.group(1)) if match else 10**9
    return (str(path.parent), idx, str(path))


def collect_records(inputs: list[Path], *, recursive: bool) -> list[tuple[str, Path, dict[str, Any]]]:
    records: list[tuple[str, Path, dict[str, Any]]] = []
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if input_path.is_dir():
            pattern = "**/conversation_*.json" if recursive else "conversation_*.json"
            conversation_files = sorted(input_path.glob(pattern), key=conversation_sort_key)
            if conversation_files:
                for file_path in conversation_files:
                    records.append((file_path.stem, file_path, json.loads(read_text(file_path))))
                continue
            jsonl_files = sorted(input_path.glob("**/online_conversations.jsonl" if recursive else "online_conversations.jsonl"))
            for file_path in jsonl_files:
                for idx, row in enumerate(read_jsonl(file_path), start=1):
                    records.append((f"{file_path.stem}_{idx}", file_path, row))
            continue
        if input_path.suffix == ".jsonl":
            for idx, row in enumerate(read_jsonl(input_path), start=1):
                records.append((f"{input_path.stem}_{idx}", input_path, row))
            continue
        if input_path.suffix == ".json":
            records.append((input_path.stem, input_path, json.loads(read_text(input_path))))
            continue
        raise ValueError(f"Unsupported input path: {input_path}")
    return records


def mode_from_record(record: dict[str, Any], forced_mode: str) -> str:
    if forced_mode != "auto":
        return forced_mode
    roles = {event.get("role") for event in (record.get("trace") or []) if isinstance(event, dict)}
    if roles.intersection({"ORC", "GA", "PA", "VRA"}):
        return "multiagent"
    if record.get("plan") and record.get("steps"):
        return "multiagent"
    return "singleagent"


def expected_answer_from_record(record: dict[str, Any]) -> Any:
    """Return the reference answer from either multi- or single-agent output."""

    if "expected_final_answer" in record:
        return record.get("expected_final_answer")
    return record.get("gold")


def predicted_answer_from_record(record: dict[str, Any]) -> Any:
    """Return the prediction without changing its native JSON/string shape."""

    if "final_answer" in record:
        return record.get("final_answer")
    return record.get("prediction")


def question_from_record(record: dict[str, Any]) -> str:
    """Read a direct question or recover it from a saved single-agent prompt."""

    question = record.get("question")
    if isinstance(question, dict):
        question = question.get("text") or question.get("question")
    if question:
        return str(question)
    messages = record.get("messages") or []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = str(message.get("content") or "").strip()
        if text.startswith("QUESTION\n"):
            text = text[len("QUESTION\n") :]
        for marker in ("\n\nREGISTERED INPUTS", "\n\nANSWER SCHEMA", "\n\nSTATE"):
            if marker in text:
                text = text.split(marker, 1)[0]
                break
        if text:
            return text.strip()
    for event in record.get("trace") or []:
        text = str(event.get("user_message") or "").strip()
        if not text:
            continue
        if text.startswith("QUESTION\n"):
            text = text[len("QUESTION\n") :]
        for marker in ("\n\nREGISTERED INPUTS", "\n\nANSWER SCHEMA", "\n\nSTATE"):
            if marker in text:
                text = text.split(marker, 1)[0]
                break
        return text.strip()
    return ""


def plan_from_record(record: dict[str, Any]) -> Any:
    """Return an explicit top-level plan or the Plan action saved in a trace."""

    if record.get("plan"):
        return record.get("plan")
    for event in record.get("trace") or []:
        assistant = event.get("assistant") or {}
        action = assistant.get("action") or {}
        if action.get("tool") == "Plan":
            return action
    return None


def summarize_shared_state(state: dict[str, Any] | None, *, max_text: int) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    artifacts = state.get("artifacts") or {}
    layers = state.get("layers") or {}
    fields = state.get("fields") or {}
    return {
        "artifact_handles": sorted(map(str, artifacts.keys()))[:80] if isinstance(artifacts, dict) else [],
        "layers": compact(layers, max_text=max_text, max_items=24),
        "fields": compact(fields, max_text=max_text, max_items=80),
    }


def build_step_agent_map(record: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for step in record.get("steps") or []:
        if isinstance(step, dict):
            out[str(step.get("id"))] = str(step.get("agent"))
    plan = record.get("plan") or {}
    for step in plan.get("steps") or []:
        if isinstance(step, dict):
            out[str(step.get("id"))] = str(step.get("agent"))
    return out


def collect_tool_calls(record: dict[str, Any], *, max_rows: int, max_text: int) -> list[dict[str, Any]]:
    step_agents = build_step_agent_map(record)
    rows: list[dict[str, Any]] = []
    last_agent = ""
    for event in record.get("trace") or []:
        role = str(event.get("role") or "")
        if role not in {"tool", "user", "runtime"}:
            last_agent = "MAIN" if role == "agent" else role
        # Multi-agent traces label tool events with phase=tool_call. Current
        # single-agent traces use role=tool and intentionally omit phase.
        if role != "tool":
            continue
        payload = event.get("payload") or {}
        produced = payload.get("produced") or {}
        step_id = str(event.get("step_id") or "")
        rows.append(
            {
                "agent": step_agents.get(step_id) or payload.get("agent") or last_agent or "MAIN",
                "step_id": step_id,
                "turn": event.get("turn"),
                "tool": payload.get("tool"),
                "args": compact(payload.get("args") or {}, max_text=max_text, max_items=24),
                "ok": payload.get("ok"),
                "output_text": clip_text(payload.get("text", ""), max_text),
                "produced_fields": compact((produced.get("fields") or {}), max_text=max_text, max_items=36),
                "produced_artifacts": compact((produced.get("artifacts") or []), max_text=max_text, max_items=12),
                "produced_layers": compact((produced.get("layers") or []), max_text=max_text, max_items=12),
            }
        )
        if len(rows) >= max_rows:
            rows.append({"truncated": True, "remaining_tool_calls": "not shown"})
            break
    return rows


def collect_agent_events(record: dict[str, Any], *, max_rows: int, max_text: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in record.get("trace") or []:
        role = str(event.get("role") or "")
        phase = str(event.get("phase") or "")
        if role in {"user", "tool"}:
            continue
        payload = event.get("payload") or {}
        assistant = event.get("assistant") or {}
        if role == "agent" and assistant:
            action = assistant.get("action") or {}
            action_tool = str(action.get("tool") or "")
            rows.append(
                {
                    "role": "MAIN",
                    "phase": (
                        "plan"
                        if action_tool == "Plan"
                        else "complete"
                        if action_tool == "FinalAnswer"
                        else "step"
                    ),
                    "turn": event.get("turn"),
                    "thought": clip_text(assistant.get("thought", ""), max_text),
                    "action": compact(action, max_text=max_text, max_items=18),
                    **(
                        {"error": clip_text(event.get("error"), max_text)}
                        if event.get("error")
                        else {}
                    ),
                }
            )
            if len(rows) >= max_rows:
                rows.append({"truncated": True, "remaining_agent_events": "not shown"})
                break
            continue
        row: dict[str, Any] = {
            "role": role or "MAIN",
            "phase": phase,
            "step_id": event.get("step_id"),
            "turn": event.get("turn"),
        }
        if phase == "plan":
            row["plan"] = compact(payload.get("plan"), max_text=max_text, max_items=18)
        elif phase in {"delegate", "decide"}:
            row["decision"] = compact(payload.get("decision"), max_text=max_text, max_items=12)
            if "contract" in payload:
                row["contract"] = compact(payload.get("contract"), max_text=max_text, max_items=12)
            if "error" in payload:
                row["error"] = clip_text(payload.get("error"), max_text)
        elif phase == "step":
            turn = payload.get("turn") or {}
            row["thought"] = clip_text(turn.get("thought", ""), max_text)
            row["action"] = compact(turn.get("action") or {}, max_text=max_text, max_items=18)
        elif phase == "complete":
            row["completion"] = compact(payload, max_text=max_text, max_items=12)
        else:
            row["payload"] = compact(payload, max_text=max_text, max_items=12)
        rows.append(row)
        if len(rows) >= max_rows:
            rows.append({"truncated": True, "remaining_agent_events": "not shown"})
            break
    return rows


def reference_toolchain(record: dict[str, Any], *, max_text: int) -> Any:
    for key in (
        "reference_toolchain",
        "expected_toolchain",
        "expected_tool_calls",
        "gt_toolchain",
        "gt_chain",
        "toolchain",
        "executions",
    ):
        if key in record and record[key]:
            return compact(record[key], max_text=max_text, max_items=80)
    return None


def build_case_payload(
    record_id: str,
    source_path: Path,
    record: dict[str, Any],
    *,
    forced_mode: str,
    max_tool_rows: int,
    max_agent_events: int,
    max_text: int,
) -> dict[str, Any]:
    case_id = (
        record.get("dataset_id")
        or record.get("dataset_index")
        or record.get("idx")
        or record.get("case_id")
        or record_id
    )
    expected_answer = expected_answer_from_record(record)
    predicted_answer = predicted_answer_from_record(record)
    deterministic_answer = score_answer(expected_answer, predicted_answer)
    mode = mode_from_record(record, forced_mode)
    source = record.get("source")
    if not source and record.get("dataset_id"):
        source = str(record["dataset_id"]).split("/", 1)[0]
    return {
        "case_id": case_id,
        "record_id": record_id,
        "source_file": str(source_path),
        "mode": mode,
        "source": source,
        "question_type": record.get("question_type"),
        "status": record.get("status"),
        "abort_reason": record.get("abort_reason"),
        "question": clip_text(question_from_record(record), max_text * 3),
        "initial_artifacts": compact(record.get("initial_artifacts") or {}, max_text=max_text, max_items=40),
        "expected_final_answer": compact(expected_answer, max_text=max_text, max_items=60),
        "predicted_final_answer": compact(predicted_answer, max_text=max_text, max_items=60),
        "deterministic_answer_evaluation": {
            "correct": bool(deterministic_answer["correct"]),
            "valid_json": bool(deterministic_answer["valid_json"]),
            "scoring_policy": deterministic_answer["scoring_policy"],
            "absolute_tolerance": deterministic_answer["absolute_tolerance"],
            "relative_tolerance": deterministic_answer["relative_tolerance"],
            "diff": compact(
                structured_diff(expected_answer, predicted_answer),
                max_text=max_text,
                max_items=100,
            ),
        },
        "plan": compact(plan_from_record(record), max_text=max_text, max_items=40),
        "steps": compact(record.get("steps") or [], max_text=max_text, max_items=40),
        "agent_events": collect_agent_events(record, max_rows=max_agent_events, max_text=max_text),
        "tool_calls": collect_tool_calls(record, max_rows=max_tool_rows, max_text=max_text),
        "shared_state_summary": summarize_shared_state(record.get("shared_state"), max_text=max_text),
        "reference_toolchain_or_replay_hints": reference_toolchain(record, max_text=max_text),
        "deterministic_metrics": compact(
            {
                "metrics": record.get("metrics"),
                "alignment_metrics": record.get("alignment_metrics"),
                "agent_metrics": record.get("agent_metrics"),
                "sequence_metrics": record.get("sequence_metrics"),
                "failure_buckets": record.get("failure_buckets"),
            },
            max_text=max_text,
            max_items=80,
        ),
        "metric_applicability": {
            "delegation": "not_applicable" if mode == "singleagent" else "applicable",
            "role_scores": ["MAIN"] if mode == "singleagent" else ["ORC", "GA", "PA", "VRA"],
        },
    }


def make_request(
    payload: dict[str, Any],
    *,
    system_prompt: str,
    user_template: str,
) -> dict[str, Any]:
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    user_prompt = user_template.replace("{case_payload_json}", payload_json)
    return {
        "custom_id": str(payload.get("case_id")),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "case_payload": payload,
    }


def call_openai_compatible(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return str(obj["choices"][0]["message"]["content"])


def normalize_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        score = float(value)
    except Exception:
        return None
    if score > 1.0 and score <= 100.0:
        score /= 100.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def canonical_score_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if isinstance(out.get("deterministic_answer_correct"), bool):
        strict = bool(out["deterministic_answer_correct"])
        out["answer_correct"] = strict
        out["strict_correct"] = strict
        out["StrictCorrect"] = 1.0 if strict else 0.0
    for key in ALL_SCORE_KEYS:
        if key in out:
            out[key] = normalize_score(out.get(key))
    out["agent_scores"] = out.get("agent_scores") or {}
    if isinstance(out["agent_scores"], dict):
        for agent, scores in list(out["agent_scores"].items()):
            if not isinstance(scores, dict):
                continue
            for key in set(ALL_SCORE_KEYS + ["turn_count", "tool_call_count"]):
                if key in scores and key not in {"turn_count", "tool_call_count"}:
                    scores[key] = normalize_score(scores.get(key))
    tags = out.get("failure_tags") or []
    if isinstance(tags, str):
        tags = [tags]
    out["failure_tags"] = [tag for tag in tags if tag in FAILURE_TAGS]
    return out


def mean_pct(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [normalize_score(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(100.0 * sum(vals) / len(vals), 4)


def bool_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in rows if isinstance(row.get(key), bool)]
    if not vals:
        return None
    return round(100.0 * sum(1 for v in vals if v) / len(vals), 4)


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean_rows = [canonical_score_row(row) for row in rows if not row.get("judge_error")]
    summary: dict[str, Any] = {
        "cases_scored": len(clean_rows),
        "cases_with_errors": sum(1 for row in rows if row.get("judge_error")),
        "table3": {key: mean_pct(clean_rows, key) for key in TABLE3_KEYS},
        "task_metrics": {key: mean_pct(clean_rows, key) for key in TASK_KEYS},
        "table4": {key: mean_pct(clean_rows, key) for key in TABLE4_KEYS},
        "answer_correct_rate": bool_rate(clean_rows, "answer_correct"),
        "constraint_satisfied_rate": bool_rate(clean_rows, "constraint_satisfied"),
        "execution_success_rate": bool_rate(clean_rows, "execution_success"),
    }
    tag_counter = Counter(tag for row in clean_rows for tag in (row.get("failure_tags") or []))
    summary["failure_tag_counts"] = dict(tag_counter.most_common())
    summary["failure_tag_rates"] = {
        tag: round(100.0 * count / len(clean_rows), 4) if clean_rows else 0.0
        for tag, count in tag_counter.most_common()
    }

    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        scores = row.get("agent_scores") or {}
        if isinstance(scores, dict):
            for agent, agent_row in scores.items():
                if isinstance(agent_row, dict):
                    by_agent[str(agent)].append(agent_row)
    summary["per_agent"] = {
        agent: {
            "n": len(items),
            **{key: mean_pct(items, key) for key in TABLE3_KEYS + ["PlanAcc", "DelegAcc"]},
            "mean_turn_count": round(
                sum(float(x.get("turn_count") or 0) for x in items) / len(items), 4
            )
            if items
            else 0.0,
            "mean_tool_call_count": round(
                sum(float(x.get("tool_call_count") or 0) for x in items) / len(items), 4
            )
            if items
            else 0.0,
        }
        for agent, items in sorted(by_agent.items())
    }

    def group_by(meta_key: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in clean_rows:
            label = str(row.get(meta_key) or "unknown")
            buckets[label].append(row)
        return {
            label: {
                "n": len(items),
                "TSR": mean_pct(items, "TSR"),
                "Ans": mean_pct(items, "Ans"),
                "Inst": mean_pct(items, "Inst"),
                "Tool": mean_pct(items, "Tool"),
                "ArgN": mean_pct(items, "ArgN"),
                "ArgV": mean_pct(items, "ArgV"),
                "Summ": mean_pct(items, "Summ"),
            }
            for label, items in sorted(buckets.items())
        }

    summary["by_source"] = group_by("source")
    summary["by_question_type"] = group_by("question_type")
    return summary


def score_requests(args: argparse.Namespace, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.openai_base_url or os.environ.get("OPENAI_BASE_URL", "")
    model = args.openai_model or os.environ.get("OPENAI_MODEL", "")
    if not base_url or not model:
        raise SystemExit("openai backend requires OPENAI_BASE_URL and OPENAI_MODEL, or matching CLI flags")
    for idx, req in enumerate(requests, start=1):
        try:
            content = call_openai_compatible(
                req["messages"],
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout_s=args.timeout_s,
            )
            parsed = extract_first_json_object(content)
            if parsed is None:
                parsed = {
                    "case_id": req["custom_id"],
                    "judge_error": "judge_returned_non_json",
                    "raw_response": clip_text(content, 2000),
                }
            parsed.setdefault("case_id", req["custom_id"])
            payload = req.get("case_payload") or {}
            parsed.setdefault("source", payload.get("source"))
            parsed.setdefault("question_type", payload.get("question_type"))
            parsed.setdefault("mode", payload.get("mode"))
            deterministic = payload.get("deterministic_answer_evaluation") or {}
            if isinstance(deterministic.get("correct"), bool):
                strict = bool(deterministic["correct"])
                parsed["judge_answer_correct_raw"] = parsed.get("answer_correct")
                parsed["deterministic_answer_correct"] = strict
                parsed["deterministic_answer_diff"] = deterministic.get("diff") or []
                parsed["answer_correct"] = strict
                parsed["strict_correct"] = strict
                parsed["StrictCorrect"] = 1.0 if strict else 0.0
                if not strict:
                    tags = parsed.get("failure_tags") or []
                    if isinstance(tags, str):
                        tags = [tags]
                    if "SynthErr" not in tags:
                        tags.append("SynthErr")
                    parsed["failure_tags"] = tags
            scores.append(parsed)
        except (urllib.error.URLError, TimeoutError, Exception) as exc:
            scores.append({"case_id": req["custom_id"], "judge_error": repr(exc)})
        if args.sleep_s and idx < len(requests):
            time.sleep(args.sleep_s)
    return scores


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_metrics_md(path: Path, summary: dict[str, Any], *, scores_path: Path) -> None:
    table3 = summary.get("table3") or {}
    table4 = summary.get("table4") or {}
    task = summary.get("task_metrics") or {}
    lines = [
        "# GeoDisaster Conversation Judge Metrics",
        "",
        f"- Scored cases: `{summary.get('cases_scored', 0)}`",
        f"- Judge errors: `{summary.get('cases_with_errors', 0)}`",
        f"- Scores JSONL: `{scores_path}`",
        "",
        "## Table 3 Style Metrics",
        "",
        markdown_table(["Inst.", "Tool.", "ArgN.", "ArgV.", "Summ."], [[table3.get(k) for k in TABLE3_KEYS]]),
        "",
        "## Task Metrics",
        "",
        markdown_table(
            ["TSR", "Ans.", "Strict Correct", "CSR", "ESR", "PlanAcc", "DelegAcc", "Answer Correct", "Constraint OK", "Execution OK"],
            [[
                task.get("TSR"),
                task.get("Ans"),
                task.get("StrictCorrect"),
                task.get("CSR"),
                task.get("ESR"),
                task.get("PlanAcc"),
                task.get("DelegAcc"),
                summary.get("answer_correct_rate"),
                summary.get("constraint_satisfied_rate"),
                summary.get("execution_success_rate"),
            ]],
        ),
        "",
        "## Table 4 Style Metrics",
        "",
        markdown_table(
            ["Per.", "Op.", "Logic.", "GIS.", "AnyOr.", "SameO.", "Uni.", "Ans.", "Gen."],
            [[table4.get(k) for k in TABLE4_KEYS]],
        ),
        "",
        "## Per-Agent Metrics",
        "",
        markdown_table(
            ["Agent", "N", "Inst.", "Tool.", "ArgN.", "ArgV.", "Summ.", "PlanAcc", "DelegAcc", "Turns", "Tools"],
            [
                [
                    agent,
                    item.get("n"),
                    item.get("Inst"),
                    item.get("Tool"),
                    item.get("ArgN"),
                    item.get("ArgV"),
                    item.get("Summ"),
                    item.get("PlanAcc"),
                    item.get("DelegAcc"),
                    item.get("mean_turn_count"),
                    item.get("mean_tool_call_count"),
                ]
                for agent, item in (summary.get("per_agent") or {}).items()
            ],
        ),
        "",
        "## Failure Tags",
        "",
        json.dumps(summary.get("failure_tag_rates") or {}, indent=2, ensure_ascii=False),
        "",
        "## Source Breakdown",
        "",
        markdown_table(
            ["Source", "N", "TSR", "Ans.", "Inst.", "Tool.", "ArgN.", "ArgV.", "Summ."],
            [
                [
                    label,
                    item.get("n"),
                    item.get("TSR"),
                    item.get("Ans"),
                    item.get("Inst"),
                    item.get("Tool"),
                    item.get("ArgN"),
                    item.get("ArgV"),
                    item.get("Summ"),
                ]
                for label, item in (summary.get("by_source") or {}).items()
            ],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dry_run_md(path: Path, *, requests_path: Path, payloads_path: Path, prompt_path: Path) -> None:
    lines = [
        "# GeoDisaster Judge Dry Run",
        "",
        "No judge model was called. The judge requests and compact case payloads are ready.",
        "",
        f"- Prompt: `{prompt_path}`",
        f"- Requests: `{requests_path}`",
        f"- Case payloads: `{payloads_path}`",
        "",
        "To call an OpenAI-compatible judge:",
        "",
        "```bash",
        "OPENAI_BASE_URL=http://localhost:8000/v1 \\",
        "OPENAI_API_KEY=dummy \\",
        "OPENAI_MODEL=your-judge-model \\",
        "python3 scripts/final_dataset/judge_geodisaster_conversations.py \\",
        "  --input /path/to/conversation_dir \\",
        "  --output_dir /path/to/judge_out \\",
        "  --backend openai",
        "```",
        "",
        "To aggregate an already-produced `judge_scores.jsonl`:",
        "",
        "```bash",
        "python3 scripts/final_dataset/judge_geodisaster_conversations.py \\",
        "  --scores /path/to/judge_scores.jsonl \\",
        "  --output_dir /path/to/judge_out",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="inputs", action="append", default=[], help="Conversation JSON, JSONL, or directory. Repeatable.")
    parser.add_argument("--scores", type=Path, help="Existing judge_scores.jsonl to aggregate without rebuilding requests.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prompt_md", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--backend", choices=["dry_run", "openai"], default="dry_run")
    parser.add_argument("--mode", choices=["auto", "multiagent", "singleagent"], default="auto")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_tool_rows", type=int, default=80)
    parser.add_argument("--max_agent_events", type=int, default=120)
    parser.add_argument("--max_text_chars", type=int, default=700)
    parser.add_argument("--openai_base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--openai_model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--timeout_s", type=int, default=180)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.scores:
        rows = [canonical_score_row(row) for row in read_jsonl(args.scores)]
        summary = aggregate_scores(rows)
        write_json(args.output_dir / "judge_summary.json", summary)
        write_metrics_md(args.output_dir / "judge_metrics.md", summary, scores_path=args.scores)
        print(f"Wrote aggregate metrics to {args.output_dir}")
        return 0

    if not args.inputs:
        raise SystemExit("Provide at least one --input, or --scores for aggregation-only mode.")

    system_prompt, user_template = load_prompt(args.prompt_md)
    records = collect_records([Path(p) for p in args.inputs], recursive=args.recursive)
    if args.limit:
        records = records[: args.limit]
    payloads: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for record_id, source_path, record in records:
        payload = build_case_payload(
            record_id,
            source_path,
            record,
            forced_mode=args.mode,
            max_tool_rows=args.max_tool_rows,
            max_agent_events=args.max_agent_events,
            max_text=args.max_text_chars,
        )
        payloads.append(payload)
        requests.append(make_request(payload, system_prompt=system_prompt, user_template=user_template))

    payloads_path = args.output_dir / "judge_case_payloads.jsonl"
    requests_path = args.output_dir / "judge_requests.jsonl"
    write_jsonl(payloads_path, payloads)
    write_jsonl(requests_path, requests)
    write_json(
        args.output_dir / "judge_manifest.json",
        {
            "input_count": len(args.inputs),
            "case_count": len(requests),
            "backend": args.backend,
            "prompt_md": str(args.prompt_md),
            "requests_path": str(requests_path),
            "payloads_path": str(payloads_path),
        },
    )

    if args.backend == "dry_run":
        write_dry_run_md(
            args.output_dir / "README.md",
            requests_path=requests_path,
            payloads_path=payloads_path,
            prompt_path=args.prompt_md,
        )
        print(f"Built {len(requests)} judge requests in {args.output_dir}")
        return 0

    scores = score_requests(args, requests)
    scores = [canonical_score_row(row) for row in scores]
    scores_path = args.output_dir / "judge_scores.jsonl"
    write_jsonl(scores_path, scores)
    summary = aggregate_scores(scores)
    write_json(args.output_dir / "judge_summary.json", summary)
    write_metrics_md(args.output_dir / "judge_metrics.md", summary, scores_path=scores_path)
    print(f"Judged {len(scores)} cases; wrote metrics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
