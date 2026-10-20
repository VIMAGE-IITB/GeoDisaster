from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent  # code/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluation.structured_answer import score_answer, structured_diff  # noqa: E402

FAILURE_TAGS = [
    "PlanErr", "AgentErr", "ToolErr", "ArgErr", "FormatErr", "TermErr",
    "LoopErr", "ReplanErr", "SynthErr",
]
FAILURE_TAG_ALIASES = {
    "planningfailure": "PlanErr",
    "orcplaninvalidity": "PlanErr",
    "agenterr": "AgentErr",
    "wrongdelegation": "AgentErr",
    "interagentinconsistency": "AgentErr",
    "toolselectionfailure": "ToolErr",
    "toolfailure": "ToolErr",
    "toolerr": "ToolErr",
    "argumentgroundingfailure": "ArgErr",
    "argerr": "ArgErr",
    "formaterr": "FormatErr",
    "jsonerr": "FormatErr",
    "prematuretermination": "TermErr",
    "termerr": "TermErr",
    "maxturnfailure": "LoopErr",
    "redundantordegenerateloops": "LoopErr",
    "looperr": "LoopErr",
    "replanerr": "ReplanErr",
}
SCORE_KEYS = [
    "answer_score",
    "constraint_score",
    "execution_score",
    "planning_score",
    "delegation_score",
    "tool_use_score",
]
DEFAULT_LOCAL_QWEN = os.environ.get("GEODISASTER_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SYSTEM_PROMPT = """You are an evaluation assistant responsible for fair, strict evaluation of a geospatial tool-use agent.

You are given a single case containing:
- a Question
- a Ground Truth answer and, when available, the Ground Truth multi-agent plan and toolchain
- a Predicted answer produced by the evaluated agent
- the evaluated agent's plan, tool calls, status, and failure buckets

Judge the prediction with both answer correctness and trajectory quality:
1. Answer correctness:
   - For numeric answers, accept values within +/-10% of the ground-truth value unless the question demands exactness.
   - For bounding boxes, accept IoU >= 0.5.
   - For categorical/descriptive outputs, use semantic equivalence.
   - Ignore extraneous numbers that do not directly answer the question.
   - A bare placeholder such as "<image>" is correct only if the trace actually produced the requested relevant visual artifact.
   - If the prediction omits all required answer elements, fabricates unsupported entities/relationships, or says unknown/no data for a required answer, score it as incorrect.
2. Constraint and execution correctness:
   - Check whether the predicted trajectory follows key constraints in the question and GT toolchain.
   - Penalize wrong tool names, wrong arguments, missing required tools, invalid tool outputs, max-turn failures, repeated degenerate calls, and premature termination.
   - Do not require an exact word-for-word plan when a different valid plan reaches the same required result.
3. Planning and delegation:
   - Credit plans that assign the right specialists: ORC coordinates, GA handles geospatial/vector/map work, VRA handles image/raster/vision work, and PA handles arithmetic/plot/synthesis.
   - Penalize generic subtasks, wrong delegation, missing data dependencies, and re-planning that does not fix a failure.
   - Delegation is not applicable to single-agent runs and must be returned as null there.

Return strictly one JSON object and nothing else."""

USER_PROMPT_TEMPLATE = """Evaluate this case and return the JSON object described below.

Required output schema:
{
  "case_id": <int>,
  "answer_score": <float 0.0-1.0>,
  "answer_correct": <bool>,
  "constraint_score": <float 0.0-1.0>,
  "constraint_satisfied": <bool>,
  "execution_score": <float 0.0-1.0>,
  "execution_success": <bool>,
  "planning_score": <float 0.0-1.0>,
  "delegation_score": <float 0.0-1.0|null>,
  "tool_use_score": <float 0.0-1.0>,
  "failure_tags": [<zero or more of: PlanErr, AgentErr, ToolErr, ArgErr, FormatErr, TermErr, LoopErr, ReplanErr>],
  "justification": "<1-2 concise sentences comparing GT and prediction>"
}

Scoring guidance:
- answer_correct should usually be true when answer_score >= 0.80.
- constraint_satisfied should usually be true when constraint_score >= 0.80.
- execution_success should usually be true when execution_score >= 0.80.
- Use failure_tags to identify the dominant reasons for lost score.

Case payload:
{payload}
"""


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
                    parsed = json.loads(raw[start : idx + 1])
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
    return None


def _clip_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _load_teacher_dir(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for file_path in sorted(path.glob("multiagent_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        records[int(file_path.stem.split("_")[-1])] = json.loads(file_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"No multiagent_*.json files found in {path}")
    return records


def _load_eval_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for file_path in sorted(path.glob("conversation_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        record = json.loads(file_path.read_text(encoding="utf-8"))
        # Current single-agent filenames are zero-based and are output positions,
        # not canonical dataset indices. Prefer the stable numeric dataset index.
        raw_id = record.get("idx", record.get("dataset_index", record.get("case_id")))
        sample_no = int(raw_id) if isinstance(raw_id, int) else int(file_path.stem.split("_")[-1])
        if sample_no in records:
            raise ValueError(f"Duplicate numeric case id {sample_no} in {path}")
        records[sample_no] = record
    if not records:
        raise ValueError(f"No conversation_*.json files found in {path}")
    return records


def _teacher_tool_rows(record: dict[str, Any], *, max_rows: int, max_text_chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for execution in record.get("executions") or []:
        agent = str(execution.get("agent", ""))
        step_id = str(execution.get("step_id", ""))
        for turn in execution.get("turns") or []:
            action = turn.get("action") or {}
            tool = str(action.get("tool", ""))
            if not tool or tool == "TaskComplete":
                continue
            tool_output = turn.get("tool_output") or {}
            produced = tool_output.get("produced") or {}
            rows.append(
                {
                    "agent": agent,
                    "step_id": step_id,
                    "tool": tool,
                    "args": action.get("args") or {},
                    "ok": bool(tool_output.get("ok", True)),
                    "output_text": _clip_text(tool_output.get("text", ""), max_text_chars),
                    "produced_fields": produced.get("fields") or {},
                    "produced_layers": produced.get("layers") or [],
                    "produced_artifacts": produced.get("artifacts") or [],
                }
            )
            if len(rows) >= max_rows:
                rows.append({"truncated": True, "remaining_tool_calls": "not shown"})
                return rows
    return rows


def _actual_tool_rows(record: dict[str, Any], *, max_rows: int, max_text_chars: int) -> list[dict[str, Any]]:
    step_agents = {str(step.get("id")): str(step.get("agent")) for step in (record.get("steps") or [])}
    step_agents.update({str(step.get("id")): str(step.get("agent")) for step in ((record.get("plan") or {}).get("steps") or [])})
    rows: list[dict[str, Any]] = []
    for event in record.get("trace") or []:
        # Multi-agent traces include phase=tool_call; single-agent traces use
        # role=tool with no phase. The role is the common executable contract.
        if event.get("role") != "tool":
            continue
        payload = event.get("payload") or {}
        produced = payload.get("produced") or {}
        step_id = str(event.get("step_id") or "")
        rows.append(
            {
                "agent": step_agents.get(step_id, "") or "MAIN",
                "step_id": step_id,
                "turn": event.get("turn"),
                "tool": payload.get("tool", ""),
                "args": payload.get("args") or {},
                "ok": bool(payload.get("ok", False)),
                "output_text": _clip_text(payload.get("text", ""), max_text_chars),
                "produced_fields": produced.get("fields") or {},
                "produced_layers": produced.get("layers") or [],
                "produced_artifacts": produced.get("artifacts") or [],
            }
        )
        if len(rows) >= max_rows:
            rows.append({"truncated": True, "remaining_tool_calls": "not shown"})
            return rows
    return rows


def _compact_plan(record: dict[str, Any]) -> dict[str, Any]:
    plan = record.get("plan") or {}
    if not plan:
        for event in record.get("trace") or []:
            assistant = event.get("assistant") or {}
            action = assistant.get("action") or {}
            if action.get("tool") == "Plan":
                plan = action
                break
    return {
        "goal": plan.get("goal", ""),
        "steps": [
            {
                "id": step.get("id"),
                "agent": step.get("agent"),
                "subtask": step.get("subtask"),
                "needs": step.get("needs") or [],
            }
            for step in plan.get("steps") or []
        ],
    }


def _compact_completions(record: dict[str, Any], max_text_chars: int) -> list[dict[str, Any]]:
    out = []
    for event in record.get("trace") or []:
        if event.get("role") not in {"GA", "PA", "VRA"} or event.get("phase") != "complete":
            continue
        payload = event.get("payload") or {}
        out.append(
            {
                "agent": event.get("role"),
                "step_id": event.get("step_id"),
                "status": payload.get("status"),
                "summary": _clip_text(payload.get("summary", ""), max_text_chars),
            }
        )
    return out


def _build_case_payload(
    case_id: int,
    gt: dict[str, Any],
    actual: dict[str, Any],
    *,
    max_tool_rows: int,
    max_text_chars: int,
) -> dict[str, Any]:
    expected_answer = gt.get(
        "expected_final_answer",
        gt.get("gold", gt.get("final_answer", actual.get("gold", actual.get("label", "")))),
    )
    predicted_answer = actual.get("final_answer", actual.get("prediction", ""))
    mode = "multiagent" if any(
        event.get("role") in {"ORC", "GA", "PA", "VRA"}
        for event in (actual.get("trace") or [])
        if isinstance(event, dict)
    ) else "singleagent"
    question = gt.get("question") or actual.get("question", "")
    if isinstance(question, dict):
        question = question.get("text") or question.get("question") or ""
    if not question:
        for message in actual.get("messages") or []:
            if message.get("role") != "user":
                continue
            question = str(message.get("content") or "")
            if question.startswith("QUESTION\n"):
                question = question[len("QUESTION\n") :]
            question = question.split("\n\nREGISTERED INPUTS", 1)[0]
            question = question.split("\n\nANSWER SCHEMA", 1)[0]
            break
    if not question:
        for event in actual.get("trace") or []:
            question = str(event.get("user_message") or "")
            if not question:
                continue
            if question.startswith("QUESTION\n"):
                question = question[len("QUESTION\n") :]
            question = question.split("\n\nREGISTERED INPUTS", 1)[0]
            question = question.split("\n\nANSWER SCHEMA", 1)[0]
            break
    deterministic = score_answer(expected_answer, predicted_answer)
    return {
        "case_id": case_id,
        "dataset_id": actual.get("dataset_id", gt.get("dataset_id")),
        "mode": mode,
        "dataset_idx": actual.get("idx", gt.get("idx")),
        "question": question,
        "images": gt.get("images") or actual.get("images") or [],
        "ground_truth": {
            "final_answer": expected_answer,
            "plan": _compact_plan(gt),
            "toolchain": _teacher_tool_rows(gt, max_rows=max_tool_rows, max_text_chars=max_text_chars),
        },
        "prediction": {
            "status": actual.get("status"),
            "abort_reason": actual.get("abort_reason", ""),
            "final_answer": predicted_answer,
            "plan": _compact_plan(actual),
            "tool_calls": _actual_tool_rows(actual, max_rows=max_tool_rows, max_text_chars=max_text_chars),
            "agent_completions": _compact_completions(actual, max_text_chars),
            "failure_buckets": actual.get("failure_buckets") or {},
        },
        "deterministic_answer_evaluation": {
            "correct": bool(deterministic["correct"]),
            "valid_json": bool(deterministic["valid_json"]),
            "scoring_policy": deterministic["scoring_policy"],
            "absolute_tolerance": deterministic["absolute_tolerance"],
            "relative_tolerance": deterministic["relative_tolerance"],
            "diff": structured_diff(expected_answer, predicted_answer),
        },
    }


def _apply_deterministic_answer_gate(
    row: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Make strict native-JSON scoring authoritative for answer correctness."""

    deterministic = payload.get("deterministic_answer_evaluation") or {}
    if not isinstance(deterministic.get("correct"), bool):
        return row
    strict = bool(deterministic["correct"])
    row["judge_answer_correct_raw"] = row.get("answer_correct")
    row["deterministic_answer_correct"] = strict
    row["deterministic_answer_diff"] = deterministic.get("diff") or []
    row["answer_correct"] = strict
    if not strict:
        tags = list(row.get("failure_tags") or [])
        if "SynthErr" not in tags:
            tags.append("SynthErr")
        row["failure_tags"] = tags
    return row


def _messages_for_case(payload: dict[str, Any], system_prompt: str, user_prompt_template: str) -> list[dict[str, str]]:
    payload_text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if "{payload}" in user_prompt_template:
        user_content = user_prompt_template.replace("{payload}", payload_text)
    else:
        user_content = user_prompt_template.replace(
            "<JSON containing question, ground_truth, prediction, plans, tool calls, status, and failure buckets>",
            payload_text,
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, number))


def _coerce_bool(value: Any, fallback_score: float, threshold: float) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return fallback_score >= threshold


def _coerce_judgment(case_id: int, parsed: dict[str, Any], raw_response: str, threshold: float) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": int(parsed.get("case_id", case_id))}
    for key in SCORE_KEYS:
        row[key] = None if key == "delegation_score" and parsed.get(key) is None else _clamp01(parsed.get(key, 0.0))
    row["answer_correct"] = _coerce_bool(parsed.get("answer_correct"), row["answer_score"], threshold)
    row["constraint_satisfied"] = _coerce_bool(parsed.get("constraint_satisfied"), row["constraint_score"], threshold)
    row["execution_success"] = _coerce_bool(parsed.get("execution_success"), row["execution_score"], threshold)
    raw_tags = parsed.get("failure_tags") or []
    tags = []
    for tag in raw_tags:
        text = str(tag)
        canonical = text if text in FAILURE_TAGS else FAILURE_TAG_ALIASES.get(re.sub(r"[^a-z0-9]", "", text.lower()))
        if canonical and canonical not in tags:
            tags.append(canonical)
    row["failure_tags"] = tags
    row["justification"] = _clip_text(parsed.get("justification", ""), 600)
    row["raw_response"] = raw_response
    return row


def _call_judge(messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
    )
    return str(response.choices[0].message.content or "")


def _install_local_qwen_judge(args: argparse.Namespace):
    from evaluation.eval_live import _install_local_qwen_backend, _maybe_init_accelerate
    from multi_agent import llm as llm_mod

    _maybe_init_accelerate()
    os.environ["MODEL_BACKEND"] = "local_qwen"
    _install_local_qwen_backend(
        pretrained=args.model_path,
        max_new_tokens=args.max_new_tokens,
        adapter_root="",
    )

    def _local_call(messages: list[dict[str, str]]) -> str:
        return str(llm_mod._BACKEND_CALLS["local_qwen"](messages))

    return _local_call


def _fenced_block_after(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        return ""
    fence_start = text.find("```", start)
    if fence_start < 0:
        return ""
    line_end = text.find("\n", fence_start)
    if line_end < 0:
        return ""
    fence_end = text.find("```", line_end + 1)
    if fence_end < 0:
        return ""
    return text[line_end + 1 : fence_end].strip()


def _load_prompt_templates(path: str) -> tuple[str, str]:
    if not path:
        return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    prompt_path = Path(path).resolve()
    text = prompt_path.read_text(encoding="utf-8")
    system_prompt = _fenced_block_after(text, "System prompt:") or SYSTEM_PROMPT
    user_prompt = _fenced_block_after(text, "User prompt template:") or USER_PROMPT_TEMPLATE
    return system_prompt, user_prompt


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, bool))]
    return float(sum(values) / len(values)) if values else 0.0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tag_counts = Counter(tag for row in rows for tag in row.get("failure_tags", []))
    n = len(rows)
    strict_successes = [
        row
        for row in rows
        if row.get("answer_correct") and row.get("constraint_satisfied") and row.get("execution_success")
    ]
    return {
        "n_cases": n,
        "judge_answer_score_avg": _mean(rows, "answer_score"),
        "judge_answer_correct_rate": _mean(rows, "answer_correct"),
        "judge_constraint_score_avg": _mean(rows, "constraint_score"),
        "judge_constraint_satisfied_rate": _mean(rows, "constraint_satisfied"),
        "judge_execution_score_avg": _mean(rows, "execution_score"),
        "judge_execution_success_rate": _mean(rows, "execution_success"),
        "judge_planning_score_avg": _mean(rows, "planning_score"),
        "judge_delegation_score_avg": _mean(rows, "delegation_score"),
        "judge_tool_use_score_avg": _mean(rows, "tool_use_score"),
        "judge_strict_tsr_rate": len(strict_successes) / n if n else 0.0,
        "failure_tag_counts": dict(sorted(tag_counts.items())),
        "failure_tag_rates": {tag: count / n for tag, count in sorted(tag_counts.items())} if n else {},
    }


def _existing_case_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload.get("case_id"), int):
            out.add(int(payload["case_id"]))
    return out


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_eval_dir(
    eval_dir: Path,
    teacher_records: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    judge_call,
    system_prompt: str,
    user_prompt_template: str,
) -> dict[str, Any]:
    eval_records = _load_eval_records(eval_dir)
    case_ids = [
        case_id
        for case_id in sorted(eval_records)
        if (not teacher_records or case_id in teacher_records) and case_id >= args.start_case
    ]
    if args.max_cases:
        case_ids = case_ids[: args.max_cases]

    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else eval_dir / "llm_judge_scores.jsonl"
    output_summary = Path(args.output_summary).resolve() if args.output_summary else eval_dir / "llm_judge_summary.json"
    request_jsonl = Path(args.request_jsonl).resolve() if args.request_jsonl else eval_dir / "llm_judge_requests.jsonl"

    if args.dry_run:
        request_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with request_jsonl.open("w", encoding="utf-8") as f:
            for case_id in case_ids:
                payload = _build_case_payload(
                    case_id,
                    teacher_records.get(case_id, eval_records[case_id]),
                    eval_records[case_id],
                    max_tool_rows=args.max_tool_rows,
                    max_text_chars=args.max_text_chars,
                )
                f.write(
                    json.dumps(
                        {"case_id": case_id, "messages": _messages_for_case(payload, system_prompt, user_prompt_template)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return {"mode": "dry_run", "eval_dir": str(eval_dir), "request_jsonl": str(request_jsonl), "n_cases": len(case_ids)}

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen = _existing_case_ids(output_jsonl) if args.resume else set()
    with output_jsonl.open("a" if args.resume else "w", encoding="utf-8") as f:
        for case_id in case_ids:
            if case_id in seen:
                continue
            payload = _build_case_payload(
                case_id,
                teacher_records.get(case_id, eval_records[case_id]),
                eval_records[case_id],
                max_tool_rows=args.max_tool_rows,
                max_text_chars=args.max_text_chars,
            )
            messages = _messages_for_case(payload, system_prompt, user_prompt_template)
            started = time.time()
            raw_response = judge_call(messages)
            parsed = _extract_first_json_object(raw_response)
            if parsed is None:
                row = {
                    "case_id": case_id,
                    "error": "could_not_parse_json",
                    "raw_response": raw_response,
                    "latency_s": round(time.time() - started, 3),
                }
            else:
                row = _coerce_judgment(case_id, parsed, raw_response, args.score_threshold)
                row = _apply_deterministic_answer_gate(row, payload)
                row["mode"] = payload.get("mode")
                if payload.get("mode") == "singleagent":
                    row["delegation_score"] = None
                row["latency_s"] = round(time.time() - started, 3)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()

    rows = [row for row in _read_rows(output_jsonl) if not row.get("error")]
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(_summarize(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"eval_dir": str(eval_dir), "output_jsonl": str(output_jsonl), "output_summary": str(output_summary), "n_scored": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or prepare an LLM-as-judge evaluation over model eval outputs.")
    parser.add_argument("--eval_dir", action="append", required=True)
    parser.add_argument(
        "--teacher_dir",
        default="",
        help="Optional multiagent_*.json reference directory. Single-agent outputs carry gold answers directly.",
    )
    parser.add_argument("--output_jsonl", default="", help="Defaults to eval_dir/llm_judge_scores.jsonl")
    parser.add_argument("--output_summary", default="", help="Defaults to eval_dir/llm_judge_summary.json")
    parser.add_argument("--request_jsonl", default="", help="Defaults to eval_dir/llm_judge_requests.jsonl in --dry_run mode")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model_backend", choices=["openai", "local_qwen"], default="openai")
    parser.add_argument("--model_path", default=DEFAULT_LOCAL_QWEN)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--judge_prompt_md", default="", help="Optional markdown override; empty uses the built-in judge prompt.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--start_case", type=int, default=0)
    parser.add_argument("--max_tool_rows", type=int, default=40)
    parser.add_argument("--max_text_chars", type=int, default=900)
    parser.add_argument("--score_threshold", type=float, default=0.8)
    parser.add_argument("--dry_run", action="store_true", help="Write judge request JSONL instead of calling a model.")
    parser.add_argument("--resume", action="store_true", help="Skip cases already present in output_jsonl.")
    args = parser.parse_args()

    teacher_records = _load_teacher_dir(Path(args.teacher_dir).resolve()) if args.teacher_dir else {}
    system_prompt, user_prompt_template = _load_prompt_templates(args.judge_prompt_md)

    if args.output_jsonl and len(args.eval_dir) > 1:
        raise SystemExit("--output_jsonl can only be used with one --eval_dir")
    if args.output_summary and len(args.eval_dir) > 1:
        raise SystemExit("--output_summary can only be used with one --eval_dir")
    if args.request_jsonl and len(args.eval_dir) > 1:
        raise SystemExit("--request_jsonl can only be used with one --eval_dir")

    judge_call = None
    if not args.dry_run and args.model_backend == "local_qwen":
        judge_call = _install_local_qwen_judge(args)
    elif not args.dry_run:
        if not args.model:
            raise SystemExit("Pass --model or set OPENAI_MODEL.")
        judge_call = lambda messages: _call_judge(messages, args)

    outputs = []
    for eval_dir_arg in args.eval_dir:
        outputs.append(
            _run_eval_dir(
                Path(eval_dir_arg).resolve(),
                teacher_records,
                args,
                judge_call,
                system_prompt,
                user_prompt_template,
            )
        )
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
