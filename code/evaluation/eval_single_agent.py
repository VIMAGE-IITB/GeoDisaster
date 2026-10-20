#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any



ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MAS_AGENT_PROFILE", "geodisaster")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.structured_answer import score_answer as _score_answer  # noqa: E402

from multi_agent import llm as llm_mod  # noqa: E402
from multi_agent.answer_contract import validate_explicit_answer  # noqa: E402
from multi_agent.llm import LLMTransportError, LLMValidationError  # noqa: E402
from single_agent.prompts import (  # noqa: E402
    SINGLE_AGENT_SYSTEM,
    build_initial_user_message,
    build_runtime_user_message,
)
from single_agent.runtime import (  # noqa: E402
    absorb_tool_result,
    call_single_tool,
    clear_geodisaster_context,
    install_geodisaster_context,
    new_single_state,
    render_memory,
    resolve_answer_ref,
)
from single_agent.schemas import SingleAgentTurn, SingleFinalAnswer, SinglePlan, SingleToolCall  # noqa: E402


DEFAULT_DATASET_ROOT = Path(
    os.environ.get("GEODISASTER_DATASET_ROOT", str(ROOT.parent / "geodisaster-agentic-2921"))
).expanduser()
DEFAULT_TEST = DEFAULT_DATASET_ROOT / "data" / "single_agent" / "test_reference_conversations.jsonl"

OPENAI_CALL_LOG: list[dict[str, Any]] = []

OPENAI_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    # Standard API token prices. Keep this conservative: cached-input discounts
    # are not applied here because not every endpoint returns cached-token detail.
    "o4-mini": (1.10, 4.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.5": (5.00, 30.00),
}


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def _resolve_core_jsonl(dataset: Path, explicit: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    # Public layout: data/single_agent/<file> and data/core/<canonical>.
    if dataset.parent.name == "single_agent":
        candidates.append(
            dataset.parent.parent / "core" / "geodisaster_2921_source_with_splits.jsonl"
        )
    dataset_root = os.environ.get("GEODISASTER_DATASET_ROOT", "").strip()
    if dataset_root:
        candidates.append(
            Path(dataset_root) / "data" / "core" / "geodisaster_2921_source_with_splits.jsonl"
        )
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.is_file():
            return path
    return None


def _hydrate_from_core(
    rows: list[dict[str, Any]], core_jsonl: Path | None
) -> list[dict[str, Any]]:
    """Join normalized dialogue rows to canonical task inputs by stable id."""

    needs_core = any(
        not isinstance(row.get("initial_artifacts"), dict)
        or "expected_final_answer" not in row
        or not row.get("question")
        for row in rows
    )
    if not needs_core:
        return rows
    if core_jsonl is None:
        raise ValueError(
            "single-agent rows require their canonical core file; pass --core_jsonl "
            "or use the public data/single_agent + data/core directory layout"
        )
    canonical = {
        str(row.get("dataset_id") or row.get("id") or ""): row
        for row in load_jsonl(core_jsonl)
    }
    hydrated: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row.get("dataset_id") or row.get("id") or "")
        core = canonical.get(record_id)
        if core is None:
            raise ValueError(f"single-agent id is absent from canonical core: {record_id!r}")
        merged = dict(core)
        merged.update(row)
        # The normalized dialogue export deliberately omits these fields, so
        # their sole authority is the canonical core row.
        for key in (
            "question",
            "raw_question",
            "initial_artifacts",
            "available_initial_handles",
            "answer_schema",
            "expected_final_answer",
            "gt_chain",
        ):
            if key in core:
                merged[key] = core[key]
        hydrated.append(merged)
    return hydrated


def _teacher_stub_responses(record: dict[str, Any]) -> list[str]:
    """Return published single-agent decisions for an explicit infra test."""

    responses = [
        str(message.get("content") or "")
        for message in (record.get("messages") or [])
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not responses:
        raise ValueError(
            f"record {record.get('dataset_id') or record.get('id')!r} has no teacher responses"
        )
    return responses


def _single_failure_category(
    status: str,
    *,
    answer_correct: bool,
    model_error: BaseException | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> str:
    """Separate model failures from runtime and answer failures."""

    if status == "DONE":
        return "success" if answer_correct else "answer_mismatch"
    if status == "MAX_TURNS":
        return "agent_max_turns"
    failed_tool_text = " ".join(
        str((event.get("payload") or {}).get("text") or "")
        for event in (trace or [])
        if event.get("role") == "tool" and (event.get("payload") or {}).get("ok") is False
    ).lower()
    if "no declared sentinel-1 raster is available" in failed_tool_text:
        return "required_asset_unavailable"
    if failed_tool_text:
        return "tool_or_worker_error"
    if isinstance(model_error, LLMValidationError):
        return "model_schema_error"
    if isinstance(model_error, LLMTransportError):
        return "model_transport_error"
    return "agent_runtime_error"


def _new_session_id(record: dict[str, Any]) -> str:
    """Unique transport session; dataset ids are not reusable state ids."""

    return f"single_{record.get('idx', 'case')}_{uuid.uuid4().hex[:12]}"


def _usage_value(obj: Any, key: str, default: int = 0) -> int:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return int(obj.get(key) or default)
    return int(getattr(obj, key, default) or default)


def _estimate_openai_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = OPENAI_PRICE_PER_1M.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens / 1_000_000.0) * input_price + (output_tokens / 1_000_000.0) * output_price


def _uses_modern_completion_param(model: str) -> bool:
    m = model.lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5"))


def _chat_completion_with_fallback(client: Any, *, model: str, messages: list[dict[str, Any]], max_tokens: int) -> Any:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip()
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    if os.environ.get("OPENAI_RESPONSE_FORMAT_JSON", "").strip().lower() in {"1", "true", "yes", "json"}:
        request["response_format"] = {"type": "json_object"}
    if _uses_modern_completion_param(model):
        request["max_completion_tokens"] = max_tokens
    else:
        request["temperature"] = 0.0
        request["max_tokens"] = max_tokens

    try:
        return client.chat.completions.create(**request)
    except Exception as exc:
        text = str(exc)
        # Some OpenAI-compatible deployments still expect the legacy parameter,
        # while current reasoning models prefer max_completion_tokens. Retry once
        # with the alternate spelling when the endpoint says the first one is bad.
        if "max_completion_tokens" in request and ("max_completion_tokens" in text or "unsupported" in text.lower()):
            request.pop("max_completion_tokens", None)
            request["max_tokens"] = max_tokens
            return client.chat.completions.create(**request)
        if "max_tokens" in request and ("max_tokens" in text or "unsupported" in text.lower()):
            request.pop("max_tokens", None)
            request["max_completion_tokens"] = max_tokens
            return client.chat.completions.create(**request)
        if "temperature" in request and ("temperature" in text or "unsupported" in text.lower()):
            request.pop("temperature", None)
            return client.chat.completions.create(**request)
        if "reasoning_effort" in request and ("reasoning_effort" in text or "unsupported" in text.lower()):
            request.pop("reasoning_effort", None)
            return client.chat.completions.create(**request)
        if "response_format" in request and ("response_format" in text or "unsupported" in text.lower()):
            request.pop("response_format", None)
            return client.chat.completions.create(**request)
        raise


def install_openai_backend(base_url: str, api_key: str, model: str, max_tokens: int) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    def _call(messages: list[dict[str, Any]]) -> str:
        started = time.time()
        resp = _chat_completion_with_fallback(client, model=model, messages=messages, max_tokens=max_tokens)
        elapsed = time.time() - started
        usage = getattr(resp, "usage", None)
        input_tokens = _usage_value(usage, "prompt_tokens")
        output_tokens = _usage_value(usage, "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens", input_tokens + output_tokens)
        OPENAI_CALL_LOG.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "elapsed_s": elapsed,
                "estimated_cost_usd": _estimate_openai_cost_usd(model, input_tokens, output_tokens),
            }
        )
        return resp.choices[0].message.content or ""

    llm_mod._BACKEND_CALLS["single_openai"] = _call
    os.environ["MODEL_BACKEND"] = "single_openai"


def _maybe_init_accelerate() -> None:
    """Initialize Accelerate state before constructing a local Transformers model."""
    try:
        from accelerate import Accelerator

        Accelerator()
    except Exception:
        pass


def install_local_qwen_backend(pretrained: str, max_new_tokens: int, adapter_path: str = "") -> None:
    _maybe_init_accelerate()
    from evaluation.local_model_backend import build_local_chat_backend

    llm_mod._BACKEND_CALLS["single_local_qwen"] = build_local_chat_backend(
        pretrained=pretrained,
        max_new_tokens=max_new_tokens,
        adapter_path=adapter_path,
    )
    os.environ["MODEL_BACKEND"] = "single_local_qwen"


def run_case(record: dict[str, Any], *, max_turns: int) -> dict[str, Any]:
    state = new_single_state(record)
    # A stable dataset index is not a safe worker-session identifier: reruns
    # would reopen stale state (source family, handles, and counters). Make the
    # transport session unique while retaining the case id for traceability.
    session_id = install_geodisaster_context(
        record,
        session_id=_new_session_id(record),
    )
    messages: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    openai_call_start = len(OPENAI_CALL_LOG)
    started = time.time()
    plan: dict[str, Any] | None = None
    last_observation: dict[str, Any] | None = None
    final_answer = ""
    status = "RUNNING"
    model_error: BaseException | None = None
    try:
        for turn_no in range(1, max_turns + 1):
            user_msg = (
                build_initial_user_message(record)
                if turn_no == 1
                else build_runtime_user_message(
                    record,
                    plan=plan,
                    memory_text=render_memory(state),
                    last_observation=last_observation,
                    turn_no=turn_no,
                )
            )
            try:
                turn = llm_mod.llm_call(
                    system=SINGLE_AGENT_SYSTEM,
                    user=user_msg,
                    schema=SingleAgentTurn,
                    max_retries=1,
                )
                assistant_payload = turn.model_dump()
            except Exception as exc:
                model_error = exc
                status = "SCHEMA_OR_MODEL_ERROR"
                trace.append({"turn": turn_no, "role": "agent", "error": str(exc), "user_message": user_msg})
                break

            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)})
            trace.append({"turn": turn_no, "role": "agent", "user_message": user_msg, "assistant": assistant_payload})
            action = turn.action
            if isinstance(action, SinglePlan):
                plan = action.model_dump()
                state["plan"] = plan
                last_observation = {"tool": "Plan", "status": "OK", "summary": "plan recorded"}
                continue
            if isinstance(action, SingleFinalAnswer):
                resolution = resolve_answer_ref(state, action.answer_ref)
                if not resolution.ok:
                    last_observation = {
                        "tool": "FinalAnswer",
                        "status": "ERROR",
                        "failure_category": "answer_reference_not_ready",
                        "summary": resolution.error,
                    }
                    trace.append(
                        {"turn": turn_no, "role": "runtime", "payload": last_observation}
                    )
                    continue
                validation = validate_explicit_answer(record, resolution.answer)
                if validation.ok:
                    final_answer = json.dumps(
                        validation.answer, ensure_ascii=False, sort_keys=True
                    )
                    status = "DONE"
                    break
                last_observation = {
                    "tool": "FinalAnswer",
                    "status": "ERROR",
                    "failure_category": "answer_contract_failure",
                    "summary": "Final answer rejected by the declared JSON Schema.",
                    "validation_errors": list(validation.errors),
                }
                trace.append(
                    {
                        "turn": turn_no,
                        "role": "runtime",
                        "payload": last_observation,
                    }
                )
                continue
            if isinstance(action, SingleToolCall):
                result = call_single_tool(action.tool, action.args)
                obs = absorb_tool_result(state, action.tool, action.args, result)
                last_observation = obs
                trace.append({"turn": turn_no, "role": "tool", "payload": obs})
                continue
        else:
            status = "MAX_TURNS"
    finally:
        clear_geodisaster_context()

    gold = record.get("expected_final_answer", record.get("final_answer", ""))
    score = _score_answer(gold, final_answer)
    failure_category = _single_failure_category(
        status,
        answer_correct=bool(score.get("correct")),
        model_error=model_error,
        trace=trace,
    )
    openai_calls = OPENAI_CALL_LOG[openai_call_start:]
    openai_usage = {
        "calls": len(openai_calls),
        "input_tokens": sum(int(c.get("input_tokens") or 0) for c in openai_calls),
        "output_tokens": sum(int(c.get("output_tokens") or 0) for c in openai_calls),
        "total_tokens": sum(int(c.get("total_tokens") or 0) for c in openai_calls),
        "estimated_cost_usd": sum(
            float(c.get("estimated_cost_usd") or 0.0)
            for c in openai_calls
            if c.get("estimated_cost_usd") is not None
        ),
        "elapsed_s": time.time() - started,
    }
    return {
        "idx": record.get("idx"),
        "dataset_id": record.get("dataset_id"),
        "source": record.get("source") or str(record.get("dataset_id") or "").split("/", 1)[0],
        "question_type": record.get("question_type"),
        # Evaluation metadata is written only after the episode. It makes each
        # transcript independently judgeable without exposing gold state to the
        # agent-facing runtime context.
        "question": record.get("question"),
        "initial_artifacts": record.get("initial_artifacts") or {},
        "reference_toolchain": record.get("gt_chain") or [],
        "plan": plan,
        "session_id": session_id,
        "status": status,
        "failure_category": failure_category,
        "gold": gold,
        "prediction": final_answer,
        "score": score,
        "messages": messages,
        "trace": trace,
        "openai_usage": openai_usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    env_backend = os.environ.get("MODEL_BACKEND", "openai")
    default_backend = env_backend if env_backend in {"openai", "local_qwen"} else "openai"
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--core_jsonl",
        type=Path,
        default=None,
        help="Canonical core file used to hydrate normalized single-agent rows by id.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_turns", type=int, default=18)
    parser.add_argument("--model_backend", choices=["openai", "local_qwen"], default=default_backend)
    parser.add_argument(
        "--stub_from_teacher",
        action="store_true",
        help=(
            "Explicit integration test: execute published teacher decisions through "
            "the current single-agent runtime and real workers."
        ),
    )
    parser.add_argument("--openai_base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--openai_model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument(
        "--model_path",
        default=os.environ.get(
            "GEODISASTER_MODEL_PATH",
            os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
        ),
    )
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("SINGLE_AGENT_MAX_NEW_TOKENS", "1024")))
    args = parser.parse_args()

    if args.stub_from_teacher:
        os.environ["MODEL_BACKEND"] = "stub"
    elif args.model_backend == "openai":
        if not args.openai_base_url or not args.openai_model:
            raise SystemExit("openai backend requires --openai_base_url and --openai_model")
        install_openai_backend(args.openai_base_url, args.openai_api_key, args.openai_model, args.max_new_tokens)
    else:
        install_local_qwen_backend(args.model_path, args.max_new_tokens, adapter_path=args.adapter_path)

    rows = load_jsonl(args.dataset, args.limit)
    core_jsonl = _resolve_core_jsonl(args.dataset, args.core_jsonl)
    rows = _hydrate_from_core(rows, core_jsonl)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for pos, record in enumerate(rows):
        if args.stub_from_teacher:
            llm_mod.set_stub_responses(_teacher_stub_responses(record))
        result = run_case(record, max_turns=args.max_turns)
        results.append(result)
        (out / f"conversation_{pos:04d}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{pos+1}/{len(rows)}] {result['status']} acc={result['score']['answer_acc']:.3f} {result.get('dataset_id')}")

    with (out / "conversations.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    mean_acc = sum(float(r["score"]["answer_acc"]) for r in results) / max(len(results), 1)
    correct = sum(1 for r in results if r["score"].get("correct"))
    summary = {
        "dataset": str(args.dataset),
        "core_jsonl": str(core_jsonl) if core_jsonl else "embedded",
        "mode": "teacher_integration" if args.stub_from_teacher else "online_model",
        "records": len(results),
        "mean_answer_acc": mean_acc,
        "correct": correct,
        "correct_rate": correct / max(len(results), 1),
        "statuses": dict(
            sorted(Counter(str(r.get("status") or "unknown") for r in results).items())
        ),
        "failure_categories": dict(
            sorted(
                Counter(
                    str(r.get("failure_category") or "unknown") for r in results
                ).items()
            )
        ),
        "openai_usage": {
            "calls": sum(int((r.get("openai_usage") or {}).get("calls") or 0) for r in results),
            "input_tokens": sum(int((r.get("openai_usage") or {}).get("input_tokens") or 0) for r in results),
            "output_tokens": sum(int((r.get("openai_usage") or {}).get("output_tokens") or 0) for r in results),
            "total_tokens": sum(int((r.get("openai_usage") or {}).get("total_tokens") or 0) for r in results),
            "estimated_cost_usd": sum(float((r.get("openai_usage") or {}).get("estimated_cost_usd") or 0.0) for r in results),
            "elapsed_s": sum(float((r.get("openai_usage") or {}).get("elapsed_s") or 0.0) for r in results),
        },
        "out_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Single-Agent GeoDisaster Eval\n\n"
        f"- Records: {summary['records']}\n"
        f"- Mean answer accuracy: {summary['mean_answer_acc']:.4f}\n"
        f"- Correct rate: {summary['correct_rate']:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
