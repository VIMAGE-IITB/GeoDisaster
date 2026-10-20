#!/usr/bin/env python3
"""Evaluate GeoDisaster dataset traces, workers, and live agents.

Default ``trace_audit`` mode checks published ORC/user flow shape, tool
ownership, TaskComplete discipline, and terminal structured answers without
executing workers. ``deterministic`` executes each prescribed chain.

`--mode online` runs the same graph/user-flow style as the EO live evaluator,
with `MAS_AGENT_PROFILE=geodisaster`, local/live GeoDisaster workers, and
conversation logs suitable for the existing analysis scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("GEODISASTER_DATASET_ROOT", str(ROOT.parent / "geodisaster-agentic-2921"))
).expanduser()
DEFAULT_MULTIAGENTIC = DEFAULT_DATASET_ROOT / "data" / "multi_agent" / "test_conversations.jsonl"
DEFAULT_OUT = ROOT / "runs" / "geodisaster_eval"

os.environ.setdefault("MAS_AGENT_PROFILE", "geodisaster")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_agent.final_dataset_profile import FINAL_DATASET_AGENT_TOOLS, TOOL_TO_AGENT  # noqa: E402
from evaluation.structured_answer import score_answer  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def pct(num: int, den: int) -> float:
    return round((num / den) * 100, 4) if den else 0.0


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("dataset_id") or record.get("id") or record.get("sample_id") or "")


def record_source(record: dict[str, Any]) -> str:
    if record.get("source"):
        return str(record["source"])
    rid = record_id(record)
    parts = rid.split("/")
    if len(parts) >= 2 and parts[0] == "multi_hazard":
        return "/".join(parts[:2])
    return parts[0] if parts and parts[0] else "unknown"


def logical_question_type(record: dict[str, Any]) -> str:
    rid = record_id(record)
    parts = rid.split("/")
    qtype = str(record.get("question_type") or (parts[-1] if parts else "unknown"))
    if parts and parts[0] == "multi_hazard" and len(parts) >= 2:
        return f"multi_hazard/{parts[1]}/{qtype}"
    family = parts[0] if parts and parts[0] else record_source(record)
    return f"{family}/{qtype}"


def eval_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    plan = record.get("plan") or {}
    plan_steps = plan.get("steps") or []
    step_by_id = {str(step.get("id")): step for step in plan_steps}
    executions = record.get("executions") or []
    expected_execution_ids = [str(step.get("id")) for step in plan_steps]
    actual_execution_ids = [str(exe.get("step_id")) for exe in executions]
    if expected_execution_ids != actual_execution_ids:
        errors.append("execution_order_mismatch")

    tool_calls = 0
    ownership_ok = 0
    taskcomplete_ok = 0
    tool_sequence: list[str] = []
    agent_sequence: list[str] = []
    for exe in executions:
        step_id = str(exe.get("step_id"))
        agent = str(exe.get("agent"))
        agent_sequence.append(agent)
        step = step_by_id.get(step_id)
        if not step:
            errors.append(f"missing_plan_step:{step_id}")
        elif step.get("agent") != agent:
            errors.append(f"agent_mismatch:{step_id}")
        turns = exe.get("turns") or []
        if not turns or ((turns[-1].get("action") or {}).get("tool") != "TaskComplete"):
            errors.append(f"missing_taskcomplete:{step_id}")
        else:
            status = (turns[-1].get("action") or {}).get("status")
            if status == "TASK_COMPLETE":
                taskcomplete_ok += 1
            else:
                errors.append(f"contract_failure:{step_id}")
        for turn in turns:
            action = turn.get("action") or {}
            tool = action.get("tool")
            if not tool or tool == "TaskComplete":
                continue
            tool_calls += 1
            tool_sequence.append(str(tool))
            if TOOL_TO_AGENT.get(str(tool)) == agent and tool in FINAL_DATASET_AGENT_TOOLS.get(agent, []):
                ownership_ok += 1
            else:
                errors.append(f"wrong_owner:{agent}:{tool}")

    pure_ok = record.get("pure_toolchain_sim_status") == "passed"
    support_score = ((record.get("answer_support") or {}).get("score"))
    if support_score is None:
        support_score = 1.0 if pure_ok else 0.0

    answer_metrics = score_answer(
        record.get("expected_final_answer"), record.get("resolved_final_answer")
    )
    terminal_ref = str(record.get("final_answer_ref") or "")
    if terminal_ref != "ANSWER_JSON:answer_1":
        errors.append("invalid_terminal_answer_ref")

    return {
        "dataset_id": record_id(record),
        "source": record_source(record),
        "question_type": record.get("question_type"),
        "logical_question_type": logical_question_type(record),
        "plan_step_count": len(plan_steps),
        "execution_count": len(executions),
        "tool_call_count": tool_calls,
        "tool_ownership_ok": ownership_ok,
        "tool_ownership_total": tool_calls,
        "taskcomplete_ok": taskcomplete_ok,
        "taskcomplete_total": len(executions),
        "terminate_ok": terminal_ref == "ANSWER_JSON:answer_1",
        "pure_worker_pass": pure_ok,
        "answer_support_score": float(support_score),
        "answer": answer_metrics,
        "tool_sequence": tool_sequence,
        "agent_sequence": agent_sequence,
        "errors": errors,
        "ok": not errors and pure_ok and bool(answer_metrics["correct"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    tool_total = sum(r["tool_ownership_total"] for r in rows)
    tool_ok = sum(r["tool_ownership_ok"] for r in rows)
    tc_total = sum(r["taskcomplete_total"] for r in rows)
    tc_ok = sum(r["taskcomplete_ok"] for r in rows)
    tool_counts = Counter(tool for r in rows for tool in r["tool_sequence"])
    agent_step_counts = Counter(agent for r in rows for agent in r["agent_sequence"])
    source_counts = Counter(r["source"] for r in rows)
    qtype_counts = Counter(r["logical_question_type"] for r in rows)
    by_source: dict[str, Any] = {}
    for src in sorted(source_counts):
        items = [r for r in rows if r["source"] == src]
        by_source[src] = {
            "records": len(items),
            "ok_rate": pct(sum(1 for r in items if r["ok"]), len(items)),
            "mean_answer_support": round(sum(r["answer_support_score"] for r in items) / len(items), 6),
            "mean_tool_calls": round(sum(r["tool_call_count"] for r in items) / len(items), 4),
        }
    return {
        "records": n,
        "ok_records": sum(1 for r in rows if r["ok"]),
        "ok_rate": pct(sum(1 for r in rows if r["ok"]), n),
        "plan_valid_rate": pct(sum(1 for r in rows if not any(e.startswith("execution_order") or e.startswith("agent_mismatch") for e in r["errors"])), n),
        "tool_ownership_rate": pct(tool_ok, tool_total),
        "taskcomplete_rate": pct(tc_ok, tc_total),
        "terminate_rate": pct(sum(1 for r in rows if r["terminate_ok"]), n),
        "pure_worker_pass_rate": pct(sum(1 for r in rows if r["pure_worker_pass"]), n),
        "terminal_answer_correct_rate": pct(sum(1 for r in rows if r["answer"]["correct"]), n),
        "mean_answer_support": round(sum(r["answer_support_score"] for r in rows) / n, 6) if rows else 0.0,
        "mean_plan_steps": round(sum(r["plan_step_count"] for r in rows) / n, 4) if rows else 0.0,
        "mean_tool_calls": round(sum(r["tool_call_count"] for r in rows) / n, 4) if rows else 0.0,
        "tool_counts": dict(tool_counts.most_common()),
        "agent_step_counts": dict(agent_step_counts),
        "source_counts": dict(source_counts),
        "question_type_count": len(qtype_counts),
        "by_source": by_source,
    }


def write_report(out_dir: Path, summary: dict[str, Any], cases_path: Path) -> None:
    lines = [
        "# GeoDisaster Dataset Multi-Agent Eval",
        "",
        f"- Cases: `{summary['records']}`",
        f"- OK records: `{summary['ok_records']}/{summary['records']}` (`{summary['ok_rate']}%`)",
        f"- Plan valid rate: `{summary['plan_valid_rate']}%`",
        f"- Tool ownership rate: `{summary['tool_ownership_rate']}%`",
        f"- TaskComplete rate: `{summary['taskcomplete_rate']}%`",
        f"- Terminate rate: `{summary['terminate_rate']}%`",
        f"- Pure worker pass rate: `{summary['pure_worker_pass_rate']}%`",
        f"- Terminal structured-answer correctness: `{summary['terminal_answer_correct_rate']}%`",
        f"- Mean answer support: `{summary['mean_answer_support']}`",
        f"- Mean plan steps: `{summary['mean_plan_steps']}`",
        f"- Mean tool calls: `{summary['mean_tool_calls']}`",
        "",
        "## Agent Step Counts",
        "",
        json.dumps(summary["agent_step_counts"], indent=2, ensure_ascii=False),
        "",
        "## Tool Counts",
        "",
        json.dumps(summary["tool_counts"], indent=2, ensure_ascii=False),
        "",
        "## Source Breakdown",
        "",
        "| Source | Records | OK % | Mean Answer Support | Mean Tool Calls |",
        "|---|---:|---:|---:|---:|",
    ]
    for src, item in summary["by_source"].items():
        lines.append(f"| `{src}` | {item['records']} | {item['ok_rate']} | {item['mean_answer_support']} | {item['mean_tool_calls']} |")
    lines.extend(["", f"Per-case metrics: `{cases_path}`"])
    (out_dir / "major_agentic_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gt_replay(input_jsonl: Path, output_dir: Path, limit: int = 0) -> dict[str, Any]:
    records = read_jsonl(input_jsonl)
    if limit:
        records = records[:limit]
    rows = [eval_trace_record(rec) for rec in records]
    summary = aggregate(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "case_metrics.jsonl"
    write_jsonl(cases_path, rows)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, cases_path)
    return summary


def _install_model_backend(args: argparse.Namespace) -> None:
    os.environ["MODEL_BACKEND"] = args.model_backend
    if args.model_backend in {"stub", "hf", "ollama"}:
        return

    from evaluation.eval_live import _install_local_qwen_backend, _install_openai_backend, _maybe_init_accelerate

    _maybe_init_accelerate()
    try:
        from accelerate import Accelerator
        from accelerate.state import PartialState

        PartialState.is_main_process = property(lambda self: getattr(self, "process_index", 0) == 0)  # type: ignore[attr-defined]
        Accelerator()
        PartialState()
    except Exception:
        pass
    if args.model_backend == "local_qwen":
        _install_local_qwen_backend(
            pretrained=args.model_path,
            max_new_tokens=args.max_new_tokens,
            adapter_root=args.adapter_root,
        )
        return
    if args.model_backend == "openai":
        if not args.openai_base_url:
            raise SystemExit("MODEL_BACKEND=openai requires --openai_base_url or OPENAI_BASE_URL")
        _install_openai_backend(
            base_url=args.openai_base_url,
            model=args.openai_model,
            api_key=args.openai_api_key,
        )
        return
    raise SystemExit(f"Unsupported model_backend: {args.model_backend}")


def _shared_state_for_log(final_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifacts": final_state.get("artifacts"),
        "layers": final_state.get("layers"),
        "fields": {
            name: {"value": value["value"], "produced_by_step": value["produced_by_step"]}
            for name, value in (final_state.get("fields") or {}).items()
            if isinstance(value, dict) and "value" in value
        },
    }


def _online_failure_category(
    status: str | None,
    abort_reason: str | None,
    *,
    answer_correct: bool = False,
    trace: list[dict[str, Any]] | None = None,
) -> str:
    if status == "DONE":
        return "success" if answer_correct else "answer_mismatch"
    failed_tool_text = " ".join(
        str((event.get("payload") or {}).get("text") or "")
        for event in (trace or [])
        if event.get("role") == "tool" and (event.get("payload") or {}).get("ok") is False
    ).lower()
    if "no declared sentinel-1 raster is available" in failed_tool_text:
        return "required_asset_unavailable"
    if failed_tool_text:
        return "tool_or_worker_error"
    reason = str(abort_reason or "").lower()
    if "transport" in reason or "stub queue empty" in reason or "inference call failed" in reason:
        return "model_transport_error"
    if "_invalid" in reason or "validation" in reason or "schema" in reason:
        return "model_schema_error"
    if "tool" in reason or "worker" in reason:
        return "tool_or_worker_error"
    if "recursion" in reason or "loop" in reason:
        return "agent_loop_error"
    return "agent_runtime_error"


def _aggregate_online_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    from evaluation.eval_live import (
        _aggregate_failure_buckets,
        _aggregate_sequence_metrics,
        _finalize_agent_metrics,
        _turn_agent_metrics_template,
    )

    n = max(len(results), 1)
    aggregate_agent_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in results:
        for agent, metrics in (row.get("agent_raw_metrics") or {}).items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    aggregate_agent_counts[agent][key] += float(value)

    per_agent_summary: dict[str, Any] = {}
    for agent, totals in sorted(aggregate_agent_counts.items()):
        raw = _turn_agent_metrics_template()
        for key in raw:
            raw[key] = totals.get(key, 0.0)
        per_agent_summary[agent] = _finalize_agent_metrics(raw)

    aggregate_overall_raw = _turn_agent_metrics_template()
    for row in results:
        metrics = row.get("alignment_raw_metrics") or {}
        for key in aggregate_overall_raw:
            aggregate_overall_raw[key] += float(metrics.get(key, 0.0))

    by_type: dict[str, dict[str, Any]] = {}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        bucket = str(
            row.get("logical_question_type")
            or f"{row.get('source')}::{row.get('question_type')}"
        )
        buckets[bucket].append(row)
    for key, rows in sorted(buckets.items()):
        by_type[key] = {
            "records": len(rows),
            "done_rate": sum(1 for row in rows if row.get("status") == "DONE") / len(rows),
            "correct_rate": sum(1 for row in rows if (row.get("metrics") or {}).get("correct")) / len(rows),
            "answer_acc_avg": sum((row.get("metrics") or {}).get("answer_acc", 0.0) for row in rows) / len(rows),
            "mean_tool_calls": sum(
                len([event for event in (row.get("trace") or []) if event.get("phase") == "tool_call"])
                for row in rows
            )
            / len(rows),
        }

    return {
        "mode": "online",
        "n_samples": len(results),
        "task_complete_rate": sum(1 for row in results if row.get("status") == "DONE") / n,
        "answer_acc_avg": sum((row.get("metrics") or {}).get("answer_acc", 0.0) for row in results) / n,
        "token_f1_avg": sum((row.get("metrics") or {}).get("token_f1", 0.0) for row in results) / n,
        "sequence_ratio_avg": sum((row.get("metrics") or {}).get("sequence_ratio", 0.0) for row in results) / n,
        "exact_match_rate": sum(1 for row in results if (row.get("metrics") or {}).get("exact_match")) / n,
        "numeric_match_rate": sum(1 for row in results if (row.get("metrics") or {}).get("numeric_match")) / n,
        "numeric_subset_match_rate": sum(
            1 for row in results if (row.get("metrics") or {}).get("numeric_subset_match")
        )
        / n,
        "correct_rate": sum(1 for row in results if (row.get("metrics") or {}).get("correct")) / n,
        "failure_categories": dict(
            sorted(Counter(str(row.get("failure_category") or "unknown") for row in results).items())
        ),
        "alignment_metrics": _finalize_agent_metrics(aggregate_overall_raw),
        "sequence_metrics": _aggregate_sequence_metrics(results),
        "failure_buckets": _aggregate_failure_buckets(results),
        "per_agent_metrics": per_agent_summary,
        "by_question_type": by_type,
    }


def _write_online_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GeoDisaster Online Agent Evaluation",
        "",
        f"- Cases: `{summary['n_samples']}`",
        f"- Task complete rate: `{summary['task_complete_rate']:.4f}`",
        f"- Correct rate: `{summary['correct_rate']:.4f}`",
        f"- Answer accuracy avg: `{summary['answer_acc_avg']:.4f}`",
        f"- Token F1 avg: `{summary['token_f1_avg']:.4f}`",
        "",
        "## Alignment Metrics",
        "",
        json.dumps(summary["alignment_metrics"], indent=2, ensure_ascii=False),
        "",
        "## Per-Agent Metrics",
        "",
        json.dumps(summary["per_agent_metrics"], indent=2, ensure_ascii=False),
        "",
        "## Question-Type Breakdown",
        "",
        "| Question Type | N | Done | Correct | Answer Acc | Mean Tool Calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary["by_question_type"].items():
        lines.append(
            f"| `{key}` | {item['records']} | {item['done_rate']:.3f} | "
            f"{item['correct_rate']:.3f} | {item['answer_acc_avg']:.3f} | {item['mean_tool_calls']:.2f} |"
        )
    (output_dir / "online_major_agentic_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _teacher_stub_responses(record: dict[str, Any]) -> list[str]:
    """Convert a published teacher trajectory into model responses.

    This is an explicit integration-test backend: LangGraph still performs all
    routing and every ToolCall still reaches the configured worker backend.
    It tests the agent/worker contract without attributing model quality to a
    canned stub.  It is never enabled implicitly.
    """
    responses: list[str] = []
    for event in record.get("multiagentic_gt_chain") or []:
        role = str(event.get("role") or "")
        output = event.get("output")
        if role == "USER" or not isinstance(output, dict):
            continue
        if role == "ORC":
            payload = output
        else:
            payload = {
                "thought": str(event.get("thought") or "Execute the next contract action."),
                "action": output,
            }
        responses.append(json.dumps(payload, ensure_ascii=False, default=str))
    if not responses:
        raise ValueError(f"record {record_id(record)!r} has no teacher model responses")
    return responses


def run_online(
    input_jsonl: Path,
    output_dir: Path,
    limit: int = 0,
    recursion_limit: int = 120,
    *,
    one_per_type: bool = False,
    stub_from_teacher: bool = False,
) -> dict[str, Any]:
    from evaluation.eval_live import _analyze_record, _score_answer
    from multi_agent.graph import build_graph
    from multi_agent.llm import set_stub_responses
    from multi_agent.tool_gateway import clear_final_dataset_context, set_final_dataset_context
    from multi_agent.run import _build_conversation_v1_shape
    from multi_agent.state import new_episode

    records = read_jsonl(input_jsonl)
    if one_per_type:
        selected: dict[str, dict[str, Any]] = {}
        for record in records:
            selected.setdefault(logical_question_type(record), record)
        records = [selected[key] for key in sorted(selected)]
    if limit:
        records = records[:limit]
    graph = build_graph()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    agent_conversations_path = output_dir / "agent_conversations.jsonl"
    tool_calls_path = output_dir / "tool_calls.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    failure_tags_path = output_dir / "alignment_failure_tags.jsonl"
    for export_path in (agent_conversations_path, tool_calls_path, trajectories_path, failure_tags_path):
        if export_path.exists():
            export_path.unlink()
    for idx, record in enumerate(records, start=1):
        if stub_from_teacher:
            set_stub_responses(_teacher_stub_responses(record))
        sid = set_final_dataset_context(record, session_id=f"online_{idx:05d}_{int(time.time())}")
        state = new_episode(question=str(record.get("raw_question") or record["question"]))
        for handle, card in (record.get("initial_artifacts") or {}).items():
            state["artifacts"][handle] = {
                "kind": str(card.get("type") or "data"),
                "path": "",
                "produced_by_step": "",
                "note": str(card.get("description") or card.get("type") or "input artifact"),
            }
        start = time.time()
        final_state: dict[str, Any] = state
        try:
            final_state = graph.invoke(state, {"recursion_limit": recursion_limit})
            status = final_state.get("status")
            final_answer = final_state.get("final_answer")
            trace = final_state.get("history")
        except Exception as exc:
            status = "ABORTED"
            final_answer = ""
            state["status"] = "ABORTED"
            state["abort_reason"] = f"runtime_exception: {exc}"
            state["final_answer"] = ""
            trace = state.get("history") or [{"error": str(exc)}]
        finally:
            clear_final_dataset_context()
        metrics = _score_answer(record.get("expected_final_answer", ""), final_answer or "")
        row = {
            "dataset_index": idx - 1,
            "idx": record_id(record) or idx,
            "dataset_id": record_id(record),
            "source": record_source(record),
            "question_type": record.get("question_type"),
            "logical_question_type": logical_question_type(record),
            "question": record.get("question"),
            "label": record.get("expected_final_answer", ""),
            "initial_artifacts": record.get("initial_artifacts", {}),
            "status": status,
            "abort_reason": final_state.get("abort_reason", ""),
            "final_answer": final_answer,
            "expected_final_answer": record.get("expected_final_answer"),
            "gt_chain": record.get("gt_chain", []),
            "wall_time_s": round(time.time() - start, 3),
            "conversation": _build_conversation_v1_shape(final_state),
            "plan": final_state.get("plan").model_dump() if final_state.get("plan") is not None else None,
            "steps": final_state.get("steps"),
            "shared_state": _shared_state_for_log(final_state),
            "trace": trace,
            "metrics": metrics,
            "session_id": sid,
        }
        row["failure_category"] = _online_failure_category(
            str(status) if status is not None else None,
            str(row["abort_reason"] or ""),
            answer_correct=bool(metrics.get("correct")),
            trace=trace,
        )
        analysis = _analyze_record(row)
        row["alignment_metrics"] = analysis["overall_metrics"]
        row["alignment_raw_metrics"] = analysis["overall_raw_metrics"]
        row["agent_metrics"] = analysis["agent_metrics"]
        row["agent_raw_metrics"] = analysis["agent_raw_metrics"]
        row["sequence_metrics"] = analysis["sequence_metrics"]
        row["failure_buckets"] = analysis["failure_buckets"]
        results.append(row)
        write_json(output_dir / f"conversation_{idx}.json", row)
        for row_item in analysis["datasets"]["agent_conversations"]:
            with agent_conversations_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row_item, ensure_ascii=False, default=str) + "\n")
        for row_item in analysis["datasets"]["tool_calls"]:
            with tool_calls_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row_item, ensure_ascii=False, default=str) + "\n")
        for row_item in analysis["datasets"]["trajectories"]:
            with trajectories_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row_item, ensure_ascii=False, default=str) + "\n")
        for row_item in analysis["alignment_failure_tags"]:
            with failure_tags_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row_item, ensure_ascii=False, default=str) + "\n")
        print(
            f"[{idx}/{len(records)}] dataset_id={record.get('dataset_id')} "
            f"status={status} answer_acc={metrics['answer_acc']:.3f} "
            f"correct={metrics['correct']} answer={str(final_answer or '')[:120]!r}",
            flush=True,
        )
    write_jsonl(output_dir / "online_conversations.jsonl", results)
    summary = _aggregate_online_summary(results)
    summary.update(
        {
            "input_jsonl": str(input_jsonl),
            "output_dir": str(output_dir),
            "exports": {
                "agent_conversations_jsonl": str(agent_conversations_path),
                "tool_calls_jsonl": str(tool_calls_path),
                "trajectories_jsonl": str(trajectories_path),
                "alignment_failure_tags_jsonl": str(failure_tags_path),
            },
        }
    )
    write_json(output_dir / "summary.json", summary)
    _write_online_report(output_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, default=DEFAULT_MULTIAGENTIC)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--mode",
        choices=["trace_audit", "gt_replay", "deterministic", "online"],
        default="trace_audit",
        help="gt_replay is retained as a backward-compatible alias for trace_audit.",
    )
    parser.add_argument("--tool_backend", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--one_per_type", action="store_true")
    parser.add_argument(
        "--stub_from_teacher",
        action="store_true",
        help="Explicit integration test: replay teacher model decisions through the live graph and workers.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recursion_limit", type=int, default=120)
    parser.add_argument("--model_backend", default=os.environ.get("MODEL_BACKEND", "local_qwen"))
    parser.add_argument(
        "--model_path",
        default=os.environ.get(
            "GEODISASTER_MODEL_PATH",
            os.environ.get("MAS_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAS_MAX_NEW_TOKENS", "1024")))
    parser.add_argument("--adapter_root", default=os.environ.get("MAS_ADAPTER_ROOT", ""))
    parser.add_argument("--openai_base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai_model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    args = parser.parse_args()
    os.environ["GEODISASTER_TOOL_BACKEND"] = args.tool_backend
    if args.mode in {"trace_audit", "gt_replay"}:
        summary = run_gt_replay(args.input_jsonl.resolve(), args.output_dir.resolve(), args.limit)
    elif args.mode == "deterministic":
        from evaluation.deterministic_replay import run_deterministic_replay

        summary = run_deterministic_replay(
            args.input_jsonl.resolve(),
            args.output_dir.resolve(),
            limit=args.limit,
            one_per_type=args.one_per_type,
        )
    else:
        _install_model_backend(args)
        if args.stub_from_teacher and args.model_backend != "stub":
            parser.error("--stub_from_teacher requires --model_backend stub")
        summary = run_online(
            args.input_jsonl.resolve(),
            args.output_dir.resolve(),
            args.limit,
            args.recursion_limit,
            one_per_type=args.one_per_type,
            stub_from_teacher=args.stub_from_teacher,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
