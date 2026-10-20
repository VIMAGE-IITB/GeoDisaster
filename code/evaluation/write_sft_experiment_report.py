from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _last_loss_line(path: Path) -> str:
    if not path.exists():
        return ""
    last = ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and '"loss"' in stripped:
            last = stripped
    return last


def _extract_steps(path: Path) -> tuple[int | None, float | None]:
    last = _last_loss_line(path)
    if not last:
        return None, None
    try:
        payload = json.loads(last)
        return int(payload.get("step")), float(payload.get("loss"))
    except Exception:
        return None, None


def _metrics_block(summary: dict[str, Any]) -> list[str]:
    alignment = summary.get("alignment_metrics", {}) or {}
    sequence = summary.get("sequence_metrics", {}) or {}
    failures = summary.get("failure_buckets", {}) or {}
    return [
        f"- `answer_acc_avg`: `{summary.get('answer_acc_avg', 0):.6f}`",
        f"- `correct_rate`: `{summary.get('correct_rate', 0):.6f}`",
        f"- `task_complete_rate`: `{summary.get('task_complete_rate', 0):.6f}`",
        f"- `TSR`: `{alignment.get('TSR', 0):.6f}`",
        f"- `orc_valid_rate`: `{alignment.get('orc_valid_rate', 0):.6f}`",
        f"- `first_tool_accuracy`: `{sequence.get('first_tool_accuracy', 0):.6f}`",
        f"- `first_divergence_index_avg`: `{sequence.get('first_divergence_index_avg', 0):.6f}`",
        f"- `overcall_count_avg`: `{sequence.get('overcall_count_avg', 0):.6f}`",
        f"- `redundant_call_count_avg`: `{sequence.get('redundant_call_count_avg', 0):.6f}`",
        f"- `stop_after_sufficient_evidence_rate`: `{sequence.get('stop_after_sufficient_evidence_rate', 0):.6f}`",
        f"- `planning_failure_rate`: `{failures.get('planning_failure_rate', 0):.6f}`",
        f"- `tool_selection_failure_rate`: `{failures.get('tool_selection_failure_rate', 0):.6f}`",
        f"- `state_update_failure_rate`: `{failures.get('state_update_failure_rate', 0):.6f}`",
        f"- `redundant_or_degenerate_loops_rate`: `{failures.get('redundant_or_degenerate_loops_rate', 0):.6f}`",
    ]


def _gt_metrics_block(report: dict[str, Any]) -> list[str]:
    summary = report.get("summary", {}) or {}
    plan = summary.get("plan", {}) or {}
    toolchain = summary.get("toolchain", {}) or {}
    contracts = summary.get("contracts", {}) or {}
    answer = summary.get("answer", {}) or {}
    return [
        f"- `gt_cases`: `{summary.get('n_cases', 0)}`",
        f"- `gt_plan_step_count_match_rate`: `{plan.get('step_count_match_rate', 0):.6f}`",
        f"- `gt_plan_role_sequence_match_rate`: `{plan.get('role_sequence_match_rate', 0):.6f}`",
        f"- `gt_exact_tool_sequence_match_rate`: `{toolchain.get('exact_sequence_match_rate', 0):.6f}`",
        f"- `gt_ordered_lcs_recall_avg`: `{toolchain.get('ordered_lcs_recall_avg', 0):.6f}`",
        f"- `gt_unordered_tool_f1_avg`: `{toolchain.get('unordered_f1_avg', 0):.6f}`",
        f"- `gt_overcall_count_avg`: `{toolchain.get('overcall_count_avg', 0):.6f}`",
        f"- `gt_missing_call_count_avg`: `{toolchain.get('missing_call_count_avg', 0):.6f}`",
        f"- `gt_arg_name_valid_rate_avg`: `{contracts.get('arg_name_valid_rate_avg', 0):.6f}`",
        f"- `gt_canonical_handle_rate_avg`: `{contracts.get('canonical_handle_rate_avg', 0):.6f}`",
        f"- `gt_answer_correct_rate`: `{answer.get('correct_rate', 0):.6f}`",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a markdown report for an SFT experiment.")
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    cfg = _read_json(run_root / "experiment_config.json")
    report_path = run_root / "EXPERIMENT_REPORT.md"

    lines: list[str] = []
    lines.append(f"# {cfg['experiment_name']}")
    lines.append("")
    lines.append("## Idea")
    lines.append("")
    lines.append(str(cfg.get("idea", "")).strip())
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append(f"- run root: `{run_root}`")
    lines.append(f"- training config: `{run_root / 'experiment_config.json'}`")
    for key in (
        "sft_data_dir",
        "targeted_manifest",
        "eval_teacher_dir",
        "test_teacher_dir",
        "teacher_train_json",
        "teacher_test_json",
        "scripted_eval_db",
        "eval_dataset",
    ):
        if cfg.get(key):
            lines.append(f"- `{key}`: `{cfg[key]}`")
    lines.append("")
    lines.append("## Role HPs")
    lines.append("")
    roles_cfg = cfg.get("roles", {}) or {}
    if roles_cfg:
        roles = list(roles_cfg)
    else:
        roles = [role.strip() for role in str(cfg.get("role_order", "ORC,GA,PA,VRA")).split(",") if role.strip()]
    for role in roles:
        role_cfg = roles_cfg.get(role, {}) if isinstance(roles_cfg, dict) else {}
        lines.append(f"### {role}")
        if role_cfg:
            for key in sorted(role_cfg):
                lines.append(f"- `{key}`: `{role_cfg[key]}`")
        else:
            for key in ("epochs", "max_length", "validation_ratio"):
                if key in cfg:
                    lines.append(f"- `{key}`: `{cfg[key]}`")
        log_path = run_root / "logs" / f"train_{role}.log"
        step, loss = _extract_steps(log_path)
        lines.append(f"- log: `{log_path}`")
        if step is not None:
            lines.append(f"- last logged step: `{step}`")
        if loss is not None:
            lines.append(f"- last logged loss: `{loss:.6f}`")
        lines.append("")

    pre_summary_path = _maybe_path(cfg.get("pre_summary_path"))
    if pre_summary_path and pre_summary_path.exists():
        lines.append("## Pre-SFT Baseline")
        lines.append("")
        lines.append(f"- summary: `{pre_summary_path}`")
        lines.extend(_metrics_block(_read_json(pre_summary_path)))
        lines.append("")

    post_eval_dir = run_root / "eval_test100_post_sft"
    if not post_eval_dir.exists():
        post_eval_dir = run_root / "eval_test50_post_sft"

    post_summary_path = post_eval_dir / "summary.json"
    if post_summary_path.exists():
        lines.append("## Post-SFT Eval")
        lines.append("")
        lines.append(f"- summary: `{post_summary_path}`")
        lines.extend(_metrics_block(_read_json(post_summary_path)))
        lines.append("")

    gt_summary_path = post_eval_dir / "gt_toolchain_metrics" / "gt_metrics_summary.json"
    if gt_summary_path.exists():
        lines.append("## Post-SFT GT Toolchain Eval")
        lines.append("")
        lines.append(f"- summary: `{gt_summary_path}`")
        lines.extend(_gt_metrics_block(_read_json(gt_summary_path)))
        lines.append("")

    lines.append("## Progress")
    lines.append("")
    lines.append("- Add notes here as the experiment progresses.")
    lines.append("- Link any follow-up comparison docs or case audits here.")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report -> {report_path}")


if __name__ == "__main__":
    main()
