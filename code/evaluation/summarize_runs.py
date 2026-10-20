#!/usr/bin/env python3
"""Summarize GeoDisaster online eval runs into one comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "geodisaster_eval_comparison.md"


def load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.exists():
        return {"missing": True, "path": str(summary_path)}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["path"] = str(summary_path)
    return payload


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}"
    except (TypeError, ValueError):
        return "NA"


def scalar(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def metric(summary: dict[str, Any], key: str) -> Any:
    if summary.get("missing"):
        return None
    return summary.get(key)


def render(runs: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "# GeoDisaster Eval Comparison",
        "",
        "All rows use the same held-out GeoDisaster test split unless noted.",
        "",
        "| Run | N | Task Complete % | Correct % | Answer Acc | Token F1 | Seq Ratio | Inst | Tool/TSR | ArgN | ArgV | Summ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in runs:
        align = summary.get("alignment_metrics") if isinstance(summary.get("alignment_metrics"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(summary.get("n_samples", "NA")) if not summary.get("missing") else "missing",
                    pct(metric(summary, "task_complete_rate")),
                    pct(metric(summary, "correct_rate")),
                    scalar(metric(summary, "answer_acc_avg")),
                    scalar(metric(summary, "token_f1_avg")),
                    scalar(metric(summary, "sequence_ratio_avg")),
                    pct(align.get("Inst") if align else None),
                    pct(align.get("TSR") if align else None),
                    pct(align.get("ArgN") if align else None),
                    pct(align.get("ArgV") if align else None),
                    pct(align.get("Summ") if align else None),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Source Files",
            "",
        ]
    )
    for name, summary in runs:
        lines.append(f"- {name}: `{summary.get('path')}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="NAME=eval_dir_or_summary_json")
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Provide at least one --run NAME=path")

    runs: list[tuple[str, dict[str, Any]]] = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run must be NAME=path, got {item!r}")
        name, raw_path = item.split("=", 1)
        runs.append((name, load_summary(Path(raw_path).resolve())))
    text = render(runs)
    args.output_md.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_md.resolve().write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
