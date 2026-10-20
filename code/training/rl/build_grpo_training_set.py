from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _ensure_repo_imports() -> None:
    code_root = Path(__file__).resolve().parents[2]  # code/
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))


_ensure_repo_imports()

from training.build_scripted_tool_db import _build_case_from_execution  # noqa: E402


QUOTAS = {
    "visual_bbox_geometry": 120,
    "geo_distance": 85,
    "farthest_pair": 35,
    "change_damage": 75,
    "counting": 55,
    "ocr_text": 45,
    "raster_index": 35,
    "plot_or_percentage": 25,
    "access_time_filter": 15,
    "mixed_hard": 10,
}


PRIORITY = [
    "access_time_filter",
    "farthest_pair",
    "raster_index",
    "plot_or_percentage",
    "ocr_text",
    "change_damage",
    "counting",
    "geo_distance",
    "visual_bbox_geometry",
    "mixed_hard",
]


def _categories(question: str, images: list[str]) -> set[str]:
    q = question.lower()
    cats: set[str] = set()
    if re.search(r"average distance|avg distance|mean distance|closest|nearest|distance between", q):
        cats.add("geo_distance")
    if re.search(r"farthest|furthest|largest distance|maximum distance", q):
        cats.add("farthest_pair")
    if re.search(r"outside|within \d+[- ]?minute|access|walking|driving|speed", q):
        cats.add("access_time_filter")
    if re.search(r"plot|percentage|classes|class|chart", q):
        cats.add("plot_or_percentage")
    if re.search(r"count|how many|number of", q):
        cats.add("counting")
    if re.search(r"gsd|smallest|largest|bbox|bounding box|baseball|tennis|ship|vehicle|airplane|car|aircraft", q):
        cats.add("visual_bbox_geometry")
    if re.search(r"ocr|read|text|sign|words|letters", q):
        cats.add("ocr_text")
    if re.search(r"ndvi|ndbi|nbr|index", q):
        cats.add("raster_index")
    if re.search(r"change|before|after|pre|post|damage|destroyed|major damaged|no-damage", q):
        cats.add("change_damage")
    if len(cats) >= 2:
        cats.add("mixed_hard")
    if images and not cats:
        cats.add("visual_bbox_geometry")
    if not cats:
        cats.add("geo_distance")
    return cats


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported train source: {path}")


def _record_key(record: dict[str, Any], fallback: int) -> int:
    try:
        return int(record.get("idx", fallback))
    except Exception:
        return fallback


def _select_records(records: list[dict[str, Any]], *, total: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    record_cats: dict[int, set[str]] = {}
    for pos, record in enumerate(records):
        cats = _categories(str(record.get("question", "")), list(record.get("images") or []))
        record_cats[pos] = cats
        for cat in cats:
            buckets[cat].append((pos, record))
    for rows in buckets.values():
        rng.shuffle(rows)

    selected_pos: set[int] = set()
    selected: list[dict[str, Any]] = []
    quota_hits = Counter()

    for cat in PRIORITY:
        target = QUOTAS.get(cat, 0)
        for pos, record in buckets.get(cat, []):
            if quota_hits[cat] >= target or len(selected) >= total:
                break
            if pos in selected_pos:
                continue
            selected_pos.add(pos)
            selected.append(record)
            for c in record_cats[pos]:
                quota_hits[c] += 1

    if len(selected) < total:
        remaining = [(pos, record) for pos, record in enumerate(records) if pos not in selected_pos]
        remaining.sort(
            key=lambda item: (
                -len(record_cats[item[0]]),
                _record_key(item[1], item[0]),
            )
        )
        for pos, record in remaining:
            if len(selected) >= total:
                break
            selected_pos.add(pos)
            selected.append(record)
            for c in record_cats[pos]:
                quota_hits[c] += 1

    selected = selected[:total]
    selected_counts = Counter()
    for record in selected:
        for cat in _categories(str(record.get("question", "")), list(record.get("images") or [])):
            selected_counts[cat] += 1

    summary = {
        "requested_total": total,
        "selected_total": len(selected),
        "seed": seed,
        "quotas": QUOTAS,
        "selected_category_counts": dict(selected_counts.most_common()),
        "source_category_counts": {key: len(value) for key, value in sorted(buckets.items())},
        "selected_idx_preview": [record.get("idx") for record in selected[:25]],
    }
    return selected, summary


def _write_outputs(records: list[dict[str, Any]], *, dataset_path: Path, scripted_path: Path, summary_path: Path, summary: dict[str, Any]) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    scripted_path.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    eval_records = []
    for case_id, record in enumerate(records):
        case = _build_case_from_execution(case_id, record, scripted_path.parent)
        cases.append(case)
        eval_records.append(
            {
                "idx": case["idx"],
                "question": case["question"],
                "label": case["label"],
                "images": case["images"],
            }
        )
    dataset_path.write_text(
        json.dumps({"meta": {"source": "targeted_grpo_train", "n_records": len(eval_records)}, "records": eval_records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    scripted_path.write_text(
        json.dumps({"meta": {"source": "targeted_grpo_train", "n_cases": len(cases)}, "cases": cases}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary["dataset_path"] = str(dataset_path)
    summary["scripted_tool_data_path"] = str(scripted_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 500-case targeted GRPO train set from EO train traces.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset_output", required=True)
    parser.add_argument("--scripted_output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    records = _load_records(Path(args.source).resolve())
    selected, summary = _select_records(records, total=args.total, seed=args.seed)
    _write_outputs(
        selected,
        dataset_path=Path(args.dataset_output).resolve(),
        scripted_path=Path(args.scripted_output).resolve(),
        summary_path=Path(args.summary_json).resolve(),
        summary=summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
