#!/usr/bin/env python3
"""Evaluator-side session bootstrap for the stateful GeoDisaster tool API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from tool_server.tool_workers.online_workers.final_dataset_worker_state import (  # noqa: E402
    new_state,
    save_state,
    stable_session_id,
)


def load_case(path: Path, record_id: str | None, index: int | None) -> dict:
    if path.suffix == ".jsonl":
        for i, line in enumerate(path.open(encoding="utf-8")):
            if not line.strip():
                continue
            row = json.loads(line)
            if (index is not None and i == index) or (record_id is not None and row.get("id") == record_id):
                return row
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("records") if isinstance(payload, dict) else payload
        if isinstance(rows, list):
            if index is not None:
                return rows[index]
            for row in rows:
                if row.get("id") == record_id:
                    return row
    raise ValueError("record not found; provide a valid --record-id or --index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record-id")
    group.add_argument("--index", type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--allow-fallback-roots",
        action="store_true",
        help="Permit runtime path lookup outside --dataset-root (unsafe for held-out evaluation).",
    )
    args = parser.parse_args()
    if args.dataset_root:
        os.environ["GEODISASTER_DATASET_ROOT"] = str(args.dataset_root.resolve())
        if not args.allow_fallback_roots:
            os.environ["GEODISASTER_STRICT_DATA_ROOT"] = "1"
    case = load_case(args.dataset.resolve(), args.record_id, args.index)
    session_id = args.session_id or stable_session_id(case)
    save_state(new_state(case, session_id))
    print(json.dumps({"session_id": session_id, "record_id": case.get("id"), "next_call": {"tool": "ReadData", "args": {"session_id": session_id, "question": "question_1", "input": "<initial-handle>"}}}, indent=2))


if __name__ == "__main__": main()
