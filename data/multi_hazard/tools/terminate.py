from __future__ import annotations

from typing import Any, Dict


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    status = args.get("status")
    if status not in {"TASK_COMPLETE", "INSUFFICIENT_DATA", "SOURCES_DISAGREE"}:
        return {"error": "invalid_status", "status": status}
    return {
        "status": status,
        "answer": args.get("answer", {}),
        "evidence": args.get("evidence", {}),
    }

