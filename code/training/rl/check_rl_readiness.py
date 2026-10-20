"""Readiness checks for the supported live-worker GeoDisaster GRPO pipeline."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

try:
    from .rl_common import ROLES, iter_jsonl, resolve_role_adapter, write_json
except ImportError:  # pragma: no cover - direct script execution
    from rl_common import ROLES, iter_jsonl, resolve_role_adapter, write_json


def _adapter_check(root: Path, roles: list[str], preferred_subdir: str = "") -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for role in roles:
        path = resolve_role_adapter(root, role, preferred_subdir=preferred_subdir)
        config = path / "adapter_config.json"
        weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
        checks[role] = {
            "path": str(path),
            "config": config.is_file(),
            "weights": any(candidate.is_file() for candidate in weights),
            "ok": config.is_file() and any(candidate.is_file() for candidate in weights),
        }
    return {"path": str(root), "roles": checks, "ok": all(row["ok"] for row in checks.values())}


def _model_check(reference: str) -> dict[str, Any]:
    path = Path(reference).expanduser()
    if path.is_dir():
        config = path / "config.json"
        return {"reference": reference, "kind": "local", "config": str(config), "ok": config.is_file()}
    looks_remote = bool(reference) and not path.is_absolute() and not reference.startswith((".", "~")) and reference.count("/") == 1
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"} or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
    return {
        "reference": reference,
        "kind": "remote" if looks_remote else "missing_local",
        "offline": offline,
        "ok": bool(looks_remote and not offline),
    }


def _logical_type(row: dict[str, Any]) -> str:
    record_id = str(row.get("dataset_id") or row.get("id") or "")
    parts = record_id.split("/")
    question_type = str(row.get("question_type") or (parts[-1] if parts else "unknown"))
    if len(parts) >= 2 and parts[0] == "multi_hazard":
        return f"multi_hazard/{parts[1]}/{question_type}"
    return f"{parts[0] if parts and parts[0] else 'unknown'}/{question_type}"


def _path_cards(value: Any) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and ("optional" in value or "exists" in value):
            cards.append(value)
        for child in value.values():
            cards.extend(_path_cards(child))
    elif isinstance(value, list):
        for child in value:
            cards.extend(_path_cards(child))
    return cards


def _training_data_check(train_jsonl: Path, dataset_root: Path, expected_types: int) -> dict[str, Any]:
    if not train_jsonl.is_file():
        return {"path": str(train_jsonl), "records": 0, "logical_types": 0, "ok": False}
    rows = list(iter_jsonl(train_jsonl))
    ids = [str(row.get("dataset_id") or row.get("id") or "") for row in rows]
    logical_types = {_logical_type(row) for row in rows}
    missing: set[str] = set()
    outside: set[str] = set()
    for row in rows:
        for card in _path_cards(row.get("initial_artifacts") or {}):
            if card.get("optional"):
                continue
            declared = Path(str(card["path"]))
            resolved = declared if declared.is_absolute() else dataset_root / declared
            try:
                resolved.resolve().relative_to(dataset_root.resolve())
            except ValueError:
                outside.add(str(card["path"]))
            if not resolved.is_file():
                missing.add(str(card["path"]))
    ok = bool(rows) and all(ids) and len(ids) == len(set(ids)) and len(logical_types) == expected_types and not missing and not outside
    return {
        "path": str(train_jsonl),
        "records": len(rows),
        "unique_ids": len(set(ids)),
        "logical_types": len(logical_types),
        "expected_logical_types": expected_types,
        "required_assets_missing": len(missing),
        "required_assets_outside_root": len(outside),
        "missing_examples": sorted(missing)[:20],
        "outside_examples": sorted(outside)[:20],
        "ok": ok,
    }


def _scorer_check(contract_root: Path) -> dict[str, Any]:
    from evaluation.structured_answer import DEFAULT_ABS_TOL, DEFAULT_REL_TOL, SCORING_POLICY_VERSION

    path = contract_root / "data/core/scoring_policy_43.json"
    try:
        declared = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "error": str(exc), "ok": False}
    numeric = declared.get("numeric") or {}
    ok = (
        declared.get("schema_version") == SCORING_POLICY_VERSION
        and numeric.get("absolute_tolerance") == DEFAULT_ABS_TOL
        and numeric.get("relative_tolerance") == DEFAULT_REL_TOL
    )
    return {
        "path": str(path),
        "declared_version": declared.get("schema_version"),
        "executable_version": SCORING_POLICY_VERSION,
        "absolute_tolerance": DEFAULT_ABS_TOL,
        "relative_tolerance": DEFAULT_REL_TOL,
        "ok": ok,
    }


def _runtime_check() -> dict[str, Any]:
    import numpy as np

    backend = os.environ.get("GEODISASTER_TOOL_BACKEND", "inprocess").strip().lower()
    scripted = os.environ.get("MAS_SCRIPTED_TOOL_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "required_numpy": "1.26.4",
        "tool_backend": backend,
        "scripted_tools_enabled": scripted,
        "ok": np.__version__ == "1.26.4" and backend in {"inprocess", "http"} and not scripted,
    }


def build_live_grpo_report(
    *,
    model_name_or_path: str,
    policy_adapter_root: Path,
    ref_adapter_root: Path,
    train_jsonl: Path,
    dataset_root: Path,
    contract_root: Path | None,
    roles: list[str],
    policy_adapter_subdir: str = "",
    ref_adapter_subdir: str = "",
    expected_types: int = 43,
) -> dict[str, Any]:
    report = {
        "pipeline": "training.train_grpo_pipeline",
        "roles": roles,
        "model": _model_check(model_name_or_path),
        "policy_adapters": _adapter_check(policy_adapter_root, roles, policy_adapter_subdir),
        "reference_adapters": _adapter_check(ref_adapter_root, roles, ref_adapter_subdir),
        "training_data": _training_data_check(train_jsonl, dataset_root, expected_types),
        "scorer": _scorer_check(contract_root or dataset_root),
        "runtime": _runtime_check(),
    }
    report["ready_for_grpo_training"] = all(
        report[key]["ok"]
        for key in ("model", "policy_adapters", "reference_adapters", "training_data", "scorer", "runtime")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the supported real-worker GeoDisaster GRPO pipeline.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--policy_adapter_root", "--sft_adapter_root", dest="policy_adapter_root", required=True)
    parser.add_argument("--ref_adapter_root", default="", help="Defaults to --policy_adapter_root.")
    parser.add_argument("--policy_adapter_subdir", default="")
    parser.add_argument("--ref_adapter_subdir", default="")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dataset_root", default=os.environ.get("GEODISASTER_DATASET_ROOT", ""))
    parser.add_argument(
        "--contract_root",
        default=os.environ.get("GEODISASTER_CONTRACT_ROOT", ""),
        help="Root containing the final scoring policy; defaults to --dataset_root.",
    )
    parser.add_argument("--roles", default=",".join(ROLES))
    parser.add_argument("--expected_types", type=int, default=43)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    roles = [role.strip().upper() for role in args.roles.split(",") if role.strip()]
    invalid = sorted(set(roles) - set(ROLES))
    if invalid:
        raise SystemExit(f"Unsupported roles: {invalid}")
    if not args.dataset_root:
        raise SystemExit("--dataset_root or GEODISASTER_DATASET_ROOT is required")
    policy_root = Path(args.policy_adapter_root).expanduser().resolve()
    report = build_live_grpo_report(
        model_name_or_path=args.model_name_or_path,
        policy_adapter_root=policy_root,
        ref_adapter_root=Path(args.ref_adapter_root).expanduser().resolve() if args.ref_adapter_root else policy_root,
        train_jsonl=Path(args.train_jsonl).expanduser().resolve(),
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        contract_root=Path(args.contract_root).expanduser().resolve() if args.contract_root else None,
        roles=roles,
        policy_adapter_subdir=args.policy_adapter_subdir,
        ref_adapter_subdir=args.ref_adapter_subdir,
        expected_types=args.expected_types,
    )
    if args.output_json:
        write_json(Path(args.output_json).expanduser().resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.strict and not report["ready_for_grpo_training"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
