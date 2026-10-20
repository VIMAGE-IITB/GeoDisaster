"""Shared session and artifact helpers for the final-dataset tool workers.

This module intentionally contains no tool-specific processing policy. Each
`*_worker.py` file owns its own `process_*` function. The helpers here only
provide state loading/saving, handle bookkeeping, path lineage resolution, and
JSON-safe output conversion so the workers can share one episode context.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import uuid
from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = ROOT / "final_dataset_gt_replay" / "eo_style_43_live_gt_dataset.json"
SESSION_DIR = Path(
    os.environ.get(
        "GEODISASTER_SESSION_DIR",
        str(ROOT / "final_dataset_gt_replay" / "tool_worker_sessions"),
    )
)
ARTIFACT_DIR = Path(
    os.environ.get(
        "GEODISASTER_ARTIFACT_DIR",
        str(ROOT / "final_dataset_gt_replay" / "tool_worker_artifacts"),
    )
)


_DECLARED_READ_SCOPE: ContextVar[dict[str, Any] | None] = ContextVar(
    "worker_declared_read_scope", default=None
)


def _argument_handles(state: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Return only artifact handles explicitly opened by this tool call."""
    candidates: list[Any] = [args.get("question"), args.get("input"), args.get("work")]
    inputs = args.get("inputs")
    if isinstance(inputs, dict):
        candidates.extend(inputs.values())
    artifacts = state.get("artifacts") or {}
    return list(
        dict.fromkeys(str(value) for value in candidates if str(value) in artifacts)
    )


def declared_paths_for_call(state: dict[str, Any], args: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for handle in _argument_handles(state, args):
        for entry in all_lineage_path_entries(state, handle):
            path_value = entry.get("path")
            if not path_value:
                continue
            path = resolve_data_path(path_value).resolve()
            key = str(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def authorize_declared_read(path: Path, operation: str = "json_read") -> Path:
    resolved = Path(path).resolve()
    scope = _DECLARED_READ_SCOPE.get()
    if scope is not None and str(resolved) not in scope["allowed"]:
        raise PermissionError(f"undeclared sibling access denied ({operation}): {resolved}")
    if scope is not None:
        record = {"path": str(resolved), "operation": operation}
        if record not in scope["reads"]:
            scope["reads"].append(record)
    return resolved


@contextmanager
def declared_read_scope(state: dict[str, Any], args: dict[str, Any]):
    paths = declared_paths_for_call(state, args)
    audit: dict[str, Any] = {
        "contract": "opened_handle_exact_path_v1",
        "opened_handles": _argument_handles(state, args),
        "declared_path_count": len(paths),
        "reads": [],
    }
    token = _DECLARED_READ_SCOPE.set(
        {"allowed": {str(path) for path in paths}, "reads": audit["reads"]}
    )
    try:
        with ExitStack() as stack:
            source = source_of(state)
            if source == "deforestation_data":
                reads = stack.enter_context(deforestation_tools().DeclaredReadScope(paths))
                audit["module_reads"] = reads
            elif source == "data/multi_hazard":
                reads = stack.enter_context(mh_modules()["common"].DeclaredReadScope(paths))
                audit["module_reads"] = reads
            elif source == "ourdata/final_routing_subset_v1":
                reads = stack.enter_context(routing_tools().DeclaredReadScope(paths))
                audit["module_reads"] = reads
                visual_reads = stack.enter_context(
                    routing_visualization_tools().DeclaredReadScope(paths)
                )
                audit["visualization_reads"] = visual_reads
            yield audit
    finally:
        _DECLARED_READ_SCOPE.reset(token)


def dataset_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("GEODISASTER_DATASET_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    strict = os.environ.get("GEODISASTER_STRICT_DATA_ROOT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if strict:
        if not roots:
            raise ValueError(
                "GEODISASTER_STRICT_DATA_ROOT requires GEODISASTER_DATASET_ROOT"
            )
        return roots
    roots.extend([ROOT.resolve(), ROOT.parent.resolve(), Path.cwd().resolve(), Path.cwd().parent.resolve()])
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def resolve_data_path(path_value: Any) -> Path:
    path = Path(str(path_value or ""))
    strict = os.environ.get("GEODISASTER_STRICT_DATA_ROOT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if path.is_absolute():
        if strict:
            root_text = os.environ.get("GEODISASTER_DATASET_ROOT", "").strip()
            if not root_text:
                raise ValueError(
                    "GEODISASTER_STRICT_DATA_ROOT requires GEODISASTER_DATASET_ROOT"
                )
            strict_root = Path(root_text).expanduser().resolve()
            try:
                path.resolve().relative_to(strict_root)
            except ValueError as exc:
                raise ValueError(
                    f"absolute data path is outside strict dataset root: {path}"
                ) from exc
        return path
    for root in dataset_roots():
        candidate = root / path
        if candidate.exists():
            return candidate
    if strict:
        return dataset_roots()[0] / path
    return path


def json_default(value: Any) -> str:
    return str(value)


def jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def compact_preview(obj: Any, max_chars: int = 1800) -> Any:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=json_default)
    if len(text) <= max_chars:
        return obj
    return {"_truncated": True, "_chars": len(text), "_preview": text[:max_chars]}


def ok(text: str, **data: Any) -> dict[str, Any]:
    payload = {"text": text, "error_code": 0}
    payload.update(data)
    return payload


def error(text: str, code: int = 1, **data: Any) -> dict[str, Any]:
    payload = {"text": text, "error_code": code}
    payload.update(data)
    return payload


def collect_path_entries(obj: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if "path" in obj and "exists" in obj:
            entries.append(obj)
        for value in obj.values():
            entries.extend(collect_path_entries(value))
    elif isinstance(obj, list):
        for value in obj:
            entries.extend(collect_path_entries(value))
    return entries


def load_json(path: Path) -> Any:
    path = authorize_declared_read(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")


def session_path(session_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in session_id)
    return SESSION_DIR / f"{safe}.json"


def safe_component(value: Any) -> str:
    text = str(value or "anon")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe[:180] or "anon"


def stable_session_id(case: dict[str, Any]) -> str:
    seed = case.get("dataset_id") or case.get("id") or case.get("question_type") or json.dumps(case.get("question", ""), sort_keys=True)
    digest = hashlib.sha1(str(seed).encode("utf-8")).hexdigest()[:12]
    return f"final_{digest}"


def load_case_from_default(index: int) -> dict[str, Any]:
    payload = load_json(DEFAULT_DATASET)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"default dataset has no records list: {DEFAULT_DATASET}")
    return records[index]


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    if "initial_artifacts" in case:
        return case
    if "records" in case:
        records = case["records"]
        if not records:
            raise ValueError("case payload contains an empty records list")
        return records[0]
    raise ValueError("case must contain initial_artifacts")


def strip_private_answer_artifacts(obj: Any) -> Any:
    """Remove private answer/GT report references before workers see a case.

    Some source folders contain useful processed inputs under names such as
    ``processed_hidden``. Those are data artifacts and are kept. This only
    removes explicit answer-bearing reports, so tools cannot read GT shortcuts
    even if a raw development JSON is passed directly to ReadData.
    """
    private_markers = (
        "hidden_" + "gt",
        "gt_" + "report",
        "ground" + "_truth",
        "expected_" + "answer",
        "expected_final_" + "answer",
        "canonical_" + "answer",
        "selected_" + "routes.json",
    )
    if isinstance(obj, dict):
        path_text = str(obj.get("path") or "").lower()
        label_text = str(obj.get("label") or "").lower()
        path_parts = {part.lower() for part in Path(path_text).parts}
        private_parts = {"gt", "ground_" + "truth", "hidden", "trajectories"}
        if path_parts & private_parts or any(
            marker in path_text or marker in label_text for marker in private_markers
        ):
            return None
        out: dict[str, Any] = {}
        for key, value in obj.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in private_markers):
                continue
            cleaned = strip_private_answer_artifacts(value)
            if cleaned is not None:
                out[key] = cleaned
        return out
    if isinstance(obj, list):
        return [cleaned for item in obj if (cleaned := strip_private_answer_artifacts(item)) is not None]
    return obj


def new_state(case: dict[str, Any], session_id: str) -> dict[str, Any]:
    agent_case_keys = {
        "answer_schema",
        "available_initial_handles",
        "category",
        "dataset_id",
        "id",
        "initial_artifacts",
        "metadata",
        "question",
        "question_type",
        "raw_question",
        "sample_id",
        "source",
        "source_id",
        "split",
        "split_group_id",
        "split_seed",
    }
    safe_case = copy.deepcopy(
        {key: value for key, value in case.items() if key in agent_case_keys}
    )
    initial_artifacts = strip_private_answer_artifacts(safe_case.get("initial_artifacts") or {})
    if not initial_artifacts:
        raise ValueError("case has no initial_artifacts")
    safe_case["initial_artifacts"] = copy.deepcopy(initial_artifacts)
    return {
        "session_id": session_id,
        "artifact_run_id": uuid.uuid4().hex[:10],
        "case": safe_case,
        "artifacts": copy.deepcopy(initial_artifacts),
        "counters": {},
        "history": [],
        "answer_candidate": None,
    }


def artifact_output_dir(state: dict[str, Any]) -> Path:
    session_id = safe_component(state.get("session_id"))
    run_id = safe_component(state.get("artifact_run_id") or "run")
    return ARTIFACT_DIR / f"{session_id}__{run_id}"


def artifact_snapshot_path(state: dict[str, Any], handle: str, artifact_type: str | None) -> Path:
    counters = state.get("counters") or {}
    prefix = str(handle).split("_", 1)[0]
    ordinal = int(counters.get(prefix, 0) or 0)
    kind = safe_component(artifact_type or "artifact")
    return artifact_output_dir(state) / f"{ordinal:04d}__{safe_component(handle)}__{kind}.json"


def persist_artifact_snapshot(state: dict[str, Any], handle: str, card: dict[str, Any]) -> dict[str, Any]:
    """Persist a generated artifact under a globally unique analysis path.

    Agents only see the canonical handle (for example ``work_1`` or
    ``stats_1``). The persisted JSON gives offline analysis a stable
    session/run/handle-qualified payload, avoiding collisions when thousands of
    cases all contain a canonical ``work_1``.
    """

    global_id = "__".join(
        [
            safe_component(state.get("session_id")),
            safe_component(state.get("artifact_run_id")),
            safe_component(handle),
        ]
    )
    path = artifact_snapshot_path(state, handle, card.get("type"))
    card["canonical_handle"] = handle
    card["global_artifact_id"] = global_id
    card["artifact_json_path"] = str(path)
    write_json(
        path,
        {
            "schema_version": "geodisaster_tool_artifact_v1",
            "session_id": state.get("session_id"),
            "artifact_run_id": state.get("artifact_run_id"),
            "canonical_handle": handle,
            "global_artifact_id": global_id,
            "artifact_type": card.get("type"),
            "source_work": card.get("source_work"),
            "source_input": card.get("source_input"),
            "processor": card.get("processor"),
            "tool_args": card.get("tool_args"),
            "artifact": jsonable(card),
        },
    )
    return card


def load_state(params: dict[str, Any], require_existing: bool = True) -> tuple[str, dict[str, Any]]:
    session_id = params.get("session_id")
    if session_id:
        path = session_path(str(session_id))
        if path.exists():
            return str(session_id), load_json(path)

    case_payload = params.get("case")
    if case_payload is not None:
        case = normalize_case(case_payload)
        session_id = str(session_id or stable_session_id(case))
        return session_id, new_state(case, session_id)

    case_path = params.get("case_path")
    if case_path:
        case = normalize_case(load_json(Path(case_path)))
        session_id = str(session_id or stable_session_id(case))
        return session_id, new_state(case, session_id)

    if "case_index" in params:
        case = load_case_from_default(int(params["case_index"]))
        session_id = str(session_id or stable_session_id(case))
        return session_id, new_state(case, session_id)

    if require_existing:
        raise ValueError("missing session_id; initialize a session with ReadData using case, case_path, or case_index")
    raise ValueError("missing case/session initializer")


def save_state(state: dict[str, Any]) -> None:
    write_json(session_path(str(state["session_id"])), state)


def clear_session(session_id: str) -> None:
    path = session_path(session_id)
    if path.exists():
        path.unlink()


def require_artifact(state: dict[str, Any], handle: str) -> dict[str, Any]:
    artifacts = state["artifacts"]
    if handle not in artifacts:
        raise ValueError(f"unknown handle: {handle}")
    card = artifacts[handle]
    if not isinstance(card, dict):
        raise ValueError(f"artifact {handle} is not an object")
    return card


def require_question(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    question_handle = args.get("question")
    if not question_handle:
        raise ValueError("missing required arg: question")
    card = require_artifact(state, str(question_handle))
    if card.get("type") != "question":
        raise ValueError(f"{question_handle} is not a question handle")
    return card


def new_handle(state: dict[str, Any], prefix: str, payload: dict[str, Any]) -> str:
    counters = Counter(state.get("counters") or {})
    counters[prefix] += 1
    state["counters"] = dict(counters)
    handle = f"{prefix}_{counters[prefix]}"
    card = {"handle": handle, **jsonable(payload)}
    state["artifacts"][handle] = persist_artifact_snapshot(state, handle, card)
    return handle


def actual_path_status(entry: dict[str, Any]) -> dict[str, Any]:
    path_value = entry.get("path")
    resolved_path = resolve_data_path(path_value) if path_value else Path("")
    path_exists = resolved_path.exists() if path_value else False
    return {
        "path": path_value,
        "resolved_path": str(resolved_path) if path_value else "",
        "label": entry.get("label"),
        "declared_exists": bool(entry.get("exists")),
        "actual_exists": path_exists,
        "optional": bool(entry.get("optional")),
    }


def path_summary(card: dict[str, Any]) -> dict[str, Any]:
    entries = [actual_path_status(entry) for entry in collect_path_entries(card.get("paths", card))]
    missing_required = [entry for entry in entries if not entry["optional"] and not entry["actual_exists"]]
    suffix_counts: Counter[str] = Counter()
    byte_total = 0
    json_preview: dict[str, Any] = {}
    sensitive_terms = ("gt", "ground" + "_truth", "expected", "ans" + "wer", "canon" + "ical")
    for entry in entries:
        path_value = entry.get("path")
        if not path_value or not entry["actual_exists"]:
            continue
        path = resolve_data_path(path_value)
        suffix_counts[path.suffix.lower() or "<dir>"] += 1
        if path.is_file():
            try:
                byte_total += path.stat().st_size
            except OSError:
                pass
        path_text = f"{path.name} {path_value} {entry.get('label', '')}".lower()
        if any(term in path_text for term in sensitive_terms):
            continue
        if path.suffix.lower() in {".json", ".geojson"} and len(json_preview) < 5:
            try:
                payload = load_json(path)
                if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
                    json_preview[path.name] = {
                        "type": "FeatureCollection",
                        "features": len(payload.get("features") or []),
                        "top_level_keys": sorted(payload.keys())[:12],
                    }
                elif isinstance(payload, dict):
                    json_preview[path.name] = {"type": "dict", "keys": sorted(payload.keys())[:20]}
                elif isinstance(payload, list):
                    json_preview[path.name] = {"type": "list", "items": len(payload)}
            except Exception as exc:
                json_preview[path.name] = {"read_error": str(exc)}
    return {
        "path_count": len(entries),
        "missing_required_path_count": len(missing_required),
        "missing_required_paths": missing_required,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "byte_total": byte_total,
        "json_preview": json_preview,
    }


def source_of(state: dict[str, Any]) -> str | None:
    case = state.get("case") or {}
    source = case.get("source")
    if source:
        return str(source)
    record_id = str(case.get("dataset_id") or case.get("id") or "")
    family = record_id.split("/", 1)[0]
    public_to_internal = {
        "deforestation": "deforestation_data",
        "multi_hazard": "data/multi_hazard",
        "xbd_damage": "geodisaster",
        "routing": "ourdata/final_routing_subset_v1",
        "sen1_flood": "ourdata/sen1_hand_final_hidden_eval_5q",
    }
    return public_to_internal.get(family)


def artifact_result(state: dict[str, Any], handle: str | None) -> Any:
    if not handle:
        return None
    try:
        card = require_artifact(state, str(handle))
    except ValueError:
        return None
    return card.get("result") or card.get("evidence")


def initial_handles_for(state: dict[str, Any], handle: str, seen: set[str] | None = None) -> list[str]:
    seen = seen or set()
    if handle in seen:
        return []
    seen.add(handle)
    try:
        card = require_artifact(state, handle)
    except ValueError:
        return []
    case_initial = set((state.get("case") or {}).get("initial_artifacts") or {})
    if handle in case_initial:
        return [handle]
    out: list[str] = []
    if card.get("source_input"):
        out.extend(initial_handles_for(state, str(card["source_input"]), seen))
    if card.get("source_work"):
        out.extend(initial_handles_for(state, str(card["source_work"]), seen))
    inputs = card.get("inputs")
    if isinstance(inputs, dict):
        for child in inputs.values():
            out.extend(initial_handles_for(state, str(child), seen))
    return list(dict.fromkeys(out))


def initial_cards_for(state: dict[str, Any], handle: str) -> list[dict[str, Any]]:
    return [require_artifact(state, h) for h in initial_handles_for(state, handle)]


def all_lineage_path_entries(state: dict[str, Any], handle: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for card in initial_cards_for(state, handle):
        entries.extend(collect_path_entries(card.get("paths", card)))
    resolved_entries: list[dict[str, Any]] = []
    for entry in entries:
        copied = dict(entry)
        if copied.get("path"):
            copied["declared_path"] = copied["path"]
            copied["path"] = str(resolve_data_path(copied["path"]))
        resolved_entries.append(copied)
    return resolved_entries


def declared_sar_windows(state: dict[str, Any], handle: str) -> dict[str, dict[str, Any]]:
    """Return authoritative per-file SAR time metadata from opened handles.

    Deforestation SAR workers must not invent acquisition dates or infer
    pre/post membership from an undeclared sibling report. The canonical handle
    cards carry the source acquisition date and temporal role on each S1 path.
    """

    windows: dict[str, dict[str, Any]] = {}
    for entry in all_lineage_path_entries(state, handle):
        path_value = entry.get("path")
        if not path_value:
            continue
        path = Path(path_value)
        if path.suffix.lower() not in {".tif", ".tiff"} or not path.stem.startswith("s1_"):
            continue
        role = str(entry.get("temporal_role") or "").lower()
        date = entry.get("acquisition_date")
        if role not in {"pre", "post"} or not date:
            raise ValueError(
                f"declared SAR path lacks authoritative temporal metadata: {entry.get('declared_path') or path}"
            )
        windows[path.stem] = {
            "temporal_role": role,
            "target_date": str(date),
            "metadata_source": "declared_handle_metadata",
        }
    if not windows:
        raise ValueError("opened handle lineage contains no declared SAR windows")
    return windows


def first_existing_path(entries: list[dict[str, Any]], *needles: str) -> Path | None:
    lowered = [n.lower() for n in needles if n]
    for entry in entries:
        path_value = entry.get("path")
        if not path_value:
            continue
        text = f"{path_value} {entry.get('label', '')}".lower()
        resolved_path = resolve_data_path(path_value)
        if all(n in text for n in lowered) and resolved_path.exists():
            return resolved_path
    return None


def dir_from_path_parts(path: Path, marker: str, depth_after_marker: int = 1) -> Path | None:
    parts = path.parts
    if marker not in parts:
        if marker == "samples_v2" and "source_assets" in parts:
            idx = parts.index("source_assets")
            end = idx + 3
            if len(parts) >= end:
                return Path(*parts[:end])
        return None
    idx = parts.index(marker)
    end = idx + 1 + depth_after_marker
    if len(parts) >= end:
        return Path(*parts[:end])
    return None


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@lru_cache(maxsize=1)
def deforestation_tools():
    return load_module_from_path("final_dataset_deforestation_tools", ROOT / "deforestation_data" / "tools" / "tools.py")


@lru_cache(maxsize=1)
def sen1_utils():
    return load_module_from_path(
        "final_dataset_sen1_utils",
        ROOT / "ourdata" / "sen1_hand_final_hidden_eval_5q" / "scripts" / "utils_sen1.py",
    )


@lru_cache(maxsize=1)
def mh_modules():
    mh_root = ROOT / "data" / "multi_hazard"
    if str(mh_root) not in sys.path:
        sys.path.insert(0, str(mh_root))
    import tools._tool_common as common
    import tools.compare_across_sources as compare_across_sources
    import tools.compute_region_stats as compute_region_stats
    import tools.compute_temporal_anomaly as compute_temporal_anomaly
    import tools.query_exposure as query_exposure
    import mhds_common

    return {
        "common": common,
        "stats": compute_region_stats,
        "compare": compare_across_sources,
        "temporal": compute_temporal_anomaly,
        "exposure": query_exposure,
        "common_defs": mhds_common,
    }


@lru_cache(maxsize=1)
def routing_tools():
    return load_module_from_path(
        "final_dataset_routing_tools",
        ROOT / "ourdata" / "final_routing_subset_v1" / "tools" / "routing_tools.py",
    )


@lru_cache(maxsize=1)
def routing_visualization_tools():
    return load_module_from_path(
        "final_dataset_routing_visualization_tools",
        ROOT / "ourdata" / "final_routing_subset_v1" / "tools" / "visualization_tools.py",
    )


def store_result_handle(
    state: dict[str, Any],
    prefix: str,
    artifact_type: str,
    tool_name: str,
    args: dict[str, Any],
    work: str,
    processor: str,
    result: Any,
) -> dict[str, Any]:
    result = jsonable(result)
    handle = new_handle(
        state,
        prefix,
        {
            "type": artifact_type,
            "source_work": work,
            "tool_args": copy.deepcopy(args),
            "processor": processor,
            "result": result,
        },
    )
    state["answer_candidate"] = result
    return {
        "status": "OK",
        "handle": handle,
        "canonical_handle": handle,
        "global_artifact_id": state["artifacts"][handle].get("global_artifact_id"),
        "artifact_json_path": state["artifacts"][handle].get("artifact_json_path"),
        "source_work": work,
        "processor": processor,
        # The model-facing observation layer will compact this again before it
        # enters a prompt. Keep the worker return itself rich enough for
        # downstream answer-package references such as ANSWER_JSON:rank_1.
        "result": compact_preview(result, 20000),
    }


def run_stateful_tool(
    tool_name: str,
    params: dict[str, Any] | None,
    process_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    require_existing: bool = True,
) -> dict[str, Any]:
    args = dict(params or {})
    try:
        session_id, state = load_state(args, require_existing=require_existing)
        state["session_id"] = session_id
        with declared_read_scope(state, args) as read_audit:
            output = process_fn(state, args)
        module_reads = read_audit.pop("module_reads", [])
        visualization_reads = read_audit.pop("visualization_reads", [])
        combined_reads = (
            read_audit.pop("reads", []) + module_reads + visualization_reads
        )
        unique_reads = []
        for item in combined_reads:
            if item not in unique_reads:
                unique_reads.append(item)
        read_audit["reads"] = unique_reads
        read_audit["read_count"] = len(unique_reads)
        read_audit["undeclared_read_count"] = 0
        read_audit["private_answer_read_count"] = 0
        output["read_authorization"] = read_audit
        state.setdefault("history", []).append(
            {
                "tool": tool_name,
                "args": args,
                "read_authorization": copy.deepcopy(read_audit),
                "output": compact_preview(output, 3200),
            }
        )
        save_state(state)
        return ok(str(output.get("status", "OK")), **output)
    except Exception as exc:
        return error(f"{tool_name} failed: {exc}")
