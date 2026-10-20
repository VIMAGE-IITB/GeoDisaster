"""Real GeoDisaster tool gateway used by agents, training, and evaluation.

This module is the only agent-facing execution boundary for the finalized
GeoDisaster tools.  It injects evaluator-owned session context, validates the
public call contract, reconciles ``ReadData`` calls with declared artifact
cards, and dispatches to either the real in-process worker function or the
real HTTP worker service.

No fixture or expected-answer data is used here.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from .final_dataset_profile import FINAL_DATASET_AGENT_TOOLS, FINAL_DATASET_TOOLS
from .geodisaster_observation import (
    make_geodisaster_state_fields,
    summarize_geodisaster_tool_result,
)
from .schemas import Produced, ToolResult


AGENT_TOOLS: dict[str, list[str]] = {
    role: list(tools) for role, tools in FINAL_DATASET_AGENT_TOOLS.items()
}

_CONTEXT: dict[str, Any] = {}
_HTTP_MANAGER: Any | None = None
_GENERIC_CARD_VALUES = {None, "", "requested"}


def _strip_hidden_paths(obj: Any) -> Any:
    """Remove evaluator-only and ground-truth path declarations recursively."""

    if isinstance(obj, dict):
        path_text = str(obj.get("path") or "").lower()
        label_text = str(obj.get("label") or "").lower()
        path_parts = {part.lower() for part in Path(path_text).parts}
        private_parts = {"gt", "ground_truth", "hidden", "trajectories"}
        private_names = {
            "gt_report",
            "hidden_gt",
            "expected_answer",
            "expected_final_answer",
            "canonical_answer",
            "selected_routes.json",
        }
        if path_parts & private_parts or any(
            marker in path_text or marker in label_text for marker in private_names
        ):
            return None
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            item = _strip_hidden_paths(value)
            if item is not None:
                cleaned[key] = item
        return cleaned
    if isinstance(obj, list):
        return [
            cleaned
            for item in obj
            if (cleaned := _strip_hidden_paths(item)) is not None
        ]
    return obj


def _sanitize_case(case: dict[str, Any]) -> dict[str, Any]:
    """Keep only question-visible metadata required by real workers."""

    allowed = {
        "dataset_id",
        "id",
        "sample_id",
        "source",
        "source_id",
        "question_type",
        "question",
        "raw_question",
        "initial_artifacts",
        "available_initial_handles",
        "metadata",
        "category",
        "answer_schema",
    }
    clean = {key: deepcopy(value) for key, value in case.items() if key in allowed}
    if "initial_artifacts" in clean:
        clean["initial_artifacts"] = _strip_hidden_paths(clean["initial_artifacts"])
    return clean


def set_final_dataset_context(case: dict[str, Any], session_id: str | None = None) -> str:
    """Install one sanitized benchmark case for subsequent stateful calls."""

    sid = session_id or f"fd_{uuid.uuid4().hex[:12]}"
    _CONTEXT.clear()
    _CONTEXT.update(
        {
            "case": _sanitize_case(dict(case)),
            "session_id": sid,
            "started": False,
        }
    )
    return sid


def clear_final_dataset_context() -> None:
    _CONTEXT.clear()


def current_sanitized_case() -> dict[str, Any] | None:
    case = _CONTEXT.get("case")
    return deepcopy(case) if isinstance(case, dict) else None


def is_tool_allowed(agent: str, tool_name: str) -> bool:
    if tool_name == "TaskComplete":
        return True
    return tool_name in AGENT_TOOLS.get(agent, [])


def _read_data_card(args: dict[str, Any]) -> dict[str, Any] | None:
    case = _CONTEXT.get("case")
    if not isinstance(case, dict):
        return None
    artifacts = case.get("initial_artifacts")
    if not isinstance(artifacts, dict):
        return None
    card = artifacts.get(str(args.get("input") or ""))
    return card if isinstance(card, dict) else None


def normalize_read_data_args(
    args: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Resolve ``ReadData`` semantics without letting stale calls override cards.

    A specific artifact-card role/time is authoritative.  A generic card value
    (currently ``requested`` or absent) delegates that choice to the call.  A
    contradiction is returned as a structured agent-contract error instead of
    being silently rewritten or reaching a worker as an infrastructure error.
    """

    normalized = dict(args or {})
    options = dict(normalized.get("options") or {}) if isinstance(normalized.get("options"), dict) else {}
    errors: list[str] = []
    notes: list[str] = []
    card = _read_data_card(normalized)
    if card is None:
        return normalized, errors, notes

    top_role = normalized.get("role")
    option_role = options.get("role")
    if top_role and option_role and top_role != option_role:
        errors.append(
            f"ReadData role conflict between role={top_role!r} and options.role={option_role!r}"
        )
    supplied_role = top_role or option_role
    card_role = card.get("role")
    if card_role not in _GENERIC_CARD_VALUES:
        if supplied_role and supplied_role != card_role:
            errors.append(
                f"ReadData role {supplied_role!r} contradicts authoritative artifact-card role {card_role!r}"
            )
        elif not supplied_role:
            normalized["role"] = card_role
            notes.append(f"role inherited from artifact card: {card_role}")
    elif not supplied_role:
        errors.append("ReadData requires role because the artifact card does not declare a specific role")

    top_time = normalized.get("time")
    option_time = options.get("time")
    if top_time and option_time and top_time != option_time:
        errors.append(
            f"ReadData time conflict between time={top_time!r} and options.time={option_time!r}"
        )
    supplied_time = top_time or option_time
    metadata = card.get("metadata") if isinstance(card.get("metadata"), dict) else {}
    card_time = card.get("time") or metadata.get("time")
    if card_time not in _GENERIC_CARD_VALUES:
        if supplied_time and supplied_time != card_time:
            errors.append(
                f"ReadData time {supplied_time!r} contradicts authoritative artifact-card time {card_time!r}"
            )
        elif not supplied_time:
            normalized["time"] = card_time
            notes.append(f"time inherited from artifact card: {card_time}")

    return normalized, errors, notes


def _failure(
    tool: str,
    message: str,
    category: str,
    *,
    original_args: dict[str, Any] | None = None,
    normalized_args: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> ToolResult:
    fields: dict[str, Any] = {
        "latest_tool": tool,
        "latest_tool_ok": False,
        "failure_category": category,
    }
    if original_args is not None:
        fields["original_tool_args"] = original_args
    if normalized_args is not None:
        fields["normalized_tool_args"] = normalized_args
    if errors:
        fields["argument_errors"] = errors
    return ToolResult(ok=False, text=message, produced=Produced(fields=fields))


def _worker_failure_category(text: str) -> str:
    lower = str(text or "").lower()
    if "unknown handle" in lower or "expects scene handles" in lower or "not a question handle" in lower:
        return "wrong_handle_type"
    if "missing required arg" in lower or "missing required argument" in lower:
        return "missing_required_argument"
    if any(token in lower for token in ("unsupported", "expected one of", "supports only")):
        return "invalid_enum"
    if any(
        token in lower
        for token in (
            "missing declared inputs",
            "missing required",
            "requires an opened",
            "could not resolve",
            "is not the required",
            "does not contain",
        )
    ):
        return "missing_dependency"
    return "worker_rejection"


def _ensure_inprocess_imports() -> None:
    """Import real process functions without starting HTTP server machinery."""

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Worker modules define their real ``process_<Tool>`` function alongside a
    # server wrapper.  Lightweight server shims prevent logger/server startup
    # side effects in in-process mode; they do not replace tool computation.
    base_name = "tool_server.tool_workers.online_workers.base_tool_worker"
    if base_name not in sys.modules:
        base_mod = types.ModuleType(base_name)

        class BaseToolWorker:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def run(self) -> None:
                pass

        base_mod.BaseToolWorker = BaseToolWorker
        sys.modules[base_name] = base_mod

    server_name = "tool_server.utils.server_utils"
    if server_name not in sys.modules:
        server_mod = types.ModuleType(server_name)

        class _Logger:
            def info(self, *args: Any, **kwargs: Any) -> None:
                pass

            def warning(self, *args: Any, **kwargs: Any) -> None:
                pass

            def error(self, *args: Any, **kwargs: Any) -> None:
                pass

        server_mod.build_logger = lambda *args, **kwargs: _Logger()
        sys.modules[server_name] = server_mod


def call_tool(name: str, args: dict[str, Any]) -> ToolResult:
    """Call one finalized GeoDisaster tool through the selected real backend."""

    from tool_server.tool_workers.online_workers.final_dataset_tool_specs import (
        validate_agent_args,
    )

    original_args = dict(args or {})
    if name not in FINAL_DATASET_TOOLS:
        return _failure(name, f"Unknown GeoDisaster tool: {name}", "unknown_tool")
    if not _CONTEXT:
        return _failure(
            name,
            "GeoDisaster case context was not installed before tool call.",
            "session_error",
            original_args=original_args,
        )

    normalized_args = dict(original_args)
    normalization_notes: list[str] = []
    if name == "ReadData":
        normalized_args, semantic_errors, normalization_notes = normalize_read_data_args(
            normalized_args
        )
        if semantic_errors:
            return _failure(
                name,
                f"ReadData semantic contract error: {'; '.join(semantic_errors)}",
                "agent_semantic_conflict",
                original_args=original_args,
                normalized_args=normalized_args,
                errors=semantic_errors,
            )

    validation_errors = validate_agent_args(name, normalized_args)
    if validation_errors:
        return _failure(
            name,
            f"{name} argument contract error: {'; '.join(validation_errors)}",
            "agent_invalid_args",
            original_args=original_args,
            normalized_args=normalized_args,
            errors=validation_errors,
        )

    call_args = dict(normalized_args)
    call_args["session_id"] = str(_CONTEXT["session_id"])
    if name != "ReadData" and not _CONTEXT.get("started"):
        return _failure(
            name,
            (
                f"{name} requires an opened GeoDisaster scene/work handle. "
                "Call ReadData on an available input handle first."
            ),
            "session_error",
            original_args=original_args,
            normalized_args=normalized_args,
        )
    if name == "ReadData" and not _CONTEXT.get("started"):
        call_args["case"] = _CONTEXT["case"]

    backend = os.environ.get("GEODISASTER_TOOL_BACKEND", "inprocess").strip().lower()
    try:
        if backend == "http":
            global _HTTP_MANAGER
            if _HTTP_MANAGER is None:
                from tool_server.tool_workers.tool_manager.base_manager import ToolManager

                _HTTP_MANAGER = ToolManager(
                    os.environ.get("MAS_CONTROLLER_ADDR", "http://127.0.0.1:20101"),
                    timeout=float(os.environ.get("GEODISASTER_TOOL_TIMEOUT", "120")),
                )
            out = _HTTP_MANAGER.call_tool(name, call_args)
        elif backend == "inprocess":
            _ensure_inprocess_imports()
            from tool_server.tool_workers.online_workers.final_dataset_worker_state import (
                run_stateful_tool,
            )

            module = importlib.import_module(
                f"tool_server.tool_workers.online_workers.{name}_worker"
            )
            process = getattr(module, f"process_{name}")
            out = run_stateful_tool(
                name,
                call_args,
                process,
                require_existing=(name != "ReadData"),
            )
        else:
            return _failure(
                name,
                f"Unsupported GEODISASTER_TOOL_BACKEND={backend!r}; use inprocess or http.",
                "configuration_error",
                original_args=original_args,
                normalized_args=normalized_args,
            )
    except Exception as exc:
        category = "transport_error" if backend == "http" else "worker_error"
        return _failure(
            name,
            f"{name} {category}: {exc}",
            category,
            original_args=original_args,
            normalized_args=normalized_args,
        )

    ok = int(out.get("error_code", 1)) == 0
    if name == "ReadData" and ok:
        _CONTEXT["started"] = True
    raw_text = str(out.get("text") or ("OK" if ok else "ERROR"))
    handle = out.get("handle")
    summary = summarize_geodisaster_tool_result(
        name,
        handle=str(handle) if handle else None,
        text=raw_text,
        result=out.get("result"),
        evidence=out.get("evidence"),
        ok=ok,
    )
    fields = make_geodisaster_state_fields(
        name,
        handle=str(handle) if handle else None,
        ok=ok,
        summary=summary,
        full_result=out.get("result"),
    )
    if normalization_notes:
        fields["argument_normalization"] = normalization_notes
    if original_args != normalized_args:
        fields["original_tool_args"] = original_args
        fields["normalized_tool_args"] = normalized_args
    if not ok and "failure_category" not in fields:
        fields["failure_category"] = _worker_failure_category(raw_text)
    text = summary.get("summary") or raw_text
    return ToolResult(ok=ok, text=text, produced=Produced(fields=fields))


__all__ = [
    "AGENT_TOOLS",
    "call_tool",
    "clear_final_dataset_context",
    "current_sanitized_case",
    "is_tool_allowed",
    "normalize_read_data_args",
    "set_final_dataset_context",
]
