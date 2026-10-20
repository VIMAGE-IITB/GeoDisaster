"""Episode state — the canonical truth.

The runtime owns this. Agents never accumulate chat history themselves; every
LLM call's user message is a deterministic projection of this state.

The state is a ``TypedDict`` (LangGraph-friendly). Convenience constructors and
mutators live below as plain functions so the data and the operations are
syntactically separate.

We deliberately do NOT use LangGraph reducers on ``history``. Each node
returns the (mutated) state dict whole — channels replace, not append — and
that keeps mutation semantics simple and readable. With reducers the same
events would double on each return.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, TypedDict

from .schemas import ArtifactRecord, LayerRecord, Plan


# ---------------------------------------------------------------------------
# Records held in state
# ---------------------------------------------------------------------------


class FieldVal(TypedDict):
    """A single named field value, latest-wins."""

    value: Any
    produced_by_step: str  # "" if produced before any step (system input)
    ts: float


class Event(TypedDict):
    """One entry in the canonical episode trace."""

    role: Literal["user", "ORC", "GA", "PA", "VRA", "tool", "system"]
    phase: Literal[
        "question",
        "plan",
        "delegate",
        "step",
        "tool_call",
        "complete",
        "decide",
        "terminate",
        "replan",
        "abort",
    ]
    step_id: str | None
    turn: int | None
    payload: dict[str, Any]
    ts: float


class StepRecord(TypedDict):
    """A step from the plan plus its runtime status and last summary."""

    id: str
    agent: Literal["GA", "PA", "VRA"]
    subtask: str
    needs: list[str]
    status: Literal["pending", "running", "done", "failed"]
    summary: str  # last specialist summary for this step


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------


class EpisodeState(TypedDict):
    question: str
    images: list[str]
    question_facts: dict[str, list[str]]
    scenario_case_id: int
    scenario_session_id: str

    plan: Plan | None
    steps: list[StepRecord]  # mutable view of plan.steps with status + summary
    current_step_id: str | None

    artifacts: dict[str, dict[str, Any]]  # handle -> {kind, path, produced_by_step, note}
    layers: dict[str, list[dict[str, Any]]]  # gpkg_handle -> [{name, produced_by_step}]
    fields: dict[str, FieldVal]  # name -> FieldVal (latest-wins)
    # Stable answer-ref -> schema-valid projection from successful worker
    # evidence. This is runtime state, never a worker artifact or gold answer.
    answer_registry: dict[str, dict[str, Any]]

    history: list[Event]  # append-only; replaced channel-wise on each node return
    contract_window: dict[str, int]  # step_id -> history index when contract started

    counters: dict[str, int]  # {"gpkg": N, "img": N, "tif": N} for handle assignment

    replans_left: int
    final_answer: str | None
    status: Literal["RUNNING", "DONE", "ABORTED"]
    abort_reason: str


# ---------------------------------------------------------------------------
# Constructors / mutators (pure functions returning state updates)
# ---------------------------------------------------------------------------


def new_episode(
    question: str,
    images: list[str] | None = None,
    question_facts: dict[str, list[str]] | None = None,  # ignored; kept for back-compat
    replans_left: int = 2,
) -> EpisodeState:
    del question_facts  # we no longer auto-extract facts; ORC embeds them in subtasks
    return EpisodeState(
        question=question,
        images=list(images or []),
        question_facts={},
        scenario_case_id=-1,
        scenario_session_id="",
        plan=None,
        steps=[],
        current_step_id=None,
        artifacts={},
        layers={},
        fields={},
        answer_registry={},
        history=[
            Event(
                role="user",
                phase="question",
                step_id=None,
                turn=None,
                payload={"question": question, "images": list(images or [])},
                ts=time.time(),
            )
        ],
        contract_window={},
        counters={"gpkg": 0, "img": 0, "tif": 0},
        replans_left=replans_left,
        final_answer=None,
        status="RUNNING",
        abort_reason="",
    )


def install_plan(state: EpisodeState, plan: Plan) -> EpisodeState:
    """Replace plan and step records (used at turn 1 and on RePlan)."""
    state["plan"] = plan
    state["steps"] = [
        StepRecord(
            id=s.id,
            agent=s.agent,
            subtask=s.subtask,
            needs=list(s.needs),
            status="pending",
            summary="",
        )
        for s in plan.steps
    ]
    state["current_step_id"] = None
    return state


def find_step(state: EpisodeState, step_id: str) -> StepRecord | None:
    for step in state["steps"]:
        if step["id"] == step_id:
            return step
    return None


def next_pending_step(state: EpisodeState) -> StepRecord | None:
    for step in state["steps"]:
        if step["status"] == "pending":
            return step
    return None


def all_steps_done(state: EpisodeState) -> bool:
    return bool(state["steps"]) and all(s["status"] == "done" for s in state["steps"])


def assign_artifact_handle(state: EpisodeState, kind: str, path: str, note: str = "", produced_by_step: str = "") -> str:
    """Register an artifact at a path. Returns the handle (existing or new)."""
    # If the same path is already registered, reuse the handle.
    for handle, rec in state["artifacts"].items():
        if rec.get("path") == path and rec.get("kind") == kind:
            return handle
    state["counters"][kind] = state["counters"].get(kind, 0) + 1
    handle_prefix = "image" if kind == "img" else kind
    handle = f"{handle_prefix}_{state['counters'][kind]}"
    state["artifacts"][handle] = {
        "kind": kind,
        "path": path,
        "produced_by_step": produced_by_step,
        "note": note,
    }
    return handle


def add_layer(state: EpisodeState, gpkg_handle: str, name: str, produced_by_step: str) -> None:
    bucket = state["layers"].setdefault(gpkg_handle, [])
    for existing in bucket:
        if existing["name"] == name:
            return
    bucket.append({"name": name, "produced_by_step": produced_by_step})


def set_field(state: EpisodeState, name: str, value: Any, produced_by_step: str) -> None:
    state["fields"][name] = FieldVal(value=value, produced_by_step=produced_by_step, ts=time.time())


def open_contract_window(state: EpisodeState, step_id: str) -> None:
    state["contract_window"][step_id] = len(state["history"])


def has_progress_since(state: EpisodeState, step_id: str) -> bool:
    """True iff any tool succeeded under this step since its window opened."""
    start = state["contract_window"].get(step_id)
    if start is None:
        return False
    for event in state["history"][start:]:
        if event["role"] == "tool" and event["payload"].get("ok") is True:
            if event["step_id"] == step_id:
                return True
    return False


def append_event(
    state: EpisodeState,
    role: str,
    phase: str,
    payload: dict[str, Any],
    step_id: str | None = None,
    turn: int | None = None,
) -> None:
    """Append a single event to history. Used inside graph nodes that don't go through reducers."""
    state["history"].append(
        Event(
            role=role,
            phase=phase,
            step_id=step_id,
            turn=turn,
            payload=payload,
            ts=time.time(),
        )
    )


def mark_step(state: EpisodeState, step_id: str, status: str, summary: str = "") -> None:
    step = find_step(state, step_id)
    if step is None:
        return
    step["status"] = status  # type: ignore[assignment]
    if summary:
        step["summary"] = summary


def render_artifact_card(handle: str, record: dict[str, Any]) -> str:
    note = record.get("note") or ""
    suffix = f" — {note}" if note else ""
    by = record.get("produced_by_step") or "input"
    return f"  {handle} [{record.get('kind', 'unknown')}]{suffix}  ({by})"


def render_layer_cards(state: EpisodeState) -> list[str]:
    out: list[str] = []
    for gpkg, layers in sorted(state["layers"].items()):
        if not layers:
            continue
        items = ", ".join(f"{lay['name']} ({lay['produced_by_step']})" for lay in layers)
        out.append(f"  {gpkg}: {items}")
    return out


def render_field_cards(
    state: EpisodeState,
    only: list[str] | None = None,
    *,
    max_fields: int | None = None,
    max_chars: int = 700,
) -> list[str]:
    out: list[str] = []
    for name, val in state["fields"].items():
        # Legacy releases placed terminal-answer payloads in normal worker
        # fields. They are never model-visible in the reference-only contract.
        if name == "latest_answer_ref" or name.endswith(("_answer_ref", "_final_answer")):
            continue
        if name.endswith(("_payload", "_payload_keys", "_history")):
            continue
        if only is not None and name not in only:
            continue
        try:
            if isinstance(val["value"], (dict, list)):
                text = json.dumps(val["value"], ensure_ascii=False, sort_keys=True)
            else:
                text = repr(val["value"])
        except Exception:
            text = repr(val["value"])
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        out.append(f"  {name} = {text}  ({val['produced_by_step']})")
        if max_fields is not None and len(out) >= max_fields:
            break
    return out
