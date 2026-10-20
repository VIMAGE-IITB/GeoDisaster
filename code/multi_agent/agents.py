"""Agent node functions.

Each function takes the ``EpisodeState`` and produces a state update. Together
with ``graph.py`` they form the FSM. None of these functions accumulates LLM
chat history themselves — every call is stateless from the model's POV.
"""

from __future__ import annotations

import json
from typing import Any

from .llm import LLMValidationError, llm_call
from .answer_contract import project_from_state, validate_explicit_answer
from .answer_registry import register_answer, resolve_answer_ref
from .tool_gateway import call_tool, current_sanitized_case, is_tool_allowed
from .prompts import (
    SYSTEM_PROMPTS,
    build_orc_decide_message,
    build_orc_plan_message,
    build_specialist_message,
)

from .schemas import (
    CallAgent,
    ORCDecision,
    Plan,
    RePlan,
    SpecialistTurn,
    TaskComplete,
    Terminate,
    ToolCall,
)
from .state import (
    EpisodeState,
    add_layer,
    all_steps_done,
    append_event,
    assign_artifact_handle,
    find_step,
    has_progress_since,
    install_plan,
    mark_step,
    next_pending_step,
    open_contract_window,
    set_field,
)


_MAX_INNER_TURNS = 8


# ---------------------------------------------------------------------------
# ORC nodes
# ---------------------------------------------------------------------------


def orc_plan_node(state: EpisodeState) -> EpisodeState:
    """Turn 1 — ORC produces the Plan."""
    user_msg = build_orc_plan_message(state)
    try:
        plan = llm_call(
            system=SYSTEM_PROMPTS["ORC_PLAN"],
            user=user_msg,
            schema=Plan,
            max_retries=1,
        )
    except LLMValidationError as exc:
        append_event(
            state,
            role="ORC",
            phase="plan",
            payload={"error": str(exc)},
        )
        state["status"] = "ABORTED"
        state["abort_reason"] = "orc_plan_invalid"
        state["final_answer"] = "Could not produce a valid plan."
        return state

    install_plan(state, plan)
    append_event(
        state,
        role="ORC",
        phase="plan",
        payload={"user_message": user_msg, "plan": plan.model_dump()},
    )
    return state


def orc_decide_node(state: EpisodeState) -> EpisodeState:
    """Turn N — ORC chooses CallAgent / Terminate / RePlan."""
    last_report = _last_specialist_report(state)
    user_msg = build_orc_decide_message(state, last_report)

    try:
        decision: ORCDecision = llm_call(
            system=SYSTEM_PROMPTS["ORC_DECIDE"],
            user=user_msg,
            schema=_OrcDecisionTypeAdapter,
            max_retries=1,
        )
    except LLMValidationError as exc:
        append_event(state, role="ORC", phase="decide", payload={"error": str(exc)})
        state["status"] = "ABORTED"
        state["abort_reason"] = "orc_decide_invalid"
        state["final_answer"] = _best_effort_answer(state)
        return state

    if isinstance(decision, CallAgent):
        return _orc_dispatch_call_agent(state, decision, user_msg)
    if isinstance(decision, Terminate):
        return _orc_dispatch_terminate(state, decision, user_msg)
    if isinstance(decision, RePlan):
        return _orc_dispatch_replan(state, decision, user_msg)
    raise RuntimeError(f"unknown decision type {type(decision)!r}")


def _orc_dispatch_call_agent(state: EpisodeState, decision: CallAgent, user_msg: str) -> EpisodeState:
    pending = next_pending_step(state)
    if pending is None or pending["id"] != decision.step_id:
        # ORC tried to call a step that's not the next pending one. We hard-fail
        # this episode rather than silently re-route — that's the no-fallback
        # principle. If it happens repeatedly we'll observe it and decide.
        append_event(
            state,
            role="ORC",
            phase="decide",
            payload={
                "error": f"step_id {decision.step_id!r} is not the next pending step "
                         f"({pending['id'] if pending else 'none'})",
                "user_message": user_msg,
                "decision": decision.model_dump(),
            },
        )
        state["status"] = "ABORTED"
        state["abort_reason"] = "orc_called_wrong_step"
        state["final_answer"] = _best_effort_answer(state)
        return state

    state["current_step_id"] = pending["id"]
    mark_step(state, pending["id"], "running")
    open_contract_window(state, pending["id"])
    append_event(
        state,
        role="ORC",
        phase="delegate",
        step_id=pending["id"],
        payload={
            "user_message": user_msg,
            "decision": decision.model_dump(),
            "contract": {
                "agent": pending["agent"],
                "subtask": pending["subtask"],
                "needs": list(pending.get("needs") or []),
            },
        },
    )
    return state


def _orc_dispatch_terminate(state: EpisodeState, decision: Terminate, user_msg: str) -> EpisodeState:
    answer_ref = decision.answer_ref
    case = current_sanitized_case()
    if case and not all_steps_done(state):
        append_event(
            state,
            role="ORC",
            phase="decide",
            payload={
                "error": "termination rejected: plan still has unfinished steps",
                "user_message": user_msg,
                "answer_ref": answer_ref,
            },
        )
        return state
    resolution = resolve_answer_ref(state, answer_ref)
    if not resolution.ok:
        state["status"] = "ABORTED"
        state["abort_reason"] = "answer_reference_not_ready"
        state["final_answer"] = None
        append_event(
            state,
            role="ORC",
            phase="abort",
            payload={
                "error": resolution.error,
                "user_message": user_msg,
                "answer_ref": answer_ref,
            },
        )
        return state
    if case:
        validation = validate_explicit_answer(case, resolution.answer)
        if not validation.ok:
            state["status"] = "ABORTED"
            state["abort_reason"] = "answer_contract_failure"
            state["final_answer"] = None
            append_event(
                state,
                role="ORC",
                phase="abort",
                payload={
                    "error": "termination answer failed the declared JSON Schema",
                    "validation_errors": list(validation.errors),
                    "user_message": user_msg,
                    "answer_ref": answer_ref,
                },
            )
            return state
        state["final_answer"] = json.dumps(
            validation.answer, ensure_ascii=False, sort_keys=True
        )
    else:
        state["final_answer"] = json.dumps(
            resolution.answer, ensure_ascii=False, sort_keys=True
        )
    state["status"] = "DONE"
    append_event(
        state,
        role="ORC",
        phase="terminate",
        payload={
            "user_message": user_msg,
            "answer_ref": answer_ref,
            "resolved_answer": state["final_answer"],
        },
    )
    return state


def _orc_dispatch_replan(state: EpisodeState, decision: RePlan, user_msg: str) -> EpisodeState:
    if state["replans_left"] <= 0:
        append_event(state, role="ORC", phase="replan",
                     payload={"error": "no replans left", "decision": decision.model_dump()})
        state["status"] = "ABORTED"
        state["abort_reason"] = "replans_exhausted"
        state["final_answer"] = _best_effort_answer(state)
        return state
    state["replans_left"] -= 1
    new_plan = Plan(goal=decision.goal, steps=decision.steps)
    install_plan(state, new_plan)
    append_event(
        state,
        role="ORC",
        phase="replan",
        payload={"user_message": user_msg, "plan": new_plan.model_dump(),
                 "replans_left": state["replans_left"]},
    )
    return state


# ---------------------------------------------------------------------------
# Specialist node (inner loop in plain Python)
# ---------------------------------------------------------------------------


def specialist_node(state: EpisodeState) -> EpisodeState:
    """Run the inner tool loop for the current step."""
    step_id = state["current_step_id"]
    if step_id is None:
        # Defensive — graph routing should never call us without a current step.
        return state
    step = find_step(state, step_id)
    if step is None:
        return state

    contract_window_index = state["contract_window"][step_id]

    last_summary = ""
    last_status = "CONTRACT_FAILURE"
    pydantic_failures_in_a_row = 0
    tool_outcomes: dict[str, tuple[bool, str]] = {}

    for turn_no in range(1, _MAX_INNER_TURNS + 1):
        user_msg = build_specialist_message(state, step, turn_no, contract_window_index)
        try:
            turn = llm_call(
                system=SYSTEM_PROMPTS[step["agent"]],
                user=user_msg,
                schema=SpecialistTurn,
                max_retries=1,
            )
            pydantic_failures_in_a_row = 0
        except LLMValidationError as exc:
            pydantic_failures_in_a_row += 1
            append_event(
                state,
                role=step["agent"],
                phase="step",
                step_id=step_id,
                turn=turn_no,
                payload={"error": str(exc), "user_message": user_msg},
            )
            if pydantic_failures_in_a_row >= 2:
                last_summary = f"Two consecutive schema-invalid responses: {exc}"
                last_status = "CONTRACT_FAILURE"
                break
            continue

        # Record the raw turn response.
        append_event(
            state,
            role=step["agent"],
            phase="step",
            step_id=step_id,
            turn=turn_no,
            payload={"user_message": user_msg, "turn": turn.model_dump()},
        )

        action = turn.action
        if isinstance(action, TaskComplete):
            last_summary = action.summary
            last_status = action.status
            break

        # Tool call path.
        assert isinstance(action, ToolCall)
        tool_key = _tool_call_key(action.tool, action.args)
        previous_outcome = tool_outcomes.get(tool_key)
        if previous_outcome is not None:
            previous_ok, _ = previous_outcome
            outcome_word = "successful" if previous_ok else "failing"
            last_summary = (
                f"Repeated {outcome_word} tool call {action.tool} with the same args; "
                "stopping to avoid a loop."
            )
            last_status = "CONTRACT_FAILURE"
            break

        if not is_tool_allowed(step["agent"], action.tool):
            text = f"{step['agent']} is not allowed to call {action.tool}"
            append_event(
                state,
                role="tool",
                phase="tool_call",
                step_id=step_id,
                turn=turn_no,
                payload={
                    "tool": action.tool,
                    "args": action.args,
                    "ok": False,
                    "text": text,
                    "produced": {"fields": {"failure_category": "forbidden_tool"}},
                },
            )
            tool_outcomes[tool_key] = (False, text)
            # Don't auto-fail; let the model see the error in LAST OBSERVATION.
            continue

        result = call_tool(action.tool, action.args)
        tool_outcomes[tool_key] = (bool(result.ok), str(result.text))
        # Resolve placeholder handles produced by tools (kind == 'gpkg' / 'img' / 'tif').
        produced_dump = _absorb_tool_result(state, step_id, action.tool, result)

        append_event(
            state,
            role="tool",
            phase="tool_call",
            step_id=step_id,
            turn=turn_no,
            payload={
                "tool": action.tool,
                "args": action.args,
                "ok": result.ok,
                "text": result.text,
                "produced": produced_dump,
            },
        )

    else:  # for-else: inner loop hit max turns without break
        last_summary = f"{step['agent']} hit max inner turns ({_MAX_INNER_TURNS})."
        last_status = "CONTRACT_FAILURE"

    # Auto-downgrade: claimed TASK_COMPLETE but no successful tool call ran.
    if last_status == "TASK_COMPLETE" and not has_progress_since(state, step_id):
        # PA synthesis steps may legitimately not call any tool — they just
        # restate evidence already present. Allow that ONLY if relevant fields
        # were already present at contract start.
        if not _pa_synthesis_grace(state, step):
            last_summary = f"Auto-downgrade: {step['agent']} returned TASK_COMPLETE but no progress logged."
            last_status = "CONTRACT_FAILURE"

    mark_step(state, step_id, "done" if last_status == "TASK_COMPLETE" else "failed", last_summary)
    state["current_step_id"] = None
    append_event(
        state,
        role=step["agent"],
        phase="complete",
        step_id=step_id,
        payload={"status": last_status, "summary": last_summary},
    )
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _absorb_tool_result(state: EpisodeState, step_id: str, tool_name: str, result: Any) -> dict[str, Any]:
    """Register artifacts/layers/fields. Resolve auto-handles to concrete handles."""
    produced = result.produced
    handle_map: dict[str, str] = {}
    out_artifacts: list[dict[str, Any]] = []
    for art in produced.artifacts:
        # Always create a fresh handle for tools that return a new artifact.
        handle = assign_artifact_handle(
            state,
            kind=art.kind,
            path=art.path,
            note=art.note,
            produced_by_step=step_id,
        )
        handle_map[art.handle] = handle
        out_artifacts.append({"handle": handle, "kind": art.kind, "path": art.path, "note": art.note})

    out_layers: list[dict[str, Any]] = []
    for lay in produced.layers:
        gpkg = lay.gpkg
        if gpkg in handle_map:
            gpkg = handle_map[gpkg]
        # If the gpkg ref doesn't resolve to a real handle, skip silently — the
        # tool call's args were wrong; surface that in LAST OBSERVATION via
        # the result.text field.
        if gpkg in state["artifacts"]:
            add_layer(state, gpkg, lay.name, produced_by_step=step_id)
        out_layers.append({"gpkg": gpkg, "name": lay.name})

    out_fields: dict[str, Any] = {}
    for fname, fval in produced.fields.items():
        set_field(state, fname, fval, produced_by_step=step_id)
        out_fields[fname] = fval

    case = current_sanitized_case()
    if result.ok and case:
        projection = project_from_state(case, state["fields"])
        if projection.ok:
            register_answer(
                state,
                case,
                projection.answer,
                produced_by_step=step_id,
                produced_by_tool=tool_name,
            )

    return {"artifacts": out_artifacts, "layers": out_layers, "fields": out_fields}


def _tool_call_key(tool_name: str, args: dict[str, Any]) -> str:
    try:
        args_text = json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        args_text = str(args or {})
    return f"{tool_name}|{args_text}"


def _last_specialist_report(state: EpisodeState) -> dict[str, Any] | None:
    for event in reversed(state["history"]):
        if event["role"] in ("GA", "PA", "VRA") and event["phase"] == "complete":
            return {
                "agent": event["role"],
                "step_id": event["step_id"],
                "status": event["payload"].get("status"),
                "summary": event["payload"].get("summary"),
            }
    return None


def _best_effort_answer(state: EpisodeState) -> str:
    # Prefer the most recent specialist summary if present.
    last = _last_specialist_report(state)
    if last and last.get("summary"):
        return last["summary"]
    if state["fields"]:
        # Render fields as `name=value` joined.
        return "; ".join(f"{n}={v['value']}" for n, v in state["fields"].items())
    return f"Unable to produce an answer for: {state['question']}"


def _pa_synthesis_grace(state: EpisodeState, step: dict[str, Any]) -> bool:
    """A PA synthesis step is allowed to TaskComplete without a tool call only
    if it had relevant evidence already in scope (via `needs`)."""
    if step.get("agent") != "PA":
        return False
    needs = step.get("needs") or []
    if not needs:
        return False
    for token in needs:
        if token.startswith("field "):
            fname = token[6:].strip()
            if fname in state["fields"]:
                return True
        elif token in state["artifacts"]:
            return True
    return False


# ---------------------------------------------------------------------------
# OrcDecision union for `instructor`-style validation via TypeAdapter
# ---------------------------------------------------------------------------


from pydantic import TypeAdapter

_OrcDecisionAdapter = TypeAdapter(ORCDecision)


class _OrcDecisionTypeAdapter:
    """Lightweight wrapper exposing the ``model_validate`` interface to llm.py."""

    @staticmethod
    def model_validate(obj: dict[str, Any]) -> Any:
        return _OrcDecisionAdapter.validate_python(obj)


# ---------------------------------------------------------------------------
# Convenience predicates used by graph routing
# ---------------------------------------------------------------------------


def episode_done(state: EpisodeState) -> bool:
    return state["status"] in ("DONE", "ABORTED")


def needs_replan(state: EpisodeState) -> bool:
    last = _last_specialist_report(state)
    if last is None:
        return False
    return last.get("status") == "CONTRACT_FAILURE" and state["replans_left"] > 0
