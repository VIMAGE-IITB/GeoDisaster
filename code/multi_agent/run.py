"""Conversation-log projection helpers for GeoDisaster agent runs.

Dataset execution is provided by :mod:`evaluation.eval_multi_agent`. This
module intentionally contains no canned questions, fixture tools, or expected
answers; training and evaluation import only the conversation projection.
"""

from __future__ import annotations

import json
from typing import Any

from .state import EpisodeState


def _build_conversation_v1_shape(state: EpisodeState) -> list[dict[str, Any]]:
    """Project runtime history into the canonical conversation-log format."""

    conversation: list[dict[str, Any]] = [
        {"role": "user", "actor": "USER", "message": state["question"]}
    ]
    history = state["history"]

    def user_message(payload: dict[str, Any]) -> str:
        message = payload.get("user_message")
        return message if isinstance(message, str) else ""

    def tool_observation(payload: dict[str, Any]) -> str:
        return (
            "OBSERVATION\n"
            f"Tool: {payload.get('tool', '?')}\n"
            f"Status: {'OK' if payload.get('ok') else 'ERROR'}\n"
            f"Args: {payload.get('args', {})}\n"
            f"Result: {payload.get('text', '')}"
        )

    for event in history:
        role = event.get("role")
        phase = event.get("phase")
        payload = event.get("payload") or {}

        if role == "user" and phase == "question":
            continue

        if role in ("ORC", "GA", "PA", "VRA") and phase in (
            "plan",
            "decide",
            "delegate",
            "step",
            "replan",
            "terminate",
        ):
            message = user_message(payload)
            if message:
                conversation.append({"role": "user", "actor": role, "message": message})

            if phase in ("plan", "decide", "replan"):
                if "decision" in payload:
                    response: Any = payload["decision"]
                elif "plan" in payload:
                    response = {
                        "thought": "",
                        "action": {"tool": "Plan", "args": payload["plan"]},
                    }
                elif "answer_ref" in payload:
                    response = {"tool": "Terminate", "answer_ref": payload["answer_ref"]}
                elif "error" in payload:
                    response = f"<<error: {payload['error']}>>"
                else:
                    response = None
                if response is not None:
                    rendered = (
                        response
                        if isinstance(response, str)
                        else json.dumps(response, ensure_ascii=False)
                    )
                    conversation.append(
                        {"role": "agent", "actor": role, "message": rendered}
                    )
            elif phase == "delegate":
                response = payload.get("decision") or {
                    "tool": "CallAgent",
                    "step_id": event.get("step_id"),
                }
                conversation.append(
                    {
                        "role": "agent",
                        "actor": role,
                        "message": json.dumps(response, ensure_ascii=False),
                    }
                )
            elif phase == "step":
                if payload.get("turn") is not None:
                    rendered = json.dumps(payload["turn"], ensure_ascii=False)
                elif "error" in payload:
                    rendered = f"<<error: {payload['error']}>>"
                else:
                    rendered = ""
                if rendered:
                    conversation.append(
                        {"role": "agent", "actor": role, "message": rendered}
                    )
            elif phase == "terminate":
                response = {
                    "tool": "Terminate",
                    "answer_ref": payload.get("answer_ref", ""),
                }
                conversation.append(
                    {
                        "role": "agent",
                        "actor": role,
                        "message": json.dumps(response, ensure_ascii=False),
                    }
                )

        elif role == "tool" and phase == "tool_call":
            actor = next(
                (
                    item["role"]
                    for item in history
                    if item.get("step_id") == event.get("step_id")
                    and item.get("role") in ("GA", "PA", "VRA")
                ),
                "AGENT",
            )
            conversation.append(
                {"role": "user", "actor": actor, "message": tool_observation(payload)}
            )

    return conversation


__all__ = ["_build_conversation_v1_shape"]
