"""LangGraph wiring for the GeoDisaster multi-agent pipeline.

Two LangGraph nodes — ``orc`` and ``specialist`` — plus conditional edges that
encode the FSM. The whole topology is in this file.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .agents import (
    episode_done,
    needs_replan,
    orc_decide_node,
    orc_plan_node,
    specialist_node,
)
from .state import EpisodeState


def _orc_node(state: EpisodeState) -> EpisodeState:
    """ORC node — plans on turn 1, decides afterward.

    State channels are full-replace (no reducers); we return the mutated dict.
    """
    if state["plan"] is None:
        orc_plan_node(state)
    else:
        orc_decide_node(state)
    return state


def _specialist_node(state: EpisodeState) -> EpisodeState:
    specialist_node(state)
    return state


def _route_after_orc(state: EpisodeState) -> Literal["specialist", "orc", "end"]:
    if episode_done(state):
        return "end"
    if state["current_step_id"] is not None:
        return "specialist"
    # Plan was just installed (replan or first plan), or ORC must decide again.
    return "orc"


def _route_after_specialist(state: EpisodeState) -> Literal["orc", "end"]:
    if episode_done(state):
        return "end"
    return "orc"


def build_graph():
    builder = StateGraph(EpisodeState)
    builder.add_node("orc", _orc_node)
    builder.add_node("specialist", _specialist_node)

    builder.add_edge(START, "orc")
    builder.add_conditional_edges(
        "orc",
        _route_after_orc,
        {
            "specialist": "specialist",
            "orc": "orc",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "specialist",
        _route_after_specialist,
        {
            "orc": "orc",
            "end": END,
        },
    )
    return builder.compile()
