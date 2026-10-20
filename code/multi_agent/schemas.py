"""Pydantic schemas for the GeoDisaster multi-agent pipeline.

Every schema's docstring carries an example JSON object that matches the schema.
``tests/test_schemas.py`` re-parses each example into its schema. That test is
the structural guarantee that prompt examples and runtime validators cannot
drift out of sync.

Four contract shapes:

* ``Plan``                — ORC turn 1 output.
* ``ORCDecision``         — ORC turn N output (CallAgent | Terminate | RePlan).
* ``SpecialistTurn``      — every specialist turn output.
* ``ToolResult``          — every tool's return shape.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class StepSpec(BaseModel):
    """One step in a plan.

    Example::

        {
          "id": "s1",
          "agent": "GA",
          "subtask": "Get Banff boundary buffered 3000m and add fire and police layers.",
          "needs": []
        }
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=16)
    agent: Literal["GA", "PA", "VRA"]
    subtask: str = Field(min_length=1)
    needs: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """The plan ORC produces on turn 1.

    Example::

        {
          "goal": "Find the closest fire-police station pair within 3000m of Banff.",
          "steps": [
            {"id": "s1", "agent": "GA", "subtask": "Boundary + POI layers", "needs": []},
            {"id": "s2", "agent": "GA", "subtask": "Compute closest pair", "needs": ["gpkg_1"]},
            {"id": "s3", "agent": "PA", "subtask": "State the answer", "needs": ["field pair", "field distance_m"]}
          ]
        }
    """

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    steps: list[StepSpec] = Field(min_length=1, max_length=6)

    @field_validator("steps")
    @classmethod
    def _unique_ids(cls, value: list[StepSpec]) -> list[StepSpec]:
        ids = [step.id for step in value]
        if len(set(ids)) != len(ids):
            raise ValueError("step ids must be unique")
        return value


# ---------------------------------------------------------------------------
# ORC decision (after a specialist returns)
# ---------------------------------------------------------------------------


class CallAgent(BaseModel):
    """ORC delegates the next pending step.

    Example::

        {"tool": "CallAgent", "step_id": "s2"}
    """

    model_config = ConfigDict(extra="forbid")

    tool: Literal["CallAgent"] = "CallAgent"
    step_id: str = Field(min_length=1)


class Terminate(BaseModel):
    """ORC ends the episode with the final answer.

    Example::

        {"tool": "Terminate", "answer_ref": "ANSWER_JSON:answer_1"}
    """

    model_config = ConfigDict(extra="forbid")

    tool: Literal["Terminate"] = "Terminate"
    answer_ref: str = Field(pattern=r"^ANSWER_JSON:answer_1$")


class RePlan(BaseModel):
    """ORC re-plans after a CONTRACT_FAILURE. Only legal when ``replans_left > 0``.

    Example::

        {
          "tool": "Plan",
          "goal": "Re-attempt with police layer reused.",
          "steps": [
            {"id": "s1", "agent": "PA", "subtask": "Compute closest pair from existing layers.", "needs": ["gpkg_1"]}
          ]
        }
    """

    model_config = ConfigDict(extra="forbid")

    tool: Literal["Plan"] = "Plan"
    goal: str = Field(min_length=1)
    steps: list[StepSpec] = Field(min_length=1, max_length=6)


ORCDecision = Annotated[
    Union[CallAgent, Terminate, RePlan],
    Field(discriminator="tool"),
]


# ---------------------------------------------------------------------------
# Specialist turn
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A specialist's tool invocation. ``tool`` may be any name except ``TaskComplete``.

    Example::

        {"tool": "AddPoisLayer", "args": {"gpkg": "gpkg_1", "query": {"amenity": "police"}, "layer_name": "police_stations"}}
    """

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool")
    @classmethod
    def _not_task_complete(cls, value: str) -> str:
        if value == "TaskComplete":
            raise ValueError("ToolCall.tool must not be 'TaskComplete'; use TaskComplete model")
        return value


class TaskComplete(BaseModel):
    """A specialist's terminal action.

    Example::

        {"tool": "TaskComplete", "status": "TASK_COMPLETE", "summary": "Built fire and police layers in gpkg_1."}
    """

    model_config = ConfigDict(extra="forbid")

    tool: Literal["TaskComplete"] = "TaskComplete"
    status: Literal["TASK_COMPLETE", "CONTRACT_FAILURE"]
    summary: str = Field(min_length=1)


SpecialistAction = Union[TaskComplete, ToolCall]


class SpecialistTurn(BaseModel):
    """Every specialist response. The ``thought`` envelope is mandatory.

    Example::

        {
          "thought": "Boundary already exists; add police layer next.",
          "action": {"tool": "AddPoisLayer", "args": {"gpkg": "gpkg_1", "query": {"amenity": "police"}, "layer_name": "police_stations"}}
        }
    """

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(min_length=1)
    action: SpecialistAction

    @field_validator("action", mode="before")
    @classmethod
    def _coerce_action(cls, value: Any) -> Any:
        # Without a discriminator, pydantic's union validation otherwise tries
        # TaskComplete first and raises on every ToolCall. Route by ``tool``.
        if isinstance(value, dict):
            if value.get("tool") == "TaskComplete":
                return TaskComplete.model_validate(value)
            return ToolCall.model_validate(value)
        return value


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------


class ArtifactRecord(BaseModel):
    """An artifact produced by a tool. ``handle`` is autoassigned by the runtime."""

    model_config = ConfigDict(extra="forbid")

    handle: str = Field(min_length=1)
    kind: Literal["gpkg", "img", "tif"]
    path: str = Field(min_length=1)
    note: str = ""


class LayerRecord(BaseModel):
    """A vector layer produced inside a gpkg."""

    model_config = ConfigDict(extra="forbid")

    gpkg: str = Field(min_length=1)
    name: str = Field(min_length=1)


class Produced(BaseModel):
    """What a tool produced this call."""

    model_config = ConfigDict(extra="forbid")

    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    layers: list[LayerRecord] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """A tool's return shape.

    Example::

        {
          "ok": true,
          "text": "Saved 3 POIs to layer 'fire_stations' in fix_aoi.gpkg",
          "produced": {
            "artifacts": [],
            "layers": [{"gpkg": "gpkg_1", "name": "fire_stations"}],
            "fields": {"fire_stations_count": 3}
          }
        }
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    text: str
    produced: Produced = Field(default_factory=Produced)


# ---------------------------------------------------------------------------
# Helpers — extract the example JSON from a model's docstring (for tests)
# ---------------------------------------------------------------------------


def example_json(model: type[BaseModel]) -> Any:
    """Return the first JSON literal in ``model.__doc__`` as a Python object.

    The docstring convention is ``Example::`` then a blank line then an indented
    JSON literal. We grab everything after ``Example::`` and strip it down to
    the JSON object, then ``json.loads`` it. This is what powers
    ``tests/test_schemas.py``.
    """
    doc = model.__doc__ or ""
    marker = "Example::"
    if marker not in doc:
        raise ValueError(f"{model.__name__} has no Example:: block in its docstring")
    after = doc.split(marker, 1)[1]
    # Find the first `{` and take from there until braces balance.
    start = after.find("{")
    if start < 0:
        raise ValueError(f"{model.__name__} Example:: block has no JSON object")
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i, ch in enumerate(after[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError(f"{model.__name__} Example:: block JSON is unbalanced")
    return json.loads(after[start:end])
