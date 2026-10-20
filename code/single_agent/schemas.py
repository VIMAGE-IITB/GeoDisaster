from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SinglePlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=16)
    objective: str = Field(min_length=1)
    needs: list[str] = Field(default_factory=list)


class SinglePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["Plan"] = "Plan"
    goal: str = Field(min_length=1)
    steps: list[SinglePlanStep] = Field(min_length=1, max_length=10)


class SingleToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool")
    @classmethod
    def _reserved(cls, value: str) -> str:
        if value in {"Plan", "FinalAnswer", "TaskComplete"}:
            raise ValueError("reserved tool name; use the dedicated action shape")
        return value


class SingleFinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["FinalAnswer"] = "FinalAnswer"
    answer_ref: str = Field(pattern=r"^ANSWER_JSON:answer_1$")


SingleAction = Union[SinglePlan, SingleFinalAnswer, SingleToolCall]


class SingleAgentTurn(BaseModel):
    """Single-agent response envelope.

    Example:
        {
          "thought": "Open the declared optical stack first.",
          "action": {"tool": "ReadData", "args": {"question": "question_1", "input": "optical_stack_1"}}
        }
    """

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(min_length=1)
    action: SingleAction

    @field_validator("action", mode="before")
    @classmethod
    def _route_action(cls, value: Any) -> Any:
        if isinstance(value, dict):
            tool = value.get("tool")
            if tool == "Plan":
                return SinglePlan.model_validate(value)
            if tool == "FinalAnswer":
                return SingleFinalAnswer.model_validate(value)
            return SingleToolCall.model_validate(value)
        return value

