# Agents

GeoDisaster uses a **contract-driven orchestrated multi-agent system**: one orchestrator plans and
delegates, three specialists execute with tools. Implementation:
[`code/multi_agent/`](../code/multi_agent/) (`agents.py`, `graph.py`, `state.py`, `schemas.py`); a
single-agent baseline lives in [`code/single_agent/`](../code/single_agent/).

## Roles

| Role | Responsibility | Tools |
|---|---|---|
| **ORC** (Orchestrator) | Reads the question, writes a plan, delegates each step to a specialist, tracks results, and terminates with the runtime-published answer reference. **Holds no tools.** | — (planning & delegation only) |
| **GA** (Geospatial Agent) | Opens and prepares evidence, measures geometry, extracts objects, maps flood extent, renders maps. | `ReadData`, `PrepareScene`, `MeasureArea`, `ExtractObjects`, `MapFlood`, `RenderMap` |
| **PA** (Planning / Analytical Agent) | Numeric and decision analytics: statistics, comparison, change detection/classification, hazard scoring, exposure, ranking, and routing. | `ComputeStats`, `CompareSources`, `DetectChange`, `ClassifyChange`, `ScoreHazard`, `EstimateExposure`, `RankRegions`, `PlanRoutes`, `CompareRoutes` |
| **VRA** (Visual Reasoning Agent) | Visual/semantic judgments over imagery: damage assessment, flood description, scene comparison. | `AssessDamage`, `DescribeFlood`, `CompareScenes` |

Per-agent ownership is declared in
[`code/tool_registry/tool_ownership.json`](../code/tool_registry/tool_ownership.json).

## Interaction flow

```
USER ──question──▶ ORC ──Plan──▶ (step 1 ▶ GA) ──result──▶ ORC ──(step 2 ▶ PA) … ──▶ ORC ──Terminate(answer_ref)──▶ runtime-resolved JSON
```

The orchestrator emits a `Plan`, then for each step issues `CallAgent(step_id)`; the selected
specialist runs a short tool loop and returns; the orchestrator may `Plan` again (re-plan) and
finally `Terminate` with a runtime-published answer reference. The whole exchange is recorded turn-by-turn in the dataset's
`multiagentic_gt_chain`.

## Message structure between ORC and agents (the contract)

The orchestrator↔specialist interface is a typed contract (Pydantic schemas in
[`code/multi_agent/schemas.py`](../code/multi_agent/schemas.py)):

**ORC plan** — what the orchestrator commits to:
```json
{"goal": "<overall objective>",
 "steps": [{"id": "s1", "agent": "GA", "subtask": "<what this specialist must do>", "needs": []},
           {"id": "s2", "agent": "PA", "subtask": "…", "needs": ["s1"]}]}
```
Each step names exactly one `agent` (`GA` | `PA` | `VRA`), a natural-language `subtask`, and
`needs` (the exact input, artifact, layer, or field tokens required by that
step, such as `question_1`, `work_1`, or `field distance_m`). A plan has 1–6
steps; step IDs identify dispatch targets and are not dependency tokens.

**ORC actions:** `CallAgent{step_id}` (dispatch a planned step), `Plan{goal, steps}` (re-plan), or
`Terminate{answer_ref}` (finish). The only valid value is the published
`ANSWER_JSON:answer_1`. Termination is accepted only after the plan is complete; the runtime then
resolves and revalidates the referenced JSON against that record's declared answer schema.

**Specialist turn** — what an agent returns each step:
```json
{"thought": "<brief reasoning>",
 "action": {"tool": "ComputeStats", "args": {...}}}
```
The `action` is either a `ToolCall{tool, args}` or a `TaskComplete{status, summary}`, where
`status` is `TASK_COMPLETE` (sub-goal met) or `CONTRACT_FAILURE` (the contract status for an
unsatisfied step, used instead of fabricating evidence).

**Tool results** flow back as `ToolResult{ok, text, produced}`, where `produced` carries the new
`artifacts` (handle + kind + path), `layers`, and structured `fields` that downstream steps
reference by handle. Handles are passed between steps — `question_1` → `scene_N` (after `ReadData`)
→ `work_N` (after `PrepareScene`) → analytic tools — so each step builds on prior evidence rather
than re-reading raw files.

## Tool and answer boundary

Agents never receive the expected answer or teacher chain as runtime context. They see the plain
question and declared input-handle cards. The answer schema remains runtime-owned because the
model returns only a reference; it is used for evidence projection, reference publication, and
strict scoring rather than being copied into every model prompt. Calls cross
[`code/multi_agent/tool_gateway.py`](../code/multi_agent/tool_gateway.py), which enforces ownership,
validates arguments, and invokes real workers in-process or over HTTP.

A worker payload may contain support fields in addition to the requested answer. After successful
execution, [`code/multi_agent/answer_contract.py`](../code/multi_agent/answer_contract.py) derives a
schema-valid terminal projection from that evidence and publishes it in a runtime-owned answer
registry as `ANSWER_JSON:answer_1`. This is a stable transport reference, not a worker-artifact
handle. The answer payload is hidden from model context, resolved at termination, validated again,
and passed to the strict evaluator. Unknown or unavailable references fail explicitly; the runtime
does not fall back to gold, prose, or a best-effort field dump.
