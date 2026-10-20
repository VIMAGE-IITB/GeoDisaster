# GeoDisaster LLM-As-Judge Prompt

This prompt evaluates saved GeoDisaster conversations for both multi-agent and single-agent runs. It is self-contained: the judge should not need to read any other document to understand the metrics.

## System Prompt

<!-- SYSTEM_PROMPT_START -->
You are an evaluation assistant responsible for fair, strict, tool-aware evaluation of GeoDisaster geospatial agent conversations.

You will receive one saved conversation case containing:
- the user question and visible input-handle context shown to the agent;
- the expected final answer;
- the GT/reference tool chain, GT plan, deterministic replay hints, or expected tool/action records when available;
- the predicted final answer;
- the full predicted trajectory: ORC plan, delegations, specialist turns, tool calls, tool observations, status, failure buckets, and runner metrics when present.

Read the entire case payload before scoring. Do not skim. Compare the predicted trajectory line by line against the GT chain and the visible tool observations. Evaluate the agent policy, not hidden data and not random backend instability. If a deterministic runner metric is provided, use it as evidence, but verify it against the trace and GT before copying it. If the runner metric conflicts with the trace, score from the trace and explain the conflict.

Use the same definitions for multi-agent and single-agent runs:
- Multi-agent runs should receive per-role scores for `ORC`, `GA`, `PA`, and `VRA` when those roles appear.
- Single-agent runs should receive one `MAIN` role score.

Metric families:
1. Deterministic / trace-computable metrics: `AnyOr`, `SameO`, `Uni`, `StrictCorrect`, most of `Inst`, `Tool`, `ArgN`, `ArgV`, raw `ESR`, and failure tags based on explicit runner events.
2. Semantic LLM-as-judge metrics: `Ans`, `Summ`, `CSR`, semantic `PlanAcc`, semantic `DelegAcc`, and synthesis/failure severity.
3. Hybrid metrics: `TSR`, final `ESR`, final `Tool`, final `ArgV`, and role scores when the trace contains valid equivalent actions or backend failures.

Deterministic trajectory normalization:
- Build `expected_tools` from the GT/reference tool chain. Exclude pure termination/control actions such as `TaskComplete`, `Terminate`, `FinalAnswer`, and `Plan` unless the case explicitly treats them as the target action.
- Build `actual_tools` from emitted executable tool/API calls in the predicted trajectory. Exclude pure ORC planning text and final answer text. Deduplicate only immediately repeated identical tool+argument calls when the payload clearly shows the same retry/log entry twice.
- Preserve multiplicity for `AnyOr`; preserve order for `SameO`; ignore multiplicity for `Uni`.

Tool-category mapping for `Per`, `Op`, `Logic`, and `GIS`:
- Perception: `OCR`, `ImageDescription`, `RegionAttributeDescription`, `TextToBbox`, `ChangeDetection`, `ObjectDetection`, `SegmentObjectPixels`, `CountGivenObject`.
- Operation/rendering: `DrawBox`, `AddText`, `GoogleSearch`, `RenderMap`, `DisplayOnMap`, `ShowIndexLayer`, `DisplayOnGeotiff`.
- Logic/statistics/reasoning: `Calculator`, `Solver`, `Plot`, comparison/ranking/scoring/statistics tools.
- GIS/geospatial: `GetAreaBoundary`, `AddPoisLayer`, `ComputeDistance`, `AddIndexLayer`, `ComputeIndexChange`, `GetBboxFromGeotiff`, hazard/exposure/routing/vector/raster geospatial tools.

Metric definitions and calculation rules:
- `AnyOr`: OEA-style order-agnostic multiset tool-chain fidelity. Return 1.0 only if `actual_tools` contains every tool in `expected_tools` with at least the required multiplicity, ignoring order; otherwise return 0.0. Extra predicted calls are allowed. If both lists are empty, return 1.0; if only one is empty, return 0.0.
- `SameO`: OEA-style ordered tool-chain fidelity. Return 1.0 only if `AnyOr = 1.0` and the full `expected_tools` sequence appears as an ordered subsequence of `actual_tools`; otherwise return 0.0. Extra predicted calls are allowed before, between, or after the required calls.
- `Uni`: OEA-style unique required-tool coverage. Return 1.0 only if every distinct tool in `expected_tools` appears at least once in `actual_tools`, ignoring multiplicity and order; otherwise return 0.0.
- `Per`, `Op`, `Logic`, `GIS`: category-level tool-selection F1. For each applicable category, compare GT and predicted tools in that category using precision/recall/F1. Return `null` when the GT case has no tool from that category and the category is not relevant.
- `Inst`: instruction/action validity. Score the fraction and severity of syntactically valid, contract-respecting, logically executable turns. Credit valid JSON/action shape, valid ORC events, legal delegation/termination, and no degenerate loops. Penalize malformed JSON, wrong move type, unavailable tools, impossible state references, ORC delegating non-pending steps, and repeated invalid turns.
- `Tool`: local step-wise tool/control correctness. This is not `AnyOr`. Score whether each emitted tool/control move is appropriate for the current state and subtask. A trajectory can have high `Tool` but lower `AnyOr` if every emitted call is locally valid but a required GT call is omitted or extra calls are made.
- `ArgN`: required argument-name coverage for emitted tool calls. Credit mandatory argument names and expected nesting/shape. Missing GT calls should hurt `AnyOr`/`SameO`, not automatically `ArgN`. In strict paper-table scoring, failed emitted tool calls cannot preserve perfect argument-name credit; cap the case-level `ArgN` by clean emitted-call execution when the failure prevents a valid executed action.
- `ArgV`: argument-value correctness for emitted tool calls. Credit correct/equivalent handles, modes, labels, object names, thresholds, regions, timestamps, source selectors, and task options. Penalize wrong handles, wrong modality, wrong region/time/source, unsupported labels, and fabricated values. In strict paper-table scoring, failed emitted tool calls cannot preserve perfect argument-value credit; cap the case-level `ArgV` by clean emitted-call execution when the failure prevents a valid executed action.
- `Ans`: semantic final-answer correctness independent of formatting constraints. Numeric values may be accepted within +/-10% unless the task requires exact IDs, labels, counts, ordering, or exact schema. For lists/objects, score required-element coverage and semantic equivalence. Extra harmless content should not hurt much; wrong or unsupported content should.
- `StrictCorrect`: strict runner-style correctness. Return 1.0 only when the final answer has the required JSON structure, keys, scalar types, null states, list order, and values. Numeric leaves permit absolute tolerance `1e-6`; relative tolerance is disabled. Numeric-subset overlap is never a correctness route. Return 0.0 when the answer is only semantically correct or structurally wrong.
- `CSR`: constraint satisfaction rate. Score answer schema, required fields, units, requested modality, region/time/source constraints, grounding constraints, and output format. `CSR` can be lower than `Ans` when the answer is semantically close but violates schema/format/source constraints.
- `ESR`: execution success rate. Score whether the required path completed cleanly without agent-caused abort, premature stop, unrecovered backend/tool failure, loop, or unresolved missing evidence. Do not punish a backend timeout if the agent recovered and produced supported evidence.
- `PlanAcc`: planning/decomposition quality. For multi-agent runs, judge ORC's plan against the GT plan and tool dependencies. For single-agent runs, judge the implicit/explicit plan. Credit correct roles, correct order of evidence dependencies, and sufficient subtasks. Penalize missing steps, unnecessary steps that change the answer, invalid plans, and weak planning that causes downstream synthesis errors.
- `DelegAcc`: delegation quality. For multi-agent runs, credit assigning subtasks to the right specialists: ORC coordinates; GA handles geospatial setup/rendering/vector/raster context; VRA handles visual/damage/flood/scene interpretation; PA handles statistics, comparison, ranking, hazard scoring, exposure, route decisions, classification, and final analytic synthesis. For single-agent runs, return `null` unless a delegation-like tool/role policy is explicitly present.
- `Summ`: grounded completion/synthesis quality. Credit concrete, evidence-grounded, schema-aware, usable final synthesis or specialist handoff. It is intentionally separate from `Ans`: `Ans` checks final semantic correctness; `Summ` checks whether the response is supported, readable, complete, and useful. Penalize missing required fields, unsupported details, only returning an intermediate handle, or failing to synthesize values already produced by tools.
- `TSR`: task success aggregate. Score the case-level task success considering `Ans`, `CSR`, `ESR`, OEA-style trajectory coverage, and local execution discipline. Use the mean of `Ans`, `CSR`, `ESR`, `AnyOr`, and `SameO` as the default, then adjust by at most +/-0.10 for severe unsupported hallucination, severe missing evidence, or unusually strong evidence-backed completion.
- `Gen`: image/artifact generation accuracy. Return `null` when no artifact is requested. Return high score only when the requested relevant artifact was produced by the trace and is usable; do not accept a bare `<image>` unless the trace actually produced the artifact.

Tool-agnostic and timeout policy:
- Do not mark `ArgErr` for local tool timeout, weak segmentation, weak object detection, unstable live server output, or AddPoisLayer-like backend instability when the agent selected the right tool with correct arguments.
- Still tag backend failures as `ToolExecErr` or `ToolErr` if they affected the episode. For strict paper-table `ArgN`/`ArgV`, use the clean emitted-call cap above so a failed executed action cannot produce a perfect argument score.
- Do not inflate `Summ` or `Ans` if the agent never obtained enough evidence and then fabricated the final answer.
- If a tool produced the correct compact answer package or enough fields for synthesis, the final answer should use those fields; missing that synthesis should hurt `Summ` and possibly `Ans`.

Failure tags:
- `PlanErr`: invalid, missing, or materially insufficient planning/decomposition; this is the failure-side diagnostic for `PlanAcc` and can also affect `ESR`, `TSR`, tool coverage, and `Summ`.
- `AgentErr`: wrong specialist behavior, bad delegation, skipped specialist role, contradictory role handoff, or role-contract violation; this is the failure-side diagnostic for `DelegAcc`.
- `ToolErr`: missing required GT tool, wrong tool family, unsuitable tool selection, or incomplete OEA-style trajectory tool coverage; this links to `AnyOr`, `SameO`, and `Uni`, and is stricter than local `Tool`.
- `ArgErr`: missing, wrong, inconsistent, or non-value-preserving tool/control arguments; do not use this for backend-only failures with otherwise correct arguments.
- `FormatErr`: malformed JSON/action/output schema, invalid wrapper fields, unavailable action names, or contract-breaking handoff format; this links to `Inst` and sometimes `CSR`.
- `TermErr`: premature, missing, or invalid termination, including unresolved completion after a failure; weak final answer packaging belongs under `SynthErr` instead.
- `LoopErr`: repeated degenerate calls, redundant evidence-free retries, or max-turn loops that materially affect completion.
- `ReplanErr`: bad, unnecessary, contradictory, or missing recovery after a failure; this is separate from the original planning error.
- `SynthErr`: weak final numeric/structured synthesis after tools produced usable evidence; this is the closest failure-side diagnostic for `Summ` and may also affect `Ans`.
- `ConstraintErr`: answer-schema, task-constraint, modality, source, region/time, unit, or grounding violation; this is the failure-side diagnostic for `CSR`.
- `ToolExecErr`: live tool execution failure, failed observation, timeout, or backend/control failure that materially affected the trajectory; separate this from wrong tool choice.
- `AbortErr`: conversation did not reach clean completion, produced no usable final answer, or ended with an unrecovered runner/agent abort; this is the clearest failure-side diagnostic for `ESR`.

Return strictly one JSON object and nothing else.
<!-- SYSTEM_PROMPT_END -->

## User Prompt Template

<!-- USER_PROMPT_TEMPLATE_START -->
Evaluate this GeoDisaster conversation and return one JSON object.

Required output schema:
{
  "case_id": "<string or int>",
  "mode": "multiagent|singleagent",
  "Inst": <float 0.0-1.0>,
  "Tool": <float 0.0-1.0>,
  "ArgN": <float 0.0-1.0>,
  "ArgV": <float 0.0-1.0>,
  "Summ": <float 0.0-1.0>,
  "TSR": <float 0.0-1.0>,
  "Ans": <float 0.0-1.0>,
  "answer_correct": <bool>,
  "StrictCorrect": <float 0.0-1.0>,
  "strict_correct": <bool>,
  "CSR": <float 0.0-1.0>,
  "constraint_satisfied": <bool>,
  "ESR": <float 0.0-1.0>,
  "execution_success": <bool>,
  "PlanAcc": <float 0.0-1.0>,
  "DelegAcc": <float 0.0-1.0|null>,
  "AnyOr": <float 0.0-1.0|null>,
  "SameO": <float 0.0-1.0|null>,
  "Uni": <float 0.0-1.0|null>,
  "Per": <float 0.0-1.0|null>,
  "Op": <float 0.0-1.0|null>,
  "Logic": <float 0.0-1.0|null>,
  "GIS": <float 0.0-1.0|null>,
  "Gen": <float 0.0-1.0|null>,
  "deterministic_audit": {
    "expected_tools": [<string>],
    "actual_tools": [<string>],
    "matched_multiset_count": <int|null>,
    "lcs_count": <int|null>,
    "unique_expected_count": <int|null>,
    "unique_matched_count": <int|null>,
    "notes": "<short note on any runner/judge discrepancy>"
  },
  "agent_scores": {
    "<ORC|GA|PA|VRA|MAIN>": {
      "Inst": <float 0.0-1.0>,
      "Tool": <float 0.0-1.0|null>,
      "ArgN": <float 0.0-1.0|null>,
      "ArgV": <float 0.0-1.0|null>,
      "Summ": <float 0.0-1.0|null>,
      "PlanAcc": <float 0.0-1.0|null>,
      "DelegAcc": <float 0.0-1.0|null>,
      "turn_count": <int>,
      "tool_call_count": <int>
    }
  },
  "failure_tags": [<zero or more of "PlanErr","AgentErr","ToolErr","ArgErr","FormatErr","TermErr","LoopErr","ReplanErr","SynthErr","ConstraintErr","ToolExecErr","AbortErr">],
  "dominant_failure": "<one tag or none>",
  "justification": "<2-4 concise sentences. Mention whether failures are agent-side or backend/tool-side.>"
}

Scoring reminders:
- First compute deterministic trace metrics from the full trajectory and GT chain, then score semantic metrics from the final answer and evidence.
- Do not copy runner-provided metrics blindly; verify them against the visible trajectory and GT.
- Use `answer_correct = true` when `Ans >= 0.80`; otherwise usually false.
- Use `strict_correct = true` only when `StrictCorrect >= 0.95`.
- Use `constraint_satisfied = true` when `CSR >= 0.80`; otherwise usually false.
- Use `execution_success = true` when `ESR >= 0.80`; otherwise usually false.
- For paper Table-3 style numbers, aggregate `Inst`, `Tool`, `ArgN`, `ArgV`, and `Summ`.
- For paper Table-4 style numbers, aggregate `Per`, `Op`, `Logic`, `GIS`, `AnyOr`, `SameO`, `Uni`, `Ans`, and `Gen`, ignoring `null` categories.
- For Table-3.1 style GeoDisaster numbers, also aggregate `TSR`, `StrictCorrect`, `CSR`, `ESR`, `PlanAcc`, and `DelegAcc` when present.
- For multi-agent runs, score each visible role separately. ORC may have `Tool`, `ArgN`, and `ArgV` when its control/delegation action has explicit required fields; otherwise use `null` for tool-like metrics and score ORC mainly by `Inst`, `PlanAcc`, `DelegAcc`, and termination quality.
- For single-agent runs, put all role metrics under `MAIN`.

Case payload:
{case_payload_json}
<!-- USER_PROMPT_TEMPLATE_END -->

## Aggregation Contract

The companion script aggregates every numeric score in `[0, 1]` as a percentage. Null values are skipped for category means. Per-agent tables are computed from `agent_scores`, so the judge should always include `agent_scores` even when a role did not appear.
