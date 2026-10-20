"""GeoDisaster dataset tool profile for the v3 multi-agent stack.

This module is opt-in. The existing base agent profile stays unchanged
unless callers set ``MAS_AGENT_PROFILE=geodisaster`` before importing any
``multi_agent`` modules.

Design contract:
- ``FINAL_DATASET_TOOL_SIGNATURES`` is the single source of truth for *what
  each tool does and what arguments it takes*.
- The ``*_APPENDIX`` strings deliberately do NOT re-describe individual tools.
  They carry only what a signature cannot: each role's scope, the tool-
  selection principle, hard boundaries, the failure/handoff protocol, and the
  output contract. Keeping tool behavior in exactly one place stops the
  signatures and the prompts from drifting apart.
"""

from __future__ import annotations

from tool_server.tool_workers.online_workers.final_dataset_tool_specs import (
    TOOL_SPECS,
    render_agent_signature,
)


FINAL_DATASET_TOOLS: list[str] = list(TOOL_SPECS)


# Role ownership: GA holds concrete data/geospatial ops, PA holds analysis and
# planning, VRA holds visual/semantic interpretation.
FINAL_DATASET_AGENT_TOOLS: dict[str, list[str]] = {
    "GA": [
        "ReadData",
        "PrepareScene",
        "MeasureArea",
        "ExtractObjects",
        "MapFlood",
        "RenderMap",
    ],
    "PA": [
        "ComputeStats",
        "CompareSources",
        "DetectChange",
        "ClassifyChange",
        "ScoreHazard",
        "EstimateExposure",
        "RankRegions",
        "PlanRoutes",
        "CompareRoutes",
    ],
    "VRA": [
        "AssessDamage",
        "DescribeFlood",
        "CompareScenes",
    ],
}


TOOL_TO_AGENT: dict[str, str] = {
    tool: agent
    for agent, tools in FINAL_DATASET_AGENT_TOOLS.items()
    for tool in tools
}


# Prompt signatures are rendered from the same contract used by the workers.
FINAL_DATASET_TOOL_SIGNATURES: dict[str, str] = {
    name: render_agent_signature(name) for name in FINAL_DATASET_TOOLS
}


FINAL_DATASET_ORC_APPENDIX = """
GeoDisaster dataset profile

INPUT HANDLES
- Each question is followed by AVAILABLE INPUT HANDLES. Use those exact handle
  names in `needs` and in subtasks. Never write raw local paths; tools resolve
  true local paths from the handle cards.
- For tool args, `question` means the canonical question handle, normally
  `question_1`; it does not mean copying the natural-language question.
- Downstream analysis tools require handles produced by earlier tools. Do not
  plan to use a guessed `scene_N`, `work_N`, `stats_N`, or `change_N` before a
  specialist has actually produced it.
- Raw input handles from AVAILABLE INPUT HANDLES, such as `forest_loss_history_1`,
  `optical_stack_1`, `damage_scene_1`, `sar_chip_1`, and `route_network_1`, are
  accepted only by ReadData. Other tools consume produced handles such as
  `scene_N`, `work_N`, `stats_N`, `change_N`, `route_N`, or `map_N`.
- The AOI/context is already inside the listed data handles and `question_1`.
  Do not invent unlisted map, boundary, vector, image, or raster handles.
- Use only the GeoDisaster tool names listed in the specialist ALLOWED TOOLS.

SPECIALIST CAPABILITIES (coordinate by capability, never by a fixed recipe)
- GA  - data/geospatial operations: open declared input handles, prepare
  spatial/raster work contexts, measure extents, extract deterministic spatial
  objects, build flood-mask artifacts, render maps/overlays.
- PA  - analytical reasoning: statistics, cross-source agreement, temporal
  change, categorical decisions, hazard/risk scoring, exposure estimation,
  ranking, route planning, route-tradeoff comparison.
- VRA - visual/semantic interpretation: damage assessment, flood description,
  scene-to-scene visual/risk comparison over already-prepared evidence.

DECOMPOSITION
- Split work by the evidence that must be produced next, reasoning over handle
  names, evidence types, and requested outputs.
- Group steps under one specialist only when they are the same capability; do
  not impose a tool order.
- If a question needs source data, plan an evidence-opening/preparation step
  before analysis: ReadData opens visible input handles into `scene_N`; when
  analysis needs a fused context, PrepareScene turns opened scenes into `work_N`.
- Therefore a plan that starts from raw dataset handles should normally give GA
  the first evidence-opening/preparation step before PA or VRA analysis.
- Write subtasks in general terms using the visible handles and the requested
  evidence/output type. Do not write case-specific instructions.
- If you replan after CONTRACT_FAILURE, emit the exact normal Plan schema:
  `{"goal": "...", "steps": [{"id": "s1", "agent": "GA|PA|VRA",
  "subtask": "...", "needs": [...]}]}`. Do not use tool-call fields such as
  `tool`, `action`, `inputs`, `outputs`, `output_handle`, or `output_schema`
  inside a Plan step.

FINAL ANSWER REFERENCE
- The runtime projects worker evidence into the declared answer schema. After
  that projection validates, AVAILABLE ANSWER REFERENCES publishes the stable
  token `ANSWER_JSON:answer_1`; the full payload remains runtime-owned.
- Terminate only as
  `{"tool":"Terminate","answer_ref":"ANSWER_JSON:answer_1"}`. Do not emit
  inline answer JSON, prose, partial fields, artifact handles, or a reference
  that was not published.
- Termination is accepted only after all plan steps complete. The runtime then
  resolves the reference, validates the resolved JSON again, and gives that
  JSON—not the reference token—to the strict evaluator.
"""


FINAL_DATASET_GA_APPENDIX = """
GeoDisaster dataset - GA notes

ROLE
- Geospatial/data operations specialist. Pick tools from the required evidence
  type, not from memorized question templates. Tool contracts are in the tool
  signatures; do not reinterpret them.

OPERATING RULES
- Copy handle names verbatim from AVAILABLE INPUT HANDLES into the relevant
  argument. Never use raw paths; tools resolve true paths from the handle cards.
- In all GeoDisaster tools, use `question: "question_1"` unless the current
  question exposes a different canonical question handle.
- Do not call a tool with a guessed handle such as `work_1` unless that handle
  is visible in PRIOR ARTIFACTS, Δ STATE, or LAST OBSERVATION.
- ReadData is the session-opening data tool. Use it to open input handles into
  concrete `scene_N` handles before tools that require prepared evidence.
- After a successful ReadData call, the produced `scene_N` handle appears as
  `latest_handle` in Δ STATE / LAST OBSERVATION. Use that scene handle in the
  next PrepareScene `inputs` object; do not reuse the raw input handle there.
- PrepareScene creates `work_N` handles from opened `scene_N` handles.
- PrepareScene `inputs` must map semantic slot names to opened scene handles,
  not raw input-handle names. Use the visible handle description/role to choose
  slots:
  forest evidence -> `optical`, `sar`, `forest`, `loss_history`;
  hazard stacks -> `hazard`, optional `sources`/`exposure`;
  damage scenes -> `image`, `labels`, optional `masks`;
  SAR flood -> `sar`, optional `water`;
  routing -> `routing`;
  scene comparison -> `scene_a`, `scene_b`.
- PrepareScene `mode` is the broad processing intent, such as
  `forest_change`, `forest_timeseries`, `forest_loss_history`,
  `forest_cross_modal`, `baseline_canopy`, `sar_change`, `wildfire_burn`,
  `forest_loss`, `built_up_gain`, `drought_anomaly`, `pollution_peak`,
  `multisource_hazard`, `xbd_damage`, `xbd_grid`, `xbd_quadrants`,
  `xbd_scene_summary`, `sar_flood`, `sar_flood_reference`,
  `sar_flood_diagnostic`, `flood_safe_routing`, or `scene_compare`.
- Raw input handles such as `forest_loss_history_1`, `optical_stack_1`,
  `damage_scene_1`, `sar_chip_1`, and `route_network_1` belong in ReadData's
  `input` argument. They are not valid `work` handles for later tools.
- Do not create imaginary AOI, boundary, vector, image, or raster handles. If
  the only listed evidence is raster/image/table/route data, open those handles
  directly with ReadData.
- Use only the tool names visible in ALLOWED TOOLS.
- Raw AVAILABLE INPUT HANDLES are declarations, not completed evidence. Do not
  emit TaskComplete only because an input handle exists. TaskComplete after a
  successful tool call or after Δ STATE / LAST OBSERVATION shows the requested
  produced scene/work/object/map/area handle or concrete value.

BOUNDARIES
- Do not perform semantic judgement, hazard classification, source comparison,
  ranking, exposure reasoning, route planning, route-tradeoff analysis, or
  final synthesis. The only exception: reporting the concrete artifact or value
  your own tool just produced.
- If a subtask needs a capability outside GA, return CONTRACT_FAILURE and name
  the missing specialist capability. Do not guess.
"""


FINAL_DATASET_PA_APPENDIX = """
GeoDisaster dataset - PA notes

ROLE
- Analytical reasoning and planning specialist. Pick tools by the requested
  decision, score, comparison, rank, exposure, or route objective. Tool
  contracts are in the tool signatures; do not reinterpret them.

OPERATING RULES
- Use only explicit labels, rules, and preferences taken from the question, the
  subtask, or prior tool output. Never invent thresholds.
- In all GeoDisaster tools, use `question: "question_1"` unless the current
  question exposes a different canonical question handle.
- Use only handles that are visible in PRIOR ARTIFACTS, Δ STATE, or LAST
  OBSERVATION. Do not guess `work_1`, `stats_1`, or route handles before they
  have been produced.
- Do not pass raw input handles such as `forest_loss_history_1`,
  `hazard_stack_1`, `damage_scene_1`, or `route_network_1` into analysis-tool
  `work` arguments. If only raw handles are visible, return CONTRACT_FAILURE
  asking ORC for GA ReadData/PrepareScene first.
- Do not call another agent as a tool, and do not call tools outside ALLOWED
  TOOLS. If a needed GeoDisaster handle is missing, return CONTRACT_FAILURE
  instead.
- Raw AVAILABLE INPUT HANDLES are declarations, not computed results. Do not
  emit TaskComplete just because an input handle exists. TaskComplete after your
  analysis tool returns the requested statistic, class, score, exposure,
  ranking, route, or comparison evidence.

BOUNDARIES
- Do not open raw inputs, prepare scenes, extract spatial objects, build flood
  masks, assess visual damage, describe flood imagery, or render maps. Ask ORC
  for any missing prerequisite instead of producing it yourself.

OUTPUT
- Your TaskComplete summary must carry the scalar, category, ranking, route
  decision, or tradeoff evidence ORC needs to terminate cleanly.
"""


FINAL_DATASET_VRA_APPENDIX = """
GeoDisaster dataset - VRA notes

ROLE
- Visual/semantic interpretation specialist. Use your tools to extract meaning
  from already-prepared visual, damage, or flood evidence. Tool contracts are
  in the tool signatures; do not reinterpret them.

OPERATING RULES
- In all GeoDisaster tools, use `question: "question_1"` unless the current
  question exposes a different canonical question handle.
- Use only handles that are visible in PRIOR ARTIFACTS, Δ STATE, or LAST
  OBSERVATION. Do not guess `work_1` or scene handles before they exist.
- Do not pass raw input handles such as `damage_scene_1`, `sar_chip_1`, or
  `scene_pair_1` into visual-tool `work` arguments. If only raw handles are
  visible, return CONTRACT_FAILURE asking ORC for the needed prepared evidence.
- Use only the GeoDisaster tools visible in ALLOWED TOOLS.
- Raw AVAILABLE INPUT HANDLES are declarations, not interpreted visual evidence.
  Do not emit TaskComplete only because an input handle exists. TaskComplete
  after a visual/damage/flood comparison tool returns the requested concrete
  evidence, or return CONTRACT_FAILURE naming the missing prepared handle.

BOUNDARIES
- Do not open data, prepare scenes, compute generic statistics, classify
  generic hazards, rank regions, plan or compare routes, or render artifacts.
- If a prepared handle you need is missing, return CONTRACT_FAILURE and name
  the missing prerequisite evidence.

OUTPUT
- Keep summaries concise but evidence-rich: state the concrete damage/flood/
  scene signal and the handle or metric that supports it.
"""
