# GeoDisaster Tool and Question Specification

This is the human-readable contract for the released GeoDisaster runtime and dataset. The
machine-readable authorities are:

- tool schemas and endpoints: `code/tool_registry/tool_api_map.json`;
- ownership: `code/tool_registry/tool_ownership.json`;
- prompt-facing signatures: `code/tool_registry/agent_tool_registry.json`;
- per-question chains and arguments: `data/core/geodisaster_2921_source_with_splits.jsonl`;
- answer schemas: `data/core/answer_schemas_43.json`;
- detailed semantics for all 43 types: `user-guide-final.md`.

## Released benchmark contract

- Canonical records: **2,921** (2,629 train, 292 evaluator-side test).
- Logical question types: **43** across deforestation, multi-hazard, xBD damage, routing, and
  Sen1Floods11 flood monitoring.
- Canonical SHA-256:
  `03988b960e6b09ae01fb622fe45b35d763eb705485cddc3cea68bcac7ae63fe1`.
- Split contract: `geodisaster_source_path_trajectory_grouped_split_v4_20261015`.
- Runtime surface: **18** GeoDisaster tools plus `TaskComplete`, which terminates a specialist step.
- Validation: 2,921/2,921 chains execute and pass field-aware checks; all required declared inputs
  resolve in the complete development snapshot.

Agents receive named artifact handles. `ReadData` validates and opens a declared input handle,
`PrepareScene` creates an aligned `work_N` context, analytical workers consume produced handles,
and `TaskComplete` returns the specialist result. Tools do not accept undeclared filesystem paths as
a substitute for handles.

## Ownership and signatures

`session_id` is injected by the runtime for every worker and is omitted from the prompt-facing
signatures below.

| Tool | Owner | Prompt-facing signature | Purpose |
|---|---|---|---|
| `ReadData` | GA | `ReadData(question, input, role?, time?, options?)` | Validate and open one declared input handle; return a scene handle. |
| `PrepareScene` | GA | `PrepareScene(question, inputs, mode, checks?)` | Validate semantic slots and assemble an aligned work context. |
| `MeasureArea` | GA | `MeasureArea(question, work, target, unit)` | Measure a mask or object footprint in the requested unit. |
| `ExtractObjects` | GA | `ExtractObjects(question, work, target, filters?)` | Extract loss patches, damaged buildings, flood polygons, or components. |
| `MapFlood` | GA | `MapFlood(question, work, method)` | Derive a flood-mask handle from the prepared SAR input. |
| `RenderMap` | GA | `RenderMap(question, work, layers, style)` | Render a fresh run-scoped artifact from current handles and declared layers. |
| `ComputeStats` | PA | `ComputeStats(question, work, metrics, region?)` | Compute requested counts, areas, summaries, distributions, or footprint statistics. |
| `CompareSources` | PA | `CompareSources(question, work, compare)` | Compare modalities, source products, or known-water overlap. |
| `DetectChange` | PA | `DetectChange(question, work, change, window?)` | Detect a temporal change or dominant change window. |
| `ClassifyChange` | PA | `ClassifyChange(question, work, labels, rule)` | Apply the declared state/change classification rule. |
| `ScoreHazard` | PA | `ScoreHazard(question, work, hazard, scale)` | Score the requested hazard class, severity, or footprint. |
| `EstimateExposure` | PA | `EstimateExposure(question, work, exposure, metrics)` | Estimate exposed population, assets, roads, schools, or facilities. |
| `RankRegions` | PA | `RankRegions(question, work, by, candidates, top_k?)` | Rank grids, quadrants, components, regions, or route candidates. |
| `PlanRoutes` | PA | `PlanRoutes(question, work, objective, avoid)` | Select from the declared flood-screened candidate routes and shelter product. |
| `CompareRoutes` | PA | `CompareRoutes(question, work, prefer)` | Report declared F1/S1/B1 route trade-offs without inventing a recommendation. |
| `AssessDamage` | VRA | `AssessDamage(question, work, aggregation, severity?)` | Aggregate damage polygons by scene, grid, quadrant, severity, or scene pair. |
| `DescribeFlood` | VRA | `DescribeFlood(question, work, metrics)` | Report flood area, percentage, components, grids, and overlap statistics. |
| `CompareScenes` | VRA | `CompareScenes(question, work, by)` | Compare two prepared damage/risk scene summaries. |

ORC owns no workers. It creates the plan, delegates typed steps to GA/PA/VRA, tracks returned
handles, and terminates with `{"tool":"Terminate","answer_ref":"ANSWER_JSON:answer_1"}` after the
runtime publishes that schema-valid worker-evidence projection.

## Question-type inventory

Each logical type has one defined canonical sequence. Counts below include train and test.

| Family | Types | Records per type | Canonical worker pattern |
|---|---|---|---|
| Deforestation | `forest_loss_detection`, `area_estimation`, `change_window_estimation`, `false_positive_rejection`, `baseline_canopy_assessment`, `multi_year_loss_attribution`, `cross_modal_concordance`, `severity_proxy_numeric` | 48 each | Open declared optical/SAR/Hansen inputs; prepare the type-specific context; compare, measure, classify, or summarize; complete. |
| Multi-hazard wildfire | `A`, `B`, `C`, `D` | 15 each | Prepare hazard evidence; compute footprint, score severity, compare sources, or estimate exposure. |
| Multi-hazard deforestation | `A`, `B`, `C`, `D` | 23 each | Prepare event loss evidence; compute footprint, score class, compare sources, or estimate exposure. |
| Multi-hazard urban expansion | `A`, `B`, `C`, `D` | 29 each | Prepare built-up evidence; compute gain, score class, compare sources, or estimate exposure. |
| Multi-hazard drought | `A`, `B`, `C`, `D` | 21 each | Prepare vegetation/temperature evidence; compute anomaly, score class, compare sources, or estimate exposure. |
| Multi-hazard air pollution | `A`, `B`, `C`, `D` | 20 each | Prepare NO2 evidence; compute footprint, score class, compare sources, or estimate exposure. |
| xBD damage | `XBD-Q1`, `XBD-Q2`, `XBD-Q3`, `XBD-Q4`, `XBD-Q5`, `XBD-Q7`, `XBD-Q8` | 177, 177, 177, 55, 121, 121, 177 | Open imagery and authoritative polygons; prepare; aggregate, rank, compare, extract, or summarize. |
| Routing | `nearest_shelter`, `fastest_safest`, `visual_overlay` | 200 each | Open the packaged route context; select/compare declared candidates; render visual overlays live when requested. |
| Sen1Floods11 | `A`, `B`, `C`, `D`, `E` | 100 each | Open S1 and, for D/E, JRC context; map flood; describe/extract/rank/compare; render the requested analysis image for E. |

The exact ordered calls and arguments are stored in each record's `gt_chain`; this table is a
readable overview rather than a second source of truth.

## Output and failure policy

- Each analytical call must use a valid handle produced earlier in the episode.
- A missing required input, invalid handle, incompatible semantic slot, empty required computation,
  or unavailable output is an explicit failure; workers do not fabricate a value.
- Optional artifacts may be absent only where the question contract marks them optional and the
  worker does not need them for the answer.
- Final answers are JSON values governed by `answer_schemas_43.json`. Deterministic correctness
  requires matching structure, keys, scalar types, null states, list order, and numeric leaves within
  the published tolerance.
- Generated maps use run-scoped artifact paths and preserve source CRS/georeferencing. Prepared
  target overlays and hidden/ground-truth products are not agent inputs.
