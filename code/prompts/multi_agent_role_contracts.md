# GeoDisaster Multi-Agent Role Contract

Enable this benchmark profile with `MAS_AGENT_PROFILE=geodisaster`. ORC plans, delegates, tracks
produced handles, and terminates; it owns no workers. The executable signatures are generated from
`tool_server/tool_workers/online_workers/final_dataset_tool_specs.py`. The JSON registries and
runtime prompts use that same authority.

## Ownership

| Agent | Tools | Responsibility |
|---|---|---|
| `GA` | `ReadData`, `PrepareScene`, `MeasureArea`, `ExtractObjects`, `MapFlood`, `RenderMap` | Open declared data, establish handle lineage, perform spatial extraction/measurement, and produce requested artifacts. |
| `PA` | `ComputeStats`, `CompareSources`, `DetectChange`, `ClassifyChange`, `ScoreHazard`, `EstimateExposure`, `RankRegions`, `PlanRoutes`, `CompareRoutes` | Quantitative analysis, comparison, classification, exposure/ranking, and route decisions. |
| `VRA` | `AssessDamage`, `DescribeFlood`, `CompareScenes` | Damage/flood interpretation and scene-level comparison from prepared evidence. |

## Agent-facing signatures

- `ReadData(question: str, input: str, role?: str, time?: str, options?: dict)`
- `PrepareScene(question: str, inputs: dict, mode: str, checks?: dict)`
- `ComputeStats(question: str, work: str, metrics: dict, region?: str)`
- `MeasureArea(question: str, work: str, target: str, unit: str)`
- `CompareSources(question: str, work: str, compare: dict)`
- `DetectChange(question: str, work: str, change: str, window?: str)`
- `ClassifyChange(question: str, work: str, labels: dict, rule: str)`
- `ScoreHazard(question: str, work: str, hazard: str, scale: str)`
- `EstimateExposure(question: str, work: str, exposure: str, metrics: dict)`
- `AssessDamage(question: str, work: str, aggregation: str, severity?: str)`
- `RankRegions(question: str, work: str, by: str, candidates: str, top_k?: int)`
- `CompareScenes(question: str, work: str, by: str)`
- `ExtractObjects(question: str, work: str, target: str, filters?: dict)`
- `MapFlood(question: str, work: str, method: str)`
- `DescribeFlood(question: str, work: str, metrics: dict)`
- `PlanRoutes(question: str, work: str, objective: str, avoid: dict)`
- `CompareRoutes(question: str, work: str, prefer: dict)`
- `RenderMap(question: str, work: str, layers: dict, style: str)`

`session_id` and case bootstrap data are injected by the evaluator transport and are not
agent-facing arguments. Specialists may call only their owned tools plus `TaskComplete`.
Every analytical call consumes a handle created earlier in the same session.

Detailed argument descriptions and required fields are in
`../tool_registry/agent_tool_registry.json` and `../tool_registry/tool_api_map.json`.
