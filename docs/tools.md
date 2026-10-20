# Tools

GeoDisaster exposes **18 disaster-oriented geospatial tools**. Each tool is a standalone worker
behind a controller; agents call tools through JSON contracts and pass results between steps by
handle.

## Tool categories

| Category | Tools | Owner |
|---|---|---|
| **Data access & preparation** | `ReadData`, `PrepareScene` | GA |
| **Geometry & extraction** | `MeasureArea`, `ExtractObjects`, `MapFlood`, `RenderMap` | GA |
| **Numeric analytics** | `ComputeStats`, `CompareSources`, `DetectChange`, `ClassifyChange` | PA |
| **Decision analytics** | `ScoreHazard`, `EstimateExposure`, `RankRegions`, `PlanRoutes`, `CompareRoutes` | PA |
| **Visual / semantic** | `AssessDamage`, `DescribeFlood`, `CompareScenes` | VRA |

High-level behaviour: `ReadData` opens a named input handle and validates its artifacts; `PrepareScene`
builds an aligned work context (per task mode: forest / hazard / xbd_damage / sar_flood / routing /
scene_compare); the analytic tools then consume those work handles to compute areas, statistics,
change masks, hazard scores, exposure estimates, region rankings, and routes; the visual tools
produce damage inventories, flood descriptions, and scene comparisons; `RenderMap` produces map
overlays.

## Tool registry (per agent)

- **Ownership** (which role may call which tool):
  [`code/tool_registry/tool_ownership.json`](../code/tool_registry/tool_ownership.json).
- **Agent-friendly registry & templates** (names, instructions, argument templates):
  [`code/tool_registry/agent_tool_registry.json`](../code/tool_registry/agent_tool_registry.json),
  [`code/tool_registry/tool_templates_43.json`](../code/tool_registry/tool_templates_43.json).
- **API map** (tool → endpoint/port wiring):
  [`code/tool_registry/tool_api_map.json`](../code/tool_registry/tool_api_map.json).
- **Authoritative tool schemas** (descriptions, properties, required args) are defined in code:
  [`code/tool_server/.../final_dataset_tool_specs.py`](../code/tool_server/tool_workers/online_workers/final_dataset_tool_specs.py).

## Tool workers path

- **Workers:** [`code/tool_server/tool_workers/online_workers/`](../code/tool_server/tool_workers/online_workers/)
  — one `<Tool>_worker.py` per tool, plus `controller.py` and the shared base/runtime modules.
- **Launch:** [`code/scripts/start_tool_workers.sh`](../code/scripts/start_tool_workers.sh) generates
  the server config and starts the controller + workers; config in
  [`code/configs/tool_server_config.yaml`](../code/configs/tool_server_config.yaml).
- **Agent-facing gateway:** [`code/multi_agent/tool_gateway.py`](../code/multi_agent/tool_gateway.py)
  validates ownership and arguments, reconciles `ReadData` with the declared handle card, and
  dispatches to the same real worker implementation either directly (`inprocess`) or through the
  controller (`http`). Neither backend substitutes fixture payloads or expected answers.

The gateway installs a sanitized per-case context. Gold answers and teacher chains never enter the
worker context. A specific role/time on an input handle card is authoritative; omitted values are
inherited, while contradictory calls fail with a structured contract error.

Worker results are deliberately richer than many terminal answer schemas. The answer-contract
layer projects evidence produced by successful calls into exactly the declared schema. A token such
as `ANSWER_JSON:answer_1` refers to that schema-valid projection, not to the worker's raw payload
or artifact handle.

## Attribution

The agentic tool-surface design — named tool workers behind a controller, JSON tool contracts, and
handle-passing between steps — follows the
[OpenEarthAgent](https://github.com/mbzuai-oryx/OpenEarthAgent) tool-server style. The GeoDisaster
tool implementations, the 18-tool surface, and the disaster task semantics are specific to this
project.

The executable authority is
[`final_dataset_tool_specs.py`](../code/tool_server/tool_workers/online_workers/final_dataset_tool_specs.py)
together with `FINAL_DATASET_AGENT_TOOLS` in
[`final_dataset_profile.py`](../code/multi_agent/final_dataset_profile.py). The JSON registries are
synchronized, human-readable mirrors checked by `scripts/sync_tool_contracts.py`; editing a JSON
mirror alone does not change runtime availability. Integrating a different tool family requires a
new explicit profile, schemas, ownership map, gateway dispatch, prompts, and tests. This public
release accepts only the GeoDisaster profile.
