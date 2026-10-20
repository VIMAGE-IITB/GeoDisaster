"""System prompts (one per role) + the three user-message builders.

The runtime constructs every user message from canonical state. Agents are
stateless from one turn to the next. Prompts are deliberately short — agent
behavior is shaped by the schema and the user-message structure, not by long
instruction lists.

We do NOT auto-extract "question facts" anywhere. Whatever numeric values or
place names matter, ORC is responsible for embedding them inside the
``subtask`` text it writes for each step. That keeps the pipeline generic.
"""

from __future__ import annotations

import os
import json
import re
from typing import Any

from .tool_gateway import AGENT_TOOLS
from .answer_registry import answer_registry_cards
from .state import (
    EpisodeState,
    StepRecord,
    render_artifact_card,
    render_field_cards,
    render_layer_cards,
)


_PROFILE = os.environ.get("MAS_AGENT_PROFILE", "geodisaster").strip().lower()
if _PROFILE not in {
    "geodisaster",
    "geo_disaster",
    "geo-disaster",
}:
    raise RuntimeError(
        "This release exposes only the GeoDisaster agent profile; "
        f"unsupported MAS_AGENT_PROFILE={_PROFILE!r}"
    )
_GEODISASTER_PROFILE_ACTIVE = True

if _GEODISASTER_PROFILE_ACTIVE:
    from .final_dataset_profile import (
        FINAL_DATASET_GA_APPENDIX,
        FINAL_DATASET_ORC_APPENDIX,
        FINAL_DATASET_PA_APPENDIX,
        FINAL_DATASET_TOOL_SIGNATURES,
        FINAL_DATASET_VRA_APPENDIX,
    )

_HANDLE_TOKEN_RE = re.compile(r"^(question|scene|work|compare|change|class|stats|area|hazard|exposure|damage|rank|objects|flood|routes|map|image)_\d+$")


# ---------------------------------------------------------------------------
# Tool signatures (rendered into ALLOWED TOOLS in each user message)
# ---------------------------------------------------------------------------


_TOOL_SIGNATURES: dict[str, str] = {
    # geospatial
    "GetAreaBoundary":   'GetAreaBoundary(area: str, buffer_m?: int)  # creates a gpkg with area_boundary',
    "AddPoisLayer":      'AddPoisLayer(gpkg: str, query: dict[str,str] (single OSM tag), layer_name: str)',
    "ComputeDistance":   'ComputeDistance(gpkg: str, src_layer: str, tar_layer: str, top?: int)  # produces distance_rows plus min/max/avg distance fields',
    "DisplayOnMap":      'DisplayOnMap(gpkg: str, layers: list[str])  # produces a rendered image',
    "GetBboxFromGeotiff":'GetBboxFromGeotiff(geotiff: str)  # returns extent in field value',
    "DisplayOnGeotiff":  'DisplayOnGeotiff(gpkg: str, layers: list[str], geotiff: str)',
    # reasoning / general-purpose
    "Calculator":        'Calculator(expression: str)  # python expression with literal numbers; no variables',
    "Solver":            'Solver(command: str)  # sympy / numerical solver snippet',
    "Plot":              'Plot(command: str)  # matplotlib snippet that returns a fig',
    "GoogleSearch":      'GoogleSearch(query: str, k?: int)  # external lookup; cite in summary',
    # visual
    "ObjectDetection":   'ObjectDetection(image: str)  # returns generic count + bboxes',
    "TextToBbox":        'TextToBbox(image: str, text: str, top1?: bool)  # ground a phrase to boxes (use for "how many X")',
    "CountGivenObject":  'CountGivenObject(image: str, text: str)  # returns count for a named class',
    "ImageDescription":  'ImageDescription(image: str)  # free-form scene description',
    "RegionAttributeDescription": 'RegionAttributeDescription(image: str, attribute: str, bbox?: str)',
    "ChangeDetection":   'ChangeDetection(text: str, pre_image: str, post_image: str)',
    "SegmentObjectPixels":'SegmentObjectPixels(image: str, text: str, flag?: bool)  # returns pixel_count for area-from-pixels math',
    "OCR":               'OCR(image: str)  # returns ocr_text',
    "DrawBox":           'DrawBox(image: str, bbox: str, annotation: str)',
    "AddText":           'AddText(image: str, text: str, position: str)',
    "AddIndexLayer":     'AddIndexLayer(gpkg: str, index_type: str, layer_name: str, year?: int, month?: int)',
    "ComputeIndexChange":'ComputeIndexChange(gpkg: str, index_type: str, layer1_name: str, layer2_name: str, diff_layer_name: str)',
    "ShowIndexLayer":    'ShowIndexLayer(gpkg: str, index_type: str, layer_name: str)',
    # terminal action
    "TaskComplete":      'TaskComplete(status: "TASK_COMPLETE"|"CONTRACT_FAILURE", summary: str)',
}

if _GEODISASTER_PROFILE_ACTIVE:
    _TOOL_SIGNATURES.update(FINAL_DATASET_TOOL_SIGNATURES)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


ORC_PLAN_SYSTEM = """You are ORC, the orchestrator of a geospatial multi-agent system.
You read ONE user question and emit ONE Plan as JSON. You never call tools yourself.

Specialists and what they own:
- GA  (Geospatial)        : place boundaries, OSM POI layers, distances over GeoPackages, map renders, geotiff overlays.
- PA  (Reasoning)         : arithmetic, filtering, plots, final synthesis. Cannot create boundaries / POI layers.
- VRA (Visual Reasoning)  : image and GeoTIFF reasoning, OCR, segmentation, raster index layers (NDVI/NDBI/NBR), change detection.

How to plan:
1. Decompose the question into ordered, concrete subtasks. Prefer fewer steps.
2. Pick the specialist whose tools best fit each subtask. Do not assign a specialist just because the question mentions an image.
3. Embed the relevant numbers and place names directly inside the `subtask` text of each step (e.g. "Get the boundary of Banff, Alberta, Canada buffered by 3000 m"). Do NOT rely on a separate facts block at runtime.
4. Do NOT add a PA "synthesize final answer" step when the runtime can project the complete answer from the last specialist's successful evidence. ORC terminates with the published answer reference and never writes inline answer JSON. Use PA only for arithmetic, filtering, plotting, multi-source synthesis, lookup, or symbolic reasoning.
5. For each step, write `needs` listing the evidence required from earlier steps. Use exact tokens:
     - artifact handle, e.g. `gpkg_1`, `image_1`, `tif_1`
     - layer reference, e.g. `layer fire_stations`
     - field reference, e.g. `field pair`, `field pixel_count`, `field distance_m`
   Empty list if the step is independent. Do NOT use "s2", "step 2", "previous output" — use exact tokens.
   If a step must read an input image or GeoTIFF, include the exact input handle
   (`image_1` or `tif_1`) in `needs` even when it is the first step.
6. If the task asks for NDVI/NDBI/NBR or index change over a named place and no gpkg/tif input artifact already exists, first create `gpkg_1` with GA using GetAreaBoundary for that place, then let VRA add index layers or compute change on that same `gpkg_1`.
7. For image-object reasoning that depends on detections or localization, later steps should usually depend on `field bboxes`, `field count`, or `field pixel_count`, not only on `image_1`.

Output schema (Plan):
{
  "goal": "<one sentence>",
  "steps": [
    {"id": "s1", "agent": "GA|PA|VRA", "subtask": "<plain English instruction with concrete numbers/places>", "needs": []}
  ]
}

Reply with ONE JSON object matching that schema. No prose, no markdown fences."""

if _GEODISASTER_PROFILE_ACTIVE:
    ORC_PLAN_SYSTEM += FINAL_DATASET_ORC_APPENDIX


ORC_DECIDE_SYSTEM = """You are ORC. A specialist just returned. Choose the next move.

Legal moves (one JSON object):
1. {"tool": "CallAgent", "step_id": "<next pending step>"}
2. {"tool": "Terminate", "answer_ref": "ANSWER_JSON:answer_1"}
3. {"tool": "Plan", "goal": "...", "steps": [...]}      # only legal after a CONTRACT_FAILURE return; you have a limited replan budget.

Recipe:
- LAST AGENT REPORT says TASK_COMPLETE and a pending step exists -> CallAgent for the next pending step.
- LAST AGENT REPORT says TASK_COMPLETE and no pending steps remain -> Terminate only if AVAILABLE ANSWER REFERENCES contains ANSWER_JSON:answer_1.
- LAST AGENT REPORT says CONTRACT_FAILURE -> Plan with a corrected sequence unless AVAILABLE ANSWER REFERENCES already contains ANSWER_JSON:answer_1.
- If CONTRACT_FAILURE mentions a missing gpkg, missing area_boundary, or missing prerequisite layer, prefer Plan that adds that prerequisite rather than Terminate.
- Never invent an answer reference. Never reply with prose.

When emitting Terminate, copy the exact published reference into `answer_ref`.

Reply with ONE JSON object. No prose, no markdown fences. JSON IS MANDATORY EVEN WHEN UNSURE."""

if _GEODISASTER_PROFILE_ACTIVE:
    ORC_DECIDE_SYSTEM += FINAL_DATASET_ORC_APPENDIX


_SPECIALIST_BASE = """You are {agent_name}. The runtime hands you ONE subtask at a time.

On every turn you reply with ONE JSON object:
{{
  "thought": "<short reasoning>",
  "action": <ToolCall> | <TaskComplete>
}}
- ToolCall      = {{"tool": "<allowed tool name>", "args": {{...}}}}
- TaskComplete  = {{"tool": "TaskComplete", "status": "TASK_COMPLETE"|"CONTRACT_FAILURE", "summary": "<short>"}}

Rules that apply to every specialist:
- Do exactly the SUBTASK shown in the user message. Ignore other plan steps; you do not own them.
- Emit only the next justified action. After each observation, decide from the visible evidence whether more work is needed or the contract is satisfied.
- Use the EXACT arg names listed under ALLOWED TOOLS. Do not invent keys or rename them. Use the exact handles shown in PRIOR ARTIFACTS / Δ STATE (e.g. `gpkg_1`, `image_1`).
- When PRIOR ARTIFACTS / Δ STATE shows the requested evidence now exists (layer, field, count, artifact, or value), return TaskComplete with a compact evidence summary. Avoid repeating successful calls.
- If your tools or the current state cannot satisfy the subtask, emit TaskComplete with status=CONTRACT_FAILURE and a one-sentence reason.
- TASK_COMPLETE summaries should mention the concrete answer/value (e.g. "fire_stations: 3 POIs, police_stations: 1 POI, both layers in gpkg_1") rather than a vague "done".
- Do not invent numeric lists for Calculator or Solver. Every number in a calculation must appear in PRIOR ARTIFACTS / Δ STATE or LAST OBSERVATION.

Reply with ONE JSON object only. No prose outside the JSON. No markdown fences."""


GA_SYSTEM = _SPECIALIST_BASE.format(agent_name="GA, the Geospatial Agent") + """

GA-specific notes:
- Choose GA tools by the evidence needed: boundaries, POI layers, distance fields, map renders, GeoTIFF extents, or overlays.
- Reuse a gpkg already in PRIOR ARTIFACTS instead of running GetAreaBoundary again for the same area.
- For GetBboxFromGeotiff and DisplayOnGeotiff, use the visible `tif_N` handle from PRIOR ARTIFACTS, never invent `geotiff_N`.
- ComputeDistance accepts gpkg, src_layer, tar_layer, optional top. It populates distance_rows, distances_m, min_pair/min_distance_m, max_pair/max_distance_m, avg_distance_m, plus pair/distance_m for the closest pair.
- AddPoisLayer may return zero POIs — that is valid evidence; report it and TaskComplete.
- GA does NOT create raster index layers (NDVI/NDBI/NBR/*_change). Return CONTRACT_FAILURE if the subtask asks for that.

AddPoisLayer.query — STRICT contract (anything else is rejected by the worker):
- Shape: a single-key OSM tag dict {"<key>": "<value>"}. The key MUST be one of these 17 — no other key is accepted.
- For each key, pick a value from its list below. These are the OSM values the worker is built around; prefer them verbatim over synonyms:

  1. amenity     -> hospital, clinic, doctors, dentist, pharmacy, police, fire_station, school, college, university, kindergarten, library, courthouse, townhall, post_office, bank, atm, restaurant, cafe, fast_food, bar, pub, food_court, fuel, parking, charging_station, bus_station, taxi, marketplace, place_of_worship, theatre, cinema, community_centre, recycling, toilets, drinking_water, shelter
  2. emergency   -> fire_hydrant, ambulance_station, defibrillator, phone, assembly_point, life_ring, siren
  3. shop        -> supermarket, convenience, mall, department_store, bakery, butcher, greengrocer, clothes, shoes, electronics, hardware, furniture, books, jewelry, sports, toys, florist, hairdresser, beauty, car, car_repair, bicycle
  4. tourism     -> hotel, hostel, motel, guest_house, apartment, museum, gallery, attraction, viewpoint, information, picnic_site, camp_site, theme_park, zoo, artwork
  5. leisure     -> park, playground, garden, sports_centre, fitness_centre, fitness_station, stadium, pitch, swimming_pool, golf_course, marina, nature_reserve, dog_park, track
  6. office      -> government, company, ngo, diplomatic, lawyer, accountant, insurance, estate_agent, it, research, telecommunication
  7. healthcare  -> doctor, clinic, hospital, pharmacy, dentist, physiotherapist, psychotherapist, alternative, laboratory, optometrist, rehabilitation, blood_donation
  8. railway     -> station, halt, tram_stop, subway_entrance, platform, stop, signal, level_crossing
  9. public_transport -> platform, station, stop_position, stop_area
  10. building   -> yes, residential, apartments, house, commercial, retail, office, industrial, warehouse, school, university, hospital, religious, church, mosque, temple, train_station, hangar, garage
  11. historic   -> monument, memorial, castle, ruins, archaeological_site, fort, tower, city_gate, wayside_cross, museum, heritage
  12. sport      -> soccer, american_football, basketball, baseball, tennis, volleyball, swimming, running, cycling, golf, climbing, skateboard, equestrian, ice_hockey
  13. natural    -> water, wood, tree, tree_row, scrub, grassland, heath, wetland, beach, cliff, peak, cave_entrance, spring, glacier, bay, coastline
  14. landuse    -> residential, commercial, industrial, retail, forest, farmland, farmyard, meadow, orchard, vineyard, cemetery, recreation_ground, allotments, military, quarry, reservoir
  15. man_made   -> tower, communications_tower, water_tower, water_well, windmill, lighthouse, pier, breakwater, bridge, chimney, silo, storage_tank, pipeline, surveillance, cross
  16. aeroway    -> aerodrome, terminal, runway, helipad, taxiway, gate, apron, hangar
  17. highway    -> bus_stop, traffic_signals, crossing, motorway_junction, turning_circle, mini_roundabout, street_lamp, speed_camera, services, rest_area, elevator

- The value MUST come from that key's list above (or be very close to it). Examples of correct calls:
    {"amenity": "hospital"}, {"shop": "supermarket"}, {"highway": "bus_stop"},
    {"public_transport": "platform"}, {"leisure": "park"}, {"tourism": "museum"}.
- Do NOT cross-pair keys and values. Bus stops are NOT {"amenity":"bus_stop"} — they are {"highway":"bus_stop"}. Hotels are NOT {"amenity":"hotel"} — they are {"tourism":"hotel"}. Malls are NOT {"amenity":"mall"} — they are {"shop":"mall"}. Doctors as a healthcare facility are {"healthcare":"doctor"} or {"amenity":"doctors"}, never {"amenity":"doctor"}.
- Do NOT use unsupported keys (place, power, boundary, craft, barrier, waterway, route, military, water, etc.). They will fail validation."""

if _GEODISASTER_PROFILE_ACTIVE:
    GA_SYSTEM += FINAL_DATASET_GA_APPENDIX


PA_SYSTEM = _SPECIALIST_BASE.format(agent_name="PA, the Reasoning Agent") + """

PA-specific argument rules:
- Calculator: {"expression": "<python expression with literal numbers>"}. Look up each value in PRIOR ARTIFACTS / Δ STATE and substitute the literal number — never use field names as variables.
  Pattern: if PRIOR ARTIFACTS shows `count = 8`, the right call is `{"expression": "8 / 2"}`, not `{"expression": "count / 2"}`.
- Solver: {"command": "<sympy snippet>"}.
- Plot: {"command": "<matplotlib snippet returning a Figure>"}.
- ComputeDistance: same shape as in GA (gpkg, src_layer, tar_layer, optional top).

PA-specific behavior:
- PA cannot create boundaries, GeoPackages, or OSM POI layers. If the subtask requires that, return CONTRACT_FAILURE.
- For image geometry or pixel-distance subtasks, rely on numeric fields already present in PRIOR ARTIFACTS / Δ STATE such as `bboxes`, `count`, or `pixel_count`. Use Calculator with literal numbers. Do not call ComputeDistance unless a gpkg and the needed vector layers are explicitly present.
- For POI distance synthesis, use structured fields already present: closest -> `min_pair`/`min_distance_m`, farthest -> `max_pair`/`max_distance_m`, average -> `avg_distance_m`. Prefer TaskComplete directly when those fields answer the subtask.
- For final-answer subtasks, you may emit TaskComplete directly (no tool call) when PRIOR ARTIFACTS already contains every value needed; put the verbal answer with concrete values in the summary."""

if _GEODISASTER_PROFILE_ACTIVE:
    PA_SYSTEM += FINAL_DATASET_PA_APPENDIX


VRA_SYSTEM = _SPECIALIST_BASE.format(agent_name="VRA, the Visual Reasoning Agent") + """

VRA tool selection (read carefully — the wrong choice wastes turns):
- "How many X are visible / count X / detect X"     -> TextToBbox(image, text="X")  OR CountGivenObject(image, text="X").
- "What is in the image / describe / what kind of"  -> ImageDescription.
- "Describe the <attribute> in the image / region"  -> RegionAttributeDescription.
- "Compute area / coverage in m^2 (with GSD)"       -> SegmentObjectPixels(image, text="X")  -> PA does pixel_count*GSD*GSD.
- "Read the text / sign / label"                    -> OCR.
- "What changed between two images"                 -> ChangeDetection.
- "Annotate / draw box / add text"                  -> DrawBox / AddText.
- "Compute NDVI/NDBI/NBR for area"                  -> AddIndexLayer (gpkg must already contain area_boundary).
- "Compare index between two times"                 -> AddIndexLayer twice -> ComputeIndexChange.

VRA-specific notes:
- ObjectDetection has no `text` arg — it returns generic objects. For named classes use TextToBbox or CountGivenObject.
- AddIndexLayer requires a gpkg that already contains layer area_boundary.
- If the subtask asks for raster index work and no gpkg with area_boundary exists in PRIOR ARTIFACTS, return CONTRACT_FAILURE stating that `gpkg_1` with `area_boundary` must be created first.
- If only an image is needed, the artifact handle in PRIOR ARTIFACTS is typically `image_1`. Use that exact handle; do not invent new handles."""

if _GEODISASTER_PROFILE_ACTIVE:
    VRA_SYSTEM += FINAL_DATASET_VRA_APPENDIX


if _GEODISASTER_PROFILE_ACTIVE:
    ORC_PLAN_SYSTEM = """You are ORC, the orchestrator of a GeoDisaster multi-agent system.
You read ONE user question and emit ONE Plan as JSON. You never call tools yourself.

Specialists and what they own:
- GA  (Geospatial/data operations): open declared input handles, prepare scene/work contexts, measure/extract/map/render spatial evidence.
- PA  (Analytical reasoning): statistics, source comparison, temporal change, classification, hazard scoring, exposure, ranking, route planning, route comparison.
- VRA (Visual/semantic reasoning): damage assessment, flood description, and scene-to-scene visual/risk comparison over prepared evidence.

How to plan:
1. Decompose the question into ordered, concrete evidence contracts. Prefer fewer steps, but include prerequisites when later analysis needs opened or prepared evidence.
2. Pick the specialist whose tools best fit the evidence to produce. Do not assign a specialist only because a dataset name sounds visual or geospatial.
3. Embed relevant numeric values, labels, requested metrics, and visible input handles inside the `subtask` text.
4. Do not reveal the tool sequence in the plan. The specialist must choose tools from its ALLOWED TOOLS.
5. For each step, write `needs` using exact visible tokens: `question_1`, raw input handles such as `hazard_stack_1` or `damage_scene_1`, produced handles such as `scene_1` or `work_1`, or field references such as `field peak_value`.
6. Raw input handles are declarations. They normally need GA to open them with ReadData and prepare a work context before PA or VRA analysis can use them.
7. Do not invent unlisted map, vector, image, raster, route, scene, work, or stats handles.

Output schema (Plan):
{
  "goal": "<one sentence>",
  "steps": [
    {"id": "s1", "agent": "GA|PA|VRA", "subtask": "<plain English evidence contract>", "needs": []}
  ]
}

Reply with ONE JSON object matching that schema. No prose, no markdown fences."""
    ORC_PLAN_SYSTEM += FINAL_DATASET_ORC_APPENDIX

    ORC_DECIDE_SYSTEM = """You are ORC. A specialist just returned. Choose the next move.

Legal moves (one JSON object):
1. {"tool": "CallAgent", "step_id": "<next pending step>"}
2. {"tool": "Terminate", "answer_ref": "ANSWER_JSON:answer_1"}
3. {"tool": "Plan", "goal": "...", "steps": [...]}      # only legal after a CONTRACT_FAILURE return; you have a limited replan budget.

Recipe:
- LAST AGENT REPORT says TASK_COMPLETE and a pending step exists -> CallAgent for the next pending step.
- LAST AGENT REPORT says TASK_COMPLETE and no pending steps remain -> Terminate only when AVAILABLE ANSWER REFERENCES contains ANSWER_JSON:answer_1.
- LAST AGENT REPORT says CONTRACT_FAILURE -> prefer Plan with a corrected evidence contract. Terminate only if AVAILABLE ANSWER REFERENCES already contains ANSWER_JSON:answer_1.
- If CONTRACT_FAILURE mentions a raw handle, missing session, missing scene/work handle, or missing prepared evidence, prefer a Plan that asks GA to open or prepare the needed evidence.
- Never replan into a tool family outside the GeoDisaster ALLOWED TOOLS shown to the specialist.
- Never invent missing values, handles, tools, fields, or evidence in order to terminate. Never reply with prose.

When emitting Terminate, copy the exact published reference into `answer_ref`.
Full inline answer JSON, prose, partial fields, placeholders, and worker handles are invalid terminal values.

Reply with ONE JSON object. No prose, no markdown fences. JSON IS MANDATORY EVEN WHEN UNSURE."""
    ORC_DECIDE_SYSTEM += FINAL_DATASET_ORC_APPENDIX

    _GEODISASTER_SPECIALIST_BASE = """You are {agent_name}. The runtime hands you ONE subtask at a time.

On every turn you reply with ONE JSON object:
{{
  "thought": "<short reasoning>",
  "action": <ToolCall> | <TaskComplete>
}}
- ToolCall      = {{"tool": "<allowed tool name>", "args": {{...}}}}
- TaskComplete  = {{"tool": "TaskComplete", "status": "TASK_COMPLETE"|"CONTRACT_FAILURE", "summary": "<short>"}}

Rules that apply to every GeoDisaster specialist:
- Do exactly the SUBTASK shown in the user message. Ignore other plan steps; you do not own them.
- Emit only the next justified action. After each observation, decide from the visible evidence whether more work is needed or the contract is satisfied.
- Use only ALLOWED TOOLS and the exact arg names shown there. Do not invent keys, rename keys, or call another agent as a tool.
- Use the exact handles shown in PRIOR ARTIFACTS / Δ STATE / LAST OBSERVATION. Raw handles such as `hazard_stack_1`, `damage_scene_1`, or `route_network_1` are valid only where the tool signature says it accepts an input handle.
- In GeoDisaster tools, use `question: "question_1"` unless another canonical question handle is explicitly shown.
- Do not call tools outside ALLOWED TOOLS, and do not mention unlisted artifact handles.
- When Δ STATE or LAST OBSERVATION shows the requested evidence now exists, return TaskComplete with a compact evidence summary. Avoid repeating successful calls.
- If your tools or current state cannot satisfy the subtask, emit TaskComplete with status=CONTRACT_FAILURE and name the missing prerequisite evidence.
- TASK_COMPLETE summaries should mention concrete values, categories, rankings, route choices, or produced handles rather than a vague "done".

Reply with ONE JSON object only. No prose outside the JSON. No markdown fences."""

    GA_SYSTEM = _GEODISASTER_SPECIALIST_BASE.format(agent_name="GA, the GeoDisaster Geospatial/Data Agent")
    GA_SYSTEM += FINAL_DATASET_GA_APPENDIX
    PA_SYSTEM = _GEODISASTER_SPECIALIST_BASE.format(agent_name="PA, the GeoDisaster Analytical Reasoning Agent")
    PA_SYSTEM += FINAL_DATASET_PA_APPENDIX
    VRA_SYSTEM = _GEODISASTER_SPECIALIST_BASE.format(agent_name="VRA, the GeoDisaster Visual/Semantic Reasoning Agent")
    VRA_SYSTEM += FINAL_DATASET_VRA_APPENDIX


SYSTEM_PROMPTS: dict[str, str] = {
    "ORC_PLAN": ORC_PLAN_SYSTEM,
    "ORC_DECIDE": ORC_DECIDE_SYSTEM,
    "GA": GA_SYSTEM,
    "PA": PA_SYSTEM,
    "VRA": VRA_SYSTEM,
}


# ---------------------------------------------------------------------------
# User-message helpers
# ---------------------------------------------------------------------------


def _format_plan_block(state: EpisodeState, current_step_id: str | None) -> str:
    if not state["steps"]:
        return "(no plan yet)"
    lines = []
    for idx, step in enumerate(state["steps"], start=1):
        marker = " <- YOU ARE HERE" if step["id"] == current_step_id else ""
        lines.append(f"{idx}. [{step['status']}] {step['id']} {step['agent']} | {step['subtask']}{marker}")
    return "\n".join(lines)


def _format_shared_memory(state: EpisodeState) -> str:
    sections: list[str] = []
    if state["artifacts"]:
        sections.append("Artifacts:")
        for handle, rec in sorted(state["artifacts"].items()):
            sections.append(render_artifact_card(handle, rec))
    layer_lines = render_layer_cards(state)
    if layer_lines:
        sections.append("Layers:")
        sections.extend(layer_lines)
    # ORC routes contracts; it does not need rich worker payload blobs. Keep a
    # bounded status/evidence view and expose terminal readiness separately.
    field_lines = render_field_cards(state, max_fields=14, max_chars=320)
    if field_lines:
        sections.append("Fields:")
        sections.extend(field_lines)
    answer_lines = answer_registry_cards(state)
    sections.append("Available answer references:")
    sections.extend(answer_lines or ["  (none; continue gathering evidence)"])
    if not sections:
        return "  (empty)"
    return "\n".join(sections)


def _format_field_value(value: Any, max_chars: int = 360) -> str:
    try:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            text = repr(value)
    except Exception:
        text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _format_last_observation(state: EpisodeState, current_step_id: str | None) -> str:
    if current_step_id is None:
        return "  (none)"
    for event in reversed(state["history"]):
        if event["step_id"] != current_step_id:
            continue
        if event["role"] == "tool":
            payload = event["payload"]
            ok = payload.get("ok")
            tool = payload.get("tool", "?")
            args = payload.get("args", {})
            text = payload.get("text", "")
            return (
                f"Tool: {tool}\n"
                f"Status: {'OK' if ok else 'ERROR'}\n"
                f"Args: {args}\n"
                f"Result: {text}"
            )
    return "  (none)"


def _format_prior_artifacts_for_step(state: EpisodeState, step: StepRecord) -> str:
    """Build PRIOR ARTIFACTS for a specialist contract.

    Primary filter: ``step.needs`` declared by ORC at plan time. Tokens supported:
    ``gpkg_X`` / ``img_X`` / ``tif_X`` (artifact handles), ``layer <name>``,
    ``field <name>``. If `needs` is empty OR none of its tokens resolve, fall
    back to a small keyword filter so the specialist still sees relevant
    evidence that already exists.
    """
    needs = step.get("needs") or []
    chosen_artifacts: dict[str, dict[str, Any]] = {}
    chosen_layers: dict[str, list[dict[str, Any]]] = {}
    chosen_fields: list[str] = []

    needs_resolved = 0
    if needs:
        for token in needs:
            tok = token.strip()
            if tok.startswith("layer "):
                name = tok[6:].strip()
                for gpkg, layers in state["layers"].items():
                    for lay in layers:
                        if lay["name"] == name:
                            chosen_layers.setdefault(gpkg, []).append(lay)
                            if gpkg in state["artifacts"]:
                                chosen_artifacts[gpkg] = state["artifacts"][gpkg]
                            needs_resolved += 1
            elif tok.startswith("field "):
                fname = tok[6:].strip()
                if fname in state["fields"]:
                    chosen_fields.append(fname)
                    needs_resolved += 1
            elif tok in state["artifacts"]:
                chosen_artifacts[tok] = state["artifacts"][tok]
                if tok in state["layers"]:
                    chosen_layers.setdefault(tok, []).extend(state["layers"][tok])
                needs_resolved += 1
            elif tok in state["layers"]:
                chosen_artifacts[tok] = state["artifacts"].get(tok, {"kind": "gpkg", "path": "?", "note": ""})
                chosen_layers.setdefault(tok, []).extend(state["layers"][tok])
                needs_resolved += 1
            elif _GEODISASTER_PROFILE_ACTIVE and _HANDLE_TOKEN_RE.match(tok):
                matched_fields = []
                for fname, fval in state["fields"].items():
                    if fname == "latest_answer_ref" or fname.endswith(
                        ("_payload", "_payload_keys", "_history", "_answer_ref", "_final_answer")
                    ):
                        continue
                    value = fval.get("value") if isinstance(fval, dict) else None
                    if fname.startswith(f"{tok}_") or value == tok:
                        matched_fields.append(fname)
                if matched_fields:
                    for fname in matched_fields:
                        if fname not in chosen_fields:
                            chosen_fields.append(fname)
                    needs_resolved += 1

    text = step["subtask"].lower()
    agent = step["agent"]

    if not needs or needs_resolved == 0:
        if any(k in text for k in ("distance", "closest", "nearest", "farthest", "furthest", "average", "travel")):
            for f in (
                "pair",
                "distance_m",
                "distance_rows",
                "distances_m",
                "distance_pair_count",
                "min_pair",
                "min_distance_m",
                "max_pair",
                "max_distance_m",
                "avg_distance_m",
                "travel_time_s",
                "travel_times_s",
                "travel_time_min",
                "result_layer",
            ):
                if f in state["fields"]:
                    chosen_fields.append(f)
            for f in ("bboxes", "count"):
                if f in state["fields"]:
                    chosen_fields.append(f)
        if any(k in text for k in ("area", "pixel", "coverage", "square meter", "sq m")):
            for f in ("pixel_count", "pixel_counts", "bboxes", "count"):
                if f in state["fields"]:
                    chosen_fields.append(f)
        if any(k in text for k in ("ocr", "read text", "sign")):
            if "ocr_text" in state["fields"]:
                chosen_fields.append("ocr_text")
        if any(k in text for k in ("count", "detect", "bbox", "how many")):
            for f in ("count", "bboxes"):
                if f in state["fields"]:
                    chosen_fields.append(f)
        for handle, rec in state["artifacts"].items():
            if agent == "GA" and rec.get("kind") == "gpkg":
                chosen_artifacts[handle] = rec
                if handle in state["layers"]:
                    chosen_layers.setdefault(handle, []).extend(state["layers"][handle])
            if agent == "GA" and rec.get("kind") == "tif" and any(k in text for k in ("geotiff", "tif", "bbox", "overlay")):
                chosen_artifacts[handle] = rec
            if agent == "VRA" and rec.get("kind") in ("img", "tif"):
                chosen_artifacts[handle] = rec
            if agent == "PA" and rec.get("kind") in ("img", "tif") and any(k in text for k in ("pixel", "bbox", "distance", "meter", "image")):
                chosen_artifacts[handle] = rec
            if agent == "PA" and rec.get("kind") == "gpkg" and any(k in text for k in ("layer", "change", "index", "ndbi", "ndvi", "nbr", "filter", "route", "map", "visualize")):
                chosen_artifacts[handle] = rec
                if handle in state["layers"]:
                    chosen_layers.setdefault(handle, []).extend(state["layers"][handle])

    # Even when `needs` resolved, enrich the specialist context with highly
    # relevant numeric evidence so PA does not hallucinate a gpkg workflow for
    # image-native geometry tasks.
    if any(k in text for k in ("distance", "pixel", "bbox", "meter", "image")):
        for f in ("bboxes", "count", "pixel_count", "pixel_counts"):
            if f in state["fields"] and f not in chosen_fields:
                chosen_fields.append(f)
    if agent == "PA" and (
        "distance_m" in chosen_fields
        or "pair" in chosen_fields
        or any(str(token).strip() in {"field distance_m", "field pair"} for token in needs)
    ):
        for f in (
            "distance_rows",
            "distances_m",
            "distance_pair_count",
            "min_pair",
            "min_distance_m",
            "max_pair",
            "max_distance_m",
            "avg_distance_m",
            "travel_times_s",
        ):
            if f in state["fields"] and f not in chosen_fields:
                chosen_fields.append(f)
    if agent == "PA" and any(k in text for k in ("layer", "change", "index", "ndbi", "ndvi", "nbr", "filter", "route", "map", "visualize")):
        for handle, rec in state["artifacts"].items():
            if rec.get("kind") == "gpkg":
                chosen_artifacts[handle] = rec
                if handle in state["layers"]:
                    chosen_layers.setdefault(handle, []).extend(state["layers"][handle])

    sections: list[str] = []
    if chosen_artifacts:
        sections.append("Artifacts:")
        for handle, rec in sorted(chosen_artifacts.items()):
            sections.append(render_artifact_card(handle, rec))
    if chosen_layers:
        sections.append("Layers:")
        for gpkg, layers in sorted(chosen_layers.items()):
            items = ", ".join(f"{lay['name']} ({lay['produced_by_step']})" for lay in layers)
            sections.append(f"  {gpkg}: {items}")
    if chosen_fields:
        sections.append("Fields:")
        for fname in chosen_fields:
            val = state["fields"][fname]
            sections.append(f"  {fname} = {_format_field_value(val['value'])}  ({val['produced_by_step']})")
    if not sections:
        return "  (none yet)"
    return "\n".join(sections)


def _format_state_delta(state: EpisodeState, step_id: str, since_history_index: int) -> str:
    new_artifacts: list[tuple[str, dict[str, Any]]] = []
    new_layers: list[tuple[str, dict[str, Any]]] = []
    new_fields: list[str] = []
    for event in state["history"][since_history_index:]:
        if event["step_id"] != step_id or event["role"] != "tool":
            continue
        if not event["payload"].get("ok"):
            continue
        produced = event["payload"].get("produced", {})
        for art in produced.get("artifacts", []):
            new_artifacts.append((art["handle"], art))
        for lay in produced.get("layers", []):
            new_layers.append((lay["gpkg"], lay))
        for fname in produced.get("fields", {}):
            if fname == "latest_answer_ref" or fname.endswith(
                ("_payload", "_payload_keys", "_history", "_answer_ref", "_final_answer")
            ):
                continue
            new_fields.append(fname)

    sections: list[str] = []
    if new_artifacts:
        sections.append("Artifacts:")
        for handle, art in new_artifacts:
            sections.append(f"  + {handle} [{art.get('kind', '?')}]")
    if new_layers:
        sections.append("Layers:")
        for gpkg, lay in new_layers:
            sections.append(f"  + layer {lay['name']} in {gpkg}")
    if new_fields:
        sections.append("Fields:")
        for fname in new_fields:
            val = state["fields"].get(fname)
            if val is not None:
                sections.append(f"  + {fname} = {_format_field_value(val['value'])}")
    if not sections:
        return "  (no change)"
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_orc_plan_message(state: EpisodeState) -> str:
    if state["artifacts"]:
        artifact_lines = [render_artifact_card(handle, rec) for handle, rec in sorted(state["artifacts"].items())]
    else:
        artifact_lines = ["  (none)"]
    if _GEODISASTER_PROFILE_ACTIVE:
        specialist_lines = [
            "- GA  : open declared input handles, prepare scene/work contexts, measure/extract/map/render geospatial evidence.",
            "- PA  : statistics, source comparison, temporal change, classification, hazard scoring, exposure, ranking, routes.",
            "- VRA : visual, damage, flood, and scene interpretation over prepared evidence.",
        ]
        needs_examples = "`question_1`, `forest_loss_history_1`, `hazard_stack_1`, `damage_scene_1`, `scene_1`, `work_1`, `field peak_value`"
    else:
        specialist_lines = [
            "- GA  : place boundaries, OSM POI layers, distance over GeoPackages, map renders.",
            "- PA  : arithmetic, filtering, plots, final synthesis.",
            "- VRA : image / GeoTIFF reasoning, OCR, segmentation, raster indices.",
        ]
        needs_examples = "`tif_1`, `image_1`, `gpkg_1`, `layer fire_stations`, `field pair`"
    parts = [
        "QUESTION",
        state["question"],
        "",
        "IMAGES",
        "  " + (", ".join(state["images"]) if state["images"] else "(none)"),
        "",
        "REGISTERED INPUT ARTIFACTS",
        *artifact_lines,
        "",
        "AVAILABLE SPECIALISTS",
        *specialist_lines,
        "",
        "RESPONSE RULE",
        f"Reply with one JSON object matching the Plan schema. Embed all relevant numbers and place names directly inside each step's `subtask`. Use exact tokens for `needs` (e.g. {needs_examples}). No prose, no markdown fences.",
    ]
    return "\n".join(parts)


def build_orc_decide_message(state: EpisodeState, last_report: dict[str, Any] | None) -> str:
    pending = next((s for s in state["steps"] if s["status"] == "pending"), None)
    next_hint = (
        f"Next pending step: {pending['id']} ({pending['agent']}) — {pending['subtask']}"
        if pending is not None
        else "All plan steps are complete or failed. Consider Terminate."
    )

    last_report_block = "  (none yet)"
    if last_report is not None:
        last_report_block = "\n".join(
            [
                f"  agent: {last_report.get('agent', '?')}",
                f"  step_id: {last_report.get('step_id', '?')}",
                f"  status: {last_report.get('status', '?')}",
                f"  summary: {last_report.get('summary', '(none)')}",
            ]
        )

    legal_moves = "Legal moves: CallAgent, Terminate"
    if last_report is not None and last_report.get("status") == "CONTRACT_FAILURE" and state["replans_left"] > 0:
        legal_moves = f"Legal moves: CallAgent, Terminate, Plan (replan; replans_left={state['replans_left']})"

    parts = [
        "QUESTION",
        state["question"],
        "",
        "PLAN",
        _format_plan_block(state, current_step_id=None),
        "",
        "SHARED MEMORY",
        _format_shared_memory(state),
        "",
        "LAST AGENT REPORT",
        last_report_block,
        "",
        "HINT",
        f"  {next_hint}",
        "",
        "RESPONSE RULE",
        "Reply with one JSON object: CallAgent, Terminate, or Plan (only after CONTRACT_FAILURE). Terminate must use `answer_ref` with the exact published reference.",
        legal_moves,
        "No prose, no markdown fences.",
    ]
    return "\n".join(parts)


def build_specialist_message(
    state: EpisodeState,
    step: StepRecord,
    turn_no: int,
    contract_window_index: int,
) -> str:
    tool_names = AGENT_TOOLS.get(step["agent"], []) + ["TaskComplete"]
    allowed_lines = [f"  - {_TOOL_SIGNATURES[name]}" for name in tool_names if name in _TOOL_SIGNATURES]

    if turn_no == 1:
        evidence_block_title = "PRIOR ARTIFACTS (filtered to your needs)"
        evidence_block_body = _format_prior_artifacts_for_step(state, step)
    else:
        evidence_block_title = "Δ STATE SINCE LAST TURN"
        evidence_block_body = _format_state_delta(state, step["id"], contract_window_index)
        # re-anchor every 4 turns
        if turn_no % 4 == 1 and turn_no > 1:
            evidence_block_title = "PRIOR ARTIFACTS (re-anchored)"
            evidence_block_body = _format_prior_artifacts_for_step(state, step)

    parts = [
        "YOUR SUBTASK",
        step["subtask"],
        "",
        "PLAN (read-only context — do not act on other steps)",
        _format_plan_block(state, current_step_id=step["id"]),
        "",
        evidence_block_title,
        evidence_block_body,
        "",
        "LAST OBSERVATION",
        _format_last_observation(state, step["id"]),
        "",
        "ALLOWED TOOLS (use these exact arg names — do not invent new keys)",
        *allowed_lines,
        "",
        "RESPONSE RULE",
        "Reply with one JSON object matching SpecialistTurn ({thought, action}). No prose, no markdown fences.",
    ]
    return "\n".join(parts)
