from __future__ import annotations

import json
import os
from typing import Any

from multi_agent.final_dataset_profile import FINAL_DATASET_TOOL_SIGNATURES


MAS_TOOL_SIGNATURES: dict[str, str] = {
    "GetAreaBoundary": "GetAreaBoundary(area: str, buffer_m?: int)  # create an AOI GeoPackage and area_boundary layer",
    "AddPoisLayer": "AddPoisLayer(gpkg: str, query: dict[str,str], layer_name: str)  # add one OSM-tag POI layer to a GeoPackage",
    "ComputeDistance": "ComputeDistance(gpkg: str, src_layer: str, tar_layer: str, top?: int)  # pairwise distances and route-time fields",
    "DisplayOnMap": "DisplayOnMap(gpkg: str, layers: list[str])  # render vector layers on a map image",
    "GetBboxFromGeotiff": "GetBboxFromGeotiff(geotiff: str)  # read GeoTIFF extent",
    "DisplayOnGeotiff": "DisplayOnGeotiff(gpkg: str, layers: list[str], geotiff: str)  # overlay layers on GeoTIFF",
    "Calculator": "Calculator(expression: str)  # Python expression with literal numbers only",
    "Solver": "Solver(command: str)  # small symbolic/numerical solver snippet",
    "Plot": "Plot(command: str)  # matplotlib snippet returning a plot artifact",
    "GoogleSearch": "GoogleSearch(query: str, k?: int)  # external lookup; cite result in answer",
    "ObjectDetection": "ObjectDetection(image: str)  # generic object detections and boxes",
    "TextToBbox": "TextToBbox(image: str, text: str, top1?: bool)  # phrase-grounded detections",
    "CountGivenObject": "CountGivenObject(image: str, text: str)  # count a named object class",
    "ImageDescription": "ImageDescription(image: str)  # free-form scene description",
    "RegionAttributeDescription": "RegionAttributeDescription(image: str, attribute: str, bbox?: str)  # attribute description for a region/image",
    "ChangeDetection": "ChangeDetection(text: str, pre_image: str, post_image: str)  # visual before/after change",
    "SegmentObjectPixels": "SegmentObjectPixels(image: str, text: str, flag?: bool)  # segmentation pixel count",
    "OCR": "OCR(image: str)  # read text from image",
    "DrawBox": "DrawBox(image: str, bbox: str, annotation: str)  # annotate image with a box",
    "AddText": "AddText(image: str, text: str, position: str)  # add text to image",
    "AddIndexLayer": "AddIndexLayer(gpkg: str, index_type: str, layer_name: str, year?: int, month?: int)  # add NDVI/NDBI/NBR-like raster index layer",
    "ComputeIndexChange": "ComputeIndexChange(gpkg: str, index_type: str, layer1_name: str, layer2_name: str, diff_layer_name: str)  # raster index delta",
    "ShowIndexLayer": "ShowIndexLayer(gpkg: str, index_type: str, layer_name: str)  # render a raster index layer",
}

_GEODISASTER_PROFILE_ACTIVE = os.environ.get("MAS_AGENT_PROFILE", "").strip().lower() in {
    "geodisaster",
    "geo_disaster",
    "geo-disaster",
}

ALL_TOOL_SIGNATURES: dict[str, str] = (
    dict(FINAL_DATASET_TOOL_SIGNATURES)
    if _GEODISASTER_PROFILE_ACTIVE
    else dict(MAS_TOOL_SIGNATURES)
)


def render_tool_signatures() -> str:
    lines = []
    for name in sorted(ALL_TOOL_SIGNATURES):
        lines.append(f"- {ALL_TOOL_SIGNATURES[name]}")
    return "\n".join(lines)


SINGLE_AGENT_SYSTEM = f"""You are a single geospatial-disaster tool-use agent.
You do your own planning and tool execution. There are no specialist agents and
no orchestrator. You receive one question plus visible handles, then solve it
by emitting one JSON object per turn.

Response schema:
{{
  "thought": "<compact reasoning for the next action>",
  "action": <Plan | ToolCall | FinalAnswer>
}}

Actions:
- Plan = {{"tool":"Plan","goal":"...","steps":[{{"id":"s1","objective":"...","needs":["..."]}}]}}
- ToolCall = {{"tool":"<tool name>","args":{{...}}}}
- FinalAnswer = {{"tool":"FinalAnswer","answer_ref":"ANSWER_JSON:answer_1"}}

Core rules:
- On the first turn, produce a high-level Plan unless the answer is already
  visible. The plan should explain evidence objectives, not list every tool by
  rote.
- After a Plan, execute one tool call per turn. Use LAST OBSERVATION and SHARED
  MEMORY to decide the next action.
- Use exact tool names and exact argument names from ALLOWED TOOLS. Do not
  invent keys or handles.
- Use visible canonical handles, not raw filesystem paths. For GeoDisaster
  questions, `question` is normally `"question_1"`.
- Raw GeoDisaster handles such as `optical_stack_1`, `damage_scene_1`,
  `sar_chip_1`, `hazard_stack_1`, and `route_network_1` are opened with
  ReadData. Later GeoDisaster tools consume produced handles such as `scene_N`,
  `work_N`, `stats_N`, `compare_N`, `rank_N`, `route_N`, or `map_N`.
- When ReadData returns `latest_handle = scene_N`, use that produced scene
  handle in later PrepareScene calls. Do not pass the raw input handle where a
  work/scene handle is required.
- For PrepareScene, choose semantic slot names from the evidence: forest
  optical/sar/forest/loss_history; hazard hazard/sources/exposure; damage
  image/labels/masks; flood sar/water; routing routing; comparison scene_a and
  scene_b.
- Stop when the evidence needed for the answer is present. Do not re-run a
  successful tool call with the same arguments.
- Finish only when AVAILABLE ANSWER REFERENCES publishes
  `ANSWER_JSON:answer_1`, then copy it into `answer_ref`. The runtime resolves
  it only from a schema-validated projection of worker evidence. Inline answer
  JSON, prose, partial fields, worker handles, and invented references are
  rejected.
- Reply with one JSON object only. No prose, no markdown fences.

ALLOWED TOOLS
{render_tool_signatures()}
"""


def compact_json(value: Any, max_chars: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def compact_artifact_line(handle: str, card: dict[str, Any]) -> str:
    kind = card.get("kind") or card.get("type") or "artifact"
    note = card.get("note") or card.get("description") or ""
    refs: list[str] = []
    paths = card.get("paths") if isinstance(card.get("paths"), dict) else {}
    for entries in paths.values():
        if isinstance(entries, list):
            for entry in entries[:2]:
                if isinstance(entry, dict) and entry.get("label"):
                    refs.append(str(entry["label"]))
    if card.get("question_path", {}).get("label"):
        refs.append(str(card["question_path"]["label"]))
    ref_text = f"; refs: {', '.join(refs[:4])}" if refs else ""
    suffix = f" - {note}{ref_text}" if note or ref_text else ""
    return f"- {handle} [{kind}]{suffix}"


def record_input_lines(record: dict[str, Any]) -> list[str]:
    artifacts = record.get("initial_artifacts") if isinstance(record.get("initial_artifacts"), dict) else {}
    if artifacts:
        return [compact_artifact_line(handle, card) for handle, card in sorted(artifacts.items())]
    images = record.get("images") or []
    lines = [f"- image_{i} [img] - {path}" for i, path in enumerate(images, start=1)]
    return lines or ["- (none)"]


def record_question_text(record: dict[str, Any]) -> str:
    question = record.get("question")
    if isinstance(question, dict):
        return str(question.get("text") or record.get("raw_question") or "").strip()
    # Derived multi-agent rows store a display context (question + handles +
    # schema) in ``question``. The single-agent prompt renders handles itself,
    # and terminal JSON is runtime-owned, so use the plain question here.
    return str(record.get("raw_question") or question or "").strip()


def build_initial_user_message(record: dict[str, Any]) -> str:
    question = record_question_text(record)
    parts = [
        "QUESTION",
        question,
        "",
        "REGISTERED INPUTS",
        *record_input_lines(record),
    ]
    parts.extend([
        "", "TERMINAL CONTRACT",
        "The runtime publishes ANSWER_JSON:answer_1 only after worker evidence satisfies the hidden declared answer schema.",
        "", "STATE",
        "No tools have been called yet.",
        "",
        "INSTRUCTION",
        "Produce a Plan action first, unless the answer is already fully visible. Then use tools one at a time.",
    ])
    return "\n".join(parts)


def build_runtime_user_message(
    record: dict[str, Any],
    *,
    plan: dict[str, Any] | None,
    memory_text: str,
    last_observation: dict[str, Any] | None,
    turn_no: int,
) -> str:
    obs = compact_json(last_observation or {"status": "none"}, max_chars=1000)
    plan_text = compact_json(plan or {}, max_chars=1200) if plan else "(no plan yet)"
    parts = [
        "QUESTION",
        record_question_text(record),
        "",
    ]
    parts.extend([
        "", "TERMINAL CONTRACT",
        "Use FinalAnswer only with a reference currently listed under Available answer references.",
        "", "CURRENT PLAN",
        plan_text,
        "",
        "SHARED MEMORY",
        memory_text or "(empty)",
        "",
        "LAST OBSERVATION",
        obs,
        "",
        "TURN",
        str(turn_no),
        "",
        "INSTRUCTION",
        "Choose the next justified action: call one tool, or FinalAnswer if the evidence is complete.",
    ])
    return "\n".join(parts)
