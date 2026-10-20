#!/usr/bin/env python3
"""Convert GeoDisaster pure toolchains into multi-agent teacher traces.

Outputs:
- multi-agent train/test JSONL traces
- role-wise SFT JSONL files compatible with the existing role adapter trainer
- tool ownership/context docs for the GeoDisaster dataset profile

The source GT chain is used only to create teacher supervision. Tool workers
remain independent and were already replayed separately.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT.parents[2]
CORRECTION_RUNS_ROOT = (DATASET_ROOT / "experimental" / "correction_runs").resolve()
V3_ROOT = ROOT / "mas" / "testing" / "v3"
DEFAULT_SPLIT_DIR = ROOT / "final_dataset_gt_replay" / "splits"
DEFAULT_SIM_JSONL = ROOT / "final_dataset_gt_replay" / "full_toolchain_simulation_2921.jsonl"
DEFAULT_OUT = ROOT / "final_dataset_gt_replay" / "multiagentic"

os.environ.setdefault("MAS_AGENT_PROFILE", "geodisaster")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_agent.final_dataset_profile import (  # noqa: E402
    FINAL_DATASET_AGENT_TOOLS,
    FINAL_DATASET_TOOL_SIGNATURES,
    TOOL_TO_AGENT,
)
from multi_agent.geodisaster_observation import (  # noqa: E402
    make_geodisaster_observation,
    make_geodisaster_state_fields,
    summarize_geodisaster_tool_result,
)
from multi_agent.prompts import (  # noqa: E402
    SYSTEM_PROMPTS,
    build_orc_decide_message,
    build_orc_plan_message,
    build_specialist_message,
)
from multi_agent.schemas import Plan  # noqa: E402
from multi_agent.answer_contract import project_from_state, validate_explicit_answer  # noqa: E402
from multi_agent.answer_registry import (  # noqa: E402
    ANSWER_REF,
    answer_sha256,
    register_answer,
    resolve_answer_ref,
)
from evaluation.structured_answer import score_answer  # noqa: E402
from multi_agent.state import (  # noqa: E402
    append_event,
    find_step,
    install_plan,
    mark_step,
    new_episode,
    open_contract_window,
    set_field,
)


ROLE_ORDER = ["ORC", "GA", "PA", "VRA"]
HANDLE_RE = re.compile(r"^(question|scene|work|compare|change|class|stats|area|hazard|exposure|damage|rank|objects|flood|routes|map)_\d+$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compact(value: Any, max_chars: int = 520) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return value
    return {"_truncated": True, "_chars": len(text), "_preview": text[:max_chars]}


def collect_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            out.extend(collect_strings(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(collect_strings(child))
    return out


def portable_teacher_value(value: Any) -> Any:
    """Remove host-specific paths from public teacher evidence.

    Packaged source paths remain repository-relative. Replay artifacts are not
    shipped with the thin release, so they are represented by stable logical
    locators rather than build-host filesystem paths.
    """

    if isinstance(value, dict):
        return {key: portable_teacher_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [portable_teacher_value(child) for child in value]
    if not isinstance(value, str) or not os.path.isabs(value):
        return value

    path = Path(value)
    # Public source paths are repository-relative regardless of which build
    # host or parent directory produced the replay.
    if "source_assets" in path.parts:
        index = path.parts.index("source_assets")
        return Path(*path.parts[index:]).as_posix()
    try:
        return str(path.relative_to(DATASET_ROOT))
    except ValueError:
        pass
    try:
        relative = path.relative_to(CORRECTION_RUNS_ROOT)
        return f"generated-artifact://{relative.as_posix()}"
    except ValueError:
        return f"external-artifact://{path.name}"


def first_path(card: dict[str, Any]) -> str:
    def walk(obj: Any) -> str:
        if isinstance(obj, dict):
            if obj.get("path"):
                return str(obj["path"])
            for child in obj.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(obj, list):
            for child in obj:
                found = walk(child)
                if found:
                    return found
        return ""

    return walk(card.get("paths", card))


def artifact_note(handle: str, card: dict[str, Any]) -> str:
    desc = str(card.get("description") or card.get("type") or "input artifact")
    path_labels: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("label"):
                path_labels.append(str(obj["label"]))
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(card.get("paths", card))
    labels = ", ".join(path_labels[:4])
    suffix = f"; refs: {labels}" if labels else ""
    return f"{handle}: {desc}{suffix}"


def question_text(record: dict[str, Any]) -> str:
    q = record.get("question")
    if isinstance(q, dict):
        return str(q.get("text") or "")
    return str(q or "")


def build_question_context(record: dict[str, Any]) -> str:
    lines = [
        question_text(record),
        "",
        "AVAILABLE INPUT HANDLES",
        "Use these exact handles in tool arguments. The tools resolve true local paths from the handle cards.",
    ]
    for handle in record.get("available_initial_handles") or sorted((record.get("initial_artifacts") or {}).keys()):
        card = (record.get("initial_artifacts") or {}).get(handle) or {}
        lines.append(f"- {artifact_note(handle, card)}")
    answer_schema = ((record.get("question") or {}).get("answer_schema") if isinstance(record.get("question"), dict) else None)
    if answer_schema:
        lines.extend(["", "ANSWER SCHEMA", json.dumps(answer_schema, ensure_ascii=False)])
    return "\n".join(lines)


def load_sim_by_id(path: Path) -> dict[str, dict[str, Any]]:
    sims = read_jsonl(path)
    return {str(row["dataset_id"]): row for row in sims}


def group_chain(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for step in chain:
        tool = step.get("tool")
        if tool == "TaskComplete":
            continue
        agent = TOOL_TO_AGENT[str(tool)]
        if groups and groups[-1]["agent"] == agent:
            groups[-1]["calls"].append(step)
        else:
            groups.append({"agent": agent, "calls": [step]})
    return groups


def handles_in_calls(calls: list[dict[str, Any]], visible_handles: set[str]) -> list[str]:
    handles: list[str] = []
    for call in calls:
        for text in collect_strings(call.get("args") or {}):
            if text in visible_handles:
                handles.append(text)
    return list(dict.fromkeys(handles))


def _short_goal(record: dict[str, Any], max_chars: int = 220) -> str:
    text = " ".join(question_text(record).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _read_inputs(group: dict[str, Any]) -> list[str]:
    handles: list[str] = []
    for call in group["calls"]:
        if call.get("tool") != "ReadData":
            continue
        args = call.get("args") or {}
        if args.get("input"):
            handles.append(str(args["input"]))
    return list(dict.fromkeys(handles))


def _first_non_question_handles(group: dict[str, Any]) -> list[str]:
    handles: list[str] = []
    for call in group["calls"]:
        for text in collect_strings(call.get("args") or {}):
            if text != "question_1" and (text.endswith("_1") or HANDLE_RE.match(text)):
                handles.append(text)
    return list(dict.fromkeys(handles))


def _domain_hint(record: dict[str, Any]) -> str:
    source = str(record.get("source") or "")
    qtype = str(record.get("question_type") or "")
    if source == "data/multi_hazard":
        domain = (
            qtype.split(":", 1)[0]
            if ":" in qtype
            else str(record.get("dataset_id") or "").split("/")[1]
        ).replace("_", " ")
        variant = qtype.split(":", 1)[-1]
        variant_map = {
            "A": "primary scalar or area statistic",
            "B": "dominant severity or class decision",
            "C": "cross-source agreement decision",
            "D": "exposure, nearest-asset, or population impact decision",
        }
        return f"multi-hazard {domain} / {variant_map.get(variant, 'hazard analysis')}"
    if source == "deforestation_data":
        return f"deterministic deforestation / {qtype.replace('_', ' ')}"
    if source == "geodisaster":
        return f"xBD building-damage / {qtype}"
    if "routing" in source:
        return f"flood-aware routing / {qtype.replace('_', ' ')}"
    if "sen1" in source:
        return f"Sentinel-1 SAR flood / diagnostic {qtype}"
    return qtype.replace("_", " ")


def _contract_prefix(record: dict[str, Any], group: dict[str, Any]) -> tuple[str, str]:
    tools = [str(call["tool"]) for call in group["calls"]]
    read_inputs = _read_inputs(group)
    handles = ", ".join(read_inputs or _first_non_question_handles(group)[:4]) or "the visible prior handles"
    domain = _domain_hint(record)
    role = str(group["agent"])
    return domain, f"{role} contract for {domain}:"


def subtask_for_group(record: dict[str, Any], group: dict[str, Any], step_idx: int) -> str:
    """Create a rich outcome contract without revealing the exact tool chain.

    ORC should teach decomposition: what evidence/result is needed, which
    specialist owns it, and what a successful handoff looks like. The precise
    tool sequence remains only inside the specialist demonstration turns.
    """

    tools = [str(call["tool"]) for call in group["calls"]]
    read_inputs = _read_inputs(group)
    handles = ", ".join(read_inputs or _first_non_question_handles(group)[:4]) or "the visible prior handles"
    domain, prefix = _contract_prefix(record, group)

    if tools[0] == "ReadData":
        if "MapFlood" in tools:
            return (
                f"{prefix} prepare SAR flood evidence from {handles}. Verify the declared sources, preserve "
                "spatial/grid context, and expose flood extent plus component/area support for later description "
                "or ranking."
            )
        if "ExtractObjects" in tools and "MeasureArea" in tools:
            return (
                f"{prefix} prepare forest-change evidence from {handles}. Verify the local stacks/masks, isolate "
                "the relevant disturbance object, and report measured area support with a reusable work handle."
            )
        if "ExtractObjects" in tools:
            return (
                f"{prefix} prepare object-level spatial evidence from {handles}. Verify the image/label references "
                "and expose the requested objects or components for downstream semantic assessment."
            )
        if "PrepareScene" in tools:
            return (
                f"{prefix} prepare reusable evidence from {handles}. Preserve units, temporal/source metadata, "
                "masks, and schema cues so later reasoning has a compact grounded work context."
            )
        return (
            f"{prefix} verify {handles} and expose a scene handle that later specialists can use."
        )

    if "PlanRoutes" in tools or "CompareRoutes" in tools:
        return (
            f"{prefix} make the requested routing decision from the prepared route/flood context. Compare "
            "reachability, time, safety, risk, and blocked-edge support as needed."
        )
    if "EstimateExposure" in tools:
        return (
            f"{prefix} estimate the requested exposure outcome from prepared hazard and asset evidence. Report "
            "the relevant counts, distances, population/asset support, and hazard class."
        )
    if "CompareSources" in tools and "ClassifyChange" in tools:
        return (
            f"{prefix} compare the prepared sources and classify the state from their combined numeric support. "
            "Report the label and the evidence values behind it."
        )
    if "CompareSources" in tools:
        return (
            f"{prefix} compare the prepared sources or references for the requested phenomenon. Report agreement "
            "or disagreement with the key support metrics."
        )
    if "DetectChange" in tools and "ClassifyChange" in tools:
        return (
            f"{prefix} identify the temporal change signal and convert it into the requested state/window label. "
            "Report the interval or label with numeric support."
        )
    if "ClassifyChange" in tools:
        return (
            f"{prefix} classify the requested state from prepared evidence using the stated labels/rules. Report "
            "the class with support values."
        )
    if "ScoreHazard" in tools:
        return (
            f"{prefix} score the requested hazard/severity level from prepared evidence. Report the class and "
            "its supporting fraction or value."
        )
    if "RankRegions" in tools:
        return (
            f"{prefix} rank the requested regions, cells, components, or candidates. Report the top item(s) "
            "with score/count/area support."
        )
    if "ComputeStats" in tools:
        return (
            f"{prefix} compute the requested statistic from prepared evidence. Report the value, units, and "
            "support fields needed by ORC."
        )

    if "AssessDamage" in tools and "CompareScenes" in tools:
        return (
            f"{prefix} interpret both prepared damage scenes and compare severe-damage burden. Report the worse "
            "scene with count/area/ratio support."
        )
    if "AssessDamage" in tools:
        return (
            f"{prefix} interpret prepared damage evidence. Report the requested counts, severity distribution, "
            "severe-area support, grid evidence, or quadrant evidence."
        )
    if "DescribeFlood" in tools:
        return (
            f"{prefix} describe prepared flood evidence in the requested terms: area/percentage, components, "
            "grid concentration, or known-water split."
        )
    if "CompareScenes" in tools:
        return (
            f"{prefix} compare prepared scene summaries and report the requested scene-level judgement with its "
            "support metric."
        )
    if "RenderMap" in tools:
        return (
            f"{prefix} render the requested map or overlay artifact from prepared evidence. Report the artifact "
            "handle/path and layer summary."
        )
    return (
        f"{prefix} produce the next grounded evidence item required by the current plan step. Report the "
        "produced handle/value and stop once the contract is satisfied."
    )


def trace_outputs_by_step(sim: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in sim.get("trace") or []:
        if row.get("tool") == "TaskComplete":
            continue
        out[int(row["step"])] = row
    return out


def tool_output_from_trace(trace_row: dict[str, Any]) -> dict[str, Any]:
    tool = str(trace_row.get("tool") or "")
    handle = trace_row.get("handle")
    ok = trace_row.get("error_code") == 0
    portable_result = portable_teacher_value(trace_row.get("result"))
    summary = summarize_geodisaster_tool_result(
        tool,
        handle=str(handle) if handle else None,
        text=str(trace_row.get("text") or "OK"),
        result=portable_result,
        ok=ok,
    )
    fields = make_geodisaster_state_fields(
        tool,
        handle=str(handle) if handle else None,
        ok=ok,
        summary=summary,
        full_result=portable_result,
    )
    text = summary.get("summary") or str(trace_row.get("text") or "OK")
    return {
        "ok": ok,
        "text": text,
        "produced": {"artifacts": [], "layers": [], "fields": fields},
        "observation": make_geodisaster_observation(
            tool,
            handle=str(handle) if handle else None,
            ok=ok,
            summary=summary,
        ),
    }


def normalize_chain_for_answer(record: dict[str, Any], chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an independent copy of the canonical chain without mutation."""
    return [copy.deepcopy(step) for step in chain]


def agent_facing_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical agent arguments without hidden schema rewrites."""
    return dict(args or {})


def agent_thought_for_call(call: dict[str, Any], group: dict[str, Any], call_index: int, call_count: int) -> str:
    tool = str(call.get("tool") or "")
    args = call.get("args") or {}
    if tool == "ReadData":
        return f"Need a verified scene handle for `{args.get('input')}` before any downstream evidence step."
    if tool == "PrepareScene":
        return (
            f"Opened scenes are ready; combine {json.dumps(args.get('inputs') or {}, ensure_ascii=False)} into "
            f"a `{args.get('mode', 'prepared')}` work context with units, masks, and metadata preserved."
        )
    if tool == "ComputeStats":
        return (
            f"The prepared work context contains the needed evidence; compute metrics "
            f"{json.dumps(args.get('metrics'), ensure_ascii=False)}."
        )
    if tool == "MeasureArea":
        return f"The target `{args.get('target')}` is available in the work context; measure area in `{args.get('unit')}`."
    if tool == "CompareSources":
        return (
            f"Prepared sources must be checked against each other; compare "
            f"{json.dumps(args.get('compare'), ensure_ascii=False)} and keep the support metrics."
        )
    if tool == "DetectChange":
        return f"Change evidence is required; detect `{args.get('change')}` over `{args.get('window', 'the available window')}`."
    if tool == "ClassifyChange":
        return (
            f"Evidence is ready; apply rule `{args.get('rule')}` to the allowed labels for a grounded decision."
        )
    if tool == "ScoreHazard":
        return (
            f"Hazard evidence is prepared; score `{args.get('hazard')}` on `{args.get('scale')}` and keep support."
        )
    if tool == "EstimateExposure":
        return f"Estimate `{args.get('exposure')}` exposure with metrics {json.dumps(args.get('metrics'), ensure_ascii=False)}."
    if tool == "AssessDamage":
        return (
            f"Damage evidence is prepared; aggregate by `{args.get('aggregation')}` with severity "
            f"`{args.get('severity', 'all')}`."
        )
    if tool == "RankRegions":
        return (
            f"Rank `{args.get('candidates')}` by `{args.get('by')}` so the top item has explicit support."
        )
    if tool == "CompareScenes":
        return f"Compare prepared scene summaries by `{args.get('by')}`."
    if tool == "ExtractObjects":
        return f"Extract `{args.get('target')}` from the prepared context using the requested filters."
    if tool == "MapFlood":
        return (
            f"SAR work context is ready; derive flood extent with `{args.get('method')}` for mask/component evidence."
        )
    if tool == "DescribeFlood":
        return f"Flood evidence is ready; report metrics {json.dumps(args.get('metrics'), ensure_ascii=False)}."
    if tool == "PlanRoutes":
        return f"Route and hazard context is ready; plan for objective `{args.get('objective')}`."
    if tool == "CompareRoutes":
        return f"Candidate routes are ready; compare with preference {json.dumps(args.get('prefer'), ensure_ascii=False)}."
    if tool == "RenderMap":
        return (
            f"Map-ready evidence is prepared; render layers with style `{args.get('style')}` for the final artifact."
        )
    return f"`{tool}` is the next grounded operation for this contract."


def _result_support_summary(trace_row: dict[str, Any] | None) -> str:
    if not trace_row:
        return ""
    result = trace_row.get("result")
    if isinstance(result, dict):
        keys = [key for key in result if key != "sample_dir"]
        if keys:
            return "fields: " + ", ".join(str(k) for k in keys[:8])
    text = str(trace_row.get("text") or "").strip()
    if text and text.upper() != "OK":
        return text[:220]
    return ""


def _contract_completion_phrase(group: dict[str, Any]) -> str:
    tools = {str(call.get("tool")) for call in group.get("calls") or []}
    if {"MapFlood", "ExtractObjects"} <= tools:
        return "flood extent, polygon/component geometry, and grid-ready evidence are available"
    if "MapFlood" in tools:
        return "flood-mask evidence is available for downstream description or ranking"
    if {"ExtractObjects", "MeasureArea"} <= tools:
        return "the target disturbance/object evidence has been isolated and measured"
    if "ExtractObjects" in tools:
        return "the requested object/component evidence has been extracted from the prepared context"
    if "RenderMap" in tools:
        return "the requested visual/map artifact has been rendered and can be cited by ORC"
    if "PrepareScene" in tools:
        return "a validated reusable work context is ready with source metadata, units, masks, and schema cues preserved"
    if "ComputeStats" in tools:
        return "the requested statistic has been computed with units/support fields"
    if "CompareSources" in tools:
        return "the agreement/disagreement evidence across sources or references has been computed"
    if "DetectChange" in tools:
        return "the requested temporal-change evidence has been detected"
    if "ClassifyChange" in tools:
        return "the requested class/state decision has been grounded in the prepared numeric evidence"
    if "ScoreHazard" in tools:
        return "the requested hazard/severity score and supporting class distribution are ready"
    if "EstimateExposure" in tools:
        return "the requested exposure counts, population/asset support, and nearest-asset evidence are ready"
    if "AssessDamage" in tools:
        return "the requested damage interpretation is ready with counts, severity, and area support"
    if "RankRegions" in tools:
        return "the requested candidate ranking is ready with top item and support metrics"
    if "DescribeFlood" in tools:
        return "the requested flood description is ready with area, percentage, component, or grid support"
    if "PlanRoutes" in tools or "CompareRoutes" in tools:
        return "the requested route decision/tradeoff is ready with distance, ETA, risk, and blockage support"
    return "the requested delegated evidence is ready"


def completion_summary(group: dict[str, Any], last_trace: dict[str, Any] | None) -> str:
    handle = last_trace.get("handle") if last_trace else None
    text = str(last_trace.get("text") or "completed") if last_trace else "completed"
    support = _result_support_summary(last_trace)
    phrase = _contract_completion_phrase(group)
    support_suffix = f"; {support}" if support else ""
    if handle:
        return f"Contract complete: {phrase}. Latest handle `{handle}`{support_suffix}."
    text_suffix = f"; tool note: {text[:180]}" if text and text.upper() != "OK" else ""
    return f"Contract complete: {phrase}{text_suffix}{support_suffix}."


def _handle_flow_cards(record: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for handle in record.get("available_initial_handles") or sorted((record.get("initial_artifacts") or {}).keys()):
        card = (record.get("initial_artifacts") or {}).get(handle) or {}
        cards.append(
            {
                "handle": handle,
                "kind": card.get("type") or "data",
                "summary": artifact_note(handle, card),
            }
        )
    return cards


def _flow_plan_status(record: dict[str, Any], current_step_id: str | None, completed: set[str]) -> list[dict[str, Any]]:
    status: list[dict[str, Any]] = []
    for step in record.get("plan", {}).get("steps", []):
        sid = step["id"]
        if sid in completed:
            state = "done"
        elif sid == current_step_id:
            state = "running"
        else:
            state = "pending"
        status.append({"id": sid, "agent": step["agent"], "status": state})
    return status


def _compact_tool_observation(turn: dict[str, Any]) -> dict[str, Any]:
    action = turn.get("action") or {}
    out = turn.get("tool_output") or {}
    if isinstance(out.get("observation"), dict):
        return out["observation"]
    produced = out.get("produced") or {}
    fields = produced.get("fields") or {}
    result_key = next((k for k in fields if str(k).endswith("_result")), None)
    obs: dict[str, Any] = {
        "tool": action.get("tool"),
        "status": "OK" if out.get("ok", True) else "ERROR",
        "text": out.get("text") or "",
    }
    if fields.get("latest_handle"):
        obs["produced_handle"] = fields["latest_handle"]
    if result_key:
        obs["result"] = fields[result_key]
    field_names = [str(k) for k in fields.keys() if k not in {"latest_tool", "latest_tool_ok", "latest_tool_text"} and not str(k).endswith("_result")]
    if field_names:
        obs["visible_fields"] = field_names[:10]
    return obs


def _visible_context_for_step(
    record: dict[str, Any],
    step: dict[str, Any],
    visible_handles: dict[str, dict[str, Any]],
    last_observation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Small review snap of the evidence a specialist can actually see.

    The real SFT/runtime message is text built by ``build_specialist_message``.
    This JSON view mirrors that message without dumping paths or raw worker
    outputs: it shows the current contract's needed handles, the most recent
    observation, and a short inventory of produced handles.
    """

    initial = record.get("initial_artifacts") or {}
    needed: list[dict[str, Any]] = []
    for token in step.get("needs") or []:
        if token in initial:
            card = initial[token] or {}
            needed.append(
                {
                    "handle": token,
                    "kind": card.get("type") or "data",
                    "summary": artifact_note(token, card),
                    "source": "initial_input",
                }
            )
        elif token in visible_handles:
            item = dict(visible_handles[token])
            item["source"] = "produced"
            needed.append(item)
        elif token == "question_1":
            needed.append(
                {
                    "handle": "question_1",
                    "kind": "question",
                    "summary": "canonical question/schema handle",
                    "source": "question",
                }
            )

    produced_recent = list(visible_handles.values())[-6:]
    out: dict[str, Any] = {
        "needed_context": needed,
        "produced_context": produced_recent,
    }
    if last_observation:
        out["last_observation"] = last_observation
    return out


def _agent_input_snap(
    record: dict[str, Any],
    step: dict[str, Any],
    execution: dict[str, Any],
    turn_index: int,
    completed: set[str],
    visible_handles: dict[str, dict[str, Any]],
    last_observation: dict[str, Any] | None,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "contract": step["subtask"],
        "needs": step.get("needs") or [],
        "plan_status": _flow_plan_status(record, step["id"], completed),
        "allowed_tools": [*FINAL_DATASET_AGENT_TOOLS[str(execution["agent"])], "TaskComplete"],
        "visible_context": _visible_context_for_step(record, step, visible_handles, last_observation),
    }
    if turn_index == 0:
        snap["instruction"] = "Initial specialist handoff: use listed needs and prior artifacts only."
    else:
        snap["instruction"] = "Continue from the latest tool observation; finish if the contract evidence is present."
    return snap


def build_interleaved_gt_flow(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the episode-shaped teacher flow used for manual inspection.

    This is deliberately not a flat tool list. It mirrors the runtime:
    user/question -> ORC plan -> ORC CallAgent -> specialist action(s) with
    compact user/tool observations -> TaskComplete -> ORC next decision ->
    Terminate. The final ORC event carries only the stable runtime answer
    reference; the schema-valid resolved answer remains runtime-owned.
    """

    flow: list[dict[str, Any]] = []
    turn = 1

    def add(role: str, phase: str, **payload: Any) -> None:
        nonlocal turn
        flow.append({"turn": turn, "role": role, "phase": phase, **payload})
        turn += 1

    add(
        "USER",
        "QuestionToORC",
        target="ORC",
        snap={
            "question": record["raw_question"],
            "available_input_handles": _handle_flow_cards(record),
            "instruction": "Plan evidence contracts; do not call tools.",
        },
    )
    add(
        "ORC",
        "Plan",
        output=record["plan"],
        reasoning_snap="Split the task into evidence contracts by specialist capability; do not expose tool order.",
    )

    completed: set[str] = set()
    last_report: dict[str, Any] | None = None
    visible_handles: dict[str, dict[str, Any]] = {}
    last_observation_by_step: dict[str, dict[str, Any]] = {}
    steps_by_id = {step["id"]: step for step in record["plan"]["steps"]}
    for execution in record.get("executions") or []:
        step = steps_by_id[execution["step_id"]]
        add(
            "USER",
            "DecisionContextToORC",
            target="ORC",
            snap={
                "last_agent_report": last_report,
                "pending_step": {"id": step["id"], "agent": step["agent"], "contract": step["subtask"], "needs": step.get("needs") or []},
                "plan_status": _flow_plan_status(record, None, completed),
            },
        )
        add(
            "ORC",
            "CallAgent",
            output={"tool": "CallAgent", "step_id": step["id"]},
            reasoning_snap=f"Delegate `{step['id']}` to {step['agent']} because that specialist owns this evidence contract.",
        )

        for turn_index, agent_turn in enumerate(execution.get("turns") or []):
            action = agent_turn.get("action") or {}
            add(
                "USER",
                "SpecialistInput",
                target=execution["agent"],
                step_id=step["id"],
                snap=_agent_input_snap(
                    record,
                    step,
                    execution,
                    turn_index,
                    completed,
                    visible_handles,
                    last_observation_by_step.get(step["id"]),
                ),
            )
            if action.get("tool") == "TaskComplete":
                add(
                    execution["agent"],
                    "TaskComplete",
                    step_id=step["id"],
                    thought=agent_turn.get("thought"),
                    output=action,
                )
                last_report = {
                    "agent": execution["agent"],
                    "step_id": step["id"],
                    "status": action.get("status"),
                    "summary": action.get("summary"),
                }
            else:
                add(
                    execution["agent"],
                    "ToolCall",
                    step_id=step["id"],
                    thought=agent_turn.get("thought"),
                    output=action,
                )
                observation = _compact_tool_observation(agent_turn)
                produced_handle = observation.get("produced_handle")
                if produced_handle:
                    visible_handles[str(produced_handle)] = {
                        "handle": produced_handle,
                        "kind": str(produced_handle).split("_", 1)[0],
                        "summary": observation.get("summary") or "",
                        "evidence": compact(observation.get("evidence"), 700) if observation.get("evidence") else {},
                    }
                last_observation_by_step[step["id"]] = observation
                add(
                    "USER",
                    "ToolObservation",
                    target=execution["agent"],
                    step_id=step["id"],
                    snap=observation,
                )
        completed.add(step["id"])

    add(
        "USER",
        "DecisionContextToORC",
        target="ORC",
        snap={
            "last_agent_report": last_report,
            "pending_step": None,
            "plan_status": _flow_plan_status(record, None, completed),
        "available_answer_refs": [record["final_answer_ref"]],
        },
    )
    add(
        "ORC",
        "Terminate",
        output={"tool": "Terminate", "answer_ref": record["final_answer_ref"]},
        reasoning_snap="All delegated contracts are complete and the runtime has published the schema-valid answer reference.",
    )
    return flow


def build_orchestration_flow(record: dict[str, Any]) -> list[dict[str, Any]]:
    flow: list[dict[str, Any]] = [{"role": "ORC", "phase": "Plan", "goal": record["plan"]["goal"], "steps": record["plan"]["steps"]}]
    for execution in record.get("executions") or []:
        step_id = execution["step_id"]
        step = next((s for s in record["plan"]["steps"] if s["id"] == step_id), {})
        flow.append({"role": "ORC", "phase": "CallAgent", "step_id": step_id, "agent": execution["agent"], "contract": step.get("subtask"), "needs": step.get("needs")})
        complete = next((t for t in reversed(execution.get("turns") or []) if (t.get("action") or {}).get("tool") == "TaskComplete"), None)
        if complete:
            flow.append({"role": execution["agent"], "phase": "TaskComplete", "step_id": step_id, "summary": (complete.get("action") or {}).get("summary")})
    flow.append(
        {
            "role": "ORC",
            "phase": "Terminate",
            "answer_ref": record["final_answer_ref"],
        }
    )
    return flow


def derive_answer_from_executions(
    record: dict[str, Any], executions: list[dict[str, Any]]
) -> tuple[Any, str, str]:
    """Reconstruct the terminal answer from teacher worker observations only."""

    fields: dict[str, Any] = {}
    state: dict[str, Any] = {"answer_registry": {}}
    for execution in executions:
        step_id = str(execution.get("step_id") or "")
        for turn in execution.get("turns") or []:
            action = turn.get("action") or {}
            if action.get("tool") == "TaskComplete":
                continue
            output = turn.get("tool_output") or {}
            if not bool(output.get("ok", True)):
                continue
            for name, value in ((output.get("produced") or {}).get("fields") or {}).items():
                fields[str(name)] = {
                    "value": value,
                    "produced_by": step_id,
                    "tool": str(action.get("tool") or ""),
                }
            projection = project_from_state(record, fields)
            if projection.ok:
                register_answer(
                    state,
                    record,
                    projection.answer,
                    produced_by_step=step_id,
                    produced_by_tool=str(action.get("tool") or ""),
                )
    resolution = resolve_answer_ref(state, ANSWER_REF)
    if not resolution.ok:
        # Historical public teacher traces compacted rich worker observations.
        # They may omit a schema field even though the live worker produced it.
        # During migration only, accept the trace's already-resolved terminal
        # JSON after full schema validation. Fresh builds have no legacy
        # ``final_answer`` field and therefore cannot take this path.
        legacy = record.get("final_answer", record.get("resolved_final_answer"))
        validation = validate_explicit_answer(record, legacy)
        if not validation.ok:
            raise ValueError(
                f"teacher observations do not produce an answer for {record.get('dataset_id')}: "
                f"{resolution.error}; legacy resolved answer is invalid: {validation.errors}"
            )
        register_answer(
            state,
            record,
            validation.answer,
            produced_by_step="export_migration",
            produced_by_tool="validated_teacher_projection_cache",
        )
        resolution = resolve_answer_ref(state, ANSWER_REF)
        provenance = "validated_teacher_projection_cache"
    else:
        provenance = "stored_worker_observations"
    legacy = record.get("final_answer", record.get("resolved_final_answer"))
    expected = record.get("expected_final_answer")
    if (
        legacy is not None
        and expected is not None
        and not score_answer(expected, resolution.answer)["correct"]
    ):
        legacy_validation = validate_explicit_answer(record, legacy)
        if not legacy_validation.ok or not score_answer(
            expected, legacy_validation.answer
        )["correct"]:
            raise ValueError(
                f"stored worker projection and legacy resolved answer both differ "
                f"from canonical expected answer for {record.get('dataset_id')}"
            )
        register_answer(
            state,
            record,
            legacy_validation.answer,
            produced_by_step="export_migration",
            produced_by_tool="validated_teacher_projection_cache",
        )
        resolution = resolve_answer_ref(state, ANSWER_REF)
        provenance = "validated_teacher_projection_cache"
    return resolution.answer, answer_sha256(state), provenance


def build_multiagent_record(record: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    normalized_chain = normalize_chain_for_answer(record, record["gt_chain"])
    groups = group_chain(normalized_chain)
    visible_handles = set(record.get("available_initial_handles") or [])
    trace_by_step = trace_outputs_by_step(sim)
    step_counter = 0
    plan_steps: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for gi, group in enumerate(groups, start=1):
        step_id = f"s{gi}"
        needs = handles_in_calls(group["calls"], visible_handles)
        subtask = subtask_for_group(record, group, gi)
        plan_steps.append({"id": step_id, "agent": group["agent"], "subtask": subtask, "needs": needs})
        turns: list[dict[str, Any]] = []
        last_trace: dict[str, Any] | None = None
        produced_by_group: list[str] = []
        for call in group["calls"]:
            step_counter += 1
            trace_row = trace_by_step.get(step_counter, {})
            last_trace = trace_row
            if trace_row.get("handle"):
                produced_by_group.append(str(trace_row["handle"]))
            turns.append(
                {
                    "thought": agent_thought_for_call(call, group, len(turns) + 1, len(group["calls"])),
                    "action": {"tool": call["tool"], "args": agent_facing_args(str(call["tool"]), dict(call.get("args") or {}))},
                    "tool_output": tool_output_from_trace(trace_row),
                }
            )
        turns.append(
            {
                "thought": "The requested evidence is visible now; hand off a compact completion summary.",
                "action": {
                    "tool": "TaskComplete",
                    "status": "TASK_COMPLETE",
                    "summary": completion_summary(group, last_trace),
                },
            }
        )
        executions.append({"step_id": step_id, "agent": group["agent"], "turns": turns})
        visible_handles.update(produced_by_group)

    question_context = build_question_context(record)
    expected_answer = record.get("expected_final_answer")
    resolved_answer, resolved_sha256, answer_provenance = derive_answer_from_executions(record, executions)
    if not score_answer(expected_answer, resolved_answer)["correct"]:
        raise ValueError(
            f"worker-derived answer differs from canonical expected answer for {record['dataset_id']}"
        )
    out = {
        "schema_version": "geodisaster_multiagentic_sft_v3",
        "idx": int(record["record_index"]),
        "dataset_id": record["dataset_id"],
        "id": record["dataset_id"],
        "sample_id": record.get("sample_id"),
        "source": record["source"],
        "dataset": record.get("dataset"),
        "final_split": record.get("final_split"),
        "question_type": record["question_type"],
        "raw_question": question_text(record),
        "question": question_context,
        "initial_artifacts": record.get("initial_artifacts") or {},
        "available_initial_handles": record.get("available_initial_handles") or [],
        "plan": {"goal": f"Resolve the {_domain_hint(record)} task using the declared handles and answer schema.", "steps": plan_steps},
        "executions": executions,
        "final_answer_ref": ANSWER_REF,
        "resolved_final_answer": resolved_answer,
        "resolved_answer_sha256": resolved_sha256,
        "resolved_answer_provenance": answer_provenance,
        # Audit copy only; model messages never include it. Canonical core is
        # still authoritative and synchronization is checked by audit_contracts.
        "expected_final_answer": record.get("expected_final_answer"),
        "gt_chain": normalized_chain,
        "pure_toolchain_sim_status": sim.get("status"),
        "answer_support": portable_teacher_value(sim.get("answer_support")),
        "split_group_id": record.get("split_group_id"),
        "training_weight": float(record.get("training_weight", 1.0)),
        "training_weight_contract": record.get("training_weight_contract"),
    }
    out["orchestration_flow"] = build_orchestration_flow(out)
    out["interleaved_gt_flow"] = build_interleaved_gt_flow(out)
    out["multiagentic_gt_chain"] = out["interleaved_gt_flow"]
    return out


def write_preview_review(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# GeoDisaster Multi-Agent Teacher Preview",
        "",
        "This 43-case preview contains one representative record per source/question type.",
        "ORC contracts describe the evidence/result to obtain; they do not reveal the specialist tool sequence.",
        "Specialist turns contain the actual tool calls and reasoning used for SFT supervision.",
        "For the exact episode-style flow, inspect `final_multiagentic_preview43_flow.json`.",
        "",
        "## Summary",
        "",
        f"- Preview records: `{len(records)}`",
        f"- Mean plan steps: `{sum(len(r['plan']['steps']) for r in records) / len(records):.2f}`",
        "",
        "## Per-Template Contracts",
        "",
    ]
    for rec in records:
        lines.extend(
            [
                f"### {rec['dataset_id']}",
                "",
                f"- Source/question type: `{rec['source']}::{rec['question_type']}`",
                f"- Final answer reference: `{rec['final_answer_ref']}`",
        "- ORC plan:",
            ]
        )
        for step in rec["plan"]["steps"]:
            lines.append(f"  - `{step['id']}` `{step['agent']}` needs={step.get('needs', [])}: {step['subtask']}")
        lines.append("- Specialist tool actions:")
        for execution in rec.get("executions") or []:
            actions = [
                turn.get("action", {}).get("tool")
                for turn in execution.get("turns") or []
                if turn.get("action", {}).get("tool") != "TaskComplete"
            ]
            lines.append(f"  - `{execution['step_id']}` `{execution['agent']}`: {' -> '.join(actions)}")
            for turn in execution.get("turns") or []:
                if (turn.get("action") or {}).get("tool") == "TaskComplete":
                    lines.append(f"    - TaskComplete: {turn.get('action', {}).get('summary')}")
        lines.append("- ORC flow:")
        for item in rec.get("orchestration_flow") or []:
            if item["phase"] == "CallAgent":
                lines.append(f"  - ORC -> `{item['agent']}` for `{item['step_id']}`: {item['contract']}")
            elif item["phase"] == "TaskComplete":
                lines.append(f"  - `{item['role']}` complete `{item['step_id']}`: {item['summary']}")
            elif item["phase"] == "Terminate":
                lines.append(f"  - ORC terminates with grounded answer reference: `{item['answer_ref']}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_preview_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick one representative record per source/question-type template."""

    picked: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: int(item.get("record_index", 0))):
        key = logical_record_type(row)
        picked.setdefault(key, row)
    return [picked[key] for key in sorted(picked)]


def register_initial_artifacts(state: dict[str, Any], record: dict[str, Any]) -> None:
    for handle, card in (record.get("initial_artifacts") or {}).items():
        state["artifacts"][handle] = {
            "kind": str(card.get("type") or "data"),
            "path": first_path(card),
            "produced_by_step": "",
            "note": artifact_note(handle, card),
        }


def absorb_tool_output(
    state: dict[str, Any],
    record: dict[str, Any],
    step_id: str,
    tool_name: str,
    tool_output: dict[str, Any],
) -> dict[str, Any]:
    produced = tool_output.get("produced") or {}
    out_fields: dict[str, Any] = {}
    for name, value in (produced.get("fields") or {}).items():
        set_field(state, str(name), value, produced_by_step=step_id)
        out_fields[str(name)] = value
    if bool(tool_output.get("ok", True)):
        projection = project_from_state(record, state["fields"])
        if projection.ok:
            register_answer(
                state,
                record,
                projection.answer,
                produced_by_step=step_id,
                produced_by_tool=tool_name,
            )
    return {"artifacts": [], "layers": [], "fields": out_fields}


def last_specialist_report(state: dict[str, Any]) -> dict[str, Any] | None:
    for event in reversed(state["history"]):
        if event["role"] in {"GA", "PA", "VRA"} and event["phase"] == "complete":
            return {
                "agent": event["role"],
                "step_id": event["step_id"],
                "status": event["payload"].get("status"),
                "summary": event["payload"].get("summary"),
            }
    return None


def build_role_rows(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    state = new_episode(question=str(record.get("raw_question") or record["question"]))
    register_initial_artifacts(state, record)

    plan_payload = dict(record["plan"])
    plan = Plan.model_validate(plan_payload)
    plan_user = build_orc_plan_message(state)
    rows["ORC"].append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["ORC_PLAN"]},
                {"role": "user", "content": plan_user},
                {"role": "assistant", "content": json_text(plan_payload)},
            ],
            "role": "ORC",
            "case_id": record["idx"],
            "dataset_id": record["dataset_id"],
            "step_id": "plan",
            "kind": "plan",
        }
    )
    install_plan(state, plan)
    append_event(state, role="ORC", phase="plan", payload={"plan": plan_payload, "user_message": plan_user})

    executions = list(record.get("executions") or [])
    for exec_index, execution in enumerate(executions):
        step_id = str(execution["step_id"])
        step = find_step(state, step_id)
        if step is None:
            raise ValueError(f"missing plan step {step_id} in {record['dataset_id']}")

        decide_user = build_orc_decide_message(state, last_specialist_report(state))
        call_payload = {"tool": "CallAgent", "step_id": step_id}
        rows["ORC"].append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS["ORC_DECIDE"]},
                    {"role": "user", "content": decide_user},
                    {"role": "assistant", "content": json_text(call_payload)},
                ],
                "role": "ORC",
                "case_id": record["idx"],
                "dataset_id": record["dataset_id"],
                "step_id": step_id,
                "kind": "decide_call_agent",
            }
        )
        append_event(state, role="ORC", phase="delegate", step_id=step_id, payload={"decision": call_payload, "user_message": decide_user})
        state["current_step_id"] = step_id
        mark_step(state, step_id, "running")
        open_contract_window(state, step_id)
        contract_start = state["contract_window"][step_id]

        completion_status = "TASK_COMPLETE"
        completion_summary = ""
        role = str(execution["agent"])
        for turn_no, turn in enumerate(execution.get("turns") or [], start=1):
            user_msg = build_specialist_message(state, step, turn_no, contract_start)
            action = dict(turn.get("action") or {})
            target_payload = {
                "thought": str(turn.get("thought") or "Follow the subtask contract."),
                "action": action,
            }
            rows[role].append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPTS[role]},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": json_text(target_payload)},
                    ],
                    "role": role,
                    "case_id": record["idx"],
                    "dataset_id": record["dataset_id"],
                    "step_id": step_id,
                    "kind": "specialist_turn",
                    "turn_no": turn_no,
                }
            )
            append_event(state, role=role, phase="step", step_id=step_id, turn=turn_no, payload={"user_message": user_msg, "turn": target_payload})
            if action.get("tool") == "TaskComplete":
                completion_status = str(action.get("status", "TASK_COMPLETE"))
                completion_summary = str(action.get("summary", ""))
                continue
            tool_output = dict(turn.get("tool_output") or {})
            produced_dump = absorb_tool_output(
                state,
                record,
                step_id,
                str(action.get("tool") or ""),
                tool_output,
            )
            append_event(
                state,
                role="tool",
                phase="tool_call",
                step_id=step_id,
                turn=turn_no,
                payload={
                    "tool": action.get("tool"),
                    "args": action.get("args") or {},
                    "ok": bool(tool_output.get("ok", True)),
                    "text": str(tool_output.get("text", "")),
                    "produced": produced_dump,
                },
            )

        mark_step(state, step_id, "done" if completion_status == "TASK_COMPLETE" else "failed", completion_summary)
        state["current_step_id"] = None
        append_event(state, role=role, phase="complete", step_id=step_id, payload={"status": completion_status, "summary": completion_summary})

        if exec_index == len(executions) - 1:
            current_resolution = resolve_answer_ref(state, ANSWER_REF)
            if (
                not current_resolution.ok
                or not score_answer(
                    record["resolved_final_answer"], current_resolution.answer
                )["correct"]
            ):
                register_answer(
                    state,
                    record,
                    record["resolved_final_answer"],
                    produced_by_step=step_id,
                    produced_by_tool="validated_teacher_projection_cache",
                )
            final_user = build_orc_decide_message(state, last_specialist_report(state))
            resolution = resolve_answer_ref(state, ANSWER_REF)
            if not resolution.ok:
                raise ValueError(
                    f"SFT state has no terminal answer reference for {record['dataset_id']}: "
                    f"{resolution.error}"
                )
            if not score_answer(record["resolved_final_answer"], resolution.answer)["correct"]:
                raise ValueError(
                    f"SFT state answer differs from conversation projection for {record['dataset_id']}"
                )
            terminate_payload = {"tool": "Terminate", "answer_ref": ANSWER_REF}
            rows["ORC"].append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPTS["ORC_DECIDE"]},
                        {"role": "user", "content": final_user},
                        {"role": "assistant", "content": json_text(terminate_payload)},
                    ],
                    "role": "ORC",
                    "case_id": record["idx"],
                    "dataset_id": record["dataset_id"],
                    "step_id": "terminate",
                    "kind": "decide_terminate",
                }
            )
    return rows


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    tool_counts = Counter()
    role_tool_counts = Counter()
    step_counts = Counter()
    qtype_counts = Counter()
    for rec in records:
        qtype_counts[logical_record_type(rec)] += 1
        for step in rec["plan"]["steps"]:
            step_counts[step["agent"]] += 1
        for call in rec["gt_chain"]:
            tool = call.get("tool")
            if tool and tool != "TaskComplete":
                tool_counts[tool] += 1
                role_tool_counts[f"{TOOL_TO_AGENT[tool]}::{tool}"] += 1
    return {
        "records": len(records),
        "question_type_count": len(qtype_counts),
        "plan_steps_mean": round(sum(len(r["plan"]["steps"]) for r in records) / len(records), 4) if records else 0.0,
        "plan_step_counts_by_agent": dict(step_counts),
        "tool_call_counts": dict(tool_counts.most_common()),
        "role_tool_call_counts": dict(role_tool_counts.most_common()),
        "question_type_counts": dict(qtype_counts),
    }


def logical_record_type(record: dict[str, Any]) -> str:
    """Stable 43-way type key; A-D are distinct within each MH domain."""
    source = str(record.get("source") or "")
    qtype = str(record.get("question_type") or "")
    if source == "data/multi_hazard":
        parts = str(record.get("dataset_id") or record.get("id") or "").split("/")
        domain = parts[1] if len(parts) > 2 else "unknown"
        return f"{source}::{domain}:{qtype}"
    return f"{source}::{qtype}"


def write_role_sft(records: list[dict[str, Any]], output_dir: Path, suffix: str) -> dict[str, Any]:
    role_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    for rec in records:
        rows = build_role_rows(rec)
        for role in ROLE_ORDER:
            for row in rows[role]:
                row["id"] = rec["dataset_id"]
                row["sample_id"] = rec.get("sample_id")
                row["question_type"] = rec.get("question_type")
                row["training_weight"] = float(rec.get("training_weight", 1.0))
                row["training_weight_contract"] = rec.get("training_weight_contract")
                row["final_answer_ref"] = rec["final_answer_ref"]
                row["resolved_answer_sha256"] = rec["resolved_answer_sha256"]
            role_rows[role].extend(rows[role])

    manifest: dict[str, Any] = {"records": len(records), "roles": {}}
    for role, rows in role_rows.items():
        name = f"{role.lower()}_{suffix}.jsonl"
        if suffix == "train":
            name = f"{role.lower()}_train.jsonl"
        path = output_dir / name
        write_jsonl(path, rows)
        manifest["roles"][role] = {"path": name, "rows": len(rows)}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def write_interleaved_flow(path_json: Path, path_jsonl: Path, records: list[dict[str, Any]]) -> None:
    rows = [
        {
            "dataset_id": rec["dataset_id"],
            "source": rec["source"],
            "question_type": rec["question_type"],
            "raw_question": rec["raw_question"],
            "flow": rec["interleaved_gt_flow"],
        }
        for rec in records
    ]
    write_json(path_json, rows)
    write_jsonl(path_jsonl, rows)


def write_ownership_docs(path: Path) -> None:
    lines = [
        "# GeoDisaster Dataset Multi-Agent Tool Ownership",
        "",
        "Enable this benchmark profile with `MAS_AGENT_PROFILE=geodisaster`. The ownership table below is the active GeoDisaster contract.",
        "",
        "## Ownership",
        "",
        "| Agent | Tools | Rationale |",
        "|---|---|---|",
    ]
    rationale = {
        "GA": "Spatial/data operations and artifact creation: open handles, prepare scenes, measure areas, extract objects, map flood extent, render overlays.",
        "PA": "Largest set: numeric/statistical reasoning, comparisons, classifications, exposure, ranking, route planning, route tradeoffs.",
        "VRA": "Small visual interpretation set: damage/flood/scene semantic summaries after evidence is prepared.",
    }
    for agent, tools in FINAL_DATASET_AGENT_TOOLS.items():
        lines.append(f"| `{agent}` | {', '.join(f'`{t}`' for t in tools)} | {rationale[agent]} |")
    lines.extend(["", "## Agent-Facing Signatures", ""])
    for tool, sig in FINAL_DATASET_TOOL_SIGNATURES.items():
        lines.append(f"- `{sig}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(split_dir: Path, sim_jsonl: Path, output_dir: Path, limit: int = 0) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_by_id = load_sim_by_id(sim_jsonl)
    all_manifests: dict[str, Any] = {
        "tool_ownership": FINAL_DATASET_AGENT_TOOLS,
        "splits": {},
    }
    write_ownership_docs(output_dir / "tool_ownership_and_prompt_context.md")
    write_json(output_dir / "tool_ownership.json", {"agent_tools": FINAL_DATASET_AGENT_TOOLS, "tool_to_agent": TOOL_TO_AGENT})

    for split in ("train", "test"):
        src = split_dir / f"final_agentic_dataset_2921_{split}.jsonl"
        rows = read_jsonl(src)
        if limit:
            rows = rows[:limit]
        records = [build_multiagent_record(row, sim_by_id[str(row["dataset_id"])]) for row in rows]
        trace_path = output_dir / f"final_multiagentic_{split}.jsonl"
        write_jsonl(trace_path, records)
        write_interleaved_flow(
            output_dir / f"final_multiagentic_{split}_flow.json",
            output_dir / f"final_multiagentic_{split}_flow.jsonl",
            records,
        )
        write_json(output_dir / f"final_multiagentic_{split}_summary.json", summarize(records))
        sft_dir = output_dir / f"sft_{split}"
        sft_manifest = write_role_sft(records, sft_dir, "train" if split == "train" else "eval")
        all_manifests["splits"][split] = {
            "input": src.name,
            "multiagentic_jsonl": trace_path.name,
            "summary": summarize(records),
            "sft": sft_manifest,
        }

    write_json(output_dir / "final_multiagentic_manifest.json", all_manifests)
    return all_manifests


def build_preview43(split_dir: Path, sim_jsonl: Path, output_dir: Path) -> dict[str, Any]:
    """Build one clean multi-agent record per unique template/question type."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sim_by_id = load_sim_by_id(sim_jsonl)
    rows = []
    for split in ("train", "test"):
        rows.extend(read_jsonl(split_dir / f"final_agentic_dataset_2921_{split}.jsonl"))
    preview_src = select_preview_records(rows)
    records = [build_multiagent_record(row, sim_by_id[str(row["dataset_id"])]) for row in preview_src]
    preview_dir = output_dir / "preview_43"
    preview_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(preview_dir / "final_multiagentic_preview43.jsonl", records)
    write_json(preview_dir / "final_multiagentic_preview43.json", records)
    write_interleaved_flow(
        preview_dir / "final_multiagentic_preview43_flow.json",
        preview_dir / "final_multiagentic_preview43_flow.jsonl",
        records,
    )
    write_json(preview_dir / "final_multiagentic_preview43_summary.json", summarize(records))
    write_preview_review(preview_dir / "README_REVIEW.md", records)
    write_role_sft(records, preview_dir / "sft_preview43", "eval")
    return {"records": len(records), "output_dir": str(preview_dir), "summary": summarize(records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--sim_jsonl", type=Path, default=DEFAULT_SIM_JSONL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split.")
    parser.add_argument("--skip_preview43", action="store_true")
    parser.add_argument("--preview_only", action="store_true")
    args = parser.parse_args()
    preview = None
    if not args.skip_preview43:
        preview = build_preview43(args.split_dir.resolve(), args.sim_jsonl.resolve(), args.output_dir.resolve())
    if args.preview_only:
        print(json.dumps({"preview_43": preview}, indent=2, ensure_ascii=False))
        return
    manifest = build(args.split_dir.resolve(), args.sim_jsonl.resolve(), args.output_dir.resolve(), args.limit)
    if preview is not None:
        manifest["preview_43"] = preview
        write_json(args.output_dir.resolve() / "final_multiagentic_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
