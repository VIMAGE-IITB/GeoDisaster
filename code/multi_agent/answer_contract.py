"""Schema-aware projection of real worker evidence into terminal answers.

Workers intentionally return rich, reusable evidence payloads.  Benchmark
answers are narrower JSON objects governed by the answer schema embedded in
each question.  This module is the sole adapter between those contracts.

The adapter never reads an expected answer, GT chain, label, or teacher
trajectory.  It uses only the evaluator-sanitized question metadata and
successful worker payloads already present in agent state.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class AnswerProjection:
    ok: bool
    answer: Any | None
    logical_type: str
    errors: tuple[str, ...] = ()


def logical_question_type(case: dict[str, Any]) -> str:
    record_id = str(case.get("dataset_id") or case.get("id") or "")
    parts = record_id.split("/")
    qtype = str(case.get("question_type") or (parts[-1] if parts else "unknown"))
    if parts and parts[0] == "multi_hazard" and len(parts) >= 2:
        return f"multi_hazard/{parts[1]}/{qtype}"
    family = parts[0] if parts and parts[0] else str(case.get("source") or "unknown")
    return f"{family}/{qtype}"


def answer_schema(case: dict[str, Any]) -> dict[str, Any] | None:
    question = case.get("question")
    if isinstance(question, dict) and isinstance(question.get("answer_schema"), dict):
        return deepcopy(question["answer_schema"])
    if isinstance(case.get("answer_schema"), dict):
        return deepcopy(case["answer_schema"])
    if isinstance(question, str):
        marker = re.search(r"(?:^|\n)ANSWER SCHEMA\s*\n", question, flags=re.I)
        if marker:
            tail = question[marker.end() :].lstrip()
            try:
                value, _ = json.JSONDecoder().raw_decode(tail)
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _unwrap_field(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value and (
        "tool" in value or "produced_by" in value or "ts" in value
    ):
        return value["value"]
    return value


def payloads_from_fields(fields: dict[str, Any]) -> list[dict[str, Any]]:
    """Return successful full payloads in state insertion order."""

    out: list[dict[str, Any]] = []
    for key, record in fields.items():
        if not key.endswith("_payload"):
            continue
        value = _unwrap_field(record)
        if isinstance(value, dict):
            out.append(value)
    return out


def _dict_nodes(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _dict_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dict_nodes(child)


class _Evidence:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.nodes: list[dict[str, Any]] = []
        # Newer tools take precedence, while each payload's root takes
        # precedence over nested provenance and preview objects.
        for payload in reversed(payloads):
            self.nodes.extend(_dict_nodes(payload))

    def value(self, key: str, *aliases: str, default: Any = None) -> Any:
        for name in (key, *aliases):
            for node in self.nodes:
                if name in node:
                    return deepcopy(node[name])
        return _MISSING if default is _MISSING else deepcopy(default)

    def node_with(self, *keys: str) -> dict[str, Any] | None:
        for node in self.nodes:
            if all(key in node for key in keys):
                return node
        return None


_MISSING = object()


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.extend(_nonfinite_paths(child, f"{path}.{key}"))
        return out
    if isinstance(value, list):
        out = []
        for index, child in enumerate(value):
            out.extend(_nonfinite_paths(child, f"{path}[{index}]"))
        return out
    return []


def _schema_value(
    schema: dict[str, Any],
    source: Any,
    evidence: _Evidence,
    *,
    aliases: dict[str, tuple[str, ...]] | None = None,
    field_name: str = "",
) -> Any:
    aliases = aliases or {}
    # A JSON Schema const is public response metadata, not hidden evidence.
    # Emit the declared constant even when a rich worker payload uses a more
    # generic internal label for the same computation.
    if "const" in schema:
        return deepcopy(schema["const"])
    schema_type = schema.get("type")
    # Preserve an explicit worker-produced null for nullable contracts.  A
    # null change window, for example, is semantically different from an
    # empty object and must not be expanded through object properties.
    if source is None and (
        schema_type == "null"
        or (isinstance(schema_type, list) and "null" in schema_type)
    ):
        return None
    if schema_type == "object" or "properties" in schema:
        local = source if isinstance(source, dict) else {}
        result: dict[str, Any] = {}
        for key, child_schema in (schema.get("properties") or {}).items():
            child = local.get(key, _MISSING)
            if child is _MISSING:
                child = evidence.value(key, *aliases.get(key, ()), default=_MISSING)
            value = _schema_value(
                child_schema,
                child,
                evidence,
                aliases=aliases,
                field_name=key,
            )
            if value is not _MISSING:
                result[key] = value
        return result
    if schema_type == "array" and isinstance(source, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return [
                _schema_value(item_schema, item, evidence, aliases=aliases)
                for item in source
            ]
        return deepcopy(source)
    if source is _MISSING:
        if "const" in schema:
            return deepcopy(schema["const"])
        return _MISSING
    return deepcopy(source)


def _project_schema(
    schema: dict[str, Any],
    evidence: _Evidence,
    source: Any = _MISSING,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> Any:
    return _schema_value(schema, source, evidence, aliases=aliases)


def _deforestation(qtype: str, schema: dict[str, Any], ev: _Evidence) -> Any:
    aliases: dict[str, tuple[str, ...]] = {}
    if qtype == "area_estimation":
        comparison = ev.node_with("hansen", "optical") or {}
        hansen = comparison.get("hansen") or {}
        optical = comparison.get("optical") or {}
        return {
            "loss_area_ha": hansen.get("loss_area_ha", ev.value("area_ha")),
            "evidence": {
                "hansen_loss_area_ha": hansen.get("loss_area_ha", ev.value("area_ha")),
                "nbr_drop": (optical.get("NBR") or {}).get("drop"),
                "ndvi_drop": (optical.get("NDVI") or {}).get("drop"),
            },
        }
    elif qtype == "baseline_canopy_assessment":
        aliases["baseline_class"] = ("predicted_class",)
    return _project_schema(schema, ev, aliases=aliases)


def _multi_hazard(domain: str, qtype: str, schema: dict[str, Any], ev: _Evidence) -> Any:
    root = ev.nodes[0] if ev.nodes else {}
    evidence_schema = schema["properties"]["evidence"]
    evidence_answer = _project_schema(evidence_schema, ev, source=root)

    if qtype == "A":
        if domain == "air_pollution":
            answer = {"units": ev.value("units"), "value": ev.value("peak_no2")}
        elif domain == "drought":
            answer = {"units": "z_score", "value": ev.value("z_score")}
            evidence_answer["ndvi_z_score"] = ev.value("z_score")
        else:
            footprint = ev.value("primary_footprint", default={})
            answer = {"units": footprint.get("units"), "value": footprint.get("area_ha")}
            evidence_answer["primary_area_ha"] = footprint.get("area_ha")
            evidence_answer["primary_handle"] = footprint.get("handle")
    elif qtype == "B":
        answer = {
            "class": ev.value("dominant_class"),
            "dominant_fraction": ev.value("dominant_fraction"),
        }
    elif qtype == "C":
        area_agreement = "agreement_score_0_100" in (
            schema["properties"]["answer"].get("properties") or {}
        )
        if area_agreement:
            score = ev.value("score")
            # Area-agreement workers retain full per-source cards.  The answer
            # contract exposes only their computed areas.
            per_source = ev.value("per_source", default={}) or {}
            per_source_area = {
                name: card.get("area_ha")
                for name, card in per_source.items()
                if isinstance(card, dict) and "area_ha" in card
            }
            evidence_answer["agreement_score_0_100"] = score
            evidence_answer["per_source_area_ha"] = per_source_area
        answer_schema = schema["properties"]["answer"]
        answer = _project_schema(answer_schema, ev, source=root)
        if area_agreement:
            answer["agreement_score_0_100"] = score
    elif qtype == "D":
        if domain in {"air_pollution", "urban_expansion"}:
            answer = {"population_in_footprint": ev.value("population_within_footprint")}
        elif domain == "drought":
            schools = (ev.value("critical_infrastructure", default={}) or {}).get("schools", {})
            answer = {"schools_in_footprint": schools.get("in_footprint")}
        elif domain == "wildfire":
            highways = (ev.value("critical_infrastructure", default={}) or {}).get("highways", {})
            answer = {"highway_representatives_within_5km": highways.get("in_buffer")}
        else:
            highways = (ev.value("critical_infrastructure", default={}) or {}).get("highways", {})
            nearest = highways.get("nearest") if isinstance(highways, dict) else {}
            answer = {
                "availability_status": ev.value("availability_status"),
                "nearest_highway_representative_distance_m": (nearest or {}).get("distance_m"),
            }
    else:
        return _project_schema(schema, ev, source=root)
    return {"answer": answer, "evidence": evidence_answer, "status": "TASK_COMPLETE"}


def _route_line(label: str, route: dict[str, Any]) -> str:
    return (
        f"{label}: {route['route_id']} ({route['route_type']}): "
        f"{route['distance_m']:.1f} m, ETA {route['eta_min']:.1f} min, "
        f"length-weighted risk {route['risk_score']:.2f}, objective cost "
        f"{route['objective_cost']:.3f} weighted min, blocked edges {route['blocked_edges']}."
    )


def _routing(qtype: str, schema: dict[str, Any], ev: _Evidence) -> Any:
    if qtype == "visual_overlay":
        render = ev.node_with("artifact", "exists", "render_validation") or {}
        answer = _project_schema(schema, ev, source=render)
        answer["terminated"] = True
        return answer
    routes = ev.value("best", default={}) or {}
    if qtype == "nearest_shelter":
        fastest = routes.get("fastest") or ev.value("fastest_route", default={})
        shelter = ev.value("nearest_reachable_shelter", default={}) or {}
        if not shelter and isinstance(fastest, dict):
            shelter = {
                "name": fastest.get("selected_shelter_name"),
                "confidence": fastest.get("shelter_confidence"),
            }
        # Preserve the benchmark's capitalization and punctuation exactly.
        text = (
            f"The supplied selected reachable shelter is {shelter['name']} ({shelter['confidence']}). "
            f"Validated fastest-objective route: {fastest['route_id']} ({fastest['route_type']}): "
            f"{fastest['distance_m']:.1f} m, ETA {fastest['eta_min']:.1f} min, "
            f"length-weighted risk {fastest['risk_score']:.2f}, objective cost "
            f"{fastest['objective_cost']:.3f} weighted min, blocked edges {fastest['blocked_edges']}."
        )
        return {"final_answer": text, "terminated": True}
    fastest = routes.get("fastest") or ev.value("fastest_route", default={})
    safest = routes.get("safest") or ev.value("safest_route", default={})
    balanced = routes.get("balanced") or ev.value("balanced_route", default={})
    delta = ev.value("delta", default={}) or {}
    text = " ".join(
        [
            _route_line("Fastest-objective route", fastest),
            _route_line("Safest-weighted route", safest),
            _route_line("Balanced-objective route", balanced),
            (
                f"The safest-weighted route adds {delta.get('safest_extra_distance_m', 0.0):.1f} m "
                f"and {delta.get('safest_extra_eta_min', 0.0):.1f} min versus the fastest route, "
                f"with a mean-risk reduction of {delta.get('safest_risk_reduction', delta.get('risk_reduction', 0.0)):.2f}."
            ),
            "Safest-weighted denotes the declared composite edge cost, not minimum mean risk alone.",
        ]
    )
    return {"final_answer": text, "terminated": True}


def _sen1(schema: dict[str, Any], ev: _Evidence) -> Any:
    aliases = {
        "flood_area_m2": ("computed_flood_area_m2",),
        "flood_percentage": ("computed_flood_percentage",),
        "prediction_method": ("method",),
    }
    report = ev.node_with("computed_flood_area_m2", "computed_flood_percentage") or {}
    source = dict(report)
    if report:
        source["flood_area_m2"] = report.get("computed_flood_area_m2")
        source["flood_percentage"] = report.get("computed_flood_percentage")
        source["prediction_method"] = report.get("method")
    return _project_schema(schema, ev, source=source, aliases=aliases)


def _xbd(qtype: str, schema: dict[str, Any], ev: _Evidence) -> Any:
    if qtype == "XBD-Q3":
        aliases = {"eligible_cell_count": ("eligible_candidate_count",)}
        answer = _project_schema(schema, ev, aliases=aliases)
        top = ev.value("top_k_cells", default=[]) or []
        if top:
            answer["classified_building_count"] = top[0].get("classified_building_count")
        answer["tie_break_applied"] = len(ev.value("top_tied_cell_ids", default=[]) or []) > 1
        return answer
    if qtype == "XBD-Q4":
        source = ev.node_with("scenes", "higher_severe_area_scene") or {}
        answer = _project_schema(schema, ev, source=source)
        scenes = source.get("scenes") or []
        if len(scenes) >= 2:
            answer["scene_a"] = _project_schema(
                schema["properties"]["scene_a"], ev, source=scenes[0]
            )
            answer["scene_b"] = _project_schema(
                schema["properties"]["scene_b"], ev, source=scenes[1]
            )
        return answer
    return _project_schema(schema, ev)


def project_final_answer(
    case: dict[str, Any], payloads: list[dict[str, Any]]
) -> AnswerProjection:
    logical_type = logical_question_type(case)
    schema = answer_schema(case)
    if schema is None:
        return AnswerProjection(False, None, logical_type, ("answer schema unavailable",))
    if not payloads:
        return AnswerProjection(False, None, logical_type, ("no successful worker payloads",))
    ev = _Evidence(payloads)
    parts = logical_type.split("/")
    try:
        if parts[0] == "deforestation":
            answer = _deforestation(parts[-1], schema, ev)
        elif parts[0] == "multi_hazard":
            answer = _multi_hazard(parts[1], parts[-1], schema, ev)
        elif parts[0] == "routing":
            answer = _routing(parts[-1], schema, ev)
        elif parts[0] == "sen1_flood":
            answer = _sen1(schema, ev)
        elif parts[0] == "xbd_damage":
            answer = _xbd(parts[-1], schema, ev)
        else:
            answer = _project_schema(schema, ev)
    except (KeyError, TypeError, ValueError) as exc:
        return AnswerProjection(False, None, logical_type, (f"projection incomplete: {exc}",))
    nonfinite = _nonfinite_paths(answer)
    if nonfinite:
        return AnswerProjection(
            False,
            answer,
            logical_type,
            tuple(f"non-finite JSON number at {path}" for path in nonfinite),
        )
    errors = tuple(
        error.message
        for error in sorted(Draft202012Validator(schema).iter_errors(answer), key=lambda e: list(e.path))
    )
    if errors:
        return AnswerProjection(False, answer, logical_type, errors)
    return AnswerProjection(True, answer, logical_type)


def project_from_state(case: dict[str, Any], fields: dict[str, Any]) -> AnswerProjection:
    return project_final_answer(case, payloads_from_fields(fields))


def validate_explicit_answer(case: dict[str, Any], answer: Any) -> AnswerProjection:
    logical_type = logical_question_type(case)
    schema = answer_schema(case)
    if schema is None:
        return AnswerProjection(False, None, logical_type, ("answer schema unavailable",))
    value = answer
    if isinstance(value, str):
        text = value.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
        if fenced:
            text = fenced.group(1).strip()
        try:
            value = json.loads(text, parse_constant=_reject_nonstandard_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            return AnswerProjection(False, None, logical_type, (f"answer is not JSON: {message}",))
    nonfinite = _nonfinite_paths(value)
    if nonfinite:
        return AnswerProjection(
            False,
            value,
            logical_type,
            tuple(f"non-finite JSON number at {path}" for path in nonfinite),
        )
    errors = tuple(
        error.message
        for error in sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: list(e.path))
    )
    if errors:
        return AnswerProjection(False, value, logical_type, errors)
    return AnswerProjection(True, value, logical_type)


__all__ = [
    "AnswerProjection",
    "answer_schema",
    "logical_question_type",
    "payloads_from_fields",
    "project_final_answer",
    "project_from_state",
    "validate_explicit_answer",
]
