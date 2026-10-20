"""Tool schemas for the final multi-source dataset workers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


TRANSPORT_ONLY_ARGS = {"session_id", "case", "case_path", "case_index"}


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _string_enum(values: list[str], description: str) -> dict[str, Any]:
    return {"type": "string", "enum": values, "description": description}


def _object(description: str) -> dict[str, Any]:
    return {"type": "object", "description": description, "additionalProperties": True}


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "ReadData": {
        "description": (
            "Open a named input handle for the current question, validate its local artifact paths, "
            "and return a scene handle containing the discovered raster/vector/image/report refs. "
            "The evaluator must initialize the session first with code/scripts/initialize_tool_session.py; "
            "case contents are not agent-visible tool arguments."
        ),
        "properties": {
            "session_id": _string("Episode/session id injected by the caller; keeps handles shared across tools."),
            "question": _string("Question handle, usually question_1."),
            "input": _string("Canonical input handle to open, such as hazard_stack_1 or sar_chip_1."),
            "role": _string_enum([
                "pre_post", "hansen", "ndvi_nbr", "vh", "wildfire", "deforestation",
                "urban_expansion", "drought", "air_pollution", "hospitals_and_schools",
                "highways", "nearest_highway", "population_in_footprint", "schools",
                "population", "post_disaster", "buildings", "target_classes", "scene_a",
                "scene_b", "roads_shelters_flood", "roads_shelters_flood_rgb", "primary",
                "known_water_reference",
            ], "Optional call role; a specific artifact-card role is authoritative."),
            "time": _string_enum(
                ["event", "event_window", "pre_post_epochs", "2001_2023"],
                "Optional temporal selector; a specific artifact-card time is authoritative.",
            ),
            "options": _object("Optional data-selection hints such as source groups or bands."),
        },
        "required": ["question", "input"],
    },
    "PrepareScene": {
        "description": (
            "Build an aligned work context from one or more opened scene handles. It checks handle "
            "types and records the processing mode before downstream analytics."
        ),
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "inputs": _object("Mapping from semantic slot names to scene handles, for example {'hazard':'scene_1'}."),
            "mode": _string_enum([
                "forest", "hazard", "xbd_damage", "sar_flood", "routing", "scene_compare",
                "forest_change", "forest_timeseries", "forest_false_positive", "baseline_canopy",
                "forest_loss_history", "forest_cross_modal", "sar_change", "wildfire_burn",
                "forest_loss", "built_up_gain", "drought_anomaly", "pollution_peak",
                "multisource_hazard", "wildfire_burn_exposure", "forest_loss_exposure",
                "built_up_gain_exposure", "drought_anomaly_exposure", "pollution_peak_exposure",
                "xbd_grid", "xbd_quadrants", "xbd_scene_summary", "flood_safe_routing",
                "sar_flood_reference", "sar_flood_diagnostic",
            ], "Supported preparation mode; task-specific names normalize to a guarded broad mode."),
            "checks": _object("Optional required checks, such as grid size or spatial alignment requirements."),
        },
        "required": ["question", "inputs", "mode"],
    },
    "ComputeStats": {
        "description": (
            "Compute numeric statistics for a work/damage/hazard handle, including area, counts, means, "
            "z-scores, class histograms, or footprint summaries."
        ),
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input work or analytic handle."),
            "metrics": _object(
                "Broad metric family requested by the question. Prefer canopy_cover, forest_loss_history, "
                "sar_change, hazard_area, vegetation_anomaly, pollution_peak, or damage_summary."
            ),
            "region": _string_enum(["aoi", "scene"], "Supported aggregation region."),
        },
        "required": ["question", "work", "metrics"],
    },
    "MeasureArea": {
        "description": (
            "Measure area for extracted objects or masks in the requested units by reading the underlying "
            "raster/vector data referenced by the input handle."
        ),
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input object/mask/work handle."),
            "target": _string_enum(["forest_loss"], "Target class to measure."),
            "unit": _string_enum(["hectares"], "Output unit."),
        },
        "required": ["question", "work", "target", "unit"],
    },
    "CompareSources": {
        "description": (
            "Compare multiple modalities or source layers, such as optical-vs-SAR, multi-source hazard "
            "agreement, or predicted-flood overlap versus pixels outside a known-water reference. "
            "Known-water context does not by itself establish temporal novelty."
        ),
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input work or analytic handle."),
            "compare": _object("Comparison specification: signals/sources/reference plus criterion."),
        },
        "required": ["question", "work", "compare"],
    },
    "DetectChange": {
        "description": "Detect or localize temporal change windows or dominant change events from a work handle.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input work/statistics handle."),
            "change": _string_enum(["largest_loss_year", "dominant_clearing_interval"], "Change concept to detect."),
            "window": _string_enum(["2001_2023", "pre_post_epochs"], "Temporal window to inspect."),
        },
        "required": ["question", "work", "change"],
    },
    "ClassifyChange": {
        "description": "Classify a change, canopy state, concordance state, or false-positive category using explicit labels/rules.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input work/change/statistics handle."),
            "labels": _object("Allowed labels or thresholds."),
            "rule": _string_enum([
                "forest_loss", "change_window", "false_positive_type", "canopy_density",
                "concordance", "cross_modal_concordance",
            ], "Supported deforestation classification rule."),
        },
        "required": ["question", "work", "labels", "rule"],
    },
    "ScoreHazard": {
        "description": "Score hazard class, severity, concentration, or impact footprint from a prepared/analytic handle.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input work/statistics/damage handle."),
            "hazard": _string_enum([
                "forest_disturbance", "wildfire", "deforestation", "urban_expansion",
                "drought", "air_pollution", "building_damage",
            ], "Hazard family to score."),
            "scale": _string_enum([
                "vh_delta_severity", "question_defined_class",
                "benchmark_relative_column_quartile", "impact_footprint",
                "spatial_concentration",
            ], "Supported scoring scale."),
        },
        "required": ["question", "work", "hazard", "scale"],
    },
    "EstimateExposure": {
        "description": "Estimate exposure of people/assets/roads/facilities to a hazard footprint or blocked-road layer.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input work handle."),
            "exposure": _string_enum([
                "blocked_roads", "high_no2_column_population", "highway_representatives_within_5km",
                "nearest_highway", "population_in_footprint", "schools",
            ], "Supported exposure target."),
            "metrics": _object("Requested exposure metrics."),
        },
        "required": ["question", "work", "exposure", "metrics"],
    },
    "AssessDamage": {
        "description": "Aggregate building-damage polygons into scene, grid, quadrant, severe-only, or pair-comparison summaries.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input work/object handle."),
            "aggregation": _string_enum(
                ["scene", "grid_4x4", "quadrant_2x2", "severe_only", "pair_compare"],
                "Supported damage aggregation mode.",
            ),
            "severity": _string_enum(["severe", "major_or_destroyed"], "Optional severity filter, when required by the aggregation."),
        },
        "required": ["question", "work", "aggregation"],
    },
    "RankRegions": {
        "description": "Rank regions, components, grid cells, quadrants, or route candidates by the requested score.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input analytic handle."),
            "by": _string_enum([
                "severe_footprint_area_m2", "severe_area_m2", "severe_damage_count",
                "severe_count", "normalized_damage_severity", "normalized_severity",
                "severity_score", "severity_score_normalized", "flood_percentage",
                "flood_pixels", "flood_area",
            ], "Supported ranking metric."),
            "candidates": _string_enum(["grid_4x4", "quadrants_2x2"], "Supported candidate set."),
            "top_k": {"type": "integer", "description": "Number of ranked entries to return."},
        },
        "required": ["question", "work", "by", "candidates"],
    },
    "CompareScenes": {
        "description": "Compare two prepared scene summaries and decide which scene has the larger damage/risk signal.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Pair-comparison damage/work handle."),
            "by": _string_enum(["severe_damage_ratio_and_footprint"], "Comparison criterion."),
        },
        "required": ["question", "work", "by"],
    },
    "ExtractObjects": {
        "description": (
            "Extract object sets such as forest-loss patches, severe buildings, flood polygons, or flood "
            "components. Sentinel flood polygons and components use 8-neighbour connectivity and produce "
            "validated run-scoped GeoJSON or label-raster artifacts."
        ),
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input work/flood handle."),
            "target": _string_enum(
                ["forest_loss_patches", "severe_buildings", "flood_polygons", "flood_components"],
                "Object target to extract.",
            ),
            "filters": _object("Optional class/year/severity filters."),
        },
        "required": ["question", "work", "target"],
    },
    "MapFlood": {
        "description": (
            "Produce a run-scoped 0/1/255 candidate-flood mask from a prepared SAR scene. The published "
            "Sentinel contract supports the validated s1_min_backscatter_otsu_v1 method."
        ),
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Prepared SAR flood work handle."),
            "method": _string_enum(["s1_min_backscatter_otsu_v1"], "Validated flood mapping method."),
        },
        "required": ["question", "work", "method"],
    },
    "DescribeFlood": {
        "description": (
            "Select and summarize requested flood-mask/component/polygon metrics, including exact-grid "
            "areas, centroids, 4x4 grids, and known-water overlap statistics from the causal handle."
        ),
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Flood/object handle."),
            "metrics": _object("Requested flood metrics."),
        },
        "required": ["question", "work", "metrics"],
    },
    "PlanRoutes": {
        "description": "Recompute route metrics and composite edge costs from declared packaged geometry/road products, then select eligible fastest, safest-weighted, and balanced candidates; this worker does not run a new graph search.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Prepared routing work handle."),
            "objective": _string_enum(
                ["nearest_reachable_shelter", "candidate_routes", "overlay_routes"],
                "Supported packaged route objective.",
            ),
            "avoid": _object("Required packaged eligibility assertion: flood_affected_roads must be true."),
        },
        "required": ["question", "work", "objective", "avoid"],
    },
    "CompareRoutes": {
        "description": "Compare requested validated fastest, safest-weighted, and balanced candidates and compute the full trade-off when all three are requested, without adding an evacuation recommendation.",
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Routes handle."),
            "prefer": _object("Boolean fastest/safest/balanced groups to include; at least one must be true."),
        },
        "required": ["question", "work", "prefer"],
    },
    "RenderMap": {
        "description": (
            "Render a fresh run-scoped map/image artifact and validation manifest from the current "
            "analytical handle and exactly the requested declared layers; prepared answer overlays and "
            "unrequested context layers are not retrieved."
        ),
        "properties": {
            "session_id": _string("Episode/session id."),
            "question": _string("Question handle."),
            "work": _string("Input map-ready handle."),
            "layers": _object("Layer toggles to render."),
            "style": _string_enum(["route_overlay", "flood_diagnostic"], "Supported rendering style."),
        },
        "required": ["question", "work", "layers", "style"],
    },
}


def agent_tool_spec(tool_name: str) -> dict[str, Any]:
    """Return the model-visible portion of a tool contract.

    Session/case bootstrap values belong to the evaluator transport. They are
    deliberately absent from prompts and model-generated calls.
    """

    spec = TOOL_SPECS[tool_name]
    properties = {
        name: deepcopy(value)
        for name, value in spec["properties"].items()
        if name not in TRANSPORT_ONLY_ARGS
    }
    return {
        "description": spec["description"],
        "properties": properties,
        "required": [name for name in spec["required"] if name in properties],
    }


def render_agent_signature(tool_name: str) -> str:
    """Render the prompt signature directly from ``TOOL_SPECS``."""

    spec = agent_tool_spec(tool_name)
    required = set(spec["required"])
    type_names = {
        "string": "str",
        "object": "dict",
        "array": "list",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }
    args = []
    for name, schema in spec["properties"].items():
        marker = "" if name in required else "?"
        rendered_type = type_names.get(schema.get("type"), "Any")
        if schema.get("enum"):
            rendered_type = "|".join(repr(value) for value in schema["enum"])
        args.append(f"{name}{marker}: {rendered_type}")
    return f"{tool_name}({', '.join(args)})  # {spec['description']}"


def validate_agent_args(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Validate names and basic JSON types before dispatching a model call."""

    if tool_name not in TOOL_SPECS:
        return [f"unknown tool: {tool_name}"]
    if not isinstance(args, dict):
        return ["tool arguments must be a JSON object"]
    spec = agent_tool_spec(tool_name)
    properties = spec["properties"]
    errors: list[str] = []
    missing = sorted(set(spec["required"]) - set(args))
    extra = sorted(set(args) - set(properties))
    if missing:
        errors.append(f"missing required arguments: {', '.join(missing)}")
    if extra:
        errors.append(f"unknown arguments: {', '.join(extra)}")
    expected_types = {
        "string": str,
        "object": dict,
        "array": list,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    for name in sorted(set(args) & set(properties)):
        expected = expected_types.get(properties[name].get("type"))
        value = args[name]
        if expected is None or value is None:
            continue
        if properties[name].get("type") in {"integer", "number"} and isinstance(value, bool):
            errors.append(f"argument {name} has type bool; expected {properties[name].get('type')}")
        elif not isinstance(value, expected):
            errors.append(
                f"argument {name} has type {type(value).__name__}; expected {properties[name].get('type')}"
            )
        elif "enum" in properties[name] and value not in properties[name]["enum"]:
            errors.append(
                f"argument {name} has unsupported value {value!r}; expected one of {properties[name]['enum']!r}"
            )
    return errors


def get_tool_instruction(tool_name: str, *, agent_visible: bool = False) -> dict[str, Any]:
    spec = TOOL_SPECS[tool_name]
    if agent_visible:
        spec = agent_tool_spec(tool_name)
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": spec["description"],
            "parameters": {
                "type": "object",
                "properties": deepcopy(spec["properties"]),
                "required": list(spec["required"]),
            },
        },
    }


def api_map(start_port: int = 20102) -> list[dict[str, Any]]:
    rows = []
    for offset, tool_name in enumerate(TOOL_SPECS):
        rows.append(
            {
                "tool": tool_name,
                "worker_file": f"{tool_name}_worker.py",
                "default_port": start_port + offset,
                "instruction": get_tool_instruction(tool_name),
            }
        )
    return rows
