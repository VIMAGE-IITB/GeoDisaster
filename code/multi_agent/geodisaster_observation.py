"""Compact GeoDisaster tool observations for teacher replay and online eval.

The workers may compute rich dictionaries, but the agent should see a small
state snapshot: produced handle, the few values needed for the next reasoning
step, and enough provenance to avoid guessing. This module is shared by SFT
generation and runtime tool calls so training and inference expose the same
kind of evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any


_ROUTE_KEYS = (
    "route_id",
    "route_type",
    "distance_km",
    "eta_min",
    "risk_score",
    "blocked_edges",
    "selected_shelter_name",
)

_NUM_RE = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

_DAMAGE_KEYS = [
    "aggregation",
    "scene_id",
    "disaster",
    "damage_distribution",
    "total_buildings",
    "classified_buildings",
    "damaged_count",
    "severe_count",
    "damaged_pct",
    "severe_pct",
    "severe_footprint_m2",
    "total_footprint_m2",
    "spatial_dispersion",
    "cell_count",
    "top_severe_cells",
    "greater_damage_scene",
]

_PRIVATE_RUNTIME_PAYLOAD_KEYS = {"sample_dir"}


def _public_runtime_payload(value: Any) -> Any:
    """Remove machine-local bookkeeping that is not analytical evidence."""

    if isinstance(value, dict):
        public: dict[str, Any] = {}
        for key, child in value.items():
            if key in _PRIVATE_RUNTIME_PAYLOAD_KEYS:
                continue
            if key.endswith("_payload_keys") and isinstance(child, list):
                child = [
                    item
                    for item in child
                    if item not in _PRIVATE_RUNTIME_PAYLOAD_KEYS
                ]
            public[key] = _public_runtime_payload(child)
        return public
    if isinstance(value, list):
        return [_public_runtime_payload(child) for child in value]
    return value


def _unpack(value: Any) -> Any:
    """Recover a compact dict when an older replay stored a truncated preview."""

    if isinstance(value, dict) and value.get("_truncated") and isinstance(value.get("_preview"), str):
        try:
            return json.loads(value["_preview"])
        except Exception:
            return value
    return value


def _preview_text(value: Any) -> str:
    if isinstance(value, dict) and value.get("_truncated") and isinstance(value.get("_preview"), str):
        return str(value["_preview"])
    return ""


def _first_number(text: str, *patterns: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern.replace("NUMBER", _NUM_RE), text, flags=re.DOTALL)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except Exception:
            continue
        return int(value) if value.is_integer() else value
    return None


def _first_string(text: str, *patterns: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1)
    return None


def _pick(obj: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: obj[key] for key in keys if key in obj and obj[key] is not None}


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_value(v) for v in value]
    return value


def _top_loss_years(hist: Any, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(hist, dict):
        return []
    rows = []
    for year, area in hist.items():
        try:
            rows.append((str(year), float(area)))
        except Exception:
            continue
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"year": year, "area_ha": round(area, 3)} for year, area in rows[:limit]]


def _route_summary(route: Any) -> dict[str, Any]:
    if not isinstance(route, dict):
        return {}
    return _round_value(_pick(route, list(_ROUTE_KEYS)))


def _scene_summary(scene: Any) -> dict[str, Any]:
    if not isinstance(scene, dict):
        return {}
    return _round_value(
        _pick(
            scene,
            [
                "scene_id",
                "disaster",
                "damage_distribution",
                "total_buildings",
                "classified_buildings",
                "damaged_count",
                "severe_count",
                "damaged_pct",
                "severe_pct",
                "severe_footprint_m2",
                "spatial_dispersion",
            ],
        )
    )


def _infra_summary(items: Any, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(items, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, item in items.items():
        if not isinstance(item, dict):
            continue
        nearest = item.get("nearest") if isinstance(item.get("nearest"), dict) else {}
        rows.append(
            {
                "type": name,
                "in_footprint": item.get("in_footprint"),
                "in_buffer": item.get("in_buffer"),
                "nearest_name": nearest.get("name"),
                "nearest_distance_m": nearest.get("distance_m"),
            }
        )
    rows.sort(key=lambda row: (float(row.get("in_footprint") or 0), float(row.get("in_buffer") or 0)), reverse=True)
    return _round_value(rows[:limit])


def _delta_summary(delta: Any) -> dict[str, Any]:
    if not isinstance(delta, dict):
        return {}
    return _round_value(
        _pick(
            delta,
            [
                "safest_extra_distance_m",
                "safest_extra_eta_min",
                "safest_risk_reduction",
                "balanced_extra_distance_m",
                "balanced_extra_eta_min",
                "balanced_risk_reduction",
            ],
        )
    )


def _damage_values(source: dict[str, Any]) -> dict[str, Any]:
    values = _pick(source, _DAMAGE_KEYS)
    if isinstance(source.get("target_mask"), dict):
        mask = source["target_mask"]
        values["target_mask"] = _pick(mask, ["available", "total_pixels", "non_background_pixels"])
        if isinstance(mask.get("histogram_pct_non_bg"), dict):
            values["target_mask"]["histogram_pct_non_bg"] = mask["histogram_pct_non_bg"]
    return values


def _compare_preview_values(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    criterion = _first_string(text, r'"criterion"\s*:\s*"([^"]+)"')
    signals = _first_string(text, r'"signals"\s*:\s*"([^"]+)"')
    if criterion or signals:
        values["compare"] = {k: v for k, v in {"criterion": criterion, "signals": signals}.items() if v}
    rules_block = _first_string(text, r'"classification_rules"\s*:\s*\[(.*?)\]')
    if rules_block:
        rules = re.findall(r'"([^"]+)"', rules_block)
        if rules:
            values["classification_rules"] = rules[:5]
    hansen = {
        "loss_area_ha": _first_number(text, r'"hansen"\s*:\s*\{.*?"loss_area_ha"\s*:\s*NUMBER'),
        "treecover2000_mean_pct": _first_number(text, r'"hansen"\s*:\s*\{.*?"treecover2000_mean_pct"\s*:\s*NUMBER'),
    }
    values["hansen"] = {k: v for k, v in hansen.items() if v is not None}
    optical: dict[str, Any] = {}
    for index_name in ("NDVI", "NBR"):
        drop = _first_number(text, rf'"{index_name}"\s*:\s*\{{.*?"drop"\s*:\s*NUMBER')
        pre = _first_number(text, rf'"{index_name}"\s*:\s*\{{.*?"pre_mean"\s*:\s*NUMBER')
        post = _first_number(text, rf'"{index_name}"\s*:\s*\{{.*?"post_mean"\s*:\s*NUMBER')
        row = {k: v for k, v in {"drop": drop, "pre_mean": pre, "post_mean": post}.items() if v is not None}
        if row:
            optical[index_name] = row
    if optical:
        values["optical"] = optical
    sar_delta = _first_number(
        text,
        r'"VH"\s*:\s*\{.*?"delta_db"\s*:\s*NUMBER',
        r"sar_vh_delta\(\s*NUMBER\s*dB\)",
    )
    if sar_delta is not None:
        values["sar"] = {"VH": {"delta_db": sar_delta}}
    return {k: v for k, v in values.items() if v}


def _damage_preview_values(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    aggregation = _first_string(text, r'"aggregation"\s*:\s*"([^"]+)"')
    cell_count = _first_number(text, r'"cell_count"\s*:\s*NUMBER')
    if aggregation:
        values["aggregation"] = aggregation
    if cell_count is not None:
        values["cell_count"] = cell_count
    cells: list[dict[str, Any]] = []
    for match in re.finditer(r'\{[^{}]*"cell_id"\s*:\s*"([^"]+)"[^{}]*\}', text, flags=re.DOTALL):
        chunk = match.group(0)
        row: dict[str, Any] = {"cell_id": match.group(1)}
        dominant = _first_string(chunk, r'"dominant_damage"\s*:\s*"([^"]+)"')
        if dominant:
            row["dominant_damage"] = dominant
        for key in ("building_count", "severe_count", "severe_pct", "severe_area_m2", "severity_score", "total_area_m2"):
            number = _first_number(chunk, rf'"{key}"\s*:\s*NUMBER')
            if number is not None:
                row[key] = number
        cells.append(row)
        if len(cells) >= 5:
            break
    if cells:
        values["sample_cells"] = cells
    return values


def _flood_preview_values(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    chip_id = _first_string(text, r'"chip_id"\s*:\s*"([^"]+)"')
    if chip_id:
        values["chip_id"] = chip_id
    for key in (
        "computed_flood_percentage",
        "computed_flood_area_m2",
        "computed_flood_area_km2",
        "computed_flood_pixel_count",
        "computed_valid_pixel_count",
        "component_count",
        "polygon_count",
        "largest_component_area_m2",
        "concentration_score",
    ):
        number = _first_number(text, rf'"{key}"\s*:\s*NUMBER')
        if number is not None:
            values[key] = number
    cells: list[dict[str, Any]] = []
    for match in re.finditer(r'\{[^{}]*"cell_id"\s*:\s*"([^"]+)"[^{}]*\}', text, flags=re.DOTALL):
        chunk = match.group(0)
        row: dict[str, Any] = {"cell_id": match.group(1)}
        for key in ("flood_percentage", "flood_pixels", "fraction_of_total_flood", "valid_pixels"):
            number = _first_number(chunk, rf'"{key}"\s*:\s*NUMBER')
            if number is not None:
                row[key] = number
        cells.append(row)
        if len(cells) >= 5:
            break
    if cells:
        values["top_grid_cells_preview"] = sorted(
            cells,
            key=lambda row: float(row.get("flood_percentage") or 0),
            reverse=True,
        )[:3]
    return values


def summarize_geodisaster_tool_result(
    tool: str,
    *,
    handle: str | None = None,
    text: str = "",
    result: Any = None,
    evidence: Any = None,
    ok: bool = True,
) -> dict[str, Any]:
    """Return the compact observation/state payload for a GeoDisaster tool."""

    preview = _preview_text(result) or _preview_text(evidence)
    result = _unpack(result)
    evidence = _unpack(evidence)
    source = result if isinstance(result, dict) else {}
    if not source and isinstance(evidence, dict):
        source = evidence

    summary = text if text and text.upper() != "OK" else ""
    values: dict[str, Any] = {}

    if not ok:
        summary = summary or f"{tool} failed"
    elif tool == "ReadData":
        summary = f"opened input as `{handle}`" if handle else "opened input handle"
    elif tool == "PrepareScene":
        summary = f"prepared reusable work context `{handle}`" if handle else "prepared reusable work context"
    elif tool == "ComputeStats":
        values = _pick(
            source,
            [
                "aoi_id",
                "region",
                "handle",
                "modality",
                "units",
                "mean",
                "median",
                "min",
                "max",
                "peak_no2",
                "p95_no2",
                "target_value",
                "baseline_mean",
                "baseline_std",
                "z_score",
                "anomaly_class",
                "canopy_class",
                "treecover2000_mean_pct",
                "loss_area_ha",
                "ndvi_drop",
                "nbr_drop",
                "sar_vh_delta_db",
            ],
        )
        if isinstance(source.get("metrics"), dict):
            values["metrics"] = source["metrics"]
        if isinstance(source.get("primary_footprint"), dict):
            values["primary_footprint"] = _pick(
                source["primary_footprint"],
                ["handle", "area_ha", "area_m2", "pixel_count", "pixel_area_ha"],
            )
        if isinstance(source.get("stats"), dict):
            values["stats"] = _damage_values(source["stats"])
        if isinstance(source.get("percentiles"), dict):
            values["percentiles"] = _pick(source["percentiles"], ["p50", "p95", "p99"])
        top_years = _top_loss_years(
            source.get("loss_year_histogram_ha")
            or source.get("loss_year_histogram_ha_approx")
        )
        if top_years:
            values["top_loss_years_ha"] = top_years
        summary = f"computed statistics `{handle}`" if handle else "computed statistics"
    elif tool == "MeasureArea":
        values = _pick(source, ["target", "unit", "area", "area_ha", "area_m2", "component_count", "largest_patch_pixels"])
        summary = f"measured area `{handle}`" if handle else "measured area"
    elif tool == "CompareSources":
        values = _pick(
            source,
            [
                "verdict",
                "agreement_class",
                "n_sources_used",
                "weakest_source",
                "score",
                "score_contract",
                "pairwise_iou",
                "pairwise_state_agreement",
                "pairwise_categorical_kappa",
                "pairwise_correlation",
                "pairwise_pct_delta",
                "median_area_ha",
                "median_pct_delta",
                "overlap_pct_of_flood",
                "outside_reference_pct_of_flood",
                "overlap_pixels",
                "outside_reference_pixels",
            ],
        )
        if isinstance(source.get("classification_rules"), list):
            values["classification_rules"] = source["classification_rules"][:5]
        for key in ("optical", "sar", "hansen"):
            if isinstance(source.get(key), dict):
                values[key] = _pick(
                    source[key],
                    [
                        "loss_area_ha",
                        "treecover2000_mean_pct",
                        "ndvi_drop",
                        "nbr_drop",
                        "sar_vh_delta_db",
                        "agreement_class",
                        "area_ha",
                        "pct_delta",
                    ],
                )
        if isinstance(source.get("per_source"), dict):
            values["per_source"] = {
                label: _pick(row, ["handle", "area_ha", "scale_m", "units"])
                for label, row in source["per_source"].items()
                if isinstance(row, dict)
            }
        if isinstance(source.get("optical"), dict):
            optical: dict[str, Any] = {}
            for index_name in ("NDVI", "NBR"):
                if isinstance(source["optical"].get(index_name), dict):
                    optical[index_name] = _pick(source["optical"][index_name], ["drop", "pre_mean", "post_mean"])
            if optical:
                values["optical"] = optical
        if isinstance(source.get("sar"), dict):
            sar: dict[str, Any] = {}
            polarizations = source["sar"].get("polarizations")
            if isinstance(polarizations, dict):
                for pol in ("VH", "VV"):
                    if isinstance(polarizations.get(pol), dict):
                        sar[pol] = _pick(polarizations[pol], ["delta_db", "pre", "post"])
            vh_minus_vv = source["sar"].get("VH_minus_VV")
            if isinstance(vh_minus_vv, dict):
                sar["VH_minus_VV"] = _pick(vh_minus_vv, ["delta", "pre", "post"])
            if sar:
                values["sar"] = sar
        if isinstance(source.get("mask_summary"), dict):
            values["mask_summary"] = _pick(
                source["mask_summary"],
                [
                    "computed_flood_percentage",
                    "computed_flood_area_m2",
                    "component_count",
                    "largest_component_area_m2",
                    "concentration_score",
                ],
            )
        if preview:
            for key, value in _compare_preview_values(preview).items():
                if key not in values or not values[key]:
                    values[key] = value
        summary = f"compared sources `{handle}`" if handle else "compared sources"
    elif tool == "DetectChange":
        values = _pick(source, ["best_window", "drop_magnitude", "signal_used", "confidence"])
        summary = f"detected change `{handle}`" if handle else "detected change"
    elif tool == "ClassifyChange":
        values = _pick(
            source,
            [
                "predicted_class",
                "confidence",
                "rules_fired",
                "false_positive_flags",
                "change_window",
                "source",
                "treecover2000_mean_pct",
            ],
        )
        if isinstance(source.get("evidence"), dict):
            values["evidence"] = _pick(
                source["evidence"],
                ["hansen_loss_area_ha", "ndvi_drop", "nbr_drop", "sar_vh_delta_db", "treecover2000_mean_pct"],
            )
        summary = f"classified change `{handle}`" if handle else "classified change"
    elif tool == "ScoreHazard":
        values = _pick(
            source,
            [
                "hazard",
                "scale",
                "severity",
                "assessment",
                "dominant_class",
                "concentration_score",
                "impact_footprint",
                "loss_area_ha",
                "sar_vh_delta_db",
                "mean",
                "max",
                "units",
            ],
        )
        if isinstance(source.get("by_class"), dict):
            values["by_class"] = source["by_class"]
        summary = f"scored hazard `{handle}`" if handle else "scored hazard"
    elif tool == "EstimateExposure":
        values = _pick(
            source,
            [
                "footprint_area_ha",
                "population_within_footprint",
                "population_within_buffer",
                "population_source",
                "edge_count",
                "blocked_edges",
                "risky_edges",
                "mean_risk",
                "max_risk",
                "road_buffer_m",
                "risk_buffer_m",
                "status",
            ],
        )
        if isinstance(source.get("critical_infrastructure"), dict):
            infra = _infra_summary(source["critical_infrastructure"])
            if infra:
                values["critical_infrastructure"] = infra
        summary = f"estimated exposure `{handle}`" if handle else "estimated exposure"
    elif tool == "AssessDamage":
        values = _damage_values(source)
        if isinstance(source.get("scenes"), list):
            values["scenes"] = [_scene_summary(scene) for scene in source["scenes"][:2]]
        if preview:
            for key, value in _damage_preview_values(preview).items():
                if key not in values or not values[key]:
                    values[key] = value
        summary = f"assessed damage `{handle}`" if handle else "assessed damage"
    elif tool == "RankRegions":
        ranked = source.get("ranked") if isinstance(source.get("ranked"), list) else []
        values = _pick(source, ["by", "candidates", "top_k"])
        values["top_ranked"] = [_round_value(item) for item in ranked[:3]]
        summary = f"ranked candidates `{handle}`" if handle else "ranked candidates"
    elif tool == "CompareScenes":
        values = _pick(source, ["by", "greater_damage_scene"])
        if isinstance(source.get("scenes"), list):
            values["scenes"] = [_scene_summary(scene) for scene in source["scenes"][:2]]
        summary = f"compared scenes `{handle}`" if handle else "compared scenes"
    elif tool == "ExtractObjects":
        values = _pick(
            source,
            [
                "target",
                "object_count",
                "component_count",
                "polygon_count",
                "largest_component_area_m2",
                "largest_component_centroid",
                "largest_component_pixel_count",
                "computed_flood_area_m2",
                "computed_flood_percentage",
                "total_area_m2",
                "total_area_ha",
                "area_m2",
                "pixel_count",
                "largest_patch_pixels",
                "event_year",
                "mask_kind",
            ],
        )
        if isinstance(source.get("top_3_grid_cells"), list):
            values["top_3_grid_cells"] = source["top_3_grid_cells"][:3]
        if isinstance(source.get("sample_objects"), list):
            values["sample_objects"] = source["sample_objects"][:2]
        summary = f"extracted objects `{handle}`" if handle else "extracted objects"
    elif tool == "MapFlood":
        mask_summary = source.get("mask_summary") if isinstance(source.get("mask_summary"), dict) else {}
        values = _pick(source, ["method", "mask_source"])
        values.update(
            _pick(
                mask_summary,
                [
                    "chip_id",
                    "computed_flood_percentage",
                    "computed_flood_area_m2",
                    "computed_flood_area_km2",
                    "component_count",
                    "polygon_count",
                    "largest_component_area_m2",
                    "largest_component_centroid",
                    "concentration_score",
                ],
            )
        )
        if isinstance(mask_summary.get("top_3_grid_cells"), list):
            values["top_3_grid_cells"] = mask_summary["top_3_grid_cells"][:3]
        if preview:
            for key, value in _flood_preview_values(preview).items():
                if key not in values or not values[key]:
                    values[key] = value
        summary = f"mapped flood `{handle}`" if handle else "mapped flood"
    elif tool == "DescribeFlood":
        values = _pick(
            source,
            [
                "chip_id",
                "method",
                "computed_flood_percentage",
                "computed_flood_area_m2",
                "computed_flood_area_km2",
                "computed_flood_pixel_count",
                "computed_valid_pixel_count",
                "component_count",
                "polygon_count",
                "largest_component_area_m2",
                "largest_component_centroid",
                "concentration_score",
                "overlap_pct_of_flood",
                "outside_reference_pct_of_flood",
            ],
        )
        if isinstance(source.get("top_3_grid_cells"), list):
            values["top_3_grid_cells"] = source["top_3_grid_cells"][:3]
        if preview:
            for key, value in _flood_preview_values(preview).items():
                if key not in values or not values[key]:
                    values[key] = value
        summary = f"described flood `{handle}`" if handle else "described flood"
    elif tool == "PlanRoutes":
        best = source.get("best") if isinstance(source.get("best"), dict) else {}
        values = {"best": {name: _route_summary(route) for name, route in best.items() if isinstance(route, dict)}}
        summary = f"planned routes `{handle}`" if handle else "planned routes"
    elif tool == "CompareRoutes":
        values = _pick(source, ["prefer"])
        if isinstance(source.get("delta"), dict):
            values["delta"] = _delta_summary(source["delta"])
        for key in ("fastest_route", "safest_route", "balanced_route"):
            if key in source:
                values[key] = _route_summary(source[key])
        summary = f"compared routes `{handle}`" if handle else "compared routes"
    elif tool == "RenderMap":
        values = _pick(source, ["style", "layers", "image_path", "artifact", "exists"])
        summary = f"rendered map `{handle}`" if handle else "rendered map"
    else:
        values = source
        summary = summary or f"{tool} completed"

    values = _round_value(values)
    return {
        "summary": summary,
        "values": values,
    }


def make_geodisaster_state_fields(
    tool: str,
    *,
    handle: str | None,
    ok: bool,
    summary: dict[str, Any],
    full_result: Any | None = None,
) -> dict[str, Any]:
    """Fields to store in shared state and show in runtime prompts."""

    fields: dict[str, Any] = {
        "latest_tool": tool,
        "latest_tool_ok": bool(ok),
        "latest_summary": summary.get("summary") or "",
    }
    if handle:
        fields["latest_handle"] = handle
        fields[f"{handle}_summary"] = summary.get("summary") or ""
        values = summary.get("values")
        if values:
            fields[f"{handle}_evidence"] = values
        if ok and full_result is not None and tool not in {"ReadData", "PrepareScene"}:
            public_result = _public_runtime_payload(full_result)
            fields[f"{handle}_payload"] = _round_value(public_result)
            if isinstance(public_result, dict):
                fields[f"{handle}_payload_keys"] = list(public_result.keys())[:24]
            elif isinstance(public_result, list):
                fields[f"{handle}_payload_items"] = len(public_result)
    elif summary.get("values"):
        fields[f"{tool}_evidence"] = summary["values"]
    if not ok:
        fields["latest_error"] = summary.get("summary") or f"{tool} failed"
    return fields


def make_geodisaster_observation(tool: str, *, handle: str | None, ok: bool, summary: dict[str, Any]) -> dict[str, Any]:
    """Small ToolObservation row for review JSONs."""

    obs: dict[str, Any] = {
        "tool": tool,
        "status": "OK" if ok else "ERROR",
        "summary": summary.get("summary") or "",
    }
    if handle:
        obs["produced_handle"] = handle
    if summary.get("values"):
        obs["evidence"] = summary["values"]
    return obs
