"""Shared, exact-path xBD damage computations used by online workers."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image

from tool_server.tool_workers.online_workers.final_dataset_worker_state import (
    authorize_declared_read,
    collect_path_entries,
    initial_cards_for,
    load_json,
    resolve_data_path,
)


XBD_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed", "unclassified"]
CLASSIFIED_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SEVERE_CLASSES = {"major-damage", "destroyed"}
CLASS_SCORE = {"unclassified": -1, "no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}


def subtype(props: dict[str, Any]) -> str:
    value = str(props.get("subtype") or "unclassified")
    return "unclassified" if value in {"un-classified", "unclassified"} else value


def _path_key(entry: dict[str, Any], path: Path) -> str:
    label = str(entry.get("label") or "").lower()
    text = f"{label} {path.name.lower()}"
    if "building_damage_xy" in text or "damage_polygons_xy" in text:
        return "damage_polygons_xy"
    if "building_damage_lnglat" in text or "damage_polygons_geo" in text:
        return "damage_polygons_geo"
    if "grid_4x4" in text:
        return "grid_4x4"
    if "quadrant" in text:
        return "quadrants"
    if "post_target" in text:
        return "post_target_mask"
    if "post" in text and path.suffix.lower() == ".png":
        return "post_image"
    if "pre" in text and path.suffix.lower() == ".png":
        return "pre_image"
    return str(entry.get("label") or path.name)


def scene_contexts(state: dict[str, Any], handle: str) -> list[dict[str, Any]]:
    """Resolve scene inputs only from exact paths declared on opened lineage."""
    contexts: dict[str, dict[str, Any]] = {}
    for card in initial_cards_for(state, handle):
        metadata = dict(card.get("metadata") or {})
        for entry in collect_path_entries(card.get("paths", card)):
            path_value = entry.get("path")
            if not path_value:
                continue
            path = resolve_data_path(path_value).resolve()
            if "xbd_damage" not in path.parts:
                continue
            marker = path.parts.index("xbd_damage")
            if len(path.parts) <= marker + 1:
                continue
            scene_id = path.parts[marker + 1]
            context = contexts.setdefault(
                scene_id,
                {"scene_id": scene_id, "paths": {}, "metadata": {}},
            )
            context["metadata"].update(metadata)
            context["paths"][_path_key(entry, path)] = {
                **dict(entry),
                "path": path,
            }
    if not contexts:
        raise ValueError("opened handle lineage contains no declared xBD scene paths")
    return list(contexts.values())


def required_path(context: dict[str, Any], key: str) -> Path:
    entry = (context.get("paths") or {}).get(key)
    if not entry:
        raise ValueError(f"{context.get('scene_id')}: opened lineage does not declare required {key}")
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{context.get('scene_id')}: missing required {key}: {path}")
    return path


def optional_path(context: dict[str, Any], key: str) -> Path | None:
    entry = (context.get("paths") or {}).get(key)
    if not entry:
        return None
    path = Path(entry["path"])
    return path if path.is_file() else None


def load_damage_collection(context: dict[str, Any]) -> dict[str, Any]:
    return load_json(required_path(context, "damage_polygons_xy"))


def grid_rows(context: dict[str, Any], grid_key: str) -> list[dict[str, Any]]:
    keep = {
        "cell_id", "row", "col", "building_count", "classified_building_count",
        "severe_count", "severe_pct", "severe_area_m2", "total_area_m2",
        "severity_score", "dominant_damage", "no_damage", "minor_damage",
        "major_damage", "destroyed", "unclassified",
    }
    collection = load_json(required_path(context, grid_key))
    return [
        {key: value for key, value in (feature.get("properties") or {}).items() if key in keep}
        for feature in collection.get("features") or []
    ]


def classified_count(row: dict[str, Any]) -> int:
    if row.get("classified_building_count") is not None:
        return int(row["classified_building_count"])
    return sum(int(row.get(key) or 0) for key in ("no_damage", "minor_damage", "major_damage", "destroyed"))


def severity_fraction(row: dict[str, Any]) -> Fraction | None:
    denominator = classified_count(row)
    if denominator <= 0:
        return None
    numerator = (
        int(row.get("minor_damage") or 0)
        + 2 * int(row.get("major_damage") or 0)
        + 3 * int(row.get("destroyed") or 0)
    )
    return Fraction(numerator, 3 * denominator)


def damage_summary(context: dict[str, Any]) -> dict[str, Any]:
    collection = load_damage_collection(context)
    props = [feature.get("properties") or {} for feature in collection.get("features") or []]
    counts = Counter(subtype(item) for item in props)
    classified = sum(int(counts.get(name, 0)) for name in CLASSIFIED_CLASSES)
    damaged = sum(int(counts.get(name, 0)) for name in ("minor-damage", "major-damage", "destroyed"))
    severe = [item for item in props if subtype(item) in SEVERE_CLASSES]
    total_area = sum(float(item.get("area_m2") or 0.0) for item in props)
    severe_area = sum(float(item.get("area_m2") or 0.0) for item in severe)
    scene_id = str(context["scene_id"])
    metadata = context.get("metadata") or {}
    disaster = str(metadata.get("disaster") or scene_id.rsplit("_", 1)[0])
    disaster_type = metadata.get("disaster_type")
    if not disaster_type:
        raise ValueError(f"{scene_id}: declared scene metadata lacks disaster_type")
    return {
        "scene_id": scene_id,
        "disaster": disaster,
        "disaster_type": str(disaster_type),
        "gsd": collection.get("gsd_m_per_px"),
        "features": list(collection.get("features") or []),
        "damage_distribution": {
            "total_buildings": len(props),
            "classified_buildings": classified,
            "counts": {name: int(counts.get(name, 0)) for name in XBD_CLASSES},
            "damaged_count": damaged,
            "severe_count": len(severe),
            "damaged_pct": round(100.0 * damaged / classified, 2) if classified else 0.0,
            "severe_pct": round(100.0 * len(severe) / classified, 2) if classified else 0.0,
        },
        "area_stats": {
            "total_footprint_m2": round(total_area, 4),
            "severe_footprint_m2": round(severe_area, 4),
            "major_damage_area_m2": round(
                sum(float(item.get("area_m2") or 0.0) for item in props if subtype(item) == "major-damage"),
                4,
            ),
            "destroyed_area_m2": round(
                sum(float(item.get("area_m2") or 0.0) for item in props if subtype(item) == "destroyed"),
                4,
            ),
            "area_method": "xy_polygon_area_times_gsd_squared",
        },
    }


def spatial_summary(quadrants: list[dict[str, Any]]) -> dict[str, Any]:
    per_quadrant = {str(row["cell_id"]): int(row.get("severe_count") or 0) for row in quadrants}
    total = sum(per_quadrant.values())
    maximum = max(per_quadrant.values(), default=0)
    concentration = round(maximum / total, 4) if total else None
    ties = sorted(key for key, value in per_quadrant.items() if value == maximum) if total else []
    if not total:
        assessment = "no severe damage"
    elif concentration is not None and concentration >= 0.70:
        assessment = "highly concentrated"
    elif concentration is not None and concentration >= 0.45:
        assessment = "moderately concentrated"
    else:
        assessment = "spatially dispersed"
    return {
        "total_severe": total,
        "concentration_score": concentration,
        "assessment": assessment,
        "worst_quadrant": ties[0] if ties else None,
        "top_tied_quadrant_ids": ties,
        "per_quadrant_severe": per_quadrant,
    }


def target_mask_summary(context: dict[str, Any], damage_counts: dict[str, int]) -> dict[str, Any]:
    path = required_path(context, "post_target_mask")
    authorize_declared_read(path, "image_read")
    with Image.open(path) as image:
        if image.size != (1024, 1024):
            raise ValueError(f"{context.get('scene_id')}: target mask is not 1024x1024")
        colors = image.getcolors(maxcolors=image.width * image.height + 1) or []
    raw = {str(int(value)): int(count) for count, value in colors}
    invalid = sorted(int(value) for value in raw if int(value) not in range(5))
    if invalid:
        raise ValueError(f"{context.get('scene_id')}: unsupported target-mask values {invalid}")
    histogram = {str(index): int(raw.get(str(index), 0)) for index in range(5)}
    non_background = sum(histogram[str(index)] for index in range(1, 5))
    class_to_code = {"no-damage": "1", "minor-damage": "2", "major-damage": "3", "destroyed": "4"}
    presence = {
        name: bool(int(damage_counts.get(name, 0))) == bool(histogram[code])
        for name, code in class_to_code.items()
    }
    return {
        "available": True,
        "encoding": "0=background,1=no-damage,2=minor-damage,3=major-damage,4=destroyed",
        "unclassified_policy": "excluded_from_target_mask",
        "total_pixels": sum(histogram.values()),
        "non_background_pixels": non_background,
        "histogram": histogram,
        "histogram_pct_non_bg": {
            str(index): (
                0.0
                if index == 0 or not non_background
                else round(100.0 * histogram[str(index)] / non_background, 4)
            )
            for index in range(5)
        },
        "mask_severe_pixels": histogram["3"] + histogram["4"],
        "class_presence_checks": presence,
        "class_presence_consistent": all(presence.values()),
    }


def scene_report(
    context: dict[str, Any], *, include_spatial: bool = False, include_mask: bool = False
) -> dict[str, Any]:
    report = damage_summary(context)
    if include_spatial:
        quadrants = grid_rows(context, "quadrants")
        report["quadrant_analysis"] = {"quadrant_table": quadrants}
        report["spatial_dispersion"] = spatial_summary(quadrants)
    if include_mask:
        report["target_mask"] = target_mask_summary(
            context, report["damage_distribution"]["counts"]
        )
    return report


def scene_answer_evidence(report: dict[str, Any]) -> dict[str, Any]:
    distribution = report["damage_distribution"]
    area = report["area_stats"]
    result = {
        "scene_id": report["scene_id"],
        "disaster": report["disaster"],
        "disaster_type": report["disaster_type"],
        "gsd": report["gsd"],
        "damage_distribution": distribution["counts"],
        "total_buildings": distribution["total_buildings"],
        "classified_buildings": distribution["classified_buildings"],
        "unclassified_buildings": distribution["counts"]["unclassified"],
        "damaged_count": distribution["damaged_count"],
        "severe_count": distribution["severe_count"],
        "damaged_pct": distribution["damaged_pct"],
        "severe_pct": distribution["severe_pct"],
        "severe_pct_denominator": "classified_buildings",
        "severe_pct_denominator_count": distribution["classified_buildings"],
        "severe_pct_status": "scored" if distribution["classified_buildings"] else "no_classified_buildings",
        "severe_footprint_m2": area["severe_footprint_m2"],
        "total_footprint_m2": area["total_footprint_m2"],
        "area_method": area["area_method"],
    }
    if report.get("spatial_dispersion") is not None:
        spatial = report["spatial_dispersion"]
        result["spatial_analysis"] = {
            "dispersion_assessment": spatial["assessment"],
            "concentration_score": spatial["concentration_score"],
            "worst_quadrant": spatial["worst_quadrant"],
            "top_tied_quadrant_ids": spatial["top_tied_quadrant_ids"],
        }
    if report.get("target_mask") is not None:
        result["target_mask"] = report["target_mask"]
    return result


def comprehensive_answer(report: dict[str, Any]) -> dict[str, Any]:
    """Return the publication-facing XBD-Q8 object from computed evidence."""
    distribution = report["damage_distribution"]
    area = report["area_stats"]
    spatial = report["spatial_dispersion"]
    target = report["target_mask"]
    return {
        "disaster": report["disaster"],
        "disaster_type": report["disaster_type"],
        "gsd": report["gsd"],
        "damage_summary": {
            "total_buildings": distribution["total_buildings"],
            "classified_buildings": distribution["classified_buildings"],
            "counts": distribution["counts"],
            "severe_count": distribution["severe_count"],
            "severe_pct": distribution["severe_pct"],
            "damaged_pct": distribution["damaged_pct"],
        },
        "area_summary": {
            "total_footprint_m2": area["total_footprint_m2"],
            "severe_footprint_m2": area["severe_footprint_m2"],
            "area_method": area["area_method"],
        },
        "spatial_analysis": {
            "dispersion_assessment": spatial["assessment"],
            "concentration_score": spatial["concentration_score"],
            "worst_quadrant": spatial["worst_quadrant"],
            "top_tied_quadrant_ids": spatial["top_tied_quadrant_ids"],
        },
        "mask_validation": {
            "mask_available": target["available"],
            "encoding": target["encoding"],
            "unclassified_policy": target["unclassified_policy"],
            "mask_severe_pixels": target["mask_severe_pixels"],
            "polygon_severe_count": distribution["severe_count"],
            "pixel_histogram": target["histogram"],
            "pixel_pct_non_background": target["histogram_pct_non_bg"],
            "class_presence_checks": target["class_presence_checks"],
            "class_presence_consistent": target["class_presence_consistent"],
        },
    }
