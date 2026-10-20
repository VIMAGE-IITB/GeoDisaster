import argparse
import copy
import hashlib
import re
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import get_tool_instruction
from tool_server.tool_workers.online_workers.final_dataset_worker_state import (
    ROOT,
    all_lineage_path_entries,
    artifact_output_dir,
    artifact_result,
    collect_path_entries,
    compact_preview,
    deforestation_tools,
    dir_from_path_parts,
    first_existing_path,
    jsonable,
    load_json,
    mh_modules,
    new_handle,
    path_summary,
    require_artifact,
    require_question,
    resolve_data_path,
    routing_tools,
    run_stateful_tool,
    sen1_utils,
    source_of,
    store_result_handle,
)
from tool_server.utils.server_utils import build_logger
from tool_server.tool_workers.online_workers.sen1_prediction_runtime import prediction_report
from tool_server.tool_workers.online_workers import xbd_damage_runtime as xbd



def _deforestation_sample_dir(state: dict[str, Any], work: str) -> Path:
    for entry in all_lineage_path_entries(state, work):
        path_value = entry.get("path")
        if not path_value:
            continue
        base = dir_from_path_parts(Path(path_value), "samples_v2", 1)
        if base and base.exists():
            return base
    raise ValueError("could not resolve deforestation sample directory from input handles")


def _event_year_from_args(question_card: dict[str, Any], args: dict[str, Any]) -> int | None:
    """Resolve and cross-check the question and requested event year."""
    filters = args.get("filters")
    if not isinstance(filters, dict) or set(filters) != {"event_year"}:
        raise ValueError(
            "deforestation ExtractObjects filters must contain only event_year"
        )
    try:
        requested_year = int(filters["event_year"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid deforestation event_year: {filters.get('event_year')!r}"
        ) from exc

    text = str(question_card.get("text") or question_card.get("question") or "")
    match = re.search(r"\b(20\d{2})\b", text)
    if match:
        question_year = int(match.group(1))
        if requested_year != question_year:
            raise ValueError(
                f"event_year {requested_year} does not match question year {question_year}"
            )
        return question_year
    metadata = question_card.get("metadata") or {}
    year = metadata.get("event_year") or metadata.get("year")
    if year:
        question_year = int(year)
        if requested_year != question_year:
            raise ValueError(
                f"event_year {requested_year} does not match question year {question_year}"
            )
        return question_year
    return requested_year


def _question_context(state: dict[str, Any], question_card: dict[str, Any]) -> dict[str, Any]:
    """Merge the lightweight question handle with the full case question."""
    case_question = (state.get("case") or {}).get("question") or {}
    if not isinstance(case_question, dict):
        case_question = {}
    return {**case_question, **question_card}


def _deforestation_objects(
    sample_dir: Path,
    event_year: int | None,
    generated_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not generated_report or (
        event_year is not None
        and int(generated_report.get("event_year") or -1) != int(event_year)
    ):
        generated_report = deforestation_tools().BuildQualityReport(
            str(sample_dir), event_year
        )
    metrics = generated_report["loss_event_year"]
    return {
        "target": "forest_loss_patches",
        "event_year": generated_report["event_year"],
        "mask_kind": metrics["mask_kind"],
        "pixel_count": metrics["loss_pixel_count"],
        "total_area_ha": metrics["area_ha"],
        "area_m2": metrics["area_m2"],
        "component_count": metrics["component_count"],
        "largest_patch_pixels": metrics["largest_patch_pixels"],
        "mask_path": metrics["mask_path"],
        "measurement_method": metrics["measurement_method"],
        "fallback_used": metrics["fallback_used"],
        "crs": metrics["crs"],
        "shape": metrics["shape"],
        "generated_quality_report": generated_report,
    }


def _sen1_mask_report_from_work(state: dict[str, Any], work: str) -> dict[str, Any]:
    return prediction_report(state, work)


def _sen1_object_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_count": report.get("component_count"),
        "polygon_count": report.get("polygon_count"),
        "largest_component_area_m2": report.get("largest_component_area_m2"),
        "largest_component_centroid": report.get("largest_component_centroid"),
        "largest_component_pixel_count": report.get("largest_component_pixel_count"),
        "computed_flood_area_m2": report.get("computed_flood_area_m2"),
        "computed_flood_percentage": report.get("computed_flood_percentage"),
        "top_3_grid_cells": report.get("top_3_grid_cells"),
        "largest_component_bbox": report.get("largest_component_bbox"),
        "prediction_provenance": report.get("prediction_provenance"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_sen1_objects(
    state: dict[str, Any],
    report: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    import numpy as np
    import rasterio

    if target not in {"flood_components", "flood_polygons"}:
        raise ValueError(
            f"unsupported Sentinel-1 object target {target!r}; expected flood_components or flood_polygons"
        )
    mask_path = Path(str(report.get("prediction_mask_path") or report.get("mask_path") or ""))
    if not mask_path.is_file():
        raise ValueError(f"prediction mask is unavailable for object extraction: {mask_path}")
    utils = sen1_utils()
    output_dir = artifact_output_dir(state)
    with rasterio.open(mask_path) as dataset:
        values = dataset.read(1)
        mask = values == 1
        labels, component_count, _sizes = utils.connected_components(mask)
        if component_count != int(report.get("component_count") or -1):
            raise ValueError(
                f"component recomputation mismatch: {component_count} vs {report.get('component_count')}"
            )
        if target == "flood_components":
            artifact_path = output_dir / "sentinel1_connected_components.tif"
            profile = dataset.profile.copy()
            profile.update(count=1, dtype="int32", nodata=-1, compress="deflate")
            output = labels.astype("int32")
            if dataset.nodata is not None:
                output[values == dataset.nodata] = -1
            with rasterio.open(artifact_path, "w", **profile) as destination:
                destination.write(output, 1)
            object_count = component_count
            artifact_kind = "connected_component_label_raster"
            geometry_generated = False
            validation = {
                "connectivity": 8,
                "reported_component_count": int(report["component_count"]),
                "extracted_component_count": component_count,
                "background_value": 0,
                "nodata_value": -1,
            }
        else:
            artifact_path = output_dir / "sentinel1_flood_polygons.geojson"
            collection = utils.polygon_feature_collection(
                mask,
                dataset,
                connectivity=8,
            )
            for component_id, feature in enumerate(collection["features"], 1):
                feature.setdefault("properties", {})["component_id"] = component_id
            utils.write_json(artifact_path, collection)
            object_count = len(collection["features"])
            if object_count != component_count:
                raise ValueError(
                    f"8-neighbour polygon/component mismatch: {object_count} vs {component_count}"
                )
            artifact_kind = "flood_polygon_feature_collection"
            geometry_generated = True
            validation = {
                "connectivity": 8,
                "reported_polygon_count": int(report["polygon_count"]),
                "extracted_polygon_count": object_count,
                "geometry_type": "GeoJSON FeatureCollection",
                "crs": str(dataset.crs) if dataset.crs else None,
            }
    return {
        "object_artifact_path": str(artifact_path),
        "object_artifact_kind": artifact_kind,
        "object_artifact_sha256": _sha256(artifact_path),
        "object_artifact_bytes": artifact_path.stat().st_size,
        "object_count": object_count,
        "polygon_geometry_generated": geometry_generated,
        "component_raster_generated": target == "flood_components",
        "object_validation": validation,
    }


def _compact_xbd_feature(feat: dict[str, Any]) -> dict[str, Any]:
    props = feat.get("properties") or {}
    return {
        "subtype": props.get("subtype"),
        "area_m2": props.get("area_m2"),
        "centroid_x": props.get("centroid_x"),
        "centroid_y": props.get("centroid_y"),
        "damage_score": props.get("damage_score"),
        "is_severe": bool(props.get("is_severe")),
    }


def process_ExtractObjects(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    question = _question_context(state, require_question(state, args))
    work = str(args.get("work") or "")
    require_artifact(state, work)
    source = source_of(state)
    if source == "deforestation_data":
        if args.get("target") != "forest_loss_patches":
            raise ValueError(f"unsupported deforestation ExtractObjects target: {args.get('target')!r}")
        work_card = require_artifact(state, work)
        generated_reports = work_card.get("generated_quality_reports") or {}
        generated_report = generated_reports.get("forest") if isinstance(generated_reports, dict) else None
        result = _deforestation_objects(
            _deforestation_sample_dir(state, work),
            _event_year_from_args(question, args),
            generated_report,
        )
        processor = "deforestation_raster_tools"
    elif source == "geodisaster":
        if args.get("target") != "severe_buildings":
            raise ValueError(f"unsupported xBD ExtractObjects target: {args.get('target')!r}")
        if args.get("filters") != {"severity": "major_or_destroyed"}:
            raise ValueError(f"unsupported xBD ExtractObjects filters: {args.get('filters')!r}")
        contexts = xbd.scene_contexts(state, work)
        fc = xbd.load_damage_collection(contexts[0])
        target = str(args.get("target") or "")
        if "severe" in target:
            feats = [f for f in fc.get("features", []) if (f.get("properties") or {}).get("is_severe") or (f.get("properties") or {}).get("subtype") in {"major-damage", "destroyed"}]
        else:
            feats = list(fc.get("features", []))
        result = {
            "target": target,
            "object_count": len(feats),
            "total_area_m2": round(sum(float((f.get("properties") or {}).get("area_m2") or 0.0) for f in feats), 3),
            "sample_objects": [_compact_xbd_feature(feat) for feat in feats[:8]],
        }
        processor = "xdamage_vector_label_tools"
    elif source == "ourdata/sen1_hand_final_hidden_eval_5q":
        report = _sen1_mask_report_from_work(state, work)
        target = str(args.get("target") or "")
        if target not in {"flood_components", "flood_polygons"}:
            raise ValueError(f"unsupported Sentinel-1 ExtractObjects target: {target!r}")
        result = {
            "target": target,
            **_sen1_object_summary(report),
            **_materialize_sen1_objects(state, report, target),
        }
        processor = "sar_flood_raster_tools"
    else:
        raise ValueError(f"ExtractObjects is not implemented for source {source}")
    return store_result_handle(state, "objects", "object_extraction", TOOL_NAME, args, work, processor, result)


TOOL_NAME = "ExtractObjects"
DEFAULT_PORT = 20114
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class ExtractObjectsWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `ExtractObjects` tool."""

    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name=TOOL_NAME,
        host="0.0.0.0",
        port=None,
        limit_model_concurrency=8,
        model_semaphore=None,
        wait_timeout=120.0,
        task_timeout=120.0,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            model_name,
            False,
            False,
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info("%s worker is ready.", TOOL_NAME)

    def generate(self, params):
        payload = dict(params or {})
        return run_stateful_tool(TOOL_NAME, payload, process_ExtractObjects, require_existing=True)

    def get_tool_instruction(self):
        return get_tool_instruction(TOOL_NAME)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default=TOOL_NAME)
    parser.add_argument("--limit-model-concurrency", type=int, default=8)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = ExtractObjectsWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()


if __name__ == "__main__":
    main()
