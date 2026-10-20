import argparse
import copy
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
from tool_server.tool_workers.online_workers import xbd_damage_runtime as xbd


XBD_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed", "unclassified"]


def _compare_scene_metric(
    summaries: list[dict[str, Any]], metric: str, *, require_classified: bool = False
) -> tuple[str | None, list[str], str]:
    eligible = [
        row
        for row in summaries
        if not require_classified or int(row.get("classified_buildings") or 0) > 0
    ]
    if not eligible:
        return None, [], "not_comparable"
    maximum = max(float(row.get(metric) or 0.0) for row in eligible)
    tied = sorted(
        str(row.get("scene_id"))
        for row in eligible
        if float(row.get(metric) or 0.0) == maximum
    )
    return (tied[0] if len(tied) == 1 else None), tied, ("ranked" if len(tied) == 1 else "tie")


def _compact_damage_cell(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["subtype_histogram"] = {
        "no_damage": int(row.get("no_damage") or 0),
        "minor_damage": int(row.get("minor_damage") or 0),
        "major_damage": int(row.get("major_damage") or 0),
        "destroyed": int(row.get("destroyed") or 0),
        "unclassified": int(row.get("unclassified") or 0),
    }
    return result


def _compact_feature(feat: dict[str, Any]) -> dict[str, Any]:
    props = feat.get("properties") or {}
    return {
        "subtype": props.get("subtype"),
        "area_m2": props.get("area_m2"),
        "centroid_x": props.get("centroid_x"),
        "centroid_y": props.get("centroid_y"),
        "damage_score": props.get("damage_score"),
    }


def process_AssessDamage(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "geodisaster":
        raise ValueError(f"AssessDamage is not implemented for source {source_of(state)}")
    contexts = xbd.scene_contexts(state, work)
    aggregation = str(args.get("aggregation") or "")
    allowed_severity = {
        "scene": {None},
        "quadrant_2x2": {"severe", "major_or_destroyed"},
        "grid_4x4": {None},
        "severe_only": {"major_or_destroyed"},
        "pair_compare": {"severe"},
    }
    if aggregation not in allowed_severity:
        raise ValueError(f"unsupported AssessDamage aggregation: {aggregation!r}")
    if args.get("severity") not in allowed_severity[aggregation]:
        raise ValueError(f"unsupported AssessDamage severity: {args.get('severity')!r}")
    if aggregation == "pair_compare":
        summaries = [xbd.scene_answer_evidence(xbd.scene_report(c)) for c in contexts[:2]]
        pct_winner, pct_top, pct_status = _compare_scene_metric(
            summaries, "severe_pct", require_classified=True
        )
        area_winner, area_top, area_status = _compare_scene_metric(
            summaries, "severe_footprint_m2"
        )
        result = {
            "aggregation": aggregation,
            "scenes": summaries,
            "greater_damage_scene": pct_winner,
            "higher_severe_pct_scene": pct_winner,
            "higher_severe_area_scene": area_winner,
            "top_severe_pct_scene_ids": pct_top,
            "top_severe_area_scene_ids": area_top,
            "severe_pct_comparison_status": pct_status,
            "severe_area_comparison_status": area_status,
            "criterion": args.get("severity"),
        }
    elif aggregation in {"quadrant_2x2", "grid_4x4"}:
        grid_key = "quadrants" if aggregation == "quadrant_2x2" else "grid_4x4"
        cells = [_compact_damage_cell(cell) for cell in xbd.grid_rows(contexts[0], grid_key)]
        if aggregation == "quadrant_2x2":
            top_cells = sorted(cells, key=lambda c: (-float(c.get("severe_area_m2") or 0), str(c.get("cell_id") or "")))
        else:
            top_cells = sorted(
                [cell for cell in cells if xbd.severity_fraction(cell) is not None],
                key=lambda c: (-xbd.severity_fraction(c), str(c.get("cell_id") or "")),
            )
        result = {
            "aggregation": aggregation,
            "cell_count": len(cells),
            "cells": cells,
            "top_severe_cells": top_cells[: int(args.get("top_k") or 4)],
        }
        if aggregation == "grid_4x4":
            result.update({
                "eligible_scored_cell_count": len(top_cells),
                "excluded_empty_cell_ids": sorted(
                    str(cell.get("cell_id"))
                    for cell in cells
                    if xbd.severity_fraction(cell) is None
                ),
                "ranking_status": "ranked" if top_cells else "no_classified_buildings",
            })
    elif aggregation == "severe_only":
        fc = xbd.load_damage_collection(contexts[0])
        feats = [f for f in fc.get("features", []) if (f.get("properties") or {}).get("subtype") in {"major-damage", "destroyed"} or (f.get("properties") or {}).get("is_severe")]
        major_area = sum(
            float((feat.get("properties") or {}).get("area_m2") or 0.0)
            for feat in feats
            if xbd.subtype(feat.get("properties") or {}) == "major-damage"
        )
        destroyed_area = sum(
            float((feat.get("properties") or {}).get("area_m2") or 0.0)
            for feat in feats
            if xbd.subtype(feat.get("properties") or {}) == "destroyed"
        )
        result = {
            "aggregation": aggregation,
            "filter": "subtype in [major-damage, destroyed]",
            "polygon_source": "building_damage_xy.geojson",
            "severe_count": len(feats),
            "severe_building_count": len(feats),
            "severe_footprint_m2": round(sum(float((f.get("properties") or {}).get("area_m2") or 0.0) for f in feats), 4),
            "major_damage_area_m2": round(major_area, 4),
            "destroyed_area_m2": round(destroyed_area, 4),
            "area_method": "xy_polygon_area_times_gsd_squared",
            "sample_severe_buildings": [_compact_feature(feat) for feat in feats[:8]],
        }
    elif aggregation == "scene":
        include_detail = xbd.optional_path(contexts[0], "post_target_mask") is not None
        report = xbd.scene_report(
            contexts[0], include_spatial=include_detail, include_mask=include_detail
        )
        evidence = (
            xbd.comprehensive_answer(report)
            if include_detail
            else xbd.scene_answer_evidence(report)
        )
        result = {"aggregation": "scene", **evidence}
    return store_result_handle(state, "damage", "damage_assessment", TOOL_NAME, args, work, "xdamage_vector_label_tools", result)


TOOL_NAME = "AssessDamage"
DEFAULT_PORT = 20111
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class AssessDamageWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `AssessDamage` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_AssessDamage, require_existing=True)

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

    worker = AssessDamageWorker(
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
