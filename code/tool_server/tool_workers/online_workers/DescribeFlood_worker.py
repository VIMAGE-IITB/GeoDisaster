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
    routing_tools,
    run_stateful_tool,
    sen1_utils,
    source_of,
    store_result_handle,
)
from tool_server.utils.server_utils import build_logger
from tool_server.tool_workers.online_workers.sen1_prediction_runtime import prediction_report



def _sen1_mask_report_from_work(state: dict[str, Any], work: str) -> dict[str, Any]:
    return prediction_report(state, work)


SUPPORTED_SEN1_METRICS = {
    "percentage",
    "area",
    "components",
    "grid",
    "polygon_count",
    "total_area",
    "largest_polygon",
    "polygons",
    "concentration",
    "spatial_pattern",
    "known_water_overlap",
    "outside_reference",
}


def _select_report_fields(
    report: dict[str, Any],
    metrics: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    out = {
        "chip_id": report.get("chip_id"),
        "method": report.get("method"),
        "prediction_threshold": report.get("prediction_threshold"),
        "prediction_provenance": report.get("prediction_provenance"),
    }
    if metrics.get("percentage"):
        for key in (
            "computed_flood_percentage",
            "computed_flood_pixel_count",
            "computed_valid_pixel_count",
        ):
            out[key] = report.get(key)
    if metrics.get("area") or metrics.get("total_area"):
        for key in ("computed_flood_area_m2", "computed_flood_area_km2", "area_status"):
            out[key] = report.get(key)
    if metrics.get("components"):
        for key in (
            "component_count",
            "largest_component_pixel_count",
            "largest_component_area_m2",
            "largest_component_centroid",
            "largest_component_bbox",
        ):
            out[key] = report.get(key)
    if any(metrics.get(key) for key in ("polygon_count", "largest_polygon", "polygons")):
        for key in (
            "polygon_count",
            "largest_component_pixel_count",
            "largest_component_area_m2",
            "largest_component_centroid",
            "largest_component_bbox",
        ):
            out[key] = report.get(key)
        for key in (
            "object_artifact_path",
            "object_artifact_kind",
            "object_artifact_sha256",
            "object_artifact_bytes",
            "polygon_geometry_generated",
            "object_validation",
        ):
            if key in prior:
                out[key] = prior[key]
    if metrics.get("grid"):
        out["grid_4x4"] = report.get("grid_4x4")
        out["top_3_grid_cells"] = report.get("top_3_grid_cells")
        out["concentration_score"] = report.get("concentration_score")
        out["spatial_pattern"] = report.get("spatial_pattern")
    if metrics.get("concentration"):
        out["concentration_score"] = report.get("concentration_score")
    if metrics.get("spatial_pattern"):
        out["spatial_pattern"] = report.get("spatial_pattern")
    if metrics.get("known_water_overlap") or metrics.get("outside_reference"):
        for key in ("overlap_pixels", "outside_reference_pixels", "overlap_pct_of_flood", "outside_reference_pct_of_flood"):
            if key in prior:
                out[key] = prior.get(key)
            elif key in report:
                out[key] = report.get(key)
    return out


def process_DescribeFlood(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "ourdata/sen1_hand_final_hidden_eval_5q":
        raise ValueError(f"DescribeFlood is not implemented for source {source_of(state)}")
    report = _sen1_mask_report_from_work(state, work)
    metrics = args.get("metrics", {}) or {}
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("DescribeFlood requires a non-empty metrics object")
    unknown = sorted(set(metrics) - SUPPORTED_SEN1_METRICS)
    if unknown:
        raise ValueError(f"unsupported Sentinel-1 flood metrics: {unknown}")
    disabled = sorted(key for key, value in metrics.items() if value is False)
    if disabled:
        raise ValueError(
            f"omit disabled Sentinel-1 metrics instead of passing false: {disabled}"
        )
    prior = artifact_result(state, work) or {}
    result = {"metrics": metrics, **_select_report_fields(report, metrics, prior)}
    return store_result_handle(state, "flood_report", "flood_report", TOOL_NAME, args, work, "sar_flood_raster_tools", result)


TOOL_NAME = "DescribeFlood"
DEFAULT_PORT = 20116
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class DescribeFloodWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `DescribeFlood` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_DescribeFlood, require_existing=True)

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

    worker = DescribeFloodWorker(
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
