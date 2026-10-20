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


SEN1_METHOD = "s1_min_backscatter_otsu_v1"



def _sen1_mask_report_from_work(state: dict[str, Any], work: str) -> dict[str, Any]:
    return prediction_report(state, work)


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "chip_id": report.get("chip_id"),
        "method": report.get("method"),
        "computed_flood_percentage": report.get("computed_flood_percentage"),
        "computed_flood_area_m2": report.get("computed_flood_area_m2"),
        "computed_flood_area_km2": report.get("computed_flood_area_km2"),
        "computed_flood_pixel_count": report.get("computed_flood_pixel_count"),
        "computed_valid_pixel_count": report.get("computed_valid_pixel_count"),
        "component_count": report.get("component_count"),
        "polygon_count": report.get("polygon_count"),
        "largest_component_area_m2": report.get("largest_component_area_m2"),
        "largest_component_centroid": report.get("largest_component_centroid"),
        "concentration_score": report.get("concentration_score"),
        "top_3_grid_cells": report.get("top_3_grid_cells"),
    }


def process_MapFlood(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "ourdata/sen1_hand_final_hidden_eval_5q":
        raise ValueError(f"MapFlood is not implemented for source {source_of(state)}")
    requested_method = str(args.get("method") or "")
    if requested_method != SEN1_METHOD:
        raise ValueError(
            f"unsupported Sentinel-1 flood method {requested_method!r}; expected {SEN1_METHOD!r}"
        )
    report = _sen1_mask_report_from_work(state, work)
    result = {
        "requested_method": requested_method,
        "method": report.get("method"),
        "mask_source": "generated_from_declared_sentinel1",
        "prediction_mask_generated": True,
        "mask_report": report,
        "mask_summary": _report_summary(report),
    }
    return store_result_handle(state, "flood", "flood_mask", TOOL_NAME, args, work, "sar_flood_raster_tools", result)


TOOL_NAME = "MapFlood"
DEFAULT_PORT = 20115
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class MapFloodWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `MapFlood` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_MapFlood, require_existing=True)

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

    worker = MapFloodWorker(
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
