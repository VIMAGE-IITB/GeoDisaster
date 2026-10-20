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
from tool_server.tool_workers.online_workers.sen1_prediction_runtime import prediction_report
from tool_server.tool_workers.online_workers import xbd_damage_runtime as xbd



def _sen1_mask_report_from_work(state: dict[str, Any], work: str) -> dict[str, Any]:
    return prediction_report(state, work)

def process_RankRegions(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    source = source_of(state)
    if source == "geodisaster":
        prior = artifact_result(state, work) or {}
        candidates = str(args.get("candidates") or "")
        grid = 2 if "quadrant" in candidates else 4
        contexts = xbd.scene_contexts(state, work)
        cells = prior.get("cells") or xbd.grid_rows(
            contexts[0], "quadrants" if grid == 2 else "grid_4x4"
        )
        key = str(args.get("by") or "")
        if grid == 2:
            if key in {"severe_footprint_area_m2", "severe_area_m2"} or "area" in key:
                rank_field = "severe_area_m2"
            elif key in {"severe_damage_count", "severe_count"} or "count" in key:
                rank_field = "severe_count"
            else:
                raise ValueError(f"unsupported 2x2 xBD ranking metric: {key!r}")
        else:
            if key not in {"normalized_severity", "severity_score", "severity_score_normalized"} and "severity" not in key:
                raise ValueError(f"unsupported 4x4 xBD ranking metric: {key!r}")
            rank_field = "severity_score"
        excluded_empty_cell_ids: list[str] = []
        eligible_cells = cells
        if grid == 4:
            # A normalized building-damage severity is undefined for a cell
            # containing no buildings.  Treating that synthetic zero as a
            # measured score can make an empty G00 win an all-zero tie.
            excluded_empty_cell_ids = sorted(
                str(cell.get("cell_id"))
                for cell in cells
                if xbd.severity_fraction(cell) is None
            )
            eligible_cells = [
                cell for cell in cells if xbd.severity_fraction(cell) is not None
            ]
        value = (
            (lambda cell: xbd.severity_fraction(cell))
            if grid == 4
            else (lambda cell: float(cell.get(rank_field) or 0))
        )
        ranked_all = sorted(eligible_cells, key=lambda c: (-value(c), str(c.get("cell_id") or "")))
        top_value = value(ranked_all[0]) if ranked_all else None
        tied_top = [
            str(c.get("cell_id"))
            for c in ranked_all
            if value(c) == top_value
        ] if ranked_all else []
        top_k = int(args.get("top_k") or 3)
        expected_candidates = "quadrants_2x2" if grid == 2 else "grid_4x4"
        if candidates != expected_candidates:
            raise ValueError(f"xBD ranking candidates must be {expected_candidates!r}")
        if top_k <= 0 or top_k > len(cells):
            raise ValueError(f"xBD top_k must be between 1 and {len(cells)}, got {top_k}")
        result = {
            "by": key,
            "rank_field": rank_field,
            "candidates": args.get("candidates"),
            "candidate_count": len(cells),
            "eligible_candidate_count": len(eligible_cells),
            "excluded_unscored_cell_ids": excluded_empty_cell_ids,
            "top_k": top_k,
            "ranked": ranked_all[:top_k],
            "tie_break_policy": "ascending_cell_id",
            "top_tied_cell_ids": tied_top,
            "top_tie": len(tied_top) > 1,
            "ranking_status": "ranked" if ranked_all else "no_classified_buildings",
        }
        if len(cells) <= 32:
            result["all_candidates"] = ranked_all
        if ranked_all and grid == 2:
            top = ranked_all[0]
            result.update({
                "highest_severe_area_quadrant": top.get("cell_id"),
                "severe_count": top.get("severe_count", 0),
                "severe_footprint_m2": top.get("severe_area_m2", 0.0),
                "severe_pct_in_quadrant": top.get("severe_pct", 0.0),
                "dominant_damage": top.get("dominant_damage"),
                "top_k_quadrants": ranked_all[:top_k],
                "all_quadrants": cells,
            })
        elif ranked_all:
            top = ranked_all[0]
            result.update({
                "highest_severity_cell": top.get("cell_id"),
                "severity_score": top.get("severity_score", 0.0),
                "building_count": top.get("building_count", 0),
                "severe_count": top.get("severe_count", 0),
                "class_histogram": {
                    "no_damage": int(top.get("no_damage") or 0),
                    "minor_damage": int(top.get("minor_damage") or 0),
                    "major_damage": int(top.get("major_damage") or 0),
                    "destroyed": int(top.get("destroyed") or 0),
                    "unclassified": int(top.get("unclassified") or 0),
                },
                "dominant_damage": top.get("dominant_damage"),
                "top_k_cells": ranked_all[:top_k],
            })
        elif grid == 4:
            result.update({
                "highest_severity_cell": None,
                "severity_score": None,
                "building_count": 0,
                "severe_count": 0,
                "class_histogram": {
                    "no_damage": 0,
                    "minor_damage": 0,
                    "major_damage": 0,
                    "destroyed": 0,
                    "unclassified": 0,
                },
                "dominant_damage": "none",
                "top_k_cells": [],
            })
        processor = "xdamage_vector_label_tools"
    elif source == "ourdata/sen1_hand_final_hidden_eval_5q":
        report = _sen1_mask_report_from_work(state, work)
        cells = report.get("grid_4x4") or []
        key = str(args.get("by") or "")
        candidates = str(args.get("candidates") or "")
        if candidates != "grid_4x4":
            raise ValueError(
                f"unsupported Sentinel-1 ranking candidates {candidates!r}; expected 'grid_4x4'"
            )
        if key == "flood_percentage":
            numerical_score = lambda c: (
                float(c.get("flood_percentage") or 0),
                int(c.get("flood_pixels") or 0),
            )
        elif key in {"flood_pixels", "flood_area"}:
            numerical_score = lambda c: (
                int(c.get("flood_pixels") or 0),
                float(c.get("flood_percentage") or 0),
            )
        else:
            raise ValueError(f"unsupported Sentinel-1 ranking metric: {key!r}")
        sorter = lambda c: (*numerical_score(c), str(c.get("cell_id") or ""))
        ranked_all = sorted(cells, key=sorter, reverse=True)
        top_score = numerical_score(ranked_all[0]) if ranked_all else None
        tied_top = [
            str(cell.get("cell_id"))
            for cell in ranked_all
            if numerical_score(cell) == top_score
        ] if ranked_all else []
        top_k = int(args.get("top_k") or 3)
        if top_k <= 0 or top_k > len(cells):
            raise ValueError(
                f"Sentinel-1 top_k must be between 1 and {len(cells)}, got {top_k}"
            )
        result = {
            "by": key,
            "candidates": candidates,
            "candidate_count": len(cells),
            "top_k": top_k,
            "ranked": ranked_all[:top_k],
            "tie_break_policy": "descending_cell_id",
            "top_tied_cell_ids": tied_top,
            "top_tie": len(tied_top) > 1,
        }
        if len(cells) <= 32:
            result["all_candidates"] = ranked_all
        processor = "sar_flood_raster_tools"
    else:
        raise ValueError(f"RankRegions is not implemented for source {source}")
    return store_result_handle(state, "rank", "region_ranking", TOOL_NAME, args, work, processor, result)


TOOL_NAME = "RankRegions"
DEFAULT_PORT = 20112
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class RankRegionsWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `RankRegions` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_RankRegions, require_existing=True)

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

    worker = RankRegionsWorker(
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
