import argparse
import copy
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import get_tool_instruction
from tool_server.tool_workers.online_workers.final_dataset_worker_state import (
    ROOT,
    all_lineage_path_entries,
    artifact_result,
    collect_path_entries,
    declared_sar_windows,
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



def _deforestation_sample_dir(state: dict[str, Any], work: str) -> Path:
    for entry in all_lineage_path_entries(state, work):
        path_value = entry.get("path")
        if not path_value:
            continue
        base = dir_from_path_parts(Path(path_value), "samples_v2", 1)
        if base and base.exists():
            return base
    raise ValueError("could not resolve deforestation sample directory from input handles")


def _deforestation_bundle(
    state: dict[str, Any], work: str, sample_dir_text: str
) -> dict[str, Any]:
    tools = deforestation_tools()
    optical = tools.GetOpticalEvidence(sample_dir_text, indices=("NDVI", "NBR"), epochs="all", restrict_to_loss_mask=True)
    sar = tools.GetSAREvidence(
        sample_dir_text, s1_windows=declared_sar_windows(state, work)
    )
    mask = tools.GetForestMask(sample_dir_text)
    change = tools.DetectChangeWindow(sample_dir_text)
    classification = tools.ClassifyChangeType(optical, sar, mask)
    return jsonable({"optical": optical, "sar": sar, "mask": mask, "change": change, "classification": classification})


def process_DetectChange(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "deforestation_data":
        raise ValueError(f"DetectChange is not implemented for source {source_of(state)}")
    requested_change = str(args.get("change") or "")
    expected_window = {
        "largest_loss_year": "2001_2023",
        "dominant_clearing_interval": "pre_post_epochs",
    }.get(requested_change)
    if expected_window is None:
        raise ValueError(f"unsupported deforestation DetectChange mode: {requested_change!r}")
    if args.get("window") != expected_window:
        raise ValueError(f"unsupported DetectChange window: {args.get('window')!r}")
    if requested_change == "largest_loss_year":
        stats = artifact_result(state, work) or {}
        metrics = stats.get("metrics") or {}
        if not metrics.get("forest_loss_history") or set(metrics) != {
            "forest_loss_history", "loss_area_ha_by_year"
        }:
            raise ValueError(
                f"work {work!r} is not a 2001-2023 forest-loss-history ComputeStats result"
            )
        if stats.get("measurement_contract") != "hansen_grid_area_by_year_v1":
            raise ValueError("forest-loss history lacks the declared grid-area contract")
        if stats.get("analysis_window") != {"start_year": 2001, "end_year": 2023}:
            raise ValueError("forest-loss history has an invalid analysis window")
        histogram = stats.get("loss_year_histogram_ha") or {}
        if not isinstance(histogram, dict):
            raise ValueError("loss_year_histogram_ha must be an object")
        clean_histogram = {}
        for raw_year, raw_area in histogram.items():
            try:
                year = int(raw_year)
                area = float(raw_area)
            except (TypeError, ValueError) as exc:
                raise ValueError("loss history requires numeric year/area entries") from exc
            if str(year) != str(raw_year) or not 2001 <= year <= 2023:
                raise ValueError(f"loss-history year is outside 2001-2023: {raw_year!r}")
            if not math.isfinite(area) or area <= 0:
                raise ValueError(f"loss-history area must be finite and positive: {raw_area!r}")
            clean_histogram[year] = round(area, 6)
        if clean_histogram:
            year_key = min(
                clean_histogram,
                key=lambda year: (-clean_histogram[year], year),
            )
            result = {
                "change": "largest_loss_year",
                "window": args.get("window"),
                "year": int(year_key),
                "loss_ha": clean_histogram[year_key],
                "loss_area_ha_by_year": clean_histogram,
                "tie_break_rule": "earliest_year",
                "measurement_contract": "hansen_grid_area_by_year_v1",
            }
        else:
            result = {
                "change": "largest_loss_year",
                "window": args.get("window"),
                "year": "no_loss",
                "loss_ha": 0.0,
                "loss_area_ha_by_year": {},
                "tie_break_rule": "earliest_year",
                "measurement_contract": "hansen_grid_area_by_year_v1",
            }
        return store_result_handle(
            state,
            "change",
            "change_detection",
            TOOL_NAME,
            args,
            work,
            "deforestation_raster_tools",
            result,
        )
    sample_dir = _deforestation_sample_dir(state, work)
    bundle = _deforestation_bundle(state, work, str(sample_dir))
    result = copy.deepcopy(bundle["change"])
    if requested_change == "dominant_clearing_interval":
        mask = bundle["mask"]
        optical = bundle["optical"]
        sar = bundle["sar"]
        result = deforestation_tools().ApplyChangeWindowContract(result, mask)
        loss_area_ha = result["change_window_contract"]["observed_loss_area_ha"]
        result.update(
            {
                "change": "dominant_clearing_interval",
                "evidence": {
                    "ndvi_drop": optical.get("indices", {}).get("NDVI", {}).get("drop"),
                    "sar_vh_delta_db": sar.get("polarizations", {}).get("VH", {}).get("delta_db"),
                    "hansen_loss_area_ha": loss_area_ha,
                    "selected_window_ndvi_drop": (
                        result.get("selected_window_evidence") or {}
                    ).get("ndvi_drop"),
                    "selected_window_nbr_drop": (
                        result.get("selected_window_evidence") or {}
                    ).get("nbr_drop"),
                    "composite_drop_score": (
                        result.get("selected_window_evidence") or {}
                    ).get("composite_drop_score"),
                },
            }
        )
    return store_result_handle(state, "change", "change_detection", TOOL_NAME, args, work, "deforestation_raster_tools", result)


TOOL_NAME = "DetectChange"
DEFAULT_PORT = 20107
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class DetectChangeWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `DetectChange` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_DetectChange, require_existing=True)

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

    worker = DetectChangeWorker(
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
