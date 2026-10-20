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


def process_ReadData(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    input_handle = args.get("input")
    if not input_handle:
        raise ValueError("ReadData requires input")
    options = args.get("options") if isinstance(args.get("options"), dict) else {}
    card = require_artifact(state, str(input_handle))
    allowed_roles = {
        "pre_post", "hansen", "ndvi_nbr", "vh", "wildfire", "deforestation",
        "urban_expansion", "drought", "air_pollution", "hospitals_and_schools",
        "highways", "nearest_highway", "population_in_footprint", "schools", "population",
        "post_disaster", "buildings", "target_classes", "scene_a", "scene_b",
        "roads_shelters_flood", "roads_shelters_flood_rgb", "primary",
        "known_water_reference",
    }
    role_value = args.get("role") or options.get("role")
    time_value = args.get("time") or options.get("time")
    if role_value not in allowed_roles:
        raise ValueError(f"unsupported ReadData role: {role_value!r}")
    if time_value is not None and time_value not in {"event", "event_window", "pre_post_epochs", "2001_2023"}:
        raise ValueError(f"unsupported ReadData time: {time_value!r}")
    if set(options) - {"role", "time", "sources"} or ("sources" in options and options["sources"] != "all"):
        raise ValueError(f"unsupported ReadData options: {options!r}")
    summary = path_summary(card)
    role = args.get("role") or options.get("role") or card.get("role")
    time = args.get("time") or options.get("time")
    generated_quality_report = None
    generated_quality_report_handle = None
    declared_names = {
        Path(str(entry.get("path") or "")).name
        for entry in collect_path_entries(card.get("paths", card))
    }
    has_annual_loss_input = any(
        name.startswith("loss_year_") and name.endswith(".tif")
        for name in declared_names
    )
    if (
        source_of(state) == "deforestation_data"
        and card.get("type") == "forest_mask"
        and has_annual_loss_input
    ):
        sample_dir = None
        for entry in all_lineage_path_entries(state, str(input_handle)):
            path_value = entry.get("path")
            if not path_value:
                continue
            sample_dir = dir_from_path_parts(Path(path_value), "samples_v2", 1)
            if sample_dir and sample_dir.exists():
                break
        if not sample_dir:
            raise ValueError("could not resolve deforestation sample directory from forest mask")
        generated_quality_report = jsonable(
            deforestation_tools().BuildQualityReport(str(sample_dir))
        )
        generated_quality_report_handle = new_handle(
            state,
            "quality",
            {
                "type": "generated_quality_report",
                "source_input": input_handle,
                "processor": "declared_hansen_raster_quality_report_v1",
                "result": generated_quality_report,
            },
        )
    handle = new_handle(
        state,
        "scene",
        {
            "type": "scene",
            "source_input": input_handle,
            "source_type": card.get("type"),
            "role": role,
            "time": time,
            "metadata": copy.deepcopy(card.get("metadata") or {}),
            "path_summary": summary,
            "generated_quality_report_handle": generated_quality_report_handle,
            "generated_quality_report": generated_quality_report,
        },
    )
    status = "OK" if summary["missing_required_path_count"] == 0 else "MISSING_REQUIRED_PATHS"
    artifact = state["artifacts"][handle]
    return {
        "status": status,
        "session_id": state["session_id"],
        "handle": handle,
        "canonical_handle": handle,
        "global_artifact_id": artifact.get("global_artifact_id"),
        "artifact_json_path": artifact.get("artifact_json_path"),
        "source_input": input_handle,
        "source_type": card.get("type"),
        "role": role,
        "path_summary": summary,
        "generated_quality_report_handle": generated_quality_report_handle,
        "generated_quality_report": compact_preview(generated_quality_report, 8000)
        if generated_quality_report is not None
        else None,
    }


TOOL_NAME = "ReadData"
DEFAULT_PORT = 20102
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class ReadDataWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `ReadData` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_ReadData, require_existing=False)

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

    worker = ReadDataWorker(
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
