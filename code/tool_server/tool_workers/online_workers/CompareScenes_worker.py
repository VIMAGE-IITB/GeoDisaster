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


def process_CompareScenes(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "geodisaster":
        raise ValueError(f"CompareScenes is not implemented for source {source_of(state)}")
    if args.get("by") != "severe_damage_ratio_and_footprint":
        raise ValueError(f"unsupported CompareScenes metric: {args.get('by')!r}")
    prior = artifact_result(state, work) or {}
    summaries = prior.get("scenes") or [
        xbd.scene_answer_evidence(xbd.scene_report(context))
        for context in xbd.scene_contexts(state, work)[:2]
    ]
    pct_winner, pct_top, pct_status = _compare_scene_metric(
        summaries, "severe_pct", require_classified=True
    )
    area_winner, area_top, area_status = _compare_scene_metric(
        summaries, "severe_footprint_m2"
    )
    result = {
        "by": args.get("by"),
        "greater_damage_scene": pct_winner,
        "higher_severe_pct_scene": pct_winner,
        "higher_severe_area_scene": area_winner,
        "top_severe_pct_scene_ids": pct_top,
        "top_severe_area_scene_ids": area_top,
        "severe_pct_comparison_status": pct_status,
        "severe_area_comparison_status": area_status,
        "scenes": summaries,
    }
    return store_result_handle(state, "scene_compare", "scene_comparison", TOOL_NAME, args, work, "xdamage_vector_label_tools", result)


TOOL_NAME = "CompareScenes"
DEFAULT_PORT = 20113
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class CompareScenesWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `CompareScenes` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_CompareScenes, require_existing=True)

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

    worker = CompareScenesWorker(
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
