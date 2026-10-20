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



def _routing_sample_dir(state: dict[str, Any], work: str) -> Path:
    for entry in all_lineage_path_entries(state, work):
        p = Path(entry.get("path") or "")
        parts = p.parts
        if "finalized_samples" in parts:
            idx = parts.index("finalized_samples")
            if len(parts) > idx + 1:
                sample_dir = Path(*parts[: idx + 2])
                if sample_dir.exists():
                    return sample_dir
        if "routing" in parts:
            idx = parts.index("routing")
            if len(parts) > idx + 1:
                sample_dir = Path(*parts[: idx + 2])
                if sample_dir.exists():
                    return sample_dir
    raise ValueError("could not resolve routing sample directory")


def process_CompareRoutes(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "ourdata/final_routing_subset_v1":
        raise ValueError(f"CompareRoutes is not implemented for source {source_of(state)}")
    sample_dir = _routing_sample_dir(state, work)
    prior = artifact_result(state, work) or {}
    best = prior.get("best") if prior.get("selection_contract") == "packaged_composite_edge_cost_v2" else None
    ranking_contract = prior.get("ranking_contract")
    if not best:
        validation = routing_tools().ValidateAndRankCandidateRoutes(str(sample_dir))
        best = validation["best"]
        ranking_contract = validation["contract"]
    prefer = args.get("prefer")
    allowed = {"fastest", "safest", "balanced"}
    if not isinstance(prefer, dict) or set(prefer) - allowed:
        raise ValueError(f"CompareRoutes prefer must contain only {sorted(allowed)}")
    requested = {name for name in allowed if prefer.get(name) is True}
    if not requested:
        raise ValueError("CompareRoutes requires at least one true preference")
    invalid_values = {
        name: value for name, value in prefer.items() if not isinstance(value, bool)
    }
    if invalid_values:
        raise ValueError(f"CompareRoutes preferences must be boolean: {invalid_values}")
    selected_best = {name: best[name] for name in sorted(requested)}
    delta = None
    if requested == allowed:
        delta = routing_tools().ComputeRouteDelta(
            best["fastest"], best["safest"], best["balanced"]
        )
    result = {
        "prefer": prefer,
        "requested_groups": sorted(requested),
        "selection_contract": "packaged_composite_edge_cost_v2",
        "ranking_contract": ranking_contract,
        "graph_search_performed": False,
        "recommendation_emitted": False,
        "best": selected_best,
        "fastest_route": best.get("fastest") if "fastest" in requested else None,
        "safest_route": best.get("safest") if "safest" in requested else None,
        "balanced_route": best.get("balanced") if "balanced" in requested else None,
        "delta": delta,
    }
    return store_result_handle(state, "route_compare", "route_comparison", TOOL_NAME, args, work, "flood_safe_routing_tools", result)


TOOL_NAME = "CompareRoutes"
DEFAULT_PORT = 20118
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class CompareRoutesWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `CompareRoutes` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_CompareRoutes, require_existing=True)

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

    worker = CompareRoutesWorker(
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
