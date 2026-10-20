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


def _route_public(route: dict[str, Any] | None) -> dict[str, Any] | None:
    if not route:
        return None
    return {
        "route_id": route.get("route_id"),
        "route_type": route.get("route_type"),
        "distance_m": route.get("distance_m"),
        "distance_km": route.get("distance_km"),
        "eta_min": route.get("eta_min"),
        "risk_score": route.get("risk_score"),
        "objective_cost": route.get("objective_cost"),
        "objective_cost_unit": route.get("objective_cost_unit"),
        "blocked_edges": route.get("blocked_edges"),
        "node_count": route.get("node_count"),
        "selected_shelter_name": route.get("selected_shelter_name"),
        "shelter_confidence": route.get("shelter_confidence"),
        "shelter_candidate_id": route.get("shelter_candidate_id"),
    }


def _route_group_summary(
    grouped: dict[str, list[dict[str, Any]]], best: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for group_name, routes in grouped.items():
        summary[group_name] = {
            "candidate_count": len(routes),
            "candidate_ids": [route.get("route_id") for route in routes],
            "best": _route_public(best[group_name]),
        }
    return summary


def process_PlanRoutes(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "ourdata/final_routing_subset_v1":
        raise ValueError(f"PlanRoutes is not implemented for source {source_of(state)}")
    sample_dir = _routing_sample_dir(state, work)
    objective = str(args.get("objective") or "")
    allowed_objectives = {
        "nearest_reachable_shelter",
        "candidate_routes",
        "overlay_routes",
    }
    if objective not in allowed_objectives:
        raise ValueError(
            f"unsupported packaged routing objective {objective!r}; "
            f"expected one of {sorted(allowed_objectives)}"
        )
    avoid = args.get("avoid") or {}
    if not isinstance(avoid, dict) or set(avoid) - {"flood_affected_roads"}:
        raise ValueError("PlanRoutes supports only the flood_affected_roads avoidance assertion")
    if avoid.get("flood_affected_roads") is not True:
        raise ValueError(
            "packaged routing requires avoid.flood_affected_roads=true; "
            "unscreened routing is not implemented"
        )
    validation = routing_tools().ValidateAndRankCandidateRoutes(str(sample_dir))
    grouped = validation["routes"]
    best = validation["best"]
    public_best = {key: _route_public(route) for key, route in best.items()}
    common = {
        "objective": objective,
        "avoid": avoid,
        "selection_contract": "packaged_composite_edge_cost_v2",
        "ranking_contract": validation["contract"],
        "ranking_validation_status": validation["status"],
        "validated_route_count": validation["validated_route_count"],
        "graph_search_performed": False,
        "candidate_score_source": validation["score_path"],
        "candidate_geometry_source": validation["geometry_path"],
        "flood_risk_road_source": validation["road_path"],
        "route_groups": _route_group_summary(grouped, best),
        "best": public_best,
    }
    if objective == "nearest_reachable_shelter":
        selected = best["fastest"]
        result = {
            **common,
            "selected_route": _route_public(selected),
            "nearest_reachable_shelter": {
                "candidate_id": selected.get("shelter_candidate_id"),
                "name": selected.get("selected_shelter_name"),
                "confidence": selected.get("shelter_confidence"),
            },
        }
    else:
        result = common
    return store_result_handle(state, "routes", "route_plan", TOOL_NAME, args, work, "flood_safe_routing_tools", result)


TOOL_NAME = "PlanRoutes"
DEFAULT_PORT = 20117
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class PlanRoutesWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `PlanRoutes` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_PlanRoutes, require_existing=True)

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

    worker = PlanRoutesWorker(
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
