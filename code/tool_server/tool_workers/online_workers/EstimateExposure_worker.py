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



def _mh_domain_sid(state: dict[str, Any], work: str) -> tuple[str, str]:
    from tool_server.tool_workers.online_workers.final_dataset_worker_state import initial_cards_for

    for card in initial_cards_for(state, work):
        md = card.get("metadata") or {}
        domain = md.get("domain") or card.get("role")
        for entry in collect_path_entries(card.get("paths", card)):
            p = Path(entry.get("path") or "")
            parts = p.parts
            if "samples" in parts:
                idx = parts.index("samples")
                if len(parts) > idx + 2:
                    return str(domain or parts[idx + 1]), str(parts[idx + 2])
            if "multi_hazard" in parts:
                idx = parts.index("multi_hazard")
                if len(parts) > idx + 2:
                    return str(domain or parts[idx + 1]), str(parts[idx + 2])
        if domain and md.get("sample_id"):
            return str(domain), str(md["sample_id"])
    raise ValueError("could not resolve multi-hazard domain/sample from input handles")


def _mh_primary_handle(domain: str) -> str:
    return {
        "wildfire": "mcd64a1__event",
        "deforestation": "hansen_loss__event",
        "urban_expansion": "built_gain",
        "drought": "low_chirps_spi",
        "air_pollution": "high_no2_column",
    }[domain]


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


def process_EstimateExposure(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    source = source_of(state)
    if source == "data/multi_hazard":
        domain, sid = _mh_domain_sid(state, work)
        question_type = str((state.get("case") or {}).get("question_type") or "")
        if question_type != "D":
            raise ValueError(
                f"multi-hazard EstimateExposure is restricted to question type D, got {question_type!r}"
            )
        footprint = _mh_primary_handle(domain)
        exposure = str(args.get("exposure") or "")
        metrics = args.get("metrics") or {}
        expected = {
            "wildfire": ("highway_representatives_within_5km", {"within_km": 5}),
            "deforestation": ("nearest_highway", {"distance_to": "nearest_highway"}),
            "urban_expansion": ("population_in_footprint", {"summary": True}),
            "drought": ("schools", {"summary": True}),
            "air_pollution": ("high_no2_column_population", {"summary": True}),
        }
        if (exposure, metrics) != expected[domain]:
            raise ValueError(f"unsupported exposure/metrics contract for {domain}: {exposure!r}/{metrics!r}")
        upstream = artifact_result(state, work) or {}
        impact = upstream.get("impact_footprint") or {}
        if impact.get("handle") != footprint:
            raise ValueError(
                "upstream work is not the required domain impact-footprint ScoreHazard result"
            )
        pixel_count = impact.get("pixel_count")
        if not isinstance(pixel_count, int) or pixel_count < 0:
            raise ValueError("upstream impact footprint pixel count is invalid")
        # A zero-pixel threshold result is a valid measured outcome for the
        # air-pollution and drought tasks and must produce zero exposure, not a
        # tool error.  Wildfire-D is explicitly defined around an observed burn
        # footprint, so its contract still requires at least one burn pixel.
        if domain == "wildfire" and pixel_count == 0:
            raise ValueError("wildfire exposure requires a nonempty burn footprint")
        declared_names = {
            Path(str(entry.get("path") or "")).name
            for entry in all_lineage_path_entries(state, work)
            if entry.get("path")
        }
        required_names = {
            f"{footprint}.tif",
            f"{footprint}.json",
            "worldpop.tif",
            "worldpop.json",
            "pois.geojson",
        }
        missing = sorted(required_names - declared_names)
        if missing:
            raise ValueError(f"exposure work is missing declared inputs: {missing}")
        if exposure in {"population", "population_in_footprint", "hospitals_and_schools", "high_no2_column_population"}:
            asset_types = ["hospitals", "schools"]
        elif exposure == "highway_representatives_within_5km":
            asset_types = ["highways"]
        elif exposure == "schools":
            asset_types = ["schools"]
        elif exposure == "nearest_highway":
            asset_types = ["highways"]
        else:
            asset_types = None
        query_args: dict[str, Any] = {"aoi_id": f"{domain}__{sid}", "footprint_handle": footprint}
        if asset_types:
            query_args["asset_types"] = asset_types
        # The source QueryExposure trajectories used a 5 km context buffer for
        # every multi-hazard exposure question.  Only the wildfire-normalized
        # call retained that raw parameter explicitly; summary and distance
        # calls rely on the same source default.
        query_args["buffer_m"] = float(metrics.get("within_km", 5)) * 1000.0
        result = mh_modules()["exposure"].run(query_args)
        if isinstance(result, dict) and result.get("error"):
            raise ValueError(f"multi-hazard processor returned error: {result['error']}")
        if domain == "deforestation":
            nearest = (
                ((result.get("critical_infrastructure") or {}).get("highways") or {})
                .get("nearest")
                or {}
            )
            result["availability_status"] = (
                "found"
                if nearest.get("distance_m") is not None
                else "no_cached_highway_representative"
            )
        if domain == "air_pollution":
            result["threshold_contract"] = "train_only_s5p_pixel_q75_v1"
        result["upstream_impact_footprint"] = impact
        processor = "multi_hazard_cached_raster_tools"
    elif source == "ourdata/final_routing_subset_v1":
        sample_dir = _routing_sample_dir(state, work)
        result = routing_tools().BuildFloodRiskRoadGraph(str(sample_dir))
        processor = "flood_safe_routing_tools"
    else:
        raise ValueError(f"EstimateExposure is not implemented for source {source}")
    return store_result_handle(state, "exposure", "exposure_estimate", TOOL_NAME, args, work, processor, result)


TOOL_NAME = "EstimateExposure"
DEFAULT_PORT = 20110
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class EstimateExposureWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `EstimateExposure` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_EstimateExposure, require_existing=True)

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

    worker = EstimateExposureWorker(
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
