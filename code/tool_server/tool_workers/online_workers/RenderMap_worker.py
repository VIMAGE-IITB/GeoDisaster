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
    artifact_output_dir,
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
    routing_visualization_tools,
    run_stateful_tool,
    sen1_utils,
    source_of,
    store_result_handle,
)
from tool_server.utils.server_utils import build_logger
from tool_server.tool_workers.online_workers.sen1_prediction_runtime import render_prediction



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


def process_RenderMap(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    source = source_of(state)
    if source == "ourdata/final_routing_subset_v1":
        sample_dir = _routing_sample_dir(state, work)
        entries = all_lineage_path_entries(state, work)
        required = {
            "rgb_path": first_existing_path(entries, "rgb.tif"),
            "routes_path": first_existing_path(entries, "candidate_routes.geojson"),
            "roads_path": first_existing_path(entries, "roads_with_flood_risk.geojson"),
            "flood_extent_path": first_existing_path(entries, "flood_extent.geojson"),
            "metadata_path": first_existing_path(entries, "metadata.json"),
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            raise ValueError(f"routing render is missing declared inputs: {missing}")
        prior = artifact_result(state, work) or {}
        if prior.get("selection_contract") != "packaged_composite_edge_cost_v2":
            raise ValueError(
                "routing render requires a current composite-cost-validated route handle"
            )
        best = prior.get("best") or {}
        if set(best) != {"fastest", "safest", "balanced"}:
            raise ValueError(
                "routing render requires fastest, safest, and balanced selections"
            )
        selected = {
            group: str((best.get(group) or {}).get("route_id") or "")
            for group in ("fastest", "safest", "balanced")
        }
        rendered = routing_visualization_tools().RenderRoutesOnRGB(
            str(sample_dir),
            **{name: str(path) for name, path in required.items()},
            output_path=str(artifact_output_dir(state) / "routing_live_overlay.png"),
            current_route_handle=work,
            selected_route_ids=selected,
            layers=args.get("layers"),
        )
        overlay = Path(rendered["overlay_path"])
        result = {
            "style": args.get("style"),
            "layers": args.get("layers", {}),
            "image_path": str(overlay),
            "render_manifest_path": rendered["render_manifest_path"],
            "render_validation": rendered["render_validation"],
            "artifact": "<image>",
            "exists": overlay.exists() and overlay.stat().st_size > 0,
            "render_provenance": (
                "generated live from declared RGB, roads, flood, metadata, and "
                "current candidate-route handle; prepared overlays are not used"
            ),
        }
        processor = "flood_safe_routing_tools"
    elif source == "ourdata/sen1_hand_final_hidden_eval_5q":
        rendered = render_prediction(
            state,
            work,
            args.get("layers"),
            str(args.get("style") or ""),
        )
        overlay = Path(rendered["overlay_path"])
        result = {
            "style": args.get("style"),
            "layers": args.get("layers", {}),
            "image_path": str(overlay),
            "render_manifest_path": str(rendered["render_manifest_path"]),
            "render_validation": rendered["render_validation"],
            "visualization_generated": True,
            "artifact": "<image>",
            "exists": overlay.exists() and overlay.stat().st_size > 0,
            "render_provenance": "generated from the current S1-derived prediction; prepared public overlays are not used",
        }
        processor = "sar_flood_raster_tools"
    else:
        raise ValueError(f"RenderMap is not implemented for source {source}")
    out = store_result_handle(state, "image", "rendered_map", TOOL_NAME, args, work, processor, result)
    out["artifact"] = "<image>"
    return out


TOOL_NAME = "RenderMap"
DEFAULT_PORT = 20119
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class RenderMapWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `RenderMap` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_RenderMap, require_existing=True)

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

    worker = RenderMapWorker(
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
