import argparse
import copy
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



def _deforestation_sample_dir(state: dict[str, Any], work: str) -> Path:
    for entry in all_lineage_path_entries(state, work):
        path_value = entry.get("path")
        if not path_value:
            continue
        base = dir_from_path_parts(Path(path_value), "samples_v2", 1)
        if base and base.exists():
            return base
    raise ValueError("could not resolve deforestation sample directory from input handles")


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


def _mh_area(domain: str, sid: str, handle: str) -> dict[str, Any]:
    import numpy as np
    mods = mh_modules()
    common = mods["common"]
    mask = common.comparison_mask(domain, sid, handle)
    _arr, sidecar = common.load_handle_array(domain, sid, handle)
    area_ha = common.mask_area_ha(mask, sidecar)
    return {
        "handle": handle,
        "area_ha": round(float(area_ha), 3),
        "pixel_count": int(np.count_nonzero(mask)),
        "area_method": "grid_aware_wgs84_ellipsoidal_or_projected_affine_v1",
    }


def _mh_primary_handle(domain: str) -> str:
    return {
        "wildfire": "mcd64a1__event",
        "deforestation": "hansen_loss__event",
        "urban_expansion": "built_gain",
        "drought": "low_chirps_spi",
        "air_pollution": "high_no2_column",
    }[domain]


def _mh_class_handle(domain: str) -> tuple[str, str, str | None, str]:
    return {
        "wildfire": ("dnbr", "@dNBR", "mcd64a1__event", "mcd64_burn_footprint"),
        "deforestation": ("ndvi_change", "@NDVI_change", "hansen_loss__event", "hansen_loss_footprint"),
        "urban_expansion": ("ndbi_change", "@NDBI_change", None, "whole_aoi"),
        "drought": ("chirps_spi__target", "@SPI_RELATIVE", None, "whole_aoi"),
        "air_pollution": ("s5p_no2__peak", "@NO2_COLUMN_RELATIVE", None, "whole_aoi"),
    }[domain]


def process_ScoreHazard(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    work_card = require_artifact(state, work)
    source = source_of(state)
    if source == "deforestation_data":
        if args.get("hazard") != "forest_disturbance" or args.get("scale") != "vh_delta_severity":
            raise ValueError(f"unsupported deforestation ScoreHazard hazard/scale: {args.get('hazard')!r}/{args.get('scale')!r}")
        stats = artifact_result(state, work) or {}
        if stats.get("metrics") != {"sar_change": True, "vh_delta_db": True}:
            raise ValueError(
                f"work {work!r} is not the declared vh_delta_db ComputeStats result"
            )
        if stats.get("measurement_contract") != "declared_s1_vh_delta_v1":
            raise ValueError(
                f"work {work!r} does not use the declared S1 VH-delta contract"
            )
        vh_delta_db = stats.get("sar_vh_delta_db")
        if vh_delta_db is None:
            raise ValueError("upstream statistics do not contain sar_vh_delta_db")
        scored = deforestation_tools().ScoreVHDeltaSeverity(vh_delta_db)
        result = {
            "hazard": args.get("hazard"),
            "scale": args.get("scale"),
            "vh_delta_db": scored["vh_delta_db"],
            "sar_vh_delta_db": scored["vh_delta_db"],
            "severity": scored["severity"],
            "severity_contract": scored["version"],
            "severity_metric": scored["metric"],
            "units": scored["units"],
            "thresholds_db": scored["thresholds_db"],
        }
        processor = "deforestation_raster_tools"
    elif source == "data/multi_hazard":
        domain, sid = _mh_domain_sid(state, work)
        question_type = str((state.get("case") or {}).get("question_type") or "")
        required_scale = {
            "B": "benchmark_relative_column_quartile" if domain == "air_pollution" else "question_defined_class",
            "D": "impact_footprint",
        }.get(question_type)
        if required_scale is None:
            raise ValueError(
                f"ScoreHazard is not prescribed for multi-hazard question type {question_type!r}"
            )
        b_modes = {
            "wildfire": "wildfire_burn",
            "deforestation": "forest_loss",
            "urban_expansion": "built_up_gain",
            "drought": "drought_anomaly",
            "air_pollution": "pollution_peak",
        }
        d_modes = {domain_name: f"{mode}_exposure" for domain_name, mode in b_modes.items()}
        expected_mode = b_modes[domain] if question_type == "B" else d_modes[domain]
        expected_roles = {"hazard": domain}
        if question_type == "D":
            expected_roles["exposure"] = {
                "wildfire": "highways",
                "deforestation": "nearest_highway",
                "urban_expansion": "population_in_footprint",
                "drought": "schools",
                "air_pollution": "population",
            }[domain]
        if (
            work_card.get("type") != "work"
            or work_card.get("requested_mode") != expected_mode
            or work_card.get("input_roles") != expected_roles
        ):
            raise ValueError(
                f"work {work!r} is not the required {domain} type-{question_type} PrepareScene result"
            )
        expected_scales = {
            "wildfire": {"question_defined_class", "impact_footprint"},
            "deforestation": {"question_defined_class", "impact_footprint"},
            "urban_expansion": {"question_defined_class", "impact_footprint"},
            "drought": {"question_defined_class", "impact_footprint"},
            "air_pollution": {"benchmark_relative_column_quartile", "impact_footprint"},
        }
        if args.get("hazard") != domain:
            raise ValueError(f"unsupported ScoreHazard hazard {args.get('hazard')!r}; declared domain is {domain!r}")
        if args.get("scale") != required_scale:
            raise ValueError(
                f"unsupported ScoreHazard scale for question type {question_type}: "
                f"{args.get('scale')!r}; expected {required_scale!r}"
            )
        if args.get("scale") not in expected_scales[domain]:
            raise ValueError(f"unsupported ScoreHazard scale for {domain}: {args.get('scale')!r}")
        if args.get("scale") == "impact_footprint":
            result = {"aoi_id": f"{domain}__{sid}", "hazard": domain, "impact_footprint": _mh_area(domain, sid, _mh_primary_handle(domain))}
        else:
            handle, threshold, mask_layer, scope = _mh_class_handle(domain)
            declared_names = {
                Path(str(entry.get("path") or "")).name
                for entry in all_lineage_path_entries(state, work)
                if entry.get("path")
            }
            required_names = {f"{handle}.tif", f"{handle}.json", "available_handles.json"}
            if mask_layer:
                required_names.update({f"{mask_layer}.tif", f"{mask_layer}.json"})
            if domain == "air_pollution":
                required_names.add("normalization.json")
            if domain == "drought":
                required_names.add("drought_normalization.json")
            missing = sorted(required_names - declared_names)
            if missing:
                raise ValueError(f"classification work is missing declared inputs: {missing}")
            run_args = {
                "aoi_id": f"{domain}__{sid}",
                "handle": handle,
                "by_class": threshold,
                "percentiles": [5, 50, 95],
                "analysis_scope": scope,
            }
            if mask_layer:
                run_args["mask_layer"] = mask_layer
            result = mh_modules()["stats"].run(run_args)
            if domain == "air_pollution":
                # The generic statistics engine reports how it partitions a
                # continuous array.  This benchmark-specific label records
                # where those interval boundaries were fitted, which is part
                # of the air-pollution B answer contract.
                result["classification_contract"] = "train_only_s5p_pixel_quartiles_v1"
        if isinstance(result, dict) and result.get("error"):
            raise ValueError(f"multi-hazard processor returned error: {result['error']}")
        processor = "multi_hazard_cached_raster_tools"
    elif source == "geodisaster":
        if args.get("hazard") != "building_damage" or args.get("scale") != "spatial_concentration":
            raise ValueError(f"unsupported xBD ScoreHazard hazard/scale: {args.get('hazard')!r}/{args.get('scale')!r}")
        prior = artifact_result(state, work) or {}
        cells = prior.get("all_candidates") or prior.get("all_quadrants") or prior.get("cells") or []
        if not cells:
            cells = xbd.grid_rows(xbd.scene_contexts(state, work)[0], "quadrants")
        if cells:
            per_quadrant = {str(c.get("cell_id")): int(c.get("severe_count") or 0) for c in cells}
            total_severe = sum(int(c.get("severe_count") or 0) for c in cells)
            max_severe = max([int(c.get("severe_count") or 0) for c in cells] or [0])
            concentration = round(max_severe / total_severe, 4) if total_severe else 0.0
        tied_quadrants = (
            sorted(
                cell_id
                for cell_id, count in per_quadrant.items()
                if int(count or 0) == max_severe
            )
            if total_severe
            else []
        )
        if total_severe == 0:
            concentration = None
            assessment = "no_severe_damage"
        elif concentration is not None and concentration >= 0.7:
            assessment = "highly concentrated"
        elif concentration is not None and concentration >= 0.45:
            assessment = "moderately concentrated"
        else:
            assessment = "spatially dispersed"
        result = {
            "hazard": args.get("hazard"),
            "scale": args.get("scale"),
            "total_severe": total_severe,
            "max_severe_in_cell": max_severe,
            "concentration_score": concentration,
            "assessment": assessment,
            "concentration_status": "scored" if total_severe else "no_severe_damage",
            "top_tied_quadrant_ids": tied_quadrants,
            "worst_quadrant": tied_quadrants[0] if tied_quadrants else None,
            "tie_break_policy": "ascending_cell_id",
            "tie_break_applied": len(tied_quadrants) > 1,
            "per_quadrant_severe": per_quadrant,
            "grid_used": "quadrant_2x2",
        }
        processor = "xdamage_vector_label_tools"
    else:
        raise ValueError(f"ScoreHazard is not implemented for source {source}")
    return store_result_handle(state, "score", "hazard_score", TOOL_NAME, args, work, processor, result)


TOOL_NAME = "ScoreHazard"
DEFAULT_PORT = 20109
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class ScoreHazardWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `ScoreHazard` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_ScoreHazard, require_existing=True)

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

    worker = ScoreHazardWorker(
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
