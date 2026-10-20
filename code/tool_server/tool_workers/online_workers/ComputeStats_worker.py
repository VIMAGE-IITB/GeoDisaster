import argparse
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import get_tool_instruction
from tool_server.tool_workers.online_workers.final_dataset_worker_state import (
    all_lineage_path_entries,
    artifact_result,
    collect_path_entries,
    declared_sar_windows,
    deforestation_tools,
    dir_from_path_parts,
    jsonable,
    load_json,
    mh_modules,
    require_artifact,
    require_question,
    resolve_data_path,
    run_stateful_tool,
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


def _classify_canopy(mean_pct: float | None) -> str:
    try:
        value = float(mean_pct)
    except (TypeError, ValueError) as exc:
        raise ValueError("canopy mean must be numeric") from exc
    if not __import__("math").isfinite(value) or not 0 <= value <= 100:
        raise ValueError("canopy mean must be finite and within 0..100 percent")
    if value > 60:
        return "dense_forest"
    if value >= 30:
        return "open_canopy"
    return "sparse"


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
    arr, sidecar = common.load_handle_array(domain, sid, handle)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"primary footprint must be single-band: {handle}")
        arr = arr[0]
    valid = np.isfinite(arr)
    if not np.any(valid):
        raise ValueError(f"primary footprint has no valid pixels: {handle}")
    values = set(np.unique(arr[valid]).tolist())
    if not values <= {0.0, 1.0}:
        raise ValueError(
            f"primary footprint must be a binary 0/1 mask: {handle}; values={sorted(values)[:8]}"
        )
    mask = valid & (arr > 0.5)
    area_ha = common.mask_area_ha(mask, sidecar)
    return {
        "handle": handle,
        "area_ha": round(float(area_ha), 3),
        "units": "ha",
        "pixel_count": int(np.count_nonzero(mask)),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "area_method": "grid_aware_wgs84_ellipsoidal_or_projected_affine_v1",
        "mask_contract": "finite_binary_primary_footprint_v1",
    }


def _mh_primary_handle(domain: str) -> str:
    return {
        "wildfire": "mcd64a1__event",
        "deforestation": "hansen_loss__event",
        "urban_expansion": "built_gain",
        "drought": "low_chirps_spi",
        "air_pollution": "high_no2_column",
    }[domain]


METRIC_ALIASES: dict[str, set[str]] = {
    "canopy_cover": {"canopy_cover", "treecover_mean_pct"},
    "forest_loss_history": {"forest_loss_history", "loss_area_ha_by_year"},
    "sar_change": {"sar_change", "vh_delta_db"},
    "hazard_area": {"hazard_area", "burned_area", "forest_loss", "built_up_gain"},
    "vegetation_anomaly": {"vegetation_anomaly", "ndvi_z_score"},
    "pollution_peak": {"pollution_peak", "peak_no2_column"},
    "damage_summary": {
        "damage_summary",
        "building_count_by_damage",
        "severe_ratio",
        "footprint_area",
        "target_mask_distribution",
        "severe_footprint_area",
    },
}


def _normalise_metrics(metrics: Any) -> dict[str, bool]:
    if isinstance(metrics, str):
        raw = {metrics: True}
    elif isinstance(metrics, list):
        raw = {str(k): True for k in metrics}
    elif isinstance(metrics, dict):
        raw = {str(k): bool(v) for k, v in metrics.items() if v}
    else:
        raw = {}
    normalised: dict[str, bool] = {}
    for broad, aliases in METRIC_ALIASES.items():
        if aliases & set(raw):
            normalised[broad] = True
    for key, value in raw.items():
        normalised.setdefault(key, value)
    return normalised


def _metric_requested(metrics: dict[str, bool], *keys: str) -> bool:
    return any(bool(metrics.get(key)) for key in keys)


def process_ComputeStats(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    work_card = require_artifact(state, work)
    source = source_of(state)
    metrics = _normalise_metrics(args.get("metrics") or {})
    raw_metric_keys = {str(k) for k, value in (args.get("metrics") or {}).items() if value} if isinstance(args.get("metrics"), dict) else set()
    allowed_raw = {
        "deforestation_data": {"treecover_mean_pct", "loss_area_ha_by_year", "vh_delta_db"},
        "data/multi_hazard": {"burned_area", "forest_loss", "built_up_gain", "ndvi_z_score", "peak_no2_column"},
        "geodisaster": {"building_count_by_damage", "footprint_area", "severe_ratio", "damage_summary", "severe_footprint_area", "target_mask_distribution"},
    }
    if not raw_metric_keys or not raw_metric_keys <= allowed_raw.get(source, set()):
        raise ValueError(f"unsupported ComputeStats metrics: {sorted(raw_metric_keys)!r}")
    expected_region = "scene" if source == "geodisaster" else "aoi"
    if args.get("region") != expected_region:
        raise ValueError(f"unsupported ComputeStats region: {args.get('region')!r}")
    if source == "deforestation_data":
        sample_dir = _deforestation_sample_dir(state, work)
        tools = deforestation_tools()
        needs_canopy = _metric_requested(metrics, "canopy_cover")
        needs_loss = _metric_requested(metrics, "forest_loss_history")
        needs_sar = _metric_requested(metrics, "sar_change")
        if not needs_canopy and not needs_loss and not needs_sar:
            raise ValueError(
                f"unsupported deforestation ComputeStats metrics: {sorted(metrics)}"
            )
        # Read only the modality opened by the prescribed chain.  The old
        # all-modal bundle implicitly touched optical/SAR/mask siblings even
        # for a single-modality question.
        canopy = (
            jsonable(tools.GetBaselineCanopyStats(str(sample_dir)))
            if needs_canopy
            else {}
        )
        loss_history = (
            jsonable(tools.GetForestLossHistory(str(sample_dir)))
            if needs_loss
            else {}
        )
        sar = (
            jsonable(
                tools.GetSAREvidence(
                    str(sample_dir), s1_windows=declared_sar_windows(state, work)
                )
            )
            if needs_sar
            else {}
        )
        result = {"sample_dir": str(sample_dir), "metrics": metrics}
        if needs_canopy:
            mean_pct = canopy.get("treecover2000_mean_pct")
            result.update(
                treecover2000_mean_pct=mean_pct,
                canopy_class=_classify_canopy(mean_pct),
                valid_pixel_count=canopy.get("valid_pixel_count"),
                measurement_contract=canopy.get("measurement_contract"),
            )
        if needs_loss:
            result.update(
                loss_year_histogram_ha=loss_history.get("loss_year_histogram_ha", {}),
                valid_pixel_count=loss_history.get("valid_pixel_count"),
                analysis_window=loss_history.get("analysis_window"),
                measurement_contract=loss_history.get("measurement_contract"),
            )
        if needs_sar:
            result.update(
                sar_vh_delta_db=sar.get("polarizations", {}).get("VH", {}).get("delta_db"),
                measurement_contract="declared_s1_vh_delta_v1",
                temporal_metadata_source=sar.get("temporal_metadata_source"),
            )
        processor = "deforestation_raster_tools"
    elif source == "data/multi_hazard":
        domain, sid = _mh_domain_sid(state, work)
        question_type = str((state.get("case") or {}).get("question_type") or "")
        expected_mode = {
            "wildfire": "wildfire_burn",
            "deforestation": "forest_loss",
            "urban_expansion": "built_up_gain",
            "drought": "drought_anomaly",
            "air_pollution": "pollution_peak",
        }[domain]
        if question_type != "A":
            raise ValueError(
                f"multi-hazard ComputeStats is restricted to question type A, got {question_type!r}"
            )
        if (
            work_card.get("type") != "work"
            or work_card.get("requested_mode") != expected_mode
            or work_card.get("input_roles") != {"hazard": domain}
        ):
            raise ValueError(
                f"work {work!r} is not the required {domain} type-A PrepareScene result"
            )
        expected_metric = {
            "wildfire": "burned_area",
            "deforestation": "forest_loss",
            "urban_expansion": "built_up_gain",
            "drought": "ndvi_z_score",
            "air_pollution": "peak_no2_column",
        }[domain]
        if raw_metric_keys != {expected_metric}:
            raise ValueError(
                f"unsupported ComputeStats metrics for {domain}: {sorted(raw_metric_keys)!r}; "
                f"expected {[expected_metric]!r}"
            )
        aoi_id = f"{domain}__{sid}"
        mods = mh_modules()
        if _metric_requested(metrics, "vegetation_anomaly", "ndvi_z_score"):
            required_names = {
                "modis_ndvi__target.tif",
                "modis_ndvi__target.json",
                "modis_ndvi_anomaly.csv",
                "available_handles.json",
            }
            declared_names = {
                Path(str(entry.get("path") or "")).name
                for entry in all_lineage_path_entries(state, work)
                if entry.get("path")
            }
            missing = sorted(required_names - declared_names)
            if missing:
                raise ValueError(f"drought anomaly work is missing declared inputs: {missing}")
            result = mods["temporal"].run(
                {
                    "aoi_id": aoi_id,
                    "modality": "modis_ndvi_anomaly",
                    "target_handle": "modis_ndvi__target",
                    "method": "z_score",
                }
            )
            result["metrics"] = metrics
        elif _metric_requested(metrics, "pollution_peak", "peak_no2_column"):
            required_names = {
                "s5p_no2__peak.tif",
                "s5p_no2__peak.json",
                "available_handles.json",
            }
            declared_names = {
                Path(str(entry.get("path") or "")).name
                for entry in all_lineage_path_entries(state, work)
                if entry.get("path")
            }
            missing = sorted(required_names - declared_names)
            if missing:
                raise ValueError(f"air-pollution peak work is missing declared inputs: {missing}")
            result = mods["stats"].run({"aoi_id": aoi_id, "handle": "s5p_no2__peak", "percentiles": [50, 95, 99]})
            result["metrics"] = metrics
            result["peak_no2"] = result.get("max")
            result["p95_no2"] = (result.get("percentiles") or {}).get("p95")
        else:
            handle = _mh_primary_handle(domain)
            declared_names = {
                Path(str(entry.get("path") or "")).name
                for entry in all_lineage_path_entries(state, work)
                if entry.get("path")
            }
            required_names = {f"{handle}.tif", f"{handle}.json", "available_handles.json"}
            missing = sorted(required_names - declared_names)
            if missing:
                raise ValueError(f"area work is missing declared inputs: {missing}")
            result = {"aoi_id": aoi_id, "region": args.get("region"), "metrics": metrics, "primary_footprint": _mh_area(domain, sid, handle)}
        if isinstance(result, dict) and result.get("error"):
            raise ValueError(f"multi-hazard processor returned error: {result['error']}")
        processor = "multi_hazard_cached_raster_tools"
    elif source == "geodisaster":
        prior = artifact_result(state, work)
        if not prior:
            context = xbd.scene_contexts(state, work)[0]
            prior = {"scene": xbd.scene_answer_evidence(xbd.scene_report(context))}
        result = {"metrics": metrics, "region": args.get("region"), "stats": prior}
        processor = "xdamage_vector_label_tools"
    else:
        raise ValueError(f"ComputeStats is not implemented for source {source}")
    return store_result_handle(state, "stats", "statistics", TOOL_NAME, args, work, processor, result)


TOOL_NAME = "ComputeStats"
DEFAULT_PORT = 20104
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class ComputeStatsWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `ComputeStats` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_ComputeStats, require_existing=True)

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

    worker = ComputeStatsWorker(
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
