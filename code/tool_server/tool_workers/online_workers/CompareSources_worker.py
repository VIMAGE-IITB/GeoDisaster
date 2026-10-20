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
from tool_server.tool_workers.online_workers.sen1_prediction_runtime import (
    known_water_overlap,
    paths as sen1_prediction_paths,
    prediction_report,
)



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


def _cross_modal_concordance(ndvi_drop: Any, vh_delta_db: Any) -> str:
    try:
        ndvi = float(ndvi_drop)
        vh = float(vh_delta_db)
    except (TypeError, ValueError) as exc:
        raise ValueError("concordance requires numeric NDVI drop and VH delta") from exc
    if not math.isfinite(ndvi) or not math.isfinite(vh):
        raise ValueError("concordance requires finite NDVI drop and VH delta")
    optical_change = ndvi > 0.10
    sar_change = vh < -1.0
    if optical_change and sar_change:
        return "concordant"
    if not optical_change and not sar_change:
        return "no_change"
    return "discordant"


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


def _require_mh_comparison_work(
    state: dict[str, Any], work: str, expected_handles: list[str]
) -> None:
    card = require_artifact(state, work)
    question_type = str((state.get("case") or {}).get("question_type") or "")
    if question_type != "C":
        raise ValueError(
            f"multi-hazard CompareSources is restricted to question type C, got {question_type!r}"
        )
    if card.get("type") != "work" or card.get("requested_mode") != "multisource_hazard":
        raise ValueError(
            f"work {work!r} is not a multisource_hazard PrepareScene result"
        )
    inputs = card.get("inputs") or {}
    if set(inputs) != {"hazard", "sources"}:
        raise ValueError(
            "multisource_hazard comparison requires exactly hazard and sources scene inputs"
        )
    declared_names = {
        Path(str(entry.get("path") or "")).name
        for entry in all_lineage_path_entries(state, work)
        if entry.get("path")
    }
    required_names = {
        name
        for handle in expected_handles
        for name in (f"{handle}.tif", f"{handle}.json")
    }
    missing = sorted(required_names - declared_names)
    if missing:
        raise ValueError(
            f"multisource_hazard work is missing declared comparison inputs: {missing}"
        )


def _sen1_mask_report_from_work(state: dict[str, Any], work: str) -> dict[str, Any]:
    return prediction_report(state, work)


def _sen1_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "chip_id": report.get("chip_id"),
        "method": report.get("method"),
        "computed_flood_percentage": report.get("computed_flood_percentage"),
        "computed_flood_area_m2": report.get("computed_flood_area_m2"),
        "component_count": report.get("component_count"),
        "polygon_count": report.get("polygon_count"),
        "largest_component_area_m2": report.get("largest_component_area_m2"),
        "top_3_grid_cells": report.get("top_3_grid_cells"),
    }


def process_CompareSources(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    source = source_of(state)
    if source == "deforestation_data":
        sample_dir = _deforestation_sample_dir(state, work)
        compare = args.get("compare", {})
        criterion = str(compare.get("criterion") or "")
        signals = str(compare.get("signals") or "")
        allowed = {
            ("forest_loss_support", "optical_sar_hansen"),
            ("validation", "ndvi_nbr_vs_hansen"),
            ("cause_attribution", "optical_sar_hansen"),
            ("concordance", "ndvi_nbr_vs_vh"),
        }
        if (criterion, signals) not in allowed:
            raise ValueError(
                f"unsupported deforestation comparison criterion/signals: "
                f"{criterion!r}/{signals!r}"
            )
        if criterion == "concordance":
            # Cross-modal concordance is defined over the whole AOI. Other
            # deforestation comparisons intentionally retain event-mask-
            # restricted optical evidence.
            tools = deforestation_tools()
            optical = jsonable(tools.GetOpticalEvidence(str(sample_dir), indices=("NDVI", "NBR"), epochs="all"))
            sar = jsonable(
                tools.GetSAREvidence(
                    str(sample_dir), s1_windows=declared_sar_windows(state, work)
                )
            )
            ndvi_drop = optical.get("indices", {}).get("NDVI", {}).get("drop")
            nbr_drop = optical.get("indices", {}).get("NBR", {}).get("drop")
            vh_delta_db = sar.get("polarizations", {}).get("VH", {}).get("delta_db")
            try:
                nbr_value = float(nbr_drop)
            except (TypeError, ValueError) as exc:
                raise ValueError("concordance requires numeric contextual NBR drop") from exc
            if not math.isfinite(nbr_value):
                raise ValueError("concordance requires finite contextual NBR drop")
            concordance = _cross_modal_concordance(ndvi_drop, vh_delta_db)
            result = {
                "sample_dir": str(sample_dir),
                "compare": compare,
                "optical": {
                    "NDVI": optical.get("indices", {}).get("NDVI"),
                    "NBR": optical.get("indices", {}).get("NBR"),
                    "cloud_flags_per_epoch": optical.get("cloud_flags_per_epoch"),
                },
                "sar": sar,
                "concordance": concordance,
                "evidence": {
                    "ndvi_drop": float(ndvi_drop),
                    "nbr_drop": nbr_value,
                    "vh_delta_db": float(vh_delta_db),
                },
                "thresholds": {
                    "optical_change_ndvi_drop_gt": 0.10,
                    "sar_change_vh_delta_db_lt": -1.0,
                },
            }
        elif criterion == "validation" and signals == "ndvi_nbr_vs_hansen":
            tools = deforestation_tools()
            optical = jsonable(
                tools.GetOpticalEvidence(
                    str(sample_dir),
                    indices=("NDVI", "NBR"),
                    epochs="all",
                    restrict_to_loss_mask=True,
                )
            )
            mask = jsonable(tools.GetForestMask(str(sample_dir)))
            work_card = require_artifact(state, work)
            generated_reports = work_card.get("generated_quality_reports") or {}
            generated_report = (
                generated_reports.get("forest")
                if isinstance(generated_reports, dict)
                else None
            )
            if generated_report:
                metrics = generated_report["loss_event_year"]
                mask.update(
                    {
                        "loss_area_ha": metrics["area_ha"],
                        "loss_pixel_count": metrics["loss_pixel_count"],
                        "loss_area_measurement_method": metrics["measurement_method"],
                        "generated_quality_report": generated_report,
                    }
                )
            result = {
                "sample_dir": str(sample_dir),
                "compare": compare,
                "optical": {
                    "NDVI": optical.get("indices", {}).get("NDVI"),
                    "NBR": optical.get("indices", {}).get("NBR"),
                    "cloud_flags_per_epoch": optical.get("cloud_flags_per_epoch"),
                },
                "hansen": mask,
                "classification_rules": [],
            }
        else:
            bundle = copy.deepcopy(
                _deforestation_bundle(state, work, str(sample_dir))
            )
            work_card = require_artifact(state, work)
            generated_reports = work_card.get("generated_quality_reports") or {}
            generated_report = (
                generated_reports.get("forest")
                if isinstance(generated_reports, dict)
                else None
            )
            if generated_report:
                metrics = generated_report["loss_event_year"]
                bundle["mask"].update(
                    {
                        "loss_area_ha": metrics["area_ha"],
                        "loss_pixel_count": metrics["loss_pixel_count"],
                        "loss_area_measurement_method": metrics["measurement_method"],
                        "generated_quality_report": generated_report,
                    }
                )
                bundle["classification"] = jsonable(
                    deforestation_tools().ClassifyChangeType(
                        bundle["optical"], bundle["sar"], bundle["mask"]
                    )
                )
            optical = bundle["optical"]
            result = {
                "sample_dir": str(sample_dir),
                "compare": compare,
                "optical": {
                    "NDVI": optical.get("indices", {}).get("NDVI"),
                    "NBR": optical.get("indices", {}).get("NBR"),
                    "cloud_flags_per_epoch": optical.get("cloud_flags_per_epoch"),
                },
                "sar": bundle["sar"],
                "hansen": bundle["mask"],
                "classification": bundle["classification"],
                "classification_rules": bundle["classification"].get("rules_fired", []),
            }
        processor = "deforestation_raster_tools"
    elif source == "data/multi_hazard":
        domain, sid = _mh_domain_sid(state, work)
        mods = mh_modules()
        handles = mods["common_defs"].COMPARISON_HANDLES[domain]
        _require_mh_comparison_work(state, work, handles)
        rule = mods["common_defs"].INTAKE_RULES[domain]
        metric = rule["metric"]
        compare_args = args.get("compare") or {}
        if compare_args.get("sources") != "all":
            raise ValueError(f"unsupported multi-hazard comparison sources: {compare_args.get('sources')!r}")
        requested = str(compare_args.get("criterion") or metric)
        if domain == "air_pollution":
            if requested != "benchmark_relative_intensity_agreement":
                raise ValueError(
                    "air-pollution comparison requires criterion "
                    "'benchmark_relative_intensity_agreement'"
                )
            expected_fraction = float(rule["required_agreement_fraction"])
            if float(compare_args.get("required_agreement_fraction", -1)) != expected_fraction:
                raise ValueError(
                    "air-pollution comparison requires required_agreement_fraction "
                    f"{expected_fraction}"
                )
            if compare_args.get("normalization_fit_split") != "train":
                raise ValueError("air-pollution comparison requires the train normalization fit")
            metric = requested
        elif domain == "drought":
            if requested != "iou":
                raise ValueError("drought comparison requires declared stress-mask IoU")
            metric = "iou"
            expected_minimum = float(rule.get("min_iou", 0.30))
            if float(compare_args.get("minimum_iou", -1)) != expected_minimum:
                raise ValueError(
                    f"drought comparison requires minimum_iou {expected_minimum}"
                )
            if compare_args.get("threshold_contract") != "per_source_train_pixel_q25_v1":
                raise ValueError(
                    "drought comparison requires threshold_contract "
                    "'per_source_train_pixel_q25_v1'"
                )
            if compare_args.get("normalization_fit_split") != "train":
                raise ValueError("drought comparison requires the train normalization fit")
        elif requested != metric:
            raise ValueError(f"unsupported comparison criterion for {domain}: {requested}")
        if metric == "area_agreement":
            expected_threshold = float(rule["max_median_pct_delta"])
            supplied_threshold = compare_args.get("maximum_median_pct_delta")
            if supplied_threshold is None or float(supplied_threshold) != expected_threshold:
                raise ValueError(
                    f"{domain} area agreement requires maximum_median_pct_delta "
                    f"{expected_threshold}, got {supplied_threshold!r}"
                )
        run_args = {
                "aoi_id": f"{domain}__{sid}",
                "handles": handles,
                "metric": metric,
                "minimum_correlation": compare_args.get(
                    "minimum_median_correlation", rule.get("min_correlation", 0.65)
                ),
                "require_all_pairs_defined": compare_args.get(
                    "require_all_pairs_defined",
                    rule.get("require_all_pairs_defined", True),
                ),
                "minimum_agreement": compare_args.get("minimum_agreement", rule.get("min_agreement", 0.70)),
                "minimum_iou": compare_args.get("minimum_iou", rule.get("min_iou", 0.50)),
                "minimum_kappa": compare_args.get("minimum_kappa", rule.get("min_kappa", 0.50)),
                "maximum_median_pct_delta": compare_args.get(
                    "maximum_median_pct_delta",
                    rule.get("max_pct_delta", rule.get("max_median_pct_delta", 30.0)),
                ),
                "required_agreement_fraction": compare_args.get(
                    "required_agreement_fraction",
                    rule.get("required_agreement_fraction", 1.0),
                ),
            }
        result = mods["compare"].run(run_args)
        if isinstance(result, dict) and result.get("error"):
            raise ValueError(f"multi-hazard processor returned error: {result['error']}")
        if domain == "drought":
            normalization = mods["common"].drought_normalization(sid)
            result.update(
                metric="median_pairwise_iou",
                median_pairwise_iou=result.get("score"),
                minimum_iou=float(compare_args["minimum_iou"]),
                thresholds={
                    handle: float(normalization["sources"][handle]["pixel_q25"])
                    for handle in handles
                },
                threshold_fit_split=normalization["fit_split"],
                stress_mask_contract="per_source_train_pixel_q25_v1",
            )
        result["agree"] = result.get("verdict") == "sources_agree"
        processor = "multi_hazard_cached_raster_tools"
    elif source == "ourdata/sen1_hand_final_hidden_eval_5q":
        compare = args.get("compare") or {}
        if not isinstance(compare, dict):
            raise ValueError("Sentinel-1 CompareSources requires a compare object")
        reference = str(compare.get("reference") or "")
        criterion = str(compare.get("criterion") or "")
        if reference != "known_water":
            raise ValueError(
                f"unsupported Sentinel-1 comparison reference {reference!r}; expected 'known_water'"
            )
        if criterion not in {"overlap_vs_outside_reference", "diagnostic_overlap"}:
            raise ValueError(
                f"unsupported Sentinel-1 comparison criterion {criterion!r}"
            )
        report = _sen1_mask_report_from_work(state, work)
        result = known_water_overlap(report, sen1_prediction_paths(state, work).get("jrc"))
        if not result.get("available"):
            raise ValueError("declared JRC known-water reference is required for this comparison")
        result["compare"] = {
            "reference": reference,
            "criterion": criterion,
        }
        result["mask_summary"] = _sen1_report_summary(report)
        result["prediction_provenance"] = report.get("prediction_provenance")
        processor = "sar_flood_raster_tools"
    else:
        raise ValueError(f"CompareSources is not implemented for source {source}")
    return store_result_handle(state, "compare", "source_comparison", TOOL_NAME, args, work, processor, result)


TOOL_NAME = "CompareSources"
DEFAULT_PORT = 20106
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class CompareSourcesWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `CompareSources` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_CompareSources, require_existing=True)

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

    worker = CompareSourcesWorker(
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
