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


def process_ClassifyChange(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "deforestation_data":
        raise ValueError(f"ClassifyChange is not implemented for source {source_of(state)}")
    sample_dir = _deforestation_sample_dir(state, work)
    rule = str(args.get("rule") or "")
    expected_labels = {
        "forest_loss": {"positive": "deforestation", "negative": "no_deforestation"},
        "change_window": {"none": "null", "changed": "start_epoch_to_end_epoch"},
        "false_positive_type": {
            "deforestation": True,
            "plantation_harvest": True,
            "water_flood_edge": True,
            "burn_bare_soil": True,
            "agriculture": True,
            "no_change": True,
        },
        "canopy_density": {
            "dense_forest": ">60",
            "open_canopy": "30<=mean<=60",
            "sparse": "<30",
        },
        "concordance": {"concordant": True, "discordant": True, "no_change": True},
        "cross_modal_concordance": {"concordant": True, "discordant": True, "no_change": True},
    }
    if rule not in expected_labels:
        raise ValueError(f"unsupported deforestation classification rule: {rule!r}")
    if args.get("labels") != expected_labels[rule]:
        raise ValueError(f"unsupported labels for {rule!r}: {args.get('labels')!r}")
    if rule == "change_window":
        # DetectChange already computes and Hansen-gates the complete temporal
        # result from the declared optical, SAR, and mask inputs.  Consume that
        # artifact directly so this step is a causal classification/formatting
        # step and cannot silently recompute a different source bundle.
        prior = artifact_result(state, work) or {}
        if prior.get("change") != "dominant_clearing_interval":
            raise ValueError(
                f"work {work!r} is not a "
                "dominant_clearing_interval DetectChange result"
            )
        label = prior.get("label")
        if label not in {"deforestation", "no_deforestation"}:
            raise ValueError("change-window result has no valid Hansen-gated label")
        contract = prior.get("change_window_contract")
        evidence = prior.get("evidence")
        if not isinstance(contract, dict) or not isinstance(evidence, dict):
            raise ValueError("change-window result lacks its contract or evidence")
        required_evidence = {
            "ndvi_drop",
            "sar_vh_delta_db",
            "hansen_loss_area_ha",
            "selected_window_ndvi_drop",
            "selected_window_nbr_drop",
            "composite_drop_score",
        }
        if not required_evidence.issubset(evidence):
            raise ValueError("change-window result lacks required evidence fields")
        result = {
            "rule": rule,
            "label": label,
            "predicted_class": label,
            "change_window": copy.deepcopy(prior.get("best_window")),
            "confidence": prior.get("confidence"),
            "change_window_contract": copy.deepcopy(contract),
            "evidence": copy.deepcopy(evidence),
            "source": copy.deepcopy(prior),
        }
        return store_result_handle(
            state,
            "class",
            "classification",
            TOOL_NAME,
            args,
            work,
            "deforestation_raster_tools",
            result,
        )
    if rule in {"false_positive_type", "forest_loss"}:
        prior = artifact_result(state, work) or {}
        expected_compare = {
            "false_positive_type": {
                "signals": "optical_sar_hansen",
                "criterion": "cause_attribution",
            },
            "forest_loss": {
                "signals": "optical_sar_hansen",
                "criterion": "forest_loss_support",
            },
        }[rule]
        if prior.get("compare") != expected_compare:
            raise ValueError(
                f"work {work!r} is not the required {expected_compare['criterion']!r} "
                "CompareSources result"
            )
        required_sections = {"optical", "sar", "hansen", "classification"}
        if not required_sections.issubset(prior):
            raise ValueError(f"comparison result lacks required sections for {rule}")
        opt = {"indices": prior["optical"]}
        sar = prior["sar"]
        mask = prior["hansen"]
        cls = prior["classification"]
        predicted = cls.get("predicted_class")
        allowed_causes = set(expected_labels["false_positive_type"])
        if predicted not in allowed_causes:
            raise ValueError(f"comparison result has invalid cause {predicted!r}")
        rules_fired = cls.get("rules_fired")
        if not isinstance(rules_fired, list) or not rules_fired:
            raise ValueError("comparison result lacks non-empty rules_fired")
        common_evidence = {
            "ndvi_drop": opt.get("indices", {}).get("NDVI", {}).get("drop"),
            "sar_vh_delta_db": sar.get("polarizations", {}).get("VH", {}).get("delta_db"),
            "hansen_loss_area_ha": mask.get("loss_area_ha"),
            "rules_fired": copy.deepcopy(rules_fired),
        }
        if rule == "false_positive_type":
            result = {
                "rule": rule,
                "cause": predicted,
                "predicted_class": predicted,
                "confidence": cls.get("confidence"),
                "rules_fired": copy.deepcopy(rules_fired),
                "false_positive_flags": copy.deepcopy(
                    cls.get("false_positive_flags", [])
                ),
                "evidence": {
                    **common_evidence,
                    "nbr_drop": opt.get("indices", {}).get("NBR", {}).get("drop"),
                    "sar_vv_post_mean_db": sar.get("polarizations", {}).get("VV", {}).get("post_mean_db"),
                    "treecover2000_mean_pct": mask.get("treecover2000_mean_pct"),
                },
            }
        else:
            label = "deforestation" if predicted == "deforestation" else "no_deforestation"
            result = {
                "rule": rule,
                "label": label,
                "predicted_class": predicted,
                "confidence": cls.get("confidence"),
                "rules_fired": copy.deepcopy(rules_fired),
                "false_positive_flags": copy.deepcopy(
                    cls.get("false_positive_flags", [])
                ),
                "evidence": common_evidence,
            }
        return store_result_handle(
            state,
            "class",
            "classification",
            TOOL_NAME,
            args,
            work,
            "deforestation_raster_tools",
            result,
        )
    if rule == "canopy_density":
        stats = artifact_result(state, work) or {}
        metrics = stats.get("metrics") or {}
        if not metrics.get("canopy_cover") or set(metrics) != {
            "canopy_cover", "treecover_mean_pct"
        }:
            raise ValueError(
                f"work {work!r} is not a baseline-canopy ComputeStats result"
            )
        mean_pct = stats.get("treecover2000_mean_pct")
        predicted = _classify_canopy(mean_pct)
        if stats.get("canopy_class") != predicted:
            raise ValueError("baseline-canopy statistics contain an inconsistent class")
        result = {
            "rule": rule,
            "treecover2000_mean_pct": mean_pct,
            "predicted_class": predicted,
            "measurement_contract": stats.get("measurement_contract"),
        }
        return store_result_handle(
            state,
            "class",
            "classification",
            TOOL_NAME,
            args,
            work,
            "deforestation_raster_tools",
            result,
        )
    if rule in {"concordance", "cross_modal_concordance"}:
        prior = artifact_result(state, work) or {}
        expected_compare = {"signals": "ndvi_nbr_vs_vh", "criterion": "concordance"}
        if prior.get("compare") != expected_compare:
            raise ValueError(
                f"work {work!r} is not the required cross-modal concordance CompareSources result"
            )
        evidence = prior.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != {
            "ndvi_drop", "nbr_drop", "vh_delta_db"
        }:
            raise ValueError("concordance comparison lacks its exact evidence fields")
        try:
            ndvi_drop = float(evidence["ndvi_drop"])
            nbr_drop = float(evidence["nbr_drop"])
            vh_delta_db = float(evidence["vh_delta_db"])
        except (TypeError, ValueError) as exc:
            raise ValueError("concordance evidence must be numeric") from exc
        if not all(math.isfinite(value) for value in (ndvi_drop, nbr_drop, vh_delta_db)):
            raise ValueError("concordance evidence must be finite")
        if prior.get("thresholds") != {
            "optical_change_ndvi_drop_gt": 0.10,
            "sar_change_vh_delta_db_lt": -1.0,
        }:
            raise ValueError("concordance comparison has inconsistent thresholds")
        concordance = _cross_modal_concordance(ndvi_drop, vh_delta_db)
        if prior.get("concordance") != concordance:
            raise ValueError("concordance comparison contains an inconsistent class")
        result = {
            "rule": rule,
            "concordance": concordance,
            "predicted_class": concordance,
            "evidence": {
                "ndvi_drop": ndvi_drop,
                "nbr_drop": nbr_drop,
                "vh_delta_db": vh_delta_db,
            },
            "thresholds": {
                "optical_change_ndvi_drop_gt": 0.10,
                "sar_change_vh_delta_db_lt": -1.0,
            },
        }
        return store_result_handle(
            state,
            "class",
            "classification",
            TOOL_NAME,
            args,
            work,
            "deforestation_raster_tools",
            result,
        )
    raise AssertionError(f"unhandled classification rule: {rule}")


TOOL_NAME = "ClassifyChange"
DEFAULT_PORT = 20108
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class ClassifyChangeWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `ClassifyChange` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_ClassifyChange, require_existing=True)

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

    worker = ClassifyChangeWorker(
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
