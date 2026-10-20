import argparse
import copy
import uuid
from typing import Any

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import get_tool_instruction
from tool_server.tool_workers.online_workers.final_dataset_worker_state import (
    new_handle,
    require_artifact,
    require_question,
    run_stateful_tool,
)
from tool_server.utils.server_utils import build_logger


BROAD_MODE_ALIASES: dict[str, str] = {
    "forest_change": "forest",
    "forest_timeseries": "forest",
    "forest_false_positive": "forest",
    "baseline_canopy": "forest",
    "forest_loss_history": "forest",
    "forest_cross_modal": "forest",
    "sar_change": "forest",
    "wildfire_burn": "hazard",
    "forest_loss": "hazard",
    "built_up_gain": "hazard",
    "drought_anomaly": "hazard",
    "pollution_peak": "hazard",
    "multisource_hazard": "hazard",
    "wildfire_burn_exposure": "hazard",
    "forest_loss_exposure": "hazard",
    "built_up_gain_exposure": "hazard",
    "drought_anomaly_exposure": "hazard",
    "pollution_peak_exposure": "hazard",
    "xbd_damage": "xbd_damage",
    "xbd_grid": "xbd_damage",
    "xbd_quadrants": "xbd_damage",
    "xbd_scene_summary": "xbd_damage",
    "flood_safe_routing": "routing",
    "sar_flood_reference": "sar_flood",
    "sar_flood_diagnostic": "sar_flood",
}


MODE_ALLOWED_SLOTS: dict[str, set[str]] = {
    "forest": {"optical", "sar", "forest", "loss_history"},
    "hazard": {"hazard", "sources", "exposure"},
    "xbd_damage": {"image", "labels", "masks"},
    "sar_flood": {"sar", "water"},
    "routing": {"routing"},
    "scene_compare": {"scene_a", "scene_b"},
}


MODE_CORE_SLOTS: dict[str, set[str]] = {
    "forest": {"optical", "sar", "forest", "loss_history"},
    "hazard": {"hazard"},
    "xbd_damage": {"image", "labels", "masks"},
    "sar_flood": {"sar"},
    "routing": {"routing"},
    "scene_compare": {"scene_a", "scene_b"},
}


def process_PrepareScene(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    require_question(state, args)
    inputs = args.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("PrepareScene requires a non-empty inputs object")
    raw_mode = args.get("mode")
    if not raw_mode:
        raise ValueError("PrepareScene requires mode")
    mode = BROAD_MODE_ALIASES.get(str(raw_mode), str(raw_mode))
    input_cards = {slot: require_artifact(state, str(handle)) for slot, handle in inputs.items()}
    non_scene_inputs = {
        slot: handle
        for slot, handle in inputs.items()
        if input_cards[slot].get("type") != "scene"
    }
    if non_scene_inputs:
        raise ValueError(f"PrepareScene expects scene handles from ReadData, got non-scene inputs: {non_scene_inputs}")
    allowed_slots = MODE_ALLOWED_SLOTS.get(str(mode))
    core_slots = MODE_CORE_SLOTS.get(str(mode), set())
    provided_slots = set(inputs)
    if allowed_slots is None:
        raise ValueError(f"PrepareScene unknown broad mode: {mode}")
    extra_slots = sorted(provided_slots - allowed_slots)
    if extra_slots:
        raise ValueError(f"PrepareScene mode {mode} received unsupported input slots: {extra_slots}")
    missing_core_slots = sorted(core_slots) if not (provided_slots & core_slots) else []
    if missing_core_slots:
        raise ValueError(f"PrepareScene mode {mode} needs at least one of these core input slots: {missing_core_slots}")
    if mode == "scene_compare" and not {"scene_a", "scene_b"} <= provided_slots:
        raise ValueError("PrepareScene mode scene_compare requires both scene_a and scene_b")
    checks = args.get("checks") or {}
    if not isinstance(checks, dict):
        raise ValueError("PrepareScene checks must be an object")
    if mode == "sar_flood":
        unknown_checks = sorted(set(checks) - {"grid"})
        if unknown_checks:
            raise ValueError(f"unsupported SAR-flood preparation checks: {unknown_checks}")
        if "grid" in checks and checks["grid"] != "4x4":
            raise ValueError("SAR-flood grid check supports only '4x4'")
    input_types = {slot: card.get("source_type") or card.get("type") for slot, card in input_cards.items()}
    input_sources = {slot: card.get("source_input") for slot, card in input_cards.items()}
    input_roles = {slot: card.get("role") for slot, card in input_cards.items()}
    generated_quality_reports = {
        slot: copy.deepcopy(card.get("generated_quality_report"))
        for slot, card in input_cards.items()
        if card.get("generated_quality_report") is not None
    }
    generated_quality_report_handles = {
        slot: card.get("generated_quality_report_handle")
        for slot, card in input_cards.items()
        if card.get("generated_quality_report_handle")
    }
    input_path_status = {
        slot: {
            "path_count": (card.get("path_summary") or {}).get("path_count", 0),
            "missing_required_path_count": (card.get("path_summary") or {}).get("missing_required_path_count", 0),
            "suffix_counts": (card.get("path_summary") or {}).get("suffix_counts", {}),
        }
        for slot, card in input_cards.items()
    }
    ready = all(v["missing_required_path_count"] == 0 for v in input_path_status.values())
    handle = new_handle(
        state,
        "work",
        {
            "type": "work",
            "mode": mode,
            "requested_mode": raw_mode,
            "checks": copy.deepcopy(checks),
            "inputs": copy.deepcopy(inputs),
            "allowed_slots": sorted(allowed_slots),
            "core_slots": sorted(core_slots),
            "extra_slots": extra_slots,
            "input_types": input_types,
            "input_sources": input_sources,
            "input_roles": input_roles,
            "generated_quality_reports": generated_quality_reports,
            "generated_quality_report_handles": generated_quality_report_handles,
            "input_path_status": input_path_status,
            "ready": ready,
        },
    )
    artifact = state["artifacts"][handle]
    return {
        "status": "OK" if ready else "INPUT_PATH_WARNING",
        "session_id": state["session_id"],
        "handle": handle,
        "canonical_handle": handle,
        "global_artifact_id": artifact.get("global_artifact_id"),
        "artifact_json_path": artifact.get("artifact_json_path"),
        "mode": mode,
        "requested_mode": raw_mode,
        "inputs": inputs,
        "allowed_slots": sorted(allowed_slots),
        "core_slots": sorted(core_slots),
        "extra_slots": extra_slots,
        "input_types": input_types,
        "input_sources": input_sources,
        "input_roles": input_roles,
        "generated_quality_report_handles": generated_quality_report_handles,
        "input_path_status": input_path_status,
        "ready": ready,
    }


TOOL_NAME = "PrepareScene"
DEFAULT_PORT = 20103
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class PrepareSceneWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `PrepareScene` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_PrepareScene, require_existing=True)

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

    worker = PrepareSceneWorker(
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
