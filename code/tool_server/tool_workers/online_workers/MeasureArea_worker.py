import argparse
import re
import uuid
from pathlib import Path
from typing import Any

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import get_tool_instruction
from tool_server.tool_workers.online_workers.final_dataset_worker_state import (
    all_lineage_path_entries,
    artifact_result,
    deforestation_tools,
    dir_from_path_parts,
    jsonable,
    load_json,
    require_artifact,
    require_question,
    run_stateful_tool,
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


def _event_year_from_question(question_card: dict[str, Any]) -> int | None:
    text = str(question_card.get("text") or question_card.get("question") or "")
    event_match = re.search(r"\bevent\s+year\s+(20\d{2})\b", text, flags=re.IGNORECASE)
    if event_match:
        return int(event_match.group(1))
    years = [int(x) for x in re.findall(r"\b20\d{2}\b", text)]
    if years:
        return years[0]
    metadata = question_card.get("metadata") or {}
    year = metadata.get("event_year") or metadata.get("year")
    return int(year) if year else None


def _question_context(state: dict[str, Any], question_card: dict[str, Any]) -> dict[str, Any]:
    """Merge the lightweight question handle with the full case question."""
    case_question = (state.get("case") or {}).get("question") or {}
    if not isinstance(case_question, dict):
        case_question = {}
    return {**case_question, **question_card}


def _deforestation_area(
    sample_dir: Path,
    event_year: int | None,
    generated_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not generated_report or (
        event_year is not None
        and int(generated_report.get("event_year") or -1) != int(event_year)
    ):
        generated_report = deforestation_tools().BuildQualityReport(
            str(sample_dir), event_year
        )
    metrics = generated_report["loss_event_year"]
    return {
        "target": "forest_loss_patches",
        "event_year": generated_report["event_year"],
        "mask_kind": metrics["mask_kind"],
        "pixel_count": metrics["loss_pixel_count"],
        "area_m2": metrics["area_m2"],
        "total_area_ha": metrics["area_ha"],
        "measurement_method": metrics["measurement_method"],
        "fallback_used": metrics["fallback_used"],
        "crs": metrics["crs"],
        "shape": metrics["shape"],
        "component_count": metrics["component_count"],
        "largest_patch_pixels": metrics["largest_patch_pixels"],
        "mask_path": metrics["mask_path"],
        "generated_quality_report": generated_report,
    }


def process_MeasureArea(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    question = _question_context(state, require_question(state, args))
    work = str(args.get("work") or "")
    require_artifact(state, work)
    if source_of(state) != "deforestation_data":
        raise ValueError(f"MeasureArea is not implemented for source {source_of(state)}")
    if args.get("target") != "forest_loss":
        raise ValueError(f"unsupported MeasureArea target: {args.get('target')!r}")
    if args.get("unit") != "hectares":
        raise ValueError(f"unsupported MeasureArea unit: {args.get('unit')!r}")
    sample_dir = _deforestation_sample_dir(state, work)
    prior = artifact_result(state, work)
    generated_report = (
        prior.get("generated_quality_report")
        if isinstance(prior, dict)
        else None
    )
    measured = _deforestation_area(
        sample_dir,
        _event_year_from_question(question),
        generated_report,
    )
    unit = str(args.get("unit") or "hectares").lower()
    area_ha = float(measured["total_area_ha"])
    area_m2 = float(measured["area_m2"])
    area = area_m2 if unit in {"m2", "sqm", "square_meters", "square_metres"} else area_ha
    result = {
        "target": args.get("target"),
        "unit": args.get("unit"),
        "area": area,
        "area_ha": area_ha,
        "area_m2": area_m2,
        "source": measured,
        "input_object_summary": {
            "target": prior.get("target"),
            "component_count": prior.get("component_count"),
            "largest_patch_pixels": prior.get("largest_patch_pixels"),
        } if isinstance(prior, dict) else None,
    }
    return store_result_handle(state, "area", "area_measurement", TOOL_NAME, args, work, "deforestation_raster_tools", result)


TOOL_NAME = "MeasureArea"
DEFAULT_PORT = 20105
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{TOOL_NAME}_worker_{worker_id}.log")


class MeasureAreaWorker(BaseToolWorker):
    """Online worker for the final multi-source dataset `MeasureArea` tool."""

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
        return run_stateful_tool(TOOL_NAME, payload, process_MeasureArea, require_existing=True)

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

    worker = MeasureAreaWorker(
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
