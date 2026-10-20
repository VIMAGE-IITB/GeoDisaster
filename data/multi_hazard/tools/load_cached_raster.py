from __future__ import annotations

from typing import Any, Dict

from ._tool_common import ToolError, error_response, load_handle_array, parse_sample, resolve_handle


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        domain, sid = parse_sample(args)
        handle = str(args["handle"])
        path, sidecar = resolve_handle(domain, sid, handle)
        arr, _ = load_handle_array(domain, sid, handle)
        return {
            "handle": handle,
            "aoi_id": f"{domain}__{sid}",
            "modality": sidecar["modality"],
            "source": sidecar["source"],
            "units": sidecar["units"],
            "geotiff_path": str(path),
            "scale_m": sidecar["scale_m"],
            "shape": [int(arr.shape[-2]), int(arr.shape[-1])],
            "n_bands": int(arr.shape[0]) if arr.ndim == 3 else 1,
            "acquisition_dates": sidecar.get("acquisition_dates", []),
            "clear_fraction_in_aoi": sidecar.get("clear_fraction_in_aoi"),
            "fetched_at_build_time": sidecar.get("fetched_at_build_time"),
            "source_mode": sidecar.get("source_mode"),
        }
    except Exception as exc:
        return error_response(exc)

