from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ._tool_common import (
    classify_array,
    comparison_mask,
    error_response,
    load_handle_array,
    parse_sample,
    pixel_area_ha,
)


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        domain, sid = parse_sample(args)
        handle = str(args["handle"])
        arr, sidecar = load_handle_array(domain, sid, handle)
        if arr.ndim == 3:
            arr = arr[0]
        arr = np.asarray(arr, dtype=float)
        analysis_mask = np.ones(arr.shape, dtype=bool)
        mask_layer = args.get("mask_layer")
        if mask_layer:
            analysis_mask = comparison_mask(domain, sid, str(mask_layer))
            _, mask_sidecar = load_handle_array(domain, sid, str(mask_layer))
            if analysis_mask.shape != arr.shape:
                raise ValueError(
                    f"analysis_mask_shape_mismatch:{mask_layer}:{analysis_mask.shape}:{arr.shape}"
                )
            value_profile = sidecar.get("profile") or {}
            mask_profile = mask_sidecar.get("profile") or {}
            if str(value_profile.get("crs")) != str(mask_profile.get("crs")):
                raise ValueError(f"analysis_mask_crs_mismatch:{mask_layer}")
            if value_profile.get("transform") != mask_profile.get("transform"):
                raise ValueError(f"analysis_mask_transform_mismatch:{mask_layer}")
        valid = np.isfinite(arr) & analysis_mask
        values = arr[valid]
        if not values.size:
            raise ValueError("analysis_population_empty")
        pct_names = args.get("percentiles") or [5, 50, 95]
        percentiles = {
            f"p{int(p):02d}": round(float(np.nanpercentile(values, float(p))), 6)
            for p in pct_names
            if values.size
        }
        px_ha = pixel_area_ha(sidecar, domain, sid)
        out: Dict[str, Any] = {
            "handle": handle,
            "aoi_id": f"{domain}__{sid}",
            "modality": sidecar["modality"],
            "units": sidecar["units"],
            "n_valid_pixels": int(values.size),
            "nan_pct": round(float((~valid).mean() * 100.0), 6),
            "mean": None if values.size == 0 else round(float(np.nanmean(values)), 6),
            "median": None if values.size == 0 else round(float(np.nanmedian(values)), 6),
            "std": None if values.size == 0 else round(float(np.nanstd(values)), 6),
            "min": None if values.size == 0 else round(float(np.nanmin(values)), 6),
            "max": None if values.size == 0 else round(float(np.nanmax(values)), 6),
            "percentiles": percentiles,
            "mask_applied": mask_layer,
            "mask_pixel_count": int(np.count_nonzero(analysis_mask)) if mask_layer else None,
            "analysis_scope": args.get("analysis_scope") or ("mask" if mask_layer else "whole_aoi"),
            "pixel_area_m2": round(px_ha * 10000.0, 3),
        }
        by_class = args.get("by_class")
        if by_class:
            classified = classify_array(
                arr,
                by_class,
                px_ha,
                domain,
                sid,
                sidecar=sidecar,
                analysis_mask=analysis_mask,
            )
            classified_count = sum(
                int(row["pixel_count"])
                for row in classified["by_class"].values()
            )
            if classified_count != int(values.size):
                raise ValueError(
                    f"classification_does_not_cover_analysis_population:"
                    f"{classified_count}:{int(values.size)}"
                )
            dominant = classified["dominant_class"]
            dominant_count = int(classified["by_class"][dominant]["pixel_count"])
            out.update(classified)
            out["dominant_fraction"] = round(dominant_count / classified_count, 6)
            out["classification_contract"] = "complete_half_open_intervals_v1"
            out["tie_break_rule"] = "first class in declared interval order"
        return out
    except Exception as exc:
        return error_response(exc)
