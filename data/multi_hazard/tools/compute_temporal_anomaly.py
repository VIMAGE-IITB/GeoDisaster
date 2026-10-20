from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ._tool_common import error_response, load_handle_array, parse_sample, read_baseline_csv


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        domain, sid = parse_sample(args)
        modality = str(args["modality"])
        target_handle = str(args.get("target_handle") or "")
        if not target_handle:
            raise ValueError("target_handle is required")
        target_date = str(args.get("target_date") or "")
        rows = read_baseline_csv(domain, sid, modality)
        baseline = [float(r["value"]) for r in rows if str(r.get("is_target", "")).lower() != "true"]
        target_rows = [r for r in rows if str(r.get("is_target", "")).lower() == "true"]
        if len(baseline) < 2:
            raise ValueError("at least two non-target baseline observations are required")
        if len(target_rows) != 1:
            raise ValueError("the baseline table must declare exactly one target date row")
        declared_target_date = str(target_rows[0].get("date") or "")
        if target_date and target_date != declared_target_date:
            raise ValueError("target_date does not match the declared baseline target row")
        target_date = declared_target_date
        values, sidecar = load_handle_array(domain, sid, target_handle)
        values = np.asarray(values, dtype=float)
        if values.ndim == 3:
            if values.shape[0] != 1:
                raise ValueError("temporal anomaly target must be single-band")
            values = values[0]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("temporal anomaly target has no finite pixels")
        if modality == "modis_ndvi_anomaly" and (
            float(np.min(finite)) < -1.0 or float(np.max(finite)) > 1.0
        ):
            raise ValueError("MODIS NDVI target values must be within [-1, 1]")
        target_value = float(np.mean(finite))
        mean = float(np.mean(baseline))
        std = float(np.std(baseline, ddof=1))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("baseline mean/std must be finite and std must be positive")
        z = (target_value - mean) / std
        if z <= -2.5:
            cls = "extreme_negative"
        elif z <= -1.5:
            cls = "severe_negative"
        elif z <= -0.75:
            cls = "mild_negative"
        elif z >= 1.5:
            cls = "positive"
        else:
            cls = "normal"
        return {
            "aoi_id": f"{domain}__{sid}",
            "modality": modality,
            "target_handle": target_handle,
            "target_date": target_date,
            "target_value": round(float(target_value), 6),
            "target_valid_pixel_count": int(finite.size),
            "target_source": sidecar.get("source"),
            "baseline_mean": round(mean, 6),
            "baseline_std": round(std, 6),
            "baseline_years_used": len({r["date"][:4] for r in rows if str(r.get("is_target", "")).lower() != "true"}),
            "n_baseline_observations": len(baseline),
            "z_score": round(float(z), 6),
            "anomaly_class": cls,
            "method": args.get("method", "z_score"),
            "target_statistic": "finite_aoi_pixel_mean",
        }
    except Exception as exc:
        return error_response(exc)

