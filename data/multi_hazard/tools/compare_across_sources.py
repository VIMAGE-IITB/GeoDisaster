from __future__ import annotations

from itertools import combinations
from typing import Any, Dict

import numpy as np

from ._tool_common import (
    air_pollution_normalization,
    comparison_mask,
    error_response,
    load_handle_array,
    mask_area_ha,
    parse_sample,
    source_label,
)


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        domain, sid = parse_sample(args)
        handles = [str(h) for h in args["handles"]]
        metric = str(args.get("metric", "area_agreement"))
        masks = {}
        arrays = {}
        valid_pixels = {}
        grid_signatures = {}
        exact_areas: Dict[str, float] = {}
        per_source: Dict[str, Dict[str, Any]] = {}
        for handle in handles:
            mask = comparison_mask(domain, sid, handle)
            raw, sidecar = load_handle_array(domain, sid, handle)
            raw = np.asarray(raw, dtype=float)
            if raw.ndim == 3 and raw.shape[0] == 1:
                raw = raw[0]
            if raw.ndim != 2:
                raise ValueError(f"correlation_requires_single_band_raster:{handle}:{raw.shape}")
            profile = sidecar.get("profile") or {}
            valid = np.isfinite(raw)
            nodata = profile.get("nodata")
            if nodata is not None and np.isfinite(float(nodata)):
                valid &= raw != float(nodata)
            if not np.any(valid):
                raise ValueError(f"no_valid_pixels:{handle}")
            if domain in {"wildfire", "urban_expansion"}:
                unique = set(np.unique(raw[valid]).tolist())
                if not unique <= {0.0, 1.0}:
                    raise ValueError(f"comparison_mask_not_binary:{domain}:{handle}:{sorted(unique)}")
            if domain == "deforestation":
                unique = set(np.unique(raw[valid]).tolist())
                allowed = {0.0, 1.0, 2.0} if handle == "radd_alerts__event" else {0.0, 1.0}
                if not unique <= allowed:
                    raise ValueError(
                        f"deforestation_comparison_mask_value_invalid:{handle}:{sorted(unique)}"
                    )
            label = source_label(handle)
            masks[handle] = mask
            arrays[handle] = raw
            valid_pixels[handle] = valid
            grid_signatures[handle] = {
                "shape": [int(raw.shape[0]), int(raw.shape[1])],
                "crs": str(profile.get("crs")),
                "transform": [float(value) for value in profile.get("transform")[:6]],
            }
            exact_area = mask_area_ha(mask, sidecar)
            exact_areas[label] = exact_area
            per_source[label] = {
                "handle": handle,
                "area_ha": round(exact_area, 3),
                "scale_m": sidecar["scale_m"],
                "units": sidecar["units"],
            }
        if metric == "benchmark_relative_intensity_agreement":
            contract = air_pollution_normalization(sid)
            boundaries = [float(value) for value in contract["state_z_boundaries"]]
            required_fraction = float(contract["required_agreement_fraction"])
            source_values: Dict[str, float] = {}
            source_z_scores: Dict[str, float] = {}
            source_states: Dict[str, str] = {}
            for handle in handles:
                label = source_label(handle)
                values = arrays[handle][valid_pixels[handle]]
                if values.size == 0:
                    raise ValueError(f"no_valid_pixels:{handle}")
                value = float(np.mean(values))
                fit = contract["sources"][handle]
                spread = float(fit["aoi_mean_std"])
                if not np.isfinite(spread) or spread <= 0:
                    raise ValueError(f"invalid_train_spread:{handle}")
                z_score = (value - float(fit["aoi_mean_mean"])) / spread
                if z_score < boundaries[0]:
                    state = "below_training_typical"
                elif z_score > boundaries[1]:
                    state = "above_training_typical"
                else:
                    state = "training_typical"
                source_values[label] = round(value, 9)
                source_z_scores[label] = round(float(z_score), 6)
                source_states[label] = state
            state_counts = {
                state: list(source_states.values()).count(state)
                for state in sorted(set(source_states.values()))
            }
            agreement_fraction = max(state_counts.values()) / max(1, len(source_states))
            verdict = (
                "sources_agree"
                if agreement_fraction >= required_fraction
                else "sources_disagree"
            )
            return {
                "metric": metric,
                "aoi_id": f"{domain}__{sid}",
                "source_spatial_means": source_values,
                "source_units": {
                    label: row["units"] for label, row in per_source.items()
                },
                "source_train_z_scores": source_z_scores,
                "source_intensity_states": source_states,
                "state_counts": state_counts,
                "agreement_fraction": round(float(agreement_fraction), 6),
                "required_agreement_fraction": required_fraction,
                "state_z_boundaries": boundaries,
                "normalization_fit_split": contract["fit_split"],
                "normalization_train_sample_ids": contract["train_sample_ids"],
                "agree": verdict == "sources_agree",
                "verdict": verdict,
                "n_sources_used": len(handles),
            }
        pairwise_pct_delta = {}
        ious = {}
        state_agreements = {}
        kappas = {}
        correlations = {}
        correlation_details = {}
        for a, b in combinations(handles, 2):
            la, lb = source_label(a), source_label(b)
            area_a = exact_areas[la]
            area_b = exact_areas[lb]
            denom = max((area_a + area_b) / 2.0, 1e-6)
            pairwise_pct_delta[f"{la}__{lb}"] = round(float(abs(area_b - area_a) / denom * 100.0), 3)
            if grid_signatures[a] != grid_signatures[b]:
                raise ValueError(f"comparison_grid_mismatch:{a}:{b}")
            common_valid = valid_pixels[a] & valid_pixels[b]
            if not np.any(common_valid):
                raise ValueError(f"no_common_valid_pixels:{a}:{b}")
            ma = masks[a][common_valid].astype(bool)
            mb = masks[b][common_valid].astype(bool)
            union = np.count_nonzero(ma | mb)
            ious[f"{la}__{lb}"] = round(float(np.count_nonzero(ma & mb) / union), 6) if union else 1.0
            pair_key = f"{la}__{lb}"
            state_agreements[pair_key] = round(float(np.mean(ma == mb)), 6)
            observed = float(np.mean(ma == mb))
            pa = float(np.mean(ma))
            pb = float(np.mean(mb))
            expected = pa * pb + (1.0 - pa) * (1.0 - pb)
            kappas[pair_key] = round((observed - expected) / (1.0 - expected), 6) if expected < 1.0 else 1.0
            if metric in {"correlation", "spatial_pearson_correlation"}:
                aa = arrays[a][common_valid]
                bb = arrays[b][common_valid]
                if aa.size < 2:
                    correlation = None
                    reason = "fewer_than_two_common_valid_pixels"
                elif float(np.ptp(aa)) == 0.0 or float(np.ptp(bb)) == 0.0:
                    correlation = None
                    reason = "constant_source"
                else:
                    correlation = round(float(np.corrcoef(aa, bb)[0, 1]), 6)
                    reason = None
                correlations[pair_key] = correlation
                correlation_details[pair_key] = {
                    "valid_pixel_count": int(aa.size),
                    "defined": correlation is not None,
                    "undefined_reason": reason,
                }
            else:
                aa = ma.astype(float)
                bb = mb.astype(float)
                correlation = (
                    round(float(np.corrcoef(aa, bb)[0, 1]), 6)
                    if float(np.ptp(aa)) > 0.0 and float(np.ptp(bb)) > 0.0
                    else None
                )
                correlations[pair_key] = correlation
                correlation_details[pair_key] = {
                    "valid_pixel_count": int(aa.size),
                    "defined": correlation is not None,
                    "undefined_reason": None if correlation is not None else "constant_mask",
                }
        deltas = [abs(v) for v in pairwise_pct_delta.values()]
        median_delta = float(np.median(deltas)) if deltas else 0.0
        median_area = float(np.median(list(exact_areas.values()))) if exact_areas else 0.0
        valid_correlations = [value for value in correlations.values() if value is not None]
        required_pair_count = len(correlations)
        if metric in {"correlation", "spatial_pearson_correlation"}:
            score = float(np.median(valid_correlations)) if valid_correlations else None
            minimum_correlation = float(args.get("minimum_correlation", 0.65))
            require_all_pairs_defined = bool(args.get("require_all_pairs_defined", True))
            if require_all_pairs_defined and len(valid_correlations) != required_pair_count:
                verdict = "insufficient_variation"
            else:
                verdict = (
                    "sources_agree"
                    if score is not None and score >= minimum_correlation
                    else "sources_disagree"
                )
        elif metric == "iou":
            score = float(np.median(list(ious.values()))) if ious else 1.0
            verdict = "sources_agree" if score >= float(args.get("minimum_iou", 0.50)) else "sources_disagree"
        elif metric == "categorical_kappa":
            score = float(np.median(list(kappas.values()))) if kappas else 1.0
            verdict = "sources_agree" if score >= float(args.get("minimum_kappa", 0.50)) else "sources_disagree"
        elif metric in {"sign_agreement", "thresholded_drought_state_agreement"}:
            score = float(np.median(list(state_agreements.values()))) if state_agreements else 1.0
            verdict = "sources_agree" if score >= float(args.get("minimum_agreement", 0.70)) else "sources_disagree"
        else:
            score = max(0.0, min(100.0, 100.0 - median_delta))
            verdict = "sources_agree" if median_delta <= float(args.get("maximum_median_pct_delta", 30.0)) else "sources_disagree"
        return {
            "metric": metric,
            "aoi_id": f"{domain}__{sid}",
            "per_source": per_source,
            "pairwise_pct_delta": pairwise_pct_delta,
            "pairwise_iou": ious,
            "pairwise_state_agreement": state_agreements,
            "pairwise_categorical_kappa": kappas,
            "pairwise_correlation": correlations,
            "median_area_ha": round(median_area, 3),
            "median_pct_delta": round(median_delta, 3),
            "agreement_class": (
                "high"
                if verdict == "sources_agree"
                else ("insufficient" if verdict == "insufficient_variation" else "low")
            ),
            "verdict": verdict,
            "n_sources_used": len(handles),
            "weakest_source": max(per_source.items(), key=lambda kv: abs(kv[1]["area_ha"] - median_area))[0],
            "score": None if score is None else round(float(score), 6),
            "score_contract": (
                "bounded_0_100_area_agreement_v1"
                if metric not in {
                    "correlation",
                    "spatial_pearson_correlation",
                    "iou",
                    "categorical_kappa",
                    "sign_agreement",
                    "thresholded_drought_state_agreement",
                }
                else metric
            ),
            "median_correlation": (
                None if score is None else round(float(score), 6)
            ) if metric in {"correlation", "spatial_pearson_correlation"} else None,
            "correlation_basis": (
                "co_registered_continuous_valid_pixels"
                if metric in {"correlation", "spatial_pearson_correlation"}
                else "thresholded_binary_masks"
            ),
            "pairwise_correlation_details": correlation_details,
            "valid_correlation_pair_count": len(valid_correlations),
            "required_correlation_pair_count": required_pair_count,
            "undefined_correlation_pairs": [
                key for key, value in correlations.items() if value is None
            ],
            "minimum_median_correlation": (
                float(args.get("minimum_correlation", 0.65))
                if metric in {"correlation", "spatial_pearson_correlation"}
                else None
            ),
            "require_all_pairs_defined": (
                bool(args.get("require_all_pairs_defined", True))
                if metric in {"correlation", "spatial_pearson_correlation"}
                else None
            ),
            "grid_signatures": grid_signatures,
        }
    except Exception as exc:
        return error_response(exc)
