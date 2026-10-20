"""Shared, answer-safe Sentinel-1 flood prediction runtime.

Only the declared Sentinel-1 image is used to make the flood prediction.
Hand labels and cached label-derived masks are evaluator data and are never
consulted here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tool_server.tool_workers.online_workers.final_dataset_worker_state import (
    all_lineage_path_entries,
    artifact_output_dir,
    artifact_result,
    first_existing_path,
    jsonable,
    sen1_utils,
)


METHOD = "s1_min_backscatter_otsu_v1"
SUPPORTED_RENDER_STYLE = "flood_diagnostic"
SUPPORTED_RENDER_LAYERS = {"sar", "flood_mask", "components", "grid", "jrc"}


def paths(state: dict[str, Any], work: str) -> dict[str, Path | None]:
    entries = all_lineage_path_entries(state, work)
    return {
        "s1": first_existing_path(entries, "_s1portable.tif")
        or first_existing_path(entries, "_s1hand.tif")
        or first_existing_path(entries, "s1_image"),
        "jrc": first_existing_path(entries, "_jrcwaterhand.tif")
        or first_existing_path(entries, "jrc"),
    }


def _chip_id(state: dict[str, Any], s1_path: Path) -> str:
    case = state.get("case") or {}
    fallback = s1_path.stem.replace("_S1Portable", "").replace("_S1Hand", "")
    return str(case.get("sample_id") or case.get("id") or fallback)


def _largest_bbox(mask, transform) -> list[float] | None:
    import numpy as np

    labels, _count, sizes = sen1_utils().connected_components(mask)
    if not sizes:
        return None
    rows, cols = np.where(labels == int(np.argmax(sizes)) + 1)
    x0, y0 = transform * (int(cols.min()), int(rows.min()))
    x1, y1 = transform * (int(cols.max()) + 1, int(rows.max()) + 1)
    return [round(min(x0, x1), 8), round(min(y0, y1), 8), round(max(x0, x1), 8), round(max(y0, y1), 8)]


def prediction_report(state: dict[str, Any], work: str) -> dict[str, Any]:
    """Return an S1-derived report, reusing only a prior report from this method."""
    prior = artifact_result(state, work)
    if isinstance(prior, dict):
        report = prior.get("mask_report")
        if isinstance(report, dict) and report.get("method") == METHOD:
            return report

    import numpy as np
    import rasterio

    resolved = paths(state, work)
    s1_path = resolved.get("s1")
    if not s1_path:
        raise ValueError("no declared Sentinel-1 raster is available")
    utils = sen1_utils()
    mask, details, ds = utils.compute_otsu_mask_from_s1(s1_path)
    try:
        bands = [ds.read(i).astype("float32") for i in range(1, ds.count + 1)]
        valid = np.logical_and.reduce([np.isfinite(band) for band in bands])
        nodata = ds.nodata
        if nodata is not None and np.isfinite(float(nodata)):
            valid &= np.logical_and.reduce([band != float(nodata) for band in bands])
        mask &= valid
        out_path = artifact_output_dir(state) / "sentinel1_predicted_flood_mask.tif"
        utils.write_binary_mask(out_path, mask, ds, valid)
        report = utils.compute_mask_report(
            _chip_id(state, s1_path), mask, valid, ds, "computed", str(out_path), METHOD
        )
        report.update(
            {
                "prediction_mask_path": str(out_path),
                "prediction_source_path": str(s1_path),
                "prediction_threshold": round(float(details["threshold"]), 8),
                "prediction_score": details["score"],
                "prediction_provenance": "computed directly from declared Sentinel-1 bands; no hand label or cached target mask used",
                "largest_component_bbox": _largest_bbox(mask, ds.transform),
                "spatial_pattern": "localized" if float(report["concentration_score"]) >= 0.5 else "dispersed",
            }
        )
        return jsonable(report)
    finally:
        ds.close()


def prediction_mask(report: dict[str, Any]):
    import rasterio

    path = report.get("prediction_mask_path") or report.get("mask_path")
    if not path:
        raise ValueError("prediction report has no generated mask path")
    with rasterio.open(path) as ds:
        return ds.read(1) == 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_water_overlap(report: dict[str, Any], jrc_path: Path | None) -> dict[str, Any]:
    if not jrc_path:
        return {"available": False}
    import numpy as np
    import rasterio

    mask_path = Path(str(report.get("prediction_mask_path") or report.get("mask_path") or ""))
    with rasterio.open(mask_path) as prediction_ds:
        flood = prediction_ds.read(1) == 1
        prediction_crs = prediction_ds.crs
        prediction_transform = prediction_ds.transform
    with rasterio.open(jrc_path) as jrc_ds:
        values = jrc_ds.read(1)
        jrc = values == 1
        same_grid = (
            flood.shape == values.shape
            and prediction_crs == jrc_ds.crs
            and np.allclose(tuple(prediction_transform), tuple(jrc_ds.transform), rtol=0.0, atol=1e-12)
        )
        jrc_nodata = jrc_ds.nodata
    if not same_grid:
        raise ValueError("prediction/JRC grid, CRS, or transform mismatch")
    allowed = {0.0, 1.0}
    if jrc_nodata is not None:
        allowed.add(float(jrc_nodata))
    observed = {float(value) for value in np.unique(values)}
    if not observed <= allowed:
        raise ValueError(f"unsupported JRC known-water values: {sorted(observed)}")
    total = int(flood.sum())
    overlap = int((flood & jrc).sum())
    outside = int((flood & ~jrc).sum())
    return {
        "available": True,
        "overlap_pixels": overlap,
        "outside_reference_pixels": outside,
        "overlap_pct_of_flood": round(overlap / total * 100.0, 6) if total else 0.0,
        "outside_reference_pct_of_flood": round(outside / total * 100.0, 6) if total else 0.0,
        "reference": "JRC known-water context (not flood ground truth)",
        "reference_grid_validated": True,
        "reference_values": sorted(observed),
    }


def render_prediction(
    state: dict[str, Any],
    work: str,
    layers: dict[str, Any],
    style: str,
) -> dict[str, Any]:
    """Render exactly the requested layers and emit a validation manifest."""
    import numpy as np
    import rasterio
    from PIL import Image, ImageDraw

    if style != SUPPORTED_RENDER_STYLE:
        raise ValueError(
            f"unsupported Sentinel-1 render style {style!r}; expected {SUPPORTED_RENDER_STYLE!r}"
        )
    if not isinstance(layers, dict) or not layers:
        raise ValueError("Sentinel-1 RenderMap requires a non-empty layers object")
    unknown = sorted(set(layers) - SUPPORTED_RENDER_LAYERS)
    if unknown:
        raise ValueError(f"unsupported Sentinel-1 render layers: {unknown}")
    non_boolean = sorted(key for key, value in layers.items() if not isinstance(value, bool))
    if non_boolean:
        raise ValueError(f"Sentinel-1 render layer toggles must be boolean: {non_boolean}")
    requested = {key: bool(layers.get(key, False)) for key in sorted(SUPPORTED_RENDER_LAYERS)}
    if not any(requested.values()):
        raise ValueError("Sentinel-1 RenderMap requires at least one enabled layer")

    report = prediction_report(state, work)
    resolved = paths(state, work)
    with rasterio.open(resolved["s1"]) as ds:
        band = ds.read(1).astype("float32")
        nodata = ds.nodata
    valid = np.isfinite(band)
    if nodata is not None and np.isfinite(float(nodata)):
        valid &= band != float(nodata)
    lo, hi = np.percentile(band[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    gray = np.clip((band - lo) / max(float(hi - lo), 1e-6), 0, 1)
    gray[~valid] = 0.0
    base_rgb = np.repeat((gray * 255).astype("uint8")[..., None], 3, axis=2)
    rgb = base_rgb.copy() if requested["sar"] else np.zeros_like(base_rgb)
    flood = prediction_mask(report)
    if requested["flood_mask"]:
        rgb[flood] = (0.35 * rgb[flood] + 0.65 * np.array([0, 255, 255])).astype("uint8")
    jrc_path = resolved.get("jrc")
    jrc_pixels = 0
    if requested["jrc"]:
        if not jrc_path:
            raise ValueError("JRC render layer requested but no declared JRC raster is available")
        with rasterio.open(jrc_path) as ds:
            jrc = ds.read(1) == 1
        jrc_pixels = int(jrc.sum())
        rgb[jrc & ~flood] = (0.55 * rgb[jrc & ~flood] + 0.45 * np.array([40, 80, 255])).astype("uint8")

    boundary = np.zeros(flood.shape, dtype=bool)
    if requested["components"]:
        interior = flood.copy()
        interior[1:, :] &= flood[:-1, :]
        interior[:-1, :] &= flood[1:, :]
        interior[:, 1:] &= flood[:, :-1]
        interior[:, :-1] &= flood[:, 1:]
        boundary = flood & ~interior
        rgb[boundary] = np.array([255, 0, 255], dtype="uint8")

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if requested["grid"]:
        for i in range(1, 4):
            draw.line((i * width / 4, 0, i * width / 4, height), fill=(255, 255, 255), width=1)
            draw.line((0, i * height / 4, width, i * height / 4), fill=(255, 255, 255), width=1)
    out_path = artifact_output_dir(state) / "sentinel1_prediction_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    rendered = np.asarray(image)
    validation = {
        "current_work_handle": work,
        "requested_layers": requested,
        "rendered_layers": [key for key, enabled in requested.items() if enabled],
        "prepared_overlay_used": False,
        "image_bytes": out_path.stat().st_size,
        "image_sha256": _sha256(out_path),
        "changed_pixels_from_sar_base": int(np.any(rendered != base_rgb, axis=2).sum()),
        "predicted_flood_pixels": int(flood.sum()),
        "component_count": int(report.get("component_count") or 0),
        "component_boundary_pixels": int(boundary.sum()),
        "jrc_pixels_rendered": jrc_pixels,
        "prediction_mask_path": report.get("prediction_mask_path"),
        "prediction_mask_sha256": _sha256(Path(report["prediction_mask_path"])),
        "source_s1_path": str(resolved["s1"]),
    }
    manifest_path = artifact_output_dir(state) / "sentinel1_prediction_render_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "sen1_flood_render_manifest_v1",
                "style": style,
                "overlay_path": str(out_path),
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "overlay_path": out_path,
        "render_manifest_path": manifest_path,
        "render_validation": validation,
    }
