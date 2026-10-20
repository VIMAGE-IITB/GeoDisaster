#!/usr/bin/env python3
"""Shared utilities for the final Sen1Floods11 hidden-label OEA dataset."""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import Affine

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None

try:
    from shapely.geometry import Polygon, mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover
    Polygon = None
    mapping = None
    shape = None
    shapely_transform = None

try:
    from pyproj import CRS, Geod, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Geod = None
    Transformer = None


OUT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = OUT_DIR.parent
RAW_ROOT = PROJECT_ROOT
ORACLE_DIR = PROJECT_ROOT / "sen1_hand_oea_oracle_100"
HIDDEN_EVAL_100_DIR = PROJECT_ROOT / "sen1_hand_hidden_eval_100"
OTSU_OFFICIAL_100_DIR = PROJECT_ROOT / "sen1_hand_otsu_official_100"
DUALPOL_DIR = PROJECT_ROOT / "sen1_dualpol_zbp_experiment_100"
OPTION_C_DIR = PROJECT_ROOT / "sen1_exact_zbp_option_c"
INDEX_PATH = ORACLE_DIR / "metadata" / "sen1_hand_chip_index.json"

AGENT_PROMPT = (
    "You are an OpenEarthAgent-style SAR flood-mapping agent. The hand-label target is hidden and unavailable. "
    "Use the Sentinel-1 SAR image, allowed deterministic tools, optional context layers, and geometry/statistics tools "
    "to produce a predicted flood report."
)

ALLOWED_TOOLS = [
    "LoadRasterImage",
    "SARPreprocess",
    "OtsuFloodMapper",
    "SegmentObjectPixels",
    "Calculator",
    "PolygonizeFloodMask",
    "ConnectedComponentAnalysis",
    "CreateNxNGrid",
    "ComputeGridFloodStats",
    "DisplayOnGeotiff",
    "GenerateFloodReport",
    "Terminate",
]

QUESTION_FAMILIES = [
    (
        "A",
        "Flood Emergency Inventory",
        "Using the Sentinel-1 SAR chip for {chip_id}, generate a SAR-derived flood emergency inventory. Report predicted flood percentage, predicted flood area, number of connected flood regions, largest flood region, largest-region centroid, and 4x4 spatial concentration.",
    ),
    (
        "B",
        "Flood Polygon Geometry",
        "Using SAR-derived flood extraction for chip {chip_id}, produce a flood polygon geometry report. Include predicted flood polygon count, total predicted flood area, largest connected flood polygon, bounding box, centroid, and top 3 flooded cells in a 4x4 grid.",
    ),
    (
        "C",
        "Spatial Concentration",
        "For chip {chip_id}, estimate flood extent from the SAR image and analyze spatial concentration using a 4x4 grid. Report the top 3 grid cells by predicted flood percentage, the concentration score, and whether the predicted flood is localized or dispersed.",
    ),
    (
        "D",
        "Known-Water Context",
        "Using the SAR chip and optional JRC water reference for {chip_id}, generate a flood report that separates predicted flood overlapping known water from predicted flood outside known-water reference. Also report total predicted flood percentage and largest connected flood region.",
    ),
    (
        "E",
        "Complete SAR Flood Diagnostic",
        "Generate a complete SAR flood diagnostic report for chip {chip_id}. Use the available SAR image and allowed flood-mapping tools to produce a predicted flood mask, flood area, polygon geometry, largest component, 4x4 grid concentration, visualization outputs, and a final structured answer.",
    ),
]


def ensure_dirs() -> None:
    for rel in [
        "data",
        "hidden_gt/gt_reports",
        "baselines/otsu/masks",
        "baselines/otsu/reports",
        "baselines/otsu/trajectories",
        "baselines/otsu/replay",
        "baselines/otsu/evaluation",
        "processed_public",
        "processed_hidden",
        "tools",
        "evaluator_tools",
        "schemas",
        "reports",
        "scripts",
    ]:
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(OUT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else OUT_DIR / path


def norm_img(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype="float32")
    lo, hi = np.nanpercentile(arr[finite], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype="float32")
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def event_prefix(chip_id: str) -> str:
    m = re.match(r"(.+?)_\d+", chip_id)
    return m.group(1) if m else chip_id.split("_")[0]


def discover_from_index(sample_size: str = "ALL") -> list[dict[str, Any]]:
    if INDEX_PATH.exists():
        rows = read_json(INDEX_PATH)
    else:
        rows = discover_hand_labeled()
    valid = []
    for r in rows:
        s1 = Path(r.get("s1_path") or "")
        label = Path(r.get("label_path") or "")
        if not s1.exists() or not label.exists():
            continue
        if int(r.get("valid_pixel_count") or 0) <= 0:
            try:
                with rasterio.open(label) as ds:
                    label_arr = ds.read(1)
                if int((label_arr != -1).sum()) <= 0:
                    continue
            except Exception:
                continue
        rr = dict(r)
        rr["event_or_prefix"] = rr.get("event_or_prefix") or event_prefix(rr["chip_id"])
        rr["core_valid"] = True
        valid.append(rr)
    valid = sorted(valid, key=lambda x: (x["event_or_prefix"], x["chip_id"]))
    if sample_size != "ALL":
        n = int(sample_size)
        valid = valid[:n]
    return valid


def discover_hand_labeled() -> list[dict[str, Any]]:
    hand_root = RAW_ROOT / "dataset" / "v1.1" / "data" / "flood_events" / "HandLabeled"
    s1_dir = hand_root / "S1Hand"
    label_dir = hand_root / "LabelHand"
    otsu_dir = hand_root / "S1OtsuLabelHand"
    jrc_dir = hand_root / "JRCWaterHand"
    s2_dir = hand_root / "S2Hand"
    rows = []
    for label_path in sorted(label_dir.glob("*_LabelHand.tif")):
        chip_id = label_path.name.replace("_LabelHand.tif", "")
        s1_path = s1_dir / f"{chip_id}_S1Hand.tif"
        if not s1_path.exists():
            continue
        with rasterio.open(label_path) as ds:
            label = ds.read(1)
            unique = sorted(int(x) for x in np.unique(label).tolist())
            valid_count = int((label != -1).sum())
            flood_count = int((label == 1).sum())
            height, width = label.shape
        rows.append(
            {
                "chip_id": chip_id,
                "s1_path": str(s1_path),
                "label_path": str(label_path),
                "otsu_path": str(otsu_dir / f"{chip_id}_S1OtsuLabelHand.tif") if (otsu_dir / f"{chip_id}_S1OtsuLabelHand.tif").exists() else None,
                "jrc_path": str(jrc_dir / f"{chip_id}_JRCWaterHand.tif") if (jrc_dir / f"{chip_id}_JRCWaterHand.tif").exists() else None,
                "s2_path": str(s2_dir / f"{chip_id}_S2Hand.tif") if (s2_dir / f"{chip_id}_S2Hand.tif").exists() else None,
                "has_s1": True,
                "has_label": True,
                "has_otsu": (otsu_dir / f"{chip_id}_S1OtsuLabelHand.tif").exists(),
                "has_jrc": (jrc_dir / f"{chip_id}_JRCWaterHand.tif").exists(),
                "has_s2": (s2_dir / f"{chip_id}_S2Hand.tif").exists(),
                "width": width,
                "height": height,
                "label_unique_values": unique,
                "valid_pixel_count": valid_count,
                "flood_pixel_count": flood_count,
                "flood_percentage": round(flood_count / valid_count * 100 if valid_count else 0.0, 6),
                "event_or_prefix": event_prefix(chip_id),
                "core_valid": valid_count > 0,
            }
        )
    return rows


def choose_prefix_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(r["event_or_prefix"] for r in rows)
    prefixes = sorted(counts)
    total = sum(counts.values())
    val_target = total * 0.15
    test_target = total * 0.15
    train_target = total * 0.70
    best = None
    for val_size in range(1, len(prefixes)):
        for val in itertools.combinations(prefixes, val_size):
            val = set(val)
            remaining = [p for p in prefixes if p not in val]
            for test_size in range(1, len(remaining)):
                for test in itertools.combinations(remaining, test_size):
                    test = set(test)
                    vc = sum(counts[p] for p in val)
                    tc = sum(counts[p] for p in test)
                    trc = total - vc - tc
                    if trc <= 0:
                        continue
                    penalty = abs(vc - val_target) + abs(tc - test_target) + 0.5 * abs(trc - train_target)
                    if len(val) < 2:
                        penalty += 15
                    if len(test) < 2:
                        penalty += 15
                    candidate = (penalty, sorted(val), sorted(test), vc, tc, trc)
                    if best is None or candidate < best:
                        best = candidate
    assert best is not None
    _penalty, val_prefixes, test_prefixes, _vc, _tc, _trc = best
    split_by_prefix = {}
    for p in prefixes:
        if p in val_prefixes:
            split_by_prefix[p] = "val"
        elif p in test_prefixes:
            split_by_prefix[p] = "test"
        else:
            split_by_prefix[p] = "train"
    return split_by_prefix


def split_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    split_by_prefix = choose_prefix_splits(rows)
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        splits[split_by_prefix[r["event_or_prefix"]]].append(r)
    for split in splits:
        splits[split].sort(key=lambda x: x["chip_id"])
    summary = {
        "strategy": "prefix_exclusive_near_70_15_15",
        "split_by_prefix": split_by_prefix,
        "counts": {k: len(v) for k, v in splits.items()},
        "total": sum(len(v) for v in splits.values()),
        "prefix_counts": dict(Counter(r["event_or_prefix"] for r in rows)),
        "note": "The split keeps each inferred event/location prefix in exactly one split. Exact 70/15/15 is impossible with the available prefix counts, so the nearest deterministic prefix-exclusive split is used.",
    }
    return splits, summary


def write_binary_mask(
    path: Path,
    mask: np.ndarray,
    src_ds,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Write a portable 0/1 classification with 255 reserved for nodata."""
    profile = src_ds.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    output = mask.astype("uint8")
    if valid_mask is not None:
        if valid_mask.shape != mask.shape:
            raise ValueError(
                f"valid-mask shape mismatch: {valid_mask.shape} vs {mask.shape}"
            )
        output = output.copy()
        output[~valid_mask.astype(bool)] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)


def write_float_mask(path: Path, arr: np.ndarray, src_ds) -> None:
    profile = src_ds.profile.copy()
    profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    mask = mask.astype(bool)
    if ndimage is not None:
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())[1:].astype(int).tolist()
        return labels.astype("int32"), int(count), sizes
    labels = np.zeros(mask.shape, dtype="int32")
    current = 0
    sizes = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c]:
                continue
            current += 1
            q = deque([(r, c)])
            labels[r, c] = current
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            q.append((nr, nc))
            sizes.append(size)
    return labels, current, sizes


def grid_cells(mask_shape: tuple[int, int], n: int = 4) -> list[dict[str, int | str]]:
    h, w = mask_shape
    rows = np.linspace(0, h, n + 1, dtype=int)
    cols = np.linspace(0, w, n + 1, dtype=int)
    cells = []
    for r in range(n):
        for c in range(n):
            cells.append(
                {
                    "cell_id": f"R{r + 1}_C{c + 1}",
                    "row": r + 1,
                    "col": c + 1,
                    "row_start": int(rows[r]),
                    "row_stop": int(rows[r + 1]),
                    "col_start": int(cols[c]),
                    "col_stop": int(cols[c + 1]),
                }
            )
    return cells


def grid_stats(mask: np.ndarray, valid_mask: np.ndarray | None = None, n: int = 4) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    if valid_mask is None:
        valid_mask = np.ones(mask.shape, dtype=bool)
    total_flood = int((mask & valid_mask).sum())
    rows = []
    for cell in grid_cells(mask.shape, n):
        win_flood = mask[cell["row_start"] : cell["row_stop"], cell["col_start"] : cell["col_stop"]]
        win_valid = valid_mask[cell["row_start"] : cell["row_stop"], cell["col_start"] : cell["col_stop"]]
        valid_pixels = int(win_valid.sum())
        flood_pixels = int((win_flood & win_valid).sum())
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "row": cell["row"],
                "col": cell["col"],
                "valid_pixels": valid_pixels,
                "flood_pixels": flood_pixels,
                "flood_percentage": round(flood_pixels / valid_pixels * 100 if valid_pixels else 0.0, 6),
                "fraction_of_total_flood": round(flood_pixels / total_flood if total_flood else 0.0, 6),
            }
        )
    top = sorted(rows, key=lambda x: (x["flood_percentage"], x["flood_pixels"], x["cell_id"]), reverse=True)[:3]
    concentration = max([c["flood_pixels"] for c in rows] or [0]) / total_flood if total_flood else 0.0
    return rows, top, round(float(concentration), 6)


def pixel_area_m2(ds) -> tuple[float | None, str]:
    """Return the legacy representative-pixel area for compatibility only."""
    if ds.crs is None:
        return None, "area_unavailable_missing_crs"
    if getattr(ds.crs, "is_projected", False):
        return abs(float(ds.transform.a * ds.transform.e)), "area_from_projected_transform"
    if not getattr(ds.crs, "is_geographic", False) or CRS is None or Transformer is None or Polygon is None:
        return None, "area_unavailable_unsupported_crs_or_missing_pyproj"
    h, w = ds.height, ds.width
    r, c = h // 2, w // 2
    t: Affine = ds.transform
    corners = [t * (c, r), t * (c + 1, r), t * (c + 1, r + 1), t * (c, r + 1)]
    lon = float(sum(x for x, _y in corners) / 4)
    lat = float(sum(y for _x, y in corners) / 4)
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    try:
        transformer = Transformer.from_crs(ds.crs, CRS.from_epsg(epsg), always_xy=True)
        projected = [transformer.transform(x, y) for x, y in corners]
        return float(abs(Polygon(projected).area)), f"area_from_center_pixel_reprojected_to_EPSG:{epsg}"
    except Exception as exc:
        return None, f"area_unavailable_reprojection_failed:{exc}"


def mask_area_m2(mask: np.ndarray, src_ds) -> tuple[float | None, str]:
    """Measure a mask on its packaged grid under an explicit area contract."""
    mask = mask.astype(bool)
    if src_ds.crs is None:
        return None, "area_unavailable_missing_crs"
    transform: Affine = src_ds.transform
    if getattr(src_ds.crs, "is_projected", False):
        pixel_area = abs(float(transform.a * transform.e - transform.b * transform.d))
        return float(mask.sum()) * pixel_area, "projected_affine_determinant_cell_sum_v1"
    if not getattr(src_ds.crs, "is_geographic", False) or Geod is None:
        return None, "area_unavailable_unsupported_crs_or_missing_pyproj"

    geod = Geod(ellps="WGS84")
    total = 0.0
    north_up = abs(float(transform.b)) < 1e-15 and abs(float(transform.d)) < 1e-15
    if north_up:
        # Cell area is constant within a row for a north-up geographic grid.
        for row, count in enumerate(mask.sum(axis=1)):
            count = int(count)
            if not count:
                continue
            x0, y0 = transform * (0, row)
            x1, y1 = transform * (1, row + 1)
            area, _ = geod.polygon_area_perimeter(
                [x0, x1, x1, x0], [y0, y0, y1, y1]
            )
            total += count * abs(float(area))
    else:  # pragma: no cover - current Sen1 inputs are north-up
        for row, col in np.argwhere(mask):
            p00 = transform * (int(col), int(row))
            p10 = transform * (int(col) + 1, int(row))
            p11 = transform * (int(col) + 1, int(row) + 1)
            p01 = transform * (int(col), int(row) + 1)
            area, _ = geod.polygon_area_perimeter(
                [p00[0], p10[0], p11[0], p01[0]],
                [p00[1], p10[1], p11[1], p01[1]],
            )
            total += abs(float(area))
    return total, "wgs84_ellipsoidal_cell_sum_v1"


def polygon_feature_collection(
    mask: np.ndarray,
    src_ds,
    value: int = 1,
    connectivity: int = 8,
) -> dict[str, Any]:
    if connectivity not in {4, 8}:
        raise ValueError(f"unsupported polygon connectivity: {connectivity}")
    feats = []
    for geom, val in features.shapes(
        mask.astype("uint8"),
        mask=mask.astype(bool),
        transform=src_ds.transform,
        connectivity=connectivity,
    ):
        if int(val) != value:
            continue
        feats.append({"type": "Feature", "geometry": geom, "properties": {"value": int(value)}})
    crs = str(src_ds.crs) if src_ds.crs else None
    out = {"type": "FeatureCollection", "features": feats}
    if crs:
        out["crs"] = {"type": "name", "properties": {"name": crs}}
    return out


def grid_geojson(mask_shape: tuple[int, int], src_ds, stats: list[dict[str, Any]], n: int = 4) -> dict[str, Any]:
    stats_by_cell = {s["cell_id"]: s for s in stats}
    feats = []
    for cell in grid_cells(mask_shape, n):
        r0, r1, c0, c1 = cell["row_start"], cell["row_stop"], cell["col_start"], cell["col_stop"]
        t = src_ds.transform
        coords = [t * (c0, r0), t * (c1, r0), t * (c1, r1), t * (c0, r1), t * (c0, r0)]
        props = {k: v for k, v in cell.items() if k in {"cell_id", "row", "col"}}
        props.update(stats_by_cell.get(cell["cell_id"], {}))
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]}, "properties": props})
    out = {"type": "FeatureCollection", "features": feats}
    if src_ds.crs:
        out["crs"] = {"type": "name", "properties": {"name": str(src_ds.crs)}}
    return out


def largest_component_geojson(labels: np.ndarray, sizes: list[int], src_ds) -> tuple[dict[str, Any], int | None]:
    if not sizes:
        return {"type": "FeatureCollection", "features": []}, None
    largest_label = int(np.argmax(sizes) + 1)
    mask = labels == largest_label
    fc = polygon_feature_collection(mask, src_ds, value=1)
    for feat in fc["features"]:
        feat["properties"]["component_label"] = largest_label
    return fc, largest_label


def centroid_from_mask(mask: np.ndarray, transform: Affine) -> list[float] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return [round(float(np.mean(xs)), 8), round(float(np.mean(ys)), 8)]


def compute_mask_report(
    chip_id: str,
    mask: np.ndarray,
    valid_mask: np.ndarray,
    src_ds,
    prefix: str,
    mask_path: str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    labels, component_count, sizes = connected_components(mask & valid_mask)
    flood_pixels = int((mask & valid_mask).sum())
    valid_pixels = int(valid_mask.sum())
    largest_pixels = int(max(sizes)) if sizes else None
    largest_mask = labels == (int(np.argmax(sizes)) + 1) if sizes else np.zeros(mask.shape, dtype=bool)
    flood_area, area_status = mask_area_m2(mask & valid_mask, src_ds)
    largest_area, _ = mask_area_m2(largest_mask, src_ds)
    all_grid, top3, concentration = grid_stats(mask, valid_mask)
    report = {
        "chip_id": chip_id,
        "method": method or prefix,
        f"{prefix}_flood_pixel_count": flood_pixels,
        f"{prefix}_valid_pixel_count": valid_pixels,
        f"{prefix}_flood_percentage": round(flood_pixels / valid_pixels * 100 if valid_pixels else 0.0, 6),
        f"{prefix}_flood_area_m2": round(flood_area, 3) if flood_area is not None else None,
        f"{prefix}_flood_area_km2": round(flood_area / 1_000_000, 6) if flood_area is not None else None,
        "area_status": area_status,
        "component_count": int(component_count),
        "largest_component_pixel_count": largest_pixels,
        "largest_component_area_m2": round(largest_area, 3) if largest_pixels is not None and largest_area is not None else None,
        "largest_component_centroid": centroid_from_mask(largest_mask, src_ds.transform) if largest_pixels else None,
        "polygon_count": int(component_count),
        "grid_4x4": all_grid,
        "top_3_grid_cells": top3,
        "concentration_score": concentration,
    }
    if mask_path:
        report["mask_path"] = mask_path
    return report


def compute_gt_outputs(row: dict[str, Any], split: str) -> dict[str, Any]:
    chip_id = row["chip_id"]
    out_dir = OUT_DIR / "processed_hidden" / chip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(row["label_path"]) as ds:
        label = ds.read(1)
        valid_mask = label != -1
        flood_mask = label == 1
        write_binary_mask(out_dir / "gt_flood_mask_binary.tif", flood_mask, ds)
        write_binary_mask(out_dir / "gt_valid_mask_binary.tif", valid_mask, ds)
        labels, _count, sizes = connected_components(flood_mask & valid_mask)
        fc = polygon_feature_collection(flood_mask & valid_mask, ds)
        write_json(out_dir / "gt_flood_polygon.geojson", fc)
        grid_all, _top3, _conc = grid_stats(flood_mask, valid_mask)
        write_json(out_dir / "gt_grid_4x4.geojson", grid_geojson(label.shape, ds, grid_all))
        largest_fc, _largest_label = largest_component_geojson(labels, sizes, ds)
        write_json(out_dir / "gt_largest_component.geojson", largest_fc)
        report = compute_mask_report(chip_id, flood_mask, valid_mask, ds, "gt", rel(out_dir / "gt_flood_mask_binary.tif"), "LabelHand hidden ground truth")
        report.update(
            {
                "split": split,
                "label_values": {"-1": "ignore", "0": "background/non-flood", "1": "flood"},
                "source_s1_path": row["s1_path"],
                "optional_layers": {
                    "s2_path": row.get("s2_path"),
                    "jrc_path": row.get("jrc_path"),
                    "otsu_path": row.get("otsu_path"),
                },
                "output_files": {
                    "gt_flood_mask_binary": rel(out_dir / "gt_flood_mask_binary.tif"),
                    "gt_valid_mask_binary": rel(out_dir / "gt_valid_mask_binary.tif"),
                    "gt_flood_polygon": rel(out_dir / "gt_flood_polygon.geojson"),
                    "gt_grid_4x4": rel(out_dir / "gt_grid_4x4.geojson"),
                    "gt_largest_component": rel(out_dir / "gt_largest_component.geojson"),
                },
            }
        )
    report_path = OUT_DIR / "hidden_gt" / "gt_reports" / f"{chip_id}_gt_report.json"
    write_json(report_path, report)
    return report


def question_for(idx: int, chip_id: str) -> tuple[str, str, str]:
    family_id, family_name, template = QUESTION_FAMILIES[idx % len(QUESTION_FAMILIES)]
    return family_id, family_name, template.format(chip_id=chip_id)


def public_record(idx: int, row: dict[str, Any], split: str) -> dict[str, Any]:
    chip_id = row["chip_id"]
    family_id, _family_name, question = question_for(idx, chip_id)
    images = [f"processed_public/{chip_id}/s1_preview.png", f"processed_public/{chip_id}/s1_overlay.png"]
    if row.get("s2_path"):
        images.append(f"processed_public/{chip_id}/s2_rgb_optional.png")
    if row.get("jrc_path"):
        images.append(f"processed_public/{chip_id}/jrc_preview_optional.png")
    return {
        "idx": idx,
        "type": "sen1_hidden_flood_eval",
        "images": images,
        "question": question,
        "conversation": [
            {"from": "human", "value": f"{AGENT_PROMPT}\n\nQuestion: {question}"},
            {
                "from": "gpt",
                "value": json.dumps(
                    {
                        "thought": "I need to analyze the SAR image, run a flood extraction tool, compute spatial statistics, and return a structured flood report.",
                        "actions": [],
                    }
                ),
            },
        ],
        "metadata": {
            "dataset": "Sen1Floods11-HandLabeled",
            "chip_id": chip_id,
            "query_family": family_id,
            "split": split,
            "mode": "hidden_label_evaluation",
            "available_layers": {
                "s1_image": row["s1_path"],
                "s2_image_optional": row.get("s2_path"),
                "jrc_mask_optional": row.get("jrc_path"),
            },
            "hidden_layers": {"labelhand_hidden": True},
            "allowed_tools": ALLOWED_TOOLS,
            "expected_prediction_format": "schemas/prediction_schema.json",
        },
    }


def public_text_has_label_leak(text: str) -> tuple[bool, list[str]]:
    disallowed = ["/LabelHand/", "_LabelHand.tif", "label_path", "hidden_gt", "processed_hidden"]
    leaks = [needle for needle in disallowed if needle in text]
    text_without_allowed = text.replace('"labelhand_hidden": true', "").replace('"labelhand_hidden": True', "")
    if "LabelHand" in text_without_allowed:
        leaks.append("LabelHand")
    return bool(leaks), leaks


def raster_band_stats(path: str | Path) -> dict[str, Any]:
    with rasterio.open(path) as ds:
        stats = []
        for i in range(1, ds.count + 1):
            arr = ds.read(i).astype("float32")
            finite = arr[np.isfinite(arr)]
            if finite.size:
                row = {
                    "band": i,
                    "dtype": str(arr.dtype),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "p2": float(np.percentile(finite, 2)),
                    "p98": float(np.percentile(finite, 98)),
                }
            else:
                row = {"band": i, "dtype": str(arr.dtype), "finite_pixels": 0}
            stats.append(row)
        return {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ",".join(ds.dtypes),
            "crs": str(ds.crs) if ds.crs else None,
            "transform": list(ds.transform)[:6],
            "bands": stats,
        }


def otsu_threshold(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(np.nanmean(values))
    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1))[::-1]
    variance12 = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[1:]) ** 2
    if variance12.size == 0:
        return float(np.nanmean(values))
    idx = int(np.argmax(variance12))
    return float(centers[:-1][idx])


def compute_otsu_mask_from_s1(s1_path: str | Path) -> tuple[np.ndarray, dict[str, Any], Any]:
    ds = rasterio.open(s1_path)
    b1 = ds.read(1).astype("float32")
    b2 = ds.read(2).astype("float32") if ds.count >= 2 else b1
    score = np.minimum(b1, b2)
    valid = np.isfinite(b1) & np.isfinite(b2)
    nodata = ds.nodata
    if nodata is not None and np.isfinite(float(nodata)):
        valid &= (b1 != float(nodata)) & (b2 != float(nodata))
    threshold = otsu_threshold(score[valid])
    mask = valid & (score <= threshold)
    return mask, {"threshold": threshold, "score": "min_band_backscatter", "source": "computed_from_s1"}, ds


def get_otsu_mask(row: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], Any]:
    otsu_path = row.get("otsu_path")
    if otsu_path and Path(otsu_path).exists():
        ds = rasterio.open(otsu_path)
        arr = ds.read(1)
        return arr == 1, {"threshold": None, "source": "S1OtsuLabelHand official baseline mask"}, ds
    return compute_otsu_mask_from_s1(row["s1_path"])


def compute_jrc_overlap(mask: np.ndarray, row: dict[str, Any]) -> dict[str, Any]:
    jrc_path = row.get("jrc_path")
    if not jrc_path or not Path(jrc_path).exists():
        return {"available": False}
    with rasterio.open(jrc_path) as ds:
        jrc = ds.read(1) == 1
    total = int(mask.sum())
    overlap = int((mask & jrc).sum())
    outside = int((mask & ~jrc).sum())
    return {
        "available": True,
        "predicted_flood_overlapping_jrc_pixels": overlap,
        "predicted_flood_outside_jrc_pixels": outside,
        "predicted_flood_overlapping_jrc_percentage": round(overlap / total * 100 if total else 0.0, 6),
        "predicted_flood_outside_jrc_percentage": round(outside / total * 100 if total else 0.0, 6),
        "note": "JRC is a known-water reference/context layer, not ground truth.",
    }


def final_answer_from_report(report: dict[str, Any]) -> str:
    pct = report["predicted_flood_percentage"]
    area = report.get("predicted_flood_area_m2")
    area_text = "unavailable" if area is None else f"{area:.3f} m2"
    top = ", ".join(f"{c['cell_id']} ({c['flood_percentage']:.3f}%, {c['flood_pixels']} px)" for c in report["top_3_grid_cells"])
    loc = "localized" if report["concentration_score"] >= 0.5 else "dispersed"
    return (
        f"OtsuFloodMapper prediction for chip {report['chip_id']}: predicted_flood_pixel_count={report['predicted_flood_pixel_count']}, "
        f"predicted_flood_percentage={pct:.6f}%, predicted_flood_area={area_text}, component_count={report['component_count']}, "
        f"largest_component_pixel_count={report['largest_component_pixel_count']}, largest_component_centroid={report['largest_component_centroid']}, "
        f"concentration_score={report['concentration_score']:.6f} ({loc}), top_3_grid_cells={top}. "
        "This is a SAR/Otsu baseline prediction; the hidden hand-label target is used only by the evaluator."
    )


def top_cell_ids(cells: Any) -> set[str]:
    out = set()
    for cell in cells or []:
        if isinstance(cell, str):
            out.add(cell)
        elif isinstance(cell, dict) and "cell_id" in cell:
            out.add(str(cell["cell_id"]))
    return out


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def f6(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def evaluate_prediction(pred: dict[str, Any], hidden: dict[str, Any]) -> dict[str, Any]:
    chip_id = pred["chip_id"]
    pred_mask_path = resolve_path(pred["predicted_flood_mask_path"])
    label_path = Path(hidden["label_path"])
    errors = []
    if not pred_mask_path.exists():
        errors.append(f"missing predicted mask {pred_mask_path}")
    if not label_path.exists():
        errors.append(f"missing hidden label {label_path}")
    if errors:
        return {"chip_id": chip_id, "passed": False, "errors": errors}
    with rasterio.open(pred_mask_path) as ds:
        pred_mask = ds.read(1) == 1
    with rasterio.open(label_path) as ds:
        label = ds.read(1)
        valid = label != -1
        gt = label == 1
    if pred_mask.shape != label.shape:
        return {"chip_id": chip_id, "passed": False, "errors": [f"shape mismatch {pred_mask.shape} vs {label.shape}"]}
    pred_valid = pred_mask & valid
    tp = int((gt & pred_valid).sum())
    fp = int(((label == 0) & pred_valid).sum())
    fn = int((gt & (~pred_mask) & valid).sum())
    tn = int(((label == 0) & (~pred_mask) & valid).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    iou = safe_div(tp, tp + fp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    gt_stats = hidden["gt_stats"]
    pred_report = pred.get("predicted_report", {})
    pred_count = int(pred_valid.sum())
    valid_count = int(valid.sum())
    pred_pct = pred_count / valid_count * 100 if valid_count else 0.0
    gt_top = top_cell_ids(gt_stats.get("top_3_grid_cells"))
    pred_top = top_cell_ids(pred_report.get("top_3_grid_cells"))
    top_overlap = len(gt_top & pred_top) / len(gt_top) if gt_top else None
    return {
        "chip_id": chip_id,
        "passed": True,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "IoU": f6(iou),
        "precision": f6(precision),
        "recall": f6(recall),
        "F1": f6(f1),
        "predicted_flood_pixel_count": pred_count,
        "gt_flood_pixel_count": gt_stats["flood_pixel_count"],
        "flood_pixel_count_error": int(pred_count - gt_stats["flood_pixel_count"]),
        "predicted_flood_percentage": f6(pred_pct),
        "gt_flood_percentage": gt_stats["flood_percentage"],
        "flood_percentage_absolute_error": f6(abs(pred_pct - gt_stats["flood_percentage"])),
        "area_absolute_error_m2": None
        if gt_stats.get("flood_area_m2") is None or pred_report.get("predicted_flood_area_m2") is None
        else round(abs(float(pred_report["predicted_flood_area_m2"]) - float(gt_stats["flood_area_m2"])), 3),
        "top_3_grid_cell_overlap": f6(top_overlap),
        "component_count_error": int((pred_report.get("component_count") or 0) - gt_stats["component_count"]),
        "errors": [],
    }


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("passed")]
    def mean_metric(name: str) -> float | None:
        vals = [r[name] for r in ok if r.get(name) is not None]
        return f6(float(np.mean(vals))) if vals else None
    return {
        "passed": len(ok) == len(records),
        "num_records": len(records),
        "mean_IoU": mean_metric("IoU"),
        "mean_precision": mean_metric("precision"),
        "mean_recall": mean_metric("recall"),
        "mean_F1": mean_metric("F1"),
        "mean_flood_percentage_absolute_error": mean_metric("flood_percentage_absolute_error"),
        "mean_top_3_grid_cell_overlap": mean_metric("top_3_grid_cell_overlap"),
        "aggregate_TP": int(sum(r.get("TP", 0) for r in ok)),
        "aggregate_FP": int(sum(r.get("FP", 0) for r in ok)),
        "aggregate_FN": int(sum(r.get("FN", 0) for r in ok)),
        "aggregate_TN": int(sum(r.get("TN", 0) for r in ok)),
    }


def metrics_md(title: str, result: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [
        f"# {title}",
        "",
        f"- Passed: {s['passed']}",
        f"- Records: {s['num_records']}",
        f"- Mean IoU: {s['mean_IoU']}",
        f"- Mean F1: {s['mean_F1']}",
        f"- Mean precision: {s['mean_precision']}",
        f"- Mean recall: {s['mean_recall']}",
        f"- Mean flood percentage absolute error: {s['mean_flood_percentage_absolute_error']}",
        f"- Mean top-3 grid-cell overlap: {s['mean_top_3_grid_cell_overlap']}",
        f"- Aggregate TP/FP/FN/TN: {s['aggregate_TP']} / {s['aggregate_FP']} / {s['aggregate_FN']} / {s['aggregate_TN']}",
    ]
    return "\n".join(lines) + "\n"


def write_common_schemas() -> None:
    write_json(
        OUT_DIR / "schemas" / "prediction_schema.json",
        {
            "type": "object",
            "required": ["chip_id", "predicted_flood_mask_path", "predicted_report"],
            "properties": {
                "chip_id": {"type": "string"},
                "predicted_flood_mask_path": {"type": "string"},
                "predicted_report": {
                    "type": "object",
                    "required": ["predicted_flood_percentage", "predicted_flood_area_m2", "component_count", "top_3_grid_cells"],
                },
            },
        },
    )
    for name in ["task_schema", "hidden_gt_schema", "evaluation_schema", "oea_record_schema", "trajectory_schema"]:
        write_json(OUT_DIR / "schemas" / f"{name}.json", {"schema_name": name, "dataset": "Sen1Floods11-HandLabeled hidden evaluation"})
