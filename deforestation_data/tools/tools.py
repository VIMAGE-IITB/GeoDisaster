"""
The 8 deterministic local-cache tools.

All tools read only from data/samples/<sample_id>/{visible,cache,masks,features}.
None reads from gt/ — the runtime should additionally enforce this via path checks.
"""

from __future__ import annotations
import hashlib, math
from contextlib import contextmanager
from contextvars import ContextVar
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

EPOCH_ORDER = ["pre_3", "pre_2", "pre_1", "post_1", "post_2", "post_3"]
PRE_EPOCHS  = ["pre_3", "pre_2", "pre_1"]
POST_EPOCHS = ["post_1", "post_2", "post_3"]
CHANGE_WINDOW_CONTRACT_VERSION = "hansen_confirmed_dominant_optical_interval_v1"
CONFIRMED_CLEARING_MIN_LOSS_HA = 0.5
VH_SEVERITY_CONTRACT_VERSION = "vh_delta_question_thresholds_v1"
VH_SEVERITY_THRESHOLDS_DB = {
    "severe": "delta < -3",
    "moderate": "-3 <= delta < -1",
    "mild": "-1 <= delta < 0",
    "none": "delta >= 0",
}


# The online worker installs this scope from the file cards reachable through
# the current tool arguments.  Keeping it here, next to every raster/JSON read,
# makes directory-derived filenames subject to the same exact-path contract as
# paths selected directly by a worker.
_DECLARED_READ_SCOPE: ContextVar[dict | None] = ContextVar(
    "deforestation_declared_read_scope", default=None
)


@contextmanager
def DeclaredReadScope(paths):
    allowed = {str(Path(path).resolve()) for path in paths}
    scope = {"allowed": allowed, "reads": []}
    token = _DECLARED_READ_SCOPE.set(scope)
    try:
        yield scope["reads"]
    finally:
        _DECLARED_READ_SCOPE.reset(token)


def _authorize_read(path: Path, operation: str = "read") -> Path:
    resolved = Path(path).resolve()
    lowered_parts = {part.lower() for part in resolved.parts}
    lowered_name = resolved.name.lower()
    private_markers = (
        "expected_answer",
        "expected_final_answer",
        "canonical_answer",
        "ground_truth",
        "gt_report",
        "final_answer",
    )
    if lowered_parts & {"gt", "ground_truth", "hidden", "trajectories"} or any(
        marker in lowered_name for marker in private_markers
    ):
        raise PermissionError(f"private answer path is not readable by tools: {resolved}")
    scope = _DECLARED_READ_SCOPE.get()
    if scope is not None and str(resolved) not in scope["allowed"]:
        raise PermissionError(
            f"undeclared sibling access denied ({operation}): {resolved}"
        )
    if scope is not None:
        record = {"path": str(resolved), "operation": operation}
        if record not in scope["reads"]:
            scope["reads"].append(record)
    return resolved


def _exists_for_read(path: Path) -> bool:
    return _authorize_read(path, "exists").exists()


def _declared_glob(directory: Path, pattern: str) -> list[Path]:
    scope = _DECLARED_READ_SCOPE.get()
    if scope is None:
        return sorted(directory.glob(pattern))
    parent = directory.resolve()
    matches = []
    for path_text in scope["allowed"]:
        path = Path(path_text)
        if path.parent == parent and fnmatch(path.name, pattern):
            _authorize_read(path, "enumerate")
            matches.append(path)
    return sorted(matches)


def _root(sample_dir: Path) -> Path:
    return Path(sample_dir)


def _safe_read(path: Path, band: int = 1) -> Optional[np.ndarray]:
    path = _authorize_read(path, "raster_read")
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        a = ds.read(band).astype("float32")
        a[a == 0] = np.nan          # treat 0 as no-data
        return a


def _read_all_bands(path: Path) -> Optional[np.ndarray]:
    path = _authorize_read(path, "raster_read")
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        a = ds.read().astype("float32")
        a[a == 0] = np.nan
        return a


def _nan_mean(a: np.ndarray) -> float:
    if a is None:
        return float("nan")
    m = np.nanmean(a)
    return float(m) if np.isfinite(m) else float("nan")


def _block_assert_no_gt(p: Path):
    _authorize_read(p, "path_check")


def _event_year_from_assets(base: Path) -> int | None:
    # Event year is encoded by the declared annual-mask filename.  Do not scan
    # or read sibling question/answer files to discover it.
    for path in _declared_glob(base / "masks", "loss_year_*.tif"):
        stem = path.stem.rsplit("_", 1)[-1]
        if stem.isdigit():
            return int(stem)
    return None


def _sha256(path: Path) -> str:
    path = _authorize_read(path, "hash_read")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geographic_mask_area_m2(ds, mask: np.ndarray) -> float:
    """Sum WGS84 ellipsoidal cell areas for a north-up geographic raster."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    total = 0.0
    for row in range(mask.shape[0]):
        count = int(mask[row].sum())
        if not count:
            continue
        x0, y0 = ds.transform * (0, row)
        x1, y1 = ds.transform * (1, row + 1)
        area, _ = geod.polygon_area_perimeter(
            [x0, x1, x1, x0], [y0, y0, y1, y1]
        )
        total += abs(float(area)) * count
    return total


def _mask_area_m2(ds, mask: np.ndarray) -> float:
    if ds.crs and ds.crs.is_geographic:
        return _geographic_mask_area_m2(ds, mask)
    pixel_area = abs(
        float(ds.transform.a) * float(ds.transform.e)
        - float(ds.transform.b) * float(ds.transform.d)
    )
    return float(mask.sum()) * pixel_area


def BuildQualityReport(sample_dir: str, event_year: int | None = None) -> dict:
    """Build an answer-safe Hansen report from the declared local rasters.

    This report is generated during tool execution.  It deliberately does not
    read the fetch-time ``quality_report.json`` because that file contains
    manifest labels and an Earth Engine reduction that is not reproducible from
    the packaged raster grid.
    """
    base = _root(Path(sample_dir))
    resolved_year = int(event_year) if event_year is not None else _event_year_from_assets(base)
    if resolved_year is None:
        raise ValueError(f"could not resolve event year for {base}")

    masks_dir = base / "masks"
    requested_mask = masks_dir / f"loss_year_{resolved_year}.tif"
    mask_path = requested_mask
    mask_kind = "event_year_binary_mask"
    fallback_used = False
    fallback_reason = None

    annual_valid = False
    if _exists_for_read(requested_mask):
        with rasterio.open(_authorize_read(requested_mask, "raster_read")) as ds:
            annual = ds.read(1)
            annual_valid_mask = np.isfinite(annual)
            if ds.nodata is not None and np.isfinite(ds.nodata):
                annual_valid_mask &= annual != ds.nodata
            annual_valid = bool(annual_valid_mask.any())

    if not _exists_for_read(requested_mask) or not annual_valid:
        mask_path = masks_dir / "lossyear.tif"
        if not _exists_for_read(mask_path):
            raise FileNotFoundError(
                f"neither usable {requested_mask.name} nor lossyear.tif exists in {masks_dir}"
            )
        mask_kind = "hansen_lossyear_code_mask"
        fallback_used = True
        fallback_reason = (
            "event_year_mask_missing"
            if not requested_mask.exists()
            else "event_year_mask_has_no_valid_pixels"
        )

    with rasterio.open(_authorize_read(mask_path, "raster_read")) as ds:
        arr = ds.read(1)
        valid = np.isfinite(arr)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            valid &= arr != ds.nodata
        if mask_kind == "hansen_lossyear_code_mask":
            loss = valid & (arr == (resolved_year - 2000))
        else:
            loss = valid & (arr > 0)
        area_m2 = _mask_area_m2(ds, loss)
        crs = str(ds.crs) if ds.crs else None
        transform = [
            float(ds.transform.a),
            float(ds.transform.b),
            float(ds.transform.c),
            float(ds.transform.d),
            float(ds.transform.e),
            float(ds.transform.f),
        ]
        shape = [int(ds.height), int(ds.width)]
        nodata = ds.nodata
        if nodata is not None and not np.isfinite(nodata):
            nodata = str(nodata)

    try:
        from scipy import ndimage

        labels, component_count = ndimage.label(
            loss, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.ravel())[1:]
        largest_patch_pixels = int(sizes.max()) if sizes.size else 0
    except Exception:  # pragma: no cover - scipy is optional
        component_count = int(loss.any())
        largest_patch_pixels = int(loss.sum())

    try:
        relative_mask_path = str(mask_path.relative_to(base))
    except ValueError:
        relative_mask_path = str(mask_path)

    return {
        "schema_version": "geodisaster_generated_quality_report_v1",
        "sample_id": base.name,
        "event_year": resolved_year,
        "source_dataset": "UMD/Hansen/global_forest_change_2025_v1_13",
        "generated_from_declared_inputs": True,
        "contains_expected_answer": False,
        "loss_event_year": {
            "mask_path": relative_mask_path,
            "mask_sha256": _sha256(mask_path),
            "mask_kind": mask_kind,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "crs": crs,
            "transform": transform,
            "shape": shape,
            "nodata": nodata,
            "valid_pixel_count": int(valid.sum()),
            "loss_pixel_count": int(loss.sum()),
            "component_count": int(component_count),
            "largest_patch_pixels": largest_patch_pixels,
            "area_m2": round(float(area_m2), 6),
            "area_ha": round(float(area_m2 / 10000.0), 6),
            "measurement_method": (
                "sum_wgs84_ellipsoidal_pixel_area"
                if crs == "EPSG:4326"
                else "sum_projected_pixel_area"
            ),
        },
    }


def _loss_area_ha(base: Path, event_year: int | None) -> float:
    if event_year is None:
        return 0.0
    return float(BuildQualityReport(str(base), event_year)["loss_event_year"]["area_ha"])


def _quality_report(base: Path) -> dict:
    # Always derive this execution-time context from declared source inputs.
    # The fetch-time quality_report.json is intentionally never consulted.
    event_year = _event_year_from_assets(base)
    s1_files = _declared_glob(base / "cache/s1_grd", "*.tif")
    split_idx = max(1, len(s1_files) // 2) if s1_files else 0
    s1_windows = {}
    for idx, tif in enumerate(s1_files):
        s1_windows[tif.stem] = {
            "name": tif.stem,
            "temporal_role": "pre" if idx < split_idx else "post",
            "target_date": None,
            "metadata_source": "sorted_filename_fallback",
        }

    epoch_names = {
        path.stem for path in _declared_glob(base / "cache/s2", "*.tif")
    } | {
        path.stem for path in _declared_glob(base / "visible/s2_rgb", "*.png")
    }
    epochs = {
        epoch: {"name": epoch, "start": None, "status": "ok"}
        for epoch in EPOCH_ORDER
        if epoch in epoch_names
    }

    return {
        "sample_id": base.name,
        "region_group": "unknown",
        "event_year": event_year,
        "epochs": epochs,
        "s1_windows": s1_windows,
        "hansen": {
            "loss_event_year": {
                "loss_area_ha": _loss_area_ha(base, event_year),
            }
        },
    }


# --------------------------------------------------------------------------------------
# 1. LoadAOI
# --------------------------------------------------------------------------------------

def LoadAOI(sample_dir: str) -> dict:
    base = _root(sample_dir)
    qr = _quality_report(base)
    # visible: just RGB PNGs the agent can "see"
    vis = _declared_glob(base / "visible/s2_rgb", "*.png")
    epochs = [p.stem for p in vis]
    dates = [qr["epochs"].get(e, {}).get("start") for e in epochs]
    return {
        "sample_id": qr["sample_id"],
        "aoi_type_hint": "unknown",       # never reveal class during solving
        "region_group": qr["region_group"],
        "event_year": qr["event_year"],
        "visible": {
            "modality": "s2_rgb",
            "panel_paths": [str(p.relative_to(base)) for p in vis],
            "epoch_dates": dates,
        },
        "availability": {
            "s2_full_bands": bool(_declared_glob(base / "cache/s2", "*.tif")),
            "s1_grd":        bool(_declared_glob(base / "cache/s1_grd", "*.tif")),
            "s1_slc_burst":  bool(_declared_glob(base / "cache/s1_slc_bursts", "*.tif")),
            "hansen_mask":   _exists_for_read(base / "masks/treecover2000.tif"),
        },
    }


# --------------------------------------------------------------------------------------
# 2. GetOpticalEvidence
# --------------------------------------------------------------------------------------

def _ndvi(b8, b4):
    den = b8 + b4
    return np.where(den == 0, np.nan, (b8 - b4) / den)


def _nbr(b8, b12):
    den = b8 + b12
    return np.where(den == 0, np.nan, (b8 - b12) / den)


# bands order in the cached GeoTIFF (per fetch_plan.s2_windows.bands_to_fetch)
# = ["B2","B3","B4","B8","B11","B12"]
S2_BAND_IDX = {"B2": 0, "B3": 1, "B4": 2, "B8": 3, "B11": 4, "B12": 5}


def _resize_mask_to(mask_path: Path, target_shape):
    """Nearest-neighbour resize a binary mask raster to (H,W)."""
    mask_path = _authorize_read(mask_path, "raster_read")
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as ds:
        m = ds.read(1)
        valid = np.isfinite(m)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            valid &= m != ds.nodata
        positive = valid & (m > 0)
    if m.shape == target_shape:
        return positive
    from PIL import Image as _PI
    pil = _PI.fromarray(positive.astype("uint8") * 255).resize(
        (target_shape[1], target_shape[0]), _PI.NEAREST)
    return np.array(pil) > 0


def GetOpticalEvidence(sample_dir: str, indices=("NDVI", "NBR"), epochs="all",
                        restrict_to_loss_mask: bool = False) -> dict:
    base = _root(sample_dir)
    if epochs == "all":
        epochs = EPOCH_ORDER
    out = {}
    cloud_flags = {}
    per_epoch_arrays = {}
    declared_s2 = {
        path.stem: path
        for path in _declared_glob(base / "cache/s2", "*.tif")
    }

    for ep in epochs:
        tif = declared_s2.get(ep)
        if tif is None:
            cloud_flags[ep] = "missing"
            continue
        _block_assert_no_gt(tif)
        arr = _read_all_bands(tif)
        if arr is None:
            cloud_flags[ep] = "missing"
            continue
        per_epoch_arrays[ep] = arr
        # crude cloud heuristic from B2 (high reflectance = cloud)
        b2 = arr[S2_BAND_IDX["B2"]]
        valid = np.isfinite(b2)
        if valid.mean() < 0.30:
            cloud_flags[ep] = "all_cloudy_or_missing"
        elif np.nanmean(b2) > 2000:
            cloud_flags[ep] = "partial_cloud"
        else:
            cloud_flags[ep] = "clear"

    # Optional: restrict mean to inside Hansen loss mask (for small-loss AOIs)
    loss_mask_2d = None
    if restrict_to_loss_mask and per_epoch_arrays:
        qr = _quality_report(base)
        if qr.get("event_year") is not None:
            lm_path = base / "masks" / f"loss_year_{qr['event_year']}.tif"
            first = next(iter(per_epoch_arrays.values()))
            loss_mask_2d = _resize_mask_to(lm_path, first.shape[1:])

    def _arr_mean(a):
        if loss_mask_2d is not None and loss_mask_2d.any():
            return _nan_mean(a[loss_mask_2d])
        return _nan_mean(a)

    indices_out = {}
    for idx in indices:
        per = {}
        for ep, arr in per_epoch_arrays.items():
            if idx == "NDVI":
                per[ep] = _arr_mean(_ndvi(arr[S2_BAND_IDX["B8"]], arr[S2_BAND_IDX["B4"]]))
            elif idx == "NBR":
                per[ep] = _arr_mean(_nbr(arr[S2_BAND_IDX["B8"]], arr[S2_BAND_IDX["B12"]]))
        pre_vals  = [per[e] for e in PRE_EPOCHS  if e in per and np.isfinite(per[e])]
        post_vals = [per[e] for e in POST_EPOCHS if e in per and np.isfinite(per[e])]
        pre_m  = float(np.mean(pre_vals))  if pre_vals  else float("nan")
        post_m = float(np.mean(post_vals)) if post_vals else float("nan")
        indices_out[idx] = {
            "per_epoch_mean": per,
            "pre_mean": pre_m, "post_mean": post_m,
            "drop": pre_m - post_m if np.isfinite(pre_m) and np.isfinite(post_m) else float("nan"),
        }

    aoi_pixel_count = None
    if per_epoch_arrays:
        first = next(iter(per_epoch_arrays.values()))
        aoi_pixel_count = int(np.isfinite(first[0]).sum())

    return {"indices": indices_out,
            "aoi_pixel_count": aoi_pixel_count,
            "cloud_flags_per_epoch": cloud_flags,
            "restricted_to_loss_mask": bool(loss_mask_2d is not None and loss_mask_2d.any())}


# --------------------------------------------------------------------------------------
# 3. GetSAREvidence
# --------------------------------------------------------------------------------------

def _db_mean(arr):
    """GEE COPERNICUS/S1_GRD returns sigma0 in dB already (negative values).
    If we receive positive linear values instead, convert; otherwise pass through."""
    a = arr.astype("float64").copy()
    a[a == 0] = np.nan
    if np.nanmin(a) >= 0 and np.nanmax(a) < 5:
        # positive linear (unlikely with GEE GRD but defensive)
        a[a <= 0] = np.nan
        return _nan_mean(10.0 * np.log10(a))
    return _nan_mean(a)


def GetSAREvidence(
    sample_dir: str,
    metrics=("VV", "VH", "VH-VV", "ratio_change"),
    s1_windows: Optional[dict] = None,
) -> dict:
    base = _root(sample_dir)
    files = _declared_glob(base / "cache/s1_grd", "*.tif")
    # The supported online runtime passes authoritative per-file roles and
    # acquisition dates from the opened SAR handle. Direct/raw-tool callers
    # retain a deterministic filename-order fallback, but it deliberately
    # carries no invented acquisition dates.
    if s1_windows is None:
        s1_windows = _quality_report(base)["s1_windows"]
        metadata_source = "sorted_filename_fallback"
    else:
        metadata_source = "declared_handle_metadata"
    pre_arrs, post_arrs = [], []
    pre_dates, post_dates = [], []
    for f in files:
        _block_assert_no_gt(f)
        with rasterio.open(_authorize_read(f, "raster_read")) as ds:
            arr = ds.read().astype("float32")          # (2, H, W) = VV, VH (linear)
        nm = f.stem
        window = s1_windows.get(nm) or {}
        role = str(window.get("temporal_role") or "").lower()
        date = window.get("target_date") or window.get("acquisition_date")
        if role == "pre":
            pre_arrs.append(arr)
            if date:
                pre_dates.append(str(date))
        elif role == "post":
            post_arrs.append(arr)
            if date:
                post_dates.append(str(date))
        else:
            raise ValueError(f"missing pre/post role for declared SAR file {nm}")

    if not pre_arrs or not post_arrs:
        raise ValueError(
            f"SAR evidence requires non-empty pre and post groups; "
            f"got {len(pre_arrs)} pre and {len(post_arrs)} post"
        )

    def _agg(arrs, idx):
        if not arrs:
            return float("nan")
        vals = [_db_mean(a[idx]) for a in arrs]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    vv_pre  = _agg(pre_arrs, 0); vv_post = _agg(post_arrs, 0)
    vh_pre  = _agg(pre_arrs, 1); vh_post = _agg(post_arrs, 1)

    return {
        "polarizations": {
            "VV": {"pre_mean_db": vv_pre, "post_mean_db": vv_post,
                   "delta_db": vv_post - vv_pre},
            "VH": {"pre_mean_db": vh_pre, "post_mean_db": vh_post,
                   "delta_db": vh_post - vh_pre},
        },
        "VH_minus_VV": {"pre": vh_pre - vv_pre, "post": vh_post - vv_post,
                        "delta": (vh_post - vv_post) - (vh_pre - vv_pre)},
        "ratio_change_db": (vh_post - vv_post) - (vh_pre - vv_pre),
        "epoch_count_pre": len(pre_arrs),
        "epoch_count_post": len(post_arrs),
        "pre_dates": pre_dates, "post_dates": post_dates,
        "temporal_metadata_source": metadata_source,
    }


# --------------------------------------------------------------------------------------
# 4. Hansen baseline/loss statistics
# --------------------------------------------------------------------------------------

def GetBaselineCanopyStats(sample_dir: str) -> dict:
    """Measure baseline Hansen canopy without reading unrelated loss products.

    Zero is a valid tree-cover percentage.  Only non-finite pixels and the
    raster's declared finite nodata value are excluded.  The online worker's
    declared-read scope additionally guarantees that this function can read
    only the exact treecover2000 path exposed by the question card.
    """
    base = _root(sample_dir)
    tc_path = base / "masks/treecover2000.tif"
    _block_assert_no_gt(tc_path)
    if not _exists_for_read(tc_path):
        raise FileNotFoundError(f"treecover2000 raster is unavailable: {tc_path}")

    with rasterio.open(_authorize_read(tc_path, "raster_read")) as ds:
        tc = ds.read(1).astype("float32")
        nodata_value = ds.nodata
        valid = np.isfinite(tc)
        if nodata_value is not None and np.isfinite(nodata_value):
            valid &= tc != nodata_value
        if not valid.any():
            raise ValueError(f"treecover2000 has no valid pixels: {tc_path}")
        values = tc[valid]
        if np.any((values < 0) | (values > 100)):
            raise ValueError(f"treecover2000 value outside 0..100: {tc_path}")

    mean_pct = float(np.mean(values))
    return {
        "treecover2000_mean_pct": mean_pct,
        "treecover2000_pct_gt_60": float((values > 60).mean()),
        "valid_pixel_count": int(values.size),
        "nodata_value": nodata_value,
        "measurement_contract": "hansen_aoi_mean_canopy_v1",
    }


def GetForestLossHistory(sample_dir: str, start_year: int = 2001, end_year: int = 2023) -> dict:
    """Measure annual Hansen loss area from the full lossyear raster only."""
    if (start_year, end_year) != (2001, 2023):
        raise ValueError("forest-loss history supports exactly 2001 through 2023")
    base = _root(sample_dir)
    path = base / "masks/lossyear.tif"
    _block_assert_no_gt(path)
    if not _exists_for_read(path):
        raise FileNotFoundError(f"lossyear raster is unavailable: {path}")

    with rasterio.open(_authorize_read(path, "raster_read")) as ds:
        values = ds.read(1).astype("float32")
        valid = np.isfinite(values)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            valid &= values != ds.nodata
        if not valid.any():
            raise ValueError(f"lossyear has no valid pixels: {path}")
        observed = values[valid]
        if np.any(observed != np.floor(observed)) or np.any((observed < 0) | (observed > 25)):
            raise ValueError(f"lossyear contains a value outside integer codes 0..25: {path}")
        histogram = {}
        for year in range(start_year, end_year + 1):
            mask = valid & (values == year - 2000)
            if mask.any():
                histogram[year] = round(_mask_area_m2(ds, mask) / 10000.0, 6)

    return {
        "loss_year_histogram_ha": histogram,
        "valid_pixel_count": int(valid.sum()),
        "analysis_window": {"start_year": start_year, "end_year": end_year},
        "measurement_contract": "hansen_grid_area_by_year_v1",
    }

def GetForestMask(sample_dir: str, year_filter="event_year_only") -> dict:
    base = _root(sample_dir)

    tc_path = base / "masks/treecover2000.tif"
    ly_path = base / "masks/lossyear.tif"
    _block_assert_no_gt(tc_path); _block_assert_no_gt(ly_path)

    if not _exists_for_read(tc_path) or not _exists_for_read(ly_path):
        return {"error": "hansen_unavailable"}

    with rasterio.open(_authorize_read(tc_path, "raster_read")) as ds:
        tc = ds.read(1).astype("float32")
        tc_valid = np.isfinite(tc)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            tc_valid &= tc != ds.nodata
    with rasterio.open(_authorize_read(ly_path, "raster_read")) as ds:
        ly = ds.read(1).astype("float32")
        ly_valid = np.isfinite(ly)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            ly_valid &= ly != ds.nodata

        # Calculate every yearly area on the raster's actual grid. Geographic
        # rasters use WGS84 ellipsoidal cell areas; projected rasters use the
        # affine determinant. This is the same contract as event-year area.
        histo = {}
        for y in range(1, 24):
            year_mask = ly_valid & (ly == y)
            if year_mask.any():
                histo[2000 + y] = round(_mask_area_m2(ds, year_mask) / 10000.0, 6)

    if not tc_valid.any():
        raise ValueError(f"treecover2000 has no valid pixels: {tc_path}")
    tc_mean = float(np.mean(tc[tc_valid]))
    tc_gt_60_frac = float((tc[tc_valid] > 60).mean())

    event_year = _event_year_from_assets(base)
    generated_report = BuildQualityReport(str(base), event_year)
    event_metrics = generated_report["loss_event_year"]
    event_loss_ha = event_metrics["area_ha"]
    loss_mask_present = _exists_for_read(base / "masks" / f"loss_year_{event_year}.tif") if event_year else False

    return {
        "treecover2000_mean_pct": tc_mean,
        "treecover2000_pct_gt_60": tc_gt_60_frac,
        "loss_mask_present": loss_mask_present,
        "loss_area_ha": event_loss_ha,
        "loss_pixel_count": event_metrics["loss_pixel_count"],
        "loss_area_measurement_method": event_metrics["measurement_method"],
        "generated_quality_report": generated_report,
        "loss_year_histogram_ha": histo,
        "alert_overlap_with_change_region": None,
    }


# --------------------------------------------------------------------------------------
# 5. DetectChangeWindow
# --------------------------------------------------------------------------------------

def DetectChangeWindow(sample_dir: str, signal="multi") -> dict:
    opt = GetOpticalEvidence(sample_dir, indices=("NDVI", "NBR"))
    ndvi_series = opt["indices"]["NDVI"]["per_epoch_mean"]
    nbr_series  = opt["indices"]["NBR"]["per_epoch_mean"]
    # A candidate score uses both terms, so an epoch is eligible only when
    # both index means are present and finite.  Never replace absent NBR with
    # a numeric zero, which would silently change the declared formula.
    epochs = [
        e
        for e in EPOCH_ORDER
        if e in ndvi_series
        and np.isfinite(ndvi_series[e])
        and e in nbr_series
        and np.isfinite(nbr_series[e])
    ]
    if len(epochs) < 2:
        return {"error": "insufficient_temporal_coverage"}

    best, best_drop = None, -math.inf
    best_ndvi_drop, best_nbr_drop = None, None
    for i in range(len(epochs) - 1):
        a, b = epochs[i], epochs[i + 1]
        d_ndvi = ndvi_series[a] - ndvi_series[b]
        d_nbr = (nbr_series[a] - nbr_series[b]) if signal in ("multi", "NBR") else 0
        score = d_ndvi + 0.5 * d_nbr
        if score > best_drop:
            best_drop = score
            best = (a, b)
            best_ndvi_drop = float(d_ndvi)
            best_nbr_drop = float(d_nbr) if np.isfinite(d_nbr) else None

    confidence = "high" if best_drop > 0.25 else ("medium" if best_drop > 0.10 else "low")
    return {
        "best_window": {"start_epoch": best[0], "end_epoch": best[1]} if best else None,
        "drop_magnitude": float(best_drop) if best else float("nan"),
        "signal_used": signal,
        "confidence": confidence,
        "selected_window_evidence": {
            "ndvi_drop": best_ndvi_drop,
            "nbr_drop": best_nbr_drop,
            "composite_drop_score": float(best_drop) if best else None,
        },
        "epoch_series": [
            {"epoch": e, "NDVI": ndvi_series[e], "NBR": nbr_series.get(e)}
            for e in epochs
        ],
    }


def ApplyChangeWindowContract(change: dict, mask: dict) -> dict:
    """Condition a temporal fluctuation on confirmed event-year forest loss.

    ``DetectChangeWindow`` necessarily finds the largest adjacent fluctuation
    whenever at least two valid epochs exist.  That fluctuation is only a
    clearing interval for this dataset when the packaged Hansen event-year
    mask confirms non-trivial loss.  Keeping this decision in one helper makes
    ``DetectChange`` and ``ClassifyChange`` use identical null semantics.

    Confirmed loss without a usable temporal interval is an input/processing
    failure, not ``no_deforestation``; raise instead of silently emitting a
    misleading null result.
    """
    result = dict(change or {})
    try:
        loss_area_ha = float((mask or {}).get("loss_area_ha"))
    except (TypeError, ValueError) as exc:
        raise ValueError("change-window contract requires numeric loss_area_ha") from exc
    if not math.isfinite(loss_area_ha) or loss_area_ha < 0:
        raise ValueError("change-window contract requires finite non-negative loss_area_ha")

    clearing_confirmed = loss_area_ha >= CONFIRMED_CLEARING_MIN_LOSS_HA
    window = result.get("best_window")
    valid_window = (
        isinstance(window, dict)
        and bool(window.get("start_epoch"))
        and bool(window.get("end_epoch"))
    )
    if clearing_confirmed and not valid_window:
        raise ValueError(
            "event-year loss confirms clearing but no valid temporal interval was produced"
        )

    if not clearing_confirmed:
        result["best_window"] = None
        result["confidence"] = "none"
        result["selected_window_evidence"] = {
            "ndvi_drop": None,
            "nbr_drop": None,
            "composite_drop_score": None,
        }
    result.update(
        {
            "clearing_visible": clearing_confirmed,
            "label": "deforestation" if clearing_confirmed else "no_deforestation",
            "change_window_contract": {
                "version": CHANGE_WINDOW_CONTRACT_VERSION,
                "confirmation_signal": "hansen_event_year_loss_area_ha",
                "minimum_confirmed_loss_ha": CONFIRMED_CLEARING_MIN_LOSS_HA,
                "observed_loss_area_ha": loss_area_ha,
                "null_when_unconfirmed": True,
            },
        }
    )
    return result


def ScoreVHDeltaSeverity(vh_delta_db: float) -> dict:
    """Apply the four VH-delta buckets stated verbatim in the question.

    The metric is ``post_mean_db - pre_mean_db``.  Values must be finite so a
    missing measurement cannot be silently assigned to the ``none`` bucket.
    Boundary ownership is explicit: -3 belongs to moderate, -1 belongs to mild,
    and 0 belongs to none.
    """
    try:
        delta = float(vh_delta_db)
    except (TypeError, ValueError) as exc:
        raise ValueError("VH severity requires a numeric delta in dB") from exc
    if not math.isfinite(delta):
        raise ValueError("VH severity requires a finite delta in dB")

    if delta < -3.0:
        severity = "severe"
    elif delta < -1.0:
        severity = "moderate"
    elif delta < 0.0:
        severity = "mild"
    else:
        severity = "none"
    return {
        "version": VH_SEVERITY_CONTRACT_VERSION,
        "metric": "post_mean_db_minus_pre_mean_db",
        "units": "dB",
        "vh_delta_db": delta,
        "severity": severity,
        "thresholds_db": dict(VH_SEVERITY_THRESHOLDS_DB),
    }


# --------------------------------------------------------------------------------------
# 6. GetZBPEvidence (optional)
# --------------------------------------------------------------------------------------

def GetZBPEvidence(sample_dir: str, features=("C2", "Hw", "alpha", "PZ")) -> dict:
    base = _root(sample_dir) / "cache/s1_slc_bursts"
    if not _declared_glob(base, "*"):
        return {"unavailable": True, "reason": "slc_burst_not_fetched"}
    # not implemented in pilot
    return {"unavailable": True, "reason": "not_implemented_in_pilot"}


# --------------------------------------------------------------------------------------
# 7. ClassifyChangeType
# --------------------------------------------------------------------------------------

def ClassifyChangeType(optical: dict, sar: dict, mask: dict, zbp: Optional[dict] = None) -> dict:
    ndvi_drop = optical["indices"]["NDVI"]["drop"]
    nbr_drop  = optical["indices"]["NBR"]["drop"]
    vh_delta  = sar["polarizations"]["VH"]["delta_db"]
    vv_post   = sar["polarizations"]["VV"]["post_mean_db"]
    tc        = mask.get("treecover2000_mean_pct", 0)
    loss_ha   = mask.get("loss_area_ha", 0)

    rules = []
    cls, conf = "no_change", "low"
    fp_flags = []

    def _f(x):
        try: return float(x)
        except: return float("nan")
    ndvi_drop = _f(ndvi_drop); nbr_drop = _f(nbr_drop)
    vh_delta = _f(vh_delta);   vv_post = _f(vv_post)
    tc = _f(tc);               loss_ha = _f(loss_ha)

    required = {
        "ndvi_drop": ndvi_drop,
        "nbr_drop": nbr_drop,
        "vh_delta_db": vh_delta,
        "vv_post_mean_db": vv_post,
        "treecover2000_mean_pct": tc,
        "hansen_loss_area_ha": loss_ha,
    }
    invalid = sorted(name for name, value in required.items() if not np.isfinite(value))
    if invalid:
        raise ValueError(
            "cause attribution requires finite source measurements: "
            + ", ".join(invalid)
        )
    if loss_ha < 0 or not 0 <= tc <= 100:
        raise ValueError(
            "cause attribution requires non-negative loss area and canopy in 0..100%"
        )

    # Ordered benchmark proxy rules. These categories describe the operational
    # evidence pattern; they are not independent field-verified causal labels.
    if loss_ha >= 0.5:
        if tc >= 50:
            cls, conf = "deforestation", "high"
            rules += [f"hansen_loss>=0.5ha({loss_ha:.1f})",
                      f"treecover2000>=50%({tc:.1f})"]
            if np.isfinite(ndvi_drop) and ndvi_drop > 0:
                rules.append(f"ndvi_drop_in_loss_region({ndvi_drop:.2f})")
            if np.isfinite(vh_delta) and vh_delta < 0:
                rules.append(f"sar_vh_delta({vh_delta:.2f}dB)")
        else:
            cls, conf = "plantation_harvest", "medium"
            rules += [f"hansen_loss({loss_ha:.1f}ha)", f"treecover2000<50%({tc:.1f})"]
    # No Hansen loss but strong optical AND SAR signal -> deforestation with caveat
    elif (np.isfinite(ndvi_drop) and ndvi_drop > 0.25
          and np.isfinite(vh_delta) and vh_delta < -1.5):
        cls, conf = "deforestation", "medium"
        rules += [f"ndvi_drop>0.25({ndvi_drop:.2f})", f"vh_delta<-1.5dB({vh_delta:.2f})",
                  "no_hansen_loss_but_strong_optical_sar"]
        fp_flags.append("missing_hansen_confirmation")
    # No Hansen loss: classify confuser type
    elif np.isfinite(vv_post) and vv_post < -16:
        cls, conf = "water_flood_edge", "medium"
        rules += [f"vv_post<-16dB({vv_post:.2f})"]
    elif (np.isfinite(nbr_drop) and nbr_drop > 0.20
          and np.isfinite(ndvi_drop) and nbr_drop > ndvi_drop):
        cls, conf = "burn_bare_soil", "medium"
        rules += [f"nbr_drop>0.20({nbr_drop:.2f})", "nbr_drop>ndvi_drop", "no_hansen_loss"]
    elif np.isfinite(ndvi_drop) and ndvi_drop > 0.10:
        cls, conf = "agriculture", "medium"
        rules += [f"ndvi_drop>0.10({ndvi_drop:.2f})", "no_hansen_loss"]
    else:
        cls = "no_change"
        rules.append("no_strong_signal")

    return {
        "predicted_class": cls,
        "confidence": conf,
        "rules_fired": rules,
        "false_positive_flags": fp_flags,
    }


# --------------------------------------------------------------------------------------
# 8. Terminate
# --------------------------------------------------------------------------------------

def Terminate(sample_dir: str, answer: dict) -> dict:
    out = Path(sample_dir) / "trajectories"
    out.mkdir(exist_ok=True)
    payload = {"sample_id": Path(sample_dir).name, "answer": answer}
    return payload


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

TOOLS = {
    "LoadAOI": LoadAOI,
    "GetOpticalEvidence": GetOpticalEvidence,
    "GetSAREvidence": GetSAREvidence,
    "GetBaselineCanopyStats": GetBaselineCanopyStats,
    "GetForestLossHistory": GetForestLossHistory,
    "GetForestMask": GetForestMask,
    "DetectChangeWindow": DetectChangeWindow,
    "GetZBPEvidence": GetZBPEvidence,
    "ClassifyChangeType": ClassifyChangeType,
    "Terminate": Terminate,
}
