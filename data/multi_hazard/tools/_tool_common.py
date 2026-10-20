from __future__ import annotations

import csv
import json
import math
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from pyproj import CRS, Geod

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mhds_common import (
    DATA_ROOT,
    PRIMARY_FOOTPRINT,
    SAMPLES_ROOT,
    THRESHOLDS_LIBRARY,
    domain_and_sid_from_aoi,
    handle_key,
    read_json as _raw_read_json,
    read_raster as _raw_read_raster,
    raster_path,
    sample_dir,
    sidecar_path,
    tabular_raster_path,
)


class ToolError(Exception):
    pass


_DECLARED_READ_SCOPE: ContextVar[dict[str, Any] | None] = ContextVar(
    "multi_hazard_declared_read_scope", default=None
)


def _authorize_read(path: Path, operation: str) -> Path:
    resolved = Path(path).resolve()
    scope = _DECLARED_READ_SCOPE.get()
    if scope is not None and str(resolved) not in scope["allowed"]:
        raise PermissionError(
            f"undeclared multi-hazard sibling access denied ({operation}): {resolved}"
        )
    if scope is not None:
        record = {"path": str(resolved), "operation": operation}
        if record not in scope["reads"]:
            scope["reads"].append(record)
    return resolved


@contextmanager
def DeclaredReadScope(paths: Iterable[Path]):
    """Authorize and record every file opened by multi-hazard processors."""

    reads: list[dict[str, str]] = []
    token = _DECLARED_READ_SCOPE.set(
        {"allowed": {str(Path(path).resolve()) for path in paths}, "reads": reads}
    )
    try:
        yield reads
    finally:
        _DECLARED_READ_SCOPE.reset(token)


def read_json(path: Path) -> Any:
    return _raw_read_json(_authorize_read(path, "json_read"))


def read_raster(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _raw_read_raster(_authorize_read(path, "raster_read"))


def air_pollution_normalization(sid: str | None = None) -> Dict[str, Any]:
    """Load the declared per-sample copy of the train-only normalization contract."""
    path = (
        sample_dir("air_pollution", sid) / "normalization.json"
        if sid is not None
        else SAMPLES_ROOT / "air_pollution" / "source_normalization.json"
    )
    if not path.exists():
        raise ToolError("air_pollution_source_normalization_missing")
    contract = read_json(path)
    if contract.get("fit_split") != "train":
        raise ToolError("air_pollution_normalization_not_train_only")
    return contract


def drought_normalization(sid: str | None = None) -> Dict[str, Any]:
    """Load the declared train-only drought stress normalization contract."""
    path = (
        sample_dir("drought", sid) / "drought_normalization.json"
        if sid is not None
        else SAMPLES_ROOT / "drought" / "source_normalization.json"
    )
    if not path.exists():
        raise ToolError("drought_source_normalization_missing")
    contract = read_json(path)
    if contract.get("fit_split") != "train":
        raise ToolError("drought_normalization_not_train_only")
    return contract


def parse_sample(args: Dict[str, Any]) -> Tuple[str, str]:
    aoi = args.get("aoi_id") or args.get("sample_id")
    if not aoi:
        raise ToolError("missing_aoi_id")
    return domain_and_sid_from_aoi(str(aoi))


def assert_allowed_path(path: Path, domain: str, sid: str) -> Path:
    path = path.resolve()
    root = sample_dir(domain, sid).resolve()
    allowed = [
        root / "rasters",
        root / "tabular",
        root / "osm",
        root / "baselines",
    ]
    if not any(str(path).startswith(str(p.resolve())) for p in allowed):
        raise ToolError(f"path_not_allowed:{path}")
    if "/gt/" in str(path):
        raise ToolError(f"path_leak_guard:{path}")
    return path


def available_handles(domain: str, sid: str) -> set[str]:
    path = sample_dir(domain, sid) / "available_handles.json"
    if not path.exists():
        raise ToolError("available_handles_missing")
    return set(read_json(path).get("handles", []))


def resolve_handle(domain: str, sid: str, handle: str) -> Tuple[Path, Dict[str, Any]]:
    if handle not in available_handles(domain, sid):
        raise ToolError(f"handle_not_cached:{handle}")
    path = assert_allowed_path(raster_path(domain, sid, handle), domain, sid)
    if not path.exists():
        raise ToolError(f"raster_missing:{handle}")
    sidecar = assert_allowed_path(sidecar_path(domain, sid, handle), domain, sid)
    if not sidecar.exists():
        raise ToolError(f"sidecar_missing:{handle}")
    return path, read_json(sidecar)


def load_handle_array(domain: str, sid: str, handle: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    path, sidecar = resolve_handle(domain, sid, handle)
    arr, profile = read_raster(path)
    values = np.asarray(arr)
    nodata = profile.get("nodata")
    if nodata is not None and np.isfinite(float(nodata)):
        values = np.where(values == float(nodata), np.nan, values)
    sidecar["profile"] = profile
    return values, sidecar


def aoi_pixel_metrics(domain: str, sid: str, shape: tuple[int, int]) -> Dict[str, float]:
    metadata = read_json(sample_dir(domain, sid) / "metadata.json")
    coords = metadata["geometry"]["coordinates"][0]
    xs = [float(c[0]) for c in coords]
    ys = [float(c[1]) for c in coords]
    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)
    radius_m = 6_371_008.8
    area_m2 = (
        radius_m
        * radius_m
        * abs(math.sin(math.radians(north)) - math.sin(math.radians(south)))
        * abs(math.radians(east - west))
    )
    height, width = int(shape[0]), int(shape[1])
    pixel_area_m2 = area_m2 / max(1, height * width)
    return {
        "aoi_area_ha": area_m2 / 10000.0,
        "pixel_area_ha": pixel_area_m2 / 10000.0,
        "pixel_size_m": math.sqrt(pixel_area_m2),
    }


def pixel_area_ha(sidecar: Dict[str, Any], domain: str | None = None, sid: str | None = None) -> float:
    grid = cell_area_ha_grid(sidecar)
    return float(np.mean(grid))


@lru_cache(maxsize=512)
def _cached_cell_area_grid(
    crs_text: str,
    transform_values: tuple[float, ...],
    height: int,
    width: int,
) -> np.ndarray:
    from affine import Affine

    transform = Affine(*transform_values[:6])
    crs = CRS.from_user_input(crs_text)
    if not crs.is_geographic:
        unit_factor = float(crs.axis_info[0].unit_conversion_factor or 1.0)
        cell_m2 = abs(transform.a * transform.e - transform.b * transform.d) * unit_factor**2
        return np.full((height, width), cell_m2 / 10000.0, dtype="float64")

    geod = Geod(ellps="WGS84")
    out = np.empty((height, width), dtype="float64")
    for row in range(height):
        for col in range(width):
            corners = [
                transform * (col, row),
                transform * (col + 1, row),
                transform * (col + 1, row + 1),
                transform * (col, row + 1),
            ]
            lons, lats = zip(*corners)
            out[row, col] = abs(geod.polygon_area_perimeter(lons, lats)[0]) / 10000.0
    out.setflags(write=False)
    return out


def cell_area_ha_grid(sidecar: Dict[str, Any]) -> np.ndarray:
    profile = sidecar.get("profile") or {}
    shape = sidecar.get("shape") or [profile.get("height"), profile.get("width")]
    height, width = int(shape[0]), int(shape[1])
    crs = profile.get("crs") or sidecar.get("crs")
    transform = profile.get("transform") or sidecar.get("transform")
    if crs is None or transform is None:
        raise ToolError("grid_metadata_missing_for_exact_area")
    values = tuple(float(value) for value in tuple(transform)[:6])
    return _cached_cell_area_grid(str(crs), values, height, width)


def mask_area_ha(mask: np.ndarray, sidecar: Dict[str, Any]) -> float:
    areas = cell_area_ha_grid(sidecar)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != areas.shape:
        raise ToolError(f"area_grid_shape_mismatch:{selected.shape}:{areas.shape}")
    return float(np.sum(areas[selected]))


def comparison_mask(domain: str, sid: str, handle: str) -> np.ndarray:
    arr, _ = load_handle_array(domain, sid, handle)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    if domain == "wildfire":
        if handle in {"dnbr", "landsat_dnbr"}:
            return arr > np.nanquantile(arr, 0.66)
        return arr > 0.5
    if domain == "deforestation":
        if handle == "ndvi_change":
            return arr > 0.30
        if handle == "radd_alerts__event":
            # Official RADD date_conf is decoded to 0=no event-window alert,
            # 1=low confidence, 2=high confidence.
            return arr > 0
        return arr > 0.5
    if domain == "urban_expansion":
        if handle == "ghsl_built__post":
            pre, _ = load_handle_array(domain, sid, "ghsl_built__pre")
            return (arr > 0.55) & (np.asarray(pre) <= 0.55)
        if handle == "worldcover_built__post":
            pre, _ = load_handle_array(domain, sid, "dynamic_world_built__pre")
            return (arr > 0.5) & (np.asarray(pre) <= 0.55)
        if handle == "ndbi_change":
            return arr > 0.18
        return arr > 0.5
    if domain == "drought":
        if handle in {
            "chirps_spi__target",
            "modis_ndvi__target",
            "era5_soil_moisture__target",
        }:
            contract = drought_normalization(sid)
            return arr < float(contract["sources"][handle]["pixel_q25"])
        return arr > 0.5
    if domain == "air_pollution":
        if handle == "high_no2_column":
            return arr > 0.5
        if handle in {"s5p_no2__peak", "cams_no2__peak", "modis_aod__peak"}:
            contract = air_pollution_normalization(sid)
            threshold = float(contract["sources"][handle]["pixel_q75"])
            return arr >= threshold
        if handle == "no2_anomaly":
            return arr > 0.0
        return arr > 0.5
    return arr > np.nanmedian(arr)


def source_label(handle: str) -> str:
    return handle_key(handle)


def classify_array(
    arr: np.ndarray,
    library_key: str | Dict[str, Any],
    pixel_ha: float,
    domain: str | None = None,
    sid: str | None = None,
    sidecar: Dict[str, Any] | None = None,
    analysis_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    if library_key == "@NO2_COLUMN_RELATIVE":
        if domain != "air_pollution" or sid is None:
            raise ToolError("air_pollution_sample_required_for_relative_classification")
        intervals = air_pollution_normalization(sid)["s5p_pixel_class_intervals"]
    elif library_key == "@SPI_RELATIVE":
        if domain != "drought" or sid is None:
            raise ToolError("drought_sample_required_for_relative_classification")
        intervals = drought_normalization(sid)["chirps_spi_pixel_class_intervals"]
    else:
        intervals = THRESHOLDS_LIBRARY[library_key] if isinstance(library_key, str) else library_key
    result: Dict[str, Any] = {}
    best = ("", -1)
    for name, bounds in intervals.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo < hi:
            mask = (arr >= lo) & (arr < hi)
        else:
            mask = (arr <= lo) & (arr > hi)
        mask &= np.isfinite(arr)
        if analysis_mask is not None:
            mask &= np.asarray(analysis_mask, dtype=bool)
        count = int(np.count_nonzero(mask))
        if count > best[1]:
            best = (name, count)
        values = arr[mask]
        result[name] = {
            "area_ha": round(
                mask_area_ha(mask, sidecar) if sidecar is not None else count * pixel_ha,
                3,
            ),
            "pixel_count": count,
            "mean_value": None if values.size == 0 else round(float(np.nanmean(values)), 6),
        }
    return {"by_class": result, "dominant_class": best[0]}


def read_baseline_csv(domain: str, sid: str, modality: str) -> list[dict[str, str]]:
    path = assert_allowed_path(sample_dir(domain, sid) / "baselines" / f"{modality}.csv", domain, sid)
    if not path.exists():
        raise ToolError(f"baseline_missing:{modality}")
    _authorize_read(path, "csv_read")
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_worldpop(domain: str, sid: str) -> tuple[np.ndarray, Dict[str, Any]]:
    path = assert_allowed_path(tabular_raster_path(domain, sid, "worldpop.tif"), domain, sid)
    arr, profile = read_raster(path)
    values = np.asarray(arr, dtype=float)
    nodata = profile.get("nodata")
    if nodata is not None and np.isfinite(float(nodata)):
        values = np.where(values == float(nodata), 0.0, values)
    values = np.where(np.isfinite(values), values, 0.0)
    return values, profile


def worldpop_metadata(domain: str, sid: str) -> Dict[str, Any]:
    path = assert_allowed_path(
        sample_dir(domain, sid) / "tabular" / "worldpop.json", domain, sid
    )
    if not path.exists():
        raise ToolError("worldpop_metadata_missing")
    return read_json(path)


def load_pois(domain: str, sid: str) -> list[dict[str, Any]]:
    path = assert_allowed_path(sample_dir(domain, sid) / "osm" / "pois.geojson", domain, sid)
    data = read_json(path)
    return list(data.get("features", []))


def nearest_distance_to_mask(row: int, col: int, mask: np.ndarray, scale_m: float) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("inf")
    d2 = (ys - row) ** 2 + (xs - col) ** 2
    return float(math.sqrt(float(np.min(d2))) * scale_m)


def error_response(exc: Exception) -> Dict[str, Any]:
    return {"error": str(exc)}
