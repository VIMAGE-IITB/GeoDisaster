from __future__ import annotations

import csv
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds


DATA_ROOT = Path(__file__).resolve().parent


def _resolve_samples_root() -> Path:
    env_root = os.environ.get("GEODISASTER_DATASET_ROOT", "").strip()
    if env_root:
        public_root = Path(env_root).expanduser().resolve() / "source_assets" / "multi_hazard"
        if public_root.exists():
            return public_root
    return DATA_ROOT / "samples"


SAMPLES_ROOT = _resolve_samples_root()
REJECTED_ROOT = DATA_ROOT / "_rejected"
QUESTIONS_ROOT = DATA_ROOT / "questions"
RASTER_SIZE = 128
DEFAULT_SCALE_M = 100
TARGET_ACCEPTED_PER_DOMAIN = 30
CANDIDATES_PER_DOMAIN = 40
BUILD_SOURCE_MODE = os.environ.get("MHDS_BUILD_SOURCE_MODE", "synthetic_fallback")
LIVE_FETCH_BLOCKER = "Live remote-source fetch has not been run for this cache."


DOMAINS = [
    "wildfire",
    "deforestation",
    "urban_expansion",
    "drought",
    "air_pollution",
]


DOMAIN_LABELS = {
    "wildfire": "Wildfire",
    "deforestation": "Deforestation",
    "urban_expansion": "Urban expansion",
    "drought": "Drought",
    "air_pollution": "Air pollution",
}


DOMAIN_REGIONS = {
    "wildfire": [
        ("Camp Fire", -121.62, 39.76),
        ("Dixie Fire", -121.25, 40.20),
        ("Woolsey Fire", -118.72, 34.10),
        ("Bootleg Fire", -121.32, 42.61),
    ],
    "deforestation": [
        ("Para clearing", -52.10, -3.25),
        ("Rondonia clearing", -63.91, -10.85),
        ("Madre de Dios clearing", -69.21, -12.52),
        ("Borneo clearing", 117.34, 1.04),
    ],
    "urban_expansion": [
        ("Bengaluru growth", 77.59, 12.97),
        ("Nairobi growth", 36.82, -1.29),
        ("Lagos growth", 3.38, 6.52),
        ("Phoenix growth", -112.07, 33.45),
    ],
    "drought": [
        ("Horn of Africa drought", 39.40, 3.30),
        ("California Central Valley drought", -119.60, 36.50),
        ("Murray Darling drought", 145.10, -34.90),
        ("Deccan drought", 76.90, 17.40),
    ],
    "air_pollution": [
        ("Delhi pollution episode", 77.21, 28.61),
        ("Los Angeles pollution episode", -118.24, 34.05),
        ("Milan pollution episode", 9.19, 45.46),
        ("Santiago pollution episode", -70.66, -33.45),
    ],
}


DOMAIN_DATES = {
    "wildfire": ("2021-07-01", "2021-08-15", "2021-08"),
    "deforestation": ("2022-06-01", "2022-09-30", "2022"),
    "urban_expansion": ("2019-01-01", "2024-01-01", "2019-2024"),
    "drought": ("2024-08-15", "2024-08-15", "2024-08-15"),
    "air_pollution": ("2024-11-08", "2024-11-12", "2024-11-10"),
}


SOURCE_HANDLES = {
    "wildfire": [
        "mcd64a1__event",
        "viirs_active_fire__event",
        "s2_nbr__pre",
        "s2_nbr__post",
        "s2_rgb__pre",
        "s2_rgb__post",
        "landsat_nbr__pre",
        "landsat_nbr__post",
        "nifc_perimeter__event",
    ],
    "deforestation": [
        "hansen_loss__event",
        "tmf_loss__event",
        "radd_alerts__event",
        "s2_ndvi__pre",
        "s2_ndvi__post",
        "forest_mask__pre",
    ],
    "urban_expansion": [
        "dynamic_world_built__pre",
        "dynamic_world_built__post",
        "ghsl_built__pre",
        "ghsl_built__post",
        "worldcover_built__post",
        "s2_ndbi__pre",
        "s2_ndbi__post",
    ],
    "drought": [
        "chirps_spi__target",
        "era5_soil_moisture__target",
        "modis_ndvi__target",
        "viirs_lst__target",
    ],
    "air_pollution": [
        "s5p_no2__peak",
        "s5p_so2__peak",
        "s5p_co__peak",
        "modis_aod__peak",
        "cams_no2__peak",
    ],
}


DERIVED_HANDLES = {
    "wildfire": ["dnbr", "landsat_dnbr", "s2_burn_support"],
    "deforestation": ["ndvi_change"],
    "urban_expansion": ["ndbi_change", "built_gain", "ghsl_built_gain", "worldcover_built_gain"],
    "drought": ["low_chirps_spi"],
    "air_pollution": ["no2_anomaly", "high_no2_column"],
}


COMPARISON_HANDLES = {
    "wildfire": ["mcd64a1__event", "s2_burn_support", "nifc_perimeter__event"],
    "deforestation": ["hansen_loss__event", "tmf_loss__event", "radd_alerts__event"],
    "urban_expansion": ["built_gain", "ghsl_built_gain", "worldcover_built_gain"],
    "drought": ["chirps_spi__target", "modis_ndvi__target", "era5_soil_moisture__target"],
    "air_pollution": ["s5p_no2__peak", "cams_no2__peak", "modis_aod__peak"],
}


PRIMARY_FOOTPRINT = {
    "wildfire": "mcd64a1__event",
    "deforestation": "hansen_loss__event",
    "urban_expansion": "built_gain",
    "drought": "low_chirps_spi",
    "air_pollution": "high_no2_column",
}


HANDLE_META = {
    "mcd64a1": ("burned_area_mask", "MODIS MCD64A1.061", "binary", 500),
    "viirs_active_fire": ("active_fire_mask", "NASA VIIRS VNP14A1.002", "binary", 1000),
    "s2_nbr": ("sentinel2_nbr", "Sentinel-2 SR harmonized", "index", 10),
    "s2_rgb": ("sentinel2_rgb", "Sentinel-2 SR harmonized", "reflectance", 10),
    "landsat_nbr": ("landsat_nbr", "Landsat Collection 2", "index", 30),
    "nifc_perimeter": ("incident_perimeter", "NIFC WFIGS incident perimeter", "binary", 100),
    "dnbr": ("s2_nbr_change", "derived from Sentinel-2 NBR", "index", 10),
    "landsat_dnbr": ("landsat_nbr_change", "derived from Landsat NBR", "index", 30),
    "hansen_loss": ("forest_loss_mask", "Hansen GFC v1.13", "binary", 30),
    "tmf_loss": ("forest_loss_mask", "JRC TMF annual loss", "binary", 30),
    "radd_alerts": ("radar_deforestation_alert", "WUR/GFW RADD v1 date_conf", "confidence_class", 10),
    "s2_ndvi": ("sentinel2_ndvi", "Sentinel-2 SR harmonized", "index", 10),
    "forest_mask": ("forest_mask", "Hansen GFC tree cover", "binary", 30),
    "ndvi_change": ("s2_ndvi_change", "derived from Sentinel-2 NDVI", "index", 10),
    "dynamic_world_built": ("dynamic_world_built", "Dynamic World V1", "probability", 10),
    "ghsl_built": ("ghsl_built_surface", "GHSL built-up surface", "probability", 100),
    "worldcover_built": ("esa_worldcover_built", "ESA WorldCover", "binary", 10),
    "s2_ndbi": ("sentinel2_ndbi", "Sentinel-2 SR harmonized", "index", 10),
    "ndbi_change": ("s2_ndbi_change", "derived from Sentinel-2 NDBI", "index", 10),
    "built_gain": ("built_up_gain", "derived built-up gain", "binary", 10),
    "chirps_spi": ("chirps_spi", "CHIRPS rainfall SPI", "z_score", 5500),
    "era5_soil_moisture": ("era5_soil_moisture", "ERA5-Land soil moisture", "m3_m3", 9000),
    "modis_ndvi": ("modis_ndvi_anomaly", "MODIS NDVI anomaly", "index", 250),
    "viirs_lst": ("viirs_lst", "NASA VIIRS VNP21A1D.002", "K", 1000),
    "s5p_no2": ("sentinel5p_no2", "Sentinel-5P TROPOMI NO2", "umol_m2", 5500),
    "s5p_so2": ("sentinel5p_so2", "Sentinel-5P TROPOMI SO2", "umol_m2", 5500),
    "s5p_co": ("sentinel5p_co", "Sentinel-5P TROPOMI CO", "mol_m2", 5500),
    "modis_aod": ("modis_maiac_aod", "MODIS MAIAC AOD", "aod", 1000),
    "cams_no2": ("cams_no2_column", "CAMS NRT total-column NO2", "kg_m2", 44528),
    "no2_anomaly": ("no2_anomaly", "derived NO2 anomaly", "umol_m2", 5500),
    "high_no2_column": ("high_no2_column_footprint", "derived training-relative S5P NO2-column quartile", "binary", 5500),
}


THRESHOLDS_LIBRARY = {
    "@dNBR": {
        "unburned": [-10.0, 0.1],
        "low": [0.1, 0.27],
        "moderate": [0.27, 0.44],
        "high": [0.44, 10.0],
    },
    "@NDVI_change": {
        "none": [-10.0, 0.15],
        "minor": [0.15, 0.30],
        "major": [0.30, 0.50],
        "severe": [0.50, 10.0],
    },
    "@NDBI_change": {
        "contraction": [-10.0, -0.02],
        "stable": [-0.02, 0.02],
        "minor": [0.02, 0.18],
        "major": [0.18, 10.0],
    },
    "@SPI_RELATIVE": {},
    # The numeric boundaries for this contract are loaded from the
    # training-split source-normalization artifact at runtime.  No WHO or
    # surface-concentration interpretation is attached to these quartiles.
    "@NO2_COLUMN_RELATIVE": {},
}


INTAKE_RULES = {
    "wildfire": {"metric": "area_agreement", "max_median_pct_delta": 20.0, "min_accepted": 28},
    "deforestation": {"metric": "area_agreement", "max_median_pct_delta": 25.0, "min_accepted": 28},
    "urban_expansion": {"metric": "area_agreement", "max_median_pct_delta": 30.0, "min_accepted": 28},
    "drought": {"metric": "iou", "min_iou": 0.30, "min_accepted": 28},
    "air_pollution": {
        "metric": "benchmark_relative_intensity_agreement",
        "state_z_boundaries": [-0.5, 0.5],
        "required_agreement_fraction": 1.0,
        "min_accepted": 28,
    },
}


QUESTION_TEMPLATES = {
    "wildfire": {
        "A": "What is the burned area in hectares for {event_name}?",
        "B": "What is the dominant Sentinel-2 dNBR severity class inside the MCD64A1 burn footprint for {event_name}?",
        "C": "Do the MCD64A1 burned-area, fixed-threshold Sentinel-2 dNBR support, and NIFC perimeter masks agree in area for {event_name}?",
        "D": "How many cached OSM highway representative points are within 5 km of the burn footprint for {event_name}?",
    },
    "deforestation": {
        "A": "What is the forest loss area in hectares for {event_name} during {window_label}?",
        "B": "What is the dominant Sentinel-2 NDVI-drop class inside the Hansen 2022 loss footprint for {event_name}?",
        "C": "Do the 2022 Hansen loss, JRC TMF annual-loss, and RADD event-alert masks agree in area for {event_name}?",
        "D": "What is the distance in meters from the forest-loss footprint to the nearest cached OSM highway representative point for {event_name}?",
    },
    "urban_expansion": {
        "A": "What is the new built-up area in hectares for {event_name} from 2019 to 2024?",
        "B": "What is the dominant whole-AOI Sentinel-2 NDBI change class for {event_name}?",
        "C": "Do Dynamic World 2019-2024, GHSL 2020-2025, and WorldCover 2020-2021 built-up gain masks agree in area for {event_name}?",
        "D": "What population was added inside the new built-up footprint for {event_name}?",
    },
    "drought": {
        "A": "What is the NDVI z-score over {event_name} on {target_date}?",
        "B": "What is the dominant train-relative CHIRPS SPI pixel quartile over {event_name} on {target_date}?",
        "C": "Do the train-relative bottom-quartile CHIRPS SPI, MODIS NDVI, and ERA5 soil-moisture stress masks reach median pairwise IoU 0.30 for {event_name}?",
        "D": "How many schools are inside the train-relative bottom-quartile CHIRPS SPI footprint for {event_name}?",
    },
    "air_pollution": {
        "A": "What is the peak NO2 column during {event_name}?",
        "B": "What is the training-relative S5P NO2-column quartile class at the peak for {event_name}?",
        "C": "Do Sentinel-5P NO2, CAMS NO2, and MODIS AOD agree on the training-relative episode-intensity state for {event_name}?",
        "D": "What population is located within the training-defined top-quartile S5P NO2-column footprint during {event_name}?",
    },
}


EXPECTED_SHAPES = {
    "A": {"value": "float", "units": "string"},
    "B": {"class": "string", "dominant_fraction": "float"},
    "C": {"agree": "bool", "median_area_ha": "float", "median_pct_delta": "float", "verdict": "string"},
    "D": {"population": "float", "asset_counts": "object", "nearest": "object"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sample_id(domain: str, idx: int) -> str:
    return f"sample_{idx:06d}"


def sample_dir(domain: str, sid: str) -> Path:
    return SAMPLES_ROOT / domain / sid


def rejected_dir(domain: str, sid: str) -> Path:
    return REJECTED_ROOT / domain / sid


def domain_and_sid_from_aoi(aoi_id: str) -> Tuple[str, str]:
    parts = aoi_id.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected aoi_id as '<domain>__<sample_id>', got {aoi_id!r}")
    return parts[0], parts[1]


def aoi_id(domain: str, sid: str) -> str:
    return f"{domain}__{sid}"


def handle_key(handle: str) -> str:
    candidates = sorted(HANDLE_META, key=len, reverse=True)
    for key in candidates:
        if handle == key or handle.startswith(f"{key}__"):
            return key
    return handle.split("__", 1)[0]


def handle_metadata(domain: str, handle: str) -> Dict[str, Any]:
    key = handle_key(handle)
    modality, source, units, scale_m = HANDLE_META.get(
        key, ("unknown", key, "unknown", DEFAULT_SCALE_M)
    )
    _, _, window = DOMAIN_DATES[domain]
    if "__pre" in handle:
        dates = [DOMAIN_DATES[domain][0]]
    elif "__post" in handle:
        dates = [DOMAIN_DATES[domain][1]]
    elif "__peak" in handle or "__target" in handle:
        dates = [DOMAIN_DATES[domain][2]]
    else:
        dates = [window]
    return {
        "handle": handle,
        "modality": modality,
        "source": source,
        "units": units,
        "scale_m": scale_m,
        "acquisition_dates": dates,
    }


def all_handles(domain: str) -> List[str]:
    return SOURCE_HANDLES[domain] + DERIVED_HANDLES[domain]


def raster_path(domain: str, sid: str, handle: str) -> Path:
    return sample_dir(domain, sid) / "rasters" / f"{handle}.tif"


def sidecar_path(domain: str, sid: str, handle: str) -> Path:
    return sample_dir(domain, sid) / "rasters" / f"{handle}.json"


def tabular_raster_path(domain: str, sid: str, name: str) -> Path:
    return sample_dir(domain, sid) / "tabular" / name


def geometry_bounds(geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return (west, south, east, north) for a GeoJSON geometry.

    The cache builder writes every raster on the sample AOI grid.  Bounds are
    therefore an explicit required input; a nominal source resolution is not
    sufficient to geolocate an output.
    """

    points: List[Tuple[float, float]] = []

    def collect(value: Any) -> None:
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(geometry.get("coordinates"))
    if not points:
        raise ValueError("GeoJSON geometry contains no coordinate pairs")
    xs, ys = zip(*points)
    result = (min(xs), min(ys), max(xs), max(ys))
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"Degenerate AOI bounds: {result}")
    return result


def sample_bounds(domain: str, sid: str) -> Tuple[float, float, float, float]:
    metadata_path = sample_dir(domain, sid) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Sample metadata is required before raster creation: {metadata_path}")
    metadata = read_json(metadata_path)
    geometry = metadata.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"Sample metadata has no valid geometry: {metadata_path}")
    return geometry_bounds(geometry)


def write_raster(
    path: Path,
    array: np.ndarray,
    bounds: Tuple[float, float, float, float],
    dtype: str = "float32",
    nodata: float | None = None,
    crs: str = "EPSG:4326",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.ndim == 2:
        bands = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        bands = arr
    else:
        raise ValueError(f"Unsupported raster shape {arr.shape}")
    height, width = bands.shape[1], bands.shape[2]
    west, south, east, north = (float(value) for value in bounds)
    if west >= east or south >= north:
        raise ValueError(f"Degenerate raster bounds: {bounds}")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands.shape[0],
        "dtype": dtype,
        "crs": crs,
        "transform": from_bounds(west, south, east, north, width, height),
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands.astype(dtype))


def read_raster(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rasterio.open(path) as src:
        data = src.read()
        profile = dict(src.profile)
    if data.shape[0] == 1:
        data = data[0]
    return data, profile


def write_handle_raster(
    domain: str,
    sid: str,
    handle: str,
    array: np.ndarray,
    source_mode: str | None = None,
    extra_sidecar: Dict[str, Any] | None = None,
) -> None:
    meta = handle_metadata(domain, handle)
    write_raster(raster_path(domain, sid, handle), array, bounds=sample_bounds(domain, sid))
    n_bands = int(array.shape[0]) if np.asarray(array).ndim == 3 else 1
    mode = source_mode or BUILD_SOURCE_MODE
    sidecar = {
        **meta,
        "geotiff_path": str(raster_path(domain, sid, handle).relative_to(DATA_ROOT)),
        "shape": [int(np.asarray(array).shape[-2]), int(np.asarray(array).shape[-1])],
        "n_bands": n_bands,
        "clear_fraction_in_aoi": 1.0,
        "scene_coverage": 1.0,
        "nan_pct": float(np.isnan(np.asarray(array, dtype=float)).mean() * 100.0),
        "fetched_at_build_time": utc_now(),
        "source_mode": mode,
    }
    if mode != "gee_live":
        sidecar["live_fetch_blocker"] = LIVE_FETCH_BLOCKER
    if extra_sidecar:
        sidecar.update(extra_sidecar)
    write_json(sidecar_path(domain, sid, handle), sidecar)


def feature_polygon(lon: float, lat: float, half_size_deg: float = 0.08) -> Dict[str, Any]:
    coords = [
        [lon - half_size_deg, lat - half_size_deg],
        [lon + half_size_deg, lat - half_size_deg],
        [lon + half_size_deg, lat + half_size_deg],
        [lon - half_size_deg, lat + half_size_deg],
        [lon - half_size_deg, lat - half_size_deg],
    ]
    return {"type": "Polygon", "coordinates": [coords]}


def build_manifest_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for domain in DOMAINS:
        start_date, end_date, window_label = DOMAIN_DATES[domain]
        locations = DOMAIN_REGIONS[domain]
        for idx in range(1, CANDIDATES_PER_DOMAIN + 1):
            region_name, lon, lat = locations[(idx - 1) % len(locations)]
            rng = random.Random(f"{domain}-{idx}")
            lon_j = lon + rng.uniform(-0.22, 0.22)
            lat_j = lat + rng.uniform(-0.18, 0.18)
            sid = sample_id(domain, idx)
            rows.append(
                {
                    "aoi_id": aoi_id(domain, sid),
                    "sample_id": sid,
                    "domain": domain,
                    "candidate_index": idx,
                    "event_name": f"{region_name} AOI {idx:02d}",
                    "start_date": start_date,
                    "end_date": end_date,
                    "target_date": window_label,
                    "window_label": window_label,
                    "geometry": feature_polygon(lon_j, lat_j),
                    "source_mode_expected": BUILD_SOURCE_MODE,
                    "split": "hidden_test" if idx % 4 == 0 else "train_val",
                    "accepted": idx <= TARGET_ACCEPTED_PER_DOMAIN,
                }
            )
    return rows


def write_aoi_files(row: Dict[str, Any]) -> None:
    domain = row["domain"]
    sid = row["sample_id"]
    sdir = sample_dir(domain, sid)
    sdir.mkdir(parents=True, exist_ok=True)
    source_mode = row.get("source_mode") or row.get("source_mode_expected") or BUILD_SOURCE_MODE
    metadata = {
        **row,
        "domain_label": DOMAIN_LABELS[domain],
        "source_mode": source_mode,
        "built_at": utc_now(),
    }
    if source_mode != "gee_live":
        metadata["live_fetch_blocker"] = LIVE_FETCH_BLOCKER
    write_json(sdir / "metadata.json", metadata)
    write_json(
        sdir / "aoi.geojson",
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": row["geometry"]}]},
    )


def rng_for(domain: str, sid: str) -> np.random.Generator:
    seed = abs(hash((domain, sid))) % (2**32)
    return np.random.default_rng(seed)


def synthetic_field(domain: str, sid: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng_for(domain, sid)
    y, x = np.mgrid[-1.0:1.0:complex(RASTER_SIZE), -1.0:1.0:complex(RASTER_SIZE)]
    cx, cy = rng.uniform(-0.25, 0.25, size=2)
    sx, sy = rng.uniform(0.32, 0.58, size=2)
    theta = rng.uniform(0, math.pi)
    xr = (x - cx) * math.cos(theta) + (y - cy) * math.sin(theta)
    yr = -(x - cx) * math.sin(theta) + (y - cy) * math.cos(theta)
    plume = np.exp(-((xr / sx) ** 2 + (yr / sy) ** 2))
    ridge = 0.16 * np.sin(5 * x + rng.uniform(0, 2)) + 0.11 * np.cos(4 * y + rng.uniform(0, 2))
    field = np.clip(plume + ridge + rng.normal(0, 0.035, size=plume.shape), 0, 1)
    mask = field > np.quantile(field, 0.66)
    texture = np.clip(0.50 + 0.20 * np.sin(3 * x) + 0.13 * np.cos(4 * y) + rng.normal(0, 0.04, field.shape), 0, 1)
    return field.astype("float32"), mask.astype("float32"), texture.astype("float32")


def jitter_mask(base: np.ndarray, rng: np.random.Generator, threshold_shift: float = 0.0) -> np.ndarray:
    arr = np.clip(base + rng.normal(0, 0.06, base.shape) + threshold_shift, 0, 1)
    return (arr > np.quantile(arr, 0.66)).astype("float32")


def write_source_cache_for_sample(row: Dict[str, Any]) -> None:
    domain = row["domain"]
    sid = row["sample_id"]
    sdir = sample_dir(domain, sid)
    if (sdir / "quality_report.json").exists():
        q = read_json(sdir / "quality_report.json")
        if q.get("all_sources_ok") is True:
            return
    write_aoi_files(row)
    rng = rng_for(domain, sid)
    field, mask, texture = synthetic_field(domain, sid)

    if domain == "wildfire":
        write_handle_raster(domain, sid, "mcd64a1__event", mask)
        write_handle_raster(domain, sid, "viirs_active_fire__event", jitter_mask(field, rng, -0.02))
        pre = np.clip(0.62 + 0.18 * texture + rng.normal(0, 0.025, field.shape), -1, 1)
        post = np.clip(pre - 0.72 * field - 0.05 * mask + rng.normal(0, 0.025, field.shape), -1, 1)
        l_pre = np.clip(pre + rng.normal(0, 0.03, field.shape), -1, 1)
        l_post = np.clip(post + rng.normal(0, 0.035, field.shape), -1, 1)
        write_handle_raster(domain, sid, "s2_nbr__pre", pre)
        write_handle_raster(domain, sid, "s2_nbr__post", post)
        write_handle_raster(domain, sid, "landsat_nbr__pre", l_pre)
        write_handle_raster(domain, sid, "landsat_nbr__post", l_post)
        rgb_pre = np.stack([0.30 + 0.25 * texture, 0.45 + 0.30 * texture, 0.25 + 0.18 * texture])
        rgb_post = np.stack([rgb_pre[0] + 0.35 * field, rgb_pre[1] - 0.30 * field, rgb_pre[2] - 0.12 * field])
        write_handle_raster(domain, sid, "s2_rgb__pre", np.clip(rgb_pre, 0, 1))
        write_handle_raster(domain, sid, "s2_rgb__post", np.clip(rgb_post, 0, 1))
        write_handle_raster(domain, sid, "nifc_perimeter__event", jitter_mask(field, rng, 0.01))

    elif domain == "deforestation":
        write_handle_raster(domain, sid, "hansen_loss__event", mask)
        write_handle_raster(domain, sid, "tmf_loss__event", jitter_mask(field, rng, -0.01))
        write_handle_raster(domain, sid, "radd_alerts__event", np.clip(field + rng.normal(0, 0.08, field.shape), 0, 1))
        ndvi_pre = np.clip(0.68 + 0.22 * texture + rng.normal(0, 0.025, field.shape), -1, 1)
        ndvi_post = np.clip(ndvi_pre - 0.62 * field - 0.12 * mask + rng.normal(0, 0.035, field.shape), -1, 1)
        write_handle_raster(domain, sid, "s2_ndvi__pre", ndvi_pre)
        write_handle_raster(domain, sid, "s2_ndvi__post", ndvi_post)
        write_handle_raster(domain, sid, "forest_mask__pre", (texture > 0.42).astype("float32"))

    elif domain == "urban_expansion":
        built_pre = ((texture + 0.20 * np.roll(texture, 12, axis=1)) > 0.70).astype("float32")
        gain = mask * (1.0 - built_pre)
        built_post = np.clip(built_pre + gain, 0, 1)
        write_handle_raster(domain, sid, "dynamic_world_built__pre", np.clip(built_pre + rng.normal(0, 0.06, field.shape), 0, 1))
        write_handle_raster(domain, sid, "dynamic_world_built__post", np.clip(built_post + rng.normal(0, 0.06, field.shape), 0, 1))
        write_handle_raster(domain, sid, "ghsl_built__pre", np.clip(built_pre + rng.normal(0, 0.08, field.shape), 0, 1))
        write_handle_raster(domain, sid, "ghsl_built__post", np.clip(built_post + rng.normal(0, 0.08, field.shape), 0, 1))
        write_handle_raster(domain, sid, "worldcover_built__post", (built_post > 0.5).astype("float32"))
        ndbi_pre = np.clip(-0.05 + 0.35 * built_pre + 0.12 * texture + rng.normal(0, 0.03, field.shape), -1, 1)
        ndbi_post = np.clip(ndbi_pre + 0.48 * gain + rng.normal(0, 0.03, field.shape), -1, 1)
        write_handle_raster(domain, sid, "s2_ndbi__pre", ndbi_pre)
        write_handle_raster(domain, sid, "s2_ndbi__post", ndbi_post)

    elif domain == "drought":
        spi = np.clip(-0.4 - 2.5 * field + rng.normal(0, 0.15, field.shape), -4, 2)
        sm = np.clip(0.32 - 0.22 * field + rng.normal(0, 0.025, field.shape), 0.02, 0.5)
        ndvi = np.clip(0.62 - 0.50 * field + rng.normal(0, 0.04, field.shape), -0.2, 0.9)
        lst = np.clip(2.0 + 9.0 * field + rng.normal(0, 0.7, field.shape), -3, 16)
        write_handle_raster(domain, sid, "chirps_spi__target", spi)
        write_handle_raster(domain, sid, "era5_soil_moisture__target", sm)
        write_handle_raster(domain, sid, "modis_ndvi__target", ndvi)
        write_handle_raster(domain, sid, "viirs_lst__target", lst)

    elif domain == "air_pollution":
        no2 = np.clip(35 + 150 * field + rng.normal(0, 8, field.shape), 0, 260)
        so2 = np.clip(2 + 18 * field + rng.normal(0, 1.2, field.shape), 0, 40)
        co = np.clip(0.025 + 0.055 * field + rng.normal(0, 0.004, field.shape), 0, 0.14)
        aod = np.clip(0.15 + 1.2 * field + rng.normal(0, 0.07, field.shape), 0, 2.4)
        cams = np.clip(no2 * rng.normal(0.92, 0.02) + rng.normal(0, 10, field.shape), 0, 260)
        write_handle_raster(domain, sid, "s5p_no2__peak", no2)
        write_handle_raster(domain, sid, "s5p_so2__peak", so2)
        write_handle_raster(domain, sid, "s5p_co__peak", co)
        write_handle_raster(domain, sid, "modis_aod__peak", aod)
        write_handle_raster(domain, sid, "cams_no2__peak", cams)

    write_tabular_and_osm(row, field, mask)
    write_available_handles(domain, sid, include_derived=False)
    write_json(
        sdir / "quality_report.json",
        {
            "sample_id": sid,
            "domain": domain,
            "all_sources_ok": True,
            "source_mode": BUILD_SOURCE_MODE,
            "live_fetch_blocker": LIVE_FETCH_BLOCKER,
            "source_count": len(SOURCE_HANDLES[domain]),
            "scene_coverage_min": 1.0,
            "nan_pct_max": 0.0,
            "completed_at": utc_now(),
        },
    )


def write_tabular_and_osm(row: Dict[str, Any], field: np.ndarray, mask: np.ndarray) -> None:
    domain = row["domain"]
    sid = row["sample_id"]
    sdir = sample_dir(domain, sid)
    rng = rng_for(domain, sid)
    pop_base = 2 + 45 * np.clip(field + 0.4 * np.roll(field, 10, axis=0), 0, 1)
    if domain == "urban_expansion":
        pop_base *= 2.2
    if domain == "air_pollution":
        pop_base *= 3.0
    if domain == "drought":
        pop_base *= 0.85
    worldpop = np.clip(pop_base + rng.normal(0, 3.0, field.shape), 0, None)
    write_raster(
        sdir / "tabular" / "worldpop.tif",
        worldpop.astype("float32"),
        bounds=geometry_bounds(row["geometry"]),
    )

    categories = [
        "hospitals",
        "schools",
        "fire_stations",
        "police_stations",
        "power_substations",
        "water_treatment",
        "highways",
        "rail_stations",
    ]
    features = []
    for cat in categories:
        n = {"hospitals": 4, "schools": 14, "highways": 7}.get(cat, 5)
        for i in range(n):
            if cat in {"hospitals", "schools"} and i < max(1, n // 3):
                ys, xs = np.where(mask > 0)
                pick = int(rng.integers(0, len(xs))) if len(xs) else 0
                row_i = int(ys[pick]) if len(xs) else int(rng.integers(0, RASTER_SIZE))
                col_i = int(xs[pick]) if len(xs) else int(rng.integers(0, RASTER_SIZE))
            else:
                row_i = int(rng.integers(0, RASTER_SIZE))
                col_i = int(rng.integers(0, RASTER_SIZE))
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "category": cat,
                        "name": f"{cat.replace('_', ' ').title()} {i + 1}",
                        "row": row_i,
                        "col": col_i,
                    },
                    "geometry": {"type": "Point", "coordinates": [float(col_i), float(row_i)]},
                }
            )
    write_json(sdir / "osm" / "pois.geojson", {"type": "FeatureCollection", "features": features})

    if domain == "air_pollution":
        csv_path = sdir / "tabular" / "openaq_stations.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["station_id", "name", "no2_umol_m2", "row", "col"])
            writer.writeheader()
            for i in range(6):
                r = int(rng.integers(0, RASTER_SIZE))
                c = int(rng.integers(0, RASTER_SIZE))
                writer.writerow(
                    {
                        "station_id": f"station_{i + 1:02d}",
                        "name": f"OpenAQ Station {i + 1}",
                        "no2_umol_m2": round(float(35 + 120 * field[r, c]), 3),
                        "row": r,
                        "col": c,
                    }
                )

    if domain in {"drought", "air_pollution"}:
        write_baselines(row, float(np.nanmean(field)))


def write_baselines(row: Dict[str, Any], field_mean: float) -> None:
    domain = row["domain"]
    sid = row["sample_id"]
    target_date = row["target_date"]
    rng = rng_for(domain, sid)
    if domain == "drought":
        series = {
            "modis_ndvi_anomaly": (0.56, 0.06, 0.56 - 0.38 * field_mean),
            "chirps_spi": (0.0, 0.85, -0.4 - 2.4 * field_mean),
            "era5_soil_moisture": (0.31, 0.035, 0.31 - 0.18 * field_mean),
        }
    else:
        series = {
            "sentinel5p_no2": (45.0, 8.0, 35 + 150 * field_mean),
            "cams_no2": (43.0, 9.0, 35 + 138 * field_mean),
            "modis_maiac_aod": (0.28, 0.08, 0.15 + 1.15 * field_mean),
        }
    for modality, (mean, std, target) in series.items():
        path = sample_dir(domain, sid) / "baselines" / f"{modality}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "value", "is_target"])
            writer.writeheader()
            for year in range(2018, 2024):
                writer.writerow(
                    {
                        "date": f"{year}-{target_date[5:]}",
                        "value": round(float(rng.normal(mean, std)), 6),
                        "is_target": "false",
                    }
                )
            writer.writerow({"date": target_date, "value": round(float(target), 6), "is_target": "true"})


def write_available_handles(domain: str, sid: str, include_derived: bool = True) -> None:
    handles = list(SOURCE_HANDLES[domain])
    if include_derived:
        handles.extend(DERIVED_HANDLES[domain])
    baselines = []
    bdir = sample_dir(domain, sid) / "baselines"
    if bdir.exists():
        baselines = sorted(p.stem for p in bdir.glob("*.csv"))
    write_json(
        sample_dir(domain, sid) / "available_handles.json",
        {
            "sample_id": aoi_id(domain, sid),
            "domain": domain,
            "handles": handles,
            "osm_categories": [
                "hospitals",
                "schools",
                "fire_stations",
                "police_stations",
                "power_substations",
                "water_treatment",
                "highways",
                "rail_stations",
            ],
            "baselines_available": baselines,
        },
    )


def source_registry() -> Dict[str, Any]:
    gee_status = detect_gee_credentials()
    items = []
    for domain in DOMAINS:
        for handle in SOURCE_HANDLES[domain] + DERIVED_HANDLES[domain]:
            meta = handle_metadata(domain, handle)
            items.append(
                {
                    "id": f"{domain}:{handle}",
                    "domain": domain,
                    "handle": handle,
                    "source": meta["source"],
                    "modality": meta["modality"],
                    "units": meta["units"],
                    "scale_m": meta["scale_m"],
                    "build_fetch_method": "gee_or_http_required_for_release",
                    "dry_run_method": BUILD_SOURCE_MODE,
                }
            )
    return {
        "version": "v2.0",
        "source_count": len(items),
        "sources": items,
        "live_fetch_status": {
            "mode": BUILD_SOURCE_MODE,
            **gee_status,
        },
    }


def detect_gee_credentials() -> Dict[str, Any]:
    candidates = [
        os.environ.get("OEA_GEE_KEY_FILE", ""),
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        str(DATA_ROOT.parents[1] / "oea_working/gee_key/my-project-1.json"),
        str(DATA_ROOT.parents[1] / "OpenEarthAgent/gee_key/my-project-1.json"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        project_id = payload.get("project_id") or ""
        service_account = payload.get("client_email") or ""
        if project_id and project_id != "-" and service_account and service_account != "-":
            return {
                "gee_credentials_detected": True,
                "gee_project_id": project_id,
                "gee_service_account": service_account,
                "gee_key_file": str(path),
                "blocked": False,
                "real_remote_fetch_complete": False,
                "reason": LIVE_FETCH_BLOCKER,
            }
    return {
        "gee_credentials_detected": False,
        "gee_project_id": None,
        "gee_service_account": None,
        "gee_key_file": None,
        "blocked": True,
        "real_remote_fetch_complete": False,
        "reason": "No usable Earth Engine key was found in OEA_GEE_KEY_FILE, GOOGLE_APPLICATION_CREDENTIALS, or repo gee_key paths.",
    }


def write_registries() -> None:
    gee_status = detect_gee_credentials()
    write_json(DATA_ROOT / "source_registry.json", source_registry())
    write_json(DATA_ROOT / "thresholds_library.json", THRESHOLDS_LIBRARY)
    write_json(DATA_ROOT / "intake_rules.json", INTAKE_RULES)
    write_json(
        DATA_ROOT / "live_source_status.json",
        {
            "mode": BUILD_SOURCE_MODE,
            **gee_status,
            "checked_at": utc_now(),
        },
    )


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    lo, hi = np.nanpercentile(a, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((a - lo) / (hi - lo), 0, 1)
    return (out * 255).astype("uint8")


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim == 2:
        img = Image.fromarray(normalize_to_uint8(rgb), mode="L").convert("RGB")
    else:
        arr = np.asarray(rgb)
        if arr.shape[0] in {3, 4} and arr.shape[1] == RASTER_SIZE:
            arr = np.moveaxis(arr[:3], 0, -1)
        if arr.dtype != np.uint8:
            arr = normalize_to_uint8(arr)
        img = Image.fromarray(arr[:, :, :3], mode="RGB")
    img.resize((512, 512), Image.Resampling.BILINEAR).save(path)


def list_sample_dirs(include_rejected: bool = False) -> List[Tuple[str, str, Path]]:
    rows: List[Tuple[str, str, Path]] = []
    roots = [SAMPLES_ROOT]
    if include_rejected:
        roots.append(REJECTED_ROOT)
    for root in roots:
        if not root.exists():
            continue
        for domain in DOMAINS:
            ddir = root / domain
            if not ddir.exists():
                continue
            for sdir in sorted(p for p in ddir.iterdir() if p.is_dir() and p.name.startswith("sample_")):
                rows.append((domain, sdir.name, sdir))
    return rows


def load_manifest() -> List[Dict[str, Any]]:
    path = DATA_ROOT / "manifest.json"
    if not path.exists():
        return []
    return read_json(path)["aois"]
