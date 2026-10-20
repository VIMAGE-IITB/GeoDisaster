from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable
from functools import lru_cache


def _root() -> Path:
    return Path.cwd()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _root() / p


def _iter_xy(coords) -> Iterable[tuple[float, float]]:
    if isinstance(coords, (list, tuple)) and len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
        yield float(coords[0]), float(coords[1])
    elif isinstance(coords, (list, tuple)):
        for item in coords:
            yield from _iter_xy(item)


@lru_cache(maxsize=512)
def _load_geojson(path_text: str) -> dict:
    with Path(path_text).open("r", encoding="utf-8") as f:
        return json.load(f)


def _feature_area_sqkm(feature: dict) -> float:
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or []
    rings = coords if geom.get("type") == "Polygon" else [ring for poly in coords for ring in poly]
    area = 0.0
    for ring_group in rings[:1]:
        ring = list(_iter_xy(ring_group))
        if len(ring) < 3:
            continue
        lat0 = sum(y for _, y in ring) / len(ring)
        scale_x = 111_320.0 * math.cos(math.radians(lat0))
        scale_y = 110_540.0
        pts = [(x * scale_x, y * scale_y) for x, y in ring]
        shoelace = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            shoelace += x1 * y2 - x2 * y1
        area += abs(shoelace) / 2.0
    return area / 1_000_000


@lru_cache(maxsize=512)
def _flood_summary(path_text: str) -> tuple[int, list[float] | None, float]:
    data = _load_geojson(path_text)
    features = data.get("features", [])
    xy = [pt for feat in features for pt in _iter_xy((feat.get("geometry") or {}).get("coordinates"))]
    bounds = [min(x for x, _ in xy), min(y for _, y in xy), max(x for x, _ in xy), max(y for _, y in xy)] if xy else None
    area_sqkm = sum(_feature_area_sqkm(feat) for feat in features)
    return int(len(features)), bounds, round(float(area_sqkm), 4)


def GetCopernicusFloodExtent(
    sample_dir: str,
    activation_code: str | None = None,
    aoi: str | dict | None = None,
    product_type: str | None = None,
    **_: object,
) -> dict:
    flood_path = Path(sample_dir) / "copernicus" / "flood_extent.geojson"
    full_path = _resolve(flood_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Missing flood extent: {flood_path}")
    feature_count, bounds, area_sqkm = _flood_summary(str(full_path))
    return {
        "flood_extent_path": flood_path.as_posix(),
        "feature_count": feature_count,
        "bounds_wgs84": bounds,
        "area_sqkm": area_sqkm,
        "activation_code": activation_code,
        "aoi": aoi,
        "product_type": product_type,
        "status": "loaded_cached_flood_extent",
    }
