from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache


def _root() -> Path:
    return Path.cwd()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _root() / p


@lru_cache(maxsize=512)
def _load_json(path_text: str) -> dict:
    with Path(path_text).open("r", encoding="utf-8") as f:
        return json.load(f)


def _metadata_for_image(image_path: Path) -> dict:
    try:
        sample_dir = image_path.parents[1]
        metadata_path = sample_dir / "input" / "metadata.json"
        return _load_json(str(metadata_path))
    except Exception:
        return {}


def ReadGeoTIFFMetadata(image_path: str | None = None, path: str | None = None, **_: object) -> dict:
    rel_path = image_path or path
    if not rel_path:
        raise ValueError("ReadGeoTIFFMetadata requires image_path")
    tif_path = _resolve(rel_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"Missing GeoTIFF: {rel_path}")
    metadata = _metadata_for_image(tif_path)
    width, height = metadata.get("chip_size_px", [512, 512])
    bounds_native = metadata.get("bounds_projected")
    transform = None
    if bounds_native and len(bounds_native) == 4:
        left, bottom, right, top = [float(v) for v in bounds_native]
        gsd_x = abs((right - left) / float(width))
        gsd_y = abs((top - bottom) / float(height))
        transform = [gsd_x, 0.0, left, 0.0, -gsd_y, top]
    else:
        gsd_x = float(metadata.get("gsd_m", 10))
        gsd_y = float(metadata.get("gsd_m", 10))
    valid = bool(width == 512 and height == 512 and metadata.get("crs") and transform)
    return {
        "image_path": str(rel_path),
        "crs": metadata.get("crs"),
        "width": int(width),
        "height": int(height),
        "count": 3,
        "bounds_native": bounds_native,
        "bounds_wgs84": metadata.get("bounds_wgs84"),
        "center_wgs84": [metadata.get("center_wgs84", [None, None])[1], metadata.get("center_wgs84", [None, None])[0]],
        "transform": transform,
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "valid": valid,
        "metadata_fallback": True,
    }
