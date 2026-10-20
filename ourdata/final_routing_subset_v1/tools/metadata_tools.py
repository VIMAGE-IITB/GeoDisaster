from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache


def _root() -> Path:
    return Path.cwd()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _root() / p


def _sample_id_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("sample_"):
            return part
    return None


@lru_cache(maxsize=512)
def _load_json(path_text: str) -> dict:
    with Path(path_text).open("r", encoding="utf-8") as f:
        return json.load(f)


def ReadSampleMetadata(metadata_path: str | None = None, path: str | None = None, **_: object) -> dict:
    rel_path = metadata_path or path
    if not rel_path:
        raise ValueError("ReadSampleMetadata requires metadata_path")
    metadata_file = _resolve(rel_path)
    raw = _load_json(str(metadata_file))

    sample_id = raw.get("sample_id") or _sample_id_from_path(metadata_file) or "unknown_sample"
    aoi = raw.get("aoi")
    if isinstance(aoi, dict):
        aoi_out = aoi
    else:
        aoi_out = {"name": aoi, "number": raw.get("aoi_number")}

    rgb_path = raw.get("rgb_tif_path") or raw.get("rgb.tif path") or f"samples/{sample_id}/input/rgb.tif"
    if str(rgb_path).startswith(".tmp/"):
        rgb_path = f"samples/{sample_id}/input/rgb.tif"

    return {
        "sample_id": sample_id,
        "activation_code": raw.get("activation_code"),
        "event_name": raw.get("event_name"),
        "event_date": raw.get("event_date") or raw.get("event_time"),
        "product_type": raw.get("product_type"),
        "aoi": aoi_out,
        "start_point": raw.get("start_point") or raw.get("origin") or {},
        "network_type": raw.get("network_type", "walk"),
        "rgb_path": rgb_path,
        "bounds_wgs84": raw.get("bounds_wgs84"),
        "chip_size_px": raw.get("chip_size_px", [512, 512]),
        "gsd_m": raw.get("gsd_m", 10),
        "crs": raw.get("crs"),
        "selected_shelter": raw.get("selected_shelter", {}),
        "source_metadata_path": str(rel_path),
    }
