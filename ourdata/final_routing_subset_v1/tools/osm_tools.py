from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from functools import lru_cache


def _root() -> Path:
    return Path.cwd()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _root() / p


@lru_cache(maxsize=512)
def _count_graphml(path_text: str) -> tuple[int, int]:
    path = Path(path_text)
    node_count = 0
    edge_count = 0
    for _, elem in ET.iterparse(path, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "node":
            node_count += 1
        elif tag == "edge":
            edge_count += 1
        elem.clear()
    return node_count, edge_count


@lru_cache(maxsize=512)
def _load_geojson(path_text: str) -> dict:
    with Path(path_text).open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_xy(coords):
    if isinstance(coords, (list, tuple)) and len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
        yield float(coords[0]), float(coords[1])
    elif isinstance(coords, (list, tuple)):
        for item in coords:
            yield from _iter_xy(item)


def _representative_lat_lon(geometry: dict) -> tuple[float | None, float | None]:
    pts = list(_iter_xy((geometry or {}).get("coordinates")))
    if not pts:
        return None, None
    lon = sum(x for x, _ in pts) / len(pts)
    lat = sum(y for _, y in pts) / len(pts)
    return lat, lon


@lru_cache(maxsize=512)
def _shelter_summary(path_text: str) -> tuple[int, dict[str, int], tuple[dict, ...]]:
    data = _load_geojson(path_text)
    features = data.get("features", [])
    confidence_counts: dict[str, int] = {}
    top = []
    for feature in features:
        props = feature.get("properties") or {}
        conf = str(props.get("confidence") or "unknown")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        if len(top) < 10:
            lat, lon = _representative_lat_lon(feature.get("geometry") or {})
            top.append(
                {
                    "name": str(props.get("name") or "unnamed_candidate"),
                    "confidence": conf,
                    "amenity": None if props.get("amenity") is None else str(props.get("amenity")),
                    "lat": lat,
                    "lon": lon,
                }
            )
    return int(len(features)), confidence_counts, tuple(top)


def FetchOSMRoadGraph(sample_dir: str, network_type: str = "walk", bounds_wgs84=None, **_: object) -> dict:
    graph_path = Path(sample_dir) / "osm" / "road_graph.graphml"
    roads_path = Path(sample_dir) / "osm" / "roads_raw.geojson"
    graph_full = _resolve(graph_path)
    roads_full = _resolve(roads_path)
    if graph_full.exists() or roads_full.exists():
        node_count = None
        if roads_full.exists():
            edge_count = len(_load_geojson(str(roads_full)).get("features", []))
        else:
            node_count, edge_count = _count_graphml(str(graph_full))
        return {
            "road_graph_path": graph_path.as_posix(),
            "roads_raw_path": roads_path.as_posix(),
            "node_count": None if node_count is None else int(node_count),
            "edge_count": int(edge_count),
            "network_type": network_type,
            "bounds_wgs84": bounds_wgs84,
            "status": "loaded_cached_osm_road_graph",
        }
    if roads_full.exists():
        features = _load_geojson(str(roads_full)).get("features", [])
        return {
            "road_graph_path": graph_path.as_posix(),
            "roads_raw_path": roads_path.as_posix(),
            "node_count": None,
            "edge_count": int(len(features)),
            "network_type": network_type,
            "bounds_wgs84": bounds_wgs84,
            "warning": "graphml_missing_used_roads_raw",
            "status": "loaded_cached_osm_roads_only",
        }
    raise FileNotFoundError(f"Missing OSM graph and roads files in {sample_dir}/osm")


def FetchOSMShelterCandidates(sample_dir: str, bounds_wgs84=None, **_: object) -> dict:
    path = Path(sample_dir) / "osm" / "shelter_candidates.geojson"
    full_path = _resolve(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Missing shelter candidates: {path}")
    candidate_count, confidence_counts, top = _shelter_summary(str(full_path))
    return {
        "shelter_candidates_path": path.as_posix(),
        "candidate_count": candidate_count,
        "confidence_counts": confidence_counts,
        "top_candidates": list(top),
        "bounds_wgs84": bounds_wgs84,
        "status": "loaded_cached_shelter_candidates",
    }
