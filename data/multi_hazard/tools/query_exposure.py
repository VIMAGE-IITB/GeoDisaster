from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import rowcol
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry

from ._tool_common import (
    comparison_mask,
    error_response,
    load_handle_array,
    load_pois,
    load_worldpop,
    mask_area_ha,
    parse_sample,
    worldpop_metadata,
)


def _projected_grid(sidecar: Dict[str, Any], shape_: tuple[int, int]):
    profile = sidecar.get("profile") or {}
    transform = profile.get("transform")
    source_crs = CRS.from_user_input(profile.get("crs"))
    if transform is None:
        raise ValueError("footprint_transform_missing")
    height, width = shape_
    rows, cols = np.indices((height, width), dtype="float64")
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    to_wgs84 = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(xs, ys)
    center_lon = float(np.nanmean(lon))
    center_lat = float(np.nanmean(lat))
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", local, always_xy=True)
    px, py = to_local.transform(lon, lat)
    return np.asarray(px), np.asarray(py), to_local


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        domain, sid = parse_sample(args)
        handle = str(args.get("footprint_handle") or args.get("handle"))
        asset_types = args.get("asset_types") or [
            "hospitals",
            "schools",
            "fire_stations",
            "police_stations",
            "power_substations",
            "water_treatment",
            "highways",
            "rail_stations",
        ]
        buffer_m = float(args.get("buffer_m", 0) or 0)
        mask = comparison_mask(domain, sid, handle)
        raw, sidecar = load_handle_array(domain, sid, handle)
        raw = np.asarray(raw, dtype="float64")
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        valid = np.isfinite(raw)
        if not np.any(valid):
            raise ValueError(f"no_valid_pixels:{handle}")
        unique = set(np.unique(raw[valid]).tolist())
        if not unique <= {0.0, 1.0}:
            raise ValueError(f"impact_footprint_not_binary:{handle}:{sorted(unique)}")
        pop, pop_profile = load_worldpop(domain, sid)
        pop_meta = worldpop_metadata(domain, sid)
        population_source = str(pop_meta["source_label"])
        if pop.shape != mask.shape:
            raise ValueError(f"population_grid_shape_mismatch:{pop.shape}:{mask.shape}")
        if str(pop_profile.get("crs")) != str((sidecar.get("profile") or {}).get("crs")):
            raise ValueError("population_grid_crs_mismatch")
        footprint_profile = sidecar.get("profile") or {}
        if pop_profile.get("transform") != footprint_profile.get("transform"):
            raise ValueError("population_grid_transform_mismatch")
        if np.any(pop < 0):
            raise ValueError("population_grid_contains_negative_values")
        grid_x, grid_y, to_local = _projected_grid(sidecar, tuple(mask.shape))
        mask_points = np.column_stack((grid_x[mask], grid_y[mask]))
        tree = cKDTree(mask_points) if len(mask_points) else None
        footprint_pop = float(np.sum(pop[mask]))
        if buffer_m > 0 and tree is not None:
            grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            distances, _ = tree.query(grid_points, k=1)
            buffered = distances.reshape(mask.shape) <= buffer_m
        else:
            buffered = mask.copy()
        pois = load_pois(domain, sid)
        source_crs = CRS.from_user_input(footprint_profile.get("crs"))
        from_wgs84 = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
        footprint_transform = footprint_profile.get("transform")
        critical: Dict[str, Dict[str, Any]] = {}
        for asset in asset_types:
            feats = [f for f in pois if f.get("properties", {}).get("category") == asset]
            in_footprint = 0
            in_buffer = 0
            nearest = {"name": None, "distance_m": None}
            best = float("inf")
            for feat in feats:
                prop = feat["properties"]
                geometry = shape(feat["geometry"])
                representative = geometry.representative_point()
                if representative.is_empty or not np.isfinite([representative.x, representative.y]).all():
                    raise ValueError("invalid_cached_feature_geometry")
                projected = transform_geometry(to_local.transform, representative)
                source_x, source_y = from_wgs84.transform(representative.x, representative.y)
                row, col = rowcol(footprint_transform, source_x, source_y)
                row, col = int(row), int(col)
                on_grid = 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]
                inside = bool(mask[row, col]) if on_grid else False
                distance = float(tree.query([[projected.x, projected.y]], k=1)[0][0]) if tree is not None else float("inf")
                inside_buffer = distance <= buffer_m if buffer_m > 0 else inside
                in_footprint += int(inside)
                in_buffer += int(inside_buffer)
                if distance < best:
                    best = distance
                    nearest = {
                        "name": prop["name"],
                        "distance_m": round(distance, 3),
                    }
            critical[asset] = {
                "in_footprint": in_footprint,
                "in_buffer": in_buffer,
                "nearest": nearest,
            }
        return {
            "aoi_id": f"{domain}__{sid}",
            "footprint_handle": handle,
            "footprint_area_ha": round(mask_area_ha(mask, sidecar), 3),
            "population_within_footprint": round(footprint_pop, 3),
            "population_within_buffer": round(float(np.sum(pop[buffered])), 3),
            "population_source": population_source,
            "distance_method": "local_azimuthal_equidistant_cell_center_v1",
            "asset_geometry_contract": "cached_feature_representative_point_v1",
            "critical_infrastructure": critical,
        }
    except Exception as exc:
        return error_response(exc)
