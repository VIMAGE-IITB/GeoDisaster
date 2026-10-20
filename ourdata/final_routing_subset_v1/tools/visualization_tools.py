from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable


RENDER_CONTRACT = "declared_routing_layers_live_render_v1"
ROUTE_IDS = ("F1", "F2", "F3", "S1", "S2", "S3", "B1", "B2", "B3")


_DECLARED_READ_SCOPE: ContextVar[dict | None] = ContextVar(
    "routing_visual_declared_read_scope", default=None
)


@contextmanager
def DeclaredReadScope(paths):
    scope = {
        "allowed": {str(Path(path).resolve()) for path in paths},
        "reads": [],
    }
    token = _DECLARED_READ_SCOPE.set(scope)
    try:
        yield scope["reads"]
    finally:
        _DECLARED_READ_SCOPE.reset(token)


def _authorize_input(path: Path, operation: str) -> Path:
    resolved = _resolve(path).resolve()
    name = resolved.name.lower()
    parts = {part.lower() for part in resolved.parts}
    if parts & {"gt", "ground_truth", "hidden", "trajectories"} or any(
        marker in name
        for marker in (
            "expected_answer",
            "expected_final_answer",
            "canonical_answer",
            "selected_routes.json",
            "final_answer",
        )
    ):
        raise PermissionError(f"private answer path is not readable by tools: {resolved}")
    scope = _DECLARED_READ_SCOPE.get()
    if scope is not None and str(resolved) not in scope["allowed"]:
        raise PermissionError(f"undeclared sibling access denied ({operation}): {resolved}")
    if scope is not None:
        record = {"path": str(resolved), "operation": operation}
        if record not in scope["reads"]:
            scope["reads"].append(record)
    return resolved


def _root() -> Path:
    return Path.cwd()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _root() / candidate


def _load(path: Path) -> Any:
    path = _authorize_input(path, "json_read")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _geometry_lines(geometry: dict[str, Any] | None) -> Iterable[list[list[float]]]:
    geometry = geometry or {}
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "LineString":
        yield coordinates
    elif kind == "MultiLineString":
        yield from coordinates
    elif kind == "Polygon":
        yield from coordinates
    elif kind == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgb_uint8(dataset):
    import numpy as np

    bands = dataset.read(list(range(1, min(dataset.count, 3) + 1))).astype("float32")
    if bands.shape[0] == 1:
        bands = np.repeat(bands, 3, axis=0)
    if bands.shape[0] == 2:
        bands = np.concatenate([bands, bands[-1:]], axis=0)
    output = np.zeros((dataset.height, dataset.width, 3), dtype="uint8")
    for index in range(3):
        band = bands[index]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        if float(band[finite].min()) >= 0 and float(band[finite].max()) <= 255:
            scaled = np.clip(band, 0, 255)
        else:
            low, high = np.percentile(band[finite], [2, 98])
            scaled = np.clip((band - low) / max(float(high - low), 1e-6) * 255.0, 0, 255)
        scaled[~finite] = 0
        output[:, :, index] = scaled.astype("uint8")
    return output


def _pixel_line(dataset, coordinates: list[list[float]]) -> list[tuple[float, float]]:
    from rasterio.warp import transform

    if len(coordinates) < 2:
        return []
    xs, ys = transform(
        "EPSG:4326",
        dataset.crs,
        [point[0] for point in coordinates],
        [point[1] for point in coordinates],
    )
    inverse = ~dataset.transform
    return [tuple(inverse * (x, y)) for x, y in zip(xs, ys)]


def _intersects_canvas(points: list[tuple[float, float]], width: int, height: int) -> bool:
    if not points:
        return False
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(xs) >= 0 and min(xs) < width and max(ys) >= 0 and min(ys) < height


def _point_pixel(dataset, lon: float, lat: float) -> tuple[float, float]:
    from rasterio.warp import transform

    xs, ys = transform("EPSG:4326", dataset.crs, [lon], [lat])
    return tuple((~dataset.transform) * (xs[0], ys[0]))


def _draw_marker(draw, image_size: tuple[int, int], point: tuple[float, float], color: tuple[int, int, int], shape: str) -> bool:
    x, y = point
    width, height = image_size
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    radius = 7
    if shape == "diamond":
        draw.polygon(
            [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)],
            fill=color,
            outline=(0, 0, 0),
        )
    else:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
    return True


def RenderRoutesOnRGB(
    sample_dir: str,
    rgb_path: str | None = None,
    routes_path: str | None = None,
    roads_path: str | None = None,
    flood_extent_path: str | None = None,
    metadata_path: str | None = None,
    output_path: str | None = None,
    current_route_handle: str | None = None,
    selected_route_ids: dict[str, str] | None = None,
    layers: dict[str, bool] | None = None,
    **_: object,
) -> dict:
    """Render a deterministic route overlay from raw declared local inputs."""
    import numpy as np
    import rasterio
    from PIL import Image, ImageDraw
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom

    sample = _resolve(sample_dir)
    rgb = _resolve(rgb_path or sample / "input/rgb.tif")
    routes = _resolve(routes_path or sample / "derived/candidate_routes.geojson")
    roads = _resolve(roads_path or sample / "derived/roads_with_flood_risk.geojson")
    flood = _resolve(flood_extent_path or sample / "copernicus/flood_extent.geojson")
    metadata = _resolve(metadata_path or sample / "input/metadata.json")
    output = _resolve(output_path or sample / "rendered/routing_live_overlay.png")
    required = {
        "rgb": rgb,
        "routes": routes,
        "roads": roads,
        "flood": flood,
        "metadata": metadata,
    }
    missing = [
        name
        for name, path in required.items()
        if not _authorize_input(path, "exists").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing routing render inputs: {missing}")

    supported_layers = {
        "rgb",
        "roads",
        "flood",
        "blocked_roads",
        "origin",
        "shelter",
        "routes",
        "labels",
    }
    requested_layers = {name: True for name in supported_layers}
    if layers is not None:
        if not isinstance(layers, dict) or set(layers) - supported_layers:
            raise ValueError(
                f"routing layers must contain only {sorted(supported_layers)}"
            )
        non_boolean = {
            name: value for name, value in layers.items() if not isinstance(value, bool)
        }
        if non_boolean:
            raise ValueError(f"routing layer toggles must be boolean: {non_boolean}")
        requested_layers.update(layers)
    if requested_layers["labels"] and not requested_layers["routes"]:
        raise ValueError("route labels require routes=true")

    route_data = _load(routes)
    road_data = _load(roads)
    flood_data = _load(flood)
    metadata_data = _load(metadata)
    route_features = route_data.get("features") or []
    road_features = road_data.get("features") or []
    flood_features = flood_data.get("features") or []
    route_by_id = {
        str((feature.get("properties") or {}).get("route_id")): feature
        for feature in route_features
    }
    if set(route_by_id) != set(ROUTE_IDS):
        raise ValueError(f"expected route IDs {ROUTE_IDS}, found {sorted(route_by_id)}")
    expected_selection_groups = {"fastest", "safest", "balanced"}
    if not isinstance(selected_route_ids, dict) or set(selected_route_ids) != expected_selection_groups:
        raise ValueError(f"incomplete current route selection: {selected_route_ids}")
    for group, route_id in selected_route_ids.items():
        feature = route_by_id.get(str(route_id))
        if feature is None or str((feature.get("properties") or {}).get("route_type")) != group:
            raise ValueError(f"selected {group} route is invalid: {route_id!r}")

    with rasterio.open(_authorize_input(rgb, "raster_read")) as dataset:
        rgb_base = _rgb_uint8(dataset)
        base = rgb_base if requested_layers["rgb"] else np.zeros_like(rgb_base)
        chip_height, chip_width = base.shape[:2]
        flood_shapes = []
        for feature in flood_features if requested_layers["flood"] else []:
            geometry = feature.get("geometry")
            if geometry:
                try:
                    flood_shapes.append(
                        (transform_geom("EPSG:4326", dataset.crs, geometry), 1)
                    )
                except Exception:
                    continue
        flood_mask = (
            rasterize(
                flood_shapes,
                out_shape=(chip_height, chip_width),
                transform=dataset.transform,
                fill=0,
                all_touched=True,
                dtype="uint8",
            )
            if flood_shapes
            else np.zeros((chip_height, chip_width), dtype="uint8")
        )
        rendered = base.copy()
        flooded = flood_mask == 1
        rendered[flooded] = (
            0.48 * rendered[flooded]
            + 0.52 * np.array([0, 190, 255], dtype="float32")
        ).astype("uint8")
        image = Image.fromarray(rendered, mode="RGB")
        draw = ImageDraw.Draw(image)

        road_counts = {
            "source": len(road_features),
            "rendered": 0,
            "risky_source": 0,
            "risky_rendered": 0,
            "blocked_source": 0,
            "blocked_rendered": 0,
        }
        for layer in ("base", "risky", "blocked"):
            if layer in {"base", "risky"} and not requested_layers["roads"]:
                continue
            if layer == "blocked" and not requested_layers["blocked_roads"]:
                continue
            for feature in road_features:
                props = feature.get("properties") or {}
                blocked = _boolish(props.get("blocked"))
                risky = float(props.get("flood_risk") or 0) > 0
                if layer == "base" and (blocked or risky):
                    continue
                if layer == "risky" and (blocked or not risky):
                    continue
                if layer == "blocked" and not blocked:
                    continue
                if layer == "risky":
                    road_counts["risky_source"] += 1
                if layer == "blocked":
                    road_counts["blocked_source"] += 1
                feature_drawn = False
                for coordinates in _geometry_lines(feature.get("geometry")):
                    points = _pixel_line(dataset, coordinates)
                    if _intersects_canvas(points, chip_width, chip_height):
                        color, line_width = {
                            "base": ((235, 235, 235), 1),
                            "risky": ((255, 165, 0), 2),
                            "blocked": ((220, 25, 25), 3),
                        }[layer]
                        draw.line(points, fill=color, width=line_width, joint="curve")
                        feature_drawn = True
                if feature_drawn:
                    road_counts["rendered"] += int(layer == "base")
                    road_counts["risky_rendered"] += int(layer == "risky")
                    road_counts["blocked_rendered"] += int(layer == "blocked")

        route_colors = {
            "fastest": (255, 215, 0),
            "safest": (30, 220, 90),
            "balanced": (220, 60, 220),
        }
        route_drawn: list[str] = []
        label_positions: dict[str, tuple[float, float]] = {}
        for route_id in reversed(ROUTE_IDS) if requested_layers["routes"] else ():
            feature = route_by_id[route_id]
            props = feature.get("properties") or {}
            group = str(props.get("route_type"))
            rank_one = route_id in {"F1", "S1", "B1"}
            for coordinates in _geometry_lines(feature.get("geometry")):
                points = _pixel_line(dataset, coordinates)
                if not _intersects_canvas(points, chip_width, chip_height):
                    continue
                draw.line(
                    points,
                    fill=route_colors[group],
                    width=5 if rank_one else 2,
                    joint="curve",
                )
                route_drawn.append(route_id)
                rank = int(route_id[1])
                position = min(
                    len(points) - 1,
                    max(0, int((0.22 + rank * 0.2) * (len(points) - 1))),
                )
                label_positions[route_id] = points[position]

        label_font = _font(11)
        for route_id in ROUTE_IDS if requested_layers["labels"] else ():
            if route_id not in label_positions:
                continue
            x_position, y_position = label_positions[route_id]
            text_position = (x_position + 3, y_position + 3)
            box = draw.textbbox(text_position, route_id, font=label_font, stroke_width=1)
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text(
                text_position,
                route_id,
                fill=(255, 255, 255),
                font=label_font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )

        start = metadata_data.get("start_point") or {}
        origin_pixel = _point_pixel(dataset, float(start["lon"]), float(start["lat"]))
        origin_drawn = requested_layers["origin"] and _draw_marker(
            draw, (chip_width, chip_height), origin_pixel, (25, 90, 255), "circle"
        )
        fastest_id = str(selected_route_ids["fastest"])
        fastest_coordinates = next(_geometry_lines(route_by_id[fastest_id].get("geometry")))
        shelter_lon, shelter_lat = fastest_coordinates[-1]
        shelter_pixel = _point_pixel(dataset, float(shelter_lon), float(shelter_lat))
        shelter_drawn = requested_layers["shelter"] and _draw_marker(
            draw,
            (chip_width, chip_height),
            shelter_pixel,
            (255, 255, 255),
            "diamond",
        )

    legend_width = 400
    canvas = Image.new("RGB", (chip_width + legend_width, chip_height), (22, 27, 34))
    canvas.paste(image, (0, 0))
    legend = ImageDraw.Draw(canvas)
    title_font = _font(18)
    body_font = _font(13)
    small_font = _font(11)
    x_position, y_position = chip_width + 18, 16
    legend.text(
        (x_position, y_position),
        "Flood-screened route candidates",
        fill=(255, 255, 255),
        font=title_font,
    )
    y_position += 30
    event_label = str(
        metadata_data.get("event_name")
        or metadata_data.get("activation_code")
        or "Routing scene"
    )
    if len(event_label) > 48:
        event_label = event_label[:45] + "..."
    legend.text(
        (x_position, y_position),
        event_label,
        fill=(190, 205, 220),
        font=small_font,
    )
    y_position += 30
    legend.text((x_position, y_position), "Live layers", fill=(255, 255, 255), font=body_font)
    y_position += 21
    legend_items = []
    if requested_layers["flood"]:
        legend_items.append(((0, 190, 255), f"Copernicus flood ({int(flooded.sum()):,} px)"))
    if requested_layers["roads"]:
        legend_items.extend(
            [
                ((235, 235, 235), f"OSM roads ({road_counts['rendered']:,} in chip)"),
                ((255, 165, 0), f"Flood-risk roads ({road_counts['risky_rendered']:,})"),
            ]
        )
    if requested_layers["blocked_roads"]:
        legend_items.append(((220, 25, 25), f"Blocked roads ({road_counts['blocked_rendered']:,})"))
    if requested_layers["origin"]:
        legend_items.append(((25, 90, 255), "Origin"))
    if requested_layers["shelter"]:
        legend_items.append(((255, 255, 255), "Selected shelter"))
    for color, text in legend_items:
        legend.rectangle(
            (x_position, y_position + 3, x_position + 12, y_position + 15),
            fill=color,
            outline=(120, 120, 120),
        )
        legend.text(
            (x_position + 20, y_position), text, fill=(225, 230, 235), font=small_font
        )
        y_position += 20
    y_position += 8
    if requested_layers["routes"]:
        legend.text(
            (x_position, y_position),
            "Candidates (rank one in bold)",
            fill=(255, 255, 255),
            font=body_font,
        )
        y_position += 22
        for route_id in ROUTE_IDS:
            props = route_by_id[route_id].get("properties") or {}
            group = str(props.get("route_type"))
            font = body_font if route_id.endswith("1") else small_font
            legend.line(
                (x_position, y_position + 8, x_position + 24, y_position + 8),
                fill=route_colors[group],
                width=5 if route_id.endswith("1") else 2,
            )
            legend.text(
                (x_position + 32, y_position),
                f"{route_id}  {float(props.get('eta_min') or 0):.1f} min  risk {float(props.get('risk_score') or 0):.2f}",
                fill=(255, 255, 255),
                font=font,
            )
            y_position += 21
    y_position += 8
    legend.text(
        (x_position, y_position),
        "Packaged candidates—not evacuation advice",
        fill=(255, 205, 120),
        font=small_font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False)
    changed_pixels = int(np.any(np.asarray(image) != base, axis=2).sum())
    validation = {
        "contract": RENDER_CONTRACT,
        "prepared_overlay_used": False,
        "current_route_handle": current_route_handle,
        "selected_route_ids": selected_route_ids,
        "requested_layers": requested_layers,
        "rgb_rendered": requested_layers["rgb"],
        "flood_source_features": len(flood_features),
        "flood_pixels_in_chip": int(flooded.sum()),
        "roads": road_counts,
        "route_ids_expected": list(ROUTE_IDS),
        "route_ids_rendered": sorted(set(route_drawn)),
        "route_labels_rendered": sorted(label_positions),
        "origin_rendered": origin_drawn,
        "shelter_rendered": shelter_drawn,
        "changed_chip_pixels": changed_pixels,
        "image_width": canvas.width,
        "image_height": canvas.height,
        "image_bytes": output.stat().st_size,
        "image_sha256": _sha256(output),
    }
    expected_routes = sorted(ROUTE_IDS) if requested_layers["routes"] else []
    expected_labels = sorted(ROUTE_IDS) if requested_layers["labels"] else []
    if validation["route_ids_rendered"] != expected_routes:
        raise ValueError(
            f"not all routes intersect the rendered chip: {validation['route_ids_rendered']}"
        )
    if validation["route_labels_rendered"] != expected_labels:
        raise ValueError("not all route labels were rendered")
    if requested_layers["origin"] and not origin_drawn:
        raise ValueError(f"routing origin render validation failed: {validation}")
    if requested_layers["shelter"] and not shelter_drawn:
        raise ValueError(f"routing shelter render validation failed: {validation}")
    if any(requested_layers[name] for name in supported_layers - {"rgb"}) and changed_pixels <= 0:
        raise ValueError(f"routing overlay did not change the chip: {validation}")
    if output.stat().st_size <= 0:
        raise ValueError(f"routing render validation failed: {validation}")
    manifest_path = output.with_suffix(".render.json")
    manifest = {
        "schema_version": "geodisaster_routing_render_manifest_v1",
        "inputs": {name: str(path) for name, path in required.items()},
        "output": str(output),
        "validation": validation,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "overlay_path": str(output),
        "render_manifest_path": str(manifest_path),
        "exists": True,
        "generated_fallback": False,
        "status": "generated_live_from_declared_layers",
        "render_validation": validation,
    }


def AddRouteLabels(
    overlay_path: str, route_scores: dict | None = None, **_: object
) -> dict:
    path = _resolve(overlay_path)
    labels = []
    if route_scores:
        routes = route_scores.get("routes", route_scores)
        if isinstance(routes, dict):
            for group in routes.values():
                if isinstance(group, list):
                    labels.extend(
                        route.get("route_id")
                        for route in group
                        if route.get("route_id")
                    )
        elif isinstance(routes, list):
            labels.extend(
                route.get("route_id") for route in routes if route.get("route_id")
            )
    labels = sorted(set(label for label in labels if label)) or list(ROUTE_IDS)
    return {
        "labeled_overlay_path": str(path),
        "overlay_exists": path.is_file() and path.stat().st_size > 0,
        "labels_added": labels,
        "status": "labels_rendered_live" if path.exists() else "overlay_missing",
    }
