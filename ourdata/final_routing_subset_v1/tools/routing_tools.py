from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


ROUTING_RANKING_CONTRACT = {
    "version": "packaged_composite_edge_cost_v2",
    "risk_score": "length-weighted mean edge flood_risk",
    "fastest_edge_cost": "travel_time_min",
    "safest_edge_cost": "travel_time_min * (1 + 10 * flood_risk)",
    "balanced_edge_cost": "travel_time_min * (1 + 4 * flood_risk)",
    "route_objective": "sum the corresponding edge cost",
    "eligibility": "blocked_edges == 0",
    "tie_break": ["eta_min", "distance_m", "route_id"],
}


_DECLARED_READ_SCOPE: ContextVar[dict | None] = ContextVar(
    "routing_declared_read_scope", default=None
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


def _authorize_read(path: str | Path, operation: str = "read") -> Path:
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


def _exists_for_read(path: str | Path) -> bool:
    return _authorize_read(path, "exists").exists()


def _root() -> Path:
    return Path.cwd()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _root() / p


def _load_json_cached(path_text: str) -> Any:
    with Path(path_text).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json(path: Path) -> Any:
    return _load_json_cached(str(_authorize_read(path, "json_read")))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _routes_list_from_scores(sample_dir: str) -> list[dict]:
    scores_path = _resolve(Path(sample_dir) / "derived" / "route_scores.json")
    if _exists_for_read(scores_path):
        data = _load_json(scores_path)
        routes = data.get("routes", data if isinstance(data, list) else [])
        return [dict(r) for r in routes]
    routes_path = _resolve(Path(sample_dir) / "derived" / "candidate_routes.geojson")
    if not _exists_for_read(routes_path):
        raise FileNotFoundError(f"Missing route scores and candidate routes for {sample_dir}")
    data = _load_json(routes_path)
    return [dict(feature.get("properties") or {}) for feature in data.get("features", [])]


def _group_routes(routes: list[dict]) -> dict[str, list[dict]]:
    grouped = {"fastest": [], "safest": [], "balanced": []}
    for route in routes:
        rtype = str(route.get("route_type"))
        if rtype in grouped:
            grouped[rtype].append(route)
    for group in grouped.values():
        group.sort(key=lambda r: str(r.get("route_id", "")))
    return grouped


def _risk_summary(path_text: str) -> tuple[int, int, int, float, float]:
    data = _load_json(Path(path_text))
    features = data.get("features", [])
    risks = [_num((feat.get("properties") or {}).get("flood_risk")) for feat in features]
    blocked = [_boolish((feat.get("properties") or {}).get("blocked")) for feat in features]
    return (
        int(len(features)),
        int(sum(blocked)),
        int(sum(v > 0 for v in risks)),
        round(float(sum(risks) / len(risks)), 4) if risks else 0.0,
        round(float(max(risks)), 4) if risks else 0.0,
    )


def _grouped_routes_for_sample(sample_dir: str) -> dict[str, list[dict]]:
    return _group_routes(_routes_list_from_scores(sample_dir))


def _road_edge_index(
    road_features: list[dict],
) -> dict[tuple[float, float], list[tuple[list[tuple[float, float]], dict]]]:
    starts: dict[tuple[float, float], list[tuple[list[tuple[float, float]], dict]]] = {}
    for feature in road_features:
        geometry = feature.get("geometry") or {}
        coordinates = [tuple(point) for point in geometry.get("coordinates") or []]
        if len(coordinates) < 2:
            continue
        starts.setdefault(coordinates[0], []).append((coordinates, feature))
        starts.setdefault(coordinates[-1], []).append((list(reversed(coordinates)), feature))
    return starts


def _route_edge_properties(
    route_coordinates: list[list[float]],
    starts: dict[tuple[float, float], list[tuple[list[tuple[float, float]], dict]]],
) -> list[dict]:
    """Decompose one packaged route into its exact OSM-derived edge records."""
    route = [tuple(point) for point in route_coordinates]
    matches: list[dict] = []
    position = 0
    while position < len(route) - 1:
        candidates = [
            (len(coordinates), feature)
            for coordinates, feature in starts.get(route[position], [])
            if route[position : position + len(coordinates)] == coordinates
        ]
        if not candidates:
            raise ValueError(f"candidate route cannot be matched to road edges at vertex {position}")
        length, feature = max(candidates, key=lambda item: item[0])
        matches.append(dict(feature.get("properties") or {}))
        position += length - 1
    return matches


def _assert_close(label: str, observed: Any, expected: float, tolerance: float) -> None:
    value = _num(observed, float("nan"))
    if not (abs(value - expected) <= tolerance):
        raise ValueError(f"{label} mismatch: stored={observed!r}, recomputed={expected!r}")


def ValidateAndRankCandidateRoutes(sample_dir: str, **_: object) -> dict:
    """Recompute the public route metrics/costs and derive the three winners.

    The function reads only the declared score, geometry, and flood-risk-road
    products.  It is intentionally independent of the historical private
    ``ground_truth/selected_routes.json`` cache.
    """
    sample = Path(sample_dir)
    scores_path = _resolve(sample / "derived" / "route_scores.json")
    routes_path = _resolve(sample / "derived" / "candidate_routes.geojson")
    roads_path = _resolve(sample / "derived" / "roads_with_flood_risk.geojson")
    scores_payload = _load_json(scores_path)
    routes_payload = _load_json(routes_path)
    roads_payload = _load_json(roads_path)
    routes = [dict(route) for route in (scores_payload.get("routes") or [])]
    score_by_id = {str(route.get("route_id")): route for route in routes}
    geometry_by_id = {
        str((feature.get("properties") or {}).get("route_id")): feature
        for feature in routes_payload.get("features") or []
    }
    expected_ids = {f"{prefix}{rank}" for prefix in "FSB" for rank in (1, 2, 3)}
    if set(score_by_id) != expected_ids or set(geometry_by_id) != expected_ids:
        raise ValueError(
            f"candidate ID contract failed: scores={sorted(score_by_id)}, "
            f"geometries={sorted(geometry_by_id)}"
        )
    road_features = roads_payload.get("features") or []
    road_index = _road_edge_index(road_features)
    audited: list[dict] = []
    for route_id in sorted(expected_ids):
        stored = score_by_id[route_id]
        feature = geometry_by_id[route_id]
        if (feature.get("properties") or {}) != stored:
            raise ValueError(f"score/geometry properties differ for {route_id}")
        edges = _route_edge_properties(
            (feature.get("geometry") or {}).get("coordinates") or [], road_index
        )
        distance = sum(_num(edge.get("length_m"), _num(edge.get("length"))) for edge in edges)
        eta = sum(_num(edge.get("travel_time_min")) for edge in edges)
        blocked = sum(_boolish(edge.get("blocked")) for edge in edges)
        risk = (
            sum(
                _num(edge.get("flood_risk"))
                * _num(edge.get("length_m"), _num(edge.get("length")))
                for edge in edges
            )
            / distance
            if distance
            else 0.0
        )
        route_type = str(stored.get("route_type") or "")
        if route_type not in {"fastest", "safest", "balanced"}:
            raise ValueError(f"unsupported route_type for {route_id}: {route_type!r}")
        objective_cost = sum(_num(edge.get(f"{route_type}_weight")) for edge in edges)
        _assert_close(f"{route_id}.distance_m", stored.get("distance_m"), distance, 0.011)
        _assert_close(f"{route_id}.distance_km", stored.get("distance_km"), distance / 1000, 0.00051)
        _assert_close(f"{route_id}.eta_min", stored.get("eta_min"), eta, 0.011)
        _assert_close(f"{route_id}.risk_score", stored.get("risk_score"), risk, 0.000051)
        _assert_close(
            f"{route_id}.objective_cost", stored.get("objective_cost"), objective_cost, 0.000051
        )
        if int(stored.get("node_count")) != len(edges) + 1:
            raise ValueError(f"{route_id}.node_count does not match reconstructed edges")
        if "blocked_edges" not in stored or int(stored["blocked_edges"]) != blocked:
            raise ValueError(f"{route_id}.blocked_edges does not match reconstructed edges")
        audited.append(dict(stored))

    grouped = _group_routes(audited)
    best: dict[str, dict] = {}
    for group_name, candidates in grouped.items():
        eligible = [route for route in candidates if int(route["blocked_edges"]) == 0]
        if not eligible:
            raise ValueError(f"no unblocked {group_name} candidate exists")
        best[group_name] = min(
            eligible,
            key=lambda route: (
                _num(route.get("objective_cost"), float("inf")),
                _num(route.get("eta_min"), float("inf")),
                _num(route.get("distance_m"), float("inf")),
                str(route.get("route_id") or ""),
            ),
        )
    return {
        "contract": ROUTING_RANKING_CONTRACT,
        "routes": grouped,
        "best": best,
        "validated_route_count": len(audited),
        "score_path": str(scores_path),
        "geometry_path": str(routes_path),
        "road_path": str(roads_path),
        "status": "validated_and_ranked_from_declared_edges",
    }


def BuildFloodRiskRoadGraph(sample_dir: str, road_buffer_m: float = 8, risk_buffer_m: float = 120, **_: object) -> dict:
    path = Path(sample_dir) / "derived" / "roads_with_flood_risk.geojson"
    full_path = _resolve(path)
    if not _exists_for_read(full_path):
        raise FileNotFoundError(f"Missing flood-risk graph: {path}")
    edge_count, blocked_edges, risky_edges, mean_risk, max_risk = _risk_summary(str(full_path))
    return {
        "roads_with_flood_risk_path": path.as_posix(),
        "edge_count": edge_count,
        "blocked_edges": blocked_edges,
        "risky_edges": risky_edges,
        "mean_risk": mean_risk,
        "max_risk": max_risk,
        "road_buffer_m": road_buffer_m,
        "risk_buffer_m": risk_buffer_m,
        "status": "loaded_cached_flood_risk_graph",
    }


def ComputeCandidateRoutes(
    sample_dir: str,
    origin: dict | None = None,
    k: int = 3,
    strategies: list[str] | None = None,
    **_: object,
) -> dict:
    strategies = strategies or ["fastest", "safest", "balanced"]
    grouped = _grouped_routes_for_sample(sample_dir)
    grouped = {key: grouped.get(key, [])[:k] for key in strategies}
    return {
        "candidate_routes_path": (Path(sample_dir) / "derived" / "candidate_routes.geojson").as_posix(),
        "route_scores_path": (Path(sample_dir) / "derived" / "route_scores.json").as_posix(),
        "route_count": int(sum(len(v) for v in grouped.values())),
        "routes": grouped,
        "origin": origin,
        "k": k,
        "strategies": strategies,
        "status": "loaded_cached_candidate_routes",
    }


def RankRoutes(sample_dir: str, routes: dict | None = None, **_: object) -> dict:
    validation = ValidateAndRankCandidateRoutes(sample_dir)
    selected = validation["best"]
    fastest = selected["fastest"]
    safest = selected["safest"]
    balanced = selected["balanced"]
    delta = ComputeRouteDelta(fastest, safest, balanced)
    return {
        "nearest_reachable_shelter": {
            "candidate_id": fastest.get("shelter_candidate_id"),
            "name": fastest.get("selected_shelter_name"),
            "confidence": fastest.get("shelter_confidence"),
        },
        "fastest_route": fastest,
        "safest_route": safest,
        "balanced_route": balanced,
        "route_delta": delta,
        "recommended_route": None,
        "recommended_route_id": None,
        "explanation": delta["summary"],
        "status": "ranked_routes",
        "source": "declared_edge_recomputed_composite_costs",
        "selection_contract": "packaged_composite_edge_cost_v2",
        "ranking_contract": validation["contract"],
        "validated_route_count": validation["validated_route_count"],
        "graph_search_performed": False,
    }


def ComputeRouteDelta(fastest_route: dict, safest_route: dict, balanced_route: dict, **_: object) -> dict:
    fastest_distance = _num(fastest_route.get("distance_m"))
    fastest_eta = _num(fastest_route.get("eta_min"))
    fastest_risk = _num(fastest_route.get("risk_score"))
    safest_extra_distance = _num(safest_route.get("distance_m")) - fastest_distance
    safest_extra_eta = _num(safest_route.get("eta_min")) - fastest_eta
    safest_risk_reduction = fastest_risk - _num(safest_route.get("risk_score"))
    balanced_extra_distance = _num(balanced_route.get("distance_m")) - fastest_distance
    balanced_extra_eta = _num(balanced_route.get("eta_min")) - fastest_eta
    balanced_risk_reduction = fastest_risk - _num(balanced_route.get("risk_score"))
    return {
        "safest_extra_distance_m": round(safest_extra_distance, 2),
        "safest_extra_eta_min": round(safest_extra_eta, 2),
        "safest_risk_reduction": round(safest_risk_reduction, 4),
        "risk_reduction": round(safest_risk_reduction, 4),
        "balanced_extra_distance_m": round(balanced_extra_distance, 2),
        "balanced_extra_eta_min": round(balanced_extra_eta, 2),
        "balanced_risk_reduction": round(balanced_risk_reduction, 4),
        "summary": (
            f"Compared with {fastest_route.get('route_id')}, the safest-weighted route adds {safest_extra_distance:.1f} m "
            f"and {safest_extra_eta:.1f} min while changing risk by {safest_risk_reduction:.2f}; "
            f"balanced adds {balanced_extra_distance:.1f} m and {balanced_extra_eta:.1f} min while "
            f"changing risk by {balanced_risk_reduction:.2f}."
        ),
    }


def CheckRouteFloodExposure(route: dict, **_: object) -> dict:
    blocked_edges = int(_num(route.get("blocked_edges")))
    risk_score = _num(route.get("risk_score"))
    return {
        "route_id": route.get("route_id"),
        "blocked_edges": blocked_edges,
        "risk_score": risk_score,
        "is_flood_safe": bool(blocked_edges == 0),
    }


def FilterReachableShelters(sample_dir: str, max_eta_min: float = 15, **_: object) -> dict:
    routes = _routes_list_from_scores(sample_dir)
    reachable = []
    seen = set()
    for route in routes:
        eta = _num(route.get("eta_min"))
        shelter_id = route.get("shelter_candidate_id") or route.get("selected_shelter_name")
        if eta <= max_eta_min and shelter_id not in seen:
            seen.add(shelter_id)
            reachable.append(
                {
                    "candidate_id": route.get("shelter_candidate_id"),
                    "name": route.get("selected_shelter_name"),
                    "confidence": route.get("shelter_confidence"),
                    "best_route_id": route.get("route_id"),
                    "eta_min": eta,
                    "distance_m": _num(route.get("distance_m")),
                }
            )
    return {"reachable_shelters": reachable, "unreachable_count": 0 if reachable else 1, "threshold_eta_min": max_eta_min}
