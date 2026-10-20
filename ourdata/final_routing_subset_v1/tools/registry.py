from __future__ import annotations

import traceback
from typing import Callable

from .copernicus_tools import GetCopernicusFloodExtent
from .evaluation_tools import Terminate
from .geotiff_tools import ReadGeoTIFFMetadata
from .metadata_tools import ReadSampleMetadata
from .osm_tools import FetchOSMRoadGraph, FetchOSMShelterCandidates
from .routing_tools import (
    BuildFloodRiskRoadGraph,
    CheckRouteFloodExposure,
    ComputeCandidateRoutes,
    ComputeRouteDelta,
    FilterReachableShelters,
    RankRoutes,
)
from .visualization_tools import AddRouteLabels, RenderRoutesOnRGB


TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    "ReadGeoTIFFMetadata": ReadGeoTIFFMetadata,
    "ReadSampleMetadata": ReadSampleMetadata,
    "GetCopernicusFloodExtent": GetCopernicusFloodExtent,
    "FetchOSMRoadGraph": FetchOSMRoadGraph,
    "FetchOSMShelterCandidates": FetchOSMShelterCandidates,
    "BuildFloodRiskRoadGraph": BuildFloodRiskRoadGraph,
    "ComputeCandidateRoutes": ComputeCandidateRoutes,
    "RankRoutes": RankRoutes,
    "ComputeRouteDelta": ComputeRouteDelta,
    "RenderRoutesOnRGB": RenderRoutesOnRGB,
    "AddRouteLabels": AddRouteLabels,
    "FilterReachableShelters": FilterReachableShelters,
    "CheckRouteFloodExposure": CheckRouteFloodExposure,
    "Terminate": Terminate,
}


def run_tool(name: str, arguments: dict) -> dict:
    if name not in TOOL_REGISTRY:
        return {
            "ok": False,
            "error": {
                "type": "UnknownTool",
                "message": f"Unknown tool: {name}",
            },
        }
    try:
        result = TOOL_REGISTRY[name](**(arguments or {}))
        if not isinstance(result, dict):
            return {
                "ok": False,
                "error": {
                    "type": "InvalidToolReturn",
                    "message": f"{name} returned {type(result).__name__}, expected dict",
                },
            }
        return {"ok": True, "tool": name, "result": result}
    except Exception as exc:
        return {
            "ok": False,
            "tool": name,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=5),
            },
        }

