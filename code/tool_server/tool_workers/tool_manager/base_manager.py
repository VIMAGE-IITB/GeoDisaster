"""Synchronous client for controller-discovered GeoDisaster workers.

The controller is responsible for worker registration and discovery. Actual
tool calls use each worker's JSON ``/worker_generate`` endpoint; the legacy
streaming proxy is not required for deterministic GeoDisaster tools.
"""

from __future__ import annotations

from typing import Any

import requests


class ToolManager:
    def __init__(self, controller_addr: str, timeout: float = 120.0) -> None:
        self.controller_addr = controller_addr.rstrip("/")
        self.timeout = float(timeout)

    def _post(self, url: str, payload: dict[str, Any] | None = None) -> requests.Response:
        response = requests.post(url, json=payload or {}, timeout=self.timeout)
        response.raise_for_status()
        return response

    def list_models(self) -> list[str]:
        payload = self._post(f"{self.controller_addr}/list_models").json()
        return list(payload.get("models") or [])

    def worker_address(self, tool_name: str) -> str:
        payload = self._post(
            f"{self.controller_addr}/get_worker_address", {"model": tool_name}
        ).json()
        address = str(payload.get("address") or "").rstrip("/")
        if not address:
            raise RuntimeError(f"no registered worker for tool {tool_name}")
        return address

    def get_tool_instruction(self, tool_name: str) -> dict[str, Any]:
        address = self.worker_address(tool_name)
        payload = self._post(f"{address}/tool_instruction").json()
        if int(payload.get("error_code", 1)) != 0:
            raise RuntimeError(payload.get("text") or f"instruction lookup failed for {tool_name}")
        return dict(payload.get("tool_instruction") or {})

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        address = self.worker_address(tool_name)
        payload = self._post(f"{address}/worker_generate", dict(args or {})).json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"worker {tool_name} returned a non-object response")
        return payload
