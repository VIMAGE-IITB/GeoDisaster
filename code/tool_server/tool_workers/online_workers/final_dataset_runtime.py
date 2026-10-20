"""Deprecated compatibility shim for the final-dataset workers.

The final-dataset tools now keep their processing logic in the corresponding
``<Tool>_worker.py`` files. This module intentionally contains no tool-specific
runtime implementation.
"""

from __future__ import annotations

from typing import Any


def run_tool(tool_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    raise RuntimeError(
        "final_dataset_runtime.run_tool is deprecated. Launch or import the "
        f"corresponding {tool_name}_worker.py so the tool's own process function "
        "handles the request."
    )
