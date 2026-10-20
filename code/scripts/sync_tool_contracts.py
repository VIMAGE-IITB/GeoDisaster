#!/usr/bin/env python3
"""Generate/check public tool registries from the executable worker schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from multi_agent.final_dataset_profile import (  # noqa: E402
    FINAL_DATASET_AGENT_TOOLS,
    FINAL_DATASET_TOOL_SIGNATURES,
    TOOL_TO_AGENT,
)
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import (  # noqa: E402
    TOOL_SPECS,
    agent_tool_spec,
    api_map,
)


REGISTRY_DIR = CODE_ROOT / "tool_registry"


def dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def generated_payloads() -> dict[Path, Any]:
    registry_path = REGISTRY_DIR / "agent_tool_registry.json"
    old_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    old_by_name = {row["name"]: row for row in old_registry.get("tools") or []}
    tools = []
    for name in TOOL_SPECS:
        visible = agent_tool_spec(name)
        previous = old_by_name.get(name, {})
        tools.append(
            {
                "name": name,
                "tags": list(previous.get("tags") or []),
                "signature": FINAL_DATASET_TOOL_SIGNATURES[name].split("  #", 1)[0],
                "description": visible["description"],
                "args": {
                    arg: schema.get("description", "")
                    for arg, schema in visible["properties"].items()
                },
            }
        )
    registry = {"tools": tools, "terminal_action": old_registry.get("terminal_action")}

    api_path = REGISTRY_DIR / "tool_api_map.json"
    old_api = json.loads(api_path.read_text(encoding="utf-8"))
    api = {
        "schema_version": "final_dataset_tool_api_map_v2",
        "controller": old_api.get("controller", "http://127.0.0.1:20101"),
        "worker_base_dir": old_api.get("worker_base_dir", "tool_server/tool_workers/online_workers"),
        "tools": api_map(),
        "notes": [
            "Worker instructions include transport-injected session_id; agents never generate it.",
            "Agent-visible signatures are derived from final_dataset_tool_specs.py.",
            "The evaluator initializes the first ReadData call with the sanitized case and session.",
        ],
    }
    ownership = {
        "agent_tools": FINAL_DATASET_AGENT_TOOLS,
        "tool_to_agent": TOOL_TO_AGENT,
    }
    return {
        registry_path: registry,
        api_path: api,
        REGISTRY_DIR / "tool_ownership.json": ownership,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drift = []
    for path, payload in generated_payloads().items():
        rendered = dump(payload)
        if path.exists() and path.read_text(encoding="utf-8") == rendered:
            continue
        drift.append(str(path))
        if not args.check:
            path.write_text(rendered, encoding="utf-8")
    if drift:
        if args.check:
            raise SystemExit("tool contract drift:\n" + "\n".join(drift))
        print("updated:\n" + "\n".join(drift))
    else:
        print("tool contracts are synchronized")


if __name__ == "__main__":
    main()
