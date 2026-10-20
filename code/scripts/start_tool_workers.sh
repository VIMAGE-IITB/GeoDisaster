#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"  # code/
WORKER_ROOT="$ROOT/tool_server/tool_workers"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/tool_server_config.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT/runs/final_tool_server_logs}"
CONTROLLER_PORT="${CONTROLLER_PORT:-20101}"
START_PORT="${START_PORT:-20102}"

# The public launcher always starts real computation workers. Historical EO
# fixture variables must not leak from a caller's shell into this service.
unset MAS_SCRIPTED_TOOL_ENABLE MAS_SCRIPTED_TOOL_DATA_PATH MAS_SCRIPTED_TOOL_CASE_LIMIT

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON_BIN" - "$ROOT" "$WORKER_ROOT" "$CONFIG_PATH" "$LOG_DIR" "$CONTROLLER_PORT" "$START_PORT" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
worker_root = Path(sys.argv[2])
config_path = Path(sys.argv[3])
log_dir = Path(sys.argv[4])
controller_port = int(sys.argv[5])
start_port = int(sys.argv[6])

sys.path.insert(0, str(root))
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import TOOL_SPECS

tool_worker_config = []
for offset, tool_name in enumerate(TOOL_SPECS):
    port = start_port + offset
    tool_worker_config.append({
        tool_name: {
            "worker_name": tool_name,
            "job_name": f"final_{tool_name}",
            "calculate_type": "control",
            "cmd": {
                "script-addr": f"./online_workers/{tool_name}_worker.py",
                "port": port,
                "host": "0.0.0.0",
            },
        }
    })

payload = {
    "base_dir": str(worker_root),
    "log_folder": str(log_dir),
    "tools_output_dir": str(root / "runs" / "final_tool_outputs"),
    "default_calculate_gpus": 0,
    "default_calculate_cpus": 4,
    "default_control_cpus": 2,
    "default_control_gpus": 0,
    "retry_interval": 1,
    "request_timeout": 10,
    "controller_ready_attempts": 120,
    "controller_config": {
        "worker_name": "final_dataset_controller",
        "job_name": "final_dataset_controller",
        "calculate_type": "control",
        "cmd": {
            "script-addr": "./online_workers/controller.py",
            "port": controller_port,
            "host": "0.0.0.0",
        },
    },
    "model_worker_config": [],
    "tool_worker_config": tool_worker_config,
}
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print(config_path)
PY

PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON_BIN" \
  "$ROOT/tool_server/tool_workers/scripts/launch_scripts/start_server_local.py" \
  --config "$CONFIG_PATH"
