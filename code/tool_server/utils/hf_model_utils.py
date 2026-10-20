from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple


_SNAPSHOT_RE = re.compile(r"models--([^/]+)--([^/]+)/snapshots/[^/]+/?$")


def snapshot_path_to_repo_id(value: str) -> str | None:
    normalized = str(value).replace("\\", "/")
    match = _SNAPSHOT_RE.search(normalized)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def _all_indexed_weights_exist(index_path: Path) -> bool:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    weight_map = payload.get("weight_map") or {}
    if not weight_map:
        return False

    parent = index_path.parent
    for relative_path in set(weight_map.values()):
        if not (parent / relative_path).exists():
            return False
    return True


def is_complete_pretrained_dir(path: str | Path) -> bool:
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        return False

    # Basic model metadata must exist.
    if not (candidate / "config.json").exists():
        return False

    if (candidate / "model.safetensors").exists() or (candidate / "pytorch_model.bin").exists():
        return True

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = candidate / index_name
        if index_path.exists() and _all_indexed_weights_exist(index_path):
            return True

    return False


def resolve_pretrained_reference(value: str | None) -> Tuple[str | None, bool]:
    if not value:
        return value, False

    candidate = Path(value).expanduser()
    if candidate.exists():
        resolved = str(candidate.resolve())
        if is_complete_pretrained_dir(candidate):
            return resolved, True

        repo_id = snapshot_path_to_repo_id(resolved)
        if repo_id:
            return repo_id, False
        return resolved, True

    repo_id = snapshot_path_to_repo_id(value)
    if repo_id:
        return repo_id, False
    return value, False
