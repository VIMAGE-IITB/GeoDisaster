from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROLES = ("ORC", "GA", "PA", "VRA")
REPO_ROOT = Path(__file__).resolve().parents[2]  # code/


def ensure_repo_imports() -> None:
    for path in (str(REPO_ROOT),):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_json_text(value: Any) -> str:
    if isinstance(value, str):
        parsed = extract_first_json(value)
        if parsed is not None:
            value = parsed
        else:
            return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def extract_first_json(text: str) -> Any | None:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(raw[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : idx + 1])
                except Exception:
                    return None
    return None


def final_assistant_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).lower() == "assistant":
            return str(message.get("content", ""))
    raise ValueError("No assistant message found in training row")


def prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("role", "")).lower() == "assistant":
            break
        prompt.append(dict(message))
    if not prompt:
        raise ValueError("No prompt messages found before assistant message")
    return prompt


def role_file_name(role: str) -> str:
    return f"{role.lower()}_train.jsonl"


def role_adapter_dir(adapter_root: Path, role: str) -> Path:
    role_dir = adapter_root / role.lower()
    if role_dir.exists():
        return role_dir
    upper_dir = adapter_root / role.upper()
    if upper_dir.exists():
        return upper_dir
    return role_dir


def resolve_role_adapter(adapter_root: Path, role: str, preferred_subdir: str = "") -> Path:
    role_dir = role_adapter_dir(adapter_root, role)
    if preferred_subdir:
        candidate = role_dir / preferred_subdir
        if candidate.exists():
            return candidate
    return role_dir


def copy_adapter_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("checkpoint-*", "runs", "logs", "*.log")
    shutil.copytree(src, dst, ignore=ignore)


def snapshot_adapter_root(src_root: Path, dst_root: Path, roles: Iterable[str] = ROLES, preferred_subdir: str = "") -> dict[str, str]:
    copied: dict[str, str] = {}
    for role in roles:
        src = resolve_role_adapter(src_root, role, preferred_subdir=preferred_subdir)
        if not src.exists():
            continue
        dst = dst_root / role.lower()
        copy_adapter_tree(src, dst)
        copied[role] = str(dst)
    return copied


def write_weight_manifest(
    *,
    output_path: Path,
    stage: str,
    base_model: str,
    source_adapter_root: str,
    output_adapter_root: str,
    roles: Iterable[str],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "stage": stage,
        "created_at_unix": int(time.time()),
        "base_model": base_model,
        "source_adapter_root": source_adapter_root,
        "output_adapter_root": output_adapter_root,
        "roles": list(roles),
        "adapter_dirs": {role: str(role_adapter_dir(Path(output_adapter_root), role)) for role in roles},
    }
    if extra:
        payload.update(extra)
    write_json(output_path, payload)


def split_rows_by_case(
    rows: list[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 4 or validation_ratio <= 0:
        return rows, []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        key = str(row.get("case_id", row.get("dataset_index", idx)))
        by_case.setdefault(key, []).append(row)
    case_ids = list(by_case)
    random.Random(seed).shuffle(case_ids)
    n_val = max(1, int(round(len(case_ids) * validation_ratio)))
    val_ids = set(case_ids[:n_val])
    train = [row for case_id in case_ids if case_id not in val_ids for row in by_case[case_id]]
    val = [row for case_id in case_ids if case_id in val_ids for row in by_case[case_id]]
    if not train or not val:
        return rows, []
    return train, val


def supports_bf16() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def filtered_config(config_cls: type, values: dict[str, Any]) -> Any:
    params = inspect.signature(config_cls).parameters
    return config_cls(**{key: value for key, value in values.items() if key in params})


def format_preference_rows(rows: list[dict[str, Any]], tokenizer: Any | None = None) -> list[dict[str, Any]]:
    def _completion_with_template(messages: list[dict[str, Any]], assistant_text: str, prompt_text: str) -> str:
        full_text = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": assistant_text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(full_text, str) and full_text.startswith(prompt_text):
            return full_text[len(prompt_text) :]
        return assistant_text + (tokenizer.eos_token or "")

    formatted: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt")
        messages = row.get("prompt_messages") or []
        if not prompt:
            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if tokenizer is not None:
            chosen = _completion_with_template(list(messages), chosen, str(prompt))
            rejected = _completion_with_template(list(messages), rejected, str(prompt))
        formatted.append(
            {
                "prompt": str(prompt),
                "chosen": chosen,
                "rejected": rejected,
                "role": str(row.get("role", "")),
                "case_id": row.get("case_id", row.get("dataset_index", -1)),
                "step_id": str(row.get("step_id", "")),
                "kind": str(row.get("kind", "")),
            }
        )
    return formatted


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    avg = mean(vals)
    return math.sqrt(sum((x - avg) ** 2 for x in vals) / (len(vals) - 1))
