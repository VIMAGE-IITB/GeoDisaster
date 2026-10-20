from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, TypedDict

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from torch.utils.data import WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _patch_transformers_remote_code_compat() -> None:
    """Provide small compatibility shims expected by some HF remote-code models.

    InternLM3's cached modeling file imports ``LossKwargs`` from
    ``transformers.utils``. Some local transformers builds used in this training
    env do not export that symbol, although the model only needs it as a
    TypedDict base for ``**kwargs`` annotations. Adding the symbol before
    ``from_pretrained(..., trust_remote_code=True)`` keeps the model import
    working without changing training behavior.
    """

    import transformers.utils as tf_utils

    if not hasattr(tf_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: Any

        tf_utils.LossKwargs = LossKwargs


_patch_transformers_remote_code_compat()


def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if _supports_bf16() else torch.float16,
        "low_cpu_mem_usage": True,
    }
    if os.environ.get("GEODISASTER_LOAD_DIRECT_TO_CUDA", "1") != "0" and torch.cuda.is_available():
        # The shared node can be CPU-RAM/swap constrained while GPUs are idle.
        # Stream weights straight to the single visible CUDA device to avoid
        # four parallel 7B/8B model loads fighting through host memory first.
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    attn_impl = os.environ.get("GEODISASTER_ATTN_IMPLEMENTATION", "").strip()
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    return kwargs


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _canonical_validation_groups(core_jsonl: Path) -> dict[str, str]:
    """Map case IDs to connected source/trajectory groups.

    Two cases stay together when either ``split_group_id`` or
    ``trajectory_content_group_id`` matches. Taking connected components also
    handles the uncommon case where those two equivalence relations overlap.
    """

    core_rows = _load_jsonl(core_jsonl)
    ids = [str(row.get("id") or row.get("dataset_id") or "") for row in core_rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Canonical core has missing or duplicate IDs: {core_jsonl}")
    groups = _DisjointSet(ids)
    first_by_key: dict[tuple[str, str], str] = {}
    for case_id, row in zip(ids, core_rows, strict=True):
        for field in ("split_group_id", "trajectory_content_group_id"):
            value = str(row.get(field) or "").strip()
            if not value:
                continue
            key = (field, value)
            first = first_by_key.setdefault(key, case_id)
            groups.union(first, case_id)
    return {case_id: groups.find(case_id) for case_id in ids}


def _discover_core_jsonl(train_file: Path, explicit: str = "") -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Canonical core JSONL not found: {path}")
        return path
    env_path = os.environ.get("GEODISASTER_CORE_JSONL", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"GEODISASTER_CORE_JSONL does not exist: {path}")
        return path
    dataset_root = os.environ.get("GEODISASTER_DATASET_ROOT", "").strip()
    candidates = []
    if dataset_root:
        candidates.append(
            Path(dataset_root).expanduser()
            / "data/core/geodisaster_2921_source_with_splits.jsonl"
        )
    candidates.extend(
        parent / "data/core/geodisaster_2921_source_with_splits.jsonl"
        for parent in (train_file.parent, *train_file.parents)
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _row_case_id(row: dict[str, Any], fallback: Any = "") -> str:
    return str(row.get("dataset_id") or row.get("id") or row.get("case_id") or fallback)


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    validation_groups: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 4 or val_ratio <= 0:
        return rows, []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        case_id = _row_case_id(row, len(by_case))
        by_case.setdefault(case_id, []).append(row)
    if validation_groups is not None:
        missing = sorted(case_id for case_id in by_case if case_id not in validation_groups)
        if missing:
            raise ValueError(
                "SFT rows are missing from the canonical validation-group map: "
                + json.dumps(missing[:20])
            )
    by_group: dict[str, list[str]] = {}
    for case_id in sorted(by_case):
        group_id = validation_groups[case_id] if validation_groups is not None else case_id
        by_group.setdefault(group_id, []).append(case_id)
    group_ids = sorted(by_group)
    random.Random(seed).shuffle(group_ids)
    target_val_cases = max(1, int(round(len(by_case) * val_ratio)))
    val_case_ids: set[str] = set()
    for group_id in group_ids:
        if val_case_ids and len(val_case_ids) >= target_val_cases:
            break
        val_case_ids.update(by_group[group_id])
    ordered_case_ids = [case_id for group_id in group_ids for case_id in by_group[group_id]]
    train_rows = [row for cid in ordered_case_ids if cid not in val_case_ids for row in by_case[cid]]
    val_rows = [row for cid in ordered_case_ids if cid in val_case_ids for row in by_case[cid]]
    if not train_rows or not val_rows:
        return rows, []
    return train_rows, val_rows


def _format_rows(rows: list[dict[str, Any]], tokenizer: Any | None = None) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        messages = row["messages"]
        assistant_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                assistant_idx = idx
                break
        if assistant_idx is None:
            raise ValueError(f"Row has no assistant message: case_id={row.get('case_id')}")
        prompt = messages[:assistant_idx]
        completion = [messages[assistant_idx]]
        if tokenizer is not None:
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            assistant_text = str(messages[assistant_idx].get("content", ""))
            # Derive the assistant completion with the model's own chat template.
            # This keeps Qwen, Llama, Mistral, InternLM, etc. from learning a
            # foreign end-of-turn token while preserving completion-only SFT.
            full_text = tokenizer.apply_chat_template(
                messages[: assistant_idx + 1],
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(full_text, str) and full_text.startswith(prompt_text):
                completion = full_text[len(prompt_text) :]
            else:
                completion = assistant_text + (tokenizer.eos_token or "")
            prompt = prompt_text
        formatted.append(
            {
                "prompt": prompt,
                "completion": completion,
                # Keep metadata columns as strings so mixed EO integer ids and
                # GeoDisaster semantic ids do not make pyarrow infer conflicting
                # column types when building a Dataset from combined SFT rows.
                "role": str(row.get("role", "")),
                "case_id": str(row.get("case_id", "")),
                "step_id": str(row.get("step_id", "")),
                "kind": str(row.get("kind", "")),
                "training_weight": float(row.get("training_weight", 1.0)),
            }
        )
    return formatted


def _training_weight_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = [float(row.get("training_weight", 1.0)) for row in rows]
    if any(weight <= 0 for weight in weights):
        raise ValueError("training_weight must be positive for every SFT row")
    contracts = sorted(
        {
            str(row.get("training_weight_contract"))
            for row in rows
            if row.get("training_weight_contract")
        }
    )
    return {
        "weight_sum": float(sum(weights)),
        "weight_min": float(min(weights)),
        "weight_max": float(max(weights)),
        "nonunit_weight_rows": sum(weight != 1.0 for weight in weights),
        "weight_contracts": contracts,
    }


def _weighted_trainer_class(base_cls: Any) -> Any:
    """Honor inverse-trajectory weights through weighted row sampling.

    Sampling with replacement makes the expected contribution of a duplicate
    trajectory group equal to one unique trajectory: ``k`` copies with weight
    ``1/k`` carry the same total sampling mass as one unit-weight row.  The
    sampler works before collation and therefore does not alter completion-only
    token masking or pass metadata into the model forward call.
    """

    class WeightedSFTTrainer(base_cls):
        def _get_train_sampler(self, train_dataset: Any | None = None) -> Any:
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            if dataset is None or not hasattr(dataset, "column_names"):
                return super()._get_train_sampler(train_dataset)
            if "training_weight" not in dataset.column_names:
                return super()._get_train_sampler(train_dataset)
            weights = torch.as_tensor(dataset["training_weight"], dtype=torch.double)
            if weights.numel() == 0 or bool(torch.any(weights <= 0)):
                raise ValueError("training_weight must be positive for every SFT row")
            generator = torch.Generator()
            generator.manual_seed(int(self.args.data_seed or self.args.seed))
            return WeightedRandomSampler(
                weights=weights,
                num_samples=int(weights.numel()),
                replacement=True,
                generator=generator,
            )

    return WeightedSFTTrainer


def _sequence_length_report(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    """Fail before training rather than silently truncating an action target."""

    maximum = 0
    maximum_id = ""
    over_limit: list[dict[str, Any]] = []
    over_limit_count = 0
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        token_ids = encoded.get("input_ids", []) if hasattr(encoded, "get") else encoded
        length = len(token_ids)
        if length > maximum:
            maximum = length
            maximum_id = str(row.get("dataset_id") or row.get("case_id") or "")
        if length > max_length:
            over_limit_count += 1
            if len(over_limit) < 20:
                over_limit.append(
                    {
                        "dataset_id": str(row.get("dataset_id") or row.get("case_id") or ""),
                        "kind": str(row.get("kind") or row.get("assistant_action") or ""),
                        "tokens": length,
                    }
                )
    report = {
        "rows": len(rows),
        "max_tokens": maximum,
        "max_tokens_dataset_id": maximum_id,
        "configured_max_length": max_length,
        "over_limit_count": over_limit_count,
        "over_limit_examples": over_limit,
    }
    if over_limit:
        raise ValueError(
            "SFT rows exceed --max_length; refusing silent truncation: "
            + json.dumps(report, ensure_ascii=False)
        )
    return report


def _copy_adapter_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("checkpoint-*", "final_adapter", "best_adapter")
    shutil.copytree(src, dst, ignore=ignore)


def _build_sft_config(config_cls: Any, args: argparse.Namespace, output_dir: Path, has_eval: bool) -> Any:
    sig = inspect.signature(config_cls)
    params = sig.parameters

    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "bf16": _supports_bf16(),
        "fp16": torch.cuda.is_available() and not _supports_bf16(),
        "gradient_checkpointing": True,
        "report_to": [],
        # Keep training_weight available to the weighted sampler. The TRL
        # collator consumes only token/mask columns, so metadata never reaches
        # the model forward call.
        "remove_unused_columns": False,
        "packing": False,
        "max_length": args.max_length,
        "max_seq_length": args.max_length,
        "completion_only_loss": args.completion_only_loss,
        "assistant_only_loss": False,
    }
    if has_eval:
        if args.load_best_model_at_end and args.save_strategy != args.eval_strategy:
            raise ValueError(
                "--load_best_model_at_end requires --save_strategy and --eval_strategy to match "
                f"(got {args.save_strategy!r} vs {args.eval_strategy!r})."
            )
        values.update(
            {
                "eval_steps": args.eval_steps,
                "save_strategy": args.save_strategy,
                "load_best_model_at_end": args.load_best_model_at_end,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
        if "eval_strategy" in params:
            values["eval_strategy"] = args.eval_strategy
        else:
            values["evaluation_strategy"] = args.eval_strategy
    else:
        if "eval_strategy" in params:
            values["eval_strategy"] = "no"
        else:
            values["evaluation_strategy"] = "no"

    return config_cls(**{key: value for key, value in values.items() if key in params})


def _build_trainer(
    *,
    trainer_cls: Any,
    model: Any,
    sft_config: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    tokenizer: Any,
    peft_config: LoraConfig | None,
    max_length: int,
) -> Any:
    sig = inspect.signature(trainer_cls)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if peft_config is not None:
        kwargs["peft_config"] = peft_config
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in sig.parameters:
        kwargs["max_seq_length"] = getattr(sft_config, "max_seq_length", None) or getattr(sft_config, "max_length", None) or max_length
    if "packing" in sig.parameters:
        kwargs["packing"] = False
    return trainer_cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one GeoDisaster LoRA adapter from turn-level SFT rows using TRL SFTTrainer.")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_strategy", choices=["steps", "epoch"], default="steps")
    parser.add_argument("--eval_strategy", choices=["steps", "epoch"], default="steps")
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--completion_only_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--validation_ratio", type=float, default=0.08)
    parser.add_argument(
        "--core_jsonl",
        default="",
        help=(
            "Canonical GeoDisaster JSONL used to keep split_group_id and "
            "trajectory_content_group_id intact in the internal validation split. "
            "Auto-discovered from GEODISASTER_CORE_JSONL, GEODISASTER_DATASET_ROOT, "
            "or the training-file ancestors when omitted."
        ),
    )
    parser.add_argument(
        "--validation_grouping",
        choices=("auto", "canonical", "case"),
        default="auto",
        help=(
            "auto uses canonical connected groups when every row is a GeoDisaster case; "
            "canonical requires them; case is an explicit generic-data fallback."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.08)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Validate rows, canonical validation grouping, token lengths, and formatted "
            "datasets, then exit before loading model weights."
        ),
    )
    parser.add_argument(
        "--init_adapter_dir",
        default="",
        help="Optional existing PEFT adapter directory to load trainably before continuing SFT.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default="",
        help=(
            "Optional Trainer checkpoint directory from this same run. Restores the "
            "adapter, optimizer, scheduler, RNG, and global optimizer step so an "
            "interrupted run continues rather than restarting."
        ),
    )
    args = parser.parse_args()

    try:
        from trl import SFTTrainer
        try:
            from trl import SFTConfig
        except Exception:
            from transformers import TrainingArguments as SFTConfig
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise SystemExit(
            "TRL is required for v3 SFT. Install it in the training env, e.g. "
            "`pip install 'trl<1'` for this transformers stack, then rerun this script."
        ) from exc

    set_seed(args.seed)
    train_file = Path(args.train_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(train_file)
    if not rows:
        raise ValueError(f"No rows found in {train_file}")
    weight_report = _training_weight_report(rows)
    core_jsonl = None if args.validation_grouping == "case" else _discover_core_jsonl(train_file, args.core_jsonl)
    validation_groups = _canonical_validation_groups(core_jsonl) if core_jsonl else None
    row_ids = {_row_case_id(row) for row in rows}
    if validation_groups is not None and not row_ids.issubset(validation_groups):
        if args.validation_grouping == "canonical":
            missing = sorted(row_ids - set(validation_groups))
            raise ValueError(
                "Canonical validation grouping was required, but rows are absent from core: "
                + json.dumps(missing[:20])
            )
        validation_groups = None
    if args.validation_grouping == "canonical" and validation_groups is None:
        raise ValueError("--validation_grouping canonical requires a discoverable --core_jsonl")
    train_rows, val_rows = _split_rows(
        rows,
        val_ratio=args.validation_ratio,
        seed=args.seed,
        validation_groups=validation_groups,
    )
    print(
        json.dumps(
            {
                "event": "rows_loaded",
                "role": args.role,
                "rows": len(rows),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "validation_grouping": "canonical_connected_groups" if validation_groups else "case_id",
                "core_jsonl": str(core_jsonl) if core_jsonl else "",
                "max_length": args.max_length,
                **weight_report,
            }
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    length_report = _sequence_length_report(rows, tokenizer, args.max_length)
    print(json.dumps({"event": "sequence_length_preflight", **length_report}), flush=True)

    train_dataset = Dataset.from_list(_format_rows(train_rows, tokenizer=tokenizer))
    eval_dataset = Dataset.from_list(_format_rows(val_rows, tokenizer=tokenizer)) if val_rows else None

    if args.dry_run:
        report = {
            "event": "sft_dry_run_complete",
            "role": args.role,
            "train_file": str(train_file),
            "model_name_or_path": args.model_name_or_path,
            "rows": len(rows),
            "train_rows": len(train_dataset),
            "val_rows": len(eval_dataset) if eval_dataset is not None else 0,
            "validation_grouping": (
                "canonical_connected_groups" if validation_groups else "case_id"
            ),
            "core_jsonl": str(core_jsonl) if core_jsonl else "",
            "training_weight": weight_report,
            "sequence_length": length_report,
        }
        (output_dir / "sft_preflight.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return

    model_kwargs = _model_load_kwargs(args)
    print(
        json.dumps(
            {
                "event": "model_load_start",
                "role": args.role,
                "direct_to_cuda": "device_map" in model_kwargs,
                "attn_implementation": model_kwargs.get("attn_implementation", ""),
            }
        ),
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    init_adapter_dir = Path(args.init_adapter_dir).resolve() if args.init_adapter_dir else None
    resume_checkpoint = (
        Path(args.resume_from_checkpoint).resolve()
        if args.resume_from_checkpoint
        else None
    )
    if init_adapter_dir and resume_checkpoint:
        raise ValueError(
            "--init_adapter_dir and --resume_from_checkpoint are mutually exclusive"
        )
    if resume_checkpoint and not resume_checkpoint.is_dir():
        raise FileNotFoundError(
            f"resume_from_checkpoint does not exist: {resume_checkpoint}"
        )
    if init_adapter_dir:
        if not init_adapter_dir.exists():
            raise FileNotFoundError(f"init_adapter_dir does not exist: {init_adapter_dir}")
        model = PeftModel.from_pretrained(model, str(init_adapter_dir), is_trainable=True)
        print(json.dumps({"event": "init_adapter_loaded", "role": args.role, "init_adapter_dir": str(init_adapter_dir)}), flush=True)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(json.dumps({"event": "model_loaded", "role": args.role}), flush=True)

    peft_config = None
    if init_adapter_dir is None:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )
    sft_config = _build_sft_config(SFTConfig, args, output_dir, has_eval=eval_dataset is not None)
    print(
        json.dumps(
            {
                "event": "sft_config",
                "role": args.role,
                "max_length": getattr(sft_config, "max_length", None),
                "max_seq_length": getattr(sft_config, "max_seq_length", None),
                "completion_only_loss": getattr(sft_config, "completion_only_loss", None),
                "assistant_only_loss": getattr(sft_config, "assistant_only_loss", None),
                "save_strategy": getattr(sft_config, "save_strategy", None),
                "eval_strategy": getattr(sft_config, "eval_strategy", None),
            },
            default=str,
        ),
        flush=True,
    )
    trainer = _build_trainer(
        trainer_cls=_weighted_trainer_class(SFTTrainer),
        model=model,
        sft_config=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=args.max_length,
    )

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    if best_checkpoint:
        if hasattr(trainer, "_load_best_model"):
            trainer._load_best_model()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        _copy_adapter_dir(output_dir, output_dir / "best_adapter")
    else:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        _copy_adapter_dir(output_dir, output_dir / "best_adapter")

    meta = {
        "role": args.role,
        "train_file": str(train_file),
        "model_name_or_path": args.model_name_or_path,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "num_train_epochs": args.num_train_epochs,
        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir else "",
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else "",
        "learning_rate": args.learning_rate,
        "validation_ratio": args.validation_ratio,
        "validation_grouping": "canonical_connected_groups" if validation_groups else "case_id",
        "core_jsonl": str(core_jsonl) if core_jsonl else "",
        "training_weight": weight_report,
        "best_model_checkpoint": best_checkpoint,
        "best_eval_loss": best_metric,
        "train_metrics": getattr(train_result, "metrics", {}),
        "best_adapter_dir": str(output_dir),
        "final_adapter_dir": str(final_dir),
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "training_complete", **meta}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
