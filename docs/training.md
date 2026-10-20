# Training

GeoDisaster trains per-role LoRA adapters with **Role-Contract Expectation Alignment (RCEA)**: a
contract-aligned supervised stage followed by a contract-grounded reinforcement stage. All
training code is in [`code/training/`](../code/training/).

Create a clean environment, install the PyTorch build appropriate for the host,
then install the project dependencies with the versions exercised by the
release checks:

```bash
python3 -m venv .venv
# Install the hardware-appropriate PyTorch wheel for this host first.
.venv/bin/pip install -r code/requirements.txt \
  -c code/constraints-training-verified.txt
```

The constraints cover the rapidly changing training stack without forcing a
particular CUDA build of PyTorch.

## Contract-aligned SFT

Each multi-agent role (`orc`, `ga`, `pa`, `vra`) is fine-tuned on its role-conditioned successful
turns. A separate single-agent adapter uses the same trainer over `data/single_agent/train_turns.jsonl`.
The targets teach valid plans, owned tool calls, handle transitions, successful `TaskComplete`
decisions, and the stable terminal answer-reference protocol. The released SFT set is success-only:
it contains no synthetic `CONTRACT_FAILURE` or recovery trajectories. Failure recovery is evaluated
in live runs and can receive contract-grounded reinforcement when failures occur during GRPO
rollouts.

The trainer consumes each row's published `training_weight` through a deterministic seeded
weighted sampler. Consequently, `k` equivalent complete trajectories carrying weight `1/k` have
the same expected SFT mass as one unique trajectory. Completion-only loss still applies only to
the assistant target.

- Trainer: [`code/training/train_role_adapter.py`](../code/training/train_role_adapter.py) (TRL `SFTTrainer`, completion-only loss).
- SFT row builders: [`code/training/build_role_sft_data.py`](../code/training/build_role_sft_data.py) (multi-agent), [`code/training/build_single_agent_sft.py`](../code/training/build_single_agent_sft.py) (single-agent baseline).

```bash
export GEODISASTER_DATASET_ROOT=../geodisaster-agentic-2921
python3 -m training.train_role_adapter \
  --role orc \
  --train_file $GEODISASTER_DATASET_ROOT/data/multi_agent/sft_train/orc_train.jsonl \
  --core_jsonl $GEODISASTER_DATASET_ROOT/data/core/geodisaster_2921_source_with_splits.jsonl \
  --validation_grouping canonical \
  --model_name_or_path $GEODISASTER_MODEL_PATH --output_dir runs/sft/orc

python3 -m training.train_role_adapter \
  --role single \
  --train_file $GEODISASTER_DATASET_ROOT/data/single_agent/train_turns.jsonl \
  --core_jsonl $GEODISASTER_DATASET_ROOT/data/core/geodisaster_2921_source_with_splits.jsonl \
  --validation_grouping canonical \
  --model_name_or_path $GEODISASTER_MODEL_PATH --output_dir runs/sft/single
```

The canonical validation mode keeps every connected `split_group_id` and
`trajectory_content_group_id` component wholly on one side of the internal
train/validation boundary. This affects checkpoint selection only; the published
train/test split remains unchanged.

Pass `--dry_run` to run the complete data, grouping, tokenizer-length, and
formatting preflight without loading model weights. The report is written to
`sft_preflight.json` in the output directory.

Interrupted SFT runs resume from a Trainer checkpoint with
`--resume_from_checkpoint /path/to/checkpoint-N`. Resumption restores the adapter,
optimizer, scheduler, random-number-generator state, and global optimizer step.
`--resume_from_checkpoint` and `--init_adapter_dir` are mutually exclusive: the
former continues one training run, while the latter initializes a separate run
from an adapter.

## GRPO / RCEA

GRPO refines the SFT adapters with dense, **step-level** rewards derived from the role contract
(plan validity, correct delegation, valid tool calls, evidence grounding, answer correctness, and
contract status reporting).

The GeoDisaster GRPO runner uses the same 1,024-token generation budget as online evaluation. Its
answer reward is gated by the shared strict structured scorer; answer similarity and compatibility
numeric-subset fields cannot award correctness.

The supported `training.train_grpo_pipeline` runs live graph rollouts through the real tool gateway.
It trains the four multi-agent roles; single-agent GRPO is intentionally outside the benchmark scope.
`training/rl/train_grpo.py`, `training/build_scripted_tool_db.py`, and the scripted worker runtime
form an explicitly enabled scenario harness for isolated RL experiments; they are not the default
GeoDisaster benchmark or RCEA pipeline.

- Pipeline: [`code/training/train_grpo_pipeline.py`](../code/training/train_grpo_pipeline.py).
- Shared RL modules: [`code/training/rl/`](../code/training/rl/) — `train_grpo.py` (lower-level
  policy/update helpers), `compute_reward.py` (reward shaping), `rl_common.py` (shared utilities),
  and `check_rl_readiness.py`. The readiness checker validates the supported live-worker pipeline;
  scripted/DPO artifacts are not prerequisites. The supported benchmark-level entry point is
  `training.train_grpo_pipeline`.
- GRPO prompt set builder: [`code/training/build_grpo_set.py`](../code/training/build_grpo_set.py).

```bash
export GEODISASTER_DATASET_ROOT=../geodisaster-agentic-2921
python3 -m training.build_grpo_set \
  --input_jsonl $GEODISASTER_DATASET_ROOT/data/multi_agent/train_conversations.jsonl \
  --output_jsonl runs/grpo_data/train.jsonl

python3 -m training.train_grpo_pipeline \
  --train_jsonl runs/grpo_data/train.jsonl \
  --model_name_or_path $GEODISASTER_MODEL_PATH \
  --policy_adapter_root runs/sft \
  --ref_adapter_root runs/sft \
  --output_adapter_root runs/grpo/adapters \
  --rollout_log_dir runs/grpo/rollouts \
  --dry_run
```

`--dry_run` is a strict readiness check, not a path-printing mode. It fails when the local model,
one of the four policy/reference adapters, the 43-type prompt coverage, required hydrated assets,
locked NumPy runtime, real-worker backend, or active scoring policy is unavailable.

## Configuration locations

- **Runtime / data paths and model selection:** environment variables documented in
  [`code/.env.example`](../code/.env.example) (e.g. `GEODISASTER_MODEL_PATH`, dataset path vars).
  Export the file explicitly with `set -a; . code/.env; set +a`; entry points do not implicitly
  load dotenv files.
- **Tool-server config:** [`code/configs/tool_server_config.yaml`](../code/configs/tool_server_config.yaml).
- **Training hyperparameters:** command-line flags on the trainers above (learning rate, epochs,
  max length, gradient accumulation, save/eval strategy). Defaults are sensible and contain no
  machine-specific paths or device assumptions; override per run on the command line.

Answer-correctness rewards use the same strict structure-aware scorer as evaluation. Numeric-token
overlap is retained only as a supplementary similarity signal and cannot award correctness to an
answer with the wrong fields, types, null state, or list structure.
