# GeoDisaster — Code

Orchestrated multi-agent framework for operational geospatial disaster reasoning.
An **orchestrator (ORC)** plans and delegates to three specialists — **GA** (geospatial/data),
**PA** (analytical), and **VRA** (visual/semantic) — that call **18 disaster-oriented geospatial
tools** to turn Earth-observation evidence into grounded, structured answers. The repository also
includes a single-agent baseline, the tool workers, and the training/evaluation pipelines used for
**Role-Contract Expectation Alignment (RCEA)** (contract-aligned SFT + contract-grounded GRPO).

The dataset consumed by this code is released separately on Hugging Face; see [`../dataset/`](../dataset/).

---

## End-to-end contract

The runtime, dataset, training targets, and evaluators use one shared contract:

| Part | Executable authority | Contract |
|---|---|---|
| Agents | `multi_agent/agents.py`, `multi_agent/graph.py`, `single_agent/runtime.py` | ORC plans and delegates; GA, PA, and VRA call only their owned tools; the single-agent baseline performs the same plan/tool/final sequence without delegation. |
| Prompts | `multi_agent/prompts.py`, `single_agent/prompts.py` | Every turn requires one typed JSON action. Models see the plain question, declared handle cards, current state, and tool observations; they do not see evaluator gold or teacher chains. |
| Tool surface | `tool_server/tool_workers/online_workers/final_dataset_tool_specs.py` | One schema authority generates the 18 prompt-facing signatures, argument validation, API map, and ownership audit. `session_id` and case bootstrap data are runtime-injected transport fields. |
| Worker access | `multi_agent/tool_gateway.py` | The default `inprocess` backend invokes the real `process_<Tool>` functions. The `http` backend invokes the same workers through the controller. The optional scripted scenario harness is disabled unless `MAS_SCRIPTED_TOOL_ENABLE` is explicitly set and is not used by the supported GeoDisaster evaluator or `training.train_grpo_pipeline`. |
| State and interaction | `multi_agent/state.py`, `multi_agent/schemas.py` | Initial handles become `scene_N`, `work_N`, and analytical/artifact handles. ORC dispatches typed steps; specialists return tool calls or `TaskComplete`; failed contracts remain explicit. |
| Terminal answer | `multi_agent/answer_contract.py`, `multi_agent/answer_registry.py` | Successful worker evidence is projected into the question JSON Schema and registered as `ANSWER_JSON:answer_1`. `TaskComplete` closes a specialist step and carries no final answer. ORC emits `Terminate(answer_ref=...)`; the single agent emits `FinalAnswer(answer_ref=...)`. |
| Data | the separate dataset release | Core, multi-agent, single-agent, and SFT projections share the same `id`, split, worker chain, terminal reference, and resolved-answer hash. |
| SFT | `training/train_role_adapter.py` | Completion-only LoRA training uses the released assistant actions, a canonical connected-group validation split, sequence-length preflight, inverse-trajectory `training_weight`, configurable checkpoint/evaluation cadence, and full Trainer-checkpoint resumption. |
| GRPO | `training/train_grpo_pipeline.py` | Live graph rollouts use real workers and role adapters. Strict structured correctness gates answer reward; role-contract signals provide step-level credit. |
| Evaluation | `evaluation/eval_multi_agent.py`, `evaluation/eval_single_agent.py`, `evaluation/structured_answer.py` | Trace audit, real-worker replay, online inference, strict structured scoring, sequence/contract metrics, and optional LLM judging share the same resolved-answer contract. An LLM judge cannot override strict answer failure. |

There are two reference namespaces. Artifact handles (`scene_1`, `work_1`, `stats_1`, and similar)
carry causal worker state. `ANSWER_JSON:answer_1` is a runtime-owned terminal reference to a complete,
schema-valid JSON value; it is never accepted as a worker artifact handle. Both single-agent and
multi-agent terminal targets use this same reference.

The synchronized release contains 2,921 core records (2,629 train and 292 test), 2,921 records in
each full agent form, 43 logical question types, 18 real tools, and 55,648 turn-level SFT rows.
All 5,842 terminal SFT actions use `ANSWER_JSON:answer_1`; contract audit reports zero identity,
split, chain, ownership, terminal-action, or answer-schema drift. The public thin package requires
the documented Sen1Floods11 hydration step before real-worker execution of its 500 Sen1 records.

---

## Installation

Prerequisite: Python 3.10+ with `venv` and `pip` support. On Debian/Ubuntu systems, install
`python3.10-venv` and `python3-pip` through the system package manager if they are not already
available.

For deterministic replay/audit of the published 2,921-record benchmark, use the CPython 3.10.12
hash lock shipped with the separate dataset release:

```bash
# Run from the downloaded dataset release root.
release_tools/create_c11_replay_environment.sh
GEODISASTER_REPLAY_VENV=.venv-geodisaster-replay \
  .venv-geodisaster-replay/bin/python release_tools/verify_c11_environment.py
```

That lock covers the offline tool workers and benchmark validators on Linux x86-64. Hosted model
inference, training, and the optional HTTP tool server use the environment below. Its worker-facing
NumPy/Rasterio/SciPy/PyProj/Shapely/Pillow versions match the replay contract. Install a
hardware-appropriate PyTorch build first; the companion constraints file pins the remaining ML
stack to the versions exercised by the release checks without forcing a CUDA-specific wheel.

```bash
# Python 3.10+
python3 -m venv .venv && source .venv/bin/activate
# Install the hardware-appropriate PyTorch wheel for this host first.
python3 -m pip install -r requirements.txt -c constraints-training-verified.txt
```

All code is importable from this `code/` directory (run the commands below from here, e.g.
`cd code`). Copy `.env.example` to `.env`, set `GEODISASTER_DATASET_ROOT` to the cloned
Hugging Face dataset directory, and export it into the shell before running commands:

```bash
cp .env.example .env
# edit .env, then:
set -a; . ./.env; set +a
```

The Python entry points read environment variables; they do not implicitly load `.env`.
When large hydrated assets are mounted from a separate snapshot, set
`GEODISASTER_DATASET_ROOT` to that asset root and `GEODISASTER_CONTRACT_ROOT` to the final release
root containing the active `data/core/scoring_policy_43.json`.

`MAS_AGENT_PROFILE` defaults to `geodisaster`; this repository rejects other profile names rather
than mixing a generic prompt with the GeoDisaster gateway.

---

## Quick Start

### 1. Inference (multi-agent pipeline)

`evaluation.eval_multi_agent` is the benchmark execution entry point. It initializes the
contract-driven graph, exposes only question-visible handle cards to the agents, dispatches calls
to the real workers, projects worker evidence into the declared answer schema, and writes a
per-case transcript plus aggregate metrics. `multi_agent.run` contains conversation-log projection
helpers; it is not a separate benchmark runner.

Terminal actions use one reference-only transport contract. Multi-agent ORC emits
`{"tool":"Terminate","answer_ref":"ANSWER_JSON:answer_1"}` and the single-agent baseline emits
`{"tool":"FinalAnswer","answer_ref":"ANSWER_JSON:answer_1"}`. The runtime publishes that stable
reference only after successful worker evidence projects into the question's JSON Schema. It then
resolves and revalidates the payload before strict scoring. Inline terminal JSON and worker-handle
references are rejected; `evaluation/structured_answer.py` always scores the resolved JSON value.

Audit a published GeoDisaster split without executing workers:

```bash
export GEODISASTER_DATASET_ROOT=../geodisaster-agentic-2921
MAS_AGENT_PROFILE=geodisaster python3 -m evaluation.eval_multi_agent \
  --mode trace_audit \
  --input_jsonl $GEODISASTER_DATASET_ROOT/data/multi_agent/test_conversations.jsonl \
  --output_dir runs/geodisaster_trace_audit
```

Online model-backed GeoDisaster evaluation is available through the same script:

```bash
export GEODISASTER_DATASET_ROOT=../geodisaster-agentic-2921
MAS_AGENT_PROFILE=geodisaster MODEL_BACKEND=openai python3 -m evaluation.eval_multi_agent \
  --mode online \
  --input_jsonl $GEODISASTER_DATASET_ROOT/data/multi_agent/test_conversations.jsonl \
  --output_dir runs/geodisaster_online \
  --openai_base_url $OPENAI_BASE_URL --openai_model $OPENAI_MODEL
```

### 2. Evaluation (LLM-as-judge)

Score either multi-agent or single-agent conversation outputs with the shared judge rubric. The
loader auto-detects the transcript contract; single-agent tool events are assigned to `MAIN` and
delegation is reported as not applicable.

```bash
PYTHONPATH=. python3 -m evaluation.judge_conversations \
  --input runs/geodisaster_online \
  --output_dir runs/geodisaster_online_judge \
  --backend openai \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_model "$OPENAI_MODEL"
```

Build judge requests without calling a model:

```bash
PYTHONPATH=. python3 -m evaluation.judge_conversations \
  --input runs/geodisaster_single_online \
  --output_dir runs/geodisaster_single_judge_dry_run \
  --mode singleagent --backend dry_run
```

The versioned prompt is `../docs/geodisaster_llm_as_judge_prompt.md`. Strict deterministic answer
correctness remains authoritative; the judge supplies trajectory and qualitative scores but cannot
turn a strict failure into a pass.

Trace audit validates published trajectories and terminal answers without
executing workers. Deterministic replay executes every prescribed tool call
and separately reports worker execution, terminal-answer, and end-to-end pass
counts:

```bash
python3 -m evaluation.eval_multi_agent \
  --mode deterministic \
  --input_jsonl $GEODISASTER_DATASET_ROOT/data/multi_agent/test_conversations.jsonl \
  --output_dir runs/geodisaster_deterministic
```

Model-produced `conversation_*.json` runs can additionally be compared with an exported
`multiagent_*.json` teacher directory using `evaluation.evaluate_gt_toolchain_metrics`. Final-answer
correctness is scored by `evaluation/structured_answer.py`, which requires the reference JSON
structure and permits absolute tolerance `1e-6` only at numeric leaves; relative tolerance is
disabled. `answer_acc` is binary correctness;
`answer_similarity` is reported separately and cannot turn a structurally wrong answer into a pass.

For HTTP execution, start the 18 workers and pass `--tool_backend http` (plus
`MAS_CONTROLLER_ADDR` when using a non-default controller address):

```bash
GEODISASTER_DATASET_ROOT=../geodisaster-agentic-2921 bash scripts/start_tool_workers.sh
MAS_CONTROLLER_ADDR=http://127.0.0.1:20101 python3 -m evaluation.eval_multi_agent \
  --mode deterministic --tool_backend http --one_per_type \
  --input_jsonl $GEODISASTER_DATASET_ROOT/data/multi_agent/train_conversations.jsonl \
  --output_dir runs/geodisaster_http_contract_43
```

`--mode online` evaluates a model-driven LangGraph run and reports model transport/schema,
agent-loop/runtime, and tool/worker failures separately. For an infrastructure-only integration
test, `--model_backend stub --stub_from_teacher --one_per_type` replays the published teacher
decisions through the live graph and workers; this validates plumbing and is not a model-quality
score. `--mode gt_replay` is an accepted alias for `trace_audit`.

For live multi-agent evaluation, retain the release-safe generation budget and graph bound
(`--max_new_tokens 1024 --recursion_limit 120`). Smaller values are useful only for explicitly
labelled bounded smoke tests and can truncate plans, tool calls, or recovery trajectories.

---

## Training

Two stages implement **RCEA**. Only representative commands are shown.

**Contract-aligned SFT** — train one LoRA adapter per multi-agent role (`orc`, `ga`, `pa`, `vra`)
or one adapter for the single-agent baseline. Both forms use the same completion-only trainer:

```bash
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

The trainer defaults to a 4,096-token window and saves/evaluates every 50 steps. It tokenizes every
row before loading the model and refuses to start if any row would be silently truncated. The
released Qwen2.5-tokenized maxima are ORC 2,488, GA 2,998, PA 2,414, VRA 1,771, and single-agent
4,064 tokens. Its seeded weighted sampler consumes the released inverse-trajectory
`training_weight`, while completion-only loss supervises only the assistant action. The released
role SFT set contains successful trajectories; failure recovery is exercised by online
evaluation and may be optimized from failures encountered during GRPO rollouts.
Pass `--dry_run` to validate canonical grouping, token lengths, and formatted
datasets and write `sft_preflight.json` without loading model weights.
Resume an interrupted run with
`--resume_from_checkpoint /path/to/checkpoint-N`; this restores the adapter,
optimizer, scheduler, random-number-generator state, and optimizer step.
Checkpoint resumption is mutually exclusive with `--init_adapter_dir`, which
starts a separate run from an adapter.

**GRPO / RCEA** — contract-grounded reinforcement learning over dense, step-level rewards on top of
the four multi-agent SFT adapters. Single-agent GRPO is not part of the supported benchmark design:

```bash
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

The dry run is a strict preflight: it validates the local model, all four policy/reference
adapters, the 43-type prompt set, hydrated required assets, the real-worker backend, the locked
NumPy runtime, and the active structured scorer before any model is loaded.

Reward shaping lives in [`training/rl/compute_reward.py`](training/rl/compute_reward.py). The
single-agent turn builder is [`training/build_single_agent_sft.py`](training/build_single_agent_sft.py),
and `evaluation.eval_single_agent --adapter_path runs/sft/single` evaluates its adapter.

---

## Framework Navigation

| Component | Location |
|---|---|
| **Agent implementations** | [`multi_agent/`](multi_agent/) (ORC + GA/PA/VRA: `agents.py`, `graph.py`, `state.py`), [`single_agent/`](single_agent/) (baseline) |
| **Prompt templates** | Executable prompts in [`multi_agent/prompts.py`](multi_agent/prompts.py) and [`single_agent/prompts.py`](single_agent/prompts.py); human-readable references in [`prompts/`](prompts/) |
| **Tool workers** | [`tool_server/tool_workers/online_workers/`](tool_server/tool_workers/online_workers/) (one worker per tool + `controller.py`); agent-facing in-process/HTTP dispatch in [`multi_agent/tool_gateway.py`](multi_agent/tool_gateway.py) |
| **Tool registry** | Executable schemas in [`tool_server/.../final_dataset_tool_specs.py`](tool_server/tool_workers/online_workers/final_dataset_tool_specs.py) and ownership in [`multi_agent/final_dataset_profile.py`](multi_agent/final_dataset_profile.py); [`tool_registry/`](tool_registry/) contains synchronized publication mirrors. |
| **Training scripts** | [`training/`](training/) (`train_role_adapter.py`, `train_grpo_pipeline.py`, `rl/`) |
| **Evaluation scripts** | [`evaluation/`](evaluation/) (`eval_multi_agent.py`, `eval_single_agent.py`, `structured_answer.py`, canonical `judge_conversations.py`, and shared metric helpers) |
| **Configuration files** | [`configs/`](configs/) (tool-server config), [`.env.example`](.env.example) (runtime env) |
| **Release synchronization** | [`scripts/`](scripts/) (current contract/export synchronization, tool-session initialization, and worker launch utilities) |

See [`../docs/`](../docs/) for component-level documentation.

### Adapting to a different model or dataset

- **Model:** the agent graph talks to models through [`multi_agent/llm.py`](multi_agent/llm.py),
  which selects `MODEL_BACKEND=stub`, `hf`, or `ollama`. Dataset evaluation scripts additionally
  expose `openai` and `local_qwen` backends for OpenAI-compatible servers, vLLM deployments, and
  local adapter evaluation.
- **Dataset:** this release implements the 43 GeoDisaster contracts documented in
  [`../docs/dataset_schema.md`](../docs/dataset_schema.md). A different dataset requires explicit
  tool schemas, ownership, worker semantics, answer projection, prompts, and evaluator tests; path
  changes alone are not sufficient.

---

## Tool attribution

The 18 GeoDisaster tools are domain-specific Earth-observation / GIS operators (reading rasters and
vectors, measuring areas, detecting and classifying change, scoring hazards, estimating exposure,
planning flood-safe routes, assessing building damage, etc.). The agentic tool-surface design —
named tool workers behind a controller, JSON tool contracts, and handle-passing between steps —
follows the [OpenEarthAgent](https://github.com/mbzuai-oryx/OpenEarthAgent) tool-server style; the
GeoDisaster tool implementations and disaster task semantics are specific to this project.

To integrate OpenEarthAgent tools directly, install or clone the upstream repository, expose its tool
workers behind the same controller pattern, and add the selected tool names/signatures to
`tool_registry/agent_tool_registry.json`, `tool_registry/tool_api_map.json`, and the appropriate
role list in `tool_registry/tool_ownership.json`. The prompt builders already support the inherited
OEA-style tool families; the registry files determine which concrete tools each agent can call in a
given run.

## License

Released under the repository's top-level license (see [`../LICENSE`](../LICENSE)).
