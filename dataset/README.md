# GeoDisaster — Dataset

Ground-truth **agentic tool-use chains** for geospatial disaster analysis: **2,921 tasks**
(train 2,629 / test 292), **43 question types**, across **five disaster task families**.
Every task is a question grounded in real Earth-observation artifacts (optical/SAR imagery, forest
masks, hazard rasters, building-damage polygons, and packaged routing candidate products), solved end-to-end by a tool-using
agent in **two settings**:

- **single-agent** — one agent plans and calls tools itself.
- **multi-agent** — orchestrator **ORC** delegates to specialists **GA / PA / VRA**.

The Hugging Face release contains the full JSONL benchmark and all redistributable source assets.
The selected Sen1Floods11 inputs are hydrated from the official upstream bucket with the shipped
checksum-pinned installer; they are not redistributed in the public package. This folder documents
the layout and ships representative [`samples/`](samples/); the large JSONL/asset files are not
duplicated inside the code repository.

**Task families:** `deforestation` (forest-loss detection/attribution), `multi_hazard` (fire,
pollution, vegetation, exposure), `xbd_damage` (building-damage assessment), `routing` (flood-safe
routing), `sen1_flood` (Sentinel-1 SAR flood mapping).

---

## Dataset Organization

The released files mirror the Hugging Face layout. The same 2,921 tasks appear in **three forms**,
all keyed by a shared `id`:

```
data/
  core/                                          # canonical task spec (no dialogue)
    geodisaster_2921_{train,test}_source.jsonl
  single_agent/
    {train_conversations,test_reference_conversations}.jsonl   # full episodes
    {train_turns,test_turns}.jsonl                             # per-turn SFT rows
    single_agent_system_prompt.md
  multi_agent/
    {train,test}_conversations.jsonl                           # full ORC↔GA/PA/VRA episodes
    sft_{train,test}/{orc,ga,pa,vra}_{train,eval}.jsonl        # per-role SFT rows
    tool_ownership_and_prompt_context.md
source_assets/                                   # bundled inputs + hydrated Sen1Floods11 inputs
ATTRIBUTIONS.md                                  # upstream sources & licenses
```

| Group | Files | Use |
|---|---|---|
| **Main dataset** | `core/*` + `multi_agent/{train,test}_conversations.jsonl` + `single_agent/*conversations.jsonl` | the benchmark: task specs and full solved episodes |
| **SFT** | `single_agent/*_turns.jsonl` + `multi_agent/sft_{train,test}/**` | per-turn / per-role supervised fine-tuning rows |
| **GRPO** | selected from `multi_agent/train_conversations.jsonl` via `code/training/build_grpo_set.py` | prompts for contract-grounded RL |

**Forms.** **core** = task spec + ground-truth toolchain + answer. **conversation** = the full
episode (question → turns → answer). **turns** = the same episode sliced to one assistant target per
row (for SFT). **Single-agent vs multi-agent:** single-agent files hold one agent's `messages`;
multi-agent files hold the orchestrator plan plus the ORC↔GA/PA/VRA turn-by-turn episode.

**Splits.** train 2,629 / test 292 per form under
`geodisaster_source_path_trajectory_grouped_split_v4_20261015`. They are disjoint by `id`, source
sample, declared path, normalized routing content, and exact complete model-visible single-agent
trajectory. Repeated complete trajectories retained within one split carry inverse-multiplicity
`training_weight` values.

---

## How to download from Hugging Face

```python
from datasets import load_dataset

multi  = load_dataset("vimageiitb/geodisaster-agentic-2921", "multi_agent")
single = load_dataset("vimageiitb/geodisaster-agentic-2921", "single_agent")
core   = load_dataset("vimageiitb/geodisaster-agentic-2921", "core")
```

Or clone the repository and read any JSONL line-by-line:

```bash
git lfs install
git clone https://huggingface.co/datasets/vimageiitb/geodisaster-agentic-2921
cd geodisaster-agentic-2921
python3 release_tools/fetch_sen1floods11_subset.py
python3 release_tools/fetch_sen1floods11_subset.py --verify-only
```

---

## Instance Structure

Every record — in every file — carries three shared keys: `id` (the unique linking key
`"<family>/<sample>/<question_type>"`, e.g. `deforestation/000086/multi_year_loss_attribution`),
`sample_id`, and `question_type`. Artifacts live under `source_assets/<family>/<sample>/…`.

A representative **core** record (full file: [`samples/core_record.json`](samples/core_record.json)):

```json
{
  "id": "deforestation/000086/multi_year_loss_attribution",
  "sample_id": "000086",
  "question_type": "multi_year_loss_attribution",
  "question": {"handle": "question_1", "text": "… identify the year with the largest forest loss …"},
  "initial_artifacts": {
    "forest_loss_history_1": {"handle": "forest_loss_history_1", "type": "forest_loss_history",
      "paths": {"…": [{"path": "source_assets/deforestation/000086/masks/lossyear.tif",
                       "label": "lossyear.tif", "exists": true, "optional": false}]}}
  },
  "gt_chain": [{"tool": "ReadData", "args": {"input": "forest_loss_history_1"}},
               {"tool": "ComputeStats", "args": {}}],
  "expected_final_answer": {"year": 2016, "loss_ha": 201.039833}
}
```

| Field | What it stores |
|---|---|
| `id`, `sample_id`, `question_type` | shared keys; `id` links the same task across all forms |
| `question` | the stored task/evaluator context; executable prompts use the plain question plus sanitized input-handle cards, while the answer schema remains runtime-side |
| `initial_artifacts` | the evidence the task exposes: handle cards `{path, label, exists, optional}` pointing into `source_assets/` |
| `available_initial_handles` | the handle names the task starts with |
| `gt_chain` | canonical ground-truth tool sequence (`tool` + `args`) |
| `expected_final_answer` | evaluator-side target structured answer; the core JSONL is authoritative |

The agent-facing forms add the trajectory and dialogue around this spec:

- **single-agent** (`samples/single_agent_conversation.json`): `messages`
  (`{role, content}` chat turns, assistant turns emit `{thought, action}` JSON) and `tool_sequence`.
  The SFT variant (`samples/single_agent_turn.json`) is one assistant target per row, plus `kind`,
  `assistant_action`, `assistant_turn_index`.
- **multi-agent** (`samples/multi_agent_episode.json`): `plan` (ORC goal + steps `{id, agent,
  subtask, needs}`), `multiagentic_gt_chain` (the USER↔ORC↔GA/PA/VRA turns with actions and tool
  observations), `executions` (per-step `{action, tool_output}`), diagnostic `answer_support`,
  `final_answer_ref`, `resolved_final_answer`, `resolved_answer_sha256`, and
  `resolved_answer_provenance`. The per-role SFT variant
  (`samples/multi_agent_role_sft_orc.json`) is one
  role's target turn, plus `role`, `step_id`, `turn_no`, `kind`.

Every role-SFT row carries the exact system-prompt snapshot used for that
target. The standalone single-agent system prompt and the multi-agent
role/tool context are also released with the data. Executable prompt builders
in the code repository are authoritative at inference.

Terminal model actions are reference-only. `TaskComplete` closes a specialist subgoal and carries
no final JSON. Multi-agent ORC uses `Terminate`; the single-agent baseline uses `FinalAnswer`; both
carry `ANSWER_JSON:answer_1`, which the runtime resolves to a schema-valid worker-evidence
projection for scoring.

See [`../docs/dataset_schema.md`](../docs/dataset_schema.md) for the full field-level schema.

---

## Sources & license

Annotations, questions, and reasoning chains are released under **CC-BY-4.0**. The bundled
`source_assets/` derive from multiple upstream Earth-observation datasets (xBD/xView2, Sen1Floods11,
Hansen GFC, Copernicus Sentinel-1/2, OpenStreetMap, …), each keeping its own terms — see the
`ATTRIBUTIONS.md` shipped with the Hugging Face dataset.

The public package is mixed-license and therefore declares `license: other`. It does not
redistribute Sen1Floods11 files; the installer fetches the pinned 100-chip S1/JRC subset directly
from the official Cloud to Street bucket. xBD imagery and derivatives retain the source's
non-commercial restriction. `ATTRIBUTIONS.md` is authoritative for source-level terms.
