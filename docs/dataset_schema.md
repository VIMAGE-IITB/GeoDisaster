# Dataset Schema

The dataset releases the same **2,921 tasks** in three forms and two agent settings, all keyed by a
shared `id`. See [`../dataset/README.md`](../dataset/README.md) for the folder layout and
[`../dataset/samples/`](../dataset/samples/) for full real records.

## Shared identity keys (every record, every file)

| Key | Type | Meaning |
|---|---|---|
| `id` | string | unique linking key `"<family>/<sample>/<question_type>"` — the same task has the same `id` across all files |
| `sample_id` | string | the source sample within its family |
| `question_type` | string | one of 43 question types |
| `training_weight` | number | inverse exact-trajectory multiplicity within a split; `1.0` for a unique trajectory |

Core records use `split`; full conversation records use `final_split`. Turn-level SFT files encode
the split in their containing `sft_train`/`sft_test` or `train_turns`/`test_turns` path. Every form
retains `training_weight`.

## Dataset files

| File group | Form | Contents |
|---|---|---|
| `data/core/geodisaster_2921_{train,test}_source.jsonl` | **main** | canonical task spec: `question`, `initial_artifacts`, `available_initial_handles`, `gt_chain`, `expected_final_answer`, `source_replay`, `path_summary` |
| `data/multi_agent/{train,test}_conversations.jsonl` | **main** | full ORC↔GA/PA/VRA episodes (see Trajectory below) |
| `data/single_agent/{train_conversations,test_reference_conversations}.jsonl` | **main** | full single-agent episodes: `messages`, `tool_sequence` |
| `data/single_agent/{train,test}_turns.jsonl` | **SFT** | one assistant target per row: `messages`, `kind`, `assistant_action`, `assistant_turn_index` |
| `data/multi_agent/sft_{train,test}/{orc,ga,pa,vra}_{train,eval}.jsonl` | **SFT** | one role-turn per row: `messages`, `role`, `step_id`, `turn_no`, `kind` |
| GRPO prompt sets | **GRPO** | selected from `data/multi_agent/train_conversations.jsonl` via `code/training/build_grpo_set.py` |

## Trajectory structure

**Single-agent** — `messages` is a chat list (`{role, content}`); each assistant turn is a JSON
object `{"thought": …, "action": {"tool": …, "args": …}}`, ending in
`{"tool":"FinalAnswer","answer_ref":"ANSWER_JSON:answer_1"}`.
`tool_sequence` is the ordered list of tools called.

**Multi-agent** — an episode adds:
- `plan` — the orchestrator plan `{goal, steps:[{id, agent, subtask, needs}]}`.
- `multiagentic_gt_chain` — the turn-by-turn episode (USER ↔ ORC ↔ GA/PA/VRA), each turn with its
  `thought`, `action`, and tool `observation`.
- `executions` — per-step `{action, tool_output}` log.
- `final_answer_ref` — stable model-facing terminal token (`ANSWER_JSON:answer_1`).
- `resolved_final_answer` — schema-valid worker projection used by evaluation.
- `resolved_answer_sha256` — canonical-JSON checksum linking derived exports.
- `answer_support` — evidence metadata backing the resolved answer.

## Artifact and handle structure

`initial_artifacts` maps a **handle** name to a card describing one piece of evidence:

```json
{"handle": "forest_loss_history_1", "type": "forest_loss_history",
 "paths": {"…": [{"path": "source_assets/deforestation/000086/masks/lossyear.tif",
                  "label": "lossyear.tif", "exists": true, "optional": false}]}}
```

Handles are the unit of data flow: the question starts with `question_1` plus the input handles; a
`ReadData` call turns an input handle into a `scene_N` handle; `PrepareScene` produces a `work_N`
handle; analytic tools consume those and emit further handles (`stats_N`, `change_N`, `damage_N`, …).
Tools reference evidence **by handle**, never by re-reading raw paths, and `optional: true` cards
mark evidence an agent may use but is not required to.

## Answer structure

`expected_final_answer` in core is the **structured** target answer — a JSON object whose keys depend on the question type (e.g.
`{"year": 2016, "loss_ha": 201.039833}` for forest-loss attribution, a ranked region list for exposure
questions, route comparisons for routing). Ground truth is produced by deterministic geospatial
workflows and consistency checks, not language-model annotation, which is what makes both answer
correctness and intermediate toolchain validity checkable.

SFT terminal targets do not reproduce that JSON. Both agent settings emit the stable reference
`ANSWER_JSON:answer_1`; the runtime creates that registry entry only after successful worker
evidence projects into the declared answer schema. The evaluator resolves the reference and scores
the resulting JSON. Thus the reference is transport, while the resolved object is the answer.
Active prompts contain the plain question and declared input-handle cards; the runtime owns the
schema and does not inline it into every SFT turn.

`TaskComplete` is a specialist-step status, not a terminal-answer container. Multi-agent episodes
finish with ORC `Terminate`; single-agent episodes finish with `FinalAnswer`. Both actions carry only
the registered reference. Artifact handles and answer references are separate namespaces.

## Split contract

The 2,629/292 split uses
`geodisaster_source_path_trajectory_grouped_split_v4_20261015`. Source-linked records, shared
declared inputs, normalized-equivalent routing cases, and identical complete model-visible
single-agent trajectories stay in one split. The evaluator-visible test view is
`data/core/geodisaster_2921_test_inputs.jsonl`; files containing test answers or reference
trajectories remain evaluator-side.
