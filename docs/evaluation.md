# Evaluation

Evaluation scripts live in [`code/evaluation/`](../code/evaluation/). GeoDisaster scores both
**final-answer correctness** and **intermediate reasoning validity**, using a mix of deterministic
checks and an LLM judge.

## Evaluation scripts

| Script | Purpose |
|---|---|
| [`eval_multi_agent.py`](../code/evaluation/eval_multi_agent.py) | Canonical multi-agent entry point for trace audit, deterministic real-worker replay, and online model execution. |
| [`eval_single_agent.py`](../code/evaluation/eval_single_agent.py) | Online single-agent model execution and explicit teacher-decision integration testing. |
| [`evaluate_gt_toolchain_metrics.py`](../code/evaluation/evaluate_gt_toolchain_metrics.py) | Deterministic toolchain/answer metrics against the ground-truth teacher chains. |
| [`llm_as_judge_eval.py`](../code/evaluation/llm_as_judge_eval.py) | Compact-run-directory judge; deterministic correctness is imposed by `structured_answer.py`. |
| [`judge_conversations.py`](../code/evaluation/judge_conversations.py) | Canonical full-conversation judge for multi- and single-agent runs. |
| [`structured_answer.py`](../code/evaluation/structured_answer.py) | Versioned, structure-aware JSON answer scorer shared by single- and multi-agent evaluation. |
| [`eval_live.py`](../code/evaluation/eval_live.py) | Shared sequence metrics and normalized run-directory helpers used by the canonical evaluator. |
| [`recompute_eval_metrics.py`](../code/evaluation/recompute_eval_metrics.py), [`summarize_runs.py`](../code/evaluation/summarize_runs.py) | Recompute / summarize metrics across runs. |

## LLM as a judge

`judge_conversations.py` sends each case (question, agent trajectory, final answer, and GT
reference) to a judge model behind an OpenAI-compatible endpoint and parses a structured verdict.
The canonical shared rubric is
[`geodisaster_llm_as_judge_prompt.md`](geodisaster_llm_as_judge_prompt.md). A `dry_run` backend emits
judge requests without calling a model, so prompts and normalized case payloads can be inspected.

It auto-detects the two released transcript contracts:

- Multi-agent traces retain ORC/GA/PA/VRA plan, delegation, tool, completion, and termination events.
- Single-agent traces use `role: "agent"` actions, `role: "tool"` observations, `prediction` as the
  terminal response, and `MAIN` as the sole scored role. Delegation is explicitly not applicable.

The adapter uses `dataset_id`/`idx` from each record and never infers a canonical case identifier by
subtracting from a filename.

```bash
PYTHONPATH=code python3 -m evaluation.judge_conversations \
  --input code/runs/geodisaster_online \
  --output_dir code/runs/geodisaster_online_judge \
  --backend openai \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_model "$OPENAI_MODEL"
```

Prepare and inspect all requests without contacting a judge model:

```bash
PYTHONPATH=code python3 -m evaluation.judge_conversations \
  --input code/runs/geodisaster_single_online \
  --output_dir code/runs/geodisaster_single_judge_dry_run \
  --mode singleagent --backend dry_run
```

The compact-format runner accepts single-agent directories directly; a teacher directory is
optional because the saved result contains its gold answer:

```bash
PYTHONPATH=code python3 -m evaluation.llm_as_judge_eval \
  --eval_dir code/runs/geodisaster_single_online --dry_run
```

## Metrics: deterministic vs LLM

- **Deterministic metrics** (no model): answer matching via `structured_answer.score_answer`, plus toolchain
  metrics from `evaluate_gt_toolchain_metrics.py` — tool-sequence fidelity, per-tool argument validity
  (against the live tool specs), and plan/contract structure. Reproducible and cheap.
- **LLM metrics** (judge model): graded reasoning and semantic-quality signals from the canonical
  `judge_conversations.py` rubric. The judge cannot override deterministic answer correctness.

Reporting deterministic and LLM scores side by side separates "did the agent produce the right
structured answer" from "is the agent's reasoning judged sound".

The deterministic correctness flag requires the same JSON structure, field names, scalar types,
null states, and list order as the reference. Numeric leaves use absolute tolerance `1e-6` with
relative tolerance disabled; this absorbs only sub-micro floating-point noise and does not relax
precision as values grow. Matching an unrelated number is never a correctness route. Token and
sequence similarities are supplementary similarity scores only.

Before strict scoring, the evaluator validates the terminal action, resolves
`ANSWER_JSON:answer_1` from the runtime answer registry, and validates the resolved object against
the question schema. The scorer receives that resolved JSON object, never the reference token.
Inline terminal JSON, worker-handle references, and unavailable references are contract failures.

## Execution modes

- `trace_audit` validates the stored trajectory, ownership, termination, and stored terminal answer;
  it does not execute workers.
- `deterministic` executes the prescribed chain using real workers, performs answer projection, and
  applies strict structured scoring. This is the reproducible worker/data contract check.
- `online` runs model decisions through the live LangGraph runtime and real workers, then applies the
  same strict terminal scorer. Model transport/schema, agent-loop, tool-contract, worker, and answer
  failures are reported separately.
- `online --model_backend stub --stub_from_teacher` executes published teacher decisions through the
  live graph and workers. It is an infrastructure integration test, not a model-quality result.

The single-agent baseline provides online execution and the explicit integration mode through
`evaluation.eval_single_agent --stub_from_teacher`. It executes published single-agent decisions
through the current runtime and real workers and likewise is not a model-quality result. Normalized
single-agent rows are joined to `data/core/geodisaster_2921_source_with_splits.jsonl` by stable `id`
at runtime; the evaluator discovers this file from the public directory layout or accepts
`--core_jsonl` explicitly. Its result and summary files classify strict success, answer mismatch,
model-schema failure, model-transport failure, maximum-turn exhaustion, and runtime failure as
separate terminal categories. Unknown tool names use the same structured `unknown_tool` gateway
response as the multi-agent runtime.

The in-process and HTTP tool backends invoke the same worker computation. HTTP mode additionally
validates controller registration, serialization, routing, and worker-service boundaries.

## Evaluation workflow

1. Validate release contracts with `evaluation.audit_contracts`.
2. Execute deterministic replay with `evaluation.eval_multi_agent --mode deterministic`.
3. Run model inference with `evaluation.eval_multi_agent --mode online`.
4. Optionally score reasoning with the canonical `judge_conversations.py` evaluator.
5. Aggregate runs with `summarize_runs.py` / `recompute_eval_metrics.py`.
