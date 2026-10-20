#!/usr/bin/env python3
"""GeoDisaster GRPO training for the v3 role LoRA adapters.

This reuses the EO GRPO policy/update implementation, but swaps rollout
generation to the GeoDisaster online evaluator path. That keeps the live
workers, handle context, prompt profile, and conversation analysis identical to
the GeoDisaster eval runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_JSONL = ROOT / "runs" / "grpo_data" / "train.jsonl"

os.environ.setdefault("MAS_AGENT_PROFILE", "geodisaster")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
for path in (str(ROOT),):
    if path not in sys.path:
        sys.path.insert(0, path)

from training.rl.compute_reward import episode_reward  # noqa: E402
from training.rl.check_rl_readiness import build_live_grpo_report  # noqa: E402
from training.rl.rl_common import ROLES, mean, snapshot_adapter_root, std, write_json, write_jsonl, write_weight_manifest  # noqa: E402
from training.rl.train_grpo import (  # noqa: E402
    RoleAdapterPolicy,
    _advantages,
    _ppo_grpo_update,
    _save_checkpoint,
    _summarize_actions,
    _summarize_agent_turns,
    _write_live_plots,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_id(sample: dict[str, Any], fallback: Any = "") -> str:
    return str(sample.get("dataset_id") or sample.get("id") or sample.get("sample_id") or fallback)


def sample_source(sample: dict[str, Any]) -> str:
    if sample.get("source"):
        return str(sample["source"])
    parts = sample_id(sample).split("/")
    if len(parts) >= 2 and parts[0] == "multi_hazard":
        return "/".join(parts[:2])
    return parts[0] if parts and parts[0] else "unknown"


def _install_policy_backend(policy: RoleAdapterPolicy) -> None:
    from multi_agent import llm as llm_mod

    llm_mod._BACKEND_CALLS["geodisaster_grpo_policy"] = policy.call
    os.environ["MODEL_BACKEND"] = "geodisaster_grpo_policy"


def _shared_state_for_log(final_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifacts": final_state.get("artifacts", {}),
        "layers": final_state.get("layers", {}),
        "fields": {
            name: {"value": rec["value"], "produced_by_step": rec["produced_by_step"]}
            for name, rec in (final_state.get("fields") or {}).items()
            if isinstance(rec, dict) and "value" in rec
        },
    }


def _role_float(obj: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(obj.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _role_reward_breakdowns(record: dict[str, Any], roles: list[str]) -> dict[str, dict[str, float]]:
    """Role-local rewards for GeoDisaster GRPO.

    The global task term is shared, but contract terms are specialized:
    ORC gets routing/validity/termination pressure, while specialists get
    tool/argument/summary and robustness pressure.
    """

    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    agent_metrics = record.get("agent_metrics") if isinstance(record.get("agent_metrics"), dict) else {}
    agent_raw = record.get("agent_raw_metrics") if isinstance(record.get("agent_raw_metrics"), dict) else {}
    buckets = record.get("failure_buckets") if isinstance(record.get("failure_buckets"), dict) else {}
    status = str(record.get("status") or "")

    # Keep role-local task reward on the same strict structured-correctness
    # contract as evaluation.  Similarity and numeric-subset fields are
    # descriptive only and cannot award correctness credit.
    correct = 1.0 if bool(metrics.get("correct") or metrics.get("structured_match")) else 0.0
    answer_acc = max(0.0, min(1.0, _role_float(metrics, "answer_acc")))
    done = 1.0 if status == "DONE" else 0.0
    common_task = 1.0 * correct + 0.8 * answer_acc + 0.35 * done
    common_robust = 0.0
    if "max inner turns" in str(record.get("abort_reason", "")).lower():
        common_robust -= 0.5
    if status == "ABORTED":
        common_robust -= 0.25

    out: dict[str, dict[str, float]] = {}
    for role in roles:
        role = role.upper()
        m = agent_metrics.get(role, {}) if isinstance(agent_metrics.get(role), dict) else {}
        raw = agent_raw.get(role, {}) if isinstance(agent_raw.get(role), dict) else {}
        contract = 0.0
        robustness = common_robust
        efficiency = 0.0

        if role == "ORC":
            contract += 0.45 * _role_float(m, "orc_valid_rate", 1.0)
            contract -= 0.20 * _role_float(raw, "orc_invalid_events")
            robustness -= 0.25 * float(bool(buckets.get("planning_failure")))
            robustness -= 0.25 * float(bool(buckets.get("orc_wrong_step_delegation")))
            robustness -= 0.20 * float(bool(buckets.get("premature_termination")))
        else:
            tool_calls = _role_float(raw, "tool_calls")
            if tool_calls > 0:
                contract += 0.25 * _role_float(m, "TSR")
                contract += 0.18 * _role_float(m, "ArgN")
                contract += 0.18 * _role_float(m, "ArgV")
                contract += 0.16 * _role_float(m, "Inst")
            if _role_float(raw, "summary_events") > 0:
                contract += 0.18 * _role_float(m, "Summ")
            robustness -= 0.25 * _role_float(raw, "contract_failure")
            robustness -= 0.25 * _role_float(raw, "loop_failures")
            robustness -= 0.25 * _role_float(raw, "wrong_tool_calls")
            efficiency -= 0.01 * max(tool_calls - 5.0, 0.0)

        reward = common_task + contract + robustness + efficiency
        out[role] = {
            "reward": float(reward),
            "task": float(common_task),
            "contract": float(contract),
            "robustness": float(robustness),
            "efficiency": float(efficiency),
        }
    return out


def _role_advantages_from_rewards(
    trajectories: list[dict[str, Any]],
    roles: list[str],
) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    per_role_rewards = {
        role: [float((traj.get("role_rewards") or {}).get(role, {}).get("reward", traj.get("reward", 0.0))) for traj in trajectories]
        for role in roles
    }
    per_role_adv = {role: _advantages(values) for role, values in per_role_rewards.items()}
    by_traj = [{role: per_role_adv[role][idx] for role in roles} for idx in range(len(trajectories))]
    stats = {
        role: {
            "reward_mean": mean(values),
            "reward_std": std(values),
            "reward_min": min(values),
            "reward_max": max(values),
        }
        for role, values in per_role_rewards.items()
    }
    return by_traj, stats


def _all_role_advantages_zero(role_advantages: list[dict[str, float]], roles: list[str]) -> bool:
    return not any(
        abs(float(role_advantages_for_traj.get(role, 0.0))) > 1e-8
        for role_advantages_for_traj in role_advantages
        for role in roles
    )


def _clip(value: float, limit: float) -> float:
    limit = abs(float(limit))
    if limit <= 0:
        return float(value)
    return max(-limit, min(limit, float(value)))


def _absolute_advantage(value: float, *, center: float, scale: float, clip_limit: float) -> float:
    """Convert an absolute shaped reward into a nonzero fallback advantage.

    GRPO normally learns from within-prompt reward differences. Some weaker
    models can collapse every sampled rollout into the same invalid JSON / ORC
    behavior; then variance is zero and the optimizer skips every update. This
    fallback is deliberately conservative: it only activates in that zero-variance
    case, and it uses the already-shaped reward sign to push away from bad
    rollouts or reinforce clearly good ones.
    """

    denom = max(abs(float(scale)), 1e-8)
    return _clip((float(value) - float(center)) / denom, clip_limit)


def _absolute_role_advantages_from_rewards(
    trajectories: list[dict[str, Any]],
    roles: list[str],
    *,
    center: float,
    scale: float,
    clip_limit: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for traj in trajectories:
        role_rewards = traj.get("role_rewards") or {}
        row: dict[str, float] = {}
        for role in roles:
            value = float((role_rewards.get(role) or {}).get("reward", traj.get("reward", 0.0)))
            row[role] = _absolute_advantage(value, center=center, scale=scale, clip_limit=clip_limit)
        rows.append(row)
    return rows


def _absolute_scalar_advantages(
    rewards: list[float],
    *,
    center: float,
    scale: float,
    clip_limit: float,
) -> list[float]:
    return [_absolute_advantage(value, center=center, scale=scale, clip_limit=clip_limit) for value in rewards]


def _make_orc_plan_rescue_turn(sample: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    """Build a compact supervised ORC-format target for invalid-plan rollouts.

    This is only used as a safety rail for models that cannot produce a valid
    ORC Plan often enough for GRPO to get nonzero trainable tokens. It does not
    expose final answers or tool outputs; it teaches the JSON plan contract.
    """

    plan = sample.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or not plan["steps"]:
        return None

    from multi_agent.prompts import SYSTEM_PROMPTS, build_orc_plan_message

    target = {
        "goal": str(plan.get("goal") or "Solve the requested GeoDisaster task."),
        "steps": plan["steps"],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["ORC_PLAN"]},
        {"role": "user", "content": build_orc_plan_message(state)},
    ]
    prompt_text = ""  # token_logps rebuilds this from messages.
    return {
        "role": "ORC",
        "messages": messages,
        "assistant": json.dumps(target, ensure_ascii=False),
        "prompt_text": prompt_text,
        "call_index": -1,
        "supervised_rescue": True,
        "rescue_reason": "orc_plan_invalid",
    }


def _sft_rescue_update(
    *,
    policy: RoleAdapterPolicy,
    optimizer: Any,
    trajectories: list[dict[str, Any]],
    weight: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Run a tiny CE update on explicitly marked rescue turns.

    GRPO remains the main objective. This update only fires when requested and
    only for rescue turns created after an invalid ORC plan, preventing all-zero
    "no learning" runs for weak initialization points.
    """

    if weight <= 0:
        return {
            "sft_rescue_loss": 0.0,
            "sft_rescue_tokens": 0.0,
            "sft_rescue_turns": 0.0,
        }

    torch = policy.torch
    rescue_turns: list[dict[str, Any]] = [
        turn
        for traj in trajectories
        for turn in (traj.get("turns") or [])
        if turn.get("supervised_rescue") and turn.get("role") in policy.roles and str(turn.get("assistant", "")).strip()
    ]
    if not rescue_turns:
        return {
            "sft_rescue_loss": 0.0,
            "sft_rescue_tokens": 0.0,
            "sft_rescue_turns": 0.0,
        }

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_tokens = 0
    loss_count = 0
    role_counts: dict[str, float] = {role: 0.0 for role in policy.roles}

    for turn in rescue_turns:
        role = str(turn["role"])
        logps = policy.token_logps(
            messages=turn["messages"],
            completion=str(turn["assistant"]),
            adapter_name=role,
            require_grad=True,
        )
        if logps is None or logps.numel() == 0:
            continue
        finite = torch.isfinite(logps)
        if not bool(finite.any()):
            continue
        selected = logps[finite]
        loss = -selected.mean() * float(weight)
        if not bool(torch.isfinite(loss)):
            continue
        loss.backward()
        total_loss += float(loss.detach().cpu())
        total_tokens += int(selected.numel())
        loss_count += 1
        role_counts[role] = role_counts.get(role, 0.0) + 1.0

    if loss_count:
        torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), max_grad_norm)
        optimizer.step()

    return {
        "sft_rescue_loss": total_loss / loss_count if loss_count else 0.0,
        "sft_rescue_tokens": float(total_tokens),
        "sft_rescue_turns": float(loss_count),
        **{f"sft_rescue_{role.lower()}_turns": value for role, value in role_counts.items()},
    }


def _run_one_episode(
    *,
    sample: dict[str, Any],
    policy: RoleAdapterPolicy,
    graph: Any,
    max_episode_calls: int,
    recursion_limit: int,
    sample_index: int,
    group_index: int,
) -> dict[str, Any]:
    from evaluation.eval_live import _analyze_record, _score_answer
    from multi_agent.tool_gateway import clear_final_dataset_context, set_final_dataset_context
    from multi_agent.run import _build_conversation_v1_shape
    from multi_agent.state import new_episode

    policy.start_episode(max_episode_calls)
    sid = set_final_dataset_context(
        sample,
        session_id=f"grpo_{sample_index:05d}_g{group_index:02d}_{time.time_ns()}",
    )
    state = new_episode(question=str(sample.get("raw_question") or sample["question"]))
    for handle, card in (sample.get("initial_artifacts") or {}).items():
        state["artifacts"][handle] = {
            "kind": str(card.get("type") or "data"),
            "path": "",
            "produced_by_step": "",
            "note": str(card.get("description") or card.get("type") or "input artifact"),
        }

    started = time.time()
    final_state: dict[str, Any] = state
    try:
        final_state = graph.invoke(state, {"recursion_limit": recursion_limit})
    except Exception as exc:
        final_state = state
        final_state["status"] = "ABORTED"
        final_state["abort_reason"] = (
            f"geodisaster_grpo_exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=12)}"
        )
        final_state["final_answer"] = ""
    finally:
        clear_final_dataset_context()

    metrics = _score_answer(
        sample.get("expected_final_answer", ""),
        final_state.get("final_answer") or "",
    )
    dataset_id = sample_id(sample, sample_index)
    source = sample_source(sample)
    record = {
        "dataset_index": sample_index,
        "idx": dataset_id,
        "dataset_id": dataset_id,
        "source": source,
        "question_type": sample.get("question_type"),
        "question": sample.get("question"),
        "label": sample.get("expected_final_answer", ""),
        "initial_artifacts": sample.get("initial_artifacts", {}),
        "status": final_state.get("status"),
        "abort_reason": final_state.get("abort_reason", ""),
        "final_answer": final_state.get("final_answer"),
        "expected_final_answer": sample.get("expected_final_answer"),
        "wall_time_s": round(time.time() - started, 3),
        "conversation": _build_conversation_v1_shape(final_state),
        "plan": final_state.get("plan").model_dump() if final_state.get("plan") is not None else None,
        "steps": final_state.get("steps"),
        "shared_state": _shared_state_for_log(final_state),
        "trace": final_state.get("history", []),
        "metrics": metrics,
        "session_id": sid,
    }
    analysis = _analyze_record(record)
    record["alignment_metrics"] = analysis["overall_metrics"]
    record["alignment_raw_metrics"] = analysis["overall_raw_metrics"]
    record["agent_metrics"] = analysis["agent_metrics"]
    record["agent_raw_metrics"] = analysis["agent_raw_metrics"]
    record["sequence_metrics"] = analysis["sequence_metrics"]
    record["failure_buckets"] = analysis["failure_buckets"]
    record["alignment_datasets"] = analysis.get("datasets", {})
    breakdown = episode_reward(record)
    role_breakdowns = _role_reward_breakdowns(record, roles=policy.roles)
    record["rl_reward"] = breakdown.reward
    record["rl_reward_breakdown"] = breakdown.__dict__
    record["rl_role_rewards"] = role_breakdowns
    turns = policy.consume_turns()
    if "orc_plan_invalid" in str(record.get("abort_reason", "")).lower():
        rescue_turn = _make_orc_plan_rescue_turn(sample, state)
        if rescue_turn is not None:
            turns.append(rescue_turn)
            record["grpo_rescue"] = {
                "type": "orc_plan_sft_rescue",
                "reason": "ORC plan was invalid before any useful tool trajectory could be trained.",
            }
    return {"record": record, "turns": turns, "reward": breakdown.reward, "role_rewards": role_breakdowns}


def _turn_counts(trajectories: list[dict[str, Any]]) -> dict[str, float]:
    total = 0.0
    rescue = 0.0
    by_role: dict[str, float] = {role: 0.0 for role in ROLES}
    for traj in trajectories:
        for turn in traj.get("turns") or []:
            total += 1.0
            role = str(turn.get("role", "")).upper()
            if role in by_role:
                by_role[role] += 1.0
            if turn.get("supervised_rescue"):
                rescue += 1.0
    return {
        "recorded_turns": total,
        "recorded_rescue_turns": rescue,
        **{f"recorded_{role.lower()}_turns": value for role, value in by_role.items()},
    }


def _aggregate_training_probe(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(records), 1)
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in records:
        key = f"{row.get('source') or 'unknown'}::{row.get('question_type')}"
        by_type[key]["n"] += 1
        by_type[key]["done"] += 1 if row.get("status") == "DONE" else 0
        by_type[key]["answer_acc"] += float((row.get("metrics") or {}).get("answer_acc", 0.0))
        by_type[key]["correct"] += 1 if (row.get("metrics") or {}).get("correct") else 0
    return {
        "records": len(records),
        "task_complete_rate": sum(1 for row in records if row.get("status") == "DONE") / n,
        "answer_acc_avg": sum(float((row.get("metrics") or {}).get("answer_acc", 0.0)) for row in records) / n,
        "correct_rate": sum(1 for row in records if (row.get("metrics") or {}).get("correct")) / n,
        "by_question_type": {
            key: {
                "records": int(vals["n"]),
                "done_rate": vals["done"] / vals["n"],
                "answer_acc_avg": vals["answer_acc"] / vals["n"],
                "correct_rate": vals["correct"] / vals["n"],
            }
            for key, vals in sorted(by_type.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoDisaster GRPO online training for v3 role LoRA adapters.")
    parser.add_argument("--train_jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--policy_adapter_root", required=True)
    parser.add_argument("--ref_adapter_root", required=True)
    parser.add_argument("--output_adapter_root", required=True)
    parser.add_argument("--rollout_log_dir", required=True)
    parser.add_argument("--roles", default=",".join(ROLES))
    parser.add_argument("--policy_adapter_subdir", default="")
    parser.add_argument("--ref_adapter_subdir", default="")
    parser.add_argument(
        "--dataset_root",
        default=os.environ.get("GEODISASTER_DATASET_ROOT", ""),
        help="Hydrated dataset root used by the real worker gateway.",
    )
    parser.add_argument(
        "--contract_root",
        default=os.environ.get("GEODISASTER_CONTRACT_ROOT", ""),
        help="Final release root containing scoring_policy_43.json; defaults to --dataset_root.",
    )
    parser.add_argument("--snapshot_policy_root", default="")
    parser.add_argument("--num_prompts", type=int, default=400)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_episode_calls", type=int, default=60)
    parser.add_argument("--recursion_limit", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta_kl", type=float, default=0.04)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_prompts", type=int, default=50)
    parser.add_argument("--plot_every_prompts", type=int, default=5)
    parser.add_argument("--max_kl", type=float, default=5.0)
    parser.add_argument("--kl_watch_interval", type=int, default=50)
    parser.add_argument("--normalize_by_trajectory_turns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--zero_advantage_fallback",
        choices=("none", "absolute"),
        default="none",
        help=(
            "When all sampled rollouts for a prompt receive identical role rewards, "
            "use no update (none) or an absolute shaped reward advantage (absolute)."
        ),
    )
    parser.add_argument("--absolute_reward_center", type=float, default=0.0)
    parser.add_argument("--absolute_reward_scale", type=float, default=0.35)
    parser.add_argument("--absolute_reward_clip", type=float, default=1.5)
    parser.add_argument(
        "--sft_rescue_weight",
        type=float,
        default=0.0,
        help=(
            "Optional cross-entropy weight for ORC invalid-plan rescue turns. "
            "Use 0 to keep pure GRPO updates."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    roles = [role.strip().upper() for role in args.roles.split(",") if role.strip()]
    invalid = sorted(set(roles) - set(ROLES))
    if invalid:
        raise SystemExit(f"Unsupported roles: {invalid}")
    if not args.dataset_root:
        raise SystemExit("--dataset_root or GEODISASTER_DATASET_ROOT is required")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    os.environ["GEODISASTER_DATASET_ROOT"] = str(dataset_root)

    samples = read_jsonl(args.train_jsonl.resolve())
    if args.num_prompts:
        samples = samples[: args.num_prompts]
    if not samples:
        raise SystemExit("No GeoDisaster GRPO samples selected.")

    output_adapter_root = Path(args.output_adapter_root).resolve()
    rollout_log_dir = Path(args.rollout_log_dir).resolve()
    rollout_log_dir.mkdir(parents=True, exist_ok=True)

    readiness = build_live_grpo_report(
        model_name_or_path=args.model_name_or_path,
        policy_adapter_root=Path(args.policy_adapter_root).resolve(),
        ref_adapter_root=Path(args.ref_adapter_root).resolve(),
        train_jsonl=args.train_jsonl.resolve(),
        dataset_root=dataset_root,
        contract_root=Path(args.contract_root).expanduser().resolve() if args.contract_root else None,
        roles=roles,
        policy_adapter_subdir=args.policy_adapter_subdir,
        ref_adapter_subdir=args.ref_adapter_subdir,
    )
    write_json(rollout_log_dir / "grpo_readiness.json", readiness)
    if not readiness["ready_for_grpo_training"]:
        print(json.dumps(readiness, indent=2, ensure_ascii=False, default=str))
        raise SystemExit("GRPO readiness failed; see grpo_readiness.json")

    if args.dry_run:
        summary = {
            "dry_run": True,
            "readiness": readiness,
            "samples": len(samples),
            "roles": roles,
            "first_sample": {
                "dataset_id": sample_id(samples[0]),
                "source": sample_source(samples[0]),
                "question_type": samples[0].get("question_type"),
                "question": str(samples[0].get("question", ""))[:300],
            },
            "policy_adapter_root": str(Path(args.policy_adapter_root).resolve()),
            "ref_adapter_root": str(Path(args.ref_adapter_root).resolve()),
            "output_adapter_root": str(output_adapter_root),
        }
        write_json(rollout_log_dir / "geodisaster_grpo_dry_run_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return

    if args.snapshot_policy_root:
        snapshot_adapter_root(
            Path(args.policy_adapter_root).resolve(),
            Path(args.snapshot_policy_root).resolve(),
            roles=roles,
            preferred_subdir=args.policy_adapter_subdir,
        )

    policy = RoleAdapterPolicy(
        model_name_or_path=args.model_name_or_path,
        policy_adapter_root=Path(args.policy_adapter_root).resolve(),
        ref_adapter_root=Path(args.ref_adapter_root).resolve(),
        roles=roles,
        policy_adapter_subdir=args.policy_adapter_subdir,
        ref_adapter_subdir=args.ref_adapter_subdir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    _install_policy_backend(policy)

    import torch
    from multi_agent.graph import build_graph

    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    graph = build_graph()
    metrics_path = rollout_log_dir / "geodisaster_grpo_training_metrics.jsonl"
    episode_path = rollout_log_dir / "geodisaster_grpo_episodes.jsonl"
    live_trace_path = rollout_log_dir / "geodisaster_grpo_live_trace.jsonl"
    live_plot_path = rollout_log_dir / "geodisaster_grpo_reward_live.png"
    all_metrics: list[dict[str, Any]] = []
    interval_kls: list[float] = []
    best_reward = float("-inf")
    best_saved: dict[str, str] = {}

    for prompt_index, sample in enumerate(samples, start=1):
        trajectories: list[dict[str, Any]] = []
        for group_index in range(args.group_size):
            traj = _run_one_episode(
                sample=sample,
                policy=policy,
                graph=graph,
                max_episode_calls=args.max_episode_calls,
                recursion_limit=args.recursion_limit,
                sample_index=prompt_index,
                group_index=group_index,
            )
            traj["group_index"] = group_index
            trajectories.append(traj)

        rewards = [float(traj["reward"]) for traj in trajectories]
        advantages = _advantages(rewards)
        role_advantages, role_reward_stats = _role_advantages_from_rewards(trajectories, roles)
        advantage_mode = "group_relative"
        if _all_role_advantages_zero(role_advantages, roles):
            advantage_mode = "zero_variance_skip"
            if args.zero_advantage_fallback == "absolute":
                role_advantages = _absolute_role_advantages_from_rewards(
                    trajectories,
                    roles,
                    center=args.absolute_reward_center,
                    scale=args.absolute_reward_scale,
                    clip_limit=args.absolute_reward_clip,
                )
                advantages = _absolute_scalar_advantages(
                    rewards,
                    center=args.absolute_reward_center,
                    scale=args.absolute_reward_scale,
                    clip_limit=args.absolute_reward_clip,
                )
                advantage_mode = "absolute_reward_fallback"
        update = _ppo_grpo_update(
            policy=policy,
            optimizer=optimizer,
            trajectories=trajectories,
            role_advantages=role_advantages,
            beta_kl=args.beta_kl,
            clip_range=args.clip_range,
            max_grad_norm=args.max_grad_norm,
            normalize_by_trajectory_turns=args.normalize_by_trajectory_turns,
        )
        rescue_update = _sft_rescue_update(
            policy=policy,
            optimizer=optimizer,
            trajectories=trajectories,
            weight=args.sft_rescue_weight,
            max_grad_norm=args.max_grad_norm,
        )
        interval_kls.append(float(update["kl"]))
        metric_row = {
            "prompt_index": prompt_index,
            "dataset_id": sample_id(sample, prompt_index),
            "source": sample_source(sample),
            "question_type": sample.get("question_type"),
            "reward_mean": mean(rewards),
            "reward_std": std(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "advantages": advantages,
            "role_advantages": role_advantages,
            "advantage_mode": advantage_mode,
            "absolute_reward_center": args.absolute_reward_center,
            "absolute_reward_scale": args.absolute_reward_scale,
            "absolute_reward_clip": args.absolute_reward_clip,
            "sft_rescue_weight": args.sft_rescue_weight,
            **{f"{role.lower()}_{name}": value for role, stats in role_reward_stats.items() for name, value in stats.items()},
            "reward_breakdowns": [traj["record"].get("rl_reward_breakdown", {}) for traj in trajectories],
            "role_reward_breakdowns": [traj["record"].get("rl_role_rewards", {}) for traj in trajectories],
            **_turn_counts(trajectories),
            **update,
            **rescue_update,
        }
        all_metrics.append(metric_row)
        write_jsonl(metrics_path, all_metrics)

        with episode_path.open("a", encoding="utf-8") as handle:
            for traj, advantage in zip(trajectories, advantages, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "prompt_index": prompt_index,
                            "group_index": traj["group_index"],
                            "advantage": advantage,
                            "role_advantages": role_advantages[traj["group_index"]],
                            "reward": traj["reward"],
                            "role_rewards": traj.get("role_rewards", {}),
                            "record": traj["record"],
                            "turn_count": len(traj["turns"]),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        with live_trace_path.open("a", encoding="utf-8") as handle:
            for traj, role_advantage_row in zip(trajectories, role_advantages, strict=True):
                record = traj["record"]
                handle.write(
                    json.dumps(
                        {
                            "prompt_index": prompt_index,
                            "group_index": traj["group_index"],
                            "dataset_id": sample_id(sample, prompt_index),
                            "source": sample_source(sample),
                            "question_type": sample.get("question_type"),
                            "question": sample.get("question"),
                            "label": sample.get("expected_final_answer"),
                            "status": record.get("status"),
                            "final_answer": record.get("final_answer"),
                            "reward": traj["reward"],
                            "role_rewards": traj.get("role_rewards", {}),
                            "role_advantages": role_advantage_row,
                            "metrics": record.get("metrics", {}),
                            "failure_buckets": record.get("failure_buckets", {}),
                            "actions": _summarize_actions(record),
                            "agent_outputs": _summarize_agent_turns(traj["turns"]),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

        probe = _aggregate_training_probe([traj["record"] for traj in trajectories])
        metric_row["group_probe"] = probe
        print(json.dumps(metric_row, ensure_ascii=False, default=str), flush=True)
        if metric_row["reward_mean"] > best_reward:
            best_reward = float(metric_row["reward_mean"])
            best_saved = policy.save_policy_adapters(rollout_log_dir / "checkpoints" / "best_by_reward")
        if args.save_every_prompts and prompt_index % args.save_every_prompts == 0:
            _save_checkpoint(policy, rollout_log_dir / "checkpoints", prompt_index)
        if args.plot_every_prompts and prompt_index % args.plot_every_prompts == 0:
            _write_live_plots(metrics_path, live_plot_path)
        if args.kl_watch_interval and prompt_index % args.kl_watch_interval == 0:
            interval_kl = mean(interval_kls[-args.kl_watch_interval :])
            if interval_kl > args.max_kl:
                print(json.dumps({"event": "kl_watchdog_halt", "prompt_index": prompt_index, "interval_kl": interval_kl}), flush=True)
                break

    saved = policy.save_policy_adapters(output_adapter_root)
    summary = {
        "stage": "geodisaster_sft_grpo",
        "samples_seen": len(all_metrics),
        "group_size": args.group_size,
        "reward_mean_final_window": mean(row["reward_mean"] for row in all_metrics[-20:]),
        "kl_mean_final_window": mean(row["kl"] for row in all_metrics[-20:]),
        "saved_adapters": saved,
        "best_by_reward_adapters": best_saved,
        "rollout_log_dir": str(rollout_log_dir),
        "live_trace_path": str(live_trace_path),
        "live_plot_path": str(live_plot_path),
    }
    write_json(rollout_log_dir / "geodisaster_grpo_run_summary.json", summary)
    write_weight_manifest(
        output_path=output_adapter_root / "weights_manifest.json",
        stage="geodisaster_sft_grpo",
        base_model=args.model_name_or_path,
        source_adapter_root=str(Path(args.policy_adapter_root).resolve()),
        output_adapter_root=str(output_adapter_root),
        roles=roles,
        extra=summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
