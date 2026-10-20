from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any, TypedDict

try:
    from .compute_reward import episode_reward
    from .rl_common import (
        ROLES,
        ensure_repo_imports,
        mean,
        resolve_role_adapter,
        snapshot_adapter_root,
        std,
        supports_bf16,
        write_json,
        write_jsonl,
        write_weight_manifest,
    )
except ImportError:  # pragma: no cover - direct script execution
    from compute_reward import episode_reward
    from rl_common import (
        ROLES,
        ensure_repo_imports,
        mean,
        resolve_role_adapter,
        snapshot_adapter_root,
        std,
        supports_bf16,
        write_json,
        write_jsonl,
        write_weight_manifest,
    )


class MaxEpisodeCalls(RuntimeError):
    pass


def _install_transformers_compat_shims() -> None:
    """Install narrow shims for remote model code used by some local checkpoints."""

    try:
        import transformers.utils as transformers_utils
    except Exception:
        return

    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: Any

        transformers_utils.LossKwargs = LossKwargs


class RoleAdapterPolicy:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        policy_adapter_root: Path,
        ref_adapter_root: Path,
        roles: list[str],
        policy_adapter_subdir: str = "",
        ref_adapter_subdir: str = "",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
    ) -> None:
        _install_transformers_compat_shims()
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ensure_repo_imports()
        from multi_agent.prompts import SYSTEM_PROMPTS

        self.torch = torch
        self.roles = roles
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.turns: list[dict[str, Any]] = []
        self.max_episode_calls = 60
        self.call_count = 0
        self.system_to_role = {
            SYSTEM_PROMPTS["ORC_PLAN"]: "ORC",
            SYSTEM_PROMPTS["ORC_DECIDE"]: "ORC",
            SYSTEM_PROMPTS["GA"]: "GA",
            SYSTEM_PROMPTS["PA"]: "PA",
            SYSTEM_PROMPTS["VRA"]: "VRA",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if supports_bf16() else torch.float16,
            "low_cpu_mem_usage": True,
        }
        attn_impl = os.environ.get("MAS_MODEL_ATTN_IMPLEMENTATION", "").strip()
        if not attn_impl and "internlm" in str(model_name_or_path).lower():
            # InternLM3 + this Transformers build can fail inside the SDPA
            # attention-mask warning path before any rollout JSON is produced.
            # Keep this model on eager attention for fair GRPO/eval behavior.
            attn_impl = "eager"
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        peft_model: Any | None = None
        for idx, role in enumerate(roles):
            adapter_path = resolve_role_adapter(policy_adapter_root, role, preferred_subdir=policy_adapter_subdir)
            if not adapter_path.exists():
                raise FileNotFoundError(f"Policy adapter for {role} not found: {adapter_path}")
            if idx == 0:
                peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), adapter_name=role, is_trainable=True)
            else:
                assert peft_model is not None
                peft_model.load_adapter(str(adapter_path), adapter_name=role, is_trainable=True)

        assert peft_model is not None
        for role in roles:
            ref_path = resolve_role_adapter(ref_adapter_root, role, preferred_subdir=ref_adapter_subdir)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference SFT adapter for {role} not found: {ref_path}")
            peft_model.load_adapter(str(ref_path), adapter_name=f"{role}_ref", is_trainable=False)

        self.model = peft_model
        self.model.config.use_cache = False
        self.generation_eos_token_id = self._generation_eos_token_id()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        self.device = next(self.model.parameters()).device
        self._enforce_adapter_trainability()

    def _enforce_adapter_trainability(self) -> None:
        policy_markers = tuple(f".{role}." for role in self.roles)
        ref_markers = tuple(f".{role}_ref." for role in self.roles)
        for name, param in self.model.named_parameters():
            if any(marker in name for marker in ref_markers):
                param.requires_grad = False
            elif any(marker in name for marker in policy_markers):
                param.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def start_episode(self, max_episode_calls: int) -> None:
        self.turns = []
        self.call_count = 0
        self.max_episode_calls = max_episode_calls

    def consume_turns(self) -> list[dict[str, Any]]:
        turns = self.turns
        self.turns = []
        return turns

    def _role_for_messages(self, messages: list[dict[str, Any]]) -> str:
        system_text = str(messages[0].get("content", "")) if messages else ""
        return self.system_to_role.get(system_text, "ORC")

    def _generation_eos_token_id(self) -> Any:
        """Respect model-specific chat stop ids.

        Some instruct checkpoints use a separate end-of-turn token in addition
        to the tokenizer EOS token. InternLM3, for example, has
        generation_config.eos_token_id == [2, 128131] while tokenizer.eos is
        only 2. Passing only tokenizer.eos_token_id makes rollouts continue
        beyond ``<|im_end|>`` and often drift into invalid text.
        """
        for obj in (
            getattr(self.model, "generation_config", None),
            getattr(getattr(self.model, "base_model", None), "generation_config", None),
            getattr(self.model, "config", None),
        ):
            value = getattr(obj, "eos_token_id", None) if obj is not None else None
            if value is not None:
                return value
        return self.tokenizer.eos_token_id

    def _completion_text_for_logps(self, messages: list[dict[str, Any]], completion: str) -> str:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = self.tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": str(completion).strip()}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(full_text, str) and full_text.startswith(prompt_text):
            return full_text[len(prompt_text) :]
        eos = self.generation_eos_token_id
        if isinstance(eos, list):
            eos = eos[0] if eos else self.tokenizer.eos_token_id
        eos_text = self.tokenizer.decode([int(eos)], skip_special_tokens=False) if eos is not None else ""
        return str(completion).strip() + eos_text

    def call(self, messages: list[dict[str, Any]]) -> str:
        if self.call_count >= self.max_episode_calls:
            raise MaxEpisodeCalls(f"RL rollout exceeded {self.max_episode_calls} LLM calls")
        self.call_count += 1

        role = self._role_for_messages(messages)
        self.model.set_adapter(role)
        self._enforce_adapter_trainability()
        self.model.eval()
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.generation_eos_token_id,
                remove_invalid_values=True,
                renormalize_logits=True,
            )
        new_ids = generated[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.turns.append(
            {
                "role": role,
                "messages": copy.deepcopy(messages),
                "assistant": text,
                "prompt_text": prompt_text,
                "call_index": self.call_count,
            }
        )
        return text

    def token_logps(self, *, messages: list[dict[str, Any]], completion: str, adapter_name: str, require_grad: bool):
        torch = self.torch
        self.model.set_adapter(adapter_name)
        self._enforce_adapter_trainability()
        self.model.train(require_grad)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion_text = self._completion_text_for_logps(messages, completion)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        completion_ids = self.tokenizer(completion_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if completion_ids.numel() == 0:
            return None
        full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(self.device)
        prompt_len = int(prompt_ids.numel())
        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context:
            outputs = self.model(full_ids)
            # Some remote-code models emit bf16/fp16 logits that become
            # non-finite under log_softmax during long RL prompts. Compute the
            # probability normalisation in fp32; gradients still flow back to
            # the adapter weights, while the GRPO update avoids NaN/Inf drops.
            logits = outputs.logits[:, :-1, :].float()
            # InternLM remote-code generation/logprob passes can occasionally
            # expose NaN/Inf activations under long prompts after adapter
            # swaps. If we leave them untouched, every token in an otherwise
            # useful rescue/update turn gets dropped by the finite mask and the
            # GRPO run reports fake all-zero learning. Bound them before
            # log_softmax so the update remains finite and conservative.
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
            logits = torch.clamp(logits, min=-100.0, max=100.0)
            labels = full_ids[:, 1:]
            logps = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            positions = torch.arange(labels.shape[1], device=self.device)
            mask = positions >= max(prompt_len - 1, 0)
            selected = logps[0, mask]
        return selected

    def save_policy_adapters(self, output_adapter_root: Path) -> dict[str, str]:
        import shutil

        output_adapter_root.mkdir(parents=True, exist_ok=True)
        saved: dict[str, str] = {}
        for role in self.roles:
            out_dir = output_adapter_root / role.lower()
            tmp_dir = output_adapter_root / f".{role.lower()}_peft_save_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # PEFT writes non-default named adapters under <save_dir>/<adapter_name>/.
            # The v3 evaluator expects adapter files directly under <root>/<role>/,
            # matching SFT/DPO output, so flatten the saved payload here.
            self.model.save_pretrained(str(tmp_dir), selected_adapters=[role])
            saved_payload_dir = tmp_dir / role
            if not saved_payload_dir.exists():
                saved_payload_dir = tmp_dir
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(saved_payload_dir, out_dir)
            shutil.rmtree(tmp_dir)

            self.tokenizer.save_pretrained(str(out_dir))
            final_dir = out_dir / "final_adapter"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(out_dir, final_dir, ignore=shutil.ignore_patterns("final_adapter"))
            self.tokenizer.save_pretrained(str(final_dir))
            saved[role] = str(out_dir)
        return saved


def _install_runtime(policy: RoleAdapterPolicy, *, controller_addr: str) -> Any:
    ensure_repo_imports()
    from tool_server.tool_workers.tool_manager.base_manager import ToolManager
    from evaluation import eval_live
    from multi_agent import agents as agents_mod
    from multi_agent import llm as llm_mod
    from multi_agent import tool_gateway as tool_gateway_mod
    from multi_agent import prompts as prompts_mod
    from multi_agent import schemas as schemas_mod

    llm_mod._BACKEND_CALLS["rl_policy"] = policy.call
    os.environ["MODEL_BACKEND"] = "rl_policy"

    tool_manager = ToolManager(controller_addr)
    live_call_tool = eval_live._live_call_tool_factory(tool_manager, schemas_mod)
    original_specialist_node = agents_mod.specialist_node

    def _specialist_node_with_context(state):
        previous = eval_live._LIVE_CONTEXT.get("state")
        eval_live._LIVE_CONTEXT["state"] = state
        try:
            return original_specialist_node(state)
        finally:
            eval_live._LIVE_CONTEXT["state"] = previous

    agents_mod.call_tool = live_call_tool
    agents_mod.is_tool_allowed = tool_gateway_mod.is_tool_allowed
    agents_mod.specialist_node = _specialist_node_with_context
    prompts_mod.AGENT_TOOLS = tool_gateway_mod.AGENT_TOOLS
    return eval_live


def _load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_repo_imports()
    from evaluation import eval_live

    load_all = bool(args.sample_indices_path) or int(args.shuffle_seed) >= 0
    samples = eval_live._build_samples(
        dataset_path=Path(args.dataset_path).resolve(),
        image_dir_path=Path(args.image_dir_path).resolve(),
        num_samples=0 if load_all else args.num_prompts,
        thinkgeo_only=args.thinkgeo_only,
    )
    if args.sample_indices_path:
        payload = json.loads(Path(args.sample_indices_path).resolve().read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_indices = payload.get("dataset_indices", payload.get("indices", []))
        else:
            raw_indices = payload
        wanted = [int(item) for item in raw_indices]
        by_dataset_index = {int(sample["dataset_index"]): sample for sample in samples}
        missing = [idx for idx in wanted if idx not in by_dataset_index]
        if missing:
            raise ValueError(f"{len(missing)} requested sample indices were not found, first missing={missing[:5]}")
        samples = [by_dataset_index[idx] for idx in wanted]
    if int(args.shuffle_seed) >= 0:
        random.Random(int(args.shuffle_seed)).shuffle(samples)
    if args.num_prompts:
        samples = samples[: args.num_prompts]
    return samples


def _run_one_episode(
    *,
    sample: dict[str, Any],
    policy: RoleAdapterPolicy,
    graph: Any,
    eval_live: Any,
    state_mod: Any,
    max_episode_calls: int,
    recursion_limit: int,
) -> dict[str, Any]:
    policy.start_episode(max_episode_calls)
    state = eval_live._episode_for_sample(sample, state_mod)
    started = time.time()
    try:
        final_state = graph.invoke(state, {"recursion_limit": recursion_limit})
    except Exception as exc:
        final_state = state
        final_state["status"] = "ABORTED"
        final_state["abort_reason"] = f"rl_rollout_exception: {exc}"
        final_state["final_answer"] = ""
    wall_time_s = round(time.time() - started, 3)
    metrics = eval_live._score_answer(sample["label"], final_state.get("final_answer") or "")
    record = {
        "dataset_index": sample["dataset_index"],
        "idx": sample["idx"],
        "question": sample["question"],
        "label": sample["label"],
        "images": sample["images"],
        "status": final_state.get("status"),
        "abort_reason": final_state.get("abort_reason"),
        "final_answer": final_state.get("final_answer"),
        "wall_time_s": wall_time_s,
        "plan": final_state.get("plan").model_dump() if final_state.get("plan") is not None else None,
        "steps": final_state.get("steps"),
        "shared_state": {
            "artifacts": final_state.get("artifacts", {}),
            "layers": final_state.get("layers", {}),
            "fields": {
                name: {"value": rec["value"], "produced_by_step": rec["produced_by_step"]}
                for name, rec in (final_state.get("fields") or {}).items()
            },
        },
        "trace": final_state.get("history", []),
        "metrics": metrics,
    }
    analysis = eval_live._analyze_record(record)
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
    return {"record": record, "turns": policy.consume_turns(), "reward": breakdown.reward, "role_rewards": role_breakdowns}


def _metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _role_tool_stats(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = ((record.get("alignment_datasets") or {}).get("tool_calls") or [])
    out: dict[str, dict[str, float]] = {role: {"tool_calls": 0.0, "tool_failures": 0.0, "redundant_calls": 0.0} for role in ROLES}
    seen: dict[str, set[str]] = {role: set() for role in ROLES}
    for row in rows:
        role = str(row.get("agent", "")).upper()
        if role not in out:
            continue
        out[role]["tool_calls"] += 1.0
        if not row.get("ok"):
            out[role]["tool_failures"] += 1.0
        signature = str(row.get("semantic_signature", ""))
        if signature in seen[role]:
            out[role]["redundant_calls"] += 1.0
        seen[role].add(signature)
    return out


def _role_reward_breakdowns(record: dict[str, Any], *, roles: list[str]) -> dict[str, dict[str, float]]:
    """Dense role-conditioned reward inspired by role-contract GRPO.

    Each active adapter still receives terminal task signal, but the contract
    and robustness terms are role-local so ORC is pushed on routing/termination,
    specialists on tool/argument/summary behavior, and PA on synthesis.
    """
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    agent_metrics = record.get("agent_metrics") if isinstance(record.get("agent_metrics"), dict) else {}
    agent_raw = record.get("agent_raw_metrics") if isinstance(record.get("agent_raw_metrics"), dict) else {}
    sequence = record.get("sequence_metrics") if isinstance(record.get("sequence_metrics"), dict) else {}
    buckets = record.get("failure_buckets") if isinstance(record.get("failure_buckets"), dict) else {}
    tool_stats = _role_tool_stats(record)

    # Match the published strict scorer: partial numeric overlap is not a
    # correctness route for policy rewards.
    correct = 1.0 if bool(metrics.get("correct") or metrics.get("structured_match")) else 0.0
    answer_acc = _bounded(_metric_float(metrics, "answer_acc"))
    task_complete = 1.0 if str(record.get("status", "")) == "DONE" else 0.0
    common_task = 1.0 * correct + 0.7 * answer_acc + 0.25 * task_complete
    loop_penalty = -0.45 if any("hit max inner turns" in str(event).lower() for event in record.get("trace") or []) else 0.0

    out: dict[str, dict[str, float]] = {}
    for role in roles:
        role = role.upper()
        m = agent_metrics.get(role, {}) if isinstance(agent_metrics.get(role, {}), dict) else {}
        raw = agent_raw.get(role, {}) if isinstance(agent_raw.get(role, {}), dict) else {}
        stats = tool_stats.get(role, {})

        contract = 0.0
        robustness = 0.0
        efficiency = 0.0

        if role == "ORC":
            orc_valid = _metric_float(m, "orc_valid_rate", 1.0)
            invalid = _metric_float(raw, "orc_invalid_events")
            contract += 0.45 * orc_valid - 0.2 * invalid
            contract -= 0.25 * float(bool(buckets.get("planning_failure")))
            contract -= 0.25 * float(bool(buckets.get("orc_wrong_step_delegation")))
            robustness -= 0.3 * float(bool(buckets.get("premature_termination")))
            robustness -= 0.2 * float(bool(buckets.get("redundant_or_degenerate_loops")))
            stop_good = sequence.get("stop_after_sufficient_evidence")
            if stop_good is not None:
                robustness += 0.15 * float(bool(stop_good))
        else:
            tool_calls = _metric_float(raw, "tool_calls")
            if tool_calls > 0:
                contract += 0.2 * _metric_float(m, "TSR")
                contract += 0.15 * _metric_float(m, "ArgN")
                contract += 0.15 * _metric_float(m, "ArgV")
                contract += 0.15 * _metric_float(m, "Inst")
            if _metric_float(raw, "summary_events") > 0:
                contract += 0.15 * _metric_float(m, "Summ")
            robustness -= 0.2 * _metric_float(raw, "contract_failure")
            robustness -= 0.25 * _metric_float(raw, "loop_failures")
            robustness -= 0.2 * _metric_float(raw, "wrong_tool_calls")

        if role == "GA":
            robustness -= 0.2 * _metric_float(buckets, "ga_bbox_detours")
            robustness -= 0.15 * _metric_float(buckets, "ga_repeated_boundary_poi_overcalls")
            robustness -= 0.2 * float(bool(buckets.get("state_update_failure")))
        elif role == "VRA":
            robustness -= 0.35 * _metric_float(buckets, "vra_wrong_first_tool")
            robustness -= 0.15 * _metric_float(buckets, "vra_repeated_text_boxing")
            robustness -= 0.2 * float(bool(buckets.get("tool_selection_failure")))
        elif role == "PA":
            robustness -= 0.35 * _metric_float(buckets, "pa_weak_numeric_synthesis")
            robustness -= 0.2 * float(bool(buckets.get("premature_termination")))

        efficiency -= 0.05 * _metric_float(stats, "redundant_calls")
        efficiency -= 0.08 * _metric_float(stats, "tool_failures")
        efficiency -= 0.01 * max(_metric_float(stats, "tool_calls") - 4.0, 0.0)

        reward = common_task + contract + robustness + efficiency + loop_penalty
        out[role] = {
            "reward": float(reward),
            "task": float(common_task),
            "contract": float(contract),
            "robustness": float(robustness),
            "efficiency": float(efficiency),
            "loop_penalty": float(loop_penalty),
        }
    return out


def _ppo_grpo_update(
    *,
    policy: RoleAdapterPolicy,
    optimizer: Any,
    trajectories: list[dict[str, Any]],
    role_advantages: list[dict[str, float]],
    beta_kl: float,
    clip_range: float,
    max_grad_norm: float,
    normalize_by_trajectory_turns: bool,
) -> dict[str, float]:
    torch = policy.torch
    if not any(abs(float(role_advantages_for_traj.get(role, 0.0))) > 1e-8 for role_advantages_for_traj in role_advantages for role in policy.roles):
        return {
            "loss": 0.0,
            "kl": 0.0,
            "turn_losses": 0.0,
            "tokens": 0.0,
            "skipped_zero_advantage_update": 1.0,
        }
    optimizer.zero_grad(set_to_none=True)
    loss_count = 0
    total_loss = 0.0
    total_kl = 0.0
    total_tokens = 0
    role_loss_count: dict[str, float] = {role: 0.0 for role in policy.roles}

    for traj, advantages_for_traj in zip(trajectories, role_advantages, strict=True):
        turns = [turn for turn in traj["turns"] if turn.get("role") in policy.roles and str(turn.get("assistant", "")).strip()]
        if not turns:
            continue
        traj_weight = 1.0 / len(turns) if normalize_by_trajectory_turns else 1.0
        for turn in turns:
            role = str(turn["role"])
            messages = turn["messages"]
            completion = str(turn["assistant"])
            with torch.no_grad():
                old_logps = policy.token_logps(messages=messages, completion=completion, adapter_name=role, require_grad=False)
                ref_logps = policy.token_logps(messages=messages, completion=completion, adapter_name=f"{role}_ref", require_grad=False)
            curr_logps = policy.token_logps(messages=messages, completion=completion, adapter_name=role, require_grad=True)
            if curr_logps is None or old_logps is None or ref_logps is None:
                continue
            n = min(curr_logps.numel(), old_logps.numel(), ref_logps.numel())
            if n == 0:
                continue
            curr = curr_logps[:n]
            old = old_logps[:n].detach()
            ref = ref_logps[:n].detach()
            finite_mask = torch.isfinite(curr) & torch.isfinite(old) & torch.isfinite(ref)
            if not bool(finite_mask.any()):
                continue
            curr = curr[finite_mask]
            old = old[finite_mask]
            ref = ref[finite_mask]
            n = int(curr.numel())
            if n == 0:
                continue
            adv = torch.tensor(float(advantages_for_traj.get(role, 0.0)), dtype=curr.dtype, device=curr.device)
            log_ratio_policy = torch.clamp(curr - old, min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio_policy)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            pg_loss_1 = -adv * ratio
            pg_loss_2 = -adv * clipped_ratio
            pg_loss = torch.maximum(pg_loss_1, pg_loss_2).mean()
            log_ratio_ref = torch.clamp(ref - curr, min=-20.0, max=20.0)
            kl = (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0).mean()
            loss = (pg_loss + beta_kl * kl) * traj_weight
            if not bool(torch.isfinite(loss)):
                continue
            loss.backward()
            total_loss += float(loss.detach().cpu())
            total_kl += float(kl.detach().cpu())
            total_tokens += int(n)
            loss_count += 1
            role_loss_count[role] = role_loss_count.get(role, 0.0) + 1.0

    if loss_count:
        torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), max_grad_norm)
        optimizer.step()
    return {
        "loss": total_loss / loss_count if loss_count else 0.0,
        "kl": total_kl / loss_count if loss_count else 0.0,
        "turn_losses": float(loss_count),
        "tokens": float(total_tokens),
        "skipped_zero_advantage_update": 0.0,
        **{f"{role.lower()}_turn_losses": value for role, value in role_loss_count.items()},
    }


def _advantages(rewards: list[float]) -> list[float]:
    reward_std = std(rewards)
    if reward_std <= 1e-8:
        return [0.0 for _ in rewards]
    reward_mean = mean(rewards)
    return [(reward - reward_mean) / (reward_std + 1e-8) for reward in rewards]


def _role_advantages(trajectories: list[dict[str, Any]], roles: list[str]) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    per_role_rewards = {
        role: [float((traj.get("role_rewards") or {}).get(role, {}).get("reward", traj.get("reward", 0.0))) for traj in trajectories]
        for role in roles
    }
    per_role_advantages = {role: _advantages(values) for role, values in per_role_rewards.items()}
    by_traj: list[dict[str, float]] = []
    for idx in range(len(trajectories)):
        by_traj.append({role: per_role_advantages[role][idx] for role in roles})
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


def _save_checkpoint(policy: RoleAdapterPolicy, root: Path, step: int) -> dict[str, str]:
    checkpoint_root = root / f"checkpoint_prompt_{step:05d}"
    return policy.save_policy_adapters(checkpoint_root)


def _summarize_actions(record: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    for event in record.get("trace") or []:
        if event.get("phase") != "tool_call":
            continue
        payload = event.get("payload") or {}
        actions.append(
            {
                "role": event.get("role"),
                "step_id": event.get("step_id"),
                "turn": event.get("turn"),
                "tool": payload.get("tool"),
                "args": payload.get("args"),
                "ok": payload.get("ok"),
                "text": str(payload.get("text", ""))[:1000],
            }
        )
    return actions


def _summarize_agent_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": turn.get("role"),
            "call_index": turn.get("call_index"),
            "assistant": str(turn.get("assistant", ""))[:2000],
        }
        for turn in turns
    ]


def _write_live_plots(metrics_path: Path, plot_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return

    def rolling_mean(values: list[float], window: int) -> list[float]:
        if window <= 1:
            return values
        smoothed: list[float] = []
        running = 0.0
        queue: list[float] = []
        for value in values:
            queue.append(value)
            running += value
            if len(queue) > window:
                running -= queue.pop(0)
            smoothed.append(running / len(queue))
        return smoothed

    xs = [row["prompt_index"] for row in rows]
    window = min(25, max(5, len(rows) // 20))
    plt.figure(figsize=(10, 6))
    values = [row.get("reward_mean", 0.0) for row in rows]
    plt.plot(xs, values, color="tab:blue", alpha=0.18, linewidth=0.8, label="global_reward_mean raw")
    plt.plot(xs, rolling_mean(values, window), color="tab:blue", linewidth=2.4, label=f"global_reward_mean rolling-{window}")
    for role in ROLES:
        key = f"{role.lower()}_reward_mean"
        if key in rows[-1]:
            role_values = [row.get(key, 0.0) for row in rows]
            plt.plot(xs, role_values, alpha=0.08, linewidth=0.6)
            plt.plot(xs, rolling_mean(role_values, window), label=f"{key} rolling-{window}", alpha=0.85, linewidth=1.4)
    plt.xlabel("prompt")
    plt.ylabel("reward")
    plt.title("GRPO live reward (raw traces + rolling mean)")
    plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO-style online episode training for v3 role LoRA adapters.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--image_dir_path", default="data/openearth_agent/test")
    parser.add_argument("--scripted_tool_data_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--policy_adapter_root", required=True, help="SFT+DPO adapter root used as the trainable policy.")
    parser.add_argument("--ref_adapter_root", required=True, help="SFT adapter root used as fixed KL reference.")
    parser.add_argument("--output_adapter_root", required=True)
    parser.add_argument("--rollout_log_dir", required=True)
    parser.add_argument("--roles", default=",".join(ROLES))
    parser.add_argument("--policy_adapter_subdir", default="")
    parser.add_argument("--ref_adapter_subdir", default="")
    parser.add_argument("--snapshot_dpo_root", default="")
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--sample_indices_path", default="", help="Optional JSON/JSONL-style list or {'dataset_indices': [...]} ordering for prompts.")
    parser.add_argument("--shuffle_seed", type=int, default=-1, help="Shuffle selected prompts before truncation when >= 0.")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_episode_calls", type=int, default=60)
    parser.add_argument("--recursion_limit", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta_kl", type=float, default=0.04)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_prompts", type=int, default=50)
    parser.add_argument("--kl_watch_interval", type=int, default=50)
    parser.add_argument("--max_kl", type=float, default=5.0)
    parser.add_argument("--plot_every_prompts", type=int, default=5)
    parser.add_argument("--controller_addr", default="http://localhost:20001")
    parser.add_argument("--thinkgeo_only", action="store_true")
    parser.add_argument("--normalize_by_trajectory_turns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Validate sample and path setup without loading the model.")
    args = parser.parse_args()

    roles = [role.strip().upper() for role in args.roles.split(",") if role.strip()]
    invalid = sorted(set(roles) - set(ROLES))
    if invalid:
        raise SystemExit(f"Unsupported roles: {invalid}")

    os.environ["MAS_SCRIPTED_TOOL_ENABLE"] = "1"
    os.environ["MAS_SCRIPTED_TOOL_DATA_PATH"] = str(Path(args.scripted_tool_data_path).resolve())
    os.environ["MAS_SCRIPTED_TOOL_CASE_LIMIT"] = str(args.num_prompts)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    samples = _load_samples(args)
    if not samples:
        raise SystemExit("No RL samples selected.")

    output_adapter_root = Path(args.output_adapter_root).resolve()
    rollout_log_dir = Path(args.rollout_log_dir).resolve()
    rollout_log_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        summary = {
            "dry_run": True,
            "samples": len(samples),
            "first_sample": {key: samples[0].get(key) for key in ("dataset_index", "idx", "question", "label")},
            "policy_adapter_root": str(Path(args.policy_adapter_root).resolve()),
            "ref_adapter_root": str(Path(args.ref_adapter_root).resolve()),
            "output_adapter_root": str(output_adapter_root),
        }
        write_json(rollout_log_dir / "grpo_dry_run_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.snapshot_dpo_root:
        snapshot_adapter_root(
            Path(args.policy_adapter_root).resolve(),
            Path(args.snapshot_dpo_root).resolve(),
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
    import torch

    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    eval_live = _install_runtime(policy, controller_addr=args.controller_addr)
    ensure_repo_imports()
    from multi_agent import state as state_mod
    from multi_agent.graph import build_graph

    graph = build_graph()
    metrics_path = rollout_log_dir / "grpo_training_metrics.jsonl"
    episode_path = rollout_log_dir / "grpo_episodes.jsonl"
    live_trace_path = rollout_log_dir / "grpo_live_trace.jsonl"
    live_plot_path = rollout_log_dir / "grpo_reward_live.png"
    interval_kls: list[float] = []
    all_metrics: list[dict[str, Any]] = []
    best_reward = float("-inf")
    best_saved: dict[str, str] = {}

    for prompt_index, sample in enumerate(samples, start=1):
        trajectories: list[dict[str, Any]] = []
        for group_index in range(args.group_size):
            traj = _run_one_episode(
                sample=sample,
                policy=policy,
                graph=graph,
                eval_live=eval_live,
                state_mod=state_mod,
                max_episode_calls=args.max_episode_calls,
                recursion_limit=args.recursion_limit,
            )
            traj["group_index"] = group_index
            trajectories.append(traj)

        rewards = [float(traj["reward"]) for traj in trajectories]
        advantages = _advantages(rewards)
        role_advantages, role_reward_stats = _role_advantages(trajectories, roles)
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
        interval_kls.append(float(update["kl"]))
        metric_row = {
            "prompt_index": prompt_index,
            "dataset_index": sample.get("dataset_index"),
            "idx": sample.get("idx"),
            "reward_mean": mean(rewards),
            "reward_std": std(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "advantages": advantages,
            "role_advantages": role_advantages,
            **{f"{role.lower()}_{name}": value for role, stats in role_reward_stats.items() for name, value in stats.items()},
            "reward_breakdowns": [traj["record"].get("rl_reward_breakdown", {}) for traj in trajectories],
            "role_reward_breakdowns": [traj["record"].get("rl_role_rewards", {}) for traj in trajectories],
            **update,
        }
        all_metrics.append(metric_row)
        write_jsonl(metrics_path, all_metrics)

        episode_rows = []
        for traj, advantage in zip(trajectories, advantages, strict=True):
            episode_rows.append(
                {
                    "prompt_index": prompt_index,
                    "group_index": traj["group_index"],
                    "advantage": advantage,
                    "role_advantages": role_advantages[traj["group_index"]],
                    "reward": traj["reward"],
                    "role_rewards": traj.get("role_rewards", {}),
                    "record": traj["record"],
                    "turn_count": len(traj["turns"]),
                }
            )
        with episode_path.open("a", encoding="utf-8") as handle:
            for row in episode_rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        with live_trace_path.open("a", encoding="utf-8") as handle:
            for traj, role_advantage_row in zip(trajectories, role_advantages, strict=True):
                record = traj["record"]
                handle.write(
                    json.dumps(
                        {
                            "prompt_index": prompt_index,
                            "group_index": traj["group_index"],
                            "dataset_index": sample.get("dataset_index"),
                            "idx": sample.get("idx"),
                            "question": sample.get("question"),
                            "label": sample.get("label"),
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
                print(
                    json.dumps(
                        {
                            "event": "kl_watchdog_halt",
                            "prompt_index": prompt_index,
                            "interval_kl": interval_kl,
                            "max_kl": args.max_kl,
                        }
                    ),
                    flush=True,
                )
                break

    saved = policy.save_policy_adapters(output_adapter_root)
    summary = {
        "stage": "sft_dpo_grpo",
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
    write_json(rollout_log_dir / "grpo_run_summary.json", summary)
    write_weight_manifest(
        output_path=output_adapter_root / "weights_manifest.json",
        stage="sft_dpo_grpo",
        base_model=args.model_name_or_path,
        source_adapter_root=str(Path(args.policy_adapter_root).resolve()),
        output_adapter_root=str(output_adapter_root),
        roles=roles,
        extra=summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
