from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.eval_live import _arg_value_ok, _extract_numbers
from evaluation.eval_live import _expected_case_for_record
from evaluation.eval_multi_agent import _online_failure_category
from evaluation.eval_single_agent import _hydrate_from_core, _new_session_id, _single_failure_category
from evaluation.judge_conversations import (
    DEFAULT_PROMPT,
    build_case_payload as build_conversation_judge_payload,
    canonical_score_row,
    load_prompt,
)
from evaluation.llm_as_judge_eval import (
    _apply_deterministic_answer_gate,
    _build_case_payload as build_compact_judge_payload,
    _load_eval_records,
)
from evaluation.structured_answer import score_answer, structured_diff
from multi_agent.answer_contract import project_final_answer, validate_explicit_answer
from multi_agent.answer_registry import ANSWER_REF, register_answer, resolve_answer_ref
from multi_agent.final_dataset_profile import (
    FINAL_DATASET_TOOL_SIGNATURES,
    FINAL_DATASET_TOOLS,
)
from multi_agent.tool_gateway import (
    clear_final_dataset_context,
    current_sanitized_case,
    normalize_read_data_args,
    set_final_dataset_context,
)
from multi_agent.llm import LLMTransportError, LLMValidationError
from multi_agent.prompts import SYSTEM_PROMPTS
from multi_agent.schemas import Terminate
from multi_agent.state import new_episode
from single_agent.schemas import SingleAgentTurn
from single_agent.runtime import call_single_tool
from training.build_grpo_set import bucket_key, select_rows, summarize, trajectory_signature
from training.train_role_adapter import _split_rows
from tool_server.tool_workers.online_workers.final_dataset_tool_specs import (
    TOOL_SPECS,
    agent_tool_spec,
    render_agent_signature,
    validate_agent_args,
)


class GrpoSelectionTests(unittest.TestCase):
    def test_multi_hazard_subfamilies_remain_distinct_with_internal_source(self) -> None:
        rows = []
        for family in ("wildfire", "drought"):
            for index in range(3):
                rows.append(
                    {
                        "dataset_id": f"multi_hazard/{family}/{index:06d}/A",
                        "source": "data/multi_hazard",
                        "question_type": "A",
                        "executions": [],
                    }
                )

        self.assertEqual(len({bucket_key(row) for row in rows}), 2)
        selected = select_rows(
            rows,
            target_size=2,
            min_per_bucket=1,
            max_per_bucket=1,
            seed=7,
        )
        summary = summarize(selected)
        self.assertEqual(summary["question_type_buckets"], 2)
        self.assertEqual(summary["bucket_count_min"], 1)
        self.assertEqual(summary["bucket_count_max"], 1)

    def test_equivalent_trajectories_are_selected_once(self) -> None:
        base = {
            "source": "routing",
            "question_type": "fastest_safest",
            "raw_question": "Compare packaged routes.",
            "plan": {"steps": [{"id": "s1", "agent": "PA"}]},
            "gt_chain": [{"tool": "CompareRoutes", "args": {"work": "routes_1"}}],
            "resolved_final_answer": {"route": "F1"},
            "executions": [],
            "training_weight": 0.5,
        }
        rows = [
            {**base, "dataset_id": "routing/000001/fastest_safest"},
            {**base, "dataset_id": "routing/000002/fastest_safest"},
        ]
        self.assertEqual(trajectory_signature(rows[0]), trajectory_signature(rows[1]))
        selected = select_rows(rows, target_size=2, min_per_bucket=2, max_per_bucket=2, seed=7)
        self.assertEqual(len(selected), 1)


class SftValidationSplitTests(unittest.TestCase):
    def test_connected_canonical_groups_do_not_cross_validation_boundary(self) -> None:
        rows = [
            {"dataset_id": case_id, "messages": []}
            for case_id in ("a", "b", "c", "d", "e", "f")
        ]
        groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3", "e": "g4", "f": "g5"}
        train, validation = _split_rows(
            rows,
            val_ratio=0.34,
            seed=42,
            validation_groups=groups,
        )
        train_groups = {groups[row["dataset_id"]] for row in train}
        validation_groups = {groups[row["dataset_id"]] for row in validation}
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual({row["dataset_id"] for row in train + validation}, set(groups))


class ToolContractTests(unittest.TestCase):
    def test_orc_termination_prompt_uses_structured_answer_contract(self) -> None:
        prompt = SYSTEM_PROMPTS["ORC_DECIDE"]
        self.assertIn('"answer_ref": "ANSWER_JSON:answer_1"', prompt)
        self.assertIn("Full inline answer JSON", prompt)
        self.assertIn("Never invent missing values", prompt)
        self.assertNotIn("<verbal final answer with concrete values>", prompt)

    def test_terminal_actions_are_reference_only(self) -> None:
        self.assertEqual(
            Terminate.model_validate(
                {"tool": "Terminate", "answer_ref": ANSWER_REF}
            ).answer_ref,
            ANSWER_REF,
        )
        with self.assertRaises(Exception):
            Terminate.model_validate({"tool": "Terminate", "answer": '{"value":1}'})
        with self.assertRaises(Exception):
            SingleAgentTurn.model_validate(
                {
                    "thought": "done",
                    "action": {"tool": "FinalAnswer", "answer_ref": "ANSWER_JSON:rank_1"},
                }
            )

    def test_answer_registry_resolves_only_registered_schema_valid_answer(self) -> None:
        case = {
            "id": "family/sample/type",
            "question_type": "type",
            "answer_schema": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        }
        state = new_episode("test")
        self.assertFalse(resolve_answer_ref(state, ANSWER_REF).ok)
        register_answer(
            state,
            case,
            {"value": 2.0},
            produced_by_step="s1",
            produced_by_tool="ComputeStats",
        )
        resolution = resolve_answer_ref(state, ANSWER_REF)
        self.assertTrue(resolution.ok)
        self.assertEqual(resolution.answer, {"value": 2.0})
        self.assertFalse(resolve_answer_ref(state, "ANSWER_JSON:rank_1").ok)

    def test_public_surface_is_generated_from_worker_schema(self) -> None:
        self.assertEqual(FINAL_DATASET_TOOLS, list(TOOL_SPECS))
        self.assertEqual(
            FINAL_DATASET_TOOL_SIGNATURES,
            {name: render_agent_signature(name) for name in TOOL_SPECS},
        )

    def test_transport_values_are_not_model_arguments(self) -> None:
        for name in TOOL_SPECS:
            public = agent_tool_spec(name)
            self.assertNotIn("session_id", public["properties"])
            self.assertNotIn("case", public["properties"])

    def test_argument_validation_rejects_schema_drift(self) -> None:
        self.assertEqual(
            validate_agent_args(
                "ReadData",
                {"question": "question_1", "input": "sar_chip_1", "role": "primary"},
            ),
            [],
        )
        errors = validate_agent_args("ReadData", {"question": "question_1", "session_id": "x"})
        self.assertTrue(any("missing required" in error for error in errors))
        self.assertTrue(any("unknown arguments" in error for error in errors))

    def test_argument_validation_rejects_unsupported_enum(self) -> None:
        errors = validate_agent_args(
            "RankRegions",
            {
                "question": "question_1",
                "work": "damage_1",
                "by": "damage_severity",
                "candidates": "candidate_regions",
            },
        )
        self.assertEqual(len(errors), 2)
        self.assertFalse(
            _arg_value_ok(
                "RankRegions",
                {
                    "question": "question_1",
                    "work": "damage_1",
                    "by": "damage_severity",
                    "candidates": "candidate_regions",
                },
                False,
                "worker rejected values",
                "agent_invalid_args",
            )
        )


class StructuredEvaluatorTests(unittest.TestCase):
    def test_diagnostic_number_parser_preserves_scientific_notation(self) -> None:
        self.assertEqual(_extract_numbers('{"value":8.802e-06}'), [8.802e-06])
        self.assertEqual(_extract_numbers("value=9e-06"), [9e-06])

    def test_matching_number_in_wrong_structure_is_incorrect(self) -> None:
        score = score_answer({"area_ha": 12.5}, {"unrelated": 12.5})
        self.assertFalse(score["correct"])
        self.assertEqual(score["answer_acc"], 0.0)
        self.assertGreater(score["answer_similarity"], 0.0)

    def test_numeric_tolerance_applies_only_after_structure_matches(self) -> None:
        score = score_answer({"area_ha": 12.5}, {"area_ha": 12.5000005})
        self.assertTrue(score["correct"])
        self.assertEqual(score["answer_acc"], 1.0)

    def test_relative_numeric_tolerance_cannot_hide_precision_loss(self) -> None:
        score = score_answer(
            {"loss_ha": 201.039833},
            {"loss_ha": 201.04},
        )
        self.assertFalse(score["correct"])
        self.assertEqual(score["relative_tolerance"], 0.0)

    def test_string_values_are_not_whitespace_normalized(self) -> None:
        score = score_answer({"class": "high"}, {"class": " high "})
        self.assertFalse(score["correct"])

    def test_nonstandard_json_numbers_are_rejected(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            score = score_answer({"value": 1.0}, f'{{"value": {constant}}}')
            self.assertFalse(score["valid_json"])
            self.assertFalse(score["correct"])

    def test_online_done_is_not_success_when_answer_is_wrong(self) -> None:
        self.assertEqual(
            _online_failure_category("DONE", "", answer_correct=False),
            "answer_mismatch",
        )
        self.assertEqual(
            _online_failure_category("DONE", "", answer_correct=True),
            "success",
        )

    def test_geodisaster_sequence_metrics_use_public_chain(self) -> None:
        expected = _expected_case_for_record(
            {
                "gt_chain": [
                    {"tool": "ReadData", "args": {"question": "question_1", "input": "input_1"}},
                    {"tool": "TaskComplete", "args": {"status": "TASK_COMPLETE"}},
                ]
            }
        )
        self.assertEqual(
            expected,
            {
                "turns": [
                    {"tool": "ReadData", "args": {"question": "question_1", "input": "input_1"}}
                ]
            },
        )

    def test_llm_judge_cannot_override_strict_answer_failure(self) -> None:
        row = {"answer_correct": True, "failure_tags": []}
        payload = {
            "deterministic_answer_evaluation": {
                "correct": False,
                "diff": [{"pointer": "/label", "kind": "value_mismatch"}],
            }
        }
        gated = _apply_deterministic_answer_gate(row, payload)
        self.assertTrue(gated["judge_answer_correct_raw"])
        self.assertFalse(gated["answer_correct"])
        self.assertIn("SynthErr", gated["failure_tags"])

    def test_structured_diff_reports_json_pointer(self) -> None:
        diff = structured_diff(
            {"answer": {"class": "high", "value": 4.0}},
            {"answer": {"class": "low", "value": 7.0}},
        )
        self.assertEqual(
            {(row["pointer"], row["kind"]) for row in diff},
            {
                ("/answer/class", "value_mismatch"),
                ("/answer/value", "numeric_mismatch"),
            },
        )

    def test_explicit_answer_must_match_question_schema(self) -> None:
        case = {
            "id": "test/one",
            "question": {
                "answer_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                    "additionalProperties": False,
                }
            },
        }
        self.assertTrue(validate_explicit_answer(case, '{"value": 2}').ok)
        self.assertFalse(validate_explicit_answer(case, '{"other": 2}').ok)
        self.assertFalse(validate_explicit_answer(case, '{"value": NaN}').ok)
        self.assertFalse(validate_explicit_answer(case, {"value": float("inf")}).ok)

    def test_nullable_object_preserves_worker_null(self) -> None:
        case = self._nullable_object_case()
        projection = project_final_answer(case, [{"change_window": None}])
        self.assertTrue(projection.ok, projection.errors)
        self.assertEqual(projection.answer, {"change_window": None})

    def test_nullable_object_preserves_populated_worker_object(self) -> None:
        case = self._nullable_object_case()
        value = {"start_epoch": "pre_1", "end_epoch": "post_1"}
        projection = project_final_answer(case, [{"change_window": value}])
        self.assertTrue(projection.ok, projection.errors)
        self.assertEqual(projection.answer, {"change_window": value})

    @staticmethod
    def _nullable_object_case() -> dict:
        return {
            "id": "deforestation/example/change_window_estimation",
            "question_type": "change_window_estimation",
            "answer_schema": {
                "type": "object",
                "properties": {
                    "change_window": {
                        "type": ["null", "object"],
                        "properties": {
                            "start_epoch": {"type": "string"},
                            "end_epoch": {"type": "string"},
                        },
                        "required": ["start_epoch", "end_epoch"],
                        "additionalProperties": False,
                    }
                },
                "required": ["change_window"],
                "additionalProperties": False,
            },
        }


class RealToolGatewayTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_final_dataset_context()

    def _install(self, role: str, *, time: str | None = None) -> None:
        card = {"handle": "input_1", "type": "raster", "role": role}
        if time is not None:
            card["metadata"] = {"time": time}
        set_final_dataset_context(
            {
                "id": "test/case",
                "question": {"text": "test", "answer_schema": {"type": "object"}},
                "initial_artifacts": {"input_1": card},
                "expected_final_answer": {"private": True},
                "gt_chain": [{"tool": "ReadData"}],
            },
            session_id="gateway_test",
        )

    def test_specific_card_semantics_fill_omitted_values(self) -> None:
        self._install("primary", time="event_window")
        normalized, errors, notes = normalize_read_data_args(
            {"question": "question_1", "input": "input_1"}
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized["role"], "primary")
        self.assertEqual(normalized["time"], "event_window")
        self.assertEqual(len(notes), 2)

    def test_specific_card_rejects_stale_model_role(self) -> None:
        self._install("roads_shelters_flood")
        _, errors, _ = normalize_read_data_args(
            {
                "question": "question_1",
                "input": "input_1",
                "role": "flood_roads",
            }
        )
        self.assertTrue(any("contradicts authoritative" in error for error in errors))

    def test_generic_card_requires_question_specific_role(self) -> None:
        self._install("requested")
        _, errors, _ = normalize_read_data_args(
            {"question": "question_1", "input": "input_1"}
        )
        self.assertTrue(any("requires role" in error for error in errors))

        normalized, errors, _ = normalize_read_data_args(
            {
                "question": "question_1",
                "input": "input_1",
                "role": "highways",
            }
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized["role"], "highways")

    def test_context_never_exposes_gold_or_chain(self) -> None:
        self._install("primary")
        clean = current_sanitized_case()
        self.assertIsNotNone(clean)
        self.assertNotIn("expected_final_answer", clean)
        self.assertNotIn("gt_chain", clean)


class SingleAgentEvaluatorTests(unittest.TestCase):
    def test_worker_session_ids_are_unique_across_reruns(self) -> None:
        record = {"idx": 7}
        first = _new_session_id(record)
        second = _new_session_id(record)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("single_7_"))

    @staticmethod
    def _single_conversation() -> dict:
        return {
            "idx": 82,
            "dataset_id": "deforestation/000149/area_estimation",
            "question_type": "area_estimation",
            "gold": {"area_ha": 4.0},
            "prediction": '{"area_ha": 4.0}',
            "status": "DONE",
            "messages": [
                {
                    "role": "user",
                    "content": "QUESTION\nEstimate loss.\n\nREGISTERED INPUTS\n- raster_1",
                }
            ],
            "trace": [
                {
                    "turn": 1,
                    "role": "agent",
                    "assistant": {
                        "thought": "Load the raster.",
                        "action": {
                            "tool": "Plan",
                            "goal": "Estimate loss",
                            "steps": [{"id": "s1", "objective": "Read input", "needs": ["raster_1"]}],
                        },
                    },
                },
                {
                    "turn": 2,
                    "role": "tool",
                    "payload": {
                        "tool": "ReadData",
                        "args": {"question": "question_1", "input": "raster_1"},
                        "ok": True,
                        "text": "opened",
                        "produced": {"fields": {"latest_handle": "scene_1"}},
                    },
                },
            ],
        }

    def test_single_agent_conversation_is_normalized_for_full_judge(self) -> None:
        record = self._single_conversation()
        payload = build_conversation_judge_payload(
            "conversation_0000",
            Path("conversation_0000.json"),
            record,
            forced_mode="auto",
            max_tool_rows=20,
            max_agent_events=20,
            max_text=500,
        )
        self.assertEqual(payload["case_id"], record["dataset_id"])
        self.assertEqual(payload["mode"], "singleagent")
        self.assertEqual(payload["question"], "Estimate loss.")
        self.assertEqual(payload["predicted_final_answer"], record["prediction"])
        self.assertEqual(payload["plan"]["tool"], "Plan")
        self.assertEqual(payload["tool_calls"][0]["tool"], "ReadData")
        self.assertEqual(payload["tool_calls"][0]["agent"], "MAIN")
        self.assertEqual(payload["agent_events"][0]["role"], "MAIN")
        self.assertEqual(payload["metric_applicability"]["delegation"], "not_applicable")

    def test_compact_judge_supports_current_single_agent_fields(self) -> None:
        record = self._single_conversation()
        payload = build_compact_judge_payload(
            82,
            record,
            record,
            max_tool_rows=20,
            max_text_chars=500,
        )
        self.assertEqual(payload["mode"], "singleagent")
        self.assertEqual(payload["question"], "Estimate loss.")
        self.assertEqual(payload["prediction"]["final_answer"], record["prediction"])
        self.assertEqual(payload["prediction"]["plan"]["goal"], "Estimate loss")
        self.assertEqual(payload["prediction"]["tool_calls"][0]["tool"], "ReadData")

    def test_zero_based_filename_uses_stable_single_agent_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conversation_0000.json"
            path.write_text(json.dumps(self._single_conversation()), encoding="utf-8")
            records = _load_eval_records(Path(temp_dir))
        self.assertEqual(set(records), {82})

    def test_canonical_judge_prompt_is_packaged(self) -> None:
        system_prompt, user_template = load_prompt(DEFAULT_PROMPT)
        self.assertIn("single-agent", system_prompt)
        self.assertIn("{case_payload_json}", user_template)

    def test_full_judge_cannot_override_deterministic_strict_result(self) -> None:
        row = canonical_score_row(
            {
                "deterministic_answer_correct": False,
                "answer_correct": True,
                "strict_correct": True,
                "StrictCorrect": 1.0,
            }
        )
        self.assertFalse(row["answer_correct"])
        self.assertFalse(row["strict_correct"])
        self.assertEqual(row["StrictCorrect"], 0.0)

    def test_unknown_tool_uses_gateway_failure_contract(self) -> None:
        result = call_single_tool("ToolCall", {"tool": "ReadData"})
        self.assertFalse(result.ok)
        self.assertEqual(result.produced.fields.get("failure_category"), "unknown_tool")

    def test_failure_categories_separate_model_and_runtime_outcomes(self) -> None:
        self.assertEqual(
            _single_failure_category("MAX_TURNS", answer_correct=False),
            "agent_max_turns",
        )
        self.assertEqual(
            _single_failure_category(
                "SCHEMA_OR_MODEL_ERROR",
                answer_correct=False,
                model_error=LLMValidationError("bad JSON"),
            ),
            "model_schema_error",
        )
        self.assertEqual(
            _single_failure_category(
                "SCHEMA_OR_MODEL_ERROR",
                answer_correct=False,
                model_error=LLMTransportError("offline"),
            ),
            "model_transport_error",
        )
        failed_trace = [
            {
                "role": "tool",
                "payload": {
                    "ok": False,
                    "text": "MapFlood failed: no declared Sentinel-1 raster is available",
                },
            }
        ]
        self.assertEqual(
            _single_failure_category(
                "SCHEMA_OR_MODEL_ERROR",
                answer_correct=False,
                model_error=LLMTransportError("stub queue empty"),
                trace=failed_trace,
            ),
            "required_asset_unavailable",
        )
        self.assertEqual(
            _online_failure_category(
                "ABORTED",
                "runtime_exception: stub queue empty",
                trace=failed_trace,
            ),
            "required_asset_unavailable",
        )

    def test_normalized_dialogue_is_hydrated_from_canonical_core(self) -> None:
        core = {
            "id": "family/sample/type",
            "dataset_id": "family/sample/type",
            "question": {"text": "canonical question", "answer_schema": {"type": "object"}},
            "initial_artifacts": {"input_1": {"type": "raster"}},
            "available_initial_handles": ["input_1"],
            "expected_final_answer": {"value": 1},
            "gt_chain": [{"tool": "ReadData", "args": {}}],
        }
        dialogue = {
            "id": "family/sample/type",
            "dataset_id": "family/sample/type",
            "messages": [{"role": "assistant", "content": "{}"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "core.jsonl"
            path.write_text(json.dumps(core) + "\n", encoding="utf-8")
            hydrated = _hydrate_from_core([dialogue], path)[0]
        self.assertEqual(hydrated["question"], core["question"])
        self.assertEqual(hydrated["initial_artifacts"], core["initial_artifacts"])
        self.assertEqual(hydrated["expected_final_answer"], {"value": 1})
        self.assertEqual(hydrated["messages"], dialogue["messages"])


if __name__ == "__main__":
    unittest.main()
