from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from course.shared.config import RetailCourseConfig
from course.shared.data import normalize_record, sample_records, scenario_from_record
from course.shared.rl_sampling import reward_signal_metrics
from course.shared.retail_env import ReplayRetailEnv, is_state_changing_tool
from course.shared.rewards import score_messages
from course.shared.wandb_artifacts import checkpoint_artifact_aliases, latest_checkpoint_dir, select_checkpoint_dir

record_to_next_action_examples = importlib.import_module(
    "course.03_sft_warmup.make_next_action_sft_jsonl"
).record_to_next_action_examples
teacher_sft = importlib.import_module("course.03_sft_warmup.make_teacher_next_action_sft_jsonl")
areal_sft = importlib.import_module("course.03_sft_warmup.make_areal_retail_sft_jsonl")
success_trace_sft = importlib.import_module("course.03_sft_warmup.make_success_trace_retail_sft_jsonl")
cached_weave_eval = importlib.import_module("course.02_weave_evals.evaluate_cached_checkpoint")
stage_acceptance = importlib.import_module("course.02_weave_evals.check_stage_acceptance")


def assistant_tool_call(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class RetailRewardInvariantTests(unittest.TestCase):
    def test_normalize_record_prefers_stable_record_id(self) -> None:
        row = {
            "id": "37",
            "messages": [],
            "metadata": {"record_id": "retail-sonnet35-37-4", "task_id": 37, "trial": 4},
        }

        normalized = normalize_record(row, split="train", index=0)

        self.assertEqual(normalized["id"], "retail-sonnet35-37-4")

    def test_next_action_merges_consecutive_assistant_chunks(self) -> None:
        record = sample_records()[0]

        examples = record_to_next_action_examples(record)

        self.assertEqual(len(examples), 3)
        state_action = examples[1]["messages"][-1]
        self.assertEqual(state_action["role"], "assistant")
        self.assertEqual(state_action["content"], "")
        self.assertEqual(
            state_action["tool_calls"][0]["function"]["name"],
            "cancel_pending_order",
        )

    def test_teacher_next_action_conversion_uses_final_answer_turn(self) -> None:
        row = {
            "sample_idx": 7,
            "conversations": [
                {"role": "user", "content": "Please cancel order O-1001."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_lookup",
                            "type": "function",
                            "function": {"name": "get_order_details", "arguments": "{\"order_id\":\"O-1001\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_lookup", "content": "{\"status\":\"pending\"}"},
            ],
            "answer": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_cancel",
                            "type": "function",
                            "function": {
                                "name": "cancel_pending_order",
                                "arguments": "{\"order_id\":\"O-1001\"}",
                            },
                        }
                    ],
                }
            ],
            "total_score": 1.0,
            "avg_score": 1.0,
        }
        tools = sample_records()[0]["tools"]

        example = teacher_sft.convert_teacher_row(
            row,
            row_index=0,
            tools=tools,
            dataset_id="teacher/example",
            split="train",
            drop_unknown_tools=True,
            min_total_score=1.0,
            min_avg_score=1.0,
        )

        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example["metadata"]["sft_format"], "teacher-retail-next-action")
        final_message = example["messages"][-1]
        self.assertEqual(final_message["role"], "assistant")
        self.assertEqual(final_message["content"], "")
        self.assertEqual(final_message["tool_calls"][0]["function"]["name"], "cancel_pending_order")

    def test_teacher_next_action_conversion_drops_unknown_answer_tool(self) -> None:
        row = {
            "conversations": [{"role": "user", "content": "Delete my account."}],
            "answer": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_delete",
                            "type": "function",
                            "function": {"name": "delete_customer_account", "arguments": "{}"},
                        }
                    ],
                }
            ],
            "total_score": 1.0,
            "avg_score": 1.0,
        }

        example = teacher_sft.convert_teacher_row(
            row,
            row_index=0,
            tools=sample_records()[0]["tools"],
            dataset_id="teacher/example",
            split="train",
            drop_unknown_tools=True,
            min_total_score=1.0,
            min_avg_score=1.0,
        )

        self.assertIsNone(example)

    def test_areal_conversion_strips_thinking_and_normalizes_answer_tool_call(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "# Retail agent policy\nHelp the customer."},
                {"role": "user", "content": "Please cancel order O-1001."},
                {
                    "role": "assistant",
                    "content": "",
                    "thinking": "I should look up the order.",
                    "tool_calls": [{"name": "get_order_details", "arguments": {"order_id": "O-1001"}}],
                },
                {"role": "tool", "content": "{\"status\":\"pending\"}"},
            ],
            "answer": {
                "role": "assistant",
                "content": "",
                "thinking": "Now cancel it.",
                "tool_calls": [{"name": "cancel_pending_order", "arguments": {"order_id": "O-1001"}}],
            },
            "metadata": {
                "source_dialog_id": "retail_dialog_7",
                "scenario_id": "scenario_7",
                "turn_index": 2,
                "correct": 1,
                "reward": 1.0,
            },
        }

        example = areal_sft.convert_areal_row(
            row,
            row_index=0,
            tools=sample_records()[0]["tools"],
            drop_unknown_tools=True,
            dataset_id="areal/example",
            require_correct=True,
            min_reward=1.0,
        )

        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example["metadata"]["sft_format"], "areal-retail-next-action")
        self.assertNotIn("thinking", json.dumps(example["messages"]))
        final_call = example["messages"][-1]["tool_calls"][0]
        self.assertEqual(final_call["function"]["name"], "cancel_pending_order")
        self.assertEqual(json.loads(final_call["function"]["arguments"]), {"order_id": "O-1001"})

    def test_areal_conversion_drops_unknown_answer_tool(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "# Retail agent policy\nHelp the customer."},
                {"role": "user", "content": "Delete my account."},
            ],
            "answer": {
                "role": "assistant",
                "tool_calls": [{"name": "delete_customer_account", "arguments": {}}],
            },
            "metadata": {"source_dialog_id": "retail_dialog_8", "correct": 1, "reward": 1.0},
        }

        example = areal_sft.convert_areal_row(
            row,
            row_index=0,
            tools=sample_records()[0]["tools"],
            drop_unknown_tools=True,
            dataset_id="areal/example",
            require_correct=True,
            min_reward=1.0,
        )

        self.assertIsNone(example)

    def test_success_trace_conversion_adds_system_and_tool_ids(self) -> None:
        row = {
            "task_id": "retail-7",
            "messages": [
                {"role": "assistant", "content": "Hi! How can I help you today?"},
                {"role": "user", "content": "Please cancel order O-1001."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "get_order_details", "arguments": {"order_id": "O-1001"}}],
                },
                {"role": "tool", "content": "{\"status\":\"pending\"}"},
                {"role": "assistant", "content": "I can cancel that pending order."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "cancel_pending_order", "arguments": {"order_id": "O-1001"}}],
                },
                {"role": "tool", "content": "{\"status\":\"canceled\"}"},
                {"role": "assistant", "content": "Done."},
            ],
            "canonical_reward": 1.0,
            "condition_flags": {"C_blind_strict": True},
            "memory_injected": False,
            "model": "Qwen/Qwen3.5-9B",
        }

        example = success_trace_sft.convert_trace_row(
            row,
            row_index=0,
            tools=sample_records()[0]["tools"],
            dataset_id="success/example",
            split="train",
            system_message="You are a retail support agent.",
            drop_unknown_tools=True,
            min_reward=1.0,
            require_blind_strict=True,
            allow_memory_injected=False,
        )

        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example["messages"][0]["role"], "system")
        first_call = example["messages"][3]["tool_calls"][0]
        first_tool = example["messages"][4]
        self.assertEqual(first_tool["tool_call_id"], first_call["id"])
        self.assertEqual(example["metadata"]["sft_format"], "tau2-retail-success-trace-full")

        next_actions = success_trace_sft.to_next_action_examples([example])
        self.assertGreaterEqual(len(next_actions), 2)
        self.assertTrue(all(row["metadata"]["sft_format"] == "tau2-retail-success-trace-next-action" for row in next_actions))

    def test_success_trace_conversion_filters_memory_injected_rows(self) -> None:
        row = {
            "task_id": "retail-memory",
            "messages": [
                {"role": "user", "content": "Please cancel order O-1001."},
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "cancel_pending_order", "arguments": {"order_id": "O-1001"}}],
                },
            ],
            "canonical_reward": 1.0,
            "condition_flags": {"C_blind_strict": True},
            "memory_injected": True,
        }

        example = success_trace_sft.convert_trace_row(
            row,
            row_index=0,
            tools=sample_records()[0]["tools"],
            dataset_id="success/example",
            split="train",
            system_message="You are a retail support agent.",
            drop_unknown_tools=True,
            min_reward=1.0,
            require_blind_strict=True,
            allow_memory_injected=False,
        )

        self.assertIsNone(example)

    def test_cached_weave_eval_rows_keep_horizontal_stage_metadata(self) -> None:
        dataset_rows, outputs = cached_weave_eval.build_cached_eval_rows(
            [
                {
                    "scenario_id": "retail-1",
                    "rollout_index": 0,
                    "reward": 0.75,
                    "metrics": {"outcome_success": 1.0},
                    "logs": {"duration": 3.5},
                }
            ],
            stage="grpo",
            model="LiquidAI/LFM2.5-8B-A1B",
            model_artifact="retail-support-agent-checkpoint:grpo",
        )

        self.assertEqual(len(dataset_rows), 1)
        row_id = dataset_rows[0]["row_id"]
        self.assertEqual(dataset_rows[0]["stage"], "grpo")
        self.assertEqual(dataset_rows[0]["model_artifact_path"], "retail-support-agent-checkpoint:grpo")
        self.assertEqual(outputs[row_id]["metrics"]["outcome_success"], 1.0)
        self.assertEqual(outputs[row_id]["metadata"]["model"], "LiquidAI/LFM2.5-8B-A1B")

    def test_checkpoint_selection_supports_historical_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RetailCourseConfig(project="proj", art_path=tmp, model_name="retail-model")
            root = Path(tmp) / "proj" / "models" / "retail-model" / "checkpoints"
            (root / "0001").mkdir(parents=True)
            (root / "0003").mkdir()

            self.assertEqual(latest_checkpoint_dir(cfg).name, "0003")
            self.assertEqual(select_checkpoint_dir(cfg, checkpoint_step=1).name, "0001")
            self.assertEqual(select_checkpoint_dir(cfg, checkpoint_path=root / "0003").name, "0003")
            with self.assertRaises(FileNotFoundError):
                select_checkpoint_dir(cfg, checkpoint_step=2)
            with self.assertRaises(ValueError):
                select_checkpoint_dir(cfg, checkpoint_step=1, checkpoint_path=root / "0003")

    def test_historical_checkpoint_aliases_do_not_move_latest_by_default(self) -> None:
        latest_aliases = checkpoint_artifact_aliases(stage="grpo", step=23, aliases=["grpo"], historical=False)
        historical_aliases = checkpoint_artifact_aliases(
            stage="grpo",
            step=23,
            aliases=["candidate"],
            historical=True,
        )

        self.assertEqual(latest_aliases, ["grpo", "step-0023", "latest"])
        self.assertEqual(historical_aliases, ["grpo-step-0023", "step-0023", "candidate"])
        self.assertNotIn("latest", historical_aliases)

    def test_stage_gate_rejects_rl_without_agentic_lift(self) -> None:
        criteria = stage_acceptance.Criteria()
        sft = {
            "stage": "sft_anchor",
            "reward": 0.10,
            "outcome_success": 0.10,
            "task_success": 0.10,
            "state_action_sequence_match": 0.20,
            "bad_state_action": 0.20,
            "missing_state_action": 0.80,
        }
        rl = {
            "stage": "grpo",
            "reward": 0.14,
            "outcome_success": 0.10,
            "task_success": 0.10,
            "state_action_sequence_match": 0.20,
            "bad_state_action": 0.20,
            "missing_state_action": 0.80,
        }

        decision = stage_acceptance.judge_rl(rl, sft, criteria)

        self.assertEqual(decision["decision"], "reject")
        self.assertIn("no outcome/task/state-action lift", decision["reason"])

    def test_stage_gate_accepts_rl_with_metric_and_error_lift(self) -> None:
        criteria = stage_acceptance.Criteria()
        sft = {
            "stage": "sft_anchor",
            "reward": -0.15,
            "outcome_success": 0.10,
            "task_success": 0.08,
            "state_action_sequence_match": 0.16,
            "bad_state_action": 0.20,
            "missing_state_action": 1.00,
        }
        rl = {
            "stage": "grpo",
            "reward": -0.09,
            "outcome_success": 0.16,
            "task_success": 0.14,
            "state_action_sequence_match": 0.22,
            "bad_state_action": 0.10,
            "missing_state_action": 0.90,
        }

        decision = stage_acceptance.judge_rl(rl, sft, criteria)

        self.assertEqual(decision["decision"], "accept")
        self.assertGreater(decision["deltas"]["reward"], 0.0)

    def test_sample_return_tool_is_state_changing(self) -> None:
        scenario = scenario_from_record(sample_records()[1], split="validation", index=0)
        messages = [
            {"role": "system", "content": scenario.system_message},
            {"role": "user", "content": scenario.user_message},
            {"role": "assistant", "content": scenario.expected_final_text},
        ]

        result = score_messages(messages, scenario, reward_profile="tau_sparse")

        self.assertTrue(is_state_changing_tool("return_delivered_order"))
        self.assertEqual(result.metrics["outcome_success"], 0.0)
        self.assertEqual(result.reward, 0.0)

    def test_tau_mode_does_not_replay_out_of_order_state_action(self) -> None:
        scenario = scenario_from_record(sample_records()[0], split="train", index=0)
        env = ReplayRetailEnv(scenario, terminate_on_invalid=False, strict_reference_actions=False)
        step = env.step(assistant_tool_call("cancel_pending_order", {"order_id": "O-1001"}))

        self.assertEqual(step.invalid_tool_calls, 1)
        self.assertEqual(step.invalid_state_mutations, 1)
        self.assertEqual(step.bad_state_actions, 1)
        self.assertIn("unexpected_state_action", step.tool_messages[0]["content"])

    def test_unknown_tool_is_invalid_even_in_tau_mode(self) -> None:
        scenario = scenario_from_record(sample_records()[0], split="train", index=0)
        env = ReplayRetailEnv(scenario, terminate_on_invalid=False, strict_reference_actions=False)
        step = env.step(assistant_tool_call("delete_customer_account", {"user_id": "U-1"}))

        self.assertEqual(step.invalid_tool_calls, 1)
        self.assertEqual(step.unknown_tool_calls, 1)
        self.assertIn("unknown_tool_call", step.tool_messages[0]["content"])

    def test_tau_irc_penalizes_bad_state_action_metric_directly(self) -> None:
        scenario = scenario_from_record(sample_records()[0], split="train", index=0)

        base = score_messages(
            scenario.reference_messages,
            scenario,
            invalid_state_mutations=1,
            reward_profile="tau_irc",
        )
        with_bad_state = score_messages(
            scenario.reference_messages,
            scenario,
            invalid_state_mutations=1,
            bad_state_actions=1,
            reward_profile="tau_irc",
        )

        self.assertLess(with_bad_state.reward, base.reward)

    def test_tau_irc_exposes_state_action_diagnostics(self) -> None:
        scenario = scenario_from_record(sample_records()[0], split="train", index=0)

        result = score_messages(scenario.reference_messages, scenario, reward_profile="tau_irc")

        self.assertEqual(result.metrics["outcome_success"], 1.0)
        self.assertGreater(result.metrics["state_action_expected_count"], 0.0)
        self.assertEqual(result.metrics["state_action_attempt_rate"], 1.0)
        self.assertEqual(result.metrics["valid_state_action_rate"], 1.0)
        self.assertIn("reward_component/outcome", result.metrics)
        self.assertIn("reward_component/penalty_bad_state", result.metrics)

    def test_truncated_full_reference_is_not_tau_success(self) -> None:
        scenario = scenario_from_record(sample_records()[0], split="train", index=0)

        complete = score_messages(scenario.reference_messages, scenario, reward_profile="tau_sparse")
        truncated = score_messages(
            scenario.reference_messages,
            scenario,
            truncated_by_max_turn=True,
            reward_profile="tau_sparse",
        )

        self.assertEqual(complete.metrics["outcome_success"], 1.0)
        self.assertEqual(truncated.metrics["outcome_success"], 0.0)
        self.assertEqual(truncated.reward, 0.0)

    def test_extra_read_only_lookup_does_not_make_state_action_missing(self) -> None:
        scenario = scenario_from_record(sample_records()[0], split="train", index=0)
        messages = list(scenario.reference_messages)
        extra_read = [
            assistant_tool_call("get_order_details", {"order_id": "O-1001"}),
            {"role": "tool", "tool_call_id": "call_test", "content": "{\"order_id\": \"O-1001\", \"status\": \"pending\"}"},
        ]
        messages = messages[:4] + extra_read + messages[4:]

        result = score_messages(messages, scenario, read_only_reference_mismatches=1, reward_profile="tau_irc")

        self.assertEqual(result.metrics["task_success"], 0.0)
        self.assertEqual(result.metrics["outcome_success"], 1.0)
        self.assertEqual(result.metrics["missing_state_action"], 0.0)

    def test_terminal_tool_call_does_not_inherit_previous_assistant_text(self) -> None:
        record = dict(sample_records()[0])
        record["messages"] = list(record["messages"][:-1])
        scenario = scenario_from_record(record, split="train", index=0)

        result = score_messages(scenario.reference_messages, scenario, reward_profile="tau_sparse")

        self.assertEqual(scenario.expected_final_text, "")
        self.assertEqual(result.metrics["communication_success"], 1.0)
        self.assertEqual(result.metrics["outcome_success"], 1.0)

    def test_reward_signal_metrics_include_winner_loser_diagnostics(self) -> None:
        group = SimpleNamespace(
            trajectories=[
                SimpleNamespace(
                    reward=-0.2,
                    metrics={
                        "outcome_success": 0.0,
                        "task_success": 0.0,
                        "state_action_sequence_match": 0.0,
                        "bad_state_action": 1.0,
                        "missing_state_action": 1.0,
                        "truncated_by_max_turn": 0.0,
                    },
                ),
                SimpleNamespace(
                    reward=1.0,
                    metrics={
                        "outcome_success": 1.0,
                        "task_success": 1.0,
                        "state_action_sequence_match": 1.0,
                        "bad_state_action": 0.0,
                        "missing_state_action": 0.0,
                        "truncated_by_max_turn": 0.0,
                    },
                ),
            ]
        )

        metrics = reward_signal_metrics([group], prefix="data/test")

        self.assertEqual(metrics["data/test_outcome_success_mixed_group_rate"], 1.0)
        self.assertEqual(metrics["data/test_winner_minus_loser_outcome_success"], 1.0)
        self.assertEqual(metrics["data/test_winner_minus_loser_bad_state_action"], -1.0)
        self.assertEqual(metrics["data/test_all_equal_reward_group_rate"], 0.0)
        self.assertEqual(metrics["data/test_all_outcome_success_group_rate"], 0.0)
        self.assertEqual(metrics["data/test_all_outcome_failure_group_rate"], 0.0)

    def test_reward_signal_metrics_classify_no_signal_failure_groups(self) -> None:
        group = SimpleNamespace(
            trajectories=[
                SimpleNamespace(
                    reward=-0.4,
                    metrics={
                        "outcome_success": 0.0,
                        "task_success": 0.0,
                        "invalid_tool_call": 1.0,
                        "missing_state_action": 1.0,
                        "state_action_reached_rate": 0.0,
                        "truncated_by_max_turn": 1.0,
                    },
                ),
                SimpleNamespace(
                    reward=-0.4,
                    metrics={
                        "outcome_success": 0.0,
                        "task_success": 0.0,
                        "invalid_tool_call": 2.0,
                        "missing_state_action": 1.0,
                        "state_action_reached_rate": 0.0,
                        "truncated_by_max_turn": 1.0,
                    },
                ),
            ]
        )

        metrics = reward_signal_metrics([group], prefix="data/drop")

        self.assertEqual(metrics["data/drop_all_equal_reward_group_rate"], 1.0)
        self.assertEqual(metrics["data/drop_all_outcome_failure_group_rate"], 1.0)
        self.assertEqual(metrics["data/drop_all_truncated_group_rate"], 1.0)
        self.assertEqual(metrics["data/drop_any_truncated_group_rate"], 1.0)
        self.assertEqual(metrics["data/drop_all_invalid_tool_group_rate"], 1.0)
        self.assertEqual(metrics["data/drop_all_missing_state_action_group_rate"], 1.0)
        self.assertEqual(metrics["data/drop_all_no_state_action_reached_group_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
