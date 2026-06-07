from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any

from course.shared.tracing import weave_op


def _metric(output: dict[str, Any], name: str, default: float = 0.0) -> float:
    metrics = output.get("metrics") or {}
    try:
        return float(metrics.get(name, default))
    except Exception:
        return default


@weave_op("retail_success_scorer")
def retail_success_scorer(output: dict[str, Any], **_: Any) -> dict[str, float]:
    return {
        "retail_task_success": _metric(
            output,
            "retail_task_success",
            _metric(output, "proxy_tau2_success", _metric(output, "outcome_success")),
        ),
        "reference_tool_sequence_exact_match": _metric(
            output,
            "reference_tool_sequence_exact_match",
            _metric(output, "task_success"),
        ),
        "communication_success": _metric(output, "communication_success"),
    }


@weave_op("tool_quality_scorer")
def tool_quality_scorer(output: dict[str, Any], **_: Any) -> dict[str, float]:
    return {
        "tool_sequence_success": _metric(output, "tool_sequence_success"),
        "tool_name_f1": _metric(output, "tool_name_f1"),
        "tool_order_match": _metric(output, "tool_order_match"),
        "tool_argument_match": _metric(output, "tool_argument_match"),
        "tool_call_exact_match": _metric(output, "tool_call_exact_match"),
        "state_action_match": _metric(output, "state_action_match"),
        "state_action_args": _metric(output, "state_action_args"),
        "state_action_sequence_match": _metric(output, "state_action_sequence_match"),
        "invalid_tool_call": _metric(output, "invalid_tool_call"),
        "invalid_state_mutation": _metric(output, "invalid_state_mutation"),
        "read_only_reference_mismatch": _metric(output, "read_only_reference_mismatch"),
        "unknown_tool_call": _metric(output, "unknown_tool_call"),
        "bad_read_only_call": _metric(output, "bad_read_only_call"),
        "bad_state_action": _metric(output, "bad_state_action"),
        "missing_state_action": _metric(output, "missing_state_action"),
        "truncated_by_max_turn": _metric(output, "truncated_by_max_turn"),
        "terminated_on_invalid": _metric(output, "terminated_on_invalid"),
    }


@weave_op("agentic_signal_scorer")
def agentic_signal_scorer(output: dict[str, Any], **_: Any) -> dict[str, float]:
    return {
        "state_action_expected_count": _metric(output, "state_action_expected_count"),
        "state_action_actual_count": _metric(output, "state_action_actual_count"),
        "state_action_attempt_rate": _metric(output, "state_action_attempt_rate"),
        "state_action_reached_rate": _metric(output, "state_action_reached_rate"),
        "valid_state_action_rate": _metric(output, "valid_state_action_rate"),
        "accepted_state_action_jump": _metric(output, "accepted_state_action_jump"),
        "accepted_state_action_jump_count": _metric(output, "accepted_state_action_jump_count"),
        "first_accepted_state_action_jump_turn": _metric(output, "first_accepted_state_action_jump_turn"),
        "skipped_reference_turns_before_state_action": _metric(
            output, "skipped_reference_turns_before_state_action"
        ),
        "last_accepted_reference_state_output_index": _metric(
            output, "last_accepted_reference_state_output_index"
        ),
        "last_accepted_reference_state_turn_index": _metric(output, "last_accepted_reference_state_turn_index"),
        "reward_component/outcome": _metric(output, "reward_component/outcome"),
        "reward_component/state_action": _metric(output, "reward_component/state_action"),
        "reward_component/state_action_args": _metric(output, "reward_component/state_action_args"),
        "reward_component/communication": _metric(output, "reward_component/communication"),
        "reward_component/penalty_invalid_state": _metric(output, "reward_component/penalty_invalid_state"),
        "reward_component/penalty_bad_state": _metric(output, "reward_component/penalty_bad_state"),
        "reward_component/penalty_read_only": _metric(output, "reward_component/penalty_read_only"),
        "reward_component/penalty_missing_state": _metric(output, "reward_component/penalty_missing_state"),
    }


@weave_op("response_quality_scorer")
def response_quality_scorer(output: dict[str, Any], **_: Any) -> dict[str, float]:
    return {
        "final_text_f1": _metric(output, "final_text_f1"),
        "has_final_response": _metric(output, "has_final_response"),
        "turn_count": _metric(output, "turn_count"),
    }


SCORERS = [retail_success_scorer, tool_quality_scorer, agentic_signal_scorer, response_quality_scorer]
