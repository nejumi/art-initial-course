from __future__ import annotations

import math
import os
import re
from typing import Any

from .retail_env import (
    STATE_CHANGING_TOOL_NAMES,
    choice_to_message,
    extract_tool_calls_from_messages,
    extract_tool_names_from_messages,
    is_state_changing_tool,
    parse_reference_turns,
    tool_argument_match_score,
    tool_call_matches,
    tool_call_name,
)
from .schemas import RetailScenario, RewardResult

_WORD_RE = re.compile(r"[a-z0-9_]+")
REWARD_PROFILES = (
    "dense",
    "strict_success",
    "agentic",
    "tau_sparse",
    "tau_irc",
    "tau_irc_outcome",
    "tau_irc_balanced",
    "tau_irc_guarded",
)
TAU_STYLE_REWARD_PROFILES = {
    "tau_sparse",
    "tau_irc",
    "tau_irc_outcome",
    "tau_irc_balanced",
    "tau_irc_guarded",
}


def normalize_reward_profile(profile: str | None = None) -> str:
    value = (profile or os.getenv("RETAIL_REWARD_PROFILE") or "dense").strip().lower().replace("-", "_")
    if not value:
        value = "dense"
    if value not in REWARD_PROFILES:
        allowed = ", ".join(REWARD_PROFILES)
        raise ValueError(f"Unknown RETAIL_REWARD_PROFILE={value!r}. Use one of: {allowed}.")
    return value


def token_set(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def token_f1(candidate: str, reference: str) -> float:
    cand = token_set(candidate)
    ref = token_set(reference)
    if not cand and not ref:
        return 1.0
    if not cand or not ref:
        return 0.0
    overlap = len(cand & ref)
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_prefix_score(actual: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0 if not actual else 0.5
    matches = 0
    for got, want in zip(actual, expected):
        if got == want:
            matches += 1
        else:
            break
    return matches / len(expected)


def tool_f1(actual: list[str], expected: list[str]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    remaining = list(expected)
    hits = 0
    for name in actual:
        if name in remaining:
            hits += 1
            remaining.remove(name)
    precision = hits / len(actual)
    recall = hits / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def expected_tool_calls(scenario: RetailScenario) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for turn in parse_reference_turns(scenario.reference_messages):
        calls.extend(dict(call) for call in turn.assistant_message.get("tool_calls") or [])
    return calls


def tool_argument_sequence_score(actual_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> float:
    if not expected_calls and not actual_calls:
        return 1.0
    if not expected_calls or not actual_calls:
        return 0.0

    score = 0.0
    for index, expected_call in enumerate(expected_calls):
        if index >= len(actual_calls):
            continue
        actual_call = actual_calls[index]
        if tool_call_name(actual_call) != tool_call_name(expected_call):
            continue
        score += tool_argument_match_score(actual_call, expected_call)
    return score / max(len(expected_calls), len(actual_calls))


def tool_call_exact_sequence_match(actual_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> float:
    if len(actual_calls) != len(expected_calls):
        return 0.0
    if not expected_calls:
        return 1.0
    return 1.0 if all(tool_call_matches(actual, expected) for actual, expected in zip(actual_calls, expected_calls)) else 0.0


def state_changing_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for call in calls if is_state_changing_tool(tool_call_name(call))]


def state_action_match_score(actual_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> float:
    expected_state_calls = state_changing_calls(expected_calls)
    actual_state_calls = state_changing_calls(actual_calls)
    if not expected_state_calls:
        return 1.0 if not actual_state_calls else 0.0
    if not actual_state_calls:
        return 0.0

    exact_matches = 0
    remaining = list(actual_state_calls)
    for expected_call in expected_state_calls:
        for actual_call in list(remaining):
            if tool_call_matches(actual_call, expected_call):
                exact_matches += 1
                remaining.remove(actual_call)
                break
    return exact_matches / max(len(expected_state_calls), len(actual_state_calls))


def state_action_sequence_match(actual_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> float:
    expected_state_calls = state_changing_calls(expected_calls)
    actual_state_calls = state_changing_calls(actual_calls)
    if len(actual_state_calls) != len(expected_state_calls):
        return 0.0
    if not expected_state_calls:
        return 1.0
    return 1.0 if all(tool_call_matches(actual, expected) for actual, expected in zip(actual_state_calls, expected_state_calls)) else 0.0


def state_action_argument_score(actual_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> float:
    return tool_argument_sequence_score(state_changing_calls(actual_calls), state_changing_calls(expected_calls))


def missing_state_action_count(actual_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> int:
    expected_state_calls = state_changing_calls(expected_calls)
    actual_state_calls = state_changing_calls(actual_calls)
    missing = 0
    remaining = list(actual_state_calls)
    for expected_call in expected_state_calls:
        matched = False
        for actual_call in list(remaining):
            if tool_call_matches(actual_call, expected_call):
                remaining.remove(actual_call)
                matched = True
                break
        if not matched:
            missing += 1
    return missing


def final_assistant_text(messages: list[Any]) -> str:
    for item in reversed(messages):
        msg = choice_to_message(item)
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def assistant_turn_count(messages: list[Any]) -> int:
    turns = 0
    for item in messages:
        msg = choice_to_message(item)
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            turns += 1
    return max(1, turns)


def score_messages(
    messages: list[Any],
    scenario: RetailScenario,
    *,
    invalid_tool_calls: int = 0,
    invalid_state_mutations: int = 0,
    read_only_reference_mismatches: int = 0,
    unknown_tool_calls: int = 0,
    bad_read_only_calls: int = 0,
    bad_state_actions: int = 0,
    missing_state_actions: int = 0,
    truncated_by_max_turn: bool = False,
    turn_count: int | None = None,
    reward_profile: str | None = None,
) -> RewardResult:
    profile = normalize_reward_profile(reward_profile)
    actual_calls = extract_tool_calls_from_messages(messages)
    expected_calls = expected_tool_calls(scenario)
    actual_state_calls = state_changing_calls(actual_calls)
    expected_state_calls = state_changing_calls(expected_calls)
    actual_state_count = len(actual_state_calls)
    expected_state_count = len(expected_state_calls)
    actual_tools = [tool_call_name(call) for call in actual_calls if tool_call_name(call)]
    expected_tools = [tool_call_name(call) for call in expected_calls if tool_call_name(call)]
    f1 = tool_f1(actual_tools, expected_tools)
    order = sequence_prefix_score(actual_tools, expected_tools)
    argument_score = tool_argument_sequence_score(actual_calls, expected_calls)
    exact_tool_sequence = tool_call_exact_sequence_match(actual_calls, expected_calls)
    state_action_score = state_action_match_score(actual_calls, expected_calls)
    state_action_args = state_action_argument_score(actual_calls, expected_calls)
    state_action_sequence = state_action_sequence_match(actual_calls, expected_calls)
    missing_state_actions = max(missing_state_actions, missing_state_action_count(actual_calls, expected_calls))
    final_text = final_assistant_text(messages)
    final_quality = token_f1(final_text, scenario.expected_final_text)
    has_final_response = 1.0 if final_text.strip() else 0.0
    communication_success = 1.0 if not scenario.expected_final_text.strip() else float(final_quality >= 0.15)
    turns = turn_count if turn_count is not None else assistant_turn_count(messages)
    turn_penalty = min(0.2, max(0, turns - scenario.max_turns) * 0.03)

    tool_sequence_success = 1.0 if f1 == 1.0 and order == 1.0 and invalid_tool_calls == 0 else 0.0
    task_success = 1.0 if exact_tool_sequence == 1.0 and invalid_tool_calls == 0 else 0.0
    state_action_attempt_rate = (
        1.0
        if expected_state_count == 0 and actual_state_count == 0
        else min(1.0, actual_state_count / expected_state_count)
        if expected_state_count
        else 0.0
    )
    state_action_reached = 1.0 if actual_state_count > 0 or expected_state_count == 0 else 0.0
    valid_state_action_rate = state_action_score
    outcome_success = (
        1.0
        if state_action_sequence == 1.0
        and communication_success == 1.0
        and invalid_tool_calls == 0
        and invalid_state_mutations == 0
        and bad_state_actions == 0
        and missing_state_actions == 0
        and not truncated_by_max_turn
        else 0.0
    )
    reward_components: dict[str, float] = {}
    if profile == "dense":
        invalid_penalty = min(0.5, invalid_tool_calls * 0.15)
        reward_components = {
            "reward_component/tool_name": 0.35 * f1,
            "reward_component/tool_order": 0.20 * order,
            "reward_component/tool_args": 0.20 * argument_score,
            "reward_component/final_text": 0.15 * final_quality,
            "reward_component/final_response": 0.10 * has_final_response,
            "reward_component/penalty_invalid": -invalid_penalty,
            "reward_component/penalty_turns": -turn_penalty,
        }
        reward = sum(reward_components.values())
    elif profile == "strict_success":
        invalid_penalty = min(0.75, invalid_tool_calls * 0.25)
        reward_components = {
            "reward_component/tool_name": 0.25 * f1,
            "reward_component/tool_order": 0.20 * order,
            "reward_component/tool_args": 0.20 * argument_score,
            "reward_component/final_text": 0.10 * final_quality,
            "reward_component/final_response": 0.05 * has_final_response,
            "reward_component/task": 0.30 * task_success,
            "reward_component/penalty_invalid": -invalid_penalty,
            "reward_component/penalty_turns": -turn_penalty,
        }
        reward = sum(reward_components.values())
    elif profile == "agentic":
        invalid_penalty = min(0.9, invalid_tool_calls * 0.35)
        reward_components = {
            "reward_component/tool_name": 0.15 * f1,
            "reward_component/tool_order": 0.15 * order,
            "reward_component/tool_args": 0.25 * argument_score,
            "reward_component/state_action": 0.25 * state_action_score,
            "reward_component/task": 0.15 * task_success,
            "reward_component/final_text": 0.05 * final_quality,
            "reward_component/final_response": 0.05 * has_final_response,
            "reward_component/penalty_invalid": -invalid_penalty,
            "reward_component/penalty_turns": -turn_penalty,
        }
        reward = sum(reward_components.values())
    elif profile == "tau_sparse":
        reward_components = {"reward_component/outcome": outcome_success}
        reward = sum(reward_components.values())
    elif profile in {"tau_irc", "tau_irc_outcome", "tau_irc_balanced", "tau_irc_guarded"}:
        # Tau-style calibrated shaping: reward the final verifiable outcome,
        # give soft credit only to state-changing actions, and keep read-only
        # lookups as diagnostics instead of positive reward drivers.
        if profile == "tau_irc":
            outcome_weight = 0.65
            state_action_weight = 0.20
            state_action_args_weight = 0.10
            state_action_sequence_weight = 0.00
            state_action_reached_weight = 0.00
            communication_weight = 0.05
            invalid_state_penalty = min(0.8, invalid_state_mutations * 0.4)
            bad_state_penalty = min(0.4, bad_state_actions * 0.2)
            unnecessary_read_penalty = min(0.15, read_only_reference_mismatches * 0.03)
            unknown_tool_penalty = min(0.5, unknown_tool_calls * 0.25)
            missing_state_penalty = min(0.4, missing_state_actions * 0.2)
            truncation_penalty = 0.0
        elif profile == "tau_irc_outcome":
            outcome_weight = 0.75
            state_action_weight = 0.10
            state_action_args_weight = 0.05
            state_action_sequence_weight = 0.00
            state_action_reached_weight = 0.00
            communication_weight = 0.10
            invalid_state_penalty = min(0.8, invalid_state_mutations * 0.4)
            bad_state_penalty = min(0.4, bad_state_actions * 0.2)
            unnecessary_read_penalty = min(0.15, read_only_reference_mismatches * 0.03)
            unknown_tool_penalty = min(0.5, unknown_tool_calls * 0.25)
            missing_state_penalty = min(0.35, missing_state_actions * 0.175)
            truncation_penalty = 0.10 if truncated_by_max_turn else 0.0
        elif profile == "tau_irc_balanced":
            outcome_weight = 0.55
            state_action_weight = 0.20
            state_action_args_weight = 0.10
            state_action_sequence_weight = 0.10
            state_action_reached_weight = 0.00
            communication_weight = 0.05
            invalid_state_penalty = min(0.9, invalid_state_mutations * 0.45)
            bad_state_penalty = min(0.6, bad_state_actions * 0.3)
            unnecessary_read_penalty = min(0.15, read_only_reference_mismatches * 0.03)
            unknown_tool_penalty = min(0.6, unknown_tool_calls * 0.3)
            missing_state_penalty = min(0.5, missing_state_actions * 0.25)
            truncation_penalty = 0.15 if truncated_by_max_turn else 0.0
        else:
            outcome_weight = 0.45
            state_action_weight = 0.20
            state_action_args_weight = 0.10
            state_action_sequence_weight = 0.10
            state_action_reached_weight = 0.05
            communication_weight = 0.10
            invalid_state_penalty = min(1.0, invalid_state_mutations * 0.5)
            bad_state_penalty = min(1.0, bad_state_actions * 0.5)
            unnecessary_read_penalty = min(0.2, read_only_reference_mismatches * 0.04)
            unknown_tool_penalty = min(0.6, unknown_tool_calls * 0.3)
            missing_state_penalty = min(0.6, missing_state_actions * 0.3)
            truncation_penalty = 0.20 if truncated_by_max_turn else 0.0
        reward_components = {
            "reward_component/outcome": outcome_weight * outcome_success,
            "reward_component/state_action": state_action_weight * state_action_score,
            "reward_component/state_action_args": state_action_args_weight * state_action_args,
            "reward_component/state_action_sequence": state_action_sequence_weight * state_action_sequence,
            "reward_component/state_action_reached": state_action_reached_weight * state_action_reached,
            "reward_component/communication": communication_weight * communication_success,
            "reward_component/penalty_invalid_state": -invalid_state_penalty,
            "reward_component/penalty_bad_state": -bad_state_penalty,
            "reward_component/penalty_read_only": -unnecessary_read_penalty,
            "reward_component/penalty_unknown_tool": -unknown_tool_penalty,
            "reward_component/penalty_missing_state": -missing_state_penalty,
            "reward_component/penalty_truncation": -truncation_penalty,
            "reward_component/penalty_turns": -turn_penalty,
        }
        reward = sum(reward_components.values())
    reward = max(-1.0, min(1.0, reward))

    metrics = {
        "course_eval_score": (
            outcome_success
            + 0.5 * task_success
            + 0.5 * state_action_sequence
            + 0.25 * valid_state_action_rate
            + 0.10 * communication_success
            - 0.70 * float(bad_state_actions)
            - 0.35 * float(missing_state_actions)
            - 0.30 * (1.0 if truncated_by_max_turn else 0.0)
        ),
        "task_success": task_success,
        "outcome_success": outcome_success,
        "proxy_outcome_success": outcome_success,
        "tool_sequence_success": tool_sequence_success,
        "tool_name_f1": f1,
        "tool_order_match": order,
        "tool_argument_match": argument_score,
        "tool_call_exact_match": exact_tool_sequence,
        "state_action_match": state_action_score,
        "state_action_args": state_action_args,
        "state_action_sequence_match": state_action_sequence,
        "state_action_expected_count": float(expected_state_count),
        "state_action_actual_count": float(actual_state_count),
        "state_action_attempt_rate": state_action_attempt_rate,
        "state_action_reached_rate": state_action_reached,
        "valid_state_action_rate": valid_state_action_rate,
        "communication_success": communication_success,
        "final_text_f1": final_quality,
        "has_final_response": has_final_response,
        "invalid_tool_call": float(invalid_tool_calls),
        "invalid_state_mutation": float(invalid_state_mutations),
        "read_only_reference_mismatch": float(read_only_reference_mismatches),
        "unknown_tool_call": float(unknown_tool_calls),
        "bad_read_only_call": float(bad_read_only_calls),
        "bad_state_action": float(bad_state_actions),
        "missing_state_action": float(missing_state_actions),
        "truncated_by_max_turn": 1.0 if truncated_by_max_turn else 0.0,
        "turn_count": float(turns),
    }
    metrics.update(reward_components)
    explanation = (
        f"profile={profile}; tools actual={actual_tools} expected={expected_tools}; "
        f"tool_f1={f1:.3f}; order={order:.3f}; args={argument_score:.3f}; "
        f"state_action={state_action_score:.3f}; state_sequence={state_action_sequence:.3f}; "
        f"communication={communication_success:.3f}; final_text_f1={final_quality:.3f}; "
        f"invalid={invalid_tool_calls}; invalid_state={invalid_state_mutations}; "
        f"unknown_tools={unknown_tool_calls}; bad_state={bad_state_actions}; "
        f"missing_state={missing_state_actions}; truncated={truncated_by_max_turn}"
    )
    return RewardResult(reward=float(reward), metrics=metrics, explanation=explanation)


def score_trajectory(
    trajectory: Any,
    scenario: RetailScenario,
    *,
    invalid_tool_calls: int = 0,
    invalid_state_mutations: int = 0,
    read_only_reference_mismatches: int = 0,
    unknown_tool_calls: int = 0,
    bad_read_only_calls: int = 0,
    bad_state_actions: int = 0,
    missing_state_actions: int = 0,
    truncated_by_max_turn: bool = False,
    turn_count: int | None = None,
    reward_profile: str | None = None,
) -> RewardResult:
    return score_messages(
        trajectory.messages_and_choices,
        scenario,
        invalid_tool_calls=invalid_tool_calls,
        invalid_state_mutations=invalid_state_mutations,
        read_only_reference_mismatches=read_only_reference_mismatches,
        unknown_tool_calls=unknown_tool_calls,
        bad_read_only_calls=bad_read_only_calls,
        bad_state_actions=bad_state_actions,
        missing_state_actions=missing_state_actions,
        truncated_by_max_turn=truncated_by_max_turn,
        turn_count=turn_count,
        reward_profile=reward_profile,
    )


def mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)
