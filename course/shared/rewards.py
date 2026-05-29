from __future__ import annotations

import math
import re
from typing import Any

from .retail_env import choice_to_message, extract_tool_names_from_messages
from .schemas import RetailScenario, RewardResult

_WORD_RE = re.compile(r"[a-z0-9_]+")


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


def final_assistant_text(messages: list[Any]) -> str:
    for item in reversed(messages):
        msg = choice_to_message(item) if hasattr(item, "message") else item
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def score_messages(
    messages: list[Any],
    scenario: RetailScenario,
    *,
    invalid_tool_calls: int = 0,
    turn_count: int | None = None,
) -> RewardResult:
    actual_tools = extract_tool_names_from_messages(messages)
    expected_tools = scenario.expected_tool_names
    f1 = tool_f1(actual_tools, expected_tools)
    order = sequence_prefix_score(actual_tools, expected_tools)
    final_text = final_assistant_text(messages)
    final_quality = token_f1(final_text, scenario.expected_final_text)
    has_final_response = 1.0 if final_text.strip() else 0.0
    invalid_penalty = min(0.5, invalid_tool_calls * 0.15)
    turns = turn_count if turn_count is not None else max(1, len([m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]))
    turn_penalty = min(0.2, max(0, turns - scenario.max_turns) * 0.03)

    task_success = 1.0 if f1 == 1.0 and order == 1.0 and invalid_tool_calls == 0 else 0.0
    reward = 0.45 * f1 + 0.25 * order + 0.20 * final_quality + 0.10 * has_final_response
    reward = max(-1.0, min(1.0, reward - invalid_penalty - turn_penalty))

    metrics = {
        "task_success": task_success,
        "tool_name_f1": f1,
        "tool_order_match": order,
        "final_text_f1": final_quality,
        "has_final_response": has_final_response,
        "invalid_tool_call": float(invalid_tool_calls),
        "turn_count": float(turns),
    }
    explanation = (
        f"tools actual={actual_tools} expected={expected_tools}; "
        f"tool_f1={f1:.3f}; order={order:.3f}; final_text_f1={final_quality:.3f}; "
        f"invalid={invalid_tool_calls}"
    )
    return RewardResult(reward=float(reward), metrics=metrics, explanation=explanation)


def score_trajectory(trajectory: Any, scenario: RetailScenario, *, invalid_tool_calls: int = 0) -> RewardResult:
    return score_messages(trajectory.messages_and_choices, scenario, invalid_tool_calls=invalid_tool_calls)


def mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)
