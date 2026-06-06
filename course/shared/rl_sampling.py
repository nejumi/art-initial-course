from __future__ import annotations

import math
import statistics
from typing import Any


def reward_values(group: Any) -> list[float]:
    return [float(getattr(trajectory, "reward", 0.0)) for trajectory in getattr(group, "trajectories", [])]


def has_reward_signal(group: Any, *, tolerance: float = 1e-9) -> bool:
    rewards = reward_values(group)
    if len(rewards) < 2:
        return False
    return max(rewards) - min(rewards) > tolerance


def split_reward_signal_groups(
    groups: list[Any],
    *,
    tolerance: float = 1e-9,
) -> tuple[list[Any], list[Any]]:
    trainable: list[Any] = []
    no_signal: list[Any] = []
    for group in groups:
        if has_reward_signal(group, tolerance=tolerance):
            trainable.append(group)
        else:
            no_signal.append(group)
    return trainable, no_signal


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else math.nan
    return statistics.fmean((value - statistics.fmean(values)) ** 2 for value in values) ** 0.5


def metric_values(groups: list[Any], key: str) -> list[float]:
    values: list[float] = []
    for group in groups:
        for trajectory in getattr(group, "trajectories", []):
            try:
                values.append(float(getattr(trajectory, "metrics", {}).get(key)))
            except Exception:
                continue
    return values


def _trajectory_metric(trajectory: Any, key: str) -> float | None:
    if key == "reward":
        try:
            return float(getattr(trajectory, "reward", 0.0))
        except Exception:
            return None
    try:
        return float(getattr(trajectory, "metrics", {}).get(key))
    except Exception:
        return None


def _winner_loser_deltas(groups: list[Any], key: str) -> list[float]:
    deltas: list[float] = []
    for group in groups:
        trajectories = list(getattr(group, "trajectories", []))
        if len(trajectories) < 2:
            continue
        ranked = sorted(trajectories, key=lambda trajectory: float(getattr(trajectory, "reward", 0.0)))
        loser = ranked[0]
        winner = ranked[-1]
        winner_value = _trajectory_metric(winner, key)
        loser_value = _trajectory_metric(loser, key)
        if winner_value is not None and loser_value is not None:
            deltas.append(winner_value - loser_value)
    return deltas


def _mixed_success_rate(groups: list[Any], key: str = "outcome_success") -> float:
    mixed = 0
    eligible = 0
    for group in groups:
        values = [
            value
            for trajectory in getattr(group, "trajectories", [])
            if (value := _trajectory_metric(trajectory, key)) is not None
        ]
        if len(values) < 2:
            continue
        eligible += 1
        if min(values) < 1.0 <= max(values):
            mixed += 1
    return mixed / eligible if eligible else math.nan


def group_diagnostic_metrics(groups: list[Any], *, prefix: str = "data/step") -> dict[str, float]:
    ranges = []
    stds = []
    for group in groups:
        rewards = reward_values(group)
        if len(rewards) >= 2:
            ranges.append(max(rewards) - min(rewards))
            stds.append(sample_std(rewards))
    metrics = {
        f"{prefix}_reward_range_mean": mean(ranges),
        f"{prefix}_reward_std_mean": mean(stds),
        f"{prefix}_outcome_success_mixed_group_rate": _mixed_success_rate(groups, "outcome_success"),
        f"{prefix}_task_success_mixed_group_rate": _mixed_success_rate(groups, "task_success"),
    }
    for key in [
        "outcome_success",
        "task_success",
        "state_action_match",
        "state_action_sequence_match",
        "state_action_attempt_rate",
        "valid_state_action_rate",
        "communication_success",
        "bad_state_action",
        "missing_state_action",
        "truncated_by_max_turn",
    ]:
        metrics[f"{prefix}_winner_minus_loser_{key}"] = mean(_winner_loser_deltas(groups, key))
    return {key: value for key, value in metrics.items() if not math.isnan(value)}


def reward_signal_metrics(groups: list[Any], *, prefix: str = "data/step") -> dict[str, float]:
    rewards = [reward for group in groups for reward in reward_values(group)]
    ranges = [
        max(group_rewards) - min(group_rewards)
        for group in groups
        if len(group_rewards := reward_values(group)) >= 2
    ]
    metrics = {
        f"{prefix}_reward_mean": mean(rewards),
        f"{prefix}_reward_min": min(rewards) if rewards else math.nan,
        f"{prefix}_reward_max": max(rewards) if rewards else math.nan,
        f"{prefix}_reward_range_mean": mean(ranges),
        f"{prefix}_reward_std_mean": mean(
            [sample_std(group_rewards) for group in groups if len(group_rewards := reward_values(group)) >= 2]
        ),
        f"{prefix}_outcome_success_mean": mean(metric_values(groups, "outcome_success")),
        f"{prefix}_task_success_mean": mean(metric_values(groups, "task_success")),
        f"{prefix}_state_action_reached_rate_mean": mean(metric_values(groups, "state_action_reached_rate")),
        f"{prefix}_state_action_attempt_rate_mean": mean(metric_values(groups, "state_action_attempt_rate")),
        f"{prefix}_valid_state_action_rate_mean": mean(metric_values(groups, "valid_state_action_rate")),
        f"{prefix}_invalid_tool_call_mean": mean(metric_values(groups, "invalid_tool_call")),
        f"{prefix}_invalid_state_mutation_mean": mean(metric_values(groups, "invalid_state_mutation")),
        f"{prefix}_unknown_tool_call_mean": mean(metric_values(groups, "unknown_tool_call")),
        f"{prefix}_bad_state_action_mean": mean(metric_values(groups, "bad_state_action")),
        f"{prefix}_missing_state_action_mean": mean(metric_values(groups, "missing_state_action")),
        f"{prefix}_truncated_by_max_turn_mean": mean(metric_values(groups, "truncated_by_max_turn")),
        f"{prefix}_terminated_on_invalid_mean": mean(metric_values(groups, "terminated_on_invalid")),
    }
    metrics.update(group_diagnostic_metrics(groups, prefix=prefix))
    return {key: value for key, value in metrics.items() if not math.isnan(value)}
