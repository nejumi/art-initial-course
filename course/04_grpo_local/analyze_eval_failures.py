from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from course.shared.data import load_cached_split, read_jsonl, scenarios_from_records
from course.shared.retail_env import (
    STATE_CHANGING_TOOL_NAMES,
    extract_tool_calls_from_messages,
    tool_argument_match_score,
    tool_call_name,
)
from course.shared.rewards import expected_tool_calls


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def metric_value(row: dict[str, object], key: str) -> float | None:
    metrics = row.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None
    if key not in metrics:
        return None
    try:
        return float(metrics[key])
    except Exception:
        return None


def metric(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = metric_value(row, key)
    return default if value is None else value


def format_values(values: list[float]) -> str:
    return "n/a" if not values else f"{mean(values):.4f}"


def scenario_lookup(data_dir: Path, split: str) -> dict[str, object]:
    lookup: dict[str, object] = {}
    records = load_cached_split(data_dir, split)
    for scenario in scenarios_from_records(records, split=split):
        lookup[scenario.id] = scenario
        metadata = scenario.metadata or {}
        for key in ("record_id", "id", "task_id"):
            value = metadata.get(key)
            if value is not None:
                lookup[str(value)] = scenario
        task_id = metadata.get("task_id")
        trial = metadata.get("trial")
        if task_id is not None and trial is not None:
            lookup[f"{task_id}-{trial}"] = scenario
    return lookup


def summarize(path: Path, *, data_dir: Path, split: str) -> None:
    rows = read_jsonl(path)
    scenarios = scenario_lookup(data_dir, split)
    print(f"file={path}")
    print(f"rows={len(rows)} split={split}")
    for key in [
        "reward",
        "outcome_success",
        "proxy_outcome_success",
        "task_success",
        "communication_success",
        "tool_sequence_success",
        "tool_name_f1",
        "tool_order_match",
        "tool_argument_match",
        "tool_call_exact_match",
        "state_action_match",
        "state_action_args",
        "state_action_sequence_match",
        "invalid_tool_call",
        "invalid_state_mutation",
        "read_only_reference_mismatch",
        "unknown_tool_call",
        "bad_read_only_call",
        "bad_state_action",
        "missing_state_action",
        "truncated_by_max_turn",
        "terminated_on_invalid",
        "turn_count",
    ]:
        if key == "reward":
            values = []
            for row in rows:
                try:
                    values.append(float(row.get("reward")))
                except Exception:
                    pass
        else:
            values = [
                value
                for row in rows
                if (value := metric_value(row, key)) is not None
            ]
        print(f"{key}={format_values(values)}")

    if not rows or "messages" not in rows[0]:
        print("messages=absent; rerun eval with --include-messages for detailed failure modes")
        return

    wrong_name = Counter()
    wrong_args = Counter()
    missing = Counter()
    extra = Counter()
    first_failure = Counter()
    first_state_failure = Counter()
    expected_state_actions = Counter()
    actual_state_actions = Counter()
    outcome_by_expected_state_count: dict[int, list[float]] = defaultdict(list)
    task_by_expected_state_count: dict[int, list[float]] = defaultdict(list)
    successes = 0
    for row in rows:
        scenario = scenarios.get(str(row.get("scenario_id")))
        if scenario is None:
            continue
        actual_calls = extract_tool_calls_from_messages(row.get("messages") or [])
        expected_calls = expected_tool_calls(scenario)
        expected_state_count = 0
        actual_state_count = 0
        for call in expected_calls:
            name = tool_call_name(call)
            if name in STATE_CHANGING_TOOL_NAMES:
                expected_state_count += 1
                expected_state_actions[name] += 1
        for call in actual_calls:
            name = tool_call_name(call)
            if name in STATE_CHANGING_TOOL_NAMES:
                actual_state_count += 1
                actual_state_actions[name] += 1
        outcome_by_expected_state_count[expected_state_count].append(metric(row, "outcome_success"))
        task_by_expected_state_count[expected_state_count].append(metric(row, "task_success"))
        if metric(row, "task_success") == 1.0:
            successes += 1
            continue
        failure_recorded = False
        state_failure_recorded = False
        for index, expected_call in enumerate(expected_calls):
            expected_name = tool_call_name(expected_call)
            expected_is_state = expected_name in STATE_CHANGING_TOOL_NAMES
            if index >= len(actual_calls):
                missing[expected_name] += 1
                if not failure_recorded:
                    first_failure[f"missing:{expected_name}"] += 1
                    failure_recorded = True
                if expected_is_state and not state_failure_recorded:
                    first_state_failure[f"missing:{expected_name}"] += 1
                    state_failure_recorded = True
                continue
            actual_call = actual_calls[index]
            actual_name = tool_call_name(actual_call)
            if actual_name != expected_name:
                wrong_name[(expected_name, actual_name)] += 1
                if not failure_recorded:
                    first_failure[f"name:{expected_name}->{actual_name}"] += 1
                    failure_recorded = True
                if (expected_is_state or actual_name in STATE_CHANGING_TOOL_NAMES) and not state_failure_recorded:
                    first_state_failure[f"name:{expected_name}->{actual_name}"] += 1
                    state_failure_recorded = True
                continue
            arg_score = tool_argument_match_score(actual_call, expected_call)
            if arg_score < 1.0:
                wrong_args[expected_name] += 1
                if not failure_recorded:
                    first_failure[f"args:{expected_name}"] += 1
                    failure_recorded = True
                if expected_is_state and not state_failure_recorded:
                    first_state_failure[f"args:{expected_name}"] += 1
                    state_failure_recorded = True
        for call in actual_calls[len(expected_calls) :]:
            actual_name = tool_call_name(call)
            extra[actual_name] += 1
            if not failure_recorded:
                first_failure[f"extra:{actual_name}"] += 1
                failure_recorded = True
            if actual_name in STATE_CHANGING_TOOL_NAMES and not state_failure_recorded:
                first_state_failure[f"extra:{actual_name}"] += 1
                state_failure_recorded = True

    print(f"strict_successes={successes}/{len(rows)}")
    print("expected_state_actions_top=", expected_state_actions.most_common(12))
    print("actual_state_actions_top=", actual_state_actions.most_common(12))
    print(
        "outcome_by_expected_state_count=",
        {
            key: round(mean(values), 4)
            for key, values in sorted(outcome_by_expected_state_count.items())
        },
    )
    print(
        "task_by_expected_state_count=",
        {
            key: round(mean(values), 4)
            for key, values in sorted(task_by_expected_state_count.items())
        },
    )
    print("first_failure_top=", first_failure.most_common(12))
    print("first_state_failure_top=", first_state_failure.most_common(12))
    print("wrong_name_top=", wrong_name.most_common(12))
    print("wrong_args_top=", wrong_args.most_common(12))
    print("missing_top=", missing.most_common(12))
    print("extra_top=", extra.most_common(12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze strict retail eval failures from JSONL output.")
    parser.add_argument("path")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    summarize(Path(args.path), data_dir=Path(args.data_dir), split=args.split)


if __name__ == "__main__":
    main()
