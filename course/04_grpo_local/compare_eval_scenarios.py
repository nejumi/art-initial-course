from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
from collections import Counter
from pathlib import Path

from course.shared.data import read_jsonl


DEFAULT_METRICS = [
    "reward",
    "outcome_success",
    "proxy_outcome_success",
    "task_success",
    "state_action_sequence_match",
    "communication_success",
    "tool_name_f1",
    "tool_argument_match",
    "invalid_tool_call",
    "bad_state_action",
    "missing_state_action",
    "truncated_by_max_turn",
]


def numeric(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def stage_rows(path: Path) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    for index, row in enumerate(read_jsonl(path)):
        scenario_id = str(row.get("scenario_id") or f"row-{index}")
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        values: dict[str, float | str] = {"scenario_id": scenario_id}
        reward = numeric(row.get("reward"))
        if reward is not None:
            values["reward"] = reward
        for key, value in metrics.items():  # type: ignore[union-attr]
            number = numeric(value)
            if number is not None:
                values[str(key)] = number
        rows[scenario_id] = values
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def status(value: float | None) -> str:
    if value is None:
        return "missing"
    return "success" if value >= 1.0 else "fail"


def format_float(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eval JSONLs by scenario to detect real improvements versus success churn."
    )
    parser.add_argument("paths", nargs="+", help="Eval JSONL files in stage order.")
    parser.add_argument("--stages", nargs="+", default=None, help="Stage labels, one per path.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--primary-metric", default="outcome_success")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths]
    if args.stages and len(args.stages) != len(paths):
        raise SystemExit("--stages must match number of paths")
    stages = args.stages or [path.stem.removeprefix("eval_") for path in paths]
    by_stage = [stage_rows(path) for path in paths]
    scenario_ids = sorted(set().union(*(set(rows) for rows in by_stage)))

    wide_rows: list[dict[str, object]] = []
    transition_counts: Counter[str] = Counter()
    first_stage = by_stage[0]
    last_stage = by_stage[-1]
    for scenario_id in scenario_ids:
        row: dict[str, object] = {"scenario_id": scenario_id}
        first_primary = numeric(first_stage.get(scenario_id, {}).get(args.primary_metric))
        last_primary = numeric(last_stage.get(scenario_id, {}).get(args.primary_metric))
        row["primary_transition"] = f"{status(first_primary)}->{status(last_primary)}"
        transition_counts[str(row["primary_transition"])] += 1
        if first_primary is not None and last_primary is not None:
            row[f"delta_{args.primary_metric}"] = last_primary - first_primary
        for stage, stage_data in zip(stages, by_stage):
            values = stage_data.get(scenario_id, {})
            for metric in args.metrics:
                row[f"{stage}/{metric}"] = values.get(metric)
        wide_rows.append(row)

    print(f"scenarios={len(scenario_ids)}")
    print(f"primary_metric={args.primary_metric}")
    print("primary_transitions=", dict(sorted(transition_counts.items())))
    for stage, stage_data in zip(stages, by_stage):
        print(f"\n[{stage}] rows={len(stage_data)}")
        for metric in args.metrics:
            values = [value for values in stage_data.values() if isinstance(value := values.get(metric), float)]
            if values:
                print(f"{metric}={mean(values):.4f}")

    ordered_columns = ["scenario_id", "primary_transition", f"delta_{args.primary_metric}"]
    for stage in stages:
        ordered_columns.extend(f"{stage}/{metric}" for metric in args.metrics)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(wide_rows)
        print(f"Wrote {output_path}")

    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Scenario-Level Eval Comparison",
            "",
            f"- primary_metric: `{args.primary_metric}`",
            f"- scenarios: {len(scenario_ids)}",
            f"- transitions: `{dict(sorted(transition_counts.items()))}`",
            "",
            "| " + " | ".join(ordered_columns) + " |",
            "| " + " | ".join(["---"] * len(ordered_columns)) + " |",
        ]
        for row in wide_rows:
            lines.append("| " + " | ".join(format_float(row.get(column)) for column in ordered_columns) + " |")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
