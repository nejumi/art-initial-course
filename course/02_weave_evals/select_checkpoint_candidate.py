from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
from typing import Any


DEFAULT_COLUMNS = [
    "data/step_reward_mean",
    "data/step_reward_range_mean",
    "data/step_agentic_signal_group_rate",
    "data/step_reward_only_signal_group_rate",
    "data/step_outcome_success_mean",
    "data/step_task_success_mean",
    "data/step_state_action_sequence_match_mean",
    "data/step_state_action_reached_rate_mean",
    "data/step_valid_state_action_rate_mean",
    "data/step_bad_state_action_mean",
    "data/step_missing_state_action_mean",
    "data/step_truncated_by_max_turn_mean",
    "data/step_num_groups_dropped_no_reward_signal",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None and isinstance(row.get("metrics"), dict):
        value = row["metrics"].get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def choose_candidates(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    top_k: int,
    min_step: int | None = None,
    max_step: int | None = None,
    include_skipped: bool = False,
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        if row.get("skipped") and not include_skipped:
            continue
        step = row.get("step")
        if isinstance(step, int):
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
        value = metric_value(row, metric)
        if value is None:
            continue
        scored.append((value, step if isinstance(step, int) else -1, row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _, _, row in scored[:top_k]]


def markdown_report(candidates: list[dict[str, Any]], *, metric: str) -> str:
    lines = [
        "# Checkpoint Candidate Selection",
        "",
        f"Selection metric: `{metric}`",
        "",
        "Use this as a candidate shortlist for fresh rollout evaluation. It is not a substitute for held-out eval or Weave trace inspection.",
        "",
        "| rank | algorithm | step | selection_metric | "
        + " | ".join(DEFAULT_COLUMNS)
        + " |",
        "| ---: | --- | ---: | ---: | "
        + " | ".join(["---:"] * len(DEFAULT_COLUMNS))
        + " |",
    ]
    for rank, row in enumerate(candidates, start=1):
        def fmt(key: str) -> str:
            value = metric_value(row, key)
            return "" if value is None else f"{value:.4f}"

        values = [
            str(rank),
            str(row.get("algorithm") or ""),
            str(row.get("step") or ""),
            fmt(metric),
            *[fmt(key) for key in DEFAULT_COLUMNS],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select promising RL checkpoint steps from local per-step metrics JSONL.")
    parser.add_argument("metrics_jsonl", type=Path)
    parser.add_argument("--metric", default="data/step_reward_mean")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--include-skipped", action="store_true")
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.metrics_jsonl)
    candidates = choose_candidates(
        rows,
        metric=args.metric,
        top_k=args.top_k,
        min_step=args.min_step,
        max_step=args.max_step,
        include_skipped=args.include_skipped,
    )
    result = {
        "metric": args.metric,
        "source": str(args.metrics_jsonl),
        "candidates": candidates,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.output_json}")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(candidates, metric=args.metric), encoding="utf-8")
        print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
