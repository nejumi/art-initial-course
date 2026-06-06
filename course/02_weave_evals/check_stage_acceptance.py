from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import importlib
import json
from dataclasses import dataclass
from typing import Any

compare_checkpoints = importlib.import_module("course.02_weave_evals.compare_checkpoints")
summarize = compare_checkpoints.summarize


PRIMARY_KEYS = [
    "reward",
    "outcome_success",
    "task_success",
    "state_action_sequence_match",
    "bad_state_action",
    "missing_state_action",
]


@dataclass(frozen=True)
class Criteria:
    sft_max_outcome_regression: float = 0.0
    sft_max_task_regression: float = 0.0
    rl_min_reward_delta: float = 0.05
    rl_min_outcome_delta: float = 0.05
    rl_min_task_delta: float = 0.05
    rl_min_state_action_delta: float = 0.05
    rl_min_error_reduction: float = 0.05
    rl_min_retained_sft_outcome_success_rate: float = 0.75


def numeric(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def delta(current: dict[str, Any], reference: dict[str, Any], key: str) -> float | None:
    current_value = numeric(current, key)
    reference_value = numeric(reference, key)
    if current_value is None or reference_value is None:
        return None
    return current_value - reference_value


def gte(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def lte(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def stage_success_map(path: Path, *, metric: str = "outcome_success") -> dict[str, bool]:
    rows = compare_checkpoints.read_jsonl(path)
    success: dict[str, bool] = {}
    for index, row in enumerate(rows):
        scenario_id = str(row.get("scenario_id") or f"row-{index}")
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        try:
            value = float(metrics.get(metric, 0.0))  # type: ignore[union-attr]
        except Exception:
            value = 0.0
        success[scenario_id] = success.get(scenario_id, False) or value >= 1.0
    return success


def success_churn(current: dict[str, bool], reference: dict[str, bool]) -> dict[str, Any]:
    scenario_ids = sorted(set(current) | set(reference))
    retained = 0
    lost = 0
    new = 0
    reference_successes = 0
    current_successes = 0
    for scenario_id in scenario_ids:
        was_success = bool(reference.get(scenario_id))
        is_success = bool(current.get(scenario_id))
        reference_successes += int(was_success)
        current_successes += int(is_success)
        if was_success and is_success:
            retained += 1
        elif was_success and not is_success:
            lost += 1
        elif is_success and not was_success:
            new += 1
    retention_rate = retained / reference_successes if reference_successes else None
    return {
        "reference_successes": reference_successes,
        "current_successes": current_successes,
        "retained_reference_successes": retained,
        "lost_reference_successes": lost,
        "new_successes": new,
        "retention_rate": retention_rate,
    }


def judge_sft(summary: dict[str, Any], baseline: dict[str, Any], criteria: Criteria) -> dict[str, Any]:
    outcome_delta = delta(summary, baseline, "outcome_success")
    task_delta = delta(summary, baseline, "task_success")
    state_delta = delta(summary, baseline, "state_action_sequence_match")
    reward_delta = delta(summary, baseline, "reward")
    accepted = (
        gte(outcome_delta, -criteria.sft_max_outcome_regression)
        and gte(task_delta, -criteria.sft_max_task_regression)
        and (
            gte(reward_delta, 0.0)
            or gte(state_delta, 0.0)
            or gte(outcome_delta, 0.0)
            or gte(task_delta, 0.0)
        )
    )
    reasons: list[str] = []
    if not gte(outcome_delta, -criteria.sft_max_outcome_regression):
        reasons.append("outcome_success regressed from baseline")
    if not gte(task_delta, -criteria.sft_max_task_regression):
        reasons.append("task_success regressed from baseline")
    if accepted:
        reasons.append("SFT is a usable RL parent under the configured gate")
    return {
        "stage": summary.get("stage"),
        "decision": "accept" if accepted else "reject",
        "reference_stage": baseline.get("stage"),
        "reason": "; ".join(reasons) or "no positive SFT evidence",
        "deltas": {key: delta(summary, baseline, key) for key in PRIMARY_KEYS},
    }


def judge_rl(summary: dict[str, Any], sft: dict[str, Any], criteria: Criteria) -> dict[str, Any]:
    return judge_rl_with_churn(summary, sft, criteria, churn=None)


def judge_rl_with_churn(
    summary: dict[str, Any],
    sft: dict[str, Any],
    criteria: Criteria,
    *,
    churn: dict[str, Any] | None,
) -> dict[str, Any]:
    reward_delta = delta(summary, sft, "reward")
    outcome_delta = delta(summary, sft, "outcome_success")
    task_delta = delta(summary, sft, "task_success")
    state_delta = delta(summary, sft, "state_action_sequence_match")
    bad_state_delta = delta(summary, sft, "bad_state_action")
    missing_delta = delta(summary, sft, "missing_state_action")
    performance_lift = (
        gte(outcome_delta, criteria.rl_min_outcome_delta)
        or gte(task_delta, criteria.rl_min_task_delta)
        or gte(state_delta, criteria.rl_min_state_action_delta)
    )
    error_lift = (
        lte(bad_state_delta, -criteria.rl_min_error_reduction)
        or lte(missing_delta, -criteria.rl_min_error_reduction)
    )
    retention_rate = churn.get("retention_rate") if churn else None
    retention_ok = (
        retention_rate is None
        or not (churn or {}).get("reference_successes")
        or gte(retention_rate, criteria.rl_min_retained_sft_outcome_success_rate)
    )
    accepted = gte(reward_delta, criteria.rl_min_reward_delta) and performance_lift and error_lift and retention_ok
    reasons: list[str] = []
    if not gte(reward_delta, criteria.rl_min_reward_delta):
        reasons.append("reward did not clear the RL delta gate")
    if not performance_lift:
        reasons.append("no outcome/task/state-action lift over SFT")
    if not error_lift:
        reasons.append("no bad-state or missing-state error reduction over SFT")
    if not retention_ok:
        reasons.append("RL churned away too many SFT outcome successes")
    if accepted:
        reasons.append("RL branch improves the SFT parent under the configured gate")
    return {
        "stage": summary.get("stage"),
        "decision": "accept" if accepted else "reject",
        "reference_stage": sft.get("stage"),
        "reason": "; ".join(reasons),
        "deltas": {key: delta(summary, sft, key) for key in PRIMARY_KEYS},
        "outcome_success_churn": churn or {},
    }


def markdown_report(
    decisions: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    criteria: Criteria,
) -> str:
    lines = [
        "# Stage Acceptance Gate",
        "",
        "This gate is intentionally conservative for workshop finalization. It is a diagnostic aid, not a replacement for inspecting Weave traces.",
        "",
        "## Criteria",
        "",
        f"- SFT must not regress `outcome_success` by more than `{criteria.sft_max_outcome_regression:.4f}` or `task_success` by more than `{criteria.sft_max_task_regression:.4f}` versus baseline.",
        f"- RL must improve reward by at least `{criteria.rl_min_reward_delta:.4f}` versus SFT.",
        f"- RL must improve at least one of outcome, task, or state-action sequence metrics versus SFT.",
        f"- RL must reduce either bad-state or missing-state errors versus SFT.",
        f"- RL must retain at least `{criteria.rl_min_retained_sft_outcome_success_rate:.4f}` of deterministic SFT `outcome_success` wins when SFT has any wins.",
        "",
        "## Decisions",
        "",
        "| stage | reference | decision | reason | delta_reward | delta_outcome | delta_task | delta_state_seq | delta_bad_state | delta_missing_state | retained_sft_wins | lost_sft_wins | new_wins | retention_rate |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for decision in decisions:
        deltas = decision["deltas"]
        churn = decision.get("outcome_success_churn") or {}
        def fmt(key: str) -> str:
            value = deltas.get(key)
            return "" if value is None else f"{value:+.4f}"
        def churn_fmt(key: str) -> str:
            value = churn.get(key)
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    str(decision["stage"]),
                    str(decision["reference_stage"]),
                    str(decision["decision"]),
                    str(decision["reason"]),
                    fmt("reward"),
                    fmt("outcome_success"),
                    fmt("task_success"),
                    fmt("state_action_sequence_match"),
                    fmt("bad_state_action"),
                    fmt("missing_state_action"),
                    churn_fmt("retained_reference_successes"),
                    churn_fmt("lost_reference_successes"),
                    churn_fmt("new_successes"),
                    churn_fmt("retention_rate"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Stage Metrics", ""])
    lines.append(
        "| stage | rows | reward | outcome_success | task_success | state_action_sequence_match | bad_state_action | missing_state_action |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        def metric(key: str) -> str:
            value = numeric(summary, key)
            return "" if value is None else f"{value:.4f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary.get("stage")),
                    str(summary.get("rows", "")),
                    metric("reward"),
                    metric("outcome_success"),
                    metric("task_success"),
                    metric("state_action_sequence_match"),
                    metric("bad_state_action"),
                    metric("missing_state_action"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether SFT/RL eval stages meet conservative course gates.")
    parser.add_argument("paths", nargs="+", help="Eval JSONL files in stage order.")
    parser.add_argument("--stages", nargs="+", required=True, help="Stage labels, one per JSONL path.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-reject", action="store_true")
    parser.add_argument("--sft-stage", default="sft_anchor")
    parser.add_argument("--baseline-stage", default="baseline")
    parser.add_argument("--sft-max-outcome-regression", type=float, default=0.0)
    parser.add_argument("--sft-max-task-regression", type=float, default=0.0)
    parser.add_argument("--rl-min-reward-delta", type=float, default=0.05)
    parser.add_argument("--rl-min-outcome-delta", type=float, default=0.05)
    parser.add_argument("--rl-min-task-delta", type=float, default=0.05)
    parser.add_argument("--rl-min-state-action-delta", type=float, default=0.05)
    parser.add_argument("--rl-min-error-reduction", type=float, default=0.05)
    parser.add_argument("--rl-min-retained-sft-outcome-success-rate", type=float, default=0.75)
    args = parser.parse_args()

    if len(args.stages) != len(args.paths):
        raise SystemExit("--stages must have the same length as paths")
    criteria = Criteria(
        sft_max_outcome_regression=args.sft_max_outcome_regression,
        sft_max_task_regression=args.sft_max_task_regression,
        rl_min_reward_delta=args.rl_min_reward_delta,
        rl_min_outcome_delta=args.rl_min_outcome_delta,
        rl_min_task_delta=args.rl_min_task_delta,
        rl_min_state_action_delta=args.rl_min_state_action_delta,
        rl_min_error_reduction=args.rl_min_error_reduction,
        rl_min_retained_sft_outcome_success_rate=args.rl_min_retained_sft_outcome_success_rate,
    )
    summaries = [
        summarize(Path(path), stage=stage, model=args.model)
        for path, stage in zip(args.paths, args.stages, strict=True)
    ]
    by_stage = {str(summary.get("stage")): summary for summary in summaries}
    baseline = by_stage.get(args.baseline_stage)
    sft = by_stage.get(args.sft_stage)
    if baseline is None:
        raise SystemExit(f"Missing baseline stage {args.baseline_stage!r}")
    if sft is None:
        raise SystemExit(f"Missing SFT stage {args.sft_stage!r}")

    path_by_stage = {stage: Path(path) for path, stage in zip(args.paths, args.stages, strict=True)}
    sft_successes = stage_success_map(path_by_stage[args.sft_stage])

    decisions = [judge_sft(sft, baseline, criteria)]
    for summary in summaries:
        stage = str(summary.get("stage"))
        if stage in {args.baseline_stage, args.sft_stage}:
            continue
        stage_successes = stage_success_map(path_by_stage[stage])
        decisions.append(
            judge_rl_with_churn(
                summary,
                sft,
                criteria,
                churn=success_churn(stage_successes, sft_successes),
            )
        )

    report = {
        "criteria": criteria.__dict__,
        "summaries": summaries,
        "decisions": decisions,
        "accepted": all(decision["decision"] == "accept" for decision in decisions),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_report(decisions, summaries, criteria=criteria), encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.fail_on_reject and not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
