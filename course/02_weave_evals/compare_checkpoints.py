from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import statistics
from pathlib import Path

from course.shared.config import config_from_env
from course.shared.data import read_jsonl
from course.shared.wandb_artifacts import artifact_with_alias


METRIC_KEYS = [
    "reward",
    "outcome_pass_at_k",
    "task_pass_at_k",
    "reward_std_by_scenario",
    "tau2_official_reward",
    "tau2_db_success",
    "tau2_communicate_success",
    "tau2_action_match",
    "tau2_read_action_match",
    "tau2_write_action_match",
    "tau2_terminated_cleanly",
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
    "state_action_expected_count",
    "state_action_actual_count",
    "state_action_attempt_rate",
    "state_action_reached_rate",
    "valid_state_action_rate",
    "final_text_f1",
    "has_final_response",
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
    "first_failure_observed",
    "first_failure_turn",
    "first_state_action_turn",
    "first_expected_state_action_turn",
    "read_only_reference_mismatches_before_state_action",
    "accepted_state_action_jump",
    "accepted_state_action_jump_count",
    "first_accepted_state_action_jump_turn",
    "skipped_reference_turns_before_state_action",
    "last_accepted_reference_state_output_index",
    "last_accepted_reference_state_turn_index",
    "reward_component/outcome",
    "reward_component/state_action",
    "reward_component/state_action_args",
    "reward_component/communication",
    "reward_component/tool_name",
    "reward_component/tool_order",
    "reward_component/tool_args",
    "reward_component/final_text",
    "reward_component/task",
    "reward_component/penalty_invalid",
    "reward_component/penalty_invalid_state",
    "reward_component/penalty_bad_state",
    "reward_component/penalty_read_only",
    "reward_component/penalty_unknown_tool",
    "reward_component/penalty_missing_state",
    "reward_component/penalty_turns",
]

DELTA_KEYS = [
    "reward",
    "outcome_pass_at_k",
    "task_pass_at_k",
    "tau2_official_reward",
    "tau2_db_success",
    "tau2_communicate_success",
    "tau2_write_action_match",
    "outcome_success",
    "proxy_outcome_success",
    "task_success",
    "state_action_sequence_match",
    "accepted_state_action_jump",
    "communication_success",
    "tool_name_f1",
    "bad_state_action",
    "missing_state_action",
]

COMPACT_METRIC_KEYS = [
    "reward",
    "outcome_success",
    "proxy_outcome_success",
    "task_success",
    "outcome_pass_at_k",
    "task_pass_at_k",
    "state_action_sequence_match",
    "communication_success",
    "tool_name_f1",
    "bad_state_action",
    "missing_state_action",
    "truncated_by_max_turn",
    "reward_std_by_scenario",
]

COMPACT_DELTA_KEYS = [
    "reward",
    "outcome_success",
    "task_success",
    "state_action_sequence_match",
    "communication_success",
    "bad_state_action",
    "missing_state_action",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def numeric(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped == "-":
        return None
    return stripped


def summarize(
    path: Path,
    *,
    stage: str | None = None,
    model: str | None = None,
    model_artifact_path: str | None = None,
) -> dict[str, float | int | str | None]:
    rows = read_jsonl(path)
    summary: dict[str, float | int | str | None] = {
        "path": str(path),
        "rows": len(rows),
        "stage": stage or label_for(path),
        "model": model,
        "model_artifact_path": model_artifact_path,
    }
    reward_values: list[float] = []
    metrics: dict[str, list[float]] = {}
    by_scenario: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row.get("scenario_id") or len(by_scenario)), []).append(row)
        try:
            reward_values.append(float(row.get("reward")))
        except Exception:
            pass
        for key, value in (row.get("metrics") or {}).items():
            try:
                metrics.setdefault(key, []).append(float(value))
            except Exception:
                pass
    if reward_values:
        summary["reward"] = mean(reward_values)
    if by_scenario:
        outcome_pass = []
        task_pass = []
        reward_stds = []
        for scenario_rows in by_scenario.values():
            outcome_values = []
            task_values = []
            scenario_rewards = []
            for row in scenario_rows:
                try:
                    scenario_rewards.append(float(row.get("reward")))
                except Exception:
                    pass
                row_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                for key, target in (("outcome_success", outcome_values), ("task_success", task_values)):
                    try:
                        target.append(float(row_metrics.get(key)))  # type: ignore[union-attr]
                    except Exception:
                        pass
            if outcome_values:
                outcome_pass.append(max(outcome_values))
            if task_values:
                task_pass.append(max(task_values))
            if len(scenario_rewards) >= 2:
                reward_stds.append(statistics.pstdev(scenario_rewards))
        if outcome_pass:
            summary["outcome_pass_at_k"] = mean(outcome_pass)
        if task_pass:
            summary["task_pass_at_k"] = mean(task_pass)
        if reward_stds:
            summary["reward_std_by_scenario"] = mean(reward_stds)
    for key, values in metrics.items():
        if values:
            summary[key] = mean(values)
    return summary


def label_for(path: Path) -> str:
    stem = path.stem
    for prefix in ("eval_", "after_"):
        stem = stem.removeprefix(prefix)
    return stem


def markdown_table(
    summaries: list[dict[str, float | int | str | None]],
    reference_index: int = 0,
    *,
    metric_keys: list[str] | None = None,
    delta_keys: list[str] | None = None,
) -> str:
    if not summaries:
        return ""
    metric_keys = metric_keys or METRIC_KEYS
    delta_keys = delta_keys or DELTA_KEYS
    reference = summaries[reference_index]
    header = [
        "model",
        "stage",
        "model_artifact_path",
        "rows",
        *metric_keys,
        *[f"delta_{key}" for key in delta_keys],
    ]
    reference_stage = str(reference.get("stage") or label_for(Path(str(reference["path"]))))
    lines = [
        f"Reference stage for deltas: `{reference_stage}`",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for summary in summaries:
        row: list[str] = [
            str(summary.get("model") or ""),
            str(summary.get("stage") or label_for(Path(str(summary["path"])))),
            str(summary.get("model_artifact_path") or ""),
            str(summary.get("rows", 0)),
        ]
        for key in metric_keys:
            value = numeric(summary.get(key))
            row.append(f"{value:.4f}" if value is not None else "")
        for key in delta_keys:
            current = numeric(summary.get(key))
            ref = numeric(reference.get(key))
            if current is not None and ref is not None:
                row.append(f"{current - ref:+.4f}")
            else:
                row.append("")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_csv(
    path: Path,
    summaries: list[dict[str, float | int | str | None]],
    reference_index: int = 0,
    *,
    metric_keys: list[str] | None = None,
    delta_keys: list[str] | None = None,
) -> None:
    metric_keys = metric_keys or METRIC_KEYS
    delta_keys = delta_keys or DELTA_KEYS
    reference = summaries[reference_index] if summaries else {}
    header = [
        "model",
        "stage",
        "model_artifact_path",
        "eval_path",
        "rows",
        *metric_keys,
        *[f"delta_{key}" for key in delta_keys],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for summary in summaries:
            row: list[object] = [
                summary.get("model") or "",
                summary.get("stage") or label_for(Path(str(summary["path"]))),
                summary.get("model_artifact_path") or "",
                summary.get("path") or "",
                summary.get("rows", 0),
            ]
            for key in metric_keys:
                row.append(summary.get(key))
            for key in delta_keys:
                current = numeric(summary.get(key))
                ref = numeric(reference.get(key))
                row.append(current - ref if current is not None and ref is not None else "")
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local eval JSONL files from multiple checkpoints.")
    parser.add_argument("paths", nargs="+", help="JSONL files with reward/metrics rows")
    parser.add_argument("--stages", nargs="+", default=None, help="Stage labels, one per JSONL path.")
    parser.add_argument("--model", default=None, help="Model label to put on every comparison row. Defaults to the configured base model.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional model labels, one per JSONL path.")
    parser.add_argument("--model-artifacts", nargs="+", default=None, help="Optional W&B model artifact URIs, one per JSONL path. Use '-' for no artifact.")
    parser.add_argument("--data-artifact", default=None, help="Optional dataset artifact URI to attach to the comparison run.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown report path.")
    parser.add_argument("--output-compact-md", default=None, help="Optional compact Markdown report path for slides/README.")
    parser.add_argument("--output-csv", default=None, help="Optional full horizontal CSV report path.")
    parser.add_argument("--output-compact-csv", default=None, help="Optional compact horizontal CSV report path.")
    parser.add_argument("--wandb", action="store_true", help="Log the comparison table to W&B.")
    parser.add_argument("--run-name", default="model-stage-eval-comparison")
    parser.add_argument("--reference-index", type=int, default=0, help="0-based stage index used for delta columns.")
    args = parser.parse_args()
    cfg = config_from_env()
    if args.stages is not None and len(args.stages) != len(args.paths):
        raise SystemExit("--stages must have the same length as paths")
    if args.models is not None and len(args.models) != len(args.paths):
        raise SystemExit("--models must have the same length as paths")
    if args.model_artifacts is not None and len(args.model_artifacts) != len(args.paths):
        raise SystemExit("--model-artifacts must have the same length as paths")
    default_model = args.model or cfg.base_model
    model_artifacts = [normalize_optional(value) for value in args.model_artifacts] if args.model_artifacts else [None] * len(args.paths)
    summaries = [
        summarize(
            Path(path_str),
            stage=args.stages[index] if args.stages else None,
            model=args.models[index] if args.models else default_model,
            model_artifact_path=model_artifacts[index],
        )
        for index, path_str in enumerate(args.paths)
    ]
    if not 0 <= args.reference_index < len(summaries):
        raise SystemExit(f"--reference-index must be between 0 and {len(summaries) - 1}")
    for summary in summaries:
        print(summary["path"])
        print(json.dumps(summary, indent=2, sort_keys=True))
    table_md = markdown_table(summaries, reference_index=args.reference_index)
    if table_md:
        print(table_md)
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# Checkpoint Evaluation Comparison\n\n" + table_md, encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.output_compact_md:
        output_path = Path(args.output_compact_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compact_md = markdown_table(
            summaries,
            reference_index=args.reference_index,
            metric_keys=COMPACT_METRIC_KEYS,
            delta_keys=COMPACT_DELTA_KEYS,
        )
        output_path.write_text("# Compact Checkpoint Evaluation Comparison\n\n" + compact_md, encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.output_csv:
        output_path = Path(args.output_csv)
        write_csv(output_path, summaries, reference_index=args.reference_index)
        print(f"Wrote {output_path}")
    if args.output_compact_csv:
        output_path = Path(args.output_compact_csv)
        write_csv(
            output_path,
            summaries,
            reference_index=args.reference_index,
            metric_keys=COMPACT_METRIC_KEYS,
            delta_keys=COMPACT_DELTA_KEYS,
        )
        print(f"Wrote {output_path}")
    if args.wandb:
        import wandb

        run = wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            name=args.run_name,
            job_type="eval-comparison",
            config={
                "paths": args.paths,
                "model_name": cfg.model_name,
                "model": default_model,
                "base_model": cfg.base_model,
                "stages": args.stages,
                "model_artifacts": args.model_artifacts,
                "data_artifact": args.data_artifact,
                "reference_index": args.reference_index,
                "reference_stage": str(summaries[args.reference_index].get("stage")),
            },
        )
        if args.data_artifact:
            run.use_artifact(artifact_with_alias(args.data_artifact), type="dataset")
        for artifact_uri in model_artifacts:
            if artifact_uri:
                run.use_artifact(artifact_with_alias(artifact_uri), type="model")
        delta_keys = [f"delta_{key}" for key in DELTA_KEYS]
        table = wandb.Table(
            columns=[
                "model",
                "stage",
                "model_artifact_path",
                "eval_path",
                "rows",
                *METRIC_KEYS,
                *delta_keys,
            ]
        )
        reference = summaries[args.reference_index] if summaries else {}
        for summary in summaries:
            stage = str(summary.get("stage") or label_for(Path(str(summary["path"]))))
            row_values: list[object] = [
                summary.get("model") or "",
                stage,
                summary.get("model_artifact_path") or "",
                summary["path"],
                summary.get("rows", 0),
            ]
            for key in METRIC_KEYS:
                value = numeric(summary.get(key))
                row_values.append(value)
                if value is not None:
                    run.summary[f"{stage}/{key}"] = value
            for key in DELTA_KEYS:
                current = numeric(summary.get(key))
                ref = numeric(reference.get(key))
                delta = current - ref if current is not None and ref is not None else None
                row_values.append(delta)
                if delta is not None:
                    run.summary[f"{stage}/delta_{key}"] = delta
            table.add_data(*row_values)
        run.log({"model_stage_eval_comparison": table})
        run.finish()


if __name__ == "__main__":
    main()
