from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from course.shared.config import config_from_env
from course.shared.data import read_jsonl
from course.shared.wandb_artifacts import artifact_with_alias


METRIC_KEYS = [
    "reward",
    "task_success",
    "tool_name_f1",
    "tool_order_match",
    "final_text_f1",
    "has_final_response",
    "invalid_tool_call",
    "turn_count",
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
    for row in rows:
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
    for key, values in metrics.items():
        if values:
            summary[key] = mean(values)
    return summary


def label_for(path: Path) -> str:
    stem = path.stem
    for prefix in ("eval_", "after_"):
        stem = stem.removeprefix(prefix)
    return stem


def markdown_table(summaries: list[dict[str, float | int | str | None]], reference_index: int = 0) -> str:
    if not summaries:
        return ""
    reference = summaries[reference_index]
    header = [
        "model",
        "stage",
        "model_artifact_path",
        "rows",
        *METRIC_KEYS,
        "delta_reward",
        "delta_task_success",
        "delta_tool_name_f1",
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
        for key in METRIC_KEYS:
            value = numeric(summary.get(key))
            row.append(f"{value:.4f}" if value is not None else "")
        for key in ("reward", "task_success", "tool_name_f1"):
            current = numeric(summary.get(key))
            ref = numeric(reference.get(key))
            if current is not None and ref is not None:
                row.append(f"{current - ref:+.4f}")
            else:
                row.append("")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local eval JSONL files from multiple checkpoints.")
    parser.add_argument("paths", nargs="+", help="JSONL files with reward/metrics rows")
    parser.add_argument("--stages", nargs="+", default=None, help="Stage labels, one per JSONL path.")
    parser.add_argument("--model", default=None, help="Model label to put on every comparison row. Defaults to the configured base model.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional model labels, one per JSONL path.")
    parser.add_argument("--model-artifacts", nargs="+", default=None, help="Optional W&B model artifact URIs, one per JSONL path. Use '-' for no artifact.")
    parser.add_argument("--data-artifact", default=None, help="Optional dataset artifact URI to attach to the comparison run.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown report path.")
    parser.add_argument("--wandb", action="store_true", help="Log the comparison table to W&B.")
    parser.add_argument("--run-name", default="checkpoint-eval-comparison")
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
        delta_keys = ["delta_reward", "delta_task_success", "delta_tool_name_f1"]
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
            for key in ("reward", "task_success", "tool_name_f1"):
                current = numeric(summary.get(key))
                ref = numeric(reference.get(key))
                delta = current - ref if current is not None and ref is not None else None
                row_values.append(delta)
                if delta is not None:
                    run.summary[f"{stage}/delta_{key}"] = delta
            table.add_data(*row_values)
        run.log({"checkpoint_eval_comparison": table})
        run.finish()


if __name__ == "__main__":
    main()
