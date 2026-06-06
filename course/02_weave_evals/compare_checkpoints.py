from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from course.shared.config import config_from_env
from course.shared.data import read_jsonl


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


def summarize(path: Path) -> dict[str, float | int | str]:
    rows = read_jsonl(path)
    summary: dict[str, float | int | str] = {"path": str(path), "rows": len(rows)}
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


def markdown_table(summaries: list[dict[str, float | int | str]]) -> str:
    if not summaries:
        return ""
    baseline = summaries[0]
    header = ["stage", "rows", *METRIC_KEYS, "delta_reward", "delta_task_success", "delta_tool_name_f1"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for summary in summaries:
        path = Path(str(summary["path"]))
        row: list[str] = [label_for(path), str(summary.get("rows", 0))]
        for key in METRIC_KEYS:
            value = numeric(summary.get(key))
            row.append(f"{value:.4f}" if value is not None else "")
        for key in ("reward", "task_success", "tool_name_f1"):
            current = numeric(summary.get(key))
            base = numeric(baseline.get(key))
            if current is not None and base is not None:
                row.append(f"{current - base:+.4f}")
            else:
                row.append("")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local eval JSONL files from multiple checkpoints.")
    parser.add_argument("paths", nargs="+", help="JSONL files with reward/metrics rows")
    parser.add_argument("--output-md", default=None, help="Optional Markdown report path.")
    parser.add_argument("--wandb", action="store_true", help="Log the comparison table to W&B.")
    parser.add_argument("--run-name", default="checkpoint-eval-comparison")
    args = parser.parse_args()
    summaries = [summarize(Path(path_str)) for path_str in args.paths]
    for summary in summaries:
        print(summary["path"])
        print(json.dumps(summary, indent=2, sort_keys=True))
    table_md = markdown_table(summaries)
    if table_md:
        print(table_md)
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# Checkpoint Evaluation Comparison\n\n" + table_md, encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.wandb:
        import wandb

        cfg = config_from_env()
        run = wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            name=args.run_name,
            job_type="eval-comparison",
            config={"paths": args.paths, "model_name": cfg.model_name, "base_model": cfg.base_model},
        )
        table = wandb.Table(columns=["stage", "metric", "value"])
        baseline = summaries[0] if summaries else {}
        for summary in summaries:
            stage = label_for(Path(str(summary["path"])))
            for key in METRIC_KEYS:
                value = numeric(summary.get(key))
                if value is not None:
                    table.add_data(stage, key, value)
                    run.summary[f"{stage}/{key}"] = value
                    base = numeric(baseline.get(key))
                    if base is not None:
                        run.summary[f"{stage}/delta_{key}"] = value - base
        run.log({"checkpoint_eval_comparison": table})
        run.finish()


if __name__ == "__main__":
    main()
