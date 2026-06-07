from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
from typing import Any

from course.shared.config import config_from_env
from course.shared.data import read_jsonl
from course.shared.tracing import init_weave
from course.shared.wandb_artifacts import artifact_with_alias, ensure_wandb_run, finish_wandb_run


SUMMARY_METRICS = [
    "course_eval_score",
    "outcome_success",
    "task_success",
    "state_action_sequence_match",
    "valid_state_action_rate",
    "communication_success",
    "bad_state_action",
    "missing_state_action",
    "truncated_by_max_turn",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    rewards: list[float] = []
    metrics: dict[str, list[float]] = {key: [] for key in SUMMARY_METRICS}
    for row in rows:
        try:
            rewards.append(float(row.get("reward")))
        except Exception:
            pass
        row_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        for key in SUMMARY_METRICS:
            try:
                metrics[key].append(float(row_metrics.get(key)))
            except Exception:
                pass
    summary: dict[str, float | int] = {"rows": len(rows)}
    if rewards:
        summary["reward"] = mean(rewards)
    for key, values in metrics.items():
        if values:
            summary[key] = mean(values)
    return summary


def build_cached_eval_rows(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    model: str,
    model_artifact: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    dataset_rows: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        row_id = f"{stage}:{row.get('scenario_id', index)}:{row.get('rollout_index', 0)}:{index}"
        dataset_rows.append(
            {
                "row_id": row_id,
                "scenario_id": row.get("scenario_id"),
                "rollout_index": row.get("rollout_index", 0),
                "stage": stage,
                "model": model,
                "model_artifact_path": model_artifact,
            }
        )
        outputs[row_id] = {
            "reward": row.get("reward"),
            "metrics": row.get("metrics") or {},
            "logs": row.get("logs") or {},
            "metadata": {
                "stage": stage,
                "model": model,
                "model_artifact_path": model_artifact,
                "scenario_id": row.get("scenario_id"),
                "rollout_index": row.get("rollout_index", 0),
            },
        }
    return dataset_rows, outputs


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a cached checkpoint JSONL as a Weave Evaluation without rerunning rollouts."
    )
    parser.add_argument("path")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-artifact", default=None)
    parser.add_argument("--data-artifact", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Do not create a companion W&B eval run/table.")
    args = parser.parse_args()

    cfg = config_from_env()
    rows = read_jsonl(args.path)
    run = None
    owned_run = False
    if not args.no_wandb:
        run, owned_run = ensure_wandb_run(
            cfg,
            job_type="weave-cached-eval",
            run_name=args.name or f"{args.stage}-cached-checkpoint-eval",
        )
        if run is not None:
            run.config.update(
                {
                    "stage": args.stage,
                    "model": args.model,
                    "model_artifact": args.model_artifact,
                    "data_artifact": args.data_artifact,
                    "cached_eval_path": args.path,
                    "weave_evaluation_name": args.name,
                },
                allow_val_change=True,
            )
            if args.data_artifact:
                run.use_artifact(artifact_with_alias(args.data_artifact), type="dataset")
            if args.model_artifact:
                run.use_artifact(artifact_with_alias(args.model_artifact), type="model")

            import wandb

            row_table = wandb.Table(
                columns=[
                    "stage",
                    "model",
                    "model_artifact_path",
                    "scenario_id",
                    "rollout_index",
                    "reward",
                    *SUMMARY_METRICS,
                ]
            )
            for row in rows:
                row_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                row_table.add_data(
                    args.stage,
                    args.model,
                    args.model_artifact,
                    row.get("scenario_id"),
                    row.get("rollout_index", 0),
                    row.get("reward"),
                    *[row_metrics.get(key) for key in SUMMARY_METRICS],
                )
            run.log({"cached_checkpoint_eval_rows": row_table})
            for key, value in summarize_rows(rows).items():
                run.summary[key] = value

    init_weave(cfg.project)
    import weave
    from course.weave_eval_scorers import SCORERS as scorers

    dataset_rows, outputs = build_cached_eval_rows(
        rows,
        stage=args.stage,
        model=args.model,
        model_artifact=args.model_artifact,
    )

    class CachedCheckpointModel(weave.Model):
        stage: str
        model: str
        model_artifact_path: str | None = None
        data_artifact_path: str | None = None
        outputs: dict[str, dict[str, Any]]

        @weave.op()
        async def predict(self, row_id: str, **_: Any) -> dict[str, Any]:
            return self.outputs[row_id]

    evaluation_kwargs: dict[str, Any] = {"dataset": dataset_rows, "scorers": scorers}
    if args.name:
        evaluation_kwargs["name"] = args.name
    evaluation = weave.Evaluation(**evaluation_kwargs)
    try:
        result = await evaluation.evaluate(
            CachedCheckpointModel(
                stage=args.stage,
                model=args.model,
                model_artifact_path=args.model_artifact,
                data_artifact_path=args.data_artifact,
                outputs=outputs,
            )
        )
        if run is not None:
            run.summary["weave_evaluation_result"] = str(result)
        print(result)
    finally:
        finish_wandb_run(run, owned_run)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
