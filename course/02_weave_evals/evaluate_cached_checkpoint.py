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
    args = parser.parse_args()

    cfg = config_from_env()
    init_weave(cfg.project)
    import weave
    from course.weave_eval_scorers import SCORERS as scorers

    rows = read_jsonl(args.path)
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
    result = await evaluation.evaluate(
        CachedCheckpointModel(
            stage=args.stage,
            model=args.model,
            model_artifact_path=args.model_artifact,
            data_artifact_path=args.data_artifact,
            outputs=outputs,
        )
    )
    print(result)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
