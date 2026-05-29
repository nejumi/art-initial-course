from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from course.shared.config import config_from_env
from course.shared.tracing import init_weave, weave_op


@weave_op("course_smoke_add")
def smoke_add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None)
    args = parser.parse_args()
    cfg = config_from_env()
    project = args.project or cfg.project

    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(project=project, entity=cfg.entity, job_type="smoke", reinit=True)
        wandb_run.config.update({"course": "openpipe-art-retail"})
    except Exception as exc:
        print("W&B init skipped or failed:", exc)

    init_weave(project)
    result = smoke_add(2, 3)
    print("Weave smoke op result:", result)

    if wandb_run is not None:
        wandb_run.log({"smoke/result": result})
        wandb_run.finish()


if __name__ == "__main__":
    main()
