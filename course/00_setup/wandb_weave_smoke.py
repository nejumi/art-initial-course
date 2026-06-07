from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
import time

from course.shared.config import config_from_env
from course.shared.tracing import init_weave, weave_op
from course.shared.wandb_artifacts import ensure_wandb_run, finish_wandb_run


@weave_op("course_smoke_add")
def smoke_add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--skip-run-link-check",
        action="store_true",
        help="Skip querying Weave to confirm that the smoke call is linked to the active W&B run.",
    )
    args = parser.parse_args()
    if args.project:
        os.environ["WANDB_PROJECT"] = args.project
        os.environ.setdefault("WEAVE_PROJECT", args.project)
    os.environ.setdefault("COURSE_RUN_STAGE", "setup-smoke")
    os.environ.setdefault("COURSE_RUN_KIND", "setup")
    cfg = config_from_env()

    wandb_run = None
    owned_wandb_run = False
    try:
        wandb_run, owned_wandb_run = ensure_wandb_run(cfg, job_type="smoke")
        if wandb_run is not None:
            wandb_run.config.update({"check": "wandb-weave-run-linkage"}, allow_val_change=True)
    except Exception as exc:
        print("W&B init skipped or failed:", exc)

    weave_client = init_weave()
    result = smoke_add(2, 3)
    print("Weave smoke op result:", result)

    if wandb_run is not None and weave_client is not None and not args.skip_run_link_check:
        try:
            weave_client.flush()
        except Exception:
            pass
        expected_suffix = f"/{wandb_run.id}"
        linked_run_id = None
        for _ in range(10):
            calls = weave_client.get_calls(
                limit=20,
                sort_by=[{"field": "started_at", "direction": "desc"}],
                columns=["op_name", "wb_run_id", "started_at"],
            )
            for call in calls:
                if "course_smoke_add" not in str(getattr(call, "op_name", "")):
                    continue
                linked_run_id = getattr(call, "wb_run_id", None)
                if linked_run_id and str(linked_run_id).endswith(expected_suffix):
                    print("Weave call linked to W&B run:", linked_run_id)
                    break
            if linked_run_id and str(linked_run_id).endswith(expected_suffix):
                break
            time.sleep(1)
        else:
            raise SystemExit(
                "Weave smoke call was written, but it was not linked to the active W&B run. "
                f"Expected suffix {expected_suffix!r}; observed {linked_run_id!r}."
            )

    if wandb_run is not None:
        wandb_run.log({"smoke/result": result})
        finish_wandb_run(wandb_run, owned_wandb_run)


if __name__ == "__main__":
    main()
