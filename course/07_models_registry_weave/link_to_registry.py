from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from course.shared.config import config_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Link a W&B artifact to a model registry collection.")
    parser.add_argument("artifact", help="Example: entity/project/name:version")
    parser.add_argument("--collection", default="retail-support-agent")
    parser.add_argument("--alias", default="candidate")
    args = parser.parse_args()

    cfg = config_from_env()
    import wandb

    run = wandb.init(project=cfg.project, entity=cfg.entity, job_type="registry-link")
    artifact = run.use_artifact(args.artifact)
    target = f"wandb-registry-Model/{args.collection}"
    run.link_artifact(artifact, target_path=target, aliases=[args.alias])
    run.finish()
    print(f"Linked {args.artifact} to {target} with alias {args.alias}")


if __name__ == "__main__":
    main()
