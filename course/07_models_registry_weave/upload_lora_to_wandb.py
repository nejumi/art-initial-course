from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from course.shared.config import config_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a LocalBackend LoRA checkpoint as a W&B artifact.")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--artifact-name", default=None)
    parser.add_argument("--type", default="lora")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_path)
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint path: {checkpoint}")
    cfg = config_from_env()

    import wandb

    run = wandb.init(project=cfg.project, entity=cfg.entity, job_type="upload-lora")
    artifact_name = args.artifact_name or f"{cfg.model_name}-lora"
    artifact = wandb.Artifact(artifact_name, type=args.type)
    artifact.add_dir(str(checkpoint))
    artifact.metadata.update({"task": "retail-support-agent", "base_model": cfg.base_model})
    run.log_artifact(artifact)
    run.finish()
    print("Uploaded artifact:", artifact_name)


if __name__ == "__main__":
    main()
