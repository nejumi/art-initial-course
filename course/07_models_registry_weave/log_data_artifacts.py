from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from course.shared.config import config_from_env
from course.shared.wandb_artifacts import log_retail_data_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Log retail course data as a W&B dataset artifact.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--artifact-name", default="retail-course-data")
    parser.add_argument("--aliases", nargs="+", default=["latest", "tau-retail-v1"])
    args = parser.parse_args()

    cfg = config_from_env()
    log_retail_data_artifact(
        cfg,
        data_dir=Path(args.data_dir),
        artifact_name=args.artifact_name,
        aliases=args.aliases,
    )


if __name__ == "__main__":
    main()
