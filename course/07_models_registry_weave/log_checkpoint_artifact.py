from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

from course.shared.config import config_from_env
from course.shared.wandb_artifacts import log_checkpoint_artifact


def parse_metadata(values: list[str]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise SystemExit(f"Invalid metadata item {value!r}; expected KEY=VALUE.")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Log the latest local ART checkpoint as a W&B model artifact.")
    parser.add_argument("--stage", required=True, help="Stage label for the artifact metadata and alias.")
    parser.add_argument("--artifact-name", default=None, help="Optional W&B artifact name.")
    parser.add_argument("--alias", action="append", default=[], help="Additional artifact alias; may be repeated.")
    parser.add_argument("--model-name", default=None, help="ART model name. Defaults to ART_MODEL_NAME from .env/environment.")
    parser.add_argument("--metadata", action="append", default=[], help="Extra metadata as KEY=JSON_VALUE or KEY=string.")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for the artifact upload to finish.")
    args = parser.parse_args()

    cfg = config_from_env()
    uri = log_checkpoint_artifact(
        cfg,
        stage=args.stage,
        artifact_name=args.artifact_name,
        aliases=args.alias,
        metadata=parse_metadata(args.metadata),
        model_name=args.model_name,
        wait=not args.no_wait,
    )
    if uri:
        print(uri)


if __name__ == "__main__":
    main()
