from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio

from course.shared.art_compat import make_local_backend, make_trainable_model
from course.shared.config import config_from_env


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Delete all but the latest and best ART checkpoints.")
    parser.add_argument("--best-checkpoint-metric", default="val/reward")
    args = parser.parse_args()

    cfg = config_from_env()
    backend = make_local_backend(cfg.art_path)
    model = make_trainable_model(cfg)
    await model.register(backend)
    await model.delete_checkpoints(best_checkpoint_metric=args.best_checkpoint_metric)
    print("Checkpoint pruning requested.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
