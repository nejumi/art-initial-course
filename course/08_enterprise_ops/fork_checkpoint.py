from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio

from course.shared.art_compat import make_local_backend, make_trainable_model
from course.shared.config import config_from_env


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Initialize the current model by forking a checkpoint from another LocalBackend model.")
    parser.add_argument("--from-model", required=True, help="Source ART model name")
    parser.add_argument("--from-project", default=None, help="Source ART project; defaults to current project")
    parser.add_argument("--not-after-step", type=int, default=None, help="Fork the latest checkpoint at or before this step")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = config_from_env()
    backend = make_local_backend(cfg.art_path)
    model = make_trainable_model(cfg)
    await model.register(backend)
    await backend._experimental_fork_checkpoint(
        model,
        from_model=args.from_model,
        from_project=args.from_project,
        not_after_step=args.not_after_step,
        verbose=args.verbose,
    )
    print(f"Forked checkpoint from {args.from_model} into {cfg.model_name}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
