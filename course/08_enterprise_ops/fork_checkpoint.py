from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
from dataclasses import replace
import shutil

from course.shared.art_compat import make_local_backend, make_trainable_model, register_trainable_model
from course.shared.config import RetailCourseConfig, config_from_env


def checkpoint_steps(checkpoint_base_dir: Path) -> list[int]:
    if not checkpoint_base_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_base_dir}")
    return sorted(
        int(path.name)
        for path in checkpoint_base_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def select_checkpoint_step(steps: list[int], not_after_step: int | None = None) -> int:
    if not steps:
        raise FileNotFoundError("No checkpoint step directories found.")
    if not_after_step is None:
        return steps[-1]
    candidates = [step for step in steps if step <= not_after_step]
    if not candidates:
        raise ValueError(f"No checkpoint found at or before step {not_after_step}. Available steps: {steps}")
    return candidates[-1]


def file_only_fork_checkpoint(
    config: RetailCourseConfig,
    *,
    from_model: str,
    to_model: str,
    from_project: str | None = None,
    not_after_step: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    source_project = from_project or config.project
    source_root = Path(config.art_path) / source_project / "models" / from_model / "checkpoints"
    selected_step = select_checkpoint_step(checkpoint_steps(source_root), not_after_step)
    source_checkpoint = source_root / f"{selected_step:04d}"
    dest_checkpoint = (
        Path(config.art_path)
        / config.project
        / "models"
        / to_model
        / "checkpoints"
        / f"{selected_step:04d}"
    )
    if source_checkpoint.resolve() == dest_checkpoint.resolve():
        raise ValueError(f"Source and destination checkpoint are the same: {source_checkpoint}")
    if dest_checkpoint.exists():
        if not overwrite:
            raise FileExistsError(f"Destination checkpoint already exists: {dest_checkpoint}")
        shutil.rmtree(dest_checkpoint)
    dest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Copying checkpoint from {source_checkpoint} to {dest_checkpoint}")
    shutil.copytree(source_checkpoint, dest_checkpoint)
    return dest_checkpoint


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Initialize the current model by forking a checkpoint from another LocalBackend model.")
    parser.add_argument("--from-model", required=True, help="Source ART model name")
    parser.add_argument("--to-model", default=None, help="Destination ART model name. Defaults to ART_MODEL_NAME.")
    parser.add_argument("--from-project", default=None, help="Source ART project; defaults to current project")
    parser.add_argument("--not-after-step", type=int, default=None, help="Fork the latest checkpoint at or before this step")
    parser.add_argument(
        "--file-only",
        action="store_true",
        help="Copy checkpoint files without registering LocalBackend or starting GPU/vLLM services.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination checkpoint in file-only mode.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = config_from_env()
    to_model = args.to_model or cfg.model_name
    if args.file_only:
        dest = file_only_fork_checkpoint(
            cfg,
            from_model=args.from_model,
            to_model=to_model,
            from_project=args.from_project,
            not_after_step=args.not_after_step,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        print(f"File-only forked checkpoint from {args.from_model} into {to_model}: {dest}")
        return

    model_cfg = replace(cfg, model_name=to_model)
    backend = make_local_backend(cfg.art_path)
    model = make_trainable_model(model_cfg)
    await register_trainable_model(model, backend, model_cfg)
    await backend._experimental_fork_checkpoint(
        model,
        from_model=args.from_model,
        from_project=args.from_project,
        not_after_step=args.not_after_step,
        verbose=args.verbose,
    )
    print(f"Forked checkpoint from {args.from_model} into {to_model}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
