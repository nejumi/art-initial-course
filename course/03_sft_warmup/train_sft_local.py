from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
from pathlib import Path

from course.shared.art_compat import make_local_backend, make_trainable_model
from course.shared.config import config_from_env


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run local ART SFT on retail JSONL.")
    parser.add_argument("--file", default="data/retail/sft_train.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--peak-lr", type=float, default=2e-4)
    parser.add_argument("--schedule-type", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gpu-cost-per-hour-usd", type=float, default=None)
    args = parser.parse_args()

    if not Path(args.file).exists():
        raise SystemExit(f"Missing {args.file}. Run make_sft_jsonl.py first.")

    cfg = config_from_env()
    backend = make_local_backend(cfg.art_path, gpu_cost_per_hour_usd=args.gpu_cost_per_hour_usd)
    model = make_trainable_model(cfg)
    await model.register(backend)

    from art.utils.sft import train_sft_from_file

    await train_sft_from_file(
        model=model,
        file_path=args.file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        peak_lr=args.peak_lr,
        schedule_type=args.schedule_type,
        final_step=args.max_steps,
        verbose=True,
    )
    print("SFT complete. Current step:", await model.get_step())


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
