from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import random
from pathlib import Path

from course.shared.art_compat import make_local_backend, make_trainable_model
from course.shared.config import config_from_env
from course.shared.data import load_cached_split, scenarios_from_records, write_sample_dataset
from course.shared.rollout import rollout_retail
from course.shared.tracing import init_weave


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run one GSPO-style sequence-level IS training step.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="train")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--rollouts-per-scenario", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--gpu-cost-per-hour-usd", type=float, default=None)
    parser.add_argument("--weave", action="store_true", help="Trace rollouts to Weave during this run.")
    args = parser.parse_args()

    cfg = config_from_env()
    if args.weave:
        init_weave(cfg.project)
    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)
    records = load_cached_split(data_dir, args.split)
    scenarios = scenarios_from_records(records, split=args.split)
    rng = random.Random(args.seed)

    backend = make_local_backend(cfg.art_path, gpu_cost_per_hour_usd=args.gpu_cost_per_hour_usd)
    model = make_trainable_model(cfg)
    await model.register(backend)

    import art

    for _ in range(args.steps):
        batch = rng.sample(scenarios, k=min(args.groups_per_step, len(scenarios)))
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout_retail(
                        model,
                        scenario,
                        split="train",
                        temperature=0.8,
                        request_logprobs=not args.no_logprobs,
                    )
                    for _ in range(args.rollouts_per_scenario)
                )
                for scenario in batch
            ),
            max_exceptions=0.1,
        )
        result = await backend.train(
            model,
            groups,
            learning_rate=args.learning_rate,
            importance_sampling_level="sequence",
        )
        await model.log(groups, metrics=result.metrics, step=result.step, split="train")
        print("GSPO step", result.step, result.metrics)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
