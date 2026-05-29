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

RULER_RUBRIC = """
Rank the retail support trajectories by: successful task completion, correct tool use,
policy compliance, clear customer communication, and avoiding unnecessary actions.
Prefer trajectories that solve the customer's issue without unsafe order mutations.
"""


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Train with ART RULER hybrid reward.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="train")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--rollouts-per-scenario", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--judge-model", default="openai/o3")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    cfg = config_from_env()
    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)
    records = load_cached_split(data_dir, args.split)
    scenarios = scenarios_from_records(records, split=args.split)
    rng = random.Random(args.seed)

    backend = make_local_backend(cfg.art_path)
    model = make_trainable_model(cfg)
    await model.register(backend)

    import art
    from art.rewards import ruler_score_group

    async def score_group(group):
        judged = await ruler_score_group(
            group,
            judge_model=args.judge_model,
            rubric=RULER_RUBRIC,
            swallow_exceptions=True,
        )
        if judged is None:
            return None
        for traj in judged:
            independent = float(traj.metrics.get("independent_reward", 0.0))
            ruler = float(traj.metrics.get("ruler_score", traj.reward))
            traj.reward = 0.7 * ruler + 0.3 * independent
            traj.metrics["hybrid_reward"] = traj.reward
        return judged

    for _ in range(args.steps):
        batch = rng.sample(scenarios, k=min(args.groups_per_step, len(scenarios)))
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout_retail(model, scenario, split="train")
                    for _ in range(args.rollouts_per_scenario)
                )
                for scenario in batch
            ),
            after_each=score_group,
            max_exceptions=0.1,
        )
        result = await backend.train(model, groups, learning_rate=args.learning_rate)
        await model.log(groups, metrics=result.metrics, step=result.step, split="train")
        print("step", result.step, result.metrics)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
