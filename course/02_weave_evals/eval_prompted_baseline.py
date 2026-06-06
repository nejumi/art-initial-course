from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from course.shared.art_compat import make_prompted_model
from course.shared.config import config_from_env
from course.shared.data import load_cached_split, scenarios_from_records, write_sample_dataset
from course.shared.rewards import REWARD_PROFILES
from course.shared.rollout import rollout_retail
from course.shared.tracing import init_weave, weave_op
from course.shared.schemas import RetailScenario

@weave_op("predict_retail_baseline")
async def predict(model: Any, scenario_dict: dict[str, Any], request_logprobs: bool) -> dict[str, Any]:
    scenario = RetailScenario(**scenario_dict)
    trajectory = await rollout_retail(
        model,
        scenario,
        split="eval",
        temperature=0.2,
        max_turns=scenario.max_turns,
        request_logprobs=request_logprobs,
    )
    return {
        "reward": trajectory.reward,
        "metrics": trajectory.metrics,
        "logs": trajectory.logs,
        "metadata": trajectory.metadata,
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a prompted baseline on retail scenarios.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--weave-evaluation", action="store_true")
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default=None)
    args = parser.parse_args()

    if args.reward_profile:
        os.environ["RETAIL_REWARD_PROFILE"] = args.reward_profile
    cfg = config_from_env()
    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)
    records = load_cached_split(data_dir, args.split, limit=args.limit)
    scenarios = scenarios_from_records(records, split=args.split)
    model = make_prompted_model(cfg, name="prompted-retail-baseline")

    if args.weave_evaluation:
        init_weave(cfg.project)
        import weave
        from course.weave_eval_scorers import SCORERS as scorers
        rows = [{"scenario_dict": scenario.to_dict()} for scenario in scenarios]

        class RetailBaselineModel(weave.Model):
            request_logprobs: bool = not args.no_logprobs

            @weave.op()
            async def predict(self, scenario_dict: dict[str, Any]) -> dict[str, Any]:
                return await predict(model, scenario_dict, self.request_logprobs)

        evaluation = weave.Evaluation(dataset=rows, scorers=scorers)
        result = await evaluation.evaluate(RetailBaselineModel())
        print(result)
        return

    groups = []
    import art
    for scenario in scenarios:
        traj = await rollout_retail(model, scenario, split="eval", temperature=0.2, request_logprobs=not args.no_logprobs)
        groups.append(art.TrajectoryGroup([traj]))
        print(scenario.id, traj.reward, traj.metrics)
    await model.log(groups, split="val")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
