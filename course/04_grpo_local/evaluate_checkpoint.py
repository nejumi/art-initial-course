from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import os
from pathlib import Path

from course.shared.art_compat import make_local_backend, make_trainable_model, register_trainable_model
from course.shared.config import config_from_env
from course.shared.data import load_cached_split, scenarios_from_records, write_jsonl, write_sample_dataset
from course.shared.retail_env import choice_to_message
from course.shared.rewards import REWARD_PROFILES
from course.shared.rollout import rollout_retail
from course.shared.tracing import init_weave
from course.shared.wandb_artifacts import use_wandb_artifact


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current ART checkpoint on retail scenarios.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="data/retail/eval_current.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rollouts-per-scenario", type=int, default=1)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--continue-on-invalid", action="store_true")
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--weave", action="store_true", help="Trace eval rollouts to Weave during this run.")
    parser.add_argument("--data-artifact", default=None, help="Optional W&B dataset artifact URI to mark as eval input.")
    parser.add_argument("--model-artifact", default=None, help="Optional W&B model artifact URI for the evaluated checkpoint.")
    parser.add_argument("--include-messages", action="store_true")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default=None)
    args = parser.parse_args()

    if args.reward_profile:
        os.environ["RETAIL_REWARD_PROFILE"] = args.reward_profile
    cfg = config_from_env()
    if args.weave:
        init_weave(cfg.project)
    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)
    records = load_cached_split(data_dir, args.split, limit=args.limit)
    scenarios = scenarios_from_records(records, split=args.split)

    backend = make_local_backend(cfg.art_path)
    model = make_trainable_model(cfg)
    await register_trainable_model(model, backend, cfg)
    data_artifact = args.data_artifact or os.getenv("RETAIL_DATA_ARTIFACT")
    model_artifact = args.model_artifact or os.getenv("ART_MODEL_ARTIFACT")
    if data_artifact:
        use_wandb_artifact(
            cfg,
            data_artifact,
            artifact_type="dataset",
            job_type="eval",
            use_as="eval-data",
        )
    if model_artifact:
        use_wandb_artifact(
            cfg,
            model_artifact,
            artifact_type="model",
            job_type="eval",
            use_as="evaluated-checkpoint",
        )

    import art

    groups = []
    rows = []
    rollouts_per_scenario = max(1, args.rollouts_per_scenario)
    for scenario in scenarios:
        trajectories = []
        for rollout_index in range(rollouts_per_scenario):
            traj = await rollout_retail(
                model,
                scenario,
                split="val",
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                terminate_on_invalid=False if args.continue_on_invalid else None,
                request_logprobs=not args.no_logprobs,
            )
            trajectories.append(traj)
            row = {
                "scenario_id": scenario.id,
                "rollout_index": rollout_index,
                "reward": traj.reward,
                "metrics": traj.metrics,
                "logs": traj.logs,
            }
            if args.include_messages:
                row["messages"] = [choice_to_message(item) for item in traj.messages_and_choices]
                row["expected_tool_names"] = scenario.expected_tool_names
                row["expected_final_text"] = scenario.expected_final_text
            rows.append(row)
            print(scenario.id, rollout_index, traj.reward, traj.metrics)
        groups.append(art.TrajectoryGroup(trajectories))
    await model.log(groups, split="val")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} eval rows to {args.output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
