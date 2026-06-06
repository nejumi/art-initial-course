from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import os
import random
from pathlib import Path

from course.shared.art_compat import make_local_backend, make_trainable_model, register_trainable_model
from course.shared.config import config_from_env
from course.shared.data import load_cached_split, scenarios_from_records, write_sample_dataset
from course.shared.rewards import REWARD_PROFILES
from course.shared.rl_sampling import reward_signal_metrics, split_reward_signal_groups
from course.shared.rollout import rollout_retail
from course.shared.tracing import init_weave
from course.shared.wandb_artifacts import log_checkpoint_artifact, use_wandb_artifact


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run one GSPO-style sequence-level IS training step.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--rollouts-per-scenario", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-sampling-rounds", type=int, default=4)
    parser.add_argument("--keep-zero-variance-groups", action="store_true")
    parser.add_argument("--continue-on-invalid", action="store_true", help="Keep outcome-style rollouts running after unexpected actions.")
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--gpu-cost-per-hour-usd", type=float, default=None)
    parser.add_argument("--weave", action="store_true", help="Trace rollouts to Weave during this run.")
    parser.add_argument("--data-artifact", default=None, help="Optional W&B dataset artifact URI to mark as RL input.")
    parser.add_argument("--parent-artifact", default=None, help="Optional W&B model artifact URI for the checkpoint this run starts from.")
    parser.add_argument("--checkpoint-artifact-name", default=None)
    parser.add_argument("--no-log-checkpoint-artifact", action="store_true")
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
    rng = random.Random(args.seed)

    backend = make_local_backend(cfg.art_path, gpu_cost_per_hour_usd=args.gpu_cost_per_hour_usd)
    model = make_trainable_model(cfg)
    await register_trainable_model(model, backend, cfg)
    terminate_on_invalid = False if args.continue_on_invalid else cfg.terminate_on_invalid
    data_artifact = args.data_artifact or os.getenv("RETAIL_DATA_ARTIFACT")
    parent_artifact = args.parent_artifact or os.getenv("ART_PARENT_ARTIFACT")
    if data_artifact:
        use_wandb_artifact(
            cfg,
            data_artifact,
            artifact_type="dataset",
            job_type="gspo",
            use_as="rl-training-data",
        )
    if parent_artifact:
        use_wandb_artifact(
            cfg,
            parent_artifact,
            artifact_type="model",
            job_type="gspo",
            use_as="initial-checkpoint",
        )

    import art

    last_step = await model.get_step()
    for _ in range(args.steps):
        groups = []
        all_sampled_groups = []
        all_dropped_groups = []
        submitted_groups = 0
        dropped_no_signal_groups = 0
        for _sampling_round in range(args.max_sampling_rounds):
            needed = max(args.groups_per_step - len(groups), 1)
            batch = rng.sample(scenarios, k=min(needed, len(scenarios)))
            sampled_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout_retail(
                            model,
                            scenario,
                            split=args.split,
                            temperature=args.temperature,
                            max_turns=args.max_turns,
                            max_completion_tokens=args.max_completion_tokens,
                            terminate_on_invalid=terminate_on_invalid,
                            request_logprobs=not args.no_logprobs,
                        )
                        for _ in range(args.rollouts_per_scenario)
                    )
                    for scenario in batch
                ),
                max_exceptions=0.1,
            )
            submitted_groups += len(sampled_groups)
            all_sampled_groups.extend(sampled_groups)
            if args.keep_zero_variance_groups:
                groups.extend(sampled_groups)
            else:
                trainable_groups, no_signal_groups = split_reward_signal_groups(sampled_groups)
                groups.extend(trainable_groups)
                all_dropped_groups.extend(no_signal_groups)
                dropped_no_signal_groups += len(no_signal_groups)
            if len(groups) >= args.groups_per_step:
                groups = groups[: args.groups_per_step]
                break
        dynamic_filter_metrics = {
            "data/step_num_groups_sampled_before_filter": float(submitted_groups),
            "data/step_num_groups_dropped_no_reward_signal": float(dropped_no_signal_groups),
            "data/step_trainable_group_fraction": float(len(groups) / submitted_groups) if submitted_groups else 0.0,
            **reward_signal_metrics(all_sampled_groups, prefix="data/sample"),
            **reward_signal_metrics(all_dropped_groups, prefix="data/dropped"),
        }
        if not groups:
            print("Skipping GSPO step because no sampled groups had reward variance.", dynamic_filter_metrics)
            await model.log(trajectories=None, metrics=dynamic_filter_metrics, step=last_step, split="train")
            continue
        result = await backend.train(
            model,
            groups,
            learning_rate=args.learning_rate,
            importance_sampling_level="sequence",
        )
        metrics = {**result.metrics, **dynamic_filter_metrics, **reward_signal_metrics(groups)}
        metrics.update(reward_signal_metrics(groups, prefix="data/train"))
        await model.log(groups, metrics=metrics, step=result.step, split="train")
        last_step = result.step
        print("GSPO step", result.step, metrics)
    if not args.no_log_checkpoint_artifact:
        log_checkpoint_artifact(
            cfg,
            stage="gspo",
            artifact_name=args.checkpoint_artifact_name,
            aliases=["gspo"],
            metadata={
                "algorithm": "gspo",
                "importance_sampling_level": "sequence",
                "steps": args.steps,
                "final_step": last_step,
                "groups_per_step": args.groups_per_step,
                "rollouts_per_scenario": args.rollouts_per_scenario,
                "max_sampling_rounds": args.max_sampling_rounds,
                "drop_zero_variance_groups": not args.keep_zero_variance_groups,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "max_turns": args.max_turns,
                "max_completion_tokens": args.max_completion_tokens,
                "scenario_limit": args.limit,
                "reward_profile": os.getenv("RETAIL_REWARD_PROFILE", "dense"),
                "terminate_on_invalid": terminate_on_invalid,
                "continue_on_invalid": args.continue_on_invalid,
                "request_logprobs": not args.no_logprobs,
                "data_artifact": data_artifact,
                "parent_artifact": parent_artifact,
            },
        )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
