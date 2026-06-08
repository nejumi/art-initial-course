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
from course.shared.metrics_io import append_step_metrics
from course.shared.rewards import REWARD_PROFILES
from course.shared.rl_sampling import reward_signal_metrics, split_reward_signal_groups
from course.shared.rollout import rollout_retail
from course.shared.tracing import init_weave
from course.shared.wandb_artifacts import ensure_wandb_run, log_checkpoint_artifact, log_wandb_metrics, use_wandb_artifact

RULER_RUBRIC = """
Rank the retail support trajectories by: successful task completion, correct tool use,
policy compliance, clear customer communication, and avoiding unnecessary actions.
Prefer trajectories that solve the customer's issue without unsafe order mutations.
"""


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Train with ART RULER hybrid reward.")
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--rollouts-per-scenario", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--judge-model", default="openai/gpt-5.5")
    parser.add_argument("--judge-effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--ruler-weight", type=float, default=0.3)
    parser.add_argument("--independent-weight", type=float, default=0.7)
    parser.add_argument(
        "--shuffle-ruler-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize trajectory order before LLM judging to reduce fixed-position bias.",
    )
    parser.add_argument("--seed", type=int, default=11)
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
    parser.add_argument("--metrics-jsonl", default=None, help="Optional local JSONL path for per-step train metrics.")
    args = parser.parse_args()

    if args.reward_profile:
        os.environ["RETAIL_REWARD_PROFILE"] = args.reward_profile
    cfg = config_from_env()
    if args.weave:
        ensure_wandb_run(cfg, job_type="ruler-grpo")
        init_weave()
    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)
    records = load_cached_split(data_dir, args.split, limit=args.limit)
    scenarios = scenarios_from_records(records, split=args.split)
    rng = random.Random(args.seed)
    minimum_scenarios = args.steps * args.groups_per_step
    if len(scenarios) < minimum_scenarios:
        raise ValueError(
            "RULER-GRPO requires at least one unique scenario per trajectory group: "
            f"steps={args.steps} * groups_per_step={args.groups_per_step} "
            f"needs {minimum_scenarios}, got {len(scenarios)}."
        )
    scenario_pool = list(scenarios)
    rng.shuffle(scenario_pool)

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
            job_type="ruler-grpo",
            use_as="rl-training-data",
        )
    if parent_artifact:
        use_wandb_artifact(
            cfg,
            parent_artifact,
            artifact_type="model",
            job_type="ruler-grpo",
            use_as="initial-checkpoint",
        )

    import art
    from art.rewards import ruler_score_group

    async def score_group(group):
        if args.shuffle_ruler_trajectories:
            trajectories = list(group.trajectories)
            rng.shuffle(trajectories)
            group = art.TrajectoryGroup(
                trajectories,
                exceptions=[],
                metadata=group.metadata.copy(),
                metrics=group.metrics.copy(),
                logs=group.logs.copy(),
            )
        judged = await ruler_score_group(
            group,
            judge_model=args.judge_model,
            rubric=RULER_RUBRIC,
            extra_litellm_params={"reasoning_effort": args.judge_effort},
            swallow_exceptions=True,
        )
        if judged is None:
            return None
        for traj in judged:
            independent = float(traj.metrics.get("independent_reward", 0.0))
            ruler = float(traj.metrics.get("ruler_score", traj.reward))
            traj.reward = args.ruler_weight * ruler + args.independent_weight * independent
            traj.metrics["hybrid_reward"] = traj.reward
            traj.metrics["ruler_weight"] = args.ruler_weight
            traj.metrics["independent_weight"] = args.independent_weight
        return judged

    last_step = await model.get_step()
    for _ in range(args.steps):
        groups = []
        all_sampled_groups = []
        all_dropped_groups = []
        submitted_groups = 0
        dropped_no_signal_groups = 0
        for _sampling_round in range(args.max_sampling_rounds):
            needed = max(args.groups_per_step - len(groups), 1)
            if not scenario_pool:
                break
            take = min(needed, len(scenario_pool))
            batch = [scenario_pool.pop() for _ in range(take)]
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
                after_each=score_group,
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
            "data/step_unique_scenarios_remaining": float(len(scenario_pool)),
            **reward_signal_metrics(all_sampled_groups, prefix="data/sample"),
            **reward_signal_metrics(all_dropped_groups, prefix="data/dropped"),
        }
        if not groups:
            print("Skipping RULER-GRPO step because no sampled groups had reward variance.", dynamic_filter_metrics)
            await model.log(trajectories=None, metrics=dynamic_filter_metrics, step=last_step, split="train")
            log_wandb_metrics(cfg, dynamic_filter_metrics, job_type="ruler-grpo", step=last_step)
            append_step_metrics(
                args.metrics_jsonl,
                step=last_step,
                algorithm="ruler-grpo",
                metrics=dynamic_filter_metrics,
                skipped=True,
            )
            continue
        result = await backend.train(model, groups, learning_rate=args.learning_rate)
        metrics = {
            **result.metrics,
            **dynamic_filter_metrics,
            **reward_signal_metrics(groups),
            **reward_signal_metrics(groups, prefix="data/train"),
        }
        await model.log(groups, metrics=metrics, step=result.step, split="train")
        log_wandb_metrics(cfg, metrics, job_type="ruler-grpo", step=result.step)
        last_step = result.step
        append_step_metrics(args.metrics_jsonl, step=result.step, algorithm="ruler-grpo", metrics=metrics)
        print("step", result.step, metrics)
    if not args.no_log_checkpoint_artifact:
        log_checkpoint_artifact(
            cfg,
            stage="ruler-grpo",
            artifact_name=args.checkpoint_artifact_name,
            aliases=["ruler-grpo"],
            metadata={
                "algorithm": "grpo",
                "reward_model": "ruler-hybrid",
                "ruler_rubric": RULER_RUBRIC.strip(),
                "judge_model": args.judge_model,
                "judge_effort": args.judge_effort,
                "ruler_weight": args.ruler_weight,
                "independent_weight": args.independent_weight,
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
                "independent_reward_profile": os.getenv("RETAIL_REWARD_PROFILE", "dense"),
                "shuffle_ruler_trajectories": args.shuffle_ruler_trajectories,
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
