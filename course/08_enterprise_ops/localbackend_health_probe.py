from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

from course.shared.art_compat import make_local_backend, make_trainable_model, register_trainable_model
from course.shared.config import config_from_env
from course.shared.data import load_cached_split, scenarios_from_records, write_sample_dataset
from course.shared.rollout import rollout_retail


def gpu_snapshot(label: str) -> None:
    print(f"=== gpu snapshot: {label} ===", flush=True)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("nvidia-smi unavailable", flush=True)
        return
    if result.returncode != 0:
        print((result.stderr or result.stdout).strip(), flush=True)
        return
    print(result.stdout.strip() or "<no gpu rows>", flush=True)


async def sleep_state(backend: object, model: object, label: str) -> None:
    print(f"=== vLLM sleep state: {label} ===", flush=True)
    service = getattr(backend, "_services", {}).get(getattr(model, "name", ""))
    if service is None:
        print("service=<none>", flush=True)
        return
    checker = getattr(service, "vllm_engine_is_sleeping", None)
    if checker is None:
        print("checker=<not exposed by this ART version>", flush=True)
        return
    try:
        print(await checker(), flush=True)
    except Exception as exc:
        print(f"failed {type(exc).__name__}: {exc}", flush=True)


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Probe ART LocalBackend rollout/train handoff, vLLM sleep state, and GPU memory."
    )
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="train")
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--continue-on-invalid", action="store_true")
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--model-name",
        default=os.getenv("ART_HEALTH_PROBE_MODEL_NAME", "retail-support-agent-health-probe"),
        help="Use a separate model name so the probe does not advance a course checkpoint.",
    )
    args = parser.parse_args()

    os.environ["ART_MODEL_NAME"] = args.model_name
    cfg = config_from_env()
    print(
        "CONFIG",
        json.dumps(
            {
                "project": cfg.project,
                "entity": cfg.entity,
                "art_path": cfg.art_path,
                "model_profile": cfg.model_profile,
                "base_model": cfg.base_model,
                "model_name": cfg.model_name,
                "request_logprobs": not args.no_logprobs,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    data_dir = Path(args.data_dir)
    if not (data_dir / f"{args.split}.jsonl").exists():
        write_sample_dataset(data_dir)
    records = load_cached_split(data_dir, args.split)
    scenarios = scenarios_from_records(records, split=args.split)
    if not scenarios:
        raise SystemExit(f"No scenarios found in {data_dir}/{args.split}.jsonl")
    scenario = scenarios[min(args.scenario_index, len(scenarios) - 1)]
    print("SCENARIO", scenario.id, flush=True)

    backend = make_local_backend(cfg.art_path)
    model = make_trainable_model(cfg)
    gpu_snapshot("before_register")
    await register_trainable_model(model, backend, cfg)
    await sleep_state(backend, model, "after_register")
    gpu_snapshot("after_register")

    import art

    terminate_on_invalid = False if args.continue_on_invalid else cfg.terminate_on_invalid
    groups = await art.gather_trajectory_groups(
        [
            art.TrajectoryGroup(
                rollout_retail(
                    model,
                    scenario,
                    split=args.split,
                    temperature=0.8,
                    max_turns=args.max_turns,
                    max_completion_tokens=args.max_completion_tokens,
                    terminate_on_invalid=terminate_on_invalid,
                    request_logprobs=not args.no_logprobs,
                )
                for _ in range(args.rollouts)
            )
        ],
        max_exceptions=0.5,
    )
    print("ROLLOUT groups", len(groups), [len(group.trajectories) for group in groups], flush=True)
    for group_index, group in enumerate(groups):
        for trajectory_index, trajectory in enumerate(group.trajectories):
            print(
                "TRAJ",
                group_index,
                trajectory_index,
                "reward",
                float(trajectory.reward or 0.0),
                "metrics",
                json.dumps(trajectory.metrics, sort_keys=True),
                flush=True,
            )
            if not args.skip_train:
                trajectory.metrics["probe_original_reward"] = float(trajectory.reward or 0.0)
                trajectory.reward = 1.0 if trajectory_index == 0 else 0.0
                trajectory.metrics["probe_forced_reward"] = float(trajectory.reward)
    await sleep_state(backend, model, "after_rollout")
    gpu_snapshot("after_rollout")

    if not args.skip_train and groups:
        result = await backend.train(model, groups, learning_rate=args.learning_rate)
        print(
            "TRAIN",
            json.dumps(
                {
                    "step": result.step,
                    "metrics": result.metrics,
                    "checkpoint_path": str(getattr(result, "checkpoint_path", "")),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        await sleep_state(backend, model, "after_train")
        gpu_snapshot("after_train")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
