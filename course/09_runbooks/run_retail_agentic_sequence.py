from __future__ import annotations

from pathlib import Path
import argparse
import os
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_command(
    label: str,
    command: Sequence[str],
    *,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    printable = " ".join(command)
    print(f"\n=== {label} ===")
    print(printable)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def path_arg(path: Path | str) -> str:
    return str(path)


def maybe_append(flag: str, value: object | None) -> list[str]:
    if value is None or value == "":
        return []
    return [flag, str(value)]


def artifact_uri(name: str, alias: str = "latest") -> str:
    return f"{name}:{alias}"


def retail_model_name(run_slug: str, stage: str) -> str:
    return f"retail-support-agent-{run_slug}-{stage}"


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ART_BASE_MODEL": args.base_model,
            "ART_MODEL_PROFILE": "custom",
            "ART_PATH": path_arg(args.art_path),
            "RETAIL_REWARD_PROFILE": args.reward_profile,
            "RETAIL_DATA_ARTIFACT_NAME": args.data_artifact_name,
            "RETAIL_DATA_ARTIFACT": artifact_uri(args.data_artifact_name),
            "ART_ROLLOUT_MAX_COMPLETION_TOKENS": str(args.max_completion_tokens),
            "ART_MAX_SEQ_LENGTH": str(args.art_seq_length),
            "ART_VLLM_RUNTIME_CACHE_DIR": path_arg(args.vllm_runtime_cache_dir),
            "HF_HOME": path_arg(args.hf_home),
            "TRANSFORMERS_CACHE": path_arg(args.hf_home / "transformers"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if args.vllm_gpu_memory_utilization is not None:
        env["ART_VLLM_GPU_MEMORY_UTILIZATION"] = str(args.vllm_gpu_memory_utilization)
    if args.vllm_max_model_len is not None:
        env["ART_VLLM_MAX_MODEL_LEN"] = str(args.vllm_max_model_len)
    if args.vllm_max_num_batched_tokens is not None:
        env["ART_VLLM_MAX_NUM_BATCHED_TOKENS"] = str(args.vllm_max_num_batched_tokens)
    if args.vllm_max_num_seqs is not None:
        env["ART_VLLM_MAX_NUM_SEQS"] = str(args.vllm_max_num_seqs)
    if args.continue_on_invalid:
        env["RETAIL_TERMINATE_ON_INVALID"] = "false"
    return env


def weave_args(args: argparse.Namespace) -> list[str]:
    return [] if args.no_weave else ["--weave"]


def build_or_refresh_data(args: argparse.Namespace, env: dict[str, str]) -> None:
    if not args.build_bridge and (args.data_dir / "train.jsonl").exists():
        return
    if not (args.source_dir / "train.jsonl").exists():
        run_command(
            "download retail data",
            [
                sys.executable,
                "-B",
                "course/03_sft_warmup/download_tau_retail.py",
                "--output-dir",
                path_arg(args.source_dir),
            ],
            env=env,
            dry_run=args.dry_run,
        )
        run_command(
            "make full-trajectory SFT data",
            [
                sys.executable,
                "-B",
                "course/03_sft_warmup/make_sft_jsonl.py",
                "--data-dir",
                path_arg(args.source_dir),
            ],
            env=env,
            dry_run=args.dry_run,
        )
    run_command(
        "build bridge curriculum",
        [
            sys.executable,
            "-B",
            "course/03_sft_warmup/make_bridge_curriculum.py",
            "--source-dir",
            path_arg(args.source_dir),
            "--output-dir",
            path_arg(args.data_dir),
            "--max-state-actions",
            str(args.bridge_max_state_actions),
            "--max-tool-calls",
            str(args.bridge_max_tool_calls),
            "--max-turns",
            str(args.bridge_max_turns),
            "--max-per-task",
            str(args.bridge_max_per_task),
            "--holdout-modulo",
            str(args.bridge_holdout_mod),
        ],
        env=env,
        dry_run=args.dry_run,
    )


def evaluate_command(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    split: str,
    limit: int,
    output: Path,
    model_artifact: str | None = None,
    include_messages: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        "course/04_grpo_local/evaluate_checkpoint.py",
        "--data-dir",
        path_arg(data_dir),
        "--split",
        split,
        "--limit",
        str(limit),
        "--temperature",
        str(args.eval_temperature),
        "--rollouts-per-scenario",
        str(args.eval_rollouts_per_scenario),
        "--max-completion-tokens",
        str(args.max_completion_tokens),
        "--reward-profile",
        args.reward_profile,
        "--output",
        path_arg(output),
        "--data-artifact",
        artifact_uri(args.data_artifact_name),
    ]
    if model_artifact:
        command += ["--model-artifact", model_artifact]
    if include_messages:
        command.append("--include-messages")
    if args.continue_on_invalid:
        command.append("--continue-on-invalid")
    command += weave_args(args)
    return command


def train_sft(args: argparse.Namespace, env: dict[str, str], report_dir: Path) -> str:
    anchor_model = retail_model_name(args.run_slug, "sft-anchor")
    sft_env = {**env, "ART_MODEL_NAME": anchor_model}
    sft_file = args.sft_file or args.data_dir / "sft_train_next_action.jsonl"
    command = [
        sys.executable,
        "-B",
        "course/03_sft_warmup/train_sft_local.py",
        "--file",
        path_arg(sft_file),
        "--epochs",
        str(args.sft_epochs),
        "--batch-size",
        str(args.sft_batch_size),
        "--peak-lr",
        str(args.sft_peak_lr),
        "--chunk-size-batches",
        str(args.sft_chunk_size_batches),
        "--sft-mask-mode",
        args.sft_mask_mode,
        "--data-artifact",
        artifact_uri(args.data_artifact_name),
        "--gpu-cost-per-hour-usd",
        str(args.gpu_cost_per_hour_usd),
    ]
    command += maybe_append("--max-steps", args.sft_max_steps)
    command += weave_args(args)
    run_command("SFT anchor", command, env=sft_env, dry_run=args.dry_run)
    if args.skip_sft_eval:
        return anchor_model
    run_command(
        "eval SFT anchor",
        evaluate_command(
            args,
            data_dir=args.data_dir,
            split=args.eval_split,
            limit=args.eval_limit,
            output=report_dir / "eval_01_sft_anchor.jsonl",
            model_artifact=artifact_uri(f"{anchor_model}-checkpoint", "sft-anchor"),
        ),
        env=sft_env,
        dry_run=args.dry_run,
    )
    return anchor_model


def train_rl_branch(
    args: argparse.Namespace,
    env: dict[str, str],
    *,
    algo: str,
    anchor_model: str,
    report_dir: Path,
) -> str:
    branch_stage = f"{algo}-{args.rl_suffix}"
    branch_model = retail_model_name(args.run_slug, branch_stage)
    branch_env = {**env, "ART_MODEL_NAME": branch_model}
    parent_artifact = artifact_uri(f"{anchor_model}-checkpoint", "sft-anchor")
    train_script = (
        "course/04_grpo_local/train_grpo_local.py"
        if algo == "grpo"
        else "course/06_gspo_and_configs/train_gspo_sequence.py"
    )
    run_command(
        f"fork {algo} branch from SFT anchor",
        [
            sys.executable,
            "-B",
            "course/08_enterprise_ops/fork_checkpoint.py",
            "--from-model",
            anchor_model,
            "--verbose",
        ],
        env=branch_env,
        dry_run=args.dry_run,
    )
    command = [
        sys.executable,
        "-B",
        train_script,
        "--data-dir",
        path_arg(args.data_dir),
        "--split",
        args.rl_split,
        "--limit",
        str(args.rl_train_limit),
        "--steps",
        str(args.rl_steps),
        "--groups-per-step",
        str(args.rl_groups_per_step),
        "--rollouts-per-scenario",
        str(args.rl_rollouts_per_scenario),
        "--learning-rate",
        str(args.rl_learning_rate),
        "--temperature",
        str(args.rl_temperature),
        "--max-completion-tokens",
        str(args.max_completion_tokens),
        "--reward-profile",
        args.reward_profile,
        "--data-artifact",
        artifact_uri(args.data_artifact_name),
        "--parent-artifact",
        parent_artifact,
        "--gpu-cost-per-hour-usd",
        str(args.gpu_cost_per_hour_usd),
    ]
    command += maybe_append("--max-turns", args.rl_max_turns)
    if args.continue_on_invalid:
        command.append("--continue-on-invalid")
    if args.no_logprobs:
        command.append("--no-logprobs")
    command += weave_args(args)
    run_command(f"{algo.upper()} training", command, env=branch_env, dry_run=args.dry_run)
    run_command(
        f"eval {algo} train subset",
        evaluate_command(
            args,
            data_dir=args.data_dir,
            split=args.rl_split,
            limit=args.rl_train_limit,
            output=report_dir / f"eval_{algo}_{args.rl_suffix}_train{args.rl_train_limit}.jsonl",
            model_artifact=artifact_uri(f"{branch_model}-checkpoint"),
            include_messages=True,
        ),
        env=branch_env,
        dry_run=args.dry_run,
    )
    run_command(
        f"eval {algo} validation",
        evaluate_command(
            args,
            data_dir=args.data_dir,
            split=args.eval_split,
            limit=args.eval_limit,
            output=report_dir / f"eval_{algo}_{args.rl_suffix}_validation.jsonl",
            model_artifact=artifact_uri(f"{branch_model}-checkpoint"),
            include_messages=True,
        ),
        env=branch_env,
        dry_run=args.dry_run,
    )
    return branch_model


def compare_results(
    args: argparse.Namespace,
    env: dict[str, str],
    *,
    report_dir: Path,
    anchor_model: str,
    branch_models: dict[str, str],
) -> None:
    paths = []
    stages = []
    model_artifacts = []
    baseline_path = report_dir / "eval_00_baseline.jsonl"
    if baseline_path.exists() or args.dry_run:
        paths.append(path_arg(baseline_path))
        stages.append("baseline")
        model_artifacts.append("-")
    sft_path = report_dir / "eval_01_sft_anchor.jsonl"
    if sft_path.exists() or args.dry_run:
        paths.append(path_arg(sft_path))
        stages.append("sft_anchor")
        model_artifacts.append(artifact_uri(f"{anchor_model}-checkpoint", "sft-anchor"))
    expected_branch_models = {
        algo: retail_model_name(args.run_slug, f"{algo}-{args.rl_suffix}")
        for algo in split_csv(args.rl_algos)
        if algo in {"grpo", "gspo"}
    }
    expected_branch_models.update(branch_models)
    for algo, branch_model in expected_branch_models.items():
        eval_path = report_dir / f"eval_{algo}_{args.rl_suffix}_validation.jsonl"
        if eval_path.exists() or args.dry_run:
            paths.append(path_arg(eval_path))
            stages.append(f"{algo}_{args.rl_suffix}")
            model_artifacts.append(artifact_uri(f"{branch_model}-checkpoint"))
    if len(paths) < 2:
        print("Skipping comparison because fewer than two eval outputs are available.")
        return
    command = [
        sys.executable,
        "-B",
        "course/02_weave_evals/compare_checkpoints.py",
        *paths,
        "--stages",
        *stages,
        "--model",
        args.base_model,
        "--model-artifacts",
        *model_artifacts,
        "--data-artifact",
        artifact_uri(args.data_artifact_name),
        "--output-md",
        path_arg(report_dir / "checkpoint_eval_comparison.md"),
        "--output-compact-md",
        path_arg(report_dir / "checkpoint_eval_summary.md"),
        "--output-csv",
        path_arg(report_dir / "checkpoint_eval_comparison.csv"),
        "--output-compact-csv",
        path_arg(report_dir / "checkpoint_eval_summary.csv"),
        "--run-name",
        f"{args.run_slug}-checkpoint-eval-comparison",
    ]
    if not args.no_wandb_compare:
        command.append("--wandb")
    run_command("checkpoint comparison", command, env=env, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the retail ART hands-on sequence: data artifact, baseline eval, "
            "next-action SFT, independent GRPO/GSPO branches, eval, and W&B comparison."
        )
    )
    parser.add_argument("--run-slug", default="lfm25-8b-a1b-bridge-state1")
    parser.add_argument("--base-model", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--art-path", type=Path, default=PROJECT_ROOT / ".art" / "course_runs")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "retail_bridge_state1")
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data" / "retail")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--data-artifact-name", default="retail-course-data-bridge-state1")
    parser.add_argument("--build-bridge", action="store_true")
    parser.add_argument("--bridge-max-state-actions", type=int, default=1)
    parser.add_argument("--bridge-max-tool-calls", type=int, default=6)
    parser.add_argument("--bridge-max-turns", type=int, default=28)
    parser.add_argument("--bridge-max-per-task", type=int, default=2)
    parser.add_argument("--bridge-holdout-mod", type=int, default=5)
    parser.add_argument("--reward-profile", default="tau_irc")
    parser.add_argument("--max-completion-tokens", type=int, default=768)
    parser.add_argument("--art-seq-length", type=int, default=16384)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-runtime-cache-dir", type=Path, default=PROJECT_ROOT / ".art" / "vllm_runtime_cache")
    parser.add_argument("--hf-home", type=Path, default=PROJECT_ROOT / ".art" / "hf_home")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-limit", type=int, default=48)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-rollouts-per-scenario", type=int, default=1)
    parser.add_argument("--sft-file", type=Path, default=None)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--sft-batch-size", type=int, default=1)
    parser.add_argument("--sft-peak-lr", type=float, default=1e-5)
    parser.add_argument("--sft-max-steps", type=int, default=96)
    parser.add_argument("--sft-chunk-size-batches", type=int, default=12)
    parser.add_argument("--sft-mask-mode", choices=["all-assistant", "last-assistant"], default="last-assistant")
    parser.add_argument("--rl-algos", default="grpo,gspo")
    parser.add_argument("--rl-suffix", default="bridge-s32-lr2e6")
    parser.add_argument("--rl-split", default="train")
    parser.add_argument("--rl-train-limit", type=int, default=24)
    parser.add_argument("--rl-steps", type=int, default=32)
    parser.add_argument("--rl-groups-per-step", type=int, default=4)
    parser.add_argument("--rl-rollouts-per-scenario", type=int, default=8)
    parser.add_argument("--rl-learning-rate", type=float, default=2e-6)
    parser.add_argument("--rl-temperature", type=float, default=0.9)
    parser.add_argument("--rl-max-turns", type=int, default=None)
    parser.add_argument(
        "--continue-on-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep tau-style rollouts running after invalid read/state actions so outcome rewards remain observable.",
    )
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--gpu-cost-per-hour-usd", type=float, default=3.0)
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-sft-eval", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--no-weave", action="store_true")
    parser.add_argument("--no-wandb-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.source_dir = args.source_dir.resolve()
    args.art_path = args.art_path.resolve()
    args.vllm_runtime_cache_dir = args.vllm_runtime_cache_dir.resolve()
    args.hf_home = args.hf_home.resolve()
    if args.sft_file is not None:
        args.sft_file = args.sft_file.resolve()
    report_dir = (args.report_dir or (args.data_dir / f"{args.run_slug}_runbook")).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    env = base_env(args)

    build_or_refresh_data(args, env)
    run_command(
        "log data artifact",
        [
            sys.executable,
            "-B",
            "course/07_models_registry_weave/log_data_artifacts.py",
            "--data-dir",
            path_arg(args.data_dir),
            "--artifact-name",
            args.data_artifact_name,
            "--aliases",
            "latest",
            "tau-retail-bridge",
        ],
        env=env,
        dry_run=args.dry_run,
    )
    if not args.skip_baseline_eval:
        baseline_env = {**env, "ART_MODEL_NAME": retail_model_name(args.run_slug, "baseline")}
        run_command(
            "eval baseline",
            evaluate_command(
                args,
                data_dir=args.data_dir,
                split=args.eval_split,
                limit=args.eval_limit,
                output=report_dir / "eval_00_baseline.jsonl",
            ),
            env=baseline_env,
            dry_run=args.dry_run,
        )

    anchor_model = retail_model_name(args.run_slug, "sft-anchor")
    if not args.skip_sft:
        anchor_model = train_sft(args, env, report_dir)
    branch_models: dict[str, str] = {}
    if not args.skip_rl:
        for algo in split_csv(args.rl_algos):
            if algo not in {"grpo", "gspo"}:
                raise SystemExit(f"Unsupported --rl-algos value: {algo}")
            branch_models[algo] = train_rl_branch(
                args,
                env,
                algo=algo,
                anchor_model=anchor_model,
                report_dir=report_dir,
            )
    if not args.skip_compare:
        compare_results(args, env, report_dir=report_dir, anchor_model=anchor_model, branch_models=branch_models)
    print(f"\nRunbook complete. Report directory: {report_dir}")


if __name__ == "__main__":
    main()
