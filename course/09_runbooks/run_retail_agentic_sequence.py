from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from typing import Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from course.shared.config import MODEL_PROFILES, config_from_env

DEFAULT_RUN_CONFIG = PROJECT_ROOT / "course" / "09_runbooks" / "config.yaml"
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "course" / "09_runbooks" / "base_config.yaml"
PROFILE_PATH_KEYS = {
    "art_path",
    "data_dir",
    "source_dir",
    "report_dir",
    "sft_file",
    "vllm_runtime_cache_dir",
    "hf_home",
}


def load_yaml_mapping(config_path: Path, *, required: bool = True) -> dict[str, object]:
    if not config_path.exists():
        if required:
            raise SystemExit(f"Config file not found: {config_path}")
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must be a YAML mapping: {config_path}")
    return data


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


def stage_env(
    env: dict[str, str],
    *,
    stage: str,
    kind: str,
    algo: str | None = None,
    split: str | None = None,
) -> dict[str, str]:
    scoped = {
        **env,
        "COURSE_RUN_STAGE": stage,
        "COURSE_RUN_KIND": kind,
    }
    if algo:
        scoped["COURSE_RUN_ALGO"] = algo
    else:
        scoped.pop("COURSE_RUN_ALGO", None)
    if split:
        scoped["COURSE_RUN_SPLIT"] = split
    else:
        scoped.pop("COURSE_RUN_SPLIT", None)
    return scoped


def artifact_uri(name: str, alias: str = "latest") -> str:
    return f"{name}:{alias}"


def retail_model_name(run_slug: str, stage: str) -> str:
    return f"retail-support-agent-{run_slug}-{stage}"


def candidate_model_name(branch_model: str, step: int) -> str:
    return f"{branch_model}-candidate-step{step:04d}"


def read_top_candidate_step(path: Path) -> int | None:
    steps = read_candidate_steps(path)
    return steps[0] if steps else None


def read_candidate_steps(path: Path) -> list[int]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return []
    steps: list[int] = []
    for candidate in candidates:
        step = candidate.get("step") if isinstance(candidate, dict) else None
        if step is not None:
            steps.append(int(step))
    return steps


def candidate_step_from_eval_path(path: Path) -> int | None:
    marker = "_candidate_step"
    if marker not in path.stem:
        return None
    raw = path.stem.split(marker, 1)[1].split("_", 1)[0]
    return int(raw) if raw.isdigit() else None


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_profile_path(value: object) -> object:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_profile_defaults(args: dict[str, object], *, label: str) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for key, value in args.items():
        if key in {"description", "notes"}:
            continue
        if value is None:
            continue
        defaults[key] = resolve_profile_path(value) if key in PROFILE_PATH_KEYS else value
    return defaults


def load_run_profile_defaults(profile_name: str | None, config_path: Path) -> dict[str, object]:
    if not profile_name:
        return {}
    data = load_yaml_mapping(config_path)
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise SystemExit(f"Run profile config must contain a 'profiles' mapping: {config_path}")
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(key) for key in profiles))
        raise SystemExit(f"Unknown --run-profile={profile_name!r}. Available profiles: {available}")
    args = profile.get("args", profile)
    if not isinstance(args, dict):
        raise SystemExit(f"Run profile {profile_name!r} must contain an args mapping.")
    return normalize_profile_defaults(args, label=f"profile {profile_name!r}")


def configured_run_profile(config_path: Path) -> str | None:
    data = load_yaml_mapping(config_path, required=False)
    run_profile = data.get("run_profile")
    if run_profile in {None, ""}:
        return None
    if not isinstance(run_profile, str):
        raise SystemExit("Run config 'run_profile' must be a string when set.")
    return run_profile


def load_user_config_defaults(config_path: Path, base_config_path: Path) -> dict[str, object]:
    data = load_yaml_mapping(config_path, required=False)
    defaults: dict[str, object] = {}

    model_profile = data.get("model_profile")
    if model_profile not in {None, "", "from_env"}:
        if not isinstance(model_profile, str):
            raise SystemExit("Run config 'model_profile' must be a string when set.")
        defaults["model_profile"] = model_profile
        if model_profile == "custom":
            base_model = data.get("base_model")
            if not base_model:
                raise SystemExit("Run config uses model_profile: custom, so base_model must be set.")
            defaults["base_model"] = str(base_model)
        else:
            try:
                defaults["base_model"] = MODEL_PROFILES[model_profile]
            except KeyError as exc:
                allowed = ", ".join(["from_env", "custom", *sorted(MODEL_PROFILES)])
                raise SystemExit(f"Unknown model_profile={model_profile!r}. Use one of: {allowed}.") from exc

    base_model = data.get("base_model")
    if base_model not in {None, ""}:
        defaults["base_model"] = str(base_model)

    gpu_memory_preset = data.get("gpu_memory_preset")
    if gpu_memory_preset not in {None, "", "standard"}:
        if not isinstance(gpu_memory_preset, str):
            raise SystemExit("Run config 'gpu_memory_preset' must be a string when set.")
        base_data = load_yaml_mapping(base_config_path)
        presets = base_data.get("gpu_memory_presets") or {}
        if not isinstance(presets, dict):
            raise SystemExit("Base config 'gpu_memory_presets' must be a mapping when set.")
        preset = presets.get(gpu_memory_preset)
        if not isinstance(preset, dict):
            available = ", ".join(sorted(str(key) for key in presets))
            raise SystemExit(f"Unknown gpu_memory_preset={gpu_memory_preset!r}. Available presets: {available}")
        defaults.update(normalize_profile_defaults(preset, label=f"gpu_memory_preset {gpu_memory_preset!r}"))

    overrides = data.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise SystemExit("Run config 'overrides' must be a mapping when set.")
    defaults.update(normalize_profile_defaults(overrides, label="overrides"))
    return defaults


def preparse_profile_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-profile", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--profile-config", type=Path, default=None, help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args(argv)
    args.config = args.config.resolve()
    args.base_config = (args.profile_config or args.base_config).resolve()
    args.run_profile = args.run_profile or configured_run_profile(args.config)
    return args


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ART_BASE_MODEL": args.base_model,
            "ART_MODEL_PROFILE": args.model_profile,
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
            "COURSE_RUN_PROFILE": args.run_profile or "custom",
            "COURSE_RUN_SLUG": args.run_slug,
            "COURSE_IGNORE_RUN_CONFIG": "1",
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
    retail_tool_use_instruction = getattr(args, "retail_tool_use_instruction", None)
    if retail_tool_use_instruction:
        env["RETAIL_TOOL_USE_INSTRUCTION"] = str(retail_tool_use_instruction)
    if args.continue_on_invalid:
        env["RETAIL_TERMINATE_ON_INVALID"] = "false"
    return env


def weave_args(args: argparse.Namespace) -> list[str]:
    return [] if args.no_weave else ["--weave"]


def build_or_refresh_data(args: argparse.Namespace, env: dict[str, str]) -> None:
    if not args.build_bridge and (args.data_dir / "train.jsonl").exists():
        if args.include_teacher_sft or args.include_areal_sft or args.include_success_trace_sft:
            build_augmented_sft(args, env)
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
            "--sft-remainder",
            str(args.bridge_sft_remainder),
            "--validation-remainder",
            str(args.bridge_validation_remainder),
            "--test-remainder",
            str(args.bridge_test_remainder),
        ],
        env=env,
        dry_run=args.dry_run,
    )
    if args.include_teacher_sft or args.include_areal_sft or args.include_success_trace_sft:
        build_augmented_sft(args, env)


def augmented_sft_file(args: argparse.Namespace) -> Path:
    labels = []
    if args.include_teacher_sft:
        labels.append("teacher")
    if args.include_areal_sft:
        labels.append("areal")
    if args.include_success_trace_sft:
        labels.append("success_trace")
    if labels:
        return args.data_dir / f"sft_next_action_{'_'.join(labels)}_mix.jsonl"
    return args.data_dir / "sft_next_action.jsonl"


def build_augmented_sft(args: argparse.Namespace, env: dict[str, str]) -> None:
    inputs = [args.data_dir / "sft_next_action.jsonl"]
    limits = [args.bridge_sft_limit]
    source_labels = ["bridge"]

    if args.include_teacher_sft:
        teacher_file = build_teacher_sft(args, env)
        inputs.append(teacher_file)
        limits.append(args.teacher_sft_limit)
        source_labels.append("teacher")
    if args.include_areal_sft:
        areal_file = build_areal_sft(args, env)
        inputs.append(areal_file)
        limits.append(args.areal_sft_limit)
        source_labels.append("areal")
    if args.include_success_trace_sft:
        success_trace_file = build_success_trace_sft(args, env)
        inputs.append(success_trace_file)
        limits.append(args.success_trace_sft_limit)
        source_labels.append("success-trace")

    mixed_file = augmented_sft_file(args)
    run_command(
        f"mix {'/'.join(source_labels)} SFT data",
        [
            sys.executable,
            "-B",
            "course/03_sft_warmup/mix_sft_jsonl.py",
            "--inputs",
            *[path_arg(path) for path in inputs],
            "--limits",
            *[str(limit) for limit in limits],
            "--output",
            path_arg(mixed_file),
            "--summary",
            path_arg(mixed_file.with_suffix(".summary.json")),
        ],
        env=env,
        dry_run=args.dry_run,
    )


def build_teacher_sft(args: argparse.Namespace, env: dict[str, str]) -> Path:
    teacher_file = args.data_dir / "sft_teacher_retail_next_action.jsonl"
    run_command(
        "convert public teacher next-action SFT data",
        [
            sys.executable,
            "-B",
            "course/03_sft_warmup/make_teacher_next_action_sft_jsonl.py",
            "--dataset-id",
            args.teacher_sft_dataset,
            "--split",
            args.teacher_sft_split,
            "--tools-data-dir",
            path_arg(args.data_dir),
            "--output",
            path_arg(teacher_file),
            "--limit",
            str(args.teacher_sft_limit),
            "--min-total-score",
            str(args.teacher_sft_min_total_score),
            "--min-avg-score",
            str(args.teacher_sft_min_avg_score),
            "--holdout-modulo",
            str(args.bridge_holdout_mod),
            "--sft-remainder",
            str(args.bridge_sft_remainder),
        ],
        env=env,
        dry_run=args.dry_run,
    )
    return teacher_file


def build_areal_sft(args: argparse.Namespace, env: dict[str, str]) -> Path:
    areal_file = args.data_dir / "sft_areal_retail_next_action.jsonl"
    command = [
        sys.executable,
        "-B",
        "course/03_sft_warmup/make_areal_retail_sft_jsonl.py",
        "--dataset-id",
        args.areal_sft_dataset,
        "--split",
        args.areal_sft_split,
        "--tools-data-dir",
        path_arg(args.data_dir),
        "--output",
        path_arg(areal_file),
        "--limit",
        str(args.areal_sft_limit),
        "--min-reward",
        str(args.areal_sft_min_reward),
    ]
    command += maybe_append("--max-source-rows", args.areal_sft_max_source_rows)
    if args.areal_sft_allow_uncertain_correct:
        command.append("--allow-uncertain-correct")
    run_command(
        "convert AReaL tau2 retail next-action SFT data",
        command,
        env=env,
        dry_run=args.dry_run,
    )
    return areal_file


def build_success_trace_sft(args: argparse.Namespace, env: dict[str, str]) -> Path:
    success_trace_file = args.data_dir / "sft_success_trace_retail_next_action.jsonl"
    full_trace_file = args.data_dir / "sft_success_trace_retail_full.jsonl"
    command = [
        sys.executable,
        "-B",
        "course/03_sft_warmup/make_success_trace_retail_sft_jsonl.py",
        "--dataset-id",
        args.success_trace_sft_dataset,
        "--split",
        args.success_trace_sft_split,
        "--tools-data-dir",
        path_arg(args.data_dir),
        "--output",
        path_arg(success_trace_file),
        "--full-output",
        path_arg(full_trace_file),
        "--limit",
        str(args.success_trace_sft_limit),
        "--min-reward",
        str(args.success_trace_sft_min_reward),
    ]
    command += maybe_append("--max-source-rows", args.success_trace_sft_max_source_rows)
    if not args.success_trace_sft_require_blind_strict:
        command.append("--no-require-blind-strict")
    if args.success_trace_sft_allow_memory_injected:
        command.append("--allow-memory-injected")
    run_command(
        "convert successful tau2 retail trace SFT data",
        command,
        env=env,
        dry_run=args.dry_run,
    )
    return success_trace_file


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
    sft_env = {
        **stage_env(env, stage="sft-train", kind="training", algo="sft"),
        "ART_MODEL_NAME": anchor_model,
    }
    default_sft_file = augmented_sft_file(args)
    sft_file = args.sft_file or default_sft_file
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
    sft_eval_env = {
        **stage_env(env, stage="sft-eval", kind="eval", algo="sft", split=args.eval_split),
        "ART_MODEL_NAME": anchor_model,
    }
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
        env=sft_eval_env,
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
    branch_train_env = {
        **stage_env(env, stage=f"{algo}-train", kind="training", algo=algo),
        "ART_MODEL_NAME": branch_model,
    }
    parent_artifact = artifact_uri(f"{anchor_model}-checkpoint", "sft-anchor")
    metrics_path = report_dir / f"train_metrics_{algo}_{args.rl_suffix}.jsonl"
    train_scripts = {
        "grpo": "course/04_grpo_local/train_grpo_local.py",
        "gspo": "course/06_gspo_and_configs/train_gspo_sequence.py",
        "ruler": "course/05_ruler/train_with_ruler.py",
    }
    train_script = train_scripts[algo]
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
        env=stage_env(
            {**env, "ART_MODEL_NAME": branch_model},
            stage=f"{algo}-checkpoint-fork",
            kind="artifact",
            algo=algo,
        ),
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
        "--max-sampling-rounds",
        str(args.rl_max_sampling_rounds),
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
        "--metrics-jsonl",
        path_arg(metrics_path),
    ]
    if algo == "ruler":
        command += [
            "--judge-model",
            args.ruler_judge_model,
            "--judge-effort",
            args.ruler_judge_effort,
            "--ruler-weight",
            str(args.ruler_weight),
            "--independent-weight",
            str(args.independent_weight),
        ]
    command += maybe_append("--max-turns", args.rl_max_turns)
    if args.continue_on_invalid:
        command.append("--continue-on-invalid")
    if args.no_logprobs:
        command.append("--no-logprobs")
    command += weave_args(args)
    run_command(f"{algo.upper()} training", command, env=branch_train_env, dry_run=args.dry_run)
    if args.select_rl_candidates:
        candidate_json = report_dir / f"{algo}_{args.rl_suffix}_checkpoint_candidates.json"
        run_command(
            f"select {algo} checkpoint candidates",
            [
                sys.executable,
                "-B",
                "course/02_weave_evals/select_checkpoint_candidate.py",
                path_arg(metrics_path),
                "--metric",
                args.rl_candidate_metric,
                "--top-k",
                str(args.rl_candidate_top_k),
                "--output-md",
                path_arg(report_dir / f"{algo}_{args.rl_suffix}_checkpoint_candidates.md"),
                "--output-json",
                path_arg(candidate_json),
            ],
            env=stage_env(
                {**env, "ART_MODEL_NAME": branch_model},
                stage=f"{algo}-candidate-selection",
                kind="comparison",
                algo=algo,
            ),
            dry_run=args.dry_run,
        )
        candidate_steps = [args.rl_steps] if args.dry_run else read_candidate_steps(candidate_json)
        if args.eval_rl_candidates and candidate_steps:
            for top_step in candidate_steps:
                candidate_model = candidate_model_name(branch_model, top_step)
                candidate_artifact = f"{candidate_model}-checkpoint"
                step_label = f"{top_step:04d}"
                candidate_env = {
                    **stage_env(env, stage=f"{algo}-candidate-step{step_label}", kind="artifact", algo=algo),
                    "ART_MODEL_NAME": candidate_model,
                }
                run_command(
                    f"fork {algo} candidate checkpoint step {step_label}",
                    [
                        sys.executable,
                        "-B",
                        "course/08_enterprise_ops/fork_checkpoint.py",
                        "--from-model",
                        branch_model,
                        "--to-model",
                        candidate_model,
                        "--not-after-step",
                        str(top_step),
                        "--file-only",
                        "--overwrite",
                        "--verbose",
                    ],
                    env=candidate_env,
                    dry_run=args.dry_run,
                )
                run_command(
                    f"log {algo} candidate checkpoint artifact step {step_label}",
                    [
                        sys.executable,
                        "-B",
                        "course/07_models_registry_weave/log_checkpoint_artifact.py",
                        "--model-name",
                        candidate_model,
                        "--stage",
                        f"{algo}-candidate-step{step_label}",
                        "--artifact-name",
                        candidate_artifact,
                        "--alias",
                        "validation-candidate",
                        "--metadata",
                        f"algorithm={json.dumps(algo)}",
                        "--metadata",
                        f"source_model={json.dumps(branch_model)}",
                        "--metadata",
                        f"source_step={top_step}",
                        "--metadata",
                        f"selection_metric={json.dumps(args.rl_candidate_metric)}",
                        "--metadata",
                        f"data_artifact={json.dumps(artifact_uri(args.data_artifact_name))}",
                    ],
                    env=candidate_env,
                    dry_run=args.dry_run,
                )
                candidate_validation_env = {
                    **stage_env(
                        env,
                        stage=f"{algo}-candidate-validation-eval",
                        kind="eval",
                        algo=algo,
                        split=args.eval_split,
                    ),
                    "ART_MODEL_NAME": candidate_model,
                }
                run_command(
                    f"eval {algo} candidate step {step_label} validation",
                    evaluate_command(
                        args,
                        data_dir=args.data_dir,
                        split=args.eval_split,
                        limit=args.eval_limit,
                        output=report_dir / f"eval_{algo}_{args.rl_suffix}_candidate_step{step_label}_validation.jsonl",
                        model_artifact=artifact_uri(candidate_artifact, "validation-candidate"),
                        include_messages=True,
                    ),
                    env=candidate_validation_env,
                    dry_run=args.dry_run,
                )
    if not args.skip_rl_train_eval:
        branch_train_eval_env = {
            **stage_env(env, stage=f"{algo}-train-subset-eval", kind="eval", algo=algo, split=args.rl_split),
            "ART_MODEL_NAME": branch_model,
        }
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
            env=branch_train_eval_env,
            dry_run=args.dry_run,
        )
    branch_validation_eval_env = {
        **stage_env(env, stage=f"{algo}-validation-eval", kind="eval", algo=algo, split=args.eval_split),
        "ART_MODEL_NAME": branch_model,
    }
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
        env=branch_validation_eval_env,
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
        if algo in {"grpo", "gspo", "ruler"}
    }
    expected_branch_models.update(branch_models)
    for algo, branch_model in expected_branch_models.items():
        eval_path = report_dir / f"eval_{algo}_{args.rl_suffix}_validation.jsonl"
        if eval_path.exists() or args.dry_run:
            paths.append(path_arg(eval_path))
            stages.append(f"{algo}_{args.rl_suffix}")
            model_artifacts.append(artifact_uri(f"{branch_model}-checkpoint"))
        candidate_paths = sorted(report_dir.glob(f"eval_{algo}_{args.rl_suffix}_candidate_step*_validation.jsonl"))
        if args.dry_run and args.eval_rl_candidates:
            candidate_paths = [report_dir / f"eval_{algo}_{args.rl_suffix}_candidate_step{args.rl_steps:04d}_validation.jsonl"]
        for candidate_path in candidate_paths:
            candidate_step = candidate_step_from_eval_path(candidate_path)
            if candidate_step is None:
                continue
            candidate_model = candidate_model_name(branch_model, candidate_step)
            paths.append(path_arg(candidate_path))
            stages.append(f"{algo}_{args.rl_suffix}_candidate_step{candidate_step:04d}")
            model_artifacts.append(artifact_uri(f"{candidate_model}-checkpoint", "validation-candidate"))
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
    ]
    if not args.no_wandb_compare:
        command.append("--wandb")
    run_command(
        "checkpoint comparison",
        command,
        env=stage_env(env, stage="checkpoint-comparison", kind="comparison"),
        dry_run=args.dry_run,
    )
    run_command(
        "stage acceptance gate",
        [
            sys.executable,
            "-B",
            "course/02_weave_evals/check_stage_acceptance.py",
            *paths,
            "--stages",
            *stages,
            "--model",
            args.base_model,
            "--output-md",
            path_arg(report_dir / "checkpoint_acceptance.md"),
            "--output-json",
            path_arg(report_dir / "checkpoint_acceptance.json"),
        ],
        env=stage_env(env, stage="checkpoint-acceptance", kind="comparison"),
        dry_run=args.dry_run,
    )
    if not args.no_weave and not args.skip_weave_cached_evals:
        for path, stage, model_artifact in zip(paths, stages, model_artifacts, strict=True):
            weave_command = [
                sys.executable,
                "-B",
                "course/02_weave_evals/evaluate_cached_checkpoint.py",
                path,
                "--stage",
                stage,
                "--model",
                args.base_model,
                "--data-artifact",
                artifact_uri(args.data_artifact_name),
                "--name",
                f"{args.run_slug}-{stage}-cached-checkpoint-eval",
            ]
            if model_artifact != "-":
                weave_command += ["--model-artifact", model_artifact]
            run_command(
                f"Weave cached eval {stage}",
                weave_command,
                env=stage_env(env, stage=f"cached-eval-{stage}", kind="eval"),
                dry_run=args.dry_run,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    profile_args = preparse_profile_args(argv)
    profile_defaults = load_run_profile_defaults(profile_args.run_profile, profile_args.base_config)
    profile_defaults.update(load_user_config_defaults(profile_args.config, profile_args.base_config))
    cfg = config_from_env()
    parser = argparse.ArgumentParser(
        description=(
            "Run the retail ART hands-on sequence: data artifact, baseline eval, "
            "next-action SFT, independent GRPO/GSPO branches, eval, and W&B comparison."
        )
    )
    parser.add_argument("--run-profile", default=profile_args.run_profile)
    parser.add_argument("--config", type=Path, default=profile_args.config)
    parser.add_argument("--base-config", type=Path, default=profile_args.base_config)
    parser.add_argument("--profile-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-slug", default="lfm25-8b-a1b-bridge-state1")
    parser.add_argument("--model-profile", default=cfg.model_profile)
    parser.add_argument("--base-model", default=cfg.base_model)
    parser.add_argument("--art-path", type=Path, default=Path(cfg.art_path) / "course_runs")
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
    parser.add_argument("--bridge-sft-remainder", type=int, default=2)
    parser.add_argument("--bridge-validation-remainder", type=int, default=0)
    parser.add_argument("--bridge-test-remainder", type=int, default=1)
    parser.add_argument("--bridge-sft-limit", type=int, default=-1)
    parser.add_argument("--include-teacher-sft", action="store_true")
    parser.add_argument(
        "--teacher-sft-dataset",
        default="amityco/tau-bench-retail-train-next-action-all-step-score-v0.2",
    )
    parser.add_argument("--teacher-sft-split", default="train")
    parser.add_argument("--teacher-sft-limit", type=int, default=512)
    parser.add_argument("--teacher-sft-min-total-score", type=float, default=1.0)
    parser.add_argument("--teacher-sft-min-avg-score", type=float, default=1.0)
    parser.add_argument("--include-areal-sft", action="store_true")
    parser.add_argument("--areal-sft-dataset", default="inclusionAI/AReaL-tau2-data")
    parser.add_argument("--areal-sft-split", default="sft")
    parser.add_argument("--areal-sft-limit", type=int, default=512)
    parser.add_argument("--areal-sft-min-reward", type=float, default=1.0)
    parser.add_argument("--areal-sft-max-source-rows", type=int, default=None)
    parser.add_argument(
        "--areal-sft-allow-uncertain-correct",
        action="store_true",
        help="Allow AReaL rows where metadata.correct is absent; the default keeps only correct rows.",
    )
    parser.add_argument("--include-success-trace-sft", action="store_true")
    parser.add_argument(
        "--success-trace-sft-dataset",
        default="KermitCO/qwen3.5-9B-tau2bench-retail-traces",
    )
    parser.add_argument("--success-trace-sft-split", default="train")
    parser.add_argument("--success-trace-sft-limit", type=int, default=200)
    parser.add_argument("--success-trace-sft-min-reward", type=float, default=1.0)
    parser.add_argument("--success-trace-sft-max-source-rows", type=int, default=None)
    parser.add_argument(
        "--success-trace-sft-require-blind-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only traces with the public C_blind_strict quality flag by default.",
    )
    parser.add_argument(
        "--success-trace-sft-allow-memory-injected",
        action="store_true",
        help="Allow traces generated with explicit memory/rule injection. Defaults to clean non-memory traces.",
    )
    parser.add_argument("--reward-profile", default="tau_irc")
    parser.add_argument("--retail-tool-use-instruction", default=None)
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
    parser.add_argument("--rl-max-sampling-rounds", type=int, default=1)
    parser.add_argument("--rl-learning-rate", type=float, default=2e-6)
    parser.add_argument("--rl-temperature", type=float, default=0.9)
    parser.add_argument("--rl-max-turns", type=int, default=None)
    parser.add_argument(
        "--select-rl-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write checkpoint candidate Markdown/JSON from per-step RL metrics before held-out eval.",
    )
    parser.add_argument("--rl-candidate-metric", default="data/step_course_score_mean")
    parser.add_argument("--rl-candidate-top-k", type=int, default=3)
    parser.add_argument(
        "--eval-rl-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After candidate selection, fork and evaluate the top selected RL checkpoint on validation.",
    )
    parser.add_argument("--ruler-judge-model", default="openai/gpt-5.5")
    parser.add_argument("--ruler-judge-effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--ruler-weight", type=float, default=0.3)
    parser.add_argument("--independent-weight", type=float, default=0.7)
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
    parser.add_argument(
        "--skip-rl-train-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip post-training eval on RL training scenarios; validation remains enabled.",
    )
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument(
        "--skip-weave-cached-evals",
        action="store_true",
        help="Skip publishing cached JSONL checkpoint results as Weave Evaluations after comparison.",
    )
    parser.add_argument("--no-weave", action="store_true")
    parser.add_argument("--no-wandb-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(**profile_defaults)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.run_profile:
        print(f"Using run profile: {args.run_profile}")
        print(f"User config: {args.config}")
        print(f"Base config: {args.base_config}")
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
        env=stage_env(env, stage="data-artifact", kind="data"),
        dry_run=args.dry_run,
    )
    if not args.skip_baseline_eval:
        baseline_env = {
            **stage_env(env, stage="baseline-eval", kind="eval", split=args.eval_split),
            "ART_MODEL_NAME": retail_model_name(args.run_slug, "baseline"),
        }
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
            if algo not in {"grpo", "gspo", "ruler"}:
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
