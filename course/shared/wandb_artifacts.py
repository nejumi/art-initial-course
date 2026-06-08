from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .config import RetailCourseConfig
from .tracing import bind_weave_to_active_wandb_run, clear_weave_wandb_run_context


def artifact_safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return safe[:128] or "artifact"


def artifact_with_alias(uri: str) -> str:
    last = uri.rsplit("/", 1)[-1]
    return uri if ":" in last else f"{uri}:latest"


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def jsonl_metadata_counts(path: Path) -> dict[str, Any]:
    sft_format_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            if "sft_format" in metadata:
                value = str(metadata.get("sft_format") or "unknown")
                sft_format_counts[value] = sft_format_counts.get(value, 0) + 1
            source = metadata.get("source_dataset") or metadata.get("source_repo") or metadata.get("source")
            if source:
                source_key = str(source)
                source_counts[source_key] = source_counts.get(source_key, 0) + 1
    return {
        "rows": rows,
        "sft_format_counts": sft_format_counts,
        "source_counts": source_counts,
    }


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else {"value": value}


def file_manifest(paths: Iterable[Path], *, root: Path | None = None) -> list[dict[str, Any]]:
    manifest = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        name = str(path.relative_to(root)) if root is not None else path.name
        manifest.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return manifest


def wandb_disabled() -> bool:
    return os.getenv("WANDB_MODE", "").strip().lower() == "disabled"


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _safe_tag_value(value: str) -> str:
    return artifact_safe_name(value).lower()


def _wandb_tag(label: str, value: str, *, max_length: int = 64) -> str:
    safe_label = _safe_tag_value(label)[:24] or "tag"
    safe_value = _safe_tag_value(value)
    prefix = f"{safe_label}:"
    full = f"{prefix}{safe_value}"
    if len(full) <= max_length:
        return full
    digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:8]
    room = max(1, max_length - len(prefix) - len(digest) - 1)
    shortened = safe_value[:room].rstrip(".-") or safe_value[:room]
    return f"{prefix}{shortened}-{digest}"


def _normalize_wandb_tag(tag: str) -> str:
    if ":" in tag:
        label, value = tag.split(":", 1)
    else:
        label, value = "tag", tag
    return _wandb_tag(label, value)


def _normalize_wandb_tags(tags: Iterable[str]) -> list[str]:
    normalized = [_normalize_wandb_tag(tag) for tag in tags if str(tag).strip()]
    return list(dict.fromkeys(normalized))


def _infer_stage(config: RetailCourseConfig, job_type: str) -> str:
    explicit = _env_value("COURSE_RUN_STAGE")
    if explicit:
        return explicit
    normalized_job = job_type.strip().lower()
    model_name = config.model_name.lower()
    if normalized_job in {"sft", "grpo", "gspo", "ruler-grpo"}:
        return f"{normalized_job}-train"
    if normalized_job == "eval":
        if "baseline" in model_name:
            return "baseline-eval"
        if "sft" in model_name:
            return "sft-eval"
        if "ruler" in model_name:
            return "ruler-grpo-eval"
        if "gspo" in model_name:
            return "gspo-eval"
        if "grpo" in model_name:
            return "grpo-eval"
        return "checkpoint-eval"
    if normalized_job == "data-artifact":
        return "data-artifact"
    if normalized_job == "eval-comparison":
        return "checkpoint-comparison"
    if normalized_job == "weave-cached-eval":
        return "cached-eval"
    if normalized_job == "registry-link":
        return "registry-link"
    if normalized_job == "upload-lora":
        return "upload-lora"
    if normalized_job == "smoke":
        return "setup-smoke"
    return normalized_job or "run"


def _infer_kind(job_type: str, stage: str) -> str:
    explicit = _env_value("COURSE_RUN_KIND")
    if explicit:
        return explicit
    normalized_job = job_type.strip().lower()
    normalized_stage = stage.strip().lower()
    if normalized_job in {"sft", "grpo", "gspo", "ruler-grpo"}:
        return "training"
    if normalized_job in {"eval", "weave-eval", "weave-cached-eval"} or normalized_stage.endswith("eval"):
        return "eval"
    if normalized_job == "eval-comparison":
        return "comparison"
    if normalized_job == "data-artifact":
        return "data"
    if normalized_job in {"registry-link"}:
        return "registry"
    if normalized_job in {"upload-lora", "model-checkpoint"}:
        return "artifact"
    if normalized_job == "smoke":
        return "setup"
    return normalized_job or "run"


def _infer_algorithm(job_type: str, stage: str) -> str | None:
    explicit = _env_value("COURSE_RUN_ALGO")
    if explicit:
        return explicit
    combined = f"{job_type} {stage}".lower()
    if "ruler" in combined:
        return "ruler-grpo"
    if "gspo" in combined:
        return "gspo"
    if "grpo" in combined:
        return "grpo"
    if "sft" in combined:
        return "sft"
    return None


def wandb_run_context(config: RetailCourseConfig, job_type: str) -> dict[str, str]:
    stage = _infer_stage(config, job_type)
    kind = _infer_kind(job_type, stage)
    context = {
        "stage": stage,
        "kind": kind,
        "job_type": job_type,
        "art_model_name": config.model_name,
        "base_model": config.base_model,
        "model_profile": config.model_profile,
        "dataset_id": config.dataset_id,
        "weave_project": os.getenv("WEAVE_PROJECT") or config.project,
    }
    optional_values = {
        "algorithm": _infer_algorithm(job_type, stage),
        "split": _env_value("COURSE_RUN_SPLIT"),
        "run_profile": _env_value("COURSE_RUN_PROFILE"),
        "run_slug": _env_value("COURSE_RUN_SLUG"),
        "reward_profile": _env_value("RETAIL_REWARD_PROFILE"),
        "slurm_job_id": _env_value("SLURM_JOB_ID"),
    }
    for key, value in optional_values.items():
        if value:
            context[key] = value
    return context


def default_wandb_tags(config: RetailCourseConfig, job_type: str) -> list[str]:
    context = wandb_run_context(config, job_type)
    tags = [
        _wandb_tag("stage", context["stage"]),
        _wandb_tag("kind", context["kind"]),
    ]
    if "algorithm" in context:
        tags.append(_wandb_tag("algo", context["algorithm"]))
    if "split" in context:
        tags.append(_wandb_tag("split", context["split"]))
    if "run_profile" in context:
        tags.append(_wandb_tag("profile", context["run_profile"]))
    if "reward_profile" in context and context["kind"] in {"training", "eval", "comparison"}:
        tags.append(_wandb_tag("reward", context["reward_profile"]))
    if "slurm_job_id" in context:
        tags.append(_wandb_tag("slurm", context["slurm_job_id"]))
    return list(dict.fromkeys(tags))


def default_wandb_notes(config: RetailCourseConfig, job_type: str) -> str:
    context = wandb_run_context(config, job_type)
    stage = context["stage"]
    purpose_by_stage = {
        "data-artifact": "Log the retail dataset and SFT files as versioned W&B Artifacts.",
        "baseline-eval": "Evaluate the base model before SFT or RL.",
        "sft-train": "Train the next-action SFT anchor checkpoint.",
        "sft-eval": "Evaluate the SFT anchor checkpoint.",
        "grpo-train": "Run GRPO from the SFT anchor checkpoint.",
        "gspo-train": "Run GSPO from the SFT anchor checkpoint.",
        "ruler-grpo-train": "Run GRPO with a hybrid RULER reward from the SFT anchor checkpoint.",
        "checkpoint-comparison": "Compare baseline, SFT, and RL checkpoint metrics in a W&B Table.",
        "cached-eval": "Publish cached rollout results as a Weave Evaluation and companion W&B Table.",
        "setup-smoke": "Verify that W&B metrics and Weave calls are linked to the same W&B Run.",
        "registry-link": "Link a model artifact to W&B Registry.",
        "upload-lora": "Upload a local LoRA checkpoint as a W&B model artifact.",
    }
    purpose = purpose_by_stage.get(stage)
    if purpose is None:
        if stage.startswith("cached-eval"):
            purpose = "Publish cached rollout results as a Weave Evaluation and companion W&B Table."
        elif stage.endswith("-eval"):
            purpose = f"Evaluate the {stage.removesuffix('-eval')} checkpoint."
        elif stage.endswith("-train"):
            purpose = f"Train the {stage.removesuffix('-train')} checkpoint."
        else:
            purpose = "Run one step of the retail support agent training pipeline."
    optional_lines = [
        ("Algorithm", "algorithm"),
        ("Split", "split"),
        ("Run profile", "run_profile"),
        ("Run slug", "run_slug"),
        ("Reward profile", "reward_profile"),
        ("Slurm job", "slurm_job_id"),
    ]
    context_lines = [f"{label}: {context[key]}" for label, key in optional_lines if key in context]
    return "\n".join(
        [
            purpose,
            f"Stage: {context['stage']}",
            f"Kind: {context['kind']}",
            f"Job type: {job_type}",
            *context_lines,
            f"ART model: {context['art_model_name']}",
            f"Base model: {context['base_model']}",
            f"Model profile: {context['model_profile']}",
            f"Dataset: {context['dataset_id']}",
            f"Weave project: {context['weave_project']}",
            "",
            "When Weave tracing is enabled, calls are bound to the W&B-generated run.id for this run.",
        ]
    )


def apply_wandb_run_metadata(
    run: Any,
    config: RetailCourseConfig,
    *,
    job_type: str,
    tags: Iterable[str] = (),
    notes: str | None = None,
) -> None:
    context = wandb_run_context(config, job_type)
    tag_list = _normalize_wandb_tags([*default_wandb_tags(config, job_type), *tags])
    try:
        run.tags = tuple(tag_list)
    except Exception:
        pass
    try:
        run.notes = notes or default_wandb_notes(config, job_type)
    except Exception:
        pass
    try:
        run.config.update(
            {
                "course": "openpipe-art-retail",
                "job_type": job_type,
                "stage": context["stage"],
                "kind": context["kind"],
                "art_model_name": config.model_name,
                "base_model": config.base_model,
                "model_profile": config.model_profile,
                "dataset_id": config.dataset_id,
                "weave_project": os.getenv("WEAVE_PROJECT") or config.project,
                "git_commit": current_git_commit(),
                **{key: context[key] for key in ("algorithm", "split", "run_profile", "run_slug", "reward_profile", "slurm_job_id") if key in context},
            },
            allow_val_change=True,
        )
    except Exception:
        pass


def ensure_wandb_run(
    config: RetailCourseConfig,
    *,
    job_type: str,
    tags: Iterable[str] = (),
    notes: str | None = None,
) -> tuple[Any | None, bool]:
    if wandb_disabled():
        return None, False
    try:
        import wandb
    except ImportError:
        return None, False
    if wandb.run is not None:
        apply_wandb_run_metadata(wandb.run, config, job_type=job_type, tags=tags, notes=notes)
        bind_weave_to_active_wandb_run()
        return wandb.run, False
    run = wandb.init(
        project=config.project,
        entity=config.entity,
        job_type=job_type,
        tags=_normalize_wandb_tags([*default_wandb_tags(config, job_type), *tags]),
        notes=notes or default_wandb_notes(config, job_type),
        config={
            "course": "openpipe-art-retail",
            "job_type": job_type,
            **wandb_run_context(config, job_type),
            "git_commit": current_git_commit(),
        },
    )
    bind_weave_to_active_wandb_run()
    return run, True


def finish_wandb_run(run: Any | None, owned: bool) -> None:
    if run is not None and owned:
        clear_weave_wandb_run_context(getattr(run, "id", None))
        run.finish()


def _wandb_metric_value(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if math.isfinite(numeric) else None


def log_wandb_metrics(
    config: RetailCourseConfig,
    metrics: dict[str, Any],
    *,
    job_type: str,
    step: int | None = None,
) -> None:
    run, _ = ensure_wandb_run(config, job_type=job_type)
    if run is None:
        return
    payload: dict[str, float | int] = {}
    for key, value in metrics.items():
        numeric = _wandb_metric_value(value)
        if numeric is not None:
            payload[key] = numeric
    if step is not None:
        payload["training_step"] = int(step)
    if payload:
        run.log(payload)


def use_wandb_artifact(
    config: RetailCourseConfig,
    artifact_uri: str | None,
    *,
    artifact_type: str,
    job_type: str,
    use_as: str | None = None,
) -> Any | None:
    if not artifact_uri:
        return None
    run, _ = ensure_wandb_run(config, job_type=job_type)
    if run is None:
        print(f"W&B disabled or unavailable; skipping use_artifact({artifact_uri}).")
        return None
    resolved = artifact_with_alias(artifact_uri)
    artifact = run.use_artifact(resolved, type=artifact_type)
    print(f"Using W&B artifact: {resolved}")
    return artifact


def checkpoint_root(config: RetailCourseConfig, model_name: str | None = None) -> Path:
    return (
        Path(config.art_path)
        / config.project
        / "models"
        / (model_name or config.model_name)
        / "checkpoints"
    )


def latest_checkpoint_dir(config: RetailCourseConfig, model_name: str | None = None) -> Path:
    root = checkpoint_root(config, model_name)
    if not root.exists():
        raise FileNotFoundError(f"Missing ART checkpoint root: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No ART checkpoints found in: {root}")
    return sorted(candidates, key=lambda path: int(path.name))[-1]


def checkpoint_dir_for_step(
    config: RetailCourseConfig,
    *,
    checkpoint_step: int,
    model_name: str | None = None,
) -> Path:
    root = checkpoint_root(config, model_name)
    if not root.exists():
        raise FileNotFoundError(f"Missing ART checkpoint root: {root}")
    checkpoint = root / f"{checkpoint_step:04d}"
    if not checkpoint.exists() or not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing ART checkpoint step {checkpoint_step:04d}: {checkpoint}")
    return checkpoint


def select_checkpoint_dir(
    config: RetailCourseConfig,
    *,
    model_name: str | None = None,
    checkpoint_step: int | None = None,
    checkpoint_path: Path | str | None = None,
) -> Path:
    if checkpoint_step is not None and checkpoint_path is not None:
        raise ValueError("Set only one of checkpoint_step or checkpoint_path.")
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists() or not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing ART checkpoint path: {checkpoint}")
        if not checkpoint.name.isdigit():
            raise ValueError(f"Checkpoint path must end with a numeric ART step directory: {checkpoint}")
        return checkpoint
    if checkpoint_step is not None:
        return checkpoint_dir_for_step(config, checkpoint_step=checkpoint_step, model_name=model_name)
    return latest_checkpoint_dir(config, model_name)


def checkpoint_artifact_aliases(
    *,
    stage: str,
    step: int,
    aliases: Iterable[str] = (),
    historical: bool = False,
    include_latest_alias: bool | None = None,
) -> list[str]:
    safe_stage = artifact_safe_name(stage)
    include_latest = (not historical) if include_latest_alias is None else include_latest_alias
    if historical:
        base_aliases = [f"{safe_stage}-step-{step:04d}", f"step-{step:04d}", *aliases]
    else:
        base_aliases = [safe_stage, f"step-{step:04d}", *aliases]
    if include_latest:
        base_aliases.append("latest")
    return list(dict.fromkeys(base_aliases))


def log_checkpoint_artifact(
    config: RetailCourseConfig,
    *,
    stage: str,
    artifact_name: str | None = None,
    aliases: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
    job_type: str = "model-checkpoint",
    model_name: str | None = None,
    checkpoint_step: int | None = None,
    checkpoint_path: Path | str | None = None,
    include_latest_alias: bool | None = None,
    wait: bool = True,
) -> str | None:
    try:
        import wandb
    except ImportError:
        print("W&B unavailable; skipping checkpoint artifact logging.")
        return None
    if wandb_disabled():
        print("W&B disabled; skipping checkpoint artifact logging.")
        return None
    if wandb.run is not None:
        run = wandb.run
    else:
        run, _ = ensure_wandb_run(config, job_type=job_type)
    if run is None:
        print("W&B disabled; skipping checkpoint artifact logging.")
        return None

    checkpoint = select_checkpoint_dir(
        config,
        model_name=model_name,
        checkpoint_step=checkpoint_step,
        checkpoint_path=checkpoint_path,
    )
    step = int(checkpoint.name)
    historical = checkpoint_step is not None or checkpoint_path is not None
    safe_name = artifact_safe_name(artifact_name or f"{model_name or config.model_name}-checkpoint")
    alias_list = checkpoint_artifact_aliases(
        stage=stage,
        step=step,
        aliases=aliases,
        historical=historical,
        include_latest_alias=include_latest_alias,
    )
    base_metadata: dict[str, Any] = {
        "task": "retail-support-agent",
        "stage": stage,
        "historical_checkpoint": historical,
        "art_model_name": model_name or config.model_name,
        "art_project": config.project,
        "art_path": str(config.art_path),
        "art_checkpoint_step": step,
        "art_checkpoint_path": str(checkpoint),
        "base_model": config.base_model,
        "model_profile": config.model_profile,
        "dataset_id": config.dataset_id,
        "git_commit": current_git_commit(),
        "files": file_manifest(checkpoint.rglob("*"), root=checkpoint),
    }
    if metadata:
        base_metadata.update(metadata)

    artifact = wandb.Artifact(
        safe_name,
        type="model",
        description=f"ART LoRA checkpoint for {model_name or config.model_name} at stage {stage}.",
        metadata=base_metadata,
    )
    artifact.add_dir(str(checkpoint))
    logged = run.log_artifact(artifact, aliases=alias_list)
    if wait:
        logged.wait()
    uri = f"{safe_name}:{alias_list[0]}"
    print(f"Logged W&B checkpoint artifact: {uri}")
    return uri


def log_retail_data_artifact(
    config: RetailCourseConfig,
    *,
    data_dir: Path,
    artifact_name: str = "retail-course-data",
    aliases: Iterable[str] = ("latest", "tau-retail-v1"),
    wait: bool = True,
) -> str | None:
    try:
        import wandb
    except ImportError:
        print("W&B unavailable; skipping data artifact logging.")
        return None
    run, owned = ensure_wandb_run(
        config,
        job_type="data-artifact",
        tags=[f"artifact:{artifact_safe_name(artifact_name)}"],
    )
    if run is None:
        print("W&B disabled; skipping data artifact logging.")
        return None
    data_dir = Path(data_dir)
    files = [
        data_dir / "train.jsonl",
        data_dir / "validation.jsonl",
        data_dir / "test.jsonl",
    ]
    sft_files = sorted(data_dir.glob("sft*.jsonl"))
    files.extend(sft_files)
    metadata_files = sorted(data_dir.glob("*summary*.json"))
    files.extend(metadata_files)
    existing = [path for path in files if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No retail JSONL files found in: {data_dir}")

    safe_name = artifact_safe_name(artifact_name)
    source_metadata = read_json_file(data_dir / "source_metadata.json")
    metadata = {
        "task": "retail-support-agent",
        "source_dataset": config.dataset_id,
        "data_dir": str(data_dir),
        "source_metadata": source_metadata,
        "git_commit": current_git_commit(),
        "files": [
            {
                **entry,
                **jsonl_metadata_counts(data_dir / entry["path"]),
            }
            for entry in file_manifest(existing, root=data_dir)
        ],
    }
    artifact = wandb.Artifact(
        safe_name,
        type="dataset",
        description="Retail support course data: raw TAU Retail splits plus generated SFT JSONL.",
        metadata=metadata,
    )
    for path in existing:
        artifact.add_file(str(path), name=path.name)
    alias_list = list(dict.fromkeys(aliases))
    logged = run.log_artifact(artifact, aliases=alias_list)
    if wait:
        logged.wait()
    finish_wandb_run(run, owned)
    uri = f"{safe_name}:{alias_list[0] if alias_list else 'latest'}"
    print(f"Logged W&B data artifact: {uri}")
    return uri
