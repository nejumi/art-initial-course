from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .config import RetailCourseConfig


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


def ensure_wandb_run(
    config: RetailCourseConfig,
    *,
    job_type: str,
    run_name: str | None = None,
) -> tuple[Any | None, bool]:
    if wandb_disabled():
        return None, False
    try:
        import wandb
    except ImportError:
        return None, False
    if wandb.run is not None:
        return wandb.run, False
    run = wandb.init(
        project=config.project,
        entity=config.entity,
        job_type=job_type,
        name=run_name or config.model_name,
        id=os.getenv("WANDB_RUN_ID"),
        resume="allow",
    )
    return run, True


def finish_wandb_run(run: Any | None, owned: bool) -> None:
    if run is not None and owned:
        run.finish()


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


def log_checkpoint_artifact(
    config: RetailCourseConfig,
    *,
    stage: str,
    artifact_name: str | None = None,
    aliases: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
    job_type: str = "model-checkpoint",
    model_name: str | None = None,
    wait: bool = True,
) -> str | None:
    try:
        import wandb
    except ImportError:
        print("W&B unavailable; skipping checkpoint artifact logging.")
        return None
    run, _ = ensure_wandb_run(config, job_type=job_type)
    if run is None:
        print("W&B disabled; skipping checkpoint artifact logging.")
        return None

    checkpoint = latest_checkpoint_dir(config, model_name)
    step = int(checkpoint.name)
    safe_stage = artifact_safe_name(stage)
    safe_name = artifact_safe_name(artifact_name or f"{model_name or config.model_name}-checkpoint")
    alias_list = list(dict.fromkeys([safe_stage, f"step-{step:04d}", *aliases, "latest"]))
    base_metadata: dict[str, Any] = {
        "task": "retail-support-agent",
        "stage": stage,
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
        description=f"ART LoRA checkpoint for {config.model_name} at stage {stage}.",
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
        run_name=f"{artifact_safe_name(artifact_name)}-data",
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
