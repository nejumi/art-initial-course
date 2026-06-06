from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from typing import Any

from course.shared.config import config_from_env
from course.shared.wandb_artifacts import artifact_with_alias


def wandb_project_path(entity: str | None, project: str) -> str:
    return f"{entity}/{project}" if entity else project


def qualify_artifact_uri(uri: str, *, entity: str | None, project: str) -> str:
    resolved = artifact_with_alias(uri)
    artifact_part = resolved.split(":", 1)[0]
    if "/" in artifact_part:
        return resolved
    return f"{wandb_project_path(entity, project)}/{resolved}"


def short_run(run: Any) -> dict[str, Any]:
    return {
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "job_type": getattr(run, "job_type", None),
        "url": getattr(run, "url", None),
    }


def artifact_summary(api: Any, uri: str, *, entity: str | None, project: str) -> dict[str, Any]:
    qualified = qualify_artifact_uri(uri, entity=entity, project=project)
    artifact = api.artifact(qualified)
    logged_by = None
    used_by: list[dict[str, Any]] = []
    try:
        logged_by_run = artifact.logged_by()
        logged_by = short_run(logged_by_run) if logged_by_run is not None else None
    except Exception as exc:
        logged_by = {"error": str(exc)}
    try:
        used_by = [short_run(run) for run in artifact.used_by()]
    except Exception as exc:
        used_by = [{"error": str(exc)}]
    return {
        "requested_uri": uri,
        "qualified_uri": qualified,
        "name": getattr(artifact, "name", None),
        "type": getattr(artifact, "type", None),
        "version": getattr(artifact, "version", None),
        "aliases": list(getattr(artifact, "aliases", []) or []),
        "state": getattr(artifact, "state", None),
        "created_at": str(getattr(artifact, "created_at", "") or ""),
        "updated_at": str(getattr(artifact, "updated_at", "") or ""),
        "size": getattr(artifact, "size", None),
        "metadata": getattr(artifact, "metadata", {}) or {},
        "logged_by": logged_by,
        "used_by": used_by,
    }


def run_summary(api: Any, run_path_or_name: str, *, entity: str | None, project: str) -> dict[str, Any]:
    if "/" in run_path_or_name:
        run_path = run_path_or_name
    else:
        run_path = f"{wandb_project_path(entity, project)}/{run_path_or_name}"
    run = api.run(run_path)
    logged_artifacts = []
    used_artifacts = []
    try:
        logged_artifacts = [
            {
                "name": artifact.name,
                "type": artifact.type,
                "aliases": list(getattr(artifact, "aliases", []) or []),
                "version": getattr(artifact, "version", None),
            }
            for artifact in run.logged_artifacts()
        ]
    except Exception as exc:
        logged_artifacts = [{"error": str(exc)}]
    try:
        used_artifacts = [
            {
                "name": artifact.name,
                "type": artifact.type,
                "aliases": list(getattr(artifact, "aliases", []) or []),
                "version": getattr(artifact, "version", None),
            }
            for artifact in run.used_artifacts()
        ]
    except Exception as exc:
        used_artifacts = [{"error": str(exc)}]
    summary = short_run(run)
    summary.update(
        {
            "path": run_path,
            "created_at": str(getattr(run, "created_at", "") or ""),
            "updated_at": str(getattr(run, "updated_at", "") or ""),
            "logged_artifacts": logged_artifacts,
            "used_artifacts": used_artifacts,
        }
    )
    return summary


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# W&B Lineage Inspection", ""]
    artifacts = report.get("artifacts") or []
    if artifacts:
        lines.extend(
            [
                "## Artifacts",
                "",
                "| requested | qualified | type | aliases | logged_by | used_by_count |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for item in artifacts:
            logged = item.get("logged_by") or {}
            used_by = item.get("used_by") or []
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("requested_uri") or ""),
                        str(item.get("qualified_uri") or ""),
                        str(item.get("type") or ""),
                        ", ".join(str(alias) for alias in item.get("aliases") or []),
                        str(logged.get("name") or logged.get("id") or ""),
                        str(len(used_by)),
                    ]
                )
                + " |"
            )
        lines.append("")
    runs = report.get("runs") or []
    if runs:
        lines.extend(
            [
                "## Runs",
                "",
                "| path | name | state | job_type | logged_artifacts | used_artifacts |",
                "| --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        for item in runs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("path") or ""),
                        str(item.get("name") or ""),
                        str(item.get("state") or ""),
                        str(item.get("job_type") or ""),
                        str(len(item.get("logged_artifacts") or [])),
                        str(len(item.get("used_artifacts") or [])),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect W&B run/artifact lineage for course validation.")
    parser.add_argument("--artifact", action="append", default=[], help="Artifact URI or name; may be repeated.")
    parser.add_argument("--run", action="append", default=[], help="Run path/id/name; may be repeated.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    cfg = config_from_env()
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required to inspect lineage.") from exc
    api = wandb.Api()
    report = {
        "entity": cfg.entity,
        "project": cfg.project,
        "artifacts": [
            artifact_summary(api, uri, entity=cfg.entity, project=cfg.project)
            for uri in args.artifact
        ],
        "runs": [
            run_summary(api, run, entity=cfg.entity, project=cfg.project)
            for run in args.run
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown_report(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
