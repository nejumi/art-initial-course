from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import random
from typing import Any

from course.shared.data import read_jsonl, write_jsonl
from course.shared.wandb_artifacts import sha256_file


def source_label(path: Path, row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return str(
        metadata.get("source_dataset")
        or metadata.get("source_repo")
        or metadata.get("source")
        or path.stem
    )


def limit_rows(rows: list[dict[str, Any]], limit: int | None, *, rng: random.Random) -> list[dict[str, Any]]:
    if limit is None or limit < 0 or len(rows) <= limit:
        return rows
    selected = list(rows)
    rng.shuffle(selected)
    return selected[:limit]


def format_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        key = str(metadata.get("sft_format") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix multiple SFT JSONL sources into one training file.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--limits",
        nargs="*",
        type=int,
        default=None,
        help="Optional per-input row caps. Use -1 for no cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    input_paths = [Path(value) for value in args.inputs]
    limits = args.limits or []
    if limits and len(limits) != len(input_paths):
        raise SystemExit("--limits must have the same length as --inputs when provided.")

    rng = random.Random(args.seed)
    mixed: list[dict[str, Any]] = []
    inputs_summary: list[dict[str, Any]] = []
    for index, path in enumerate(input_paths):
        rows = read_jsonl(path)
        limit = limits[index] if limits else None
        selected = limit_rows(rows, None if limit is None or limit < 0 else limit, rng=rng)
        for row in selected:
            metadata = row.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata.setdefault("mixed_sft_source_file", str(path))
        mixed.extend(selected)
        inputs_summary.append(
            {
                "path": str(path),
                "rows": len(rows),
                "selected_rows": len(selected),
                "sha256": sha256_file(path) if path.exists() else None,
                "source_counts": {
                    label: sum(1 for row in selected if source_label(path, row) == label)
                    for label in sorted({source_label(path, row) for row in selected})
                },
                "sft_format_counts": format_counts(selected),
            }
        )

    if not args.no_shuffle:
        rng.shuffle(mixed)
    count = write_jsonl(args.output, mixed)
    summary = {
        "output": args.output,
        "rows": count,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "inputs": inputs_summary,
        "sft_format_counts": format_counts(mixed),
    }
    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
