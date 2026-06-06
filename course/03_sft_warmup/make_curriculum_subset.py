from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from course.shared.data import load_cached_split, write_jsonl, write_sft_jsonl


def tool_call_count(record: dict[str, Any]) -> int:
    stats = ((record.get("metadata") or {}).get("stats") or {})
    value = stats.get("n_tool_calls")
    if isinstance(value, int):
        return value
    count = 0
    for message in record.get("messages") or []:
        if message.get("role") == "assistant":
            count += len(message.get("tool_calls") or [])
    return count


def task_key(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("task_id") or metadata.get("record_id") or record.get("id"))


def holdout_bucket(key: str, modulo: int) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def annotate(record: dict[str, Any], *, split: str, max_tool_calls: int, holdout_modulo: int) -> dict[str, Any]:
    row = dict(record)
    metadata = dict(row.get("metadata") or {})
    metadata["curriculum"] = {
        "name": f"retail-tool-calls-le-{max_tool_calls}",
        "source_split": record.get("split"),
        "source_id": record.get("id"),
        "task_key": task_key(record),
        "max_tool_calls": max_tool_calls,
        "holdout_modulo": holdout_modulo,
    }
    row["metadata"] = metadata
    row["split"] = split
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic easy retail curriculum split.")
    parser.add_argument("--source-dir", default=".art/full_data/retail")
    parser.add_argument("--output-dir", default=".art/curriculum_data/retail")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--max-tool-calls", type=int, default=5)
    parser.add_argument("--holdout-modulo", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    args = parser.parse_args()

    if args.holdout_modulo <= 1:
        raise ValueError("--holdout-modulo must be greater than 1")
    if not 0 <= args.holdout_remainder < args.holdout_modulo:
        raise ValueError("--holdout-remainder must be in [0, holdout_modulo)")

    records = load_cached_split(args.source_dir, args.source_split)
    filtered = [record for record in records if tool_call_count(record) <= args.max_tool_calls]

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for record in filtered:
        bucket = holdout_bucket(task_key(record), args.holdout_modulo)
        if bucket == args.holdout_remainder:
            validation_rows.append(
                annotate(
                    record,
                    split="validation",
                    max_tool_calls=args.max_tool_calls,
                    holdout_modulo=args.holdout_modulo,
                )
            )
        else:
            train_rows.append(
                annotate(record, split="train", max_tool_calls=args.max_tool_calls, holdout_modulo=args.holdout_modulo)
            )

    if args.limit_train is not None:
        train_rows = train_rows[: args.limit_train]
    if args.limit_validation is not None:
        validation_rows = validation_rows[: args.limit_validation]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    write_jsonl(output_dir / "test.jsonl", validation_rows)
    write_sft_jsonl(train_rows, output_dir / "sft_train.jsonl")
    write_sft_jsonl(validation_rows, output_dir / "sft_validation.jsonl")
    write_sft_jsonl(validation_rows, output_dir / "sft_test.jsonl")

    print(
        {
            "source_dir": str(Path(args.source_dir)),
            "output_dir": str(output_dir),
            "source_split": args.source_split,
            "max_tool_calls": args.max_tool_calls,
            "holdout_modulo": args.holdout_modulo,
            "holdout_remainder": args.holdout_remainder,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
        }
    )


if __name__ == "__main__":
    main()
