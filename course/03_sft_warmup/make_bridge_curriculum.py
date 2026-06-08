from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from course.shared.data import load_cached_split, write_jsonl, write_sft_jsonl
from course.shared.retail_env import STATE_CHANGING_TOOL_NAMES, extract_tool_calls_from_messages, tool_call_name


def metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata") or {}
    return value if isinstance(value, dict) else {}


def task_key(record: dict[str, Any]) -> str:
    meta = metadata(record)
    for key in ("task_id", "record_id"):
        value = meta.get(key)
        if value is not None:
            return str(value)
    return str(record.get("id"))


def holdout_bucket(key: str, modulo: int) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def source_reward(record: dict[str, Any]) -> float:
    meta = metadata(record)
    for key in ("reward", "canonical_reward", "success"):
        value = meta.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def partial_score(record: dict[str, Any]) -> float:
    meta = metadata(record)
    for key in ("partial_score", "avg_partial", "score"):
        value = meta.get(key)
        if isinstance(value, int | float):
            return float(value)
    return source_reward(record)


def tool_call_count(record: dict[str, Any]) -> int:
    stats = metadata(record).get("stats") or {}
    value = stats.get("n_tool_calls") if isinstance(stats, dict) else None
    if isinstance(value, int):
        return value
    return len(extract_tool_calls_from_messages(record.get("messages") or []))


def turn_count(record: dict[str, Any]) -> int:
    stats = metadata(record).get("stats") or {}
    value = stats.get("n_turns") if isinstance(stats, dict) else None
    if isinstance(value, int):
        return value
    return len(record.get("messages") or [])


def state_action_names(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for call in extract_tool_calls_from_messages(record.get("messages") or []):
        name = tool_call_name(call)
        if name in STATE_CHANGING_TOOL_NAMES:
            names.append(name)
    return names


def keep_record(
    record: dict[str, Any],
    *,
    min_state_actions: int,
    max_state_actions: int | None,
    max_tool_calls: int | None,
    max_turns: int | None,
) -> bool:
    state_count = len(state_action_names(record))
    if state_count < min_state_actions:
        return False
    if max_state_actions is not None and state_count > max_state_actions:
        return False
    if max_tool_calls is not None and tool_call_count(record) > max_tool_calls:
        return False
    if max_turns is not None and turn_count(record) > max_turns:
        return False
    return True


def selection_key(record: dict[str, Any]) -> tuple[float, float, int, int, str]:
    # Higher quality first; shorter, simpler trajectories break ties.
    return (
        source_reward(record),
        partial_score(record),
        -len(state_action_names(record)),
        -tool_call_count(record),
        task_key(record),
    )


def annotate(record: dict[str, Any], *, split: str, name: str, args: argparse.Namespace) -> dict[str, Any]:
    row = dict(record)
    meta = dict(metadata(record))
    state_names = state_action_names(record)
    meta["curriculum"] = {
        "name": name,
        "source_split": record.get("split"),
        "source_id": record.get("id"),
        "task_key": task_key(record),
        "source_reward": source_reward(record),
        "partial_score": partial_score(record),
        "tool_call_count": tool_call_count(record),
        "turn_count": turn_count(record),
        "state_action_count": len(state_names),
        "state_action_names": state_names,
        "min_state_actions": args.min_state_actions,
        "max_state_actions": args.max_state_actions,
        "max_tool_calls": args.max_tool_calls,
        "max_turns": args.max_turns,
        "holdout_modulo": args.holdout_modulo,
    }
    row["metadata"] = meta
    row["split"] = split
    return row


def cap_per_task(rows: list[dict[str, Any]], max_per_task: int | None) -> list[dict[str, Any]]:
    if max_per_task is None or max_per_task <= 0:
        return rows
    counts: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = task_key(row)
        if counts.get(key, 0) >= max_per_task:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(row)
    return kept


def select_rows(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filtered = [
        record
        for record in records
        if keep_record(
            record,
            min_state_actions=args.min_state_actions,
            max_state_actions=args.max_state_actions,
            max_tool_calls=args.max_tool_calls,
            max_turns=args.max_turns,
        )
    ]
    success_rows = [record for record in filtered if source_reward(record) >= args.min_source_reward]
    fallback_rows = [
        record
        for record in filtered
        if source_reward(record) < args.min_source_reward and partial_score(record) >= args.fallback_partial_threshold
    ]
    ordered = sorted(success_rows, key=selection_key, reverse=True)
    ordered.extend(sorted(fallback_rows, key=selection_key, reverse=True))
    return cap_per_task(ordered, args.max_per_task)


def split_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    buckets = {"sft": [], "train": [], "validation": [], "test": []}
    for row in rows:
        bucket = holdout_bucket(task_key(row), args.holdout_modulo)
        if bucket == args.validation_remainder:
            target = "validation"
        elif bucket == args.test_remainder:
            target = "test"
        elif bucket == args.sft_remainder:
            target = "sft"
        else:
            target = "train"
        buckets[target].append(row)
    limits = {
        "sft": args.limit_sft,
        "train": args.limit_train,
        "validation": args.limit_validation,
        "test": args.limit_test,
    }
    return {split: items[: limits[split]] if limits[split] is not None else items for split, items in buckets.items()}


def summary(rows_by_split: dict[str, list[dict[str, Any]]], *, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_dir": args.source_dir,
        "output_dir": str(output_dir),
        "source_split": args.source_split,
        "selection": {
            "min_source_reward": args.min_source_reward,
            "fallback_partial_threshold": args.fallback_partial_threshold,
            "min_state_actions": args.min_state_actions,
            "max_state_actions": args.max_state_actions,
            "max_tool_calls": args.max_tool_calls,
            "max_turns": args.max_turns,
            "max_per_task": args.max_per_task,
        },
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        state_counts = [len(state_action_names(row)) for row in rows]
        result["splits"][split] = {
            "rows": len(rows),
            "tasks": len({task_key(row) for row in rows}),
            "avg_tool_calls": round(sum(tool_call_count(row) for row in rows) / len(rows), 4) if rows else 0.0,
            "avg_turns": round(sum(turn_count(row) for row in rows) / len(rows), 4) if rows else 0.0,
            "avg_state_actions": round(sum(state_counts) / len(state_counts), 4) if state_counts else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a bridge curriculum for agentic SFT/RL from open retail trajectories."
    )
    parser.add_argument("--source-dir", default=".art/full_data/retail")
    parser.add_argument("--output-dir", default=".art/curriculum_data/retail_bridge")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--name", default="retail-bridge-state-action")
    parser.add_argument("--min-source-reward", type=float, default=1.0)
    parser.add_argument("--fallback-partial-threshold", type=float, default=0.55)
    parser.add_argument("--min-state-actions", type=int, default=1)
    parser.add_argument("--max-state-actions", type=int, default=1)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=28)
    parser.add_argument("--max-per-task", type=int, default=2)
    parser.add_argument("--holdout-modulo", type=int, default=5)
    parser.add_argument("--validation-remainder", type=int, default=0)
    parser.add_argument("--test-remainder", type=int, default=1)
    parser.add_argument("--sft-remainder", type=int, default=2)
    parser.add_argument("--limit-sft", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    if args.holdout_modulo <= 3:
        raise ValueError("--holdout-modulo must be greater than 3")
    remainders = {
        "validation": args.validation_remainder,
        "test": args.test_remainder,
        "sft": args.sft_remainder,
    }
    for name, remainder in remainders.items():
        if remainder < 0 or remainder >= args.holdout_modulo:
            raise ValueError(f"--{name}-remainder must be in [0, holdout_modulo)")
    if len(set(remainders.values())) != len(remainders):
        raise ValueError("--validation-remainder, --test-remainder, and --sft-remainder must all differ")

    records = load_cached_split(args.source_dir, args.source_split)
    selected = select_rows(records, args)
    rows_by_split = split_rows(selected, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "sft_train.jsonl",
        "sft_train_next_action.jsonl",
        "sft_validation.jsonl",
        "sft_validation_next_action.jsonl",
        "sft_test.jsonl",
        "sft_test_next_action.jsonl",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    annotated_by_split = {
        split: [annotate(row, split=split, name=args.name, args=args) for row in rows]
        for split, rows in rows_by_split.items()
    }

    for split, rows in annotated_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        if split == "sft":
            write_sft_jsonl(rows, output_dir / "sft_full.jsonl")

    next_action_module = importlib.import_module("course.03_sft_warmup.make_next_action_sft_jsonl")
    record_to_next_action_examples = next_action_module.record_to_next_action_examples

    for split, rows in annotated_by_split.items():
        if split != "sft":
            continue
        next_action_examples: list[dict[str, Any]] = []
        for row in rows:
            next_action_examples.extend(record_to_next_action_examples(row))
        write_jsonl(output_dir / "sft_next_action.jsonl", next_action_examples)

    stats = summary(annotated_by_split, output_dir=output_dir, args=args)
    (output_dir / "bridge_curriculum_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
