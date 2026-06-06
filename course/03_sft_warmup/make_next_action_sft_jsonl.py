from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from typing import Any

from course.shared.data import load_cached_split, normalize_messages_for_sft, write_jsonl
from course.shared.retail_env import merge_assistant_chunks


def assistant_spans(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        if messages[index].get("role") != "assistant":
            index += 1
            continue
        start = index
        while index < len(messages) and messages[index].get("role") == "assistant":
            index += 1
        end = index
        chunks = messages[start:end]
        if any(chunk.get("content") is not None or chunk.get("tool_calls") for chunk in chunks):
            spans.append((start, end))
    return spans


def record_to_next_action_examples(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = list(record.get("messages") or [])
    tools = list(record.get("tools") or [])
    metadata = dict(record.get("metadata") or {})
    examples: list[dict[str, Any]] = []
    for action_index, (start_index, end_index) in enumerate(assistant_spans(messages)):
        merged_assistant = merge_assistant_chunks(messages[start_index:end_index])
        prefix = [*messages[:start_index], merged_assistant]
        if not any(message.get("role") == "user" for message in prefix):
            continue
        example: dict[str, Any] = {
            "messages": normalize_messages_for_sft(prefix),
            "metadata": {
                **metadata,
                "source_record_id": metadata.get("record_id") or record.get("id"),
                "source_split": record.get("split"),
                "assistant_message_start_index": start_index,
                "assistant_message_end_index": end_index,
                "assistant_action_index": action_index,
                "sft_format": "tau-retail-next-action",
            },
        }
        if tools:
            example["tools"] = tools
        examples.append(example)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand full retail trajectories into per-turn next-action SFT rows."
    )
    parser.add_argument("--data-dir", default="data/retail")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.output) if args.output else data_dir / f"sft_{args.split}_next_action.jsonl"
    records = load_cached_split(data_dir, args.split, limit=args.limit_records)
    examples: list[dict[str, Any]] = []
    for record in records:
        examples.extend(record_to_next_action_examples(record))
        if args.max_examples is not None and len(examples) >= args.max_examples:
            examples = examples[: args.max_examples]
            break

    count = write_jsonl(output, examples)
    print(
        {
            "data_dir": str(data_dir),
            "split": args.split,
            "records": len(records),
            "examples": count,
            "output": str(output),
        }
    )


if __name__ == "__main__":
    main()
