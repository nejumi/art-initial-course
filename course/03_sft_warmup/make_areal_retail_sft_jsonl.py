from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import date, datetime
from typing import Any

from course.shared.data import load_cached_split, normalize_messages_for_sft, write_jsonl


DEFAULT_AREAL_DATASET = "inclusionAI/AReaL-tau2-data"


def load_retail_tools(data_dir: Path) -> list[dict[str, Any]]:
    for split in ("train", "validation", "test"):
        for row in load_cached_split(data_dir, split, limit=25):
            tools = row.get("tools") or []
            if tools:
                return list(tools)
    return []


def clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            cleaned_item = clean_json_value(item)
            if cleaned_item is not None:
                cleaned_items.append(cleaned_item)
        return cleaned_items
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned_item = clean_json_value(item)
            if cleaned_item is not None:
                cleaned[str(key)] = cleaned_item
        return cleaned
    return str(value)


def normalize_tool_call(call: dict[str, Any], *, row_index: int, message_index: int, call_index: int) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = call.get("name") or function.get("name")
    arguments = call.get("arguments")
    if arguments is None:
        arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments}
    arguments = clean_json_value(arguments or {})
    call_id = call.get("id") or f"call_areal_{row_index:06d}_{message_index:03d}_{call_index:02d}"
    return {
        "id": str(call_id),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
    }


def normalize_areal_messages(messages: list[dict[str, Any]], *, row_index: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    pending_tool_call_ids: list[str] = []
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if role in {"system", "user"}:
            normalized.append({"role": role, "content": str(message.get("content") or "")})
            continue
        if role == "assistant":
            out: dict[str, Any] = {"role": "assistant", "content": str(message.get("content") or "")}
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                out["tool_calls"] = [
                    normalize_tool_call(call, row_index=row_index, message_index=message_index, call_index=call_index)
                    for call_index, call in enumerate(tool_calls)
                    if isinstance(call, dict)
                ]
                pending_tool_call_ids.extend(call["id"] for call in out["tool_calls"])
            normalized.append(out)
            continue
        tool_call_id = message.get("tool_call_id") or (pending_tool_call_ids.pop(0) if pending_tool_call_ids else None)
        if tool_call_id is None:
            tool_call_id = f"call_areal_{row_index:06d}_{message_index:03d}_tool"
        normalized.append(
            {
                "role": "tool",
                "tool_call_id": str(tool_call_id),
                "content": str(message.get("content") or ""),
            }
        )
    return normalized


def is_retail_row(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    source_dialog_id = str(metadata.get("source_dialog_id") or "")
    if source_dialog_id.startswith("retail"):
        return True
    messages = row.get("messages") or []
    system = str(messages[0].get("content") or "") if messages else ""
    return "Retail agent policy" in system or "# Retail agent policy" in system


def answer_tool_names(answer: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for call in answer.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = call.get("name") or function.get("name")
        if name:
            names.add(str(name))
    return names


def convert_areal_row(
    row: dict[str, Any],
    *,
    row_index: int,
    tools: list[dict[str, Any]],
    drop_unknown_tools: bool,
    dataset_id: str,
    require_correct: bool,
    min_reward: float | None,
) -> dict[str, Any] | None:
    metadata = row.get("metadata") or {}
    correct = metadata.get("correct")
    if require_correct and correct not in (1, True):
        return None
    if not require_correct and correct not in (None, 1, True):
        return None
    reward = metadata.get("reward")
    if min_reward is not None and isinstance(reward, int | float) and float(reward) < min_reward:
        return None
    if not is_retail_row(row):
        return None
    answer = row.get("answer")
    if not isinstance(answer, dict) or answer.get("role") != "assistant":
        return None
    known_tool_names = {
        str((tool.get("function") or {}).get("name"))
        for tool in tools
        if (tool.get("function") or {}).get("name")
    }
    unknown_tools = answer_tool_names(answer) - known_tool_names if known_tool_names else set()
    if drop_unknown_tools and unknown_tools:
        return None
    messages = normalize_areal_messages(list(row.get("messages") or []), row_index=row_index)
    messages.extend(normalize_areal_messages([answer], row_index=row_index))
    example: dict[str, Any] = {
        "messages": normalize_messages_for_sft(messages),
        "metadata": {
            "source_dataset": dataset_id,
            "source_split": "sft",
            "source_dialog_id": metadata.get("source_dialog_id"),
            "scenario_id": metadata.get("scenario_id"),
            "turn_index": metadata.get("turn_index"),
            "reward": metadata.get("reward"),
            "correct": metadata.get("correct"),
            "sft_format": "areal-retail-next-action",
        },
    }
    if tools:
        example["tools"] = tools
    return example


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AReaL tau2 retail SFT rows to ART JSONL.")
    parser.add_argument("--dataset-id", default=DEFAULT_AREAL_DATASET)
    parser.add_argument("--split", default="sft")
    parser.add_argument("--tools-data-dir", default="data/retail")
    parser.add_argument("--output", default="data/retail/sft_areal_retail_next_action.jsonl")
    parser.add_argument("--limit", type=int, default=500, help="Maximum converted examples to write.")
    parser.add_argument("--max-source-rows", type=int, default=None, help="Maximum source rows to scan.")
    parser.add_argument("--keep-unknown-tools", action="store_true")
    parser.add_argument(
        "--allow-uncertain-correct",
        action="store_true",
        help="Allow rows where metadata.correct is missing. The default keeps only correct rows.",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=1.0,
        help="If metadata.reward is numeric, keep rows with reward >= this value. Use -1 to disable.",
    )
    parser.add_argument(
        "--allow-small-tool-schema",
        action="store_true",
        help="Allow converting with a tiny smoke-test tool schema. Not recommended for AReaL retail SFT.",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    tools = load_retail_tools(Path(args.tools_data_dir))
    if len(tools) < 8 and not args.allow_small_tool_schema:
        raise SystemExit(
            f"Only found {len(tools)} tools in {args.tools_data_dir}. "
            "AReaL retail conversion expects the full retail tool schema; run "
            "download_tau_retail.py first or pass --allow-small-tool-schema for a smoke diagnostic."
        )
    converted: list[dict[str, Any]] = []
    source_rows = 0
    dataset = load_dataset(args.dataset_id, split=args.split, streaming=True)
    for row_index, row in enumerate(dataset):
        source_rows += 1
        example = convert_areal_row(
            dict(row),
            row_index=row_index,
            tools=tools,
            drop_unknown_tools=not args.keep_unknown_tools,
            dataset_id=args.dataset_id,
            require_correct=not args.allow_uncertain_correct,
            min_reward=None if args.min_reward < 0 else args.min_reward,
        )
        if example is not None:
            converted.append(example)
        if args.limit is not None and len(converted) >= args.limit:
            break
        if args.max_source_rows is not None and source_rows >= args.max_source_rows:
            break

    count = write_jsonl(args.output, converted)
    print(
        {
            "dataset_id": args.dataset_id,
            "split": args.split,
            "source_rows_scanned": source_rows,
            "converted_rows": count,
            "output": args.output,
            "tools": len(tools),
        }
    )


if __name__ == "__main__":
    main()
