from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import importlib
import json
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any, Iterable

from course.shared.data import (
    load_cached_split,
    normalize_messages_for_sft,
    write_jsonl,
)
from course.shared.retail_env import first_message_content


DEFAULT_SUCCESS_TRACE_DATASET = "KermitCO/qwen3.5-9B-tau2bench-retail-traces"


def load_retail_tools(data_dir: Path) -> list[dict[str, Any]]:
    for split in ("train", "validation", "test"):
        for row in load_cached_split(data_dir, split, limit=25):
            tools = row.get("tools") or []
            if tools:
                return list(tools)
    return []


def load_retail_system_message(data_dir: Path) -> str:
    for split in ("train", "validation", "test"):
        for row in load_cached_split(data_dir, split, limit=25):
            content = first_message_content(row.get("messages") or [], "system")
            if content:
                return content
    return "You are a helpful retail support agent. Use the provided tools to solve the customer's request."


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


def parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def as_message(value: Any) -> dict[str, Any] | None:
    parsed = parse_json_maybe(value)
    if isinstance(parsed, dict):
        return parsed
    return None


def normalize_tool_call(call: dict[str, Any], *, row_index: int, message_index: int, call_index: int) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = call.get("name") or function.get("name")
    arguments = call.get("arguments")
    if arguments is None:
        arguments = function.get("arguments") or {}
    arguments = parse_json_maybe(arguments)
    if isinstance(arguments, str):
        arguments = {"raw_arguments": arguments}
    arguments = clean_json_value(arguments or {})
    call_id = call.get("id") or f"call_trace_{row_index:06d}_{message_index:03d}_{call_index:02d}"
    return {
        "id": str(call_id),
        "type": "function",
        "function": {
            "name": str(name or ""),
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
    }


def normalize_trace_messages(
    messages: list[Any],
    *,
    row_index: int,
    system_message: str | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    pending_tool_call_ids: list[str] = []
    for message_index, raw_message in enumerate(messages):
        message = as_message(raw_message)
        if message is None:
            continue
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
            tool_call_id = f"call_trace_{row_index:06d}_{message_index:03d}_tool"
        normalized.append(
            {
                "role": "tool",
                "tool_call_id": str(tool_call_id),
                "content": str(message.get("content") or ""),
            }
        )

    if system_message and (not normalized or normalized[0].get("role") != "system"):
        normalized.insert(0, {"role": "system", "content": system_message})
    return normalized


def tool_names_from_messages(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = function.get("name") or call.get("name")
            if name:
                names.add(str(name))
    return names


def numeric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


def passes_trace_filters(
    row: dict[str, Any],
    *,
    min_reward: float | None,
    require_blind_strict: bool,
    allow_memory_injected: bool,
) -> bool:
    reward = numeric_value(row, "canonical_reward")
    if reward is None:
        reward = numeric_value(row, "reward")
    if min_reward is not None and (reward is None or reward < min_reward):
        return False
    if not allow_memory_injected and bool(row.get("memory_injected")):
        return False
    flags = row.get("condition_flags") or {}
    if require_blind_strict and isinstance(flags, dict) and not flags.get("C_blind_strict"):
        return False
    return True


def convert_trace_row(
    row: dict[str, Any],
    *,
    row_index: int,
    tools: list[dict[str, Any]],
    dataset_id: str,
    split: str,
    system_message: str | None,
    drop_unknown_tools: bool,
    min_reward: float | None,
    require_blind_strict: bool,
    allow_memory_injected: bool,
) -> dict[str, Any] | None:
    if not passes_trace_filters(
        row,
        min_reward=min_reward,
        require_blind_strict=require_blind_strict,
        allow_memory_injected=allow_memory_injected,
    ):
        return None
    source_messages = row.get("messages") or []
    if not isinstance(source_messages, list):
        return None
    messages = normalize_trace_messages(source_messages, row_index=row_index, system_message=system_message)
    if not any(message.get("role") == "user" for message in messages):
        return None
    if not any(message.get("role") == "assistant" and message.get("tool_calls") for message in messages):
        return None

    known_tool_names = {
        str((tool.get("function") or {}).get("name"))
        for tool in tools
        if (tool.get("function") or {}).get("name")
    }
    unknown_tools = tool_names_from_messages(messages) - known_tool_names if known_tool_names else set()
    if drop_unknown_tools and unknown_tools:
        return None

    reward = numeric_value(row, "canonical_reward")
    if reward is None:
        reward = numeric_value(row, "reward")
    example: dict[str, Any] = {
        "id": f"success-trace-{row.get('task_id', row_index)}-{row_index}",
        "split": split,
        "messages": normalize_messages_for_sft(messages),
        "metadata": {
            "source_dataset": dataset_id,
            "source_split": split,
            "source_row_index": row_index,
            "source_task_id": row.get("task_id"),
            "source_file": row.get("source_file"),
            "source_model": row.get("model"),
            "canonical_reward": reward,
            "memory_injected": bool(row.get("memory_injected")),
            "judge_process_quality": numeric_value(row, "judge_process_quality"),
            "judge_task_completion": numeric_value(row, "judge_task_completion"),
            "judge_action_quality": numeric_value(row, "judge_action_quality"),
            "judge_execution_efficiency": numeric_value(row, "judge_execution_efficiency"),
            "sft_format": "tau2-retail-success-trace-full",
        },
    }
    if unknown_tools:
        example["metadata"]["unknown_tools"] = sorted(unknown_tools)
    if tools:
        example["tools"] = tools
    return example


def viewer_rows(dataset_id: str, split: str, *, max_source_rows: int | None) -> Iterable[dict[str, Any]]:
    query = urllib.parse.urlencode({"dataset": dataset_id})
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/splits?{query}", timeout=60) as response:
        split_data = json.loads(response.read().decode("utf-8"))
    config = "default"
    for item in split_data.get("splits") or []:
        if item.get("split") == split:
            config = str(item.get("config") or "default")
            break

    offset = 0
    seen = 0
    while True:
        if max_source_rows is not None and seen >= max_source_rows:
            break
        length = min(100, max_source_rows - seen) if max_source_rows is not None else 100
        params = urllib.parse.urlencode(
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{params}", timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        page = data.get("rows") or []
        if not page:
            break
        for item in page:
            yield dict(item.get("row", item))
            seen += 1
            if max_source_rows is not None and seen >= max_source_rows:
                break
        offset += len(page)
        total = data.get("num_rows_total")
        if total is not None and offset >= int(total):
            break


def source_rows(dataset_id: str, split: str, *, max_source_rows: int | None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        yield from viewer_rows(dataset_id, split, max_source_rows=max_source_rows)
        return
    try:
        dataset = load_dataset(dataset_id, split=split, streaming=True)
    except Exception:
        yield from viewer_rows(dataset_id, split, max_source_rows=max_source_rows)
        return
    for row_index, row in enumerate(dataset):
        if max_source_rows is not None and row_index >= max_source_rows:
            break
        yield dict(row)


def to_next_action_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    next_action_module = importlib.import_module("course.03_sft_warmup.make_next_action_sft_jsonl")
    examples: list[dict[str, Any]] = []
    for record in records:
        for example in next_action_module.record_to_next_action_examples(record):
            metadata = example.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["sft_format"] = "tau2-retail-success-trace-next-action"
            examples.append(example)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert successful public tau2 retail traces to ART-compatible SFT JSONL."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_SUCCESS_TRACE_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tools-data-dir", default="data/retail")
    parser.add_argument("--output", default="data/retail/sft_success_trace_retail_next_action.jsonl")
    parser.add_argument("--full-output", default=None)
    parser.add_argument("--limit", type=int, default=200, help="Maximum converted examples to write.")
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--keep-unknown-tools", action="store_true")
    parser.add_argument("--min-reward", type=float, default=1.0, help="Use -1 to disable reward filtering.")
    parser.add_argument(
        "--no-require-blind-strict",
        action="store_true",
        help="Keep traces without the public dataset's C_blind_strict quality flag.",
    )
    parser.add_argument(
        "--allow-memory-injected",
        action="store_true",
        help="Allow traces generated with explicit memory/rule injection. Defaults to clean non-memory traces.",
    )
    parser.add_argument(
        "--mode",
        choices=["next-action", "full-trajectory"],
        default="next-action",
        help="Write per-turn next-action rows by default; use full-trajectory for all-assistant SFT.",
    )
    parser.add_argument(
        "--allow-small-tool-schema",
        action="store_true",
        help="Allow converting with a tiny smoke-test tool schema. Not recommended for public retail traces.",
    )
    args = parser.parse_args()

    tools_data_dir = Path(args.tools_data_dir)
    tools = load_retail_tools(tools_data_dir)
    if len(tools) < 8 and not args.allow_small_tool_schema:
        raise SystemExit(
            f"Only found {len(tools)} tools in {tools_data_dir}. "
            "Success-trace conversion expects the full retail tool schema; run "
            "download_tau_retail.py first or pass --allow-small-tool-schema for a smoke diagnostic."
        )
    system_message = load_retail_system_message(tools_data_dir)
    records: list[dict[str, Any]] = []
    scanned = 0
    for row_index, row in enumerate(
        source_rows(args.dataset_id, args.split, max_source_rows=args.max_source_rows)
    ):
        scanned += 1
        record = convert_trace_row(
            row,
            row_index=row_index,
            tools=tools,
            dataset_id=args.dataset_id,
            split=args.split,
            system_message=system_message,
            drop_unknown_tools=not args.keep_unknown_tools,
            min_reward=None if args.min_reward < 0 else args.min_reward,
            require_blind_strict=not args.no_require_blind_strict,
            allow_memory_injected=args.allow_memory_injected,
        )
        if record is not None:
            records.append(record)
        if args.limit is not None and args.mode == "full-trajectory" and len(records) >= args.limit:
            break

    if args.full_output:
        write_jsonl(args.full_output, records)

    rows = records if args.mode == "full-trajectory" else to_next_action_examples(records)
    if args.limit is not None:
        rows = rows[: args.limit]
    count = write_jsonl(args.output, rows)
    print(
        {
            "dataset_id": args.dataset_id,
            "split": args.split,
            "source_rows_scanned": scanned,
            "converted_records": len(records),
            "written_rows": count,
            "mode": args.mode,
            "output": args.output,
            "full_output": args.full_output,
            "tools": len(tools),
        }
    )


if __name__ == "__main__":
    main()
