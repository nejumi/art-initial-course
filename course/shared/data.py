from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_DATASET_ID, ensure_data_dir
from .retail_env import extract_tool_names_from_messages, first_message_content
from .schemas import RetailScenario

SPLITS = ("train", "validation", "test")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_record(row: dict[str, Any], *, split: str | None = None, index: int | None = None) -> dict[str, Any]:
    messages = row.get("messages") or row.get("conversation") or []
    tools = row.get("tools") or []
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {"raw_metadata": metadata}
    record_id = row.get("id") or metadata.get("id") or metadata.get("task_id") or f"{split or 'row'}-{index or 0}"
    return {"id": str(record_id), "messages": list(messages), "tools": list(tools), "metadata": dict(metadata), "split": split or row.get("split")}


def load_cached_split(data_dir: str | Path, split: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(data_dir) / f"{split}.jsonl")
    if limit is not None:
        rows = rows[:limit]
    return [normalize_record(row, split=split, index=i) for i, row in enumerate(rows)]


def load_hf_split(dataset_id: str, split: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the datasets package to download the retail data.") from exc
    dataset = load_dataset(dataset_id, split=split)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(dataset):
        rows.append(normalize_record(dict(row), split=split, index=i))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_split(
    split: str,
    *,
    data_dir: str | Path | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    limit: int | None = None,
    prefer_cache: bool = True,
) -> list[dict[str, Any]]:
    resolved_dir = ensure_data_dir(Path(data_dir) if data_dir else None)
    cached = load_cached_split(resolved_dir, split, limit=limit) if prefer_cache else []
    if cached:
        return cached
    rows = load_hf_split(dataset_id, split, limit=limit)
    write_jsonl(resolved_dir / f"{split}.jsonl", rows)
    return rows


def record_to_sft_example(record: dict[str, Any]) -> dict[str, Any]:
    example = {"messages": record["messages"]}
    if record.get("tools"):
        example["tools"] = record["tools"]
    return example


def write_sft_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> int:
    return write_jsonl(output_path, (record_to_sft_example(record) for record in records))


def scenario_from_record(record: dict[str, Any], *, split: str | None = None, index: int = 0) -> RetailScenario:
    normalized = normalize_record(record, split=split, index=index)
    messages = normalized["messages"]
    system_message = first_message_content(messages, "system")
    user_message = first_message_content(messages, "user")
    expected_tool_names = extract_tool_names_from_messages(messages)
    expected_final_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            expected_final_text = str(msg.get("content") or "")
            break
    return RetailScenario(
        id=normalized["id"],
        split=split or normalized.get("split") or "train",
        system_message=system_message,
        user_message=user_message,
        tools=normalized["tools"],
        reference_messages=messages,
        expected_tool_names=expected_tool_names,
        expected_final_text=expected_final_text,
        metadata=normalized["metadata"],
    )


def scenarios_from_records(records: list[dict[str, Any]], *, split: str) -> list[RetailScenario]:
    return [scenario_from_record(record, split=split, index=i) for i, record in enumerate(records)]


def sample_records() -> list[dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Look up order status and line items.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_pending_order",
                "description": "Cancel an order that has not shipped.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "return_delivered_order",
                "description": "Start a return for a delivered order.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "item_id": {"type": "string"}}, "required": ["order_id", "item_id"]},
            },
        },
    ]
    system = "You are a careful retail support agent. Follow policy and use tools before changing an order."
    return [
        {
            "id": "sample-cancel-001",
            "split": "train",
            "tools": tools,
            "metadata": {"source": "course-smoke-sample", "task": "cancel_pending_order"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Hi, I want to cancel order O-1001."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_order_details", "arguments": "{\"order_id\": \"O-1001\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"order_id\": \"O-1001\", \"status\": \"pending\", \"total\": 49.99}"},
                {"role": "assistant", "content": "Order O-1001 is still pending, so it can be canceled. I will cancel it now."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "cancel_pending_order", "arguments": "{\"order_id\": \"O-1001\"}"}}]},
                {"role": "tool", "tool_call_id": "call_2", "content": "{\"order_id\": \"O-1001\", \"status\": \"canceled\"}"},
                {"role": "assistant", "content": "Done. I canceled order O-1001 and no further action is needed."},
            ],
        },
        {
            "id": "sample-return-001",
            "split": "validation",
            "tools": tools,
            "metadata": {"source": "course-smoke-sample", "task": "return_delivered_order"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "The shoes from order O-2002 arrived damaged. Can I return them?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_order_details", "arguments": "{\"order_id\": \"O-2002\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"order_id\": \"O-2002\", \"status\": \"delivered\", \"items\": [{\"item_id\": \"I-9\", \"name\": \"shoes\"}]}"},
                {"role": "assistant", "content": "The order was delivered, so I can start a return for the damaged shoes."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "return_delivered_order", "arguments": "{\"order_id\": \"O-2002\", \"item_id\": \"I-9\"}"}}]},
                {"role": "tool", "tool_call_id": "call_2", "content": "{\"return_id\": \"R-3003\", \"status\": \"started\"}"},
                {"role": "assistant", "content": "I started return R-3003 for the damaged shoes from order O-2002."},
            ],
        },
    ]


def write_sample_dataset(data_dir: str | Path | None = None) -> None:
    resolved_dir = ensure_data_dir(Path(data_dir) if data_dir else None)
    rows = sample_records()
    for split in SPLITS:
        split_rows = [row for row in rows if row.get("split") == split]
        if not split_rows and split == "test":
            split_rows = rows[:1]
        write_jsonl(resolved_dir / f"{split}.jsonl", split_rows)
