from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any, Iterable

from course.shared.data import load_cached_split, normalize_messages_for_sft, write_jsonl


DEFAULT_TEACHER_DATASET = "amityco/tau-bench-retail-train-next-action-all-step-score-v0.2"


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
    call_id = call.get("id") or f"call_teacher_{row_index:06d}_{message_index:03d}_{call_index:02d}"
    return {
        "id": str(call_id),
        "type": "function",
        "function": {
            "name": str(name or ""),
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
    }


def normalize_chat_messages(messages: list[dict[str, Any]], *, row_index: int) -> list[dict[str, Any]]:
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
            tool_call_id = f"call_teacher_{row_index:06d}_{message_index:03d}_tool"
        normalized.append(
            {
                "role": "tool",
                "tool_call_id": str(tool_call_id),
                "content": str(message.get("content") or ""),
            }
        )
    return normalized


def assistant_answer_messages(answer: Any) -> list[dict[str, Any]]:
    if isinstance(answer, dict):
        return [answer]
    if isinstance(answer, list):
        return [item for item in answer if isinstance(item, dict)]
    return []


def answer_tool_names(answer: Any) -> set[str]:
    names: set[str] = set()
    for message in assistant_answer_messages(answer):
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = call.get("name") or function.get("name")
            if name:
                names.add(str(name))
    return names


def score_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


def passes_score_filter(row: dict[str, Any], *, min_total_score: float | None, min_avg_score: float | None) -> bool:
    total_score = score_value(row, "total_score")
    avg_score = score_value(row, "avg_score")
    if min_total_score is not None and total_score is not None and total_score < min_total_score:
        return False
    if min_avg_score is not None and avg_score is not None and avg_score < min_avg_score:
        return False
    return True


def convert_teacher_row(
    row: dict[str, Any],
    *,
    row_index: int,
    tools: list[dict[str, Any]],
    dataset_id: str,
    split: str,
    drop_unknown_tools: bool,
    min_total_score: float | None,
    min_avg_score: float | None,
) -> dict[str, Any] | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None
    answer = row.get("answer")
    answer_messages = assistant_answer_messages(answer)
    if not answer_messages:
        return None
    if not passes_score_filter(row, min_total_score=min_total_score, min_avg_score=min_avg_score):
        return None

    known_tool_names = {
        str((tool.get("function") or {}).get("name"))
        for tool in tools
        if (tool.get("function") or {}).get("name")
    }
    unknown_tools = answer_tool_names(answer) - known_tool_names if known_tool_names else set()
    if drop_unknown_tools and unknown_tools:
        return None

    messages = normalize_chat_messages(conversations, row_index=row_index)
    messages.extend(normalize_chat_messages(answer_messages, row_index=row_index))
    if not messages or messages[-1].get("role") != "assistant":
        return None

    sample_idx = row.get("sample_idx")
    metadata: dict[str, Any] = {
        "source_dataset": dataset_id,
        "source_split": split,
        "source_row_index": row_index,
        "source_sample_idx": sample_idx,
        "total_score": score_value(row, "total_score"),
        "avg_score": score_value(row, "avg_score"),
        "answer_tool_names": sorted(answer_tool_names(answer)),
        "sft_format": "teacher-retail-next-action",
    }
    if unknown_tools:
        metadata["unknown_answer_tools"] = sorted(unknown_tools)

    example: dict[str, Any] = {
        "messages": normalize_messages_for_sft(messages),
        "metadata": metadata,
    }
    if tools:
        example["tools"] = tools
    return example


def viewer_rows(dataset_id: str, split: str, *, limit: int | None, max_source_rows: int | None) -> Iterable[dict[str, Any]]:
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
        if limit is not None and seen >= max(limit * 20, limit):
            # Keep viewer fallback bounded when score/unknown-tool filters are selective.
            break


def load_source_rows(dataset_id: str, split: str, *, max_source_rows: int | None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        yield from viewer_rows(dataset_id, split, limit=None, max_source_rows=max_source_rows)
        return

    try:
        dataset = load_dataset(dataset_id, split=split, streaming=True)
        for row_index, row in enumerate(dataset):
            if max_source_rows is not None and row_index >= max_source_rows:
                break
            yield dict(row)
    except Exception:
        yield from viewer_rows(dataset_id, split, limit=None, max_source_rows=max_source_rows)


def optional_score(value: float) -> float | None:
    return None if value < 0 else value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert public teacher next-action retail rows to ART SFT JSONL."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_TEACHER_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tools-data-dir", default="data/retail")
    parser.add_argument("--output", default="data/retail/sft_teacher_retail_next_action.jsonl")
    parser.add_argument("--limit", type=int, default=500, help="Maximum converted rows to write.")
    parser.add_argument("--max-source-rows", type=int, default=None, help="Maximum source rows to scan.")
    parser.add_argument("--min-total-score", type=float, default=1.0, help="Use -1 to disable this filter.")
    parser.add_argument("--min-avg-score", type=float, default=1.0, help="Use -1 to disable this filter.")
    parser.add_argument("--keep-unknown-tools", action="store_true")
    parser.add_argument(
        "--allow-small-tool-schema",
        action="store_true",
        help="Allow converting with a tiny smoke-test tool schema. Not recommended for retail teacher SFT.",
    )
    args = parser.parse_args()

    tools = load_retail_tools(Path(args.tools_data_dir))
    if len(tools) < 8 and not args.allow_small_tool_schema:
        raise SystemExit(
            f"Only found {len(tools)} tools in {args.tools_data_dir}. "
            "Teacher retail conversion expects the full retail tool schema; run "
            "download_tau_retail.py first or pass --allow-small-tool-schema for a smoke diagnostic."
        )

    converted: list[dict[str, Any]] = []
    source_rows = 0
    for row_index, row in enumerate(
        load_source_rows(args.dataset_id, args.split, max_source_rows=args.max_source_rows)
    ):
        source_rows += 1
        example = convert_teacher_row(
            row,
            row_index=row_index,
            tools=tools,
            dataset_id=args.dataset_id,
            split=args.split,
            drop_unknown_tools=not args.keep_unknown_tools,
            min_total_score=optional_score(args.min_total_score),
            min_avg_score=optional_score(args.min_avg_score),
        )
        if example is not None:
            converted.append(example)
        if args.limit is not None and len(converted) >= args.limit:
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
            "min_total_score": optional_score(args.min_total_score),
            "min_avg_score": optional_score(args.min_avg_score),
        }
    )


if __name__ == "__main__":
    main()
