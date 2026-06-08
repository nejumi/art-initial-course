from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any, Iterable

from course.shared.data import normalize_messages_for_sft, write_jsonl
from course.shared.retail_env import extract_tool_calls_from_messages


DEFAULT_DATASET_ID = "fuvty/tau-bench-synthetic"
DEFAULT_TRAJ_CONFIG = "traj-GLM5"
DEFAULT_SFT_CONFIG = "sft-GLM5"


def parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        cleaned: list[Any] = []
        for item in value:
            normalized = clean_json_value(item)
            if normalized is not None:
                cleaned.append(normalized)
        return cleaned
    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for key, item in value.items():
            normalized = clean_json_value(item)
            if normalized is not None:
                cleaned_dict[str(key)] = normalized
        return cleaned_dict
    return str(value)


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
    call_id = call.get("id") or f"call_tau_synth_{row_index:06d}_{message_index:03d}_{call_index:02d}"
    return {
        "id": str(call_id),
        "type": "function",
        "function": {
            "name": str(name or ""),
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
    }


def normalize_messages(raw_messages: Any, *, row_index: int) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(raw_messages)
    if not isinstance(parsed, list):
        return []

    messages: list[dict[str, Any]] = []
    pending_tool_call_ids: list[str] = []
    for message_index, raw_message in enumerate(parsed):
        message = parse_json_maybe(raw_message)
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if role in {"system", "user"}:
            messages.append({"role": role, "content": str(message.get("content") or "")})
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
            messages.append(out)
            continue

        tool_call_id = message.get("tool_call_id") or (pending_tool_call_ids.pop(0) if pending_tool_call_ids else None)
        if tool_call_id is None:
            tool_call_id = f"call_tau_synth_{row_index:06d}_{message_index:03d}_tool"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(tool_call_id),
                "content": str(message.get("content") or ""),
            }
        )
    return messages


def normalize_tools(raw_tools: Any) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(raw_tools)
    if not isinstance(parsed, list):
        return []
    tools: list[dict[str, Any]] = []
    for raw_tool in parsed:
        tool = parse_json_maybe(raw_tool)
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        tools.append({"type": "function", "function": clean_json_value(function)})
    return tools


def numeric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


def holdout_bucket(key: str, modulo: int) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def task_key(row: dict[str, Any], fallback: int) -> str:
    for key in ("task_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(fallback)


def viewer_rows(
    dataset_id: str,
    config: str,
    split: str,
    *,
    max_source_rows: int | None,
) -> Iterable[dict[str, Any]]:
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


def source_rows(
    dataset_id: str,
    config: str,
    split: str,
    *,
    max_source_rows: int | None,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        yield from viewer_rows(dataset_id, config, split, max_source_rows=max_source_rows)
        return
    try:
        dataset = load_dataset(dataset_id, config, split=split, streaming=True)
    except Exception:
        yield from viewer_rows(dataset_id, config, split, max_source_rows=max_source_rows)
        return
    for row_index, row in enumerate(dataset):
        if max_source_rows is not None and row_index >= max_source_rows:
            break
        yield dict(row)


def convert_traj_row(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_id: str,
    config: str,
    split: str,
    domain: str,
    min_reward: float | None,
) -> dict[str, Any] | None:
    if row.get("domain") != domain:
        return None
    reward = numeric_value(row, "reward")
    if min_reward is not None and (reward is None or reward < min_reward):
        return None
    messages = normalize_messages(row.get("messages"), row_index=row_index)
    tools = normalize_tools(row.get("tools"))
    if not messages or not tools:
        return None
    if not extract_tool_calls_from_messages(messages):
        return None
    task = task_key(row, row_index)
    trial = row.get("trial")
    record_id = f"tau-synthetic-{domain}-traj-{task}-{trial if trial is not None else row_index}"
    return {
        "id": record_id,
        "split": split,
        "messages": messages,
        "tools": tools,
        "metadata": {
            "source_dataset": dataset_id,
            "source_config": config,
            "source_split": split,
            "source_row_index": row_index,
            "source_task_id": row.get("task_id"),
            "task_id": task,
            "trial": trial,
            "domain": domain,
            "reward": reward,
            "termination_reason": row.get("termination_reason"),
            "seconds": numeric_value(row, "seconds"),
        },
    }


def convert_sft_row(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_id: str,
    config: str,
    split: str,
    domain: str,
    holdout_modulo: int,
    sft_remainder: int,
) -> dict[str, Any] | None:
    if row.get("domain") != domain:
        return None
    task = task_key(row, row_index)
    if holdout_bucket(task, holdout_modulo) != sft_remainder:
        return None
    messages = normalize_messages(row.get("messages"), row_index=row_index)
    tools = normalize_tools(row.get("tools"))
    if not messages or not tools or messages[-1].get("role") != "assistant":
        return None
    return {
        "messages": normalize_messages_for_sft(messages),
        "tools": tools,
        "metadata": {
            "source_dataset": dataset_id,
            "source_config": config,
            "source_split": split,
            "source_row_index": row_index,
            "source_task_id": row.get("task_id"),
            "task_id": task,
            "trial": row.get("trial"),
            "round": row.get("round"),
            "total_rounds": row.get("total_rounds"),
            "domain": domain,
            "sft_format": "tau-synthetic-retail-next-action",
            "sft_holdout_modulo": holdout_modulo,
            "sft_remainder": sft_remainder,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare fuvty/tau-bench-synthetic retail rows for the ART course appendix."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--traj-config", default=DEFAULT_TRAJ_CONFIG)
    parser.add_argument("--sft-config", default=DEFAULT_SFT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--output-dir", default=".art/full_data/tau_synthetic_retail")
    parser.add_argument("--sft-output", default=None)
    parser.add_argument("--max-traj-source-rows", type=int, default=None)
    parser.add_argument("--max-sft-source-rows", type=int, default=None)
    parser.add_argument("--min-traj-reward", type=float, default=1.0, help="Use -1 to keep failed traces too.")
    parser.add_argument("--holdout-modulo", type=int, default=6)
    parser.add_argument("--sft-remainder", type=int, default=2)
    args = parser.parse_args()

    if args.holdout_modulo <= 1:
        raise SystemExit("--holdout-modulo must be greater than 1.")
    if not 0 <= args.sft_remainder < args.holdout_modulo:
        raise SystemExit("--sft-remainder must be in [0, holdout_modulo).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_output = Path(args.sft_output) if args.sft_output else output_dir / "sft_next_action_tau_synthetic.jsonl"
    sft_output.parent.mkdir(parents=True, exist_ok=True)

    traj_records: list[dict[str, Any]] = []
    scanned_traj = 0
    for row_index, row in enumerate(
        source_rows(
            args.dataset_id,
            args.traj_config,
            args.split,
            max_source_rows=args.max_traj_source_rows,
        )
    ):
        scanned_traj += 1
        record = convert_traj_row(
            row,
            row_index=row_index,
            dataset_id=args.dataset_id,
            config=args.traj_config,
            split=args.split,
            domain=args.domain,
            min_reward=None if args.min_traj_reward < 0 else args.min_traj_reward,
        )
        if record is not None:
            traj_records.append(record)

    sft_rows: list[dict[str, Any]] = []
    scanned_sft = 0
    for row_index, row in enumerate(
        source_rows(
            args.dataset_id,
            args.sft_config,
            args.split,
            max_source_rows=args.max_sft_source_rows,
        )
    ):
        scanned_sft += 1
        example = convert_sft_row(
            row,
            row_index=row_index,
            dataset_id=args.dataset_id,
            config=args.sft_config,
            split=args.split,
            domain=args.domain,
            holdout_modulo=args.holdout_modulo,
            sft_remainder=args.sft_remainder,
        )
        if example is not None:
            sft_rows.append(example)

    # The downstream bridge builder performs the task-disjoint SFT/train/validation/test
    # split, so successful source trajectories are intentionally written to train.jsonl.
    train_count = write_jsonl(output_dir / "train.jsonl", traj_records)
    write_jsonl(output_dir / "validation.jsonl", [])
    write_jsonl(output_dir / "test.jsonl", [])
    sft_count = write_jsonl(sft_output, sft_rows)
    summary = {
        "dataset_id": args.dataset_id,
        "domain": args.domain,
        "traj_config": args.traj_config,
        "sft_config": args.sft_config,
        "split": args.split,
        "scanned_traj_rows": scanned_traj,
        "written_traj_rows": train_count,
        "unique_traj_tasks": len({str(row.get("metadata", {}).get("task_id")) for row in traj_records}),
        "scanned_sft_rows": scanned_sft,
        "written_sft_rows": sft_count,
        "unique_sft_tasks": len({str(row.get("metadata", {}).get("task_id")) for row in sft_rows}),
        "holdout_modulo": args.holdout_modulo,
        "sft_remainder": args.sft_remainder,
        "min_traj_reward": None if args.min_traj_reward < 0 else args.min_traj_reward,
        "output_dir": str(output_dir),
        "sft_output": str(sft_output),
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (sft_output.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
