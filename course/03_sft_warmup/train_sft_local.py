from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import json
import math
import os
import random
from pathlib import Path
from typing import Any

try:
    import unsloth  # noqa: F401
except ImportError:
    unsloth = None  # type: ignore[assignment]

from course.shared.art_compat import make_local_backend, make_trainable_model, register_trainable_model
from course.shared.config import config_from_env
from course.shared.data import normalize_messages_for_sft as normalize_demo_messages_for_sft
from course.shared.tracing import init_weave
from course.shared.wandb_artifacts import log_checkpoint_artifact, sha256_file, use_wandb_artifact

SFT_MASK_MODES = ("all-assistant", "last-assistant")


def normalize_messages_for_sft(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Make OpenAI-style teacher messages friendly to model chat templates."""
    normalized = normalize_demo_messages_for_sft(messages)
    for message in normalized:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
    return normalized


def load_sft_trajectories(path: Path) -> list[object]:
    import art

    trajectories = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            trajectories.append(
                art.Trajectory(
                    messages_and_choices=normalize_messages_for_sft(row.get("messages", [])),
                    tools=row.get("tools"),
                )
            )
    return trajectories


def inspect_sft_file(path: Path) -> dict[str, Any]:
    rows = 0
    format_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    sample_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            meta = row.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            sft_format = str(meta.get("sft_format") or "unknown")
            format_counts[sft_format] = format_counts.get(sft_format, 0) + 1
            source = str(meta.get("source_dataset") or meta.get("source") or meta.get("source_repo") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            if "course-smoke-sample" in json.dumps(meta, ensure_ascii=False):
                sample_rows += 1
    stats = {
        "rows": rows,
        "sha256": sha256_file(path),
        "sft_format_counts": format_counts,
        "source_counts": source_counts,
        "course_smoke_sample_rows": sample_rows,
    }
    print("SFT file inspection:", stats)
    return stats


def validate_mask_mode_for_sft_file(stats: dict[str, Any], *, mask_mode: str, allow_mismatch: bool) -> None:
    format_counts = stats.get("sft_format_counts") or {}
    next_action_rows = sum(
        count
        for name, count in format_counts.items()
        if isinstance(name, str) and "next-action" in name
    )
    if next_action_rows and mask_mode != "last-assistant" and not allow_mismatch:
        raise SystemExit(
            "This SFT file contains next-action rows, but --sft-mask-mode is not "
            "`last-assistant`. Re-run with `--sft-mask-mode last-assistant`, or pass "
            "`--allow-sft-mask-mismatch` only for an intentional diagnostic run."
        )


def log_sft_progress(metrics: dict[str, Any]) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log(metrics)


def mask_only_last_trainable_span(labels: list[int]) -> list[int]:
    """Keep only the final contiguous response span in response-only SFT labels."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label != -100 and start is None:
            start = index
        elif label == -100 and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    if len(spans) <= 1:
        return labels

    keep_start, keep_end = spans[-1]
    return [
        label if keep_start <= index < keep_end else -100
        for index, label in enumerate(labels)
    ]


def tokenize_sft_batch_last_assistant(
    trajectory_batch: list[object],
    learning_rate: float,
    tokenizer: Any,
    instruction_part: str,
    response_part: str,
    chat_template_kwargs: dict[str, Any] | None = None,
    chat_template_tool_schema_format: str = "default",
    max_seq_length: int | None = None,
) -> object:
    """ART LocalBackend SFT tokenizer variant for next-action teacher data.

    ART's default LocalBackend SFT path masks every assistant response in the
    rendered chat. That is right for full-trajectory imitation, but next-action
    datasets repeat previous assistant turns as context. This variant keeps only
    the final response span after ART/Unsloth response-only masking.
    """
    import torch
    from unsloth_zoo.dataset_utils import train_on_responses_only

    from art.preprocessing.tokenize import (
        SFTBatch,
        _apply_chat_template_token_ids,
        _chat_template_kwargs,
        _messages_for_chat_template,
        _normalize_tools_for_chat_template,
    )

    if max_seq_length is not None and max_seq_length < 1:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")

    train_on_responses_only_fn = train_on_responses_only(
        trainer=None,
        instruction_part=instruction_part,
        response_part=response_part,
        force_match=False,
        tokenizer=tokenizer,
        return_function=True,
    )
    trajectory_tensors = []
    num_tokens = 0
    num_trainable_tokens = 0
    num_dropped_trajectories = 0
    template_kwargs = _chat_template_kwargs(tokenizer, chat_template_kwargs)
    for trajectory in trajectory_batch:
        messages = _messages_for_chat_template(tokenizer, trajectory.messages_and_choices)
        tools = _normalize_tools_for_chat_template(
            trajectory.tools,
            tool_schema_format=chat_template_tool_schema_format,
        )
        input_ids = _apply_chat_template_token_ids(
            tokenizer,
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            **template_kwargs,
        )
        if max_seq_length is not None and len(input_ids) > max_seq_length:
            num_dropped_trajectories += 1
            continue

        attention_mask = [1] * len(input_ids)
        labels = train_on_responses_only_fn({"input_ids": [input_ids]})["labels"][0]
        labels = mask_only_last_trainable_span(list(labels))
        trajectory_tensors.append(
            {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
            }
        )
        num_tokens += sum(attention_mask)
        num_trainable_tokens += sum(1 for label in labels if label != -100)

    if num_dropped_trajectories:
        print(
            "WARNING: Dropped "
            f"{num_dropped_trajectories}/{len(trajectory_batch)} SFT trajectories "
            f"because they exceed max_seq_length={max_seq_length}."
        )

    return SFTBatch(
        trajectory_tensors=trajectory_tensors,
        learning_rate=learning_rate,
        num_trajectories=len(trajectory_tensors),
        num_tokens=num_tokens,
        num_trainable_tokens=num_trainable_tokens,
        num_dropped_trajectories=num_dropped_trajectories,
    )


def patch_local_sft_masking(mask_mode: str) -> None:
    if mask_mode == "all-assistant":
        return
    if mask_mode != "last-assistant":
        raise ValueError(f"Unsupported SFT mask mode: {mask_mode}")
    import art.local.backend as local_backend

    local_backend.tokenize_sft_batch = tokenize_sft_batch_last_assistant
    print("Patched ART LocalBackend SFT tokenizer: mask_mode=last-assistant")


def validate_sft_tokenization(
    *,
    base_model: str,
    trajectories: list[object],
    max_seq_length: int | None,
) -> tuple[list[object], dict[str, Any]]:
    from transformers import AutoTokenizer

    from art.preprocessing.tokenize import (
        _apply_chat_template_token_ids,
        _chat_template_kwargs,
        _messages_for_chat_template,
        _normalize_tools_for_chat_template,
    )
    from art.utils.model_config import get_instruction_response_parts

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    instruction_part, response_part = get_instruction_response_parts(base_model, tokenizer)
    kept = 0
    dropped = 0
    marker_hits = 0
    total_tokens = 0
    render_errors = 0
    kept_trajectories: list[object] = []
    template_kwargs = _chat_template_kwargs(tokenizer, None)
    for trajectory in trajectories:
        try:
            messages = _messages_for_chat_template(tokenizer, trajectory.messages_and_choices)
            tools = _normalize_tools_for_chat_template(trajectory.tools)
            chat = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                **template_kwargs,
            )
            token_ids = _apply_chat_template_token_ids(
                tokenizer,
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
                **template_kwargs,
            )
        except Exception as exc:
            render_errors += 1
            if render_errors <= 3:
                print("SFT tokenization preflight render error:", repr(exc))
            continue
        if max_seq_length is not None and len(token_ids) > max_seq_length:
            dropped += 1
            continue
        kept += 1
        kept_trajectories.append(trajectory)
        total_tokens += len(token_ids)
        if response_part in str(chat):
            marker_hits += 1
    stats = {
        "base_model": base_model,
        "rows": len(trajectories),
        "kept": kept,
        "dropped": dropped,
        "render_errors": render_errors,
        "max_seq_length": max_seq_length,
        "tokens": total_tokens,
        "instruction_part": instruction_part,
        "response_part": response_part,
        "response_marker_hits": marker_hits,
    }
    print("SFT tokenization preflight:", stats)
    if kept == 0:
        raise SystemExit(
            "All SFT rows exceed ART_MAX_SEQ_LENGTH. Increase ART_MAX_SEQ_LENGTH "
            "or generate shorter SFT examples before training."
        )
    if marker_hits == 0:
        raise SystemExit(
            "SFT preflight found no response markers. Check the model chat template "
            "and ART instruction/response markers before continuing."
        )
    return kept_trajectories, stats


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run local ART SFT on retail JSONL.")
    parser.add_argument("--file", default="data/retail/sft_train.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--peak-lr", type=float, default=2e-4)
    parser.add_argument("--schedule-type", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum SFT optimizer batches. With chunked SFT, ART checkpoint count is ceil(max_steps / chunk_size_batches).",
    )
    parser.add_argument(
        "--chunk-size-batches",
        type=int,
        default=24,
        help="Number of SFT optimizer batches per ART checkpoint/log point. Use 0 for a single ART SFT call.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-cost-per-hour-usd", type=float, default=None)
    parser.add_argument("--data-artifact", default=None, help="Optional W&B dataset artifact URI to mark as SFT input.")
    parser.add_argument("--checkpoint-artifact-name", default=None)
    parser.add_argument("--no-log-checkpoint-artifact", action="store_true")
    parser.add_argument("--weave", action="store_true", help="Initialize Weave for this SFT run.")
    parser.add_argument("--skip-sft-preflight", action="store_true")
    parser.add_argument(
        "--sft-mask-mode",
        choices=SFT_MASK_MODES,
        default="all-assistant",
        help=(
            "Use all assistant turns for full-trajectory imitation, or only the "
            "last assistant turn for next-action SFT datasets such as AReaL."
        ),
    )
    parser.add_argument(
        "--keep-overlength-sft-rows",
        action="store_true",
        help=(
            "Keep rows that exceed ART_MAX_SEQ_LENGTH and let ART drop them internally. "
            "By default, preflight-filtered rows are used for cleaner loss curves."
        ),
    )
    parser.add_argument(
        "--allow-sft-mask-mismatch",
        action="store_true",
        help="Allow next-action SFT rows without last-assistant masking for intentional diagnostics.",
    )
    args = parser.parse_args()

    sft_path = Path(args.file)
    if not sft_path.exists():
        raise SystemExit(f"Missing {args.file}. Run make_sft_jsonl.py first.")
    sft_file_stats = inspect_sft_file(sft_path)
    validate_mask_mode_for_sft_file(
        sft_file_stats,
        mask_mode=args.sft_mask_mode,
        allow_mismatch=args.allow_sft_mask_mismatch,
    )

    cfg = config_from_env()
    if args.weave:
        init_weave(cfg.project)
    patch_local_sft_masking(args.sft_mask_mode)
    backend = make_local_backend(cfg.art_path, gpu_cost_per_hour_usd=args.gpu_cost_per_hour_usd)
    model = make_trainable_model(cfg)
    await register_trainable_model(model, backend, cfg)
    data_artifact = args.data_artifact or os.getenv("RETAIL_DATA_ARTIFACT")
    if data_artifact:
        use_wandb_artifact(
            cfg,
            data_artifact,
            artifact_type="dataset",
            job_type="sft",
            use_as="sft-training-data",
        )

    from art.types import TrainSFTConfig
    from art.utils.sft import create_lr_schedule

    base_trajectories = load_sft_trajectories(sft_path)
    if not base_trajectories:
        raise SystemExit(f"No SFT rows found in {sft_path}")
    preflight_stats: dict[str, Any] | None = None
    if not args.skip_sft_preflight:
        filtered_trajectories, preflight_stats = validate_sft_tokenization(
            base_model=cfg.base_model,
            trajectories=base_trajectories,
            max_seq_length=cfg.art_max_seq_length,
        )
        if not args.keep_overlength_sft_rows:
            base_trajectories = filtered_trajectories

    if args.chunk_size_batches <= 0:
        single_call_trajectories: list[object] = []
        for epoch in range(args.epochs):
            epoch_trajectories = list(base_trajectories)
            random.Random(args.seed + epoch).shuffle(epoch_trajectories)
            single_call_trajectories.extend(epoch_trajectories)
        if args.max_steps is not None:
            single_call_trajectories = single_call_trajectories[: args.max_steps * args.batch_size]
        total_batches = math.ceil(len(single_call_trajectories) / args.batch_size)
        learning_rates = create_lr_schedule(
            total_steps=total_batches,
            peak_lr=args.peak_lr,
            method=args.schedule_type,
            warmup_steps=int(total_batches * args.warmup_ratio),
        )
        await model.train_sft(
            single_call_trajectories,
            TrainSFTConfig(learning_rate=learning_rates, batch_size=args.batch_size),
            verbose=True,
        )
        log_sft_progress(
            {
                "sft/chunk_index": 1,
                "sft/optimizer_batch_cursor": total_batches,
                "sft/total_batches": total_batches,
                "sft/lr_mean": sum(learning_rates) / len(learning_rates) if learning_rates else 0.0,
                "sft/rows_trainable": len(single_call_trajectories),
                "sft/mask_mode": args.sft_mask_mode,
            }
        )
        step = await model.get_step()
        if not args.no_log_checkpoint_artifact:
            log_checkpoint_artifact(
                cfg,
                stage="sft",
                artifact_name=args.checkpoint_artifact_name,
                aliases=["sft-anchor"],
                metadata={
                    "algorithm": "sft",
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "peak_lr": args.peak_lr,
                    "schedule_type": args.schedule_type,
                    "warmup_ratio": args.warmup_ratio,
                    "sft_mask_mode": args.sft_mask_mode,
                    "sft_file": str(sft_path),
                    "sft_file_sha256": sft_file_stats["sha256"],
                    "sft_format_counts": sft_file_stats["sft_format_counts"],
                    "sft_source_counts": sft_file_stats["source_counts"],
                    "data_artifact": data_artifact,
                    "sft_rows": len(base_trajectories),
                    "sft_preflight": preflight_stats,
                },
            )
        print("SFT complete. Current step:", step)
        return

    batches_per_epoch = math.ceil(len(base_trajectories) / args.batch_size)
    total_batches = batches_per_epoch * args.epochs
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps)
    learning_rates = create_lr_schedule(
        total_steps=total_batches,
        peak_lr=args.peak_lr,
        method=args.schedule_type,
        warmup_steps=int(total_batches * args.warmup_ratio),
    )

    batch_cursor = 0
    chunk_index = 0
    for epoch in range(args.epochs):
        epoch_trajectories = list(base_trajectories)
        random.Random(args.seed + epoch).shuffle(epoch_trajectories)
        chunk_stride = args.batch_size * args.chunk_size_batches
        for start in range(0, len(epoch_trajectories), chunk_stride):
            if batch_cursor >= total_batches:
                break
            chunk = epoch_trajectories[start : start + chunk_stride]
            chunk_batches = math.ceil(len(chunk) / args.batch_size)
            remaining_batches = total_batches - batch_cursor
            chunk_batches = min(chunk_batches, remaining_batches)
            chunk = chunk[: chunk_batches * args.batch_size]
            lr_slice = learning_rates[batch_cursor : batch_cursor + chunk_batches]
            if not chunk or not lr_slice:
                continue
            await model.train_sft(
                chunk,
                TrainSFTConfig(learning_rate=lr_slice, batch_size=args.batch_size),
                verbose=True,
            )
            chunk_index += 1
            batch_cursor += chunk_batches
            print(
                "SFT chunk complete:",
                {
                    "chunk": chunk_index,
                    "epoch": epoch,
                    "batches": chunk_batches,
                    "batch_cursor": batch_cursor,
                    "total_batches": total_batches,
                    "current_step": await model.get_step(),
                },
            )
            log_sft_progress(
                {
                    "sft/chunk_index": chunk_index,
                    "sft/epoch": epoch,
                    "sft/chunk_batches": chunk_batches,
                    "sft/optimizer_batch_cursor": batch_cursor,
                    "sft/total_batches": total_batches,
                    "sft/lr_mean": sum(lr_slice) / len(lr_slice) if lr_slice else 0.0,
                    "sft/rows_trainable": len(base_trajectories),
                    "sft/chunk_rows": len(chunk),
                    "sft/mask_mode": args.sft_mask_mode,
                }
            )
        if batch_cursor >= total_batches:
            break

    step = await model.get_step()
    if not args.no_log_checkpoint_artifact:
        log_checkpoint_artifact(
            cfg,
            stage="sft",
            artifact_name=args.checkpoint_artifact_name,
            aliases=["sft-anchor"],
            metadata={
                "algorithm": "sft",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "peak_lr": args.peak_lr,
                "schedule_type": args.schedule_type,
                "warmup_ratio": args.warmup_ratio,
                "sft_mask_mode": args.sft_mask_mode,
                "sft_file": str(sft_path),
                "sft_file_sha256": sft_file_stats["sha256"],
                "sft_format_counts": sft_file_stats["sft_format_counts"],
                "sft_source_counts": sft_file_stats["source_counts"],
                "chunk_size_batches": args.chunk_size_batches,
                "total_batches": total_batches,
                "chunks": chunk_index,
                "data_artifact": data_artifact,
                "sft_rows": len(base_trajectories),
                "sft_preflight": preflight_stats,
            },
        )
    print("SFT complete. Current step:", step)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
