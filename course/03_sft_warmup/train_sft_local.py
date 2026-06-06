from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import asyncio
import copy
import json
import math
import os
import random
from pathlib import Path

from course.shared.art_compat import make_local_backend, make_trainable_model, register_trainable_model
from course.shared.config import config_from_env
from course.shared.wandb_artifacts import log_checkpoint_artifact, use_wandb_artifact


def normalize_messages_for_sft(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Make OpenAI-style teacher messages friendly to model chat templates."""
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if message.get("role") == "assistant" and message.get("content") is None:
            message["content"] = ""
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


def validate_sft_tokenization(
    *,
    base_model: str,
    trajectories: list[object],
    max_seq_length: int | None,
) -> None:
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
        total_tokens += len(token_ids)
        if response_part in str(chat):
            marker_hits += 1
    print(
        "SFT tokenization preflight:",
        {
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
        },
    )
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
    parser.add_argument("--skip-sft-preflight", action="store_true")
    args = parser.parse_args()

    sft_path = Path(args.file)
    if not sft_path.exists():
        raise SystemExit(f"Missing {args.file}. Run make_sft_jsonl.py first.")

    cfg = config_from_env()
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
    from art.utils.sft import create_lr_schedule, train_sft_from_file

    if not args.skip_sft_preflight:
        validate_sft_tokenization(
            base_model=cfg.base_model,
            trajectories=load_sft_trajectories(sft_path),
            max_seq_length=cfg.art_max_seq_length,
        )

    if args.chunk_size_batches <= 0:
        await train_sft_from_file(
            model=model,
            file_path=str(sft_path),
            epochs=args.epochs,
            batch_size=args.batch_size,
            peak_lr=args.peak_lr,
            schedule_type=args.schedule_type,
            warmup_ratio=args.warmup_ratio,
            final_step=args.max_steps,
            verbose=True,
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
                    "data_artifact": data_artifact,
                    "sft_rows": len(load_sft_trajectories(sft_path)),
                },
            )
        print("SFT complete. Current step:", step)
        return

    base_trajectories = load_sft_trajectories(sft_path)
    if not base_trajectories:
        raise SystemExit(f"No SFT rows found in {sft_path}")

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
                "chunk_size_batches": args.chunk_size_batches,
                "total_batches": total_batches,
                "chunks": chunk_index,
                "data_artifact": data_artifact,
                "sft_rows": len(base_trajectories),
            },
        )
    print("SFT complete. Current step:", step)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
