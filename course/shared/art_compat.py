from __future__ import annotations

import os
from typing import Any

from .config import RetailCourseConfig


def require_art() -> Any:
    try:
        import art
    except ImportError as exc:
        raise RuntimeError('Install ART first: python -m pip install "openpipe-art[backend]"') from exc
    from .art_patches import install_art_tool_argument_normalizer

    install_art_tool_argument_normalizer()
    return art


def make_local_backend(path: str, gpu_cost_per_hour_usd: float | None = None) -> Any:
    try:
        from art.local import LocalBackend
    except ImportError as exc:
        raise RuntimeError('LocalBackend requires openpipe-art[backend].') from exc
    kwargs: dict[str, Any] = {"path": path}
    if gpu_cost_per_hour_usd is not None:
        kwargs["gpu_cost_per_hour_usd"] = gpu_cost_per_hour_usd
    return LocalBackend(**kwargs)


def use_active_wandb_run_for_art(model: Any) -> Any:
    """Prevent ART from creating a separate model-name W&B run.

    ART 0.5.x lazily calls wandb.init(name=model.name, id=model.name) from a
    private helper. The course uses W&B-generated run IDs and attaches ART
    metrics/artifacts to the active course run instead.
    """

    def course_wandb_run() -> Any | None:
        try:
            import wandb
        except ImportError:
            return None
        run = getattr(wandb, "run", None)
        if run is None or getattr(run, "_is_finished", False):
            return None
        object.__setattr__(model, "_wandb_run", run)
        sync_config = getattr(model, "_sync_wandb_config", None)
        if callable(sync_config):
            try:
                sync_config(run)
            except Exception:
                pass
        return run

    object.__setattr__(model, "_get_wandb_run", course_wandb_run)
    course_wandb_run()
    return model


def openai_server_config(config: RetailCourseConfig) -> dict[str, Any] | None:
    engine_args = vllm_engine_args(config)
    server_args: dict[str, Any] = {}
    if config.tool_call_parser:
        server_args["enable_auto_tool_choice"] = True
        server_args["tool_call_parser"] = config.tool_call_parser
    openai_config: dict[str, Any] = {}
    if engine_args:
        openai_config["engine_args"] = engine_args
    if server_args:
        openai_config["server_args"] = server_args
    return openai_config or None


def vllm_engine_args(config: RetailCourseConfig) -> dict[str, Any]:
    engine_args: dict[str, Any] = {}
    if config.vllm_max_model_len is not None:
        engine_args["max_model_len"] = config.vllm_max_model_len
    if config.vllm_gpu_memory_utilization is not None:
        engine_args["gpu_memory_utilization"] = config.vllm_gpu_memory_utilization
    if config.vllm_max_num_batched_tokens is not None:
        engine_args["max_num_batched_tokens"] = config.vllm_max_num_batched_tokens
    if config.vllm_max_num_seqs is not None:
        engine_args["max_num_seqs"] = config.vllm_max_num_seqs
    if config.vllm_enforce_eager is not None:
        engine_args["enforce_eager"] = config.vllm_enforce_eager
    return engine_args


def model_internal_config(config: RetailCourseConfig) -> dict[str, Any] | None:
    internal_config: dict[str, Any] = {}
    engine_args = vllm_engine_args(config)
    if engine_args:
        internal_config["engine_args"] = engine_args
    if config.art_max_seq_length is not None:
        max_seq = config.art_max_seq_length
        internal_config["init_args"] = {"max_seq_length": max_seq}
        internal_config["peft_args"] = {"max_seq_length": max_seq}
    return internal_config or None


async def register_trainable_model(model: Any, backend: Any, config: RetailCourseConfig) -> None:
    await model.register(backend, _openai_client_config=openai_server_config(config))


def make_trainable_model(config: RetailCourseConfig) -> Any:
    art = require_art()
    model = art.TrainableModel(
        name=config.model_name,
        project=config.project,
        entity=config.entity,
        base_model=config.base_model,
        base_path=config.art_path,
        report_metrics=[],
        _internal_config=model_internal_config(config),
    )
    use_active_wandb_run_for_art(model)
    model.update_wandb_config(
        {
            "dataset": config.dataset_id,
            "task": "retail-support-agent",
            "course": "openpipe-art-retail",
            "model_profile": config.model_profile,
            "base_model": config.base_model,
            "rollout_max_completion_tokens": config.rollout_max_completion_tokens,
            "tool_call_parser": config.tool_call_parser,
            "art_max_seq_length": config.art_max_seq_length,
            "vllm_engine_args": vllm_engine_args(config),
            "art_internal_config": model_internal_config(config),
        }
    )
    return model


def make_prompted_model(config: RetailCourseConfig, *, name: str | None = None) -> Any:
    art = require_art()
    api_key = config.inference_api_key or os.getenv("OPENAI_API_KEY")
    base_url = config.inference_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    inference_name = config.inference_model_name or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    display_name = name or inference_name
    model = art.Model(
        name=display_name,
        project=config.project,
        entity=config.entity,
        inference_api_key=api_key,
        inference_base_url=base_url,
        inference_model_name=inference_name,
        base_path=config.art_path,
        report_metrics=[],
    )
    return use_active_wandb_run_for_art(model)
