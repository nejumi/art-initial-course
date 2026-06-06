from __future__ import annotations

import os
from typing import Any

from .config import RetailCourseConfig


def require_art() -> Any:
    try:
        import art
    except ImportError as exc:
        raise RuntimeError('Install ART first: python -m pip install "openpipe-art[backend]"') from exc
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


def make_trainable_model(config: RetailCourseConfig) -> Any:
    art = require_art()
    model = art.TrainableModel(
        name=config.model_name,
        project=config.project,
        entity=config.entity,
        base_model=config.base_model,
        base_path=config.art_path,
    )
    model.update_wandb_config(
        {
            "dataset": config.dataset_id,
            "task": "retail-support-agent",
            "course": "openpipe-art-wandb-weave",
            "model_profile": config.model_profile,
            "base_model": config.base_model,
        }
    )
    return model


def make_prompted_model(config: RetailCourseConfig, *, name: str | None = None) -> Any:
    art = require_art()
    api_key = config.inference_api_key or os.getenv("OPENAI_API_KEY")
    base_url = config.inference_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    inference_name = config.inference_model_name or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    display_name = name or inference_name
    return art.Model(
        name=display_name,
        project=config.project,
        entity=config.entity,
        inference_api_key=api_key,
        inference_base_url=base_url,
        inference_model_name=inference_name,
        base_path=config.art_path,
    )
