from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ID = "lefft/tau-dev-task-retail-v1"
DEFAULT_PROJECT = os.getenv("WANDB_PROJECT", "openpipe-art-retail")
DEFAULT_ART_PATH = os.getenv("ART_PATH", str(PROJECT_ROOT / ".art"))
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "retail"


def bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def mask_secret(value: str | None) -> str:
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


@dataclass(frozen=True)
class RetailCourseConfig:
    entity: str | None = os.getenv("WANDB_ENTITY")
    project: str = DEFAULT_PROJECT
    art_path: str = DEFAULT_ART_PATH
    dataset_id: str = os.getenv("RETAIL_DATASET_ID", DEFAULT_DATASET_ID)
    data_dir: Path = DEFAULT_DATA_DIR
    base_model: str = os.getenv("ART_BASE_MODEL", "OpenPipe/Qwen3-14B-Instruct")
    model_name: str = os.getenv("ART_MODEL_NAME", "retail-support-agent")
    inference_model_name: str | None = os.getenv("INFERENCE_MODEL_NAME")
    inference_base_url: str | None = os.getenv("INFERENCE_BASE_URL")
    inference_api_key: str | None = os.getenv("INFERENCE_API_KEY") or os.getenv("OPENAI_API_KEY")
    request_logprobs: bool = bool_env("ART_REQUEST_LOGPROBS", True)


def config_from_env() -> RetailCourseConfig:
    return RetailCourseConfig()


def ensure_data_dir(path: Path | None = None) -> Path:
    data_dir = path or DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
