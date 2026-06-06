from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSE_DOTENV_PATH = PROJECT_ROOT / ".env"


def load_course_dotenv() -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(COURSE_DOTENV_PATH, override=False))


DOTENV_LOADED = load_course_dotenv()

DEFAULT_DATASET_ID = "lefft/tau-dev-task-retail-v1"
DEFAULT_PROJECT = os.getenv("WANDB_PROJECT", "openpipe-art-retail")
DEFAULT_ART_PATH = os.getenv("ART_PATH", str(PROJECT_ROOT / ".art"))
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "retail"
DEFAULT_MODEL_PROFILE = "standard"

MODEL_PROFILES: dict[str, str] = {
    "tiny": "LiquidAI/LFM2.5-1.2B-Thinking",
    "standard": "OpenPipe/Qwen3-14B-Instruct",
    "serverless": "OpenPipe/Qwen3-14B-Instruct",
    "moe": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

MODEL_PROFILE_NOTES: dict[str, str] = {
    "tiny": "1.2B local smoke-test profile for constrained GPUs; validate ART trainer compatibility before relying on it for RL.",
    "standard": "Default local-H100/course profile used by the main hands-on path.",
    "serverless": "W&B Serverless RL compatible 14B profile for managed training demos.",
    "moe": "Larger Serverless/Megatron-oriented profile for scaling discussion and advanced labs.",
}


def model_profile_from_env() -> str:
    if os.getenv("ART_BASE_MODEL") and "ART_MODEL_PROFILE" not in os.environ:
        return "custom"
    return os.getenv("ART_MODEL_PROFILE", DEFAULT_MODEL_PROFILE)


def resolve_base_model() -> str:
    explicit = os.getenv("ART_BASE_MODEL")
    if explicit:
        return explicit
    profile = model_profile_from_env()
    try:
        return MODEL_PROFILES[profile]
    except KeyError as exc:
        allowed = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(
            f"Unknown ART_MODEL_PROFILE={profile!r}. Use one of: {allowed}; "
            "or set ART_BASE_MODEL directly."
        ) from exc


def available_model_profiles() -> dict[str, str]:
    return dict(MODEL_PROFILES)


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
    entity: str | None = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))
    project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "openpipe-art-retail"))
    art_path: str = field(default_factory=lambda: os.getenv("ART_PATH", str(PROJECT_ROOT / ".art")))
    dataset_id: str = field(default_factory=lambda: os.getenv("RETAIL_DATASET_ID", DEFAULT_DATASET_ID))
    data_dir: Path = DEFAULT_DATA_DIR
    model_profile: str = field(default_factory=model_profile_from_env)
    base_model: str = field(default_factory=resolve_base_model)
    model_name: str = field(default_factory=lambda: os.getenv("ART_MODEL_NAME", "retail-support-agent"))
    inference_model_name: str | None = field(default_factory=lambda: os.getenv("INFERENCE_MODEL_NAME"))
    inference_base_url: str | None = field(default_factory=lambda: os.getenv("INFERENCE_BASE_URL"))
    inference_api_key: str | None = field(default_factory=lambda: os.getenv("INFERENCE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    request_logprobs: bool = field(default_factory=lambda: bool_env("ART_REQUEST_LOGPROBS", True))


def config_from_env() -> RetailCourseConfig:
    return RetailCourseConfig()


def ensure_data_dir(path: Path | None = None) -> Path:
    data_dir = path or DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
