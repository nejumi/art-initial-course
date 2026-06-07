from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSE_DOTENV_PATH = PROJECT_ROOT / ".env"
COURSE_RUN_CONFIG_PATH = PROJECT_ROOT / "course" / "09_runbooks" / "config.yaml"
COURSE_BASE_CONFIG_PATH = PROJECT_ROOT / "course" / "09_runbooks" / "base_config.yaml"
IGNORE_RUN_CONFIG_ENV = "COURSE_IGNORE_RUN_CONFIG"


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
DEFAULT_ART_MAX_SEQ_LENGTH = 8192
DEFAULT_VLLM_MAX_MODEL_LEN = 16384
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.70
DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS = 16384
DEFAULT_VLLM_MAX_NUM_SEQS = 8
DEFAULT_RETAIL_ALLOW_REFERENCE_STATE_ACTION_JUMPS = True
DEFAULT_RETAIL_TOOL_USE_INSTRUCTION = (
    "When a tool is needed, emit exactly one tool call and no assistant text. "
    "Do not write <think> tags or hidden reasoning in assistant messages. "
    "After all required tool work is complete, answer the customer briefly."
)

MODEL_PROFILES: dict[str, str] = {
    "tiny": "Qwen/Qwen3-0.6B",
    "standard": "LiquidAI/LFM2.5-8B-A1B",
    "openpipe": "OpenPipe/Qwen3-14B-Instruct",
    "serverless": "OpenPipe/Qwen3-14B-Instruct",
    "moe": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

MODEL_PROFILE_NOTES: dict[str, str] = {
    "tiny": "ART-supported 0.6B local profile for constrained GPUs and fast course validation.",
    "standard": "Validated local-H100/course profile used by the main hands-on path.",
    "openpipe": "OpenPipe-hosted Qwen3 14B profile for compatibility comparison and managed-training demos.",
    "serverless": "W&B Serverless RL compatible 14B profile for managed training demos.",
    "moe": "Larger Serverless/Megatron-oriented profile for scaling discussion and advanced labs.",
}


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def run_config() -> dict[str, Any]:
    if env_truthy(IGNORE_RUN_CONFIG_ENV):
        return {}
    return load_yaml_mapping(COURSE_RUN_CONFIG_PATH)


@lru_cache(maxsize=1)
def base_config() -> dict[str, Any]:
    if env_truthy(IGNORE_RUN_CONFIG_ENV):
        return {}
    return load_yaml_mapping(COURSE_BASE_CONFIG_PATH)


def nonempty_config_value(key: str) -> Any | None:
    value = run_config().get(key)
    if value is None or value == "":
        return None
    return value


def run_config_overrides() -> dict[str, Any]:
    overrides = run_config().get("overrides") or {}
    return overrides if isinstance(overrides, dict) else {}


def gpu_memory_preset_values() -> dict[str, Any]:
    preset_name = nonempty_config_value("gpu_memory_preset")
    if not preset_name or preset_name == "standard":
        return {}
    presets = base_config().get("gpu_memory_presets") or {}
    if not isinstance(presets, dict):
        return {}
    preset = presets.get(str(preset_name)) or {}
    return preset if isinstance(preset, dict) else {}


def run_config_runtime_values() -> dict[str, Any]:
    values = dict(gpu_memory_preset_values())
    values.update(run_config_overrides())
    return values


def runtime_value(key: str) -> Any | None:
    value = run_config_runtime_values().get(key)
    if value is None or value == "":
        return None
    return value


def optional_int_config(key: str) -> int | None:
    value = runtime_value(key)
    return int(value) if value is not None else None


def optional_float_config(key: str) -> float | None:
    value = runtime_value(key)
    return float(value) if value is not None else None


def configured_model_profile() -> str | None:
    profile = nonempty_config_value("model_profile")
    if profile is None or profile == "from_env":
        return None
    return str(profile)


def configured_base_model() -> str | None:
    base_model = nonempty_config_value("base_model")
    if base_model is not None:
        return str(base_model)
    profile = configured_model_profile()
    if profile is None:
        return None
    if profile == "custom":
        raise ValueError("course/09_runbooks/config.yaml uses model_profile: custom, so base_model must be set.")
    try:
        return MODEL_PROFILES[profile]
    except KeyError as exc:
        allowed = ", ".join(["custom", *sorted(MODEL_PROFILES)])
        raise ValueError(f"Unknown model_profile={profile!r} in course/09_runbooks/config.yaml. Use one of: {allowed}.") from exc


def model_profile_from_env() -> str:
    configured = configured_model_profile()
    if configured is not None:
        return configured
    if os.getenv("ART_BASE_MODEL") and "ART_MODEL_PROFILE" not in os.environ:
        return "custom"
    return os.getenv("ART_MODEL_PROFILE", DEFAULT_MODEL_PROFILE)


def resolve_base_model() -> str:
    configured = configured_base_model()
    if configured is not None:
        return configured
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


def optional_bool_env(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def terminate_on_invalid_from_env() -> bool:
    explicit = optional_bool_env("RETAIL_TERMINATE_ON_INVALID")
    if explicit is not None:
        return explicit
    legacy = optional_bool_env("RETAIL_TERMINATE_ON_INVALID_TOOL")
    if legacy is not None:
        return legacy
    return True


def retail_tool_use_instruction_from_env() -> str:
    return os.getenv("RETAIL_TOOL_USE_INSTRUCTION", DEFAULT_RETAIL_TOOL_USE_INSTRUCTION).strip()


def strict_tool_prompt_from_env() -> bool:
    return bool_env("RETAIL_STRICT_TOOL_PROMPT", True)


def default_tool_call_parser(base_model: str | None = None) -> str | None:
    explicit = os.getenv("ART_TOOL_CALL_PARSER")
    if explicit is not None:
        value = explicit.strip()
        return value or None
    model_name = base_model or resolve_base_model()
    model_name_lower = model_name.lower()
    if "liquidai/lfm" in model_name_lower or "lfm2" in model_name_lower:
        return "lfm2"
    if "qwen3.5" in model_name_lower or "qwen3_5" in model_name_lower:
        return "qwen3_xml"
    if "qwen3" in model_name_lower and "vl" not in model_name_lower:
        return "qwen3_xml"
    if "gemma-4" in model_name_lower or "gemma4" in model_name_lower:
        return "gemma4"
    return None


def mask_secret(value: str | None) -> str:
    if not value:
        return "<unset>"
    return f"<set:{len(value)} chars>"


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
    rollout_max_completion_tokens: int = field(
        default_factory=lambda: optional_int_config("max_completion_tokens")
        or optional_int_env("ART_ROLLOUT_MAX_COMPLETION_TOKENS")
        or 512
    )
    tool_call_parser: str | None = field(default_factory=default_tool_call_parser)
    art_max_seq_length: int | None = field(
        default_factory=lambda: optional_int_config("art_seq_length")
        or optional_int_env("ART_MAX_SEQ_LENGTH")
        or DEFAULT_ART_MAX_SEQ_LENGTH
    )
    vllm_max_model_len: int | None = field(
        default_factory=lambda: optional_int_config("vllm_max_model_len")
        or optional_int_env("ART_VLLM_MAX_MODEL_LEN")
        or DEFAULT_VLLM_MAX_MODEL_LEN
    )
    vllm_gpu_memory_utilization: float | None = field(
        default_factory=lambda: optional_float_config("vllm_gpu_memory_utilization")
        or optional_float_env("ART_VLLM_GPU_MEMORY_UTILIZATION")
        or DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    )
    vllm_max_num_batched_tokens: int | None = field(
        default_factory=lambda: optional_int_config("vllm_max_num_batched_tokens")
        or optional_int_env("ART_VLLM_MAX_NUM_BATCHED_TOKENS")
        or DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS
    )
    vllm_max_num_seqs: int | None = field(
        default_factory=lambda: optional_int_config("vllm_max_num_seqs")
        or optional_int_env("ART_VLLM_MAX_NUM_SEQS")
        or DEFAULT_VLLM_MAX_NUM_SEQS
    )
    vllm_enforce_eager: bool | None = field(default_factory=lambda: optional_bool_env("ART_VLLM_ENFORCE_EAGER"))
    terminate_on_invalid: bool = field(default_factory=terminate_on_invalid_from_env)
    allow_reference_state_action_jumps: bool = field(
        default_factory=lambda: bool_env(
            "RETAIL_ALLOW_REFERENCE_STATE_ACTION_JUMPS",
            DEFAULT_RETAIL_ALLOW_REFERENCE_STATE_ACTION_JUMPS,
        )
    )


def config_from_env() -> RetailCourseConfig:
    return RetailCourseConfig()


def ensure_data_dir(path: Path | None = None) -> Path:
    data_dir = path or DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
