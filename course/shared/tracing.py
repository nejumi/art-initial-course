from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from .config import DEFAULT_PROJECT, PROJECT_ROOT

F = TypeVar("F", bound=Callable[..., Any])


def default_weave_cache_dir() -> Path:
    suffix = os.getenv("SLURM_JOB_ID") or f"pid-{os.getpid()}"
    return PROJECT_ROOT / ".art" / "weave_cache" / suffix


def init_weave(project: str | None = None) -> Any | None:
    try:
        import weave
    except ImportError:
        return None
    cache_dir = Path(os.getenv("WEAVE_SERVER_CACHE_DIR") or default_weave_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WEAVE_SERVER_CACHE_DIR", str(cache_dir))
    kwargs: dict[str, Any] = {
        "global_postprocess_output": compact_for_trace,
        "settings": {"server_cache_dir": str(cache_dir)},
    }
    return weave.init(project or os.getenv("WEAVE_PROJECT") or DEFAULT_PROJECT, **kwargs)


def weave_op(name: str | None = None) -> Callable[[F], F]:
    try:
        import weave
    except ImportError:
        def identity(fn: F) -> F:
            return fn
        return identity
    return weave.op(name=name) if name else weave.op()


def compact_for_trace(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: compact_for_trace(v) for k, v in value.items() if k != "logprobs"}
    if isinstance(value, list):
        return [compact_for_trace(v) for v in value]
    if isinstance(value, tuple):
        return tuple(compact_for_trace(v) for v in value)
    if is_dataclass(value):
        return compact_for_trace(asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return compact_for_trace(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return compact_for_trace(value.to_dict())
        except Exception:
            pass
    return value
