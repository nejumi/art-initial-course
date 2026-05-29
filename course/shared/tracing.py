from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

from .config import DEFAULT_PROJECT

F = TypeVar("F", bound=Callable[..., Any])


def init_weave(project: str | None = None) -> Any | None:
    try:
        import weave
    except ImportError:
        return None
    try:
        from art.utils.strip_logprobs import strip_logprobs
    except Exception:
        strip_logprobs = None
    kwargs: dict[str, Any] = {}
    if strip_logprobs is not None:
        kwargs["global_postprocess_output"] = strip_logprobs
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
    return value
