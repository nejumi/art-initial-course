from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from .config import DEFAULT_PROJECT, PROJECT_ROOT

F = TypeVar("F", bound=Callable[..., Any])

_WEAVE_CLIENT: Any | None = None
_WEAVE_WANDB_RUN_ID: str | None = None


def _verbose_weave_context() -> bool:
    return os.getenv("COURSE_VERBOSE_WEAVE_CONTEXT", "").lower() in {"1", "true", "yes"}


def default_weave_cache_dir() -> Path:
    suffix = os.getenv("SLURM_JOB_ID") or f"pid-{os.getpid()}"
    return PROJECT_ROOT / ".art" / "weave_cache" / suffix


def init_weave(project: str | None = None) -> Any | None:
    global _WEAVE_CLIENT
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
    _WEAVE_CLIENT = weave.init(project or os.getenv("WEAVE_PROJECT") or DEFAULT_PROJECT, **kwargs)
    _set_current_weave_client(_WEAVE_CLIENT)
    bind_weave_to_active_wandb_run(_WEAVE_CLIENT)
    return _WEAVE_CLIENT


def _current_weave_client_from_context() -> Any | None:
    try:
        from weave.trace.context import weave_client_context
    except Exception:
        return None
    try:
        return weave_client_context.get_weave_client()
    except Exception:
        return None


def _set_current_weave_client(client: Any) -> None:
    try:
        from weave.trace.context import weave_client_context
    except Exception:
        return
    try:
        weave_client_context.set_weave_client_global(None)
        weave_client_context.set_weave_client_global(client)
    except Exception:
        return


def _candidate_weave_clients(client: Any | None = None) -> list[Any]:
    clients = [client, _WEAVE_CLIENT, _current_weave_client_from_context()]
    seen: set[int] = set()
    result = []
    for item in clients:
        if item is None or not hasattr(item, "set_wandb_run_context"):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def bind_weave_to_active_wandb_run(client: Any | None = None, *, step: int | None = None) -> bool:
    """Attach subsequent Weave traces to the active W&B run when both exist."""
    global _WEAVE_WANDB_RUN_ID
    try:
        import wandb
    except ImportError:
        return False
    run = getattr(wandb, "run", None)
    run_id = getattr(run, "id", None)
    if not run_id:
        return False
    if client is not None:
        _set_current_weave_client(client)
    elif _WEAVE_CLIENT is not None:
        _set_current_weave_client(_WEAVE_CLIENT)
    clients = _candidate_weave_clients(client)
    if not clients:
        return False
    bound = False
    for item in clients:
        try:
            item.set_wandb_run_context(run_id=run_id, step=step)
            bound = True
        except Exception as exc:
            if _verbose_weave_context():
                print(f"Weave/W&B run context binding skipped: {exc}")
    _WEAVE_WANDB_RUN_ID = str(run_id)
    if bound and _verbose_weave_context():
        print(f"Weave traces linked to W&B run: {run_id}")
    return bound


def clear_weave_wandb_run_context(run_id: str | None = None) -> None:
    """Clear an explicit Weave-to-W&B run override when an owned run finishes."""
    global _WEAVE_WANDB_RUN_ID
    if run_id is not None and _WEAVE_WANDB_RUN_ID not in {None, str(run_id)}:
        return
    for client in _candidate_weave_clients():
        if not hasattr(client, "clear_wandb_run_context"):
            continue
        try:
            client.clear_wandb_run_context()
        except Exception:
            continue
    _WEAVE_WANDB_RUN_ID = None


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
