from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def append_step_metrics(
    path: str | Path | None,
    *,
    step: int | None,
    algorithm: str,
    metrics: dict[str, Any],
    skipped: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step,
        "algorithm": algorithm,
        "skipped": skipped,
        "metrics": json_safe(metrics),
    }
    if extra:
        record.update(json_safe(extra))
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
