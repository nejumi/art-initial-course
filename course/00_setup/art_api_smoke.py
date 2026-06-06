from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import importlib
import inspect
from typing import Any


CHECKS: list[tuple[str, str, list[str]]] = [
    ("art", "TrainableModel", ["name", "project", "base_model"]),
    ("art", "Model", ["name", "project", "inference_model_name", "report_metrics"]),
    ("art", "gather_trajectory_groups", ["max_exceptions"]),
    ("art.local", "LocalBackend", []),
    ("art.rewards", "ruler_score_group", ["judge_model", "rubric", "extra_litellm_params", "swallow_exceptions"]),
    ("art.utils.sft", "train_sft_from_file", ["model", "file_path", "epochs", "batch_size", "peak_lr"]),
]


METHOD_CHECKS: list[tuple[str, str, str]] = [
    ("art", "Model", "update_wandb_config"),
    ("art", "Model", "read_state"),
    ("art", "Model", "merge_state"),
    ("art", "Model", "overwrite_state"),
]


def signature_params(obj: Any) -> set[str]:
    try:
        return set(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return set()


def check_callable(module_name: str, attr_name: str, required: list[str]) -> list[str]:
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    target = obj.__init__ if inspect.isclass(obj) else obj
    params = signature_params(target)
    missing = [name for name in required if name not in params]
    if missing:
        return [f"{module_name}.{attr_name} missing params: {', '.join(missing)}"]
    print(f"OK {module_name}.{attr_name}: {', '.join(sorted(params)) or '<signature unavailable>'}")
    return []


def check_method(module_name: str, class_name: str, method_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not hasattr(cls, method_name):
        return [f"{module_name}.{class_name}.{method_name} is missing"]
    print(f"OK {module_name}.{class_name}.{method_name}")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the ART APIs used by the course against the installed package.")
    parser.add_argument("--require-art", action="store_true", help="Exit non-zero if openpipe-art/art is not installed.")
    args = parser.parse_args()

    try:
        importlib.import_module("art")
    except ImportError:
        message = 'ART is not installed. Install with: python -m pip install "openpipe-art[backend]"'
        print(message)
        if args.require_art:
            raise SystemExit(1)
        return

    failures: list[str] = []
    for module_name, attr_name, required in CHECKS:
        try:
            failures.extend(check_callable(module_name, attr_name, required))
        except Exception as exc:
            failures.append(f"{module_name}.{attr_name} check failed: {exc}")

    for module_name, class_name, method_name in METHOD_CHECKS:
        try:
            failures.extend(check_method(module_name, class_name, method_name))
        except Exception as exc:
            failures.append(f"{module_name}.{class_name}.{method_name} check failed: {exc}")

    if failures:
        print()
        print("ART API smoke test failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print()
    print("ART API smoke test passed.")


if __name__ == "__main__":
    main()
