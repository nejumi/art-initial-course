from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import importlib.metadata
import os
import platform
import sys

from course.shared.config import config_from_env, mask_secret

PACKAGES = ["art", "openpipe-art", "wandb", "weave", "datasets", "torch", "transformers", "openai"]


def version_for(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "<not installed>"


def main() -> None:
    cfg = config_from_env()
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Project:", cfg.project)
    print("Entity:", cfg.entity or "<unset>")
    print("ART path:", cfg.art_path)
    print("Dataset:", cfg.dataset_id)
    print("WANDB_API_KEY:", mask_secret(os.getenv("WANDB_API_KEY")))
    print("OPENAI_API_KEY:", mask_secret(os.getenv("OPENAI_API_KEY")))
    print("\nPackages:")
    for package in PACKAGES:
        print(f"  {package}: {version_for(package)}")
    try:
        import torch
        print("\nCUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA devices:", torch.cuda.device_count())
            for idx in range(torch.cuda.device_count()):
                print(f"  [{idx}] {torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print("\nTorch CUDA check failed:", exc)


if __name__ == "__main__":
    main()
