from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import importlib.metadata
import importlib.util
import os
import platform
import sys

from course.shared.config import (
    COURSE_DOTENV_PATH,
    COURSE_RUN_CONFIG_PATH,
    DOTENV_LOADED,
    available_model_profiles,
    config_from_env,
    mask_secret,
    run_config,
)
from course.shared.art_compat import model_internal_config, vllm_engine_args

PACKAGES = ["openpipe-art", "wandb", "weave", "datasets", "torch", "transformers", "openai", "python-dotenv"]
RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "09_runbooks" / "run_retail_agentic_sequence.py"


def version_for(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "<not installed>"


def effective_runbook_args():
    spec = importlib.util.spec_from_file_location("retail_agentic_runbook", RUNBOOK_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_args(["--dry-run"])


def main() -> None:
    cfg = config_from_env()
    run_cfg = run_config()
    runbook_args = effective_runbook_args()
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Project:", cfg.project)
    print("Entity:", cfg.entity or "<unset>")
    print("ART path:", cfg.art_path)
    print("Dataset:", cfg.dataset_id)
    print(".env path:", COURSE_DOTENV_PATH)
    print(".env present:", COURSE_DOTENV_PATH.exists())
    print(".env loaded:", DOTENV_LOADED)
    print("Run config path:", COURSE_RUN_CONFIG_PATH)
    print("Run profile:", run_cfg.get("run_profile", "<unset>"))
    print("Model profile:", cfg.model_profile)
    print("Base model:", runbook_args.base_model if runbook_args is not None else cfg.base_model)
    print("GPU memory preset:", run_cfg.get("gpu_memory_preset", "<unset>"))
    print(
        "Rollout max completion tokens:",
        runbook_args.max_completion_tokens if runbook_args is not None else cfg.rollout_max_completion_tokens,
    )
    print("Tool call parser:", cfg.tool_call_parser or "<ART/vLLM default>")
    if runbook_args is not None:
        runbook_vllm_args = {
            "max_model_len": runbook_args.vllm_max_model_len,
            "gpu_memory_utilization": runbook_args.vllm_gpu_memory_utilization,
            "max_num_batched_tokens": runbook_args.vllm_max_num_batched_tokens,
            "max_num_seqs": runbook_args.vllm_max_num_seqs,
        }
        print("ART max sequence length:", runbook_args.art_seq_length)
        print("vLLM engine args:", runbook_vllm_args)
    else:
        print("ART max sequence length:", cfg.art_max_seq_length or "<ART default>")
        print("vLLM engine args:", vllm_engine_args(cfg) or "<ART/vLLM defaults>")
        print("ART internal config:", model_internal_config(cfg) or "<ART defaults>")
    print("Inference model:", cfg.inference_model_name or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini")
    print("Available profiles:", ", ".join(sorted(available_model_profiles())))
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
