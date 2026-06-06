# OpenPipe ART x W&B x Weave Retail Hands-on

This repository contains the hands-on code for a course on OpenPipe ART with W&B Models and Weave.

The main task is a Retail Customer Support Agent based on open retail tool-calling data:

- `lefft/tau-dev-task-retail-v1` for SFT and workflow validation
- tau-bench / tau2-bench retail concepts for rollout, reward, and eval design

The code is structured so the early labs can be inspected without a GPU. Local ART training labs require a CUDA-capable GPU and `openpipe-art[backend]`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your W&B/OpenAI keys and model profile.

python course/00_setup/env_check.py
python course/00_setup/art_api_smoke.py
python course/03_sft_warmup/download_tau_retail.py --sample-if-download-fails
python course/03_sft_warmup/make_sft_jsonl.py --limit 200
python course/01_art_primitives/dry_run_rollout.py --limit 3
```

To run ART training, configure W&B and a local GPU environment:

```bash
export WANDB_ENTITY=<entity>
export WANDB_PROJECT=openpipe-art-retail
export OPENAI_API_KEY=<optional-for-external-baseline>

python course/03_sft_warmup/train_sft_local.py --max-steps 20
python course/04_grpo_local/train_grpo_local.py --steps 3 --groups-per-step 4 --rollouts-per-scenario 4
```


## Model Selection

The trainable ART model is selected in `course/shared/config.py`.
Use `ART_MODEL_PROFILE` for common classroom hardware profiles, or `ART_BASE_MODEL` for a direct override.

```bash
# Constrained local GPU / compatibility smoke tests.
export ART_MODEL_PROFILE=tiny       # LiquidAI/LFM2.5-1.2B-Thinking

# Main hands-on path.
export ART_MODEL_PROFILE=standard   # OpenPipe/Qwen3-14B-Instruct

# Larger managed-training discussion/demo.
export ART_MODEL_PROFILE=moe        # Qwen/Qwen3-30B-A3B-Instruct-2507

# Explicit override always wins.
export ART_BASE_MODEL=LiquidAI/LFM2.5-1.2B-Thinking
```

`tiny` is intended to make setup and short smoke tests accessible on smaller machines. Before using it for the full RL lab, run the setup check and a short SFT/RL smoke run because ART backend compatibility depends on the model architecture, tokenizer, vLLM, and LoRA trainer path.

RULER judge selection is separate from the trainable model. It defaults to `openai/gpt-5.5` with medium reasoning effort in `course/05_ruler/train_with_ruler.py`.

## Lab Map

- `course/00_setup`: environment and W&B/Weave smoke tests
- `course/01_art_primitives`: ART trajectory and reward primitives
- `course/02_weave_evals`: Weave dataset, scorers, and baseline evals
- `course/03_sft_warmup`: retail data download and SFT
- `course/04_grpo_local`: local ART grouped RL
- `course/05_ruler`: RULER hybrid reward
- `course/06_gspo_and_configs`: config matrix and GSPO demo
- `course/07_models_registry_weave`: artifact, registry, and Weave model wrappers
- `course/08_enterprise_ops`: checkpoint, backend scaling, and enterprise operations examples

The course blueprint lives in `OPENPIPE_ART_WANDB_COURSE_BLUEPRINT.md`.

Scaling appendix: `course/08_enterprise_ops/backend_scaling.md`.
