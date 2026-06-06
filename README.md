# OpenPipe ART x W&B x Weave Retail Hands-on

This repository contains the hands-on code for a course on OpenPipe ART with W&B Models and Weave.

Japanese overview: [README.ja.md](README.ja.md)

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
export ART_MODEL_PROFILE=standard
export OPENAI_API_KEY=<optional-for-external-baseline>

python course/07_models_registry_weave/log_data_artifacts.py --data-dir data/retail

python course/03_sft_warmup/train_sft_local.py \
  --data-artifact retail-course-data:latest \
  --batch-size 1 \
  --peak-lr 3e-5 \
  --max-steps 48 \
  --chunk-size-batches 12

python course/04_grpo_local/train_grpo_local.py \
  --data-artifact retail-course-data:latest \
  --parent-artifact retail-support-agent-checkpoint:sft-anchor \
  --steps 8 \
  --groups-per-step 4 \
  --rollouts-per-scenario 4 \
  --learning-rate 5e-6
```

SFT is intentionally chunked: one ART SFT call produces one ART checkpoint/log point, so `--chunk-size-batches` keeps the SFT loss curve visible in W&B instead of collapsing a full epoch into a single point.

Training scripts use W&B Artifacts for lineage:

- `retail-course-data:latest` contains the TAU Retail JSONL splits plus generated `sft_train.jsonl`.
- SFT logs the latest local ART LoRA checkpoint as `<ART_MODEL_NAME>-checkpoint:sft-anchor`.
- GRPO, GSPO, and RULER runs call `use_artifact` on the SFT checkpoint and log their own branch checkpoint artifacts.
- Eval runs can call `use_artifact` on both the dataset and the evaluated checkpoint, while Weave stores rollout traces.

## Model Selection

The trainable ART model is selected in `course/shared/config.py`.
Use `ART_MODEL_PROFILE` for common classroom hardware profiles, or `ART_BASE_MODEL` for a direct override.

```bash
# Constrained local GPU / compatibility smoke tests.
export ART_MODEL_PROFILE=tiny       # Qwen/Qwen3-0.6B

# Main H100 hands-on path.
export ART_MODEL_PROFILE=standard   # LiquidAI/LFM2.5-8B-A1B

# OpenPipe/Qwen compatibility path.
export ART_MODEL_PROFILE=openpipe   # OpenPipe/Qwen3-14B-Instruct

# Larger managed-training discussion/demo.
export ART_MODEL_PROFILE=moe        # Qwen/Qwen3-30B-A3B-Instruct-2507

# Explicit override always wins.
export ART_BASE_MODEL=LiquidAI/LFM2.5-1.2B-Instruct
```

`standard` is the validated H100 classroom path. `tiny` is intended to make setup and short smoke tests accessible on smaller machines while staying on an ART-supported Qwen3 dense model path. Use `ART_BASE_MODEL` for direct experiments, then run the setup check and a short SFT/RL smoke run before relying on the model for the full RL lab because ART backend compatibility depends on the architecture, tokenizer, vLLM, tool-call parser, and LoRA trainer path.

When `ART_BASE_MODEL` points at an LFM2/LFM2.5 model, the course selects `ART_TOOL_CALL_PARSER=lfm2` automatically so vLLM can return model-generated tool calls as OpenAI-compatible `message.tool_calls`. Override `ART_TOOL_CALL_PARSER` only when validating another parser.

RULER judge selection is separate from the trainable model. It defaults to `openai/gpt-5.5` with medium reasoning effort in `course/05_ruler/train_with_ruler.py`.

## Validated H100 Results

The current reference run uses `LiquidAI/LFM2.5-8B-A1B`, `ART_MAX_SEQ_LENGTH=8192`, `ART_VLLM_MAX_MODEL_LEN=16384`, and 48 held-out retail eval scenarios. These are empirical workshop baselines, not a claim that every metric improves monotonically.

| Stage | Reward | Task success | Tool F1 | Tool order | Invalid calls avg | Final text F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5102 | 0.1458 | 0.7268 | 0.5542 | 0.6667 | 0.2439 |
| SFT anchor, lr=3e-5 | 0.5012 | 0.2083 | 0.7255 | 0.5750 | 0.7708 | 0.2541 |
| SFT -> GRPO, 8 steps | 0.5165 | 0.1875 | 0.7283 | 0.5740 | 0.6667 | 0.2576 |
| SFT -> GSPO, 3 steps | 0.5043 | 0.2083 | 0.7268 | 0.5792 | 0.7292 | 0.2405 |
| SFT -> RULER-GRPO, 3 steps | 0.5026 | 0.1875 | 0.7254 | 0.5750 | 0.7292 | 0.2405 |

The useful teaching arc is:

- Baseline LFM2.5 is already capable, which makes the lab realistic rather than toy-like.
- SFT improves exact task success, tool order, and final response overlap, but slightly lowers the scalar reward because invalid tool calls increase on this small eval slice.
- GRPO recovers and exceeds baseline scalar reward while keeping task success above baseline.
- GSPO is useful for contrasting sequence-level importance weighting; in the short run it preserves the SFT task-success gain and gives the best tool-order score.
- RULER shows how to blend a model judge with deterministic retail reward signals.

W&B Artifacts are logged for the dataset, the SFT checkpoint, and each RL branch checkpoint. The comparison script logs a horizontal W&B table with columns such as `model`, `stage`, `model_artifact_path`, `reward`, `task_success`, and delta metrics, while Weave stores rollout and eval traces.

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
