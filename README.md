# OpenPipe ART x W&B x Weave Retail Hands-on

This repository contains the hands-on code for a course on OpenPipe ART with W&B Models and Weave.

Japanese overview: [README.ja.md](README.ja.md)

The main task is a Retail Customer Support Agent based on open retail tool-calling data:

- `lefft/tau-dev-task-retail-v1` for SFT and workflow validation
- tau-bench / tau2-bench retail concepts for rollout, reward, and eval design
- optional tau2-style next-action SFT data from `amityco/...` and `inclusionAI/AReaL-tau2-data`

The code is structured so the early labs can be inspected without a GPU. Local ART training labs require a CUDA-capable GPU and `openpipe-art[backend]`.

## Open Training Data

The course intentionally uses public datasets so a workshop can run end to end without private task data.

| Dataset | Main use | Why it is useful | Course handling |
| --- | --- | --- | --- |
| [`lefft/tau-dev-task-retail-v1`](https://huggingface.co/datasets/lefft/tau-dev-task-retail-v1) | Default SFT and evaluation data | Successful retail trajectories already serialized in OpenAI tool-call format | Downloaded by `download_tau_retail.py`; converted to `sft_*.jsonl` |
| [`amityco/tau-bench-retail-train-next-action-all-step-score-v0.2`](https://huggingface.co/datasets/amityco/tau-bench-retail-train-next-action-all-step-score-v0.2) | Optional teacher next-action SFT | Step-level retail tool-call rows with candidate scores | Converted by `make_teacher_next_action_sft_jsonl.py`; can be mixed with bridge next-action rows |
| [`inclusionAI/AReaL-tau2-data`](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data) | Advanced next-action SFT option | Mirrors recent tau2-style SFT + verifiable-reward RL workflows | Converted by `make_areal_retail_sft_jsonl.py`; strips thinking fields and normalizes tool calls |

Full-dialog SFT is easy to explain, but it can over-supervise long dialogue style and final responses. Next-action SFT is the stronger path when we want to align with modern agent training recipes and avoid training on every prior assistant action repeatedly.

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
python course/03_sft_warmup/make_next_action_sft_jsonl.py --data-dir data/retail
python course/01_art_primitives/dry_run_rollout.py --limit 3
```

Bridge curriculum option for a shorter first RL lab:

```bash
python course/03_sft_warmup/make_bridge_curriculum.py \
  --source-dir data/retail \
  --output-dir data/retail_bridge_state1 \
  --max-state-actions 1 \
  --max-tool-calls 6 \
  --max-turns 28
python course/07_models_registry_weave/log_data_artifacts.py \
  --data-dir data/retail_bridge_state1 \
  --artifact-name retail-course-data-bridge-state1
```

Instructor runbook for a full local-H100 validation pass:

```bash
python course/09_runbooks/run_retail_agentic_sequence.py \
  --run-slug lfm25-8b-a1b-bridge-state1 \
  --base-model LiquidAI/LFM2.5-8B-A1B \
  --data-dir data/retail_bridge_state1 \
  --source-dir data/retail \
  --data-artifact-name retail-course-data-bridge-state1 \
  --build-bridge \
  --sft-max-steps 96 \
  --rl-steps 32 \
  --rl-algos grpo,gspo
```

To strengthen the SFT warm start with public teacher next-action rows:

```bash
python course/09_runbooks/run_retail_agentic_sequence.py \
  --run-slug lfm25-8b-a1b-bridge-state1-teacher \
  --base-model LiquidAI/LFM2.5-8B-A1B \
  --data-dir data/retail_bridge_state1 \
  --source-dir data/retail \
  --data-artifact-name retail-course-data-bridge-state1-teacher \
  --build-bridge \
  --include-teacher-sft \
  --teacher-sft-limit 512 \
  --sft-max-steps 144 \
  --rl-steps 32 \
  --rl-algos grpo,gspo
```

The runbook defaults to `--continue-on-invalid` for tau-style RL. Unexpected state-changing actions are still penalized, but rollouts continue long enough for final state/action and communication rewards to be observed. Use `--no-continue-on-invalid` only when demonstrating strict replay failure modes.
For the final course report, add stochastic validation with `--eval-rollouts-per-scenario 4 --eval-temperature 0.2` so `outcome_pass_at_k`, `task_pass_at_k`, and reward variance columns are meaningful. Keep the default deterministic eval for quick instructor checks.

On a Slurm H100 cluster, the same flow can be launched with:

```bash
sbatch course/09_runbooks/sunk_h100_retail_agentic_sequence.sbatch
```

Advanced SFT data option:

```bash
python course/03_sft_warmup/make_teacher_next_action_sft_jsonl.py \
  --tools-data-dir data/retail_bridge_state1 \
  --output data/retail_bridge_state1/sft_teacher_retail_next_action.jsonl \
  --limit 512

python course/03_sft_warmup/make_areal_retail_sft_jsonl.py \
  --tools-data-dir data/retail \
  --output data/retail/sft_areal_retail_next_action.jsonl \
  --limit 200
```

The default course path uses compact TAU retail trajectories. The bridge curriculum keeps short successful trajectories with exactly one state-changing action, which is useful for proving that agentic RL can improve a verifiable outcome before moving to the broader retail curriculum. The teacher and AReaL converters are provided for stronger warm starts and advanced experiments that want to align SFT data construction with recent tau2-style multi-turn tool-agent work.

Next-action SFT should be trained with last-assistant masking:

```bash
python course/03_sft_warmup/train_sft_local.py \
  --file data/retail/sft_train_next_action.jsonl \
  --sft-mask-mode last-assistant \
  --batch-size 1 \
  --peak-lr 1e-5 \
  --max-steps 48 \
  --chunk-size-batches 12
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

SFT also runs a tokenization preflight. By default, rows that exceed `ART_MAX_SEQ_LENGTH` are filtered before training so the loss curve does not contain zero-loss batches from examples that ART drops internally. Use `--keep-overlength-sft-rows` only when you explicitly want to inspect ART's raw drop behavior.

Training scripts use W&B Artifacts for lineage:

- `retail-course-data:latest` contains the TAU Retail JSONL splits plus generated `sft_train.jsonl`.
- Data artifacts include all `sft*.jsonl` files plus summary JSON files in the data directory, so full-trajectory, next-action, teacher-mixed, and AReaL-derived SFT variants can be traced.
- SFT logs the latest local ART LoRA checkpoint as `<ART_MODEL_NAME>-checkpoint:sft-anchor`, including the exact `sft_file` and `sft_mask_mode` in checkpoint metadata.
- GRPO, GSPO, and RULER runs call `use_artifact` on the SFT checkpoint and log their own branch checkpoint artifacts.
- Eval runs can call `use_artifact` on both the dataset and the evaluated checkpoint, while Weave stores rollout traces.

Reward and SFT data design notes: [course/04_grpo_local/reward_and_sft_design_notes.md](course/04_grpo_local/reward_and_sft_design_notes.md).

Official tau2 evaluation bridge: [course/02_weave_evals/official_tau2_eval_bridge.md](course/02_weave_evals/official_tau2_eval_bridge.md). Use this when you need benchmark-grade `DB * COMMUNICATE` scores in addition to the lightweight training proxy.

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

## Diagnostic H100 Results

The table below is retained as an older diagnostic run with strict reference-trajectory scoring. It is not the final expected-results table for the workshop. The course is being refreshed around tau-style outcome metrics (`tau_sparse` / `tau_irc`), where `proxy_outcome_success`, state-changing action correctness, and communication success are the primary training-proxy comparison columns. Official tau2 reporting should use `tau2_official_reward`, `tau2_db_success`, and `tau2_communicate_success` imported from the official runtime.

| Stage | Reward | Task success | Tool F1 | Tool order | Invalid calls avg | Final text F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5102 | 0.1458 | 0.7268 | 0.5542 | 0.6667 | 0.2439 |
| SFT anchor, lr=3e-5 | 0.5012 | 0.2083 | 0.7255 | 0.5750 | 0.7708 | 0.2541 |
| SFT -> GRPO, 8 steps | 0.5165 | 0.1875 | 0.7283 | 0.5740 | 0.6667 | 0.2576 |
| SFT -> GSPO, 3 steps | 0.5043 | 0.2083 | 0.7268 | 0.5792 | 0.7292 | 0.2405 |
| SFT -> RULER-GRPO, 3 steps | 0.5026 | 0.1875 | 0.7254 | 0.5750 | 0.7292 | 0.2405 |

Current tau-style diagnostics show that naive full-trajectory SFT on the small curriculum slice does not reliably beat the baseline. The course therefore treats full-trajectory SFT as a teaching baseline and validates next-action SFT before using an SFT checkpoint as the parent for GRPO/GSPO/RULER.

W&B Artifacts are logged for the dataset, the SFT checkpoint, and each RL branch checkpoint. The comparison script logs a horizontal W&B table with columns such as `model`, `stage`, `model_artifact_path`, `reward`, `task_success`, and delta metrics, while Weave stores rollout and eval traces.
RL training logs also include signal-quality diagnostics: reward range/std before and after zero-variance filtering, winner-minus-loser gaps for outcome/state-action metrics, state-action attempt/reached rates, and tau-style reward components. These columns are meant to prove that GRPO/GSPO is optimizing agentic behavior rather than noise from read-only replay differences.
The runbook writes both full audit tables (`checkpoint_eval_comparison.md/.csv`) and compact presentation tables (`checkpoint_eval_summary.md/.csv`).

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
