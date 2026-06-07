# OpenPipe ART x W&B x Weave Retail Hands-on

This repository contains the hands-on code for a course on OpenPipe ART with W&B Models and Weave.

Japanese overview: [README.ja.md](README.ja.md)

The main task is a Retail Customer Support Agent based on open retail tool-calling data:

- `lefft/tau-dev-task-retail-v1` for SFT and workflow validation
- tau-bench / tau2-bench retail concepts for rollout, reward, and eval design
- optional tau2-style next-action SFT data from `amityco/...`, `inclusionAI/AReaL-tau2-data`, and successful public tau2 retail traces

The code is structured so the early labs can be inspected without a GPU. Local ART training labs require a CUDA-capable GPU and `openpipe-art[backend]`.

## Open Training Data

The course intentionally uses public datasets so a workshop can run end to end without private task data.

| Dataset | Main use | Why it is useful | Course handling |
| --- | --- | --- | --- |
| [`lefft/tau-dev-task-retail-v1`](https://huggingface.co/datasets/lefft/tau-dev-task-retail-v1) | Default SFT and evaluation data | Successful retail trajectories already serialized in OpenAI tool-call format | Downloaded by `download_tau_retail.py`; converted to `sft_*.jsonl` |
| [`amityco/tau-bench-retail-train-next-action-all-step-score-v0.2`](https://huggingface.co/datasets/amityco/tau-bench-retail-train-next-action-all-step-score-v0.2) | Optional teacher next-action SFT | Step-level retail tool-call rows with candidate scores | Converted by `make_teacher_next_action_sft_jsonl.py`; can be mixed with bridge next-action rows |
| [`inclusionAI/AReaL-tau2-data`](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data) | Advanced next-action SFT option | Mirrors recent tau2-style SFT + verifiable-reward RL workflows | Converted by `make_areal_retail_sft_jsonl.py`; strips thinking fields and normalizes tool calls |
| [`KermitCO/qwen3.5-9B-tau2bench-retail-traces`](https://huggingface.co/datasets/KermitCO/qwen3.5-9B-tau2bench-retail-traces) | Success-trace SFT warm start | Retail-only tau2 traces with canonical reward and judge-quality fields | Converted by `make_success_trace_retail_sft_jsonl.py`; defaults to non-memory, blind-strict, reward-1 traces |
| [`Jarrodbarnes/Qwen3-4B-tau2-grpo-v1`](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1) | Prior-art reference | Public model card documents SFT -> RFT -> GRPO, turn-level shaping, and tau2 eval settings | Not used as course weights by default; informs the training/eval recipe |

Full-dialog SFT is easy to explain, but it can over-supervise long dialogue style and final responses. Next-action SFT is the stronger path when we want to align with modern agent training recipes and avoid training on every prior assistant action repeatedly.

## Task and Metric Terminology

The hands-on task is a retail customer-support agent. A user asks for help with an order, return, cancellation, exchange, address issue, or similar workflow. The agent reads customer/order/product/policy state with read-only tools, decides whether a consequential state-changing action is allowed, calls the correct state-changing tool with correct arguments, then communicates the outcome to the user.

The bridge curriculum intentionally keeps the first RL lab short: most examples have one consequential state-changing action. That makes the class focus on the core agentic loop: read state, decide policy, mutate state correctly, and communicate.

Do not confuse experiment labels with metrics:

| Term | Type | Meaning |
| --- | --- | --- |
| `bridge-only` | Experiment condition | SFT uses only local bridge next-action rows. |
| `teacher mix` | Experiment condition | SFT mixes bridge rows with public scored teacher next-action rows. |
| `success mix` | Experiment condition | SFT mixes bridge rows with successful public tau2 retail traces converted to next-action rows. |
| `task_success` | Metric | Strict exact reference tool-call sequence success with no invalid tool call. Useful for diagnosing trace imitation, but stricter than tau-style outcome scoring. |
| `outcome_success` / `proxy_outcome_success` | Metric | Lightweight tau-style proxy: correct state-changing action sequence, successful communication, no bad/missing/invalid state action, and no truncation. |
| `reward` | Metric | The configured reward profile, usually `tau_irc`: weighted outcome, state-action, argument, communication, and penalty terms. |

For workshop reporting, keep data lineage and split hygiene explicit. Every mixed SFT row carries source metadata, and data artifacts include generated summaries so instructors can check which source IDs entered training. Do not present official tau2 numbers for an eval task set that may overlap with public teacher or success-trace SFT rows; use those runs as training-proxy demos unless the held-out task IDs have been audited.

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
  --reward-profile tau_irc \
  --continue-on-invalid \
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
  --reward-profile tau_irc \
  --continue-on-invalid \
  --rl-steps 32 \
  --rl-algos grpo,gspo
```

For a research-aligned instructor validation pass, mix public teacher rows, AReaL tau2 SFT rows, and successful tau2 retail traces:

```bash
python course/09_runbooks/run_retail_agentic_sequence.py \
  --run-slug lfm25-8b-a1b-bridge-state1-teacher-areal-success \
  --base-model LiquidAI/LFM2.5-8B-A1B \
  --data-dir data/retail_bridge_state1 \
  --source-dir data/retail \
  --data-artifact-name retail-course-data-bridge-state1-teacher-areal-success \
  --build-bridge \
  --include-teacher-sft \
  --teacher-sft-limit 512 \
  --include-areal-sft \
  --areal-sft-limit 512 \
  --include-success-trace-sft \
  --success-trace-sft-limit 512 \
  --sft-max-steps 240 \
  --reward-profile tau_irc \
  --continue-on-invalid \
  --rl-steps 48 \
  --rl-algos grpo,gspo,ruler
```

The runbook defaults to `--continue-on-invalid` for tau-style RL. Unexpected state-changing actions are still penalized, but rollouts continue long enough for final state/action and communication rewards to be observed. Use `--no-continue-on-invalid` only when demonstrating strict replay failure modes.
For tau-inspired profiles, the lightweight replay environment also defaults to `RETAIL_ALLOW_REFERENCE_STATE_ACTION_JUMPS=true`: a single exact reference state-changing action can skip preceding read-only replay turns, while wrong state-changing names or arguments still count as bad state actions. This keeps the hands-on proxy closer to tau-style outcome scoring without requiring the full tau2 runtime.
For the final course report, add stochastic validation with `--eval-rollouts-per-scenario 4 --eval-temperature 0.2` so `outcome_pass_at_k`, `task_pass_at_k`, and reward variance columns are meaningful. Keep the default deterministic eval for quick instructor checks.

On a Slurm H100 cluster, the same flow can be launched with:

```bash
sbatch course/09_runbooks/sunk_h100_retail_agentic_sequence.sbatch
```

For the full instructor validation including RULER, pass `grpo,gspo,ruler` as the `RL_ALGOS` argument. RULER uses an external judge model, so keep it in instructor or enterprise validation runs unless the workshop budget explicitly includes judge calls.

Advanced SFT data option:

```bash
python course/03_sft_warmup/make_teacher_next_action_sft_jsonl.py \
  --tools-data-dir data/retail_bridge_state1 \
  --output data/retail_bridge_state1/sft_teacher_retail_next_action.jsonl \
  --limit 512

python course/03_sft_warmup/make_areal_retail_sft_jsonl.py \
  --tools-data-dir data/retail_bridge_state1 \
  --output data/retail_bridge_state1/sft_areal_retail_next_action.jsonl \
  --limit 512

python course/03_sft_warmup/make_success_trace_retail_sft_jsonl.py \
  --tools-data-dir data/retail_bridge_state1 \
  --output data/retail_bridge_state1/sft_success_trace_retail_next_action.jsonl \
  --full-output data/retail_bridge_state1/sft_success_trace_retail_full.jsonl \
  --limit 512

python course/03_sft_warmup/mix_sft_jsonl.py \
  --inputs \
    data/retail_bridge_state1/sft_train_next_action.jsonl \
    data/retail_bridge_state1/sft_teacher_retail_next_action.jsonl \
    data/retail_bridge_state1/sft_areal_retail_next_action.jsonl \
    data/retail_bridge_state1/sft_success_trace_retail_next_action.jsonl \
  --limits -1 512 512 512 \
  --output data/retail_bridge_state1/sft_train_next_action_teacher_areal_success_trace_mix.jsonl
```

The default course path uses compact TAU retail trajectories. The bridge curriculum keeps short successful trajectories with exactly one state-changing action, which is useful for proving that agentic RL can improve a verifiable outcome before moving to the broader retail curriculum. The teacher, AReaL, and success-trace converters are provided for stronger warm starts and advanced experiments that want to align SFT data construction with recent tau2-style multi-turn tool-agent work.

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
  --reward-profile tau_irc \
  --continue-on-invalid \
  --learning-rate 5e-6
```

`dense` is a replay-style teaching reward. Use `tau_irc` for the main agentic RL lab because it keeps verifiable outcome/state-changing-action signal as the anchor while treating read-only replay mismatches as diagnostics or small penalties.

SFT is intentionally chunked: one ART SFT call produces one ART checkpoint/log point, so `--chunk-size-batches` keeps the SFT loss curve visible in W&B instead of collapsing a full epoch into a single point.

SFT also runs a tokenization preflight. By default, rows that exceed `ART_MAX_SEQ_LENGTH` are filtered before training so the loss curve does not contain zero-loss batches from examples that ART drops internally. Use `--keep-overlength-sft-rows` only when you explicitly want to inspect ART's raw drop behavior.

Training scripts use W&B Artifacts and Weave Evaluations for lineage:

- `retail-course-data:latest` contains the TAU Retail JSONL splits plus generated `sft_train.jsonl`.
- Data artifacts include all `sft*.jsonl` files plus summary JSON files in the data directory, so full-trajectory, next-action, teacher-mixed, AReaL-derived, and success-trace SFT variants can be traced.
- SFT logs the latest local ART LoRA checkpoint as `<ART_MODEL_NAME>-checkpoint:sft-anchor`, including the exact `sft_file` and `sft_mask_mode` in checkpoint metadata.
- GRPO, GSPO, and RULER runs call `use_artifact` on the SFT checkpoint and log their own branch checkpoint artifacts.
- Eval runs can call `use_artifact` on both the dataset and the evaluated checkpoint, while Weave stores rollout traces.
- After checkpoint comparison, the runbook publishes each cached JSONL result as a Weave Evaluation using the same scorer set, so instructors can inspect both trace-level behavior and stage-level eval summaries without rerunning rollouts.
- `course/07_models_registry_weave/inspect_wandb_lineage.py` can verify artifact aliases, metadata, and logged/used relationships with the W&B Public API after a full run.

When a long RL run peaks before the final step, log the stable intermediate checkpoint without moving the mutable `latest` alias:

```bash
python course/02_weave_evals/select_checkpoint_candidate.py \
  data/retail_bridge_state1/lfm25-8b-a1b-bridge-state1_runbook/train_metrics_grpo_bridge-s32-lr2e6.jsonl \
  --metric data/step_reward_mean \
  --output-md data/retail_bridge_state1/lfm25-8b-a1b-bridge-state1_runbook/grpo_checkpoint_candidates.md

python course/07_models_registry_weave/log_checkpoint_artifact.py \
  --model-name retail-support-agent-lfm25-8b-a1b-bridge-state1-grpo-bridge-s32-lr2e6 \
  --stage grpo \
  --checkpoint-step 23 \
  --alias validation-candidate
```

The candidate selector reads the per-step RL metrics JSONL written by the runbook. It is meant to shortlist checkpoints for fresh held-out rollout evaluation; it does not replace validation evals or Weave trace inspection.

For a fresh rollout eval of that step on another GPU, first materialize it under a unique model name without starting LocalBackend/vLLM:

```bash
python course/08_enterprise_ops/fork_checkpoint.py \
  --from-model retail-support-agent-lfm25-8b-a1b-bridge-state1-grpo-bridge-s32-lr2e6 \
  --to-model retail-support-agent-lfm25-8b-a1b-bridge-state1-grpo-step23-eval \
  --not-after-step 23 \
  --file-only
```

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

RULER judge selection is separate from the trainable model. It defaults to `openai/gpt-5.5` with medium reasoning effort and can be overridden with `--ruler-judge-model` / `--ruler-judge-effort` in the runbook or `--judge-model` / `--judge-effort` in `course/05_ruler/train_with_ruler.py`.

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

W&B Artifacts are logged for the dataset, the SFT checkpoint, and each RL branch checkpoint. The comparison script logs a horizontal W&B table with columns such as `model`, `stage`, `model_artifact_path`, `reward`, `task_success`, and delta metrics, while Weave stores rollout traces and cached checkpoint Evaluations.
RL training logs also include signal-quality diagnostics: reward range/std before and after zero-variance filtering, winner-minus-loser gaps for outcome/state-action metrics, state-action attempt/reached rates, and tau-style reward components. These columns are meant to prove that GRPO/GSPO is optimizing agentic behavior rather than noise from read-only replay differences.
The runbook writes both full audit tables (`checkpoint_eval_comparison.md/.csv`) and compact presentation tables (`checkpoint_eval_summary.md/.csv`).
It also writes `checkpoint_acceptance.md/.json`. This conservative gate rejects a stage unless SFT preserves baseline task/outcome metrics and each RL branch improves over the SFT parent on reward plus at least one agentic success metric and one state-action error metric. Use the gate as a finalization checklist before copying expected results into this README.

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
