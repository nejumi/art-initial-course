# OpenPipe ART x W&B x Weave Retail Hands-on

This repository contains the hands-on code for a course on OpenPipe ART with W&B Models and Weave.

Japanese overview: [README.ja.md](README.ja.md)

The main task is a Retail Customer Support Agent based on open retail tool-calling data:

- `lefft/tau-dev-task-retail-v1` for SFT and workflow validation
- tau-bench / tau2-bench retail concepts for rollout, reward, and eval design
- optional tau2-style next-action SFT data from `amityco/...`, `inclusionAI/AReaL-tau2-data`, and successful public tau2 retail traces
- optional large-scale appendix data from `fuvty/tau-bench-synthetic`

The code is structured so the early labs can be inspected without a GPU. Local ART training labs require a CUDA-capable GPU and `openpipe-art[backend]`.

## Open Training Data

The course uses public datasets so the labs can run end to end without private task data.

| Dataset | Main use | Why it is useful | Course handling |
| --- | --- | --- | --- |
| [`lefft/tau-dev-task-retail-v1`](https://huggingface.co/datasets/lefft/tau-dev-task-retail-v1) | Default SFT, rollout, and evaluation source | Successful retail trajectories already serialized in OpenAI tool-call format | Downloaded by `download_tau_retail.py`; split into disjoint `sft` / `train` / `validation` / `test` buckets |
| [`amityco/tau-bench-retail-train-next-action-all-step-score-v0.2`](https://huggingface.co/datasets/amityco/tau-bench-retail-train-next-action-all-step-score-v0.2) | Optional teacher next-action SFT | Step-level retail tool-call rows with candidate scores | Converted by `make_teacher_next_action_sft_jsonl.py`; filtered to the SFT task fold before mixing |
| [`inclusionAI/AReaL-tau2-data`](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data) | Advanced next-action SFT option | Mirrors recent tau2-style SFT + verifiable-reward RL workflows | Converted by `make_areal_retail_sft_jsonl.py`; strips thinking fields and normalizes tool calls |
| [`KermitCO/qwen3.5-9B-tau2bench-retail-traces`](https://huggingface.co/datasets/KermitCO/qwen3.5-9B-tau2bench-retail-traces) | Success-trace SFT warm start | Retail-only tau2 traces with canonical reward and judge-quality fields | Converted by `make_success_trace_retail_sft_jsonl.py`; defaults to non-memory, blind-strict, reward-1 traces |
| [`fuvty/tau-bench-synthetic`](https://huggingface.co/datasets/fuvty/tau-bench-synthetic) | Large-scale appendix SFT/RL source | Larger synthetic tau-bench retail trajectories and next-action rows | Prepared by `prepare_tau_synthetic_retail.py`; intended for offline evidence runs rather than the first live workshop path |
| [`Jarrodbarnes/Qwen3-4B-tau2-grpo-v1`](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1) | Prior-art reference | Public model card documents SFT -> RFT -> GRPO, turn-level shaping, and tau2 eval settings | Not used as course weights by default; informs the training/eval recipe |

Full-dialog SFT is easy to explain, but it can over-supervise long dialogue style and final responses. Next-action SFT is the stronger path when we want to align with modern agent training recipes and avoid training on every prior assistant action repeatedly.

## Task and Metric Terminology

The hands-on task is a retail customer-support agent. A user asks for help with an order, return, cancellation, exchange, address issue, or similar workflow. The agent reads customer/order/product/policy state with read-only tools, decides whether a consequential state-changing action is allowed, calls the correct state-changing tool with correct arguments, then communicates the outcome to the user.

The bridge curriculum intentionally keeps the first RL lab short: most examples have one consequential state-changing action. That makes the class focus on the core agentic loop: read state, decide policy, mutate state correctly, and communicate.

Experiment labels and evaluation metrics are separate:

| Term | Type | Meaning |
| --- | --- | --- |
| `bridge-only` | Experiment condition | SFT uses only local bridge next-action rows. |
| `teacher mix` | Experiment condition | SFT mixes bridge rows with public scored teacher next-action rows. |
| `success mix` | Experiment condition | SFT mixes bridge rows with successful public tau2 retail traces converted to next-action rows. |
| `retail_task_success` | Metric | Primary course success signal: the agent performs the required state-changing mutation, communicates the outcome, avoids invalid actions, and finishes cleanly. |
| `reference_tool_sequence_exact_match` | Metric | Whether the model called the same tools in the same order as the reference solution. This measures reference-path imitation; a different valid tool path can still solve the task. |
| `reward` | Metric | The configured reward profile, usually `tau_irc`: weighted outcome, state-action, argument, communication, and penalty terms. |

The main course metric is `retail_task_success`. It follows the outcome-oriented spirit of tau-style evaluation while staying light enough to use inside ART rollouts: read-only tool order is diagnostic, while consequential state-changing mutations and final user communication drive the success signal. Official tau2 evaluation is available as an optional final check when a benchmark-grade runtime comparison is needed.

Evaluation has two classroom-facing levels:

| Context | Environment | Metric | Purpose |
| --- | --- | --- | --- |
| Training rollouts | Lightweight local retail environment | `data/step_retail_task_success_mean` | Fast on-policy feedback during GRPO/GSPO/RULER. |
| Checkpoint validation | Held-out scenarios in the same lightweight environment | `retail_task_success` | Fast comparison of baseline, SFT, and RL checkpoints. |

Data artifacts keep source metadata so SFT rows, RL rollout rows, validation rows, and test rows can be checked separately. The bridge curriculum splits by task hash so these buckets do not share scenario IDs or task IDs.
When public teacher next-action rows are enabled, the runbook applies the same task-hash split and keeps only the SFT fold. GRPO rollouts and checkpoint validation therefore remain task-disjoint from both local bridge SFT rows and teacher SFT rows.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your W&B/OpenAI keys.
# Edit course/09_runbooks/config.yaml for run scale, model size, and GPU memory.

python course/00_setup/env_check.py
python course/00_setup/art_api_smoke.py
python course/03_sft_warmup/download_tau_retail.py --sample-if-download-fails
python course/03_sft_warmup/make_sft_jsonl.py --limit 200
python course/03_sft_warmup/make_next_action_sft_jsonl.py --data-dir data/retail
python course/01_art_primitives/dry_run_rollout.py --limit 3
```

Workshop run settings live in [`course/09_runbooks/config.yaml`](course/09_runbooks/config.yaml). This file is the participant-facing control surface:

```yaml
run_profile: workshop_fast_h100
model_profile: standard
gpu_memory_preset: standard
overrides: {}
```

The default `run_profile` is the workshop-sized SFT -> GRPO path. For a full validation run, change it to `validated_h100`. Use `model_profile: tiny` for constrained hardware checks, or `gpu_memory_preset: low` to lower vLLM memory pressure. Detailed profile definitions are in [`course/09_runbooks/base_config.yaml`](course/09_runbooks/base_config.yaml).

You can also choose a profile from the CLI:

```bash
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile smoke_tiny
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile workshop_fast_h100
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile validated_h100
```

For the workshop runbook, `config.yaml` is the source of truth for run scale, model size, and GPU memory. `.env` is for account/project credentials and runtime secrets.

Bridge curriculum option for a shorter first RL lab:

```bash
python course/03_sft_warmup/make_bridge_curriculum.py \
  --source-dir data/retail \
  --output-dir data/retail_bridge_state1 \
  --max-state-actions 1 \
  --max-tool-calls 8 \
  --max-turns 32 \
  --holdout-modulo 6 \
  --sft-remainder 2 \
  --validation-remainder 0 \
  --test-remainder 1
python course/07_models_registry_weave/log_data_artifacts.py \
  --data-dir data/retail_bridge_state1 \
  --artifact-name retail-course-data-bridge-state1
```

Full local GPU runbook:

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

Full validation run with public teacher rows, AReaL tau2 SFT rows, and successful tau2 retail traces:

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

The runbook defaults to `--continue-on-invalid` for tau-style RL. Unexpected state-changing actions are still penalized, but rollouts continue long enough for final state/action and communication rewards to be observed. Use `--no-continue-on-invalid` to study exact reference-path failure modes.
For tau-inspired profiles, the lightweight replay environment also defaults to `RETAIL_ALLOW_REFERENCE_STATE_ACTION_JUMPS=true`: a single exact reference state-changing action can skip preceding read-only replay turns, while wrong state-changing names or arguments still count as bad state actions. This keeps `retail_task_success` focused on consequential outcomes without requiring the full tau2 runtime.
For stochastic validation, add `--eval-rollouts-per-scenario 4 --eval-temperature 0.2` so `outcome_pass_at_k`, `task_pass_at_k`, and reward variance columns are meaningful. Deterministic eval remains available for quick regression checks.

On a Slurm H100 cluster, the same flow can be launched with:

```bash
sbatch course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  validated_h100
```

To include RULER in an advanced run, set `overrides: {rl_algos: "grpo,gspo,ruler"}` or create an instructor profile in `base_config.yaml`. RULER uses an external judge model, so plan for the extra judge calls.

Large-scale appendix path:

```bash
# First build the larger synthetic retail source and SFT anchor.
sbatch course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  appendix_tau_synthetic_sft_h100

# After the SFT anchor exists, run GRPO and GSPO branches in parallel on one 8xH100 node.
sbatch course/09_runbooks/sunk_h100_parallel_profiles.sbatch \
  appendix_tau_synthetic_grpo_h100 \
  appendix_tau_synthetic_gspo_h100
```

This parallel wrapper assigns one GPU to each profile. It speeds up branch sweeps and checkpoint evaluations, but it does not turn the current single-GPU LocalBackend loop into one distributed GRPO update. For true actor/learner rollout distribution, see the scaling appendix.

Advanced SFT data option:

```bash
python course/03_sft_warmup/make_teacher_next_action_sft_jsonl.py \
  --tools-data-dir data/retail_bridge_state1 \
  --output data/retail_bridge_state1/sft_teacher_retail_next_action.jsonl \
  --limit 512 \
  --holdout-modulo 6 \
  --sft-remainder 2

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
    data/retail_bridge_state1/sft_next_action.jsonl \
    data/retail_bridge_state1/sft_teacher_retail_next_action.jsonl \
    data/retail_bridge_state1/sft_areal_retail_next_action.jsonl \
    data/retail_bridge_state1/sft_success_trace_retail_next_action.jsonl \
  --limits -1 512 512 512 \
  --output data/retail_bridge_state1/sft_next_action_teacher_areal_success_trace_mix.jsonl
```

The default course path uses compact TAU retail trajectories. The bridge curriculum keeps short successful trajectories with exactly one state-changing action and writes disjoint SFT, RL rollout, validation, and test buckets. This keeps GRPO validation independent while preserving a short, understandable task. The teacher, AReaL, and success-trace converters are provided for stronger warm starts and advanced experiments that want to align SFT data construction with recent tau2-style multi-turn tool-agent work.

The appendix path uses `fuvty/tau-bench-synthetic` to increase SFT and rollout coverage. It is useful for offline GRPO/GSPO evidence runs when a short live workshop cannot run enough RL updates to demonstrate convergence.

Next-action SFT should be trained with last-assistant masking:

```bash
python course/03_sft_warmup/train_sft_local.py \
  --file data/retail_bridge_state1/sft_next_action.jsonl \
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

`dense` is a replay-style teaching reward. Use `tau_irc` or `tau_irc_balanced` for the main agentic RL lab because they keep verifiable outcome/state-changing-action signal as the anchor while treating read-only replay mismatches as diagnostics or small penalties.

SFT is intentionally chunked: one ART SFT call produces one ART checkpoint/log point, so `--chunk-size-batches` keeps the SFT loss curve visible in W&B instead of collapsing a full epoch into a single point.

SFT also runs a tokenization preflight. By default, rows that exceed `ART_MAX_SEQ_LENGTH` are filtered before training so the loss curve does not contain zero-loss batches from examples that ART drops internally. Use `--keep-overlength-sft-rows` only when you explicitly want to inspect ART's raw drop behavior.

Training scripts use W&B Artifacts and Weave Evaluations for lineage:

- `retail-course-data:latest` contains the TAU Retail JSONL splits plus generated `sft_full.jsonl` and `sft_next_action.jsonl`.
- Data artifacts include all `sft*.jsonl` files plus summary JSON files in the data directory, so full-trajectory, next-action, teacher-mixed, AReaL-derived, and success-trace SFT variants can be traced.
- SFT logs the latest local ART LoRA checkpoint as `<ART_MODEL_NAME>-checkpoint:sft-anchor`, including the exact `sft_file` and `sft_mask_mode` in checkpoint metadata.
- GRPO, GSPO, and RULER runs call `use_artifact` on the SFT checkpoint and log their own branch checkpoint artifacts.
- Eval runs can call `use_artifact` on both the dataset and the evaluated checkpoint, while Weave stores rollout traces.
- After checkpoint comparison, the runbook publishes each cached JSONL result as a Weave Evaluation using the same scorer set, so trace-level behavior and stage-level eval summaries can be inspected without rerunning rollouts.
- `course/07_models_registry_weave/inspect_wandb_lineage.py` can verify artifact aliases, metadata, and logged/used relationships with the W&B Public API after a full run.
- Run IDs and run names are generated by W&B. Course scripts put stage identity into run tags such as `stage:sft-train`, `kind:training`, `algo:grpo`, `split:validation`, `profile:validated_h100`, and `reward:tau_irc_balanced`; notes and config hold the base model, ART model, dataset, run slug, and Weave project.
- When Weave tracing is enabled, the Weave client is explicitly bound to the active W&B-generated `run.id`, so traces and cached Evaluations can be read from the same experiment context as metrics and artifacts.

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

Official tau2 evaluation bridge: [course/02_weave_evals/official_tau2_eval_bridge.md](course/02_weave_evals/official_tau2_eval_bridge.md). Use this only when you need a benchmark-grade runtime comparison in addition to the course success metric.

## Validation Outputs

The validated H100 path uses `LiquidAI/LFM2.5-8B-A1B`, task-disjoint bridge next-action rows, task-fold-filtered public teacher next-action rows, last-assistant SFT masking, and GRPO from the SFT anchor. Candidate RL checkpoints are forked and evaluated on held-out validation, so the final comparison does not assume the latest training step is best.

The runbook writes `checkpoint_eval_summary.md/.csv` and `checkpoint_acceptance.md/.json`. For the course result, GRPO should clear the acceptance gate against the SFT anchor on held-out validation: higher reward, higher `retail_task_success` or state-action success, and lower state-action error. W&B stores the horizontal comparison table, and Weave stores the rollout traces behind each evaluation row.

## Model Selection

For the workshop runbook, choose the trainable ART model in [`course/09_runbooks/config.yaml`](course/09_runbooks/config.yaml):

```yaml
model_profile: standard   # LiquidAI/LFM2.5-8B-A1B
```

Common values are:

| `model_profile` | Model | Use |
| --- | --- | --- |
| `standard` | `LiquidAI/LFM2.5-8B-A1B` | Main H100 workshop path. |
| `tiny` | `Qwen/Qwen3-0.6B` | Setup checks and constrained GPUs. |
| `openpipe` | `OpenPipe/Qwen3-14B-Instruct` | OpenPipe/Qwen compatibility path. |
| `moe` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Larger managed-training discussion/demo. |
| `custom` | value from `base_model` | Instructor-led experiments. |

For a direct model override in the runbook:

```yaml
model_profile: custom
base_model: LiquidAI/LFM2.5-1.2B-Instruct
```

`standard` is the validated H100 classroom path. `tiny` is intended to make setup and short smoke tests accessible on smaller machines while staying on an ART-supported Qwen3 dense model path. Run the setup check and a short SFT/RL smoke run before relying on a custom model for the full RL lab because ART backend compatibility depends on the architecture, tokenizer, vLLM, tool-call parser, and LoRA trainer path.

RULER judge selection is separate from the trainable model. It defaults to `openai/gpt-5.5` with medium reasoning effort and can be overridden with `--ruler-judge-model` / `--ruler-judge-effort` in the runbook or `--judge-model` / `--judge-effort` in `course/05_ruler/train_with_ruler.py`.

## Evaluation Tables

Checkpoint comparisons are written as horizontal tables so each row is one model stage and each metric is a separate column. The compact presentation table uses `retail_task_success` as the primary success column.

| Column | Meaning |
| --- | --- |
| `model` | Base model or checkpoint family. |
| `stage` | Baseline, SFT, GRPO, GSPO, RULER, or imported official tau2 result. |
| `model_artifact_path` | W&B Artifact path for the evaluated checkpoint. |
| `reward` | Mean reward under the selected course reward profile. |
| `retail_task_success` | Primary course success signal for training rollouts and checkpoint validation. |
| `tau2_official_success` | Optional official-runtime success imported from tau2 result files. |
| `reference_tool_sequence_exact_match` | Whether the model followed the same tool path as the reference solution. |
| `state_action_sequence_match` | Whether required state-changing actions were called correctly. |
| `communication_success` | Whether the final user-facing response communicates the outcome. |
| `bad_state_action` / `missing_state_action` | State-changing action errors to inspect in Weave traces. |

W&B Artifacts are logged for the dataset, the SFT checkpoint, and each RL branch checkpoint. The comparison script logs a horizontal W&B table with columns such as `model`, `stage`, `model_artifact_path`, `reward`, `retail_task_success`, `reference_tool_sequence_exact_match`, and delta metrics, while Weave stores rollout traces and cached checkpoint Evaluations.
RL training logs also include signal-quality diagnostics: reward range/std before and after zero-variance filtering, winner-minus-loser gaps for outcome/state-action metrics, state-action attempt/reached rates, and tau-style reward components. These columns are meant to prove that GRPO/GSPO is optimizing agentic behavior rather than noise from read-only replay differences.
The runbook writes both full audit tables (`checkpoint_eval_comparison.md/.csv`) and compact presentation tables (`checkpoint_eval_summary.md/.csv`).
It also writes `checkpoint_acceptance.md/.json`, which checks that SFT preserves baseline success signals and that each RL branch improves over the SFT parent on reward plus agentic success and state-action error metrics.

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
