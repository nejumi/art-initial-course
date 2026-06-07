#!/usr/bin/env bash
#SBATCH --job-name=art-retail-rl
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=.art/sunk_runs/logs/%x-%j.out
#SBATCH --error=.art/sunk_runs/logs/%x-%j.err

set -euo pipefail

cd "${ART_COURSE_ROOT:-/mnt/home/yyamamoto+cwb607/work/art-initial-course}"

set -a
if [[ -f .env ]]; then
  source .env
fi
set +a

export PATH="$PWD/.art/venv311/bin:$PATH"
export PYTHONUNBUFFERED=1

RUN_SLUG="${1:?run slug is required}"
BASE_MODEL="${2:-LiquidAI/LFM2.5-8B-A1B}"
ART_PATH_ARG="${3:?ART path with the SFT anchor is required}"
DATA_DIR="${4:?data dir is required}"
SOURCE_DIR="${5:-$PWD/.art/full_data/retail}"
DATA_ARTIFACT_NAME="${6:?data artifact name is required}"
REWARD_PROFILE="${7:-tau_irc_balanced}"
RL_STEPS="${8:-18}"
RL_TRAIN_LIMIT="${9:-48}"
RL_ALGO="${10:-grpo}"
RL_GROUPS_PER_STEP="${11:-8}"
RL_ROLLOUTS_PER_SCENARIO="${12:-8}"
RL_LEARNING_RATE="${13:-5e-7}"
RL_SUFFIX="${14:-balanced-s18-g8-lr5e7}"
REPORT_DIR="${15:-$DATA_DIR/${RUN_SLUG}_${RL_SUFFIX}_runbook}"
MAX_COMPLETION_TOKENS="${16:-768}"
RL_CANDIDATE_METRIC="${17:-data/step_course_score_mean}"

mkdir -p .art/sunk_runs/logs
mkdir -p "$REPORT_DIR"

export ART_PATH="$ART_PATH_ARG"
export ART_BASE_MODEL="$BASE_MODEL"
export ART_MODEL_PROFILE=custom
export ART_VLLM_RUNTIME_CACHE_DIR="${ART_VLLM_RUNTIME_CACHE_DIR:-$PWD/.art/vllm_runtime_cache}"
export HF_HOME="${HF_HOME:-$PWD/.art/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_XET_HIGH_PERFORMANCE=1
export ART_VLLM_GPU_MEMORY_UTILIZATION="${ART_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export ART_VLLM_MAX_MODEL_LEN="${ART_VLLM_MAX_MODEL_LEN:-16384}"
export ART_VLLM_MAX_NUM_BATCHED_TOKENS="${ART_VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export ART_VLLM_MAX_NUM_SEQS="${ART_VLLM_MAX_NUM_SEQS:-8}"

echo "=== retail RL branch runbook ==="
echo "run_slug=$RUN_SLUG"
echo "base_model=$BASE_MODEL"
echo "art_path=$ART_PATH"
echo "data_dir=$DATA_DIR"
echo "data_artifact_name=$DATA_ARTIFACT_NAME"
echo "reward_profile=$REWARD_PROFILE"
echo "rl_algo=$RL_ALGO"
echo "rl_steps=$RL_STEPS"
echo "rl_train_limit=$RL_TRAIN_LIMIT"
echo "rl_groups_per_step=$RL_GROUPS_PER_STEP"
echo "rl_rollouts_per_scenario=$RL_ROLLOUTS_PER_SCENARIO"
echo "rl_learning_rate=$RL_LEARNING_RATE"
echo "rl_suffix=$RL_SUFFIX"
echo "rl_candidate_metric=$RL_CANDIDATE_METRIC"
echo "report_dir=$REPORT_DIR"
hostname
date -Is
nvidia-smi

python -B course/09_runbooks/run_retail_agentic_sequence.py \
  --run-slug "$RUN_SLUG" \
  --base-model "$BASE_MODEL" \
  --art-path "$ART_PATH" \
  --data-dir "$DATA_DIR" \
  --source-dir "$SOURCE_DIR" \
  --data-artifact-name "$DATA_ARTIFACT_NAME" \
  --reward-profile "$REWARD_PROFILE" \
  --skip-baseline-eval \
  --skip-sft \
  --rl-steps "$RL_STEPS" \
  --rl-train-limit "$RL_TRAIN_LIMIT" \
  --rl-algos "$RL_ALGO" \
  --rl-groups-per-step "$RL_GROUPS_PER_STEP" \
  --rl-rollouts-per-scenario "$RL_ROLLOUTS_PER_SCENARIO" \
  --rl-learning-rate "$RL_LEARNING_RATE" \
  --rl-suffix "$RL_SUFFIX" \
  --rl-candidate-metric "$RL_CANDIDATE_METRIC" \
  --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
  --report-dir "$REPORT_DIR" \
  --skip-compare \
  --vllm-runtime-cache-dir "$PWD/.art/vllm_runtime_cache" \
  --hf-home "$PWD/.art/hf_home"

date -Is
nvidia-smi
