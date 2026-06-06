#!/usr/bin/env bash
#SBATCH --job-name=art-retail-runbook
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=08:00:00
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

RUN_SLUG="${1:-lfm25-8b-a1b-bridge-state1}"
BASE_MODEL="${2:-LiquidAI/LFM2.5-8B-A1B}"
DATA_DIR="${3:-$PWD/.art/curriculum_data/retail_bridge_state1}"
SOURCE_DIR="${4:-$PWD/.art/full_data/retail}"
DATA_ARTIFACT_NAME="${5:-retail-course-data-bridge-state1}"
REWARD_PROFILE="${6:-tau_irc}"
SFT_MAX_STEPS="${7:-96}"
RL_STEPS="${8:-32}"
RL_TRAIN_LIMIT="${9:-24}"
RL_ALGOS="${10:-grpo,gspo}"
INCLUDE_TEACHER_SFT="${11:-false}"
TEACHER_SFT_LIMIT="${12:-512}"
INCLUDE_AREAL_SFT="${13:-false}"
AREAL_SFT_LIMIT="${14:-512}"
INCLUDE_SUCCESS_TRACE_SFT="${15:-false}"
SUCCESS_TRACE_SFT_LIMIT="${16:-200}"
RL_GROUPS_PER_STEP="${17:-4}"
RL_ROLLOUTS_PER_SCENARIO="${18:-8}"
RL_LEARNING_RATE="${19:-2e-6}"
RL_SUFFIX="${20:-bridge-s32-lr2e6}"

mkdir -p .art/sunk_runs/logs

echo "=== retail agentic runbook ==="
echo "run_slug=$RUN_SLUG"
echo "base_model=$BASE_MODEL"
echo "data_dir=$DATA_DIR"
echo "source_dir=$SOURCE_DIR"
echo "reward_profile=$REWARD_PROFILE"
echo "sft_max_steps=$SFT_MAX_STEPS"
echo "rl_steps=$RL_STEPS"
echo "rl_train_limit=$RL_TRAIN_LIMIT"
echo "rl_algos=$RL_ALGOS"
echo "rl_groups_per_step=$RL_GROUPS_PER_STEP"
echo "rl_rollouts_per_scenario=$RL_ROLLOUTS_PER_SCENARIO"
echo "rl_learning_rate=$RL_LEARNING_RATE"
echo "rl_suffix=$RL_SUFFIX"
echo "include_teacher_sft=$INCLUDE_TEACHER_SFT"
echo "teacher_sft_limit=$TEACHER_SFT_LIMIT"
echo "include_areal_sft=$INCLUDE_AREAL_SFT"
echo "areal_sft_limit=$AREAL_SFT_LIMIT"
echo "include_success_trace_sft=$INCLUDE_SUCCESS_TRACE_SFT"
echo "success_trace_sft_limit=$SUCCESS_TRACE_SFT_LIMIT"
hostname
date -Is
nvidia-smi

TEACHER_ARGS=()
if [[ "$INCLUDE_TEACHER_SFT" == "1" || "$INCLUDE_TEACHER_SFT" == "true" || "$INCLUDE_TEACHER_SFT" == "yes" ]]; then
  TEACHER_ARGS=(--include-teacher-sft --teacher-sft-limit "$TEACHER_SFT_LIMIT")
fi
AREAL_ARGS=()
if [[ "$INCLUDE_AREAL_SFT" == "1" || "$INCLUDE_AREAL_SFT" == "true" || "$INCLUDE_AREAL_SFT" == "yes" ]]; then
  AREAL_ARGS=(--include-areal-sft --areal-sft-limit "$AREAL_SFT_LIMIT")
fi
SUCCESS_TRACE_ARGS=()
if [[ "$INCLUDE_SUCCESS_TRACE_SFT" == "1" || "$INCLUDE_SUCCESS_TRACE_SFT" == "true" || "$INCLUDE_SUCCESS_TRACE_SFT" == "yes" ]]; then
  SUCCESS_TRACE_ARGS=(--include-success-trace-sft --success-trace-sft-limit "$SUCCESS_TRACE_SFT_LIMIT")
fi

python -B course/09_runbooks/run_retail_agentic_sequence.py \
  --run-slug "$RUN_SLUG" \
  --base-model "$BASE_MODEL" \
  --data-dir "$DATA_DIR" \
  --source-dir "$SOURCE_DIR" \
  --data-artifact-name "$DATA_ARTIFACT_NAME" \
  --reward-profile "$REWARD_PROFILE" \
  --build-bridge \
  --sft-max-steps "$SFT_MAX_STEPS" \
  --rl-steps "$RL_STEPS" \
  --rl-train-limit "$RL_TRAIN_LIMIT" \
  --rl-algos "$RL_ALGOS" \
  --rl-groups-per-step "$RL_GROUPS_PER_STEP" \
  --rl-rollouts-per-scenario "$RL_ROLLOUTS_PER_SCENARIO" \
  --rl-learning-rate "$RL_LEARNING_RATE" \
  --rl-suffix "$RL_SUFFIX" \
  "${TEACHER_ARGS[@]}" \
  "${AREAL_ARGS[@]}" \
  "${SUCCESS_TRACE_ARGS[@]}" \
  --art-path "$PWD/.art/sunk_${RUN_SLUG}_runbook" \
  --report-dir "$DATA_DIR/${RUN_SLUG}_runbook" \
  --vllm-runtime-cache-dir "$PWD/.art/vllm_runtime_cache" \
  --hf-home "$PWD/.art/hf_home"

date -Is
nvidia-smi
