#!/usr/bin/env bash
#SBATCH --job-name=art-retail-eval
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
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

ART_PATH_ARG="${1:?ART_PATH is required}"
MODEL_NAME="${2:?ART model name is required}"
DATA_DIR="${3:?data dir is required}"
OUTPUT="${4:?output JSONL path is required}"
DATA_ARTIFACT="${5:-}"
REWARD_PROFILE="${6:-tau_irc}"
LIMIT="${7:-48}"
SPLIT="${8:-validation}"
TEMPERATURE="${9:-0.0}"
ROLLOUTS_PER_SCENARIO="${10:-1}"
MAX_COMPLETION_TOKENS="${11:-768}"
MODEL_ARTIFACT_URI="${12:-}"
MODEL_ARTIFACT_STAGE="${13:-validation-candidate}"

mkdir -p .art/sunk_runs/logs
mkdir -p "$(dirname "$OUTPUT")"

export ART_PATH="$ART_PATH_ARG"
export ART_MODEL_NAME="$MODEL_NAME"
export ART_BASE_MODEL="${ART_BASE_MODEL:-LiquidAI/LFM2.5-8B-A1B}"
export ART_MODEL_PROFILE=custom
export ART_VLLM_RUNTIME_CACHE_DIR="${ART_VLLM_RUNTIME_CACHE_DIR:-$PWD/.art/vllm_runtime_cache}"
export HF_HOME="${HF_HOME:-$PWD/.art/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_XET_HIGH_PERFORMANCE=1

if [[ -n "${ART_VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  :
else
  export ART_VLLM_GPU_MEMORY_UTILIZATION=0.70
fi
export ART_VLLM_MAX_MODEL_LEN="${ART_VLLM_MAX_MODEL_LEN:-16384}"
export ART_VLLM_MAX_NUM_BATCHED_TOKENS="${ART_VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export ART_VLLM_MAX_NUM_SEQS="${ART_VLLM_MAX_NUM_SEQS:-8}"

echo "=== retail checkpoint eval ==="
echo "art_path=$ART_PATH"
echo "model_name=$ART_MODEL_NAME"
echo "data_dir=$DATA_DIR"
echo "output=$OUTPUT"
echo "data_artifact=$DATA_ARTIFACT"
echo "model_artifact_uri=${MODEL_ARTIFACT_URI:-<auto-log>}"
echo "model_artifact_stage=$MODEL_ARTIFACT_STAGE"
echo "reward_profile=$REWARD_PROFILE"
echo "split=$SPLIT limit=$LIMIT temperature=$TEMPERATURE rollouts_per_scenario=$ROLLOUTS_PER_SCENARIO"
hostname
date -Is
nvidia-smi

if [[ -z "$MODEL_ARTIFACT_URI" ]]; then
  artifact_log="$(mktemp)"
  python -B course/07_models_registry_weave/log_checkpoint_artifact.py \
    --stage "$MODEL_ARTIFACT_STAGE" \
    --model-name "$MODEL_NAME" \
    --alias validation-candidate \
    --metadata eval_split="\"$SPLIT\"" \
    --metadata eval_limit="$LIMIT" \
    --metadata reward_profile="\"$REWARD_PROFILE\"" \
    2>&1 | tee "$artifact_log"
  MODEL_ARTIFACT_URI="$(
    awk -F'Logged W&B checkpoint artifact: ' '/Logged W&B checkpoint artifact:/ {uri=$2} END {gsub(/\r/, "", uri); print uri}' "$artifact_log"
  )"
  rm -f "$artifact_log"
  if [[ -z "$MODEL_ARTIFACT_URI" ]]; then
    echo "Could not parse logged checkpoint artifact URI." >&2
    exit 1
  fi
fi
echo "resolved_model_artifact_uri=$MODEL_ARTIFACT_URI"

CMD=(
  python -B course/04_grpo_local/evaluate_checkpoint.py
  --data-dir "$DATA_DIR"
  --split "$SPLIT"
  --limit "$LIMIT"
  --temperature "$TEMPERATURE"
  --rollouts-per-scenario "$ROLLOUTS_PER_SCENARIO"
  --max-completion-tokens "$MAX_COMPLETION_TOKENS"
  --reward-profile "$REWARD_PROFILE"
  --output "$OUTPUT"
  --continue-on-invalid
  --include-messages
  --weave
)
if [[ -n "$DATA_ARTIFACT" ]]; then
  CMD+=(--data-artifact "$DATA_ARTIFACT")
fi
if [[ -n "$MODEL_ARTIFACT_URI" ]]; then
  CMD+=(--model-artifact "$MODEL_ARTIFACT_URI")
fi

"${CMD[@]}"

date -Is
nvidia-smi
