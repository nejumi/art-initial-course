#!/usr/bin/env bash
#SBATCH --job-name=art-retail-parallel
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=12:00:00
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

CONFIG_PATH="${CONFIG_PATH:-course/09_runbooks/config.yaml}"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-course/09_runbooks/base_config.yaml}"
EXTRA_ARGS=()
if [[ -n "${RUNBOOK_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$RUNBOOK_EXTRA_ARGS"
fi

if [[ "$#" -eq 0 ]]; then
  PROFILES=(
    appendix_tau_synthetic_grpo_h100
    appendix_tau_synthetic_gspo_h100
  )
else
  PROFILES=("$@")
fi

mkdir -p .art/sunk_runs/logs

echo "=== ART retail parallel profile runner ==="
echo "config=$CONFIG_PATH"
echo "base_config=$BASE_CONFIG_PATH"
echo "profiles=${PROFILES[*]}"
echo "extra_args=${EXTRA_ARGS[*]:-}"
echo "WANDB_ENTITY=${WANDB_ENTITY:-}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-}"
echo "WEAVE_PROJECT=${WEAVE_PROJECT:-}"
hostname
date -Is
nvidia-smi

declare -a PIDS=()
for index in "${!PROFILES[@]}"; do
  profile="${PROFILES[$index]}"
  gpu_index=$((index % 8))
  log_prefix=".art/sunk_runs/logs/${SLURM_JOB_ID:-manual}-${profile}"
  echo "launch profile=$profile gpu=$gpu_index"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_index"
    export WEAVE_SERVER_CACHE_DIR="$PWD/.art/weave_cache/${SLURM_JOB_ID:-manual}/${profile}"
    mkdir -p "$WEAVE_SERVER_CACHE_DIR"
    python -B course/09_runbooks/run_retail_agentic_sequence.py \
      --config "$CONFIG_PATH" \
      --base-config "$BASE_CONFIG_PATH" \
      --run-profile "$profile" \
      "${EXTRA_ARGS[@]}"
  ) >"${log_prefix}.out" 2>"${log_prefix}.err" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

date -Is
nvidia-smi
exit "$status"
