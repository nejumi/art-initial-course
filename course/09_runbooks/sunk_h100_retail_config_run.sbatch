#!/usr/bin/env bash
#SBATCH --job-name=art-retail-config-run
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
export WEAVE_SERVER_CACHE_DIR="${WEAVE_SERVER_CACHE_DIR:-$PWD/.art/weave_cache/${SLURM_JOB_ID:-manual}}"

CONFIG_PATH="${1:-course/09_runbooks/config.yaml}"
BASE_CONFIG_PATH="${2:-course/09_runbooks/base_config.yaml}"
RUN_PROFILE="${3:-}"

mkdir -p .art/sunk_runs/logs
mkdir -p "$WEAVE_SERVER_CACHE_DIR"

echo "=== retail ART config runbook ==="
echo "config=$CONFIG_PATH"
echo "base_config=$BASE_CONFIG_PATH"
echo "run_profile=${RUN_PROFILE:-from config}"
echo "WANDB_ENTITY=${WANDB_ENTITY:-}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-}"
echo "WEAVE_PROJECT=${WEAVE_PROJECT:-}"
echo "WEAVE_SERVER_CACHE_DIR=${WEAVE_SERVER_CACHE_DIR:-}"
hostname
date -Is
nvidia-smi

RUN_PROFILE_ARGS=()
if [[ -n "$RUN_PROFILE" ]]; then
  RUN_PROFILE_ARGS=(--run-profile "$RUN_PROFILE")
fi

python -B course/09_runbooks/run_retail_agentic_sequence.py \
  --config "$CONFIG_PATH" \
  --base-config "$BASE_CONFIG_PATH" \
  "${RUN_PROFILE_ARGS[@]}"

date -Is
nvidia-smi
