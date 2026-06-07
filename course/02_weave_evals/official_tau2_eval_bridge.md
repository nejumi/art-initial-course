# Official Tau2 Evaluation Bridge

The course's default `ReplayRetailEnv` is a lightweight training environment. It teaches ART rollouts, Weave traces, W&B lineage, SFT, GRPO, GSPO, and RULER without requiring the full tau2 runtime.

For benchmark-grade reporting, use the official tau2 runtime as a separate evaluation path. The current Sierra `tau2-bench` repository also contains newer tau3 extensions, but the text retail/airline/telecom task-evaluation contract remains documented in `docs/evaluation.md`. The project requires Python `>=3.12,<3.14`, so keep it separate from the ART training environment when the ART stack is pinned to Python 3.11.

## Why This Exists

Official tau2 retail scoring uses each task's `reward_basis`. The tau2 docs describe `DB` and `COMMUNICATE` as the default contract for text domains, while the current retail task files in the Sierra repository commonly use `DB` together with `NL_ASSERTION`. Inspect `evaluation_criteria.reward_basis` for the exact task set you run.

- `DB` checks whether the final predicted DB hash matches the gold DB hash.
- `COMMUNICATE` checks whether required strings were communicated.
- `NL_ASSERTION` checks task-specific natural-language conditions with a judge model.
- `evaluation_criteria.actions` is a reference trajectory used to derive the gold DB state. It is not the only valid path unless `ACTION` is explicitly in `reward_basis`.

The course metric `retail_task_success` is the main hands-on success metric for ART rollouts and checkpoint validation. It is designed for fast local training loops, while official tau2 reporting remains a separate benchmark-grade extension.

## Retail First, Telecom Later

Retail is the right first hands-on task because the environment can be taught as a single-control support workflow: the agent reads state, performs one or more state-changing tools, and communicates the result. Telecom is a stronger research target but a heavier workshop topic because tau2 introduces a dual-control setup where both the agent and the user can act with tools in a shared dynamic environment. Do not generalize replay-retail action matching into a telecom reward design; telecom needs an explicit user simulator and coordination/communication analysis.

## Setup

```bash
git clone https://github.com/sierra-research/tau2-bench .art/external/tau2-bench
cd .art/external/tau2-bench
uv sync
cp .env.example .env
```

Keep provider keys in the tau2 `.env` or your shell. Do not copy workshop `.env` files into shared artifacts.

## Run Official Retail Eval

For an external API agent:

```bash
cd .art/external/tau2-bench
uv run tau2 run \
  --domain retail \
  --agent-llm gpt-5.5 \
  --user-llm gpt-5.5 \
  --num-trials 1 \
  --num-tasks 5
```

For an ART-trained model, serve the checkpoint through an OpenAI-compatible endpoint first, then point tau2's LiteLLM agent at that endpoint. Keep this as an advanced/enterprise lab because it adds endpoint lifecycle, checkpoint export/serving, and user-simulator cost.

## Recompute Rewards

Tau2 can recompute official rewards for saved simulation results:

```bash
cd .art/external/tau2-bench
uv run tau2 evaluate-trajs \
  data/simulations/<run>/results.json \
  --output-dir data/simulations/<run>/official_rewards
```

## Import Into Course Tables

From the course repo:

```bash
python course/02_weave_evals/import_tau2_results.py \
  .art/external/tau2-bench/data/simulations/<run>/official_rewards/<updated-results>.json \
  --output data/retail/eval_tau2_official_<stage>.jsonl

python course/02_weave_evals/compare_checkpoints.py \
  data/retail/eval_tau2_official_<stage>.jsonl \
  --stages tau2-official-<stage> \
  --wandb
```

The imported rows include `tau2_official_success`, `tau2_db_success`, `tau2_communicate_success`, `tau2_nl_assertion_success`, and diagnostic action-match columns when those components are present in the source result. `tau2_official_reward` is also kept as a compatibility alias for raw tau2 outputs.
