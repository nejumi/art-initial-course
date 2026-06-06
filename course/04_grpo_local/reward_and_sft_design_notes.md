# Tau-Style SFT and Reward Design Notes

This lab intentionally separates three ideas that are easy to conflate:

1. A reference trajectory is useful SFT data.
2. A reference trajectory is not necessarily the only correct solution path.
3. RL should optimize a verifiable outcome whenever the environment can score one.

## Prior Art to Follow

- [tau-bench / tau2-bench task evaluation](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md) scores customer-service tasks by final outcome, not by exact replay of the listed reference actions. In airline, retail, and telecom, `evaluation_criteria.actions` is primarily replayed to derive a target DB state; the official reward basis is generally DB and communication.
- [Multi-Turn Reinforcement Learning for Tool-Calling Agents with Iterative Reward Calibration](https://arxiv.org/abs/2604.02869) reports that naive dense turn rewards can degrade performance. The useful recipe is to keep sparse outcome rewards as a baseline, avoid positive reward for non-discriminative read-only lookups, penalize bad state-changing actions, and calibrate shaping against rollout data.
- [AReaL-SEA](https://huggingface.co/inclusionAI/AReaL-SEA-235B-A22B) trains multi-turn tool agents with SFT followed by verifiable-reward RL on tau2-style synthetic data. The important SFT lesson is not "copy arbitrary traces"; it is "start from successful, executable, filtered trajectories that match the inference tool-call dialect."
- [From Self-Evolving Synthetic Data to Verifiable-Reward RL](https://arxiv.org/abs/2601.22607) reports a recipe that fine-tunes before GRPO-style agent RL, uses trajectory-level group-relative advantages, filters no-signal groups, and scores final state with verifiable outcome rewards. Their public dataset has separate `sft` and `rl` splits; the course uses the SFT split as an open warm-start source and keeps the RL split as a pointer for the official tau2-runtime extension.
- [Qwen3-4B-tau2-grpo-v1](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1) is a useful public worked example: the model card reports a progressive SFT -> RFT -> GRPO recipe, 21 GRPO optimizer steps, turn-level reward shaping from tau2 reward info, a local training user simulator, and a GPT-4.1-mini eval user simulator. This is not the course's target model, but it validates the overall shape of the lab.
- [MUA-RL](https://arxiv.org/abs/2508.18669) explicitly places an LLM-simulated user inside the RL loop for multi-turn tool-use agents. That supports the course decision to use LLM/user-simulator rollouts for advanced tau2 extensions, while keeping the first hands-on pass on a deterministic lightweight replay environment.

The resulting course recipe is intentionally conservative:

1. Normalize public SFT data into the same tool-call dialect used at inference.
2. Train next-action SFT with last-assistant masking and visible W&B loss/checkpoint points.
3. Evaluate SFT before RL; do not promote a weak SFT parent just because the loss decreased.
4. Run GRPO/GSPO as independent branches from the same SFT parent.
5. Drop no-signal reward groups before optimizer steps.
6. Validate RL against SFT on held-out rollouts, not just on train-step reward.
7. Publish W&B artifact lineage and Weave traces/evals for every stage.

## Course Implementation

The lightweight `ReplayRetailEnv` is intentionally smaller than the full tau2 runtime. It teaches ART trajectories, tool messages, W&B lineage, Weave traces, GRPO, GSPO, and RULER without requiring a full benchmark server.

Because replay data cannot execute arbitrary alternative read-only queries, the course exposes two scoring families:

- `dense`, `strict_success`, `agentic`: teaching profiles that reward matching the demonstration trace. These are useful for explaining failure modes and tool-call syntax, but they should not be presented as official tau-bench scoring.
- `tau_sparse`, `tau_irc`: tau-inspired profiles. They focus on state-changing actions and final communication. Read-only mismatches are diagnostics or small penalties, not core reward drivers.

## SFT Guidance

For the main hands-on path, use `lefft/tau-dev-task-retail-v1` or the derived clean curriculum subset because it is small, open, and already in OpenAI tool-calling format.

Public SFT sources used or referenced by the course:

| Source | Format | Best use in the course |
| --- | --- | --- |
| `lefft/tau-dev-task-retail-v1` | Successful full-dialog retail trajectories with OpenAI-compatible tool calls | Main open-data source, replay-env validation, and per-turn next-action SFT after expansion |
| `inclusionAI/AReaL-tau2-data` | tau2-style next-action SFT/RL rows with `messages`, `answer`, and metadata | Advanced SFT data construction and comparison to recent verifiable-reward agent training recipes |
| `amityco/tau-bench-retail-train-next-action-all-step-score-v0.2` | Retail next-action examples with candidate responses and scores | Optional public teacher warm start; convert high-score rows and train with last-answer-only SFT |
| `KermitCO/qwen3.5-9B-tau2bench-retail-traces` | Retail-only tau2 traces with canonical reward and judge-quality fields | Strong success-trace warm start; default converter keeps reward-1, non-memory, blind-strict traces and expands them to next-action SFT |
| `Jarrodbarnes/tau2-sft-v4-dataset` | Expert tau2 prompts/responses with explicit thinking-tag instructions | Reference only unless reasoning tags and tool dialect are cleaned first |

Before training, normalize demonstrations:

- Assistant turns with `tool_calls` should have empty assistant text.
- Strip or avoid inline `<thinking>` content when training models that may emit reasoning tags instead of tool calls.
- Add a strict tool-use system instruction for local open-weight models that otherwise mix text and tool calls.
- Prefer successful trajectories and keep train/validation/test split provenance in W&B Artifacts.
- Run tokenization preflight and filter overlength trajectories before SFT. If long rows are left in the batch stream, ART drops them internally, which creates zero-loss batches and noisy classroom charts.

ART's local SFT path applies response-only masking to assistant responses in the rendered chat template. For full demonstration trajectories, this is usually what we want: every assistant response in the successful trajectory is supervised. For next-action teacher datasets, be explicit about the intended format. If prior assistant turns should be context only, flatten the prior conversation or add a custom last-answer-only SFT path instead of silently training on every assistant turn.

The course now exposes both formats:

- `make_sft_jsonl.py` writes full-trajectory examples and should use `--sft-mask-mode all-assistant`.
- `make_next_action_sft_jsonl.py` expands each full trajectory into per-turn next-action rows and should use `--sft-mask-mode last-assistant`.
- `make_teacher_next_action_sft_jsonl.py` converts scored public teacher next-action rows from `amityco/...all-step-score-v0.2`; use it when the base bridge SFT is too weak for convincing RL.
- `make_areal_retail_sft_jsonl.py` converts AReaL tau2 retail next-action rows and should also use `--sft-mask-mode last-assistant`.
- `make_success_trace_retail_sft_jsonl.py` converts successful tau2 retail traces into next-action rows. It prepends the local retail system prompt, assigns missing tool-call IDs, attaches the local tool schema, and filters to clean reward-1 traces by default.
- `mix_sft_jsonl.py` mixes bridge, teacher, AReaL, and success-trace SFT sources while preserving source metadata for W&B artifact lineage.
- `make_bridge_curriculum.py` builds a shorter first RL curriculum from successful trajectories with a bounded number of state-changing actions. This is useful for proving that RL can improve a verifiable state/action objective before moving to the broader retail split.

Do not promote an SFT checkpoint to the RL parent just because the loss decreases. First compare it to the base model with Weave evals and a horizontal W&B table. For this lab, a usable SFT parent should improve or at least preserve `outcome_success` and `task_success`, while also improving tool-call diagnostics such as `tool_name_f1`, `tool_argument_match`, or state-changing action correctness.

For a production-quality or advanced course variant, consider swapping in or comparing against:

- `amityco/tau-bench-retail-train-next-action-all-step-score-v0.2`
- `inclusionAI/AReaL-tau2-data`
- `Jarrodbarnes/tau2-sft-v4-dataset`
- `HuggingFaceH4/tau2-bench-data`

The helper `course/03_sft_warmup/make_teacher_next_action_sft_jsonl.py` converts the scored amityco rows into ART-compatible next-action JSONL. The default filter keeps rows with `total_score >= 1.0` and `avg_score >= 1.0`, drops answer turns with unknown tools, and attaches the canonical retail tool schema from the local bridge data. The instructor runbook can enable this with `--include-teacher-sft`, producing `sft_train_next_action_teacher_mix.jsonl` plus a JSON summary that is included in the W&B data artifact.

The helper `course/03_sft_warmup/make_areal_retail_sft_jsonl.py` converts the retail, correct rows from `inclusionAI/AReaL-tau2-data` into ART-compatible JSONL for advanced experiments. It strips `thinking`, removes `None` tool arguments, assigns missing tool-call IDs, and attaches the canonical retail tool schemas from a local TAU retail data directory.

The helper `course/03_sft_warmup/make_success_trace_retail_sft_jsonl.py` converts `KermitCO/qwen3.5-9B-tau2bench-retail-traces`. It keeps canonical reward-1 traces, rejects memory-injected traces by default, requires the public `C_blind_strict` quality flag, and writes both full-trajectory and next-action files. In a current validation snapshot, the converter found 132 clean successful trajectories and expanded them to 512 next-action rows with zero LFM2.5 tokenization drops at 16k sequence length.

The Jarrod Barnes tau2 SFT dataset is useful as a teacher-data reference, but its prompts explicitly request `<thinking>` tags. Do not use it unfiltered with LFM Thinking models in this lab; first remove reasoning-tag instructions and convert tool calls to the same OpenAI-compatible dialect used at inference time.

## RL Guidance

Start every RL experiment with a diagnostic table:

- baseline outcome success
- SFT outcome success
- rollout reward variance within each GRPO group
- invalid state mutation rate
- read-only reference mismatch rate
- final communication success

If reward variance is near zero or SFT outcome success is far below about 30%, RL is likely optimizing noise. In that case, improve SFT data quality, model choice, prompt/tool-call parser compatibility, or task curriculum before increasing RL steps.

The runbook writes `checkpoint_acceptance.md` and `checkpoint_acceptance.json` after the horizontal comparison. The gate is deliberately stricter than "the script did not crash":

- SFT should not regress `outcome_success` or `task_success` versus baseline.
- RL should improve reward by at least 0.05 versus SFT.
- RL should improve at least one agentic metric (`outcome_success`, `task_success`, or `state_action_sequence_match`) by at least 0.05 versus SFT.
- RL should reduce at least one state-action error metric (`bad_state_action` or `missing_state_action`) by at least 0.05 versus SFT.

Rejecting a stage is a useful result. It means the class should inspect Weave traces, improve the warm start or curriculum, and rerun before presenting the table as expected workshop performance.

The GRPO and GSPO scripts implement dynamic filtering before each training step. They discard groups where every rollout receives the same reward, resample up to `--max-sampling-rounds`, and log:

- `data/step_num_groups_sampled_before_filter`
- `data/step_num_groups_dropped_no_reward_signal`
- `data/sample_all_equal_reward_group_rate`
- `data/dropped_all_outcome_failure_group_rate`
- `data/dropped_all_truncated_group_rate`
- `data/dropped_all_invalid_tool_group_rate`
- `data/dropped_all_missing_state_action_group_rate`
- `data/dropped_all_no_state_action_reached_group_rate`

Use `--keep-zero-variance-groups` only for debugging ART's raw behavior. For the workshop's main RL runs, keep filtering enabled so each optimizer step has a real group-relative advantage signal.

For tau-style outcome training, use `--continue-on-invalid` unless the lab is intentionally demonstrating strict replay failure modes. The replay environment cannot execute arbitrary non-reference retail database queries, so strict early termination can collapse exploration before the model reaches the state-changing action and final communication that tau-style scoring actually cares about. With `--continue-on-invalid`, unexpected state-changing actions are still penalized, but read-only deviations are treated as diagnostics or small penalties and the rollout can continue long enough to produce a useful trajectory-level reward.

`--continue-on-invalid` does not make state-changing actions permissive. The replay environment only returns recorded outputs for exact current-step state-changing calls. Out-of-order or unknown state-changing calls are marked invalid and counted as `bad_state_action`; unknown tool names are always invalid. This keeps the lab aligned with tau-style outcome scoring without letting a model fake DB updates by replaying a reference mutation at the wrong time.

Watch `data/step_reward_range_mean`, `data/step_outcome_success_mean`, `data/step_invalid_tool_call_mean`, `data/step_unknown_tool_call_mean`, `data/step_bad_state_action_mean`, `data/step_missing_state_action_mean`, `data/step_truncated_by_max_turn_mean`, and `data/step_num_groups_dropped_no_reward_signal` during RL. If many groups are dropped, the `data/dropped_*_group_rate` metrics show whether they were all-success, all-failure, truncated, invalid-tool, missing-state-action, or never-reached-state-action groups. A good workshop run should show non-zero group reward range and some successful outcomes during sampling before you trust validation improvements.

Eval rows and Weave scorers also record `first_failure_turn`, `first_state_action_turn`, `first_expected_state_action_turn`, and `read_only_reference_mismatches_before_state_action`. These help separate "the model never reached the consequential action" from "the model reached the action but mutated the wrong state" and from "the model spent too many turns on recoverable read-only detours."

Long multi-turn retail traces can exceed the local ART packed sequence length. If the trainer warns that it is dropping tokenized trajectories over `packed_sequence_length`, either raise `ART_MAX_SEQ_LENGTH` on H100-class hardware or reduce `ART_ROLLOUT_MAX_COMPLETION_TOKENS` / `max_turns` for constrained GPUs. For final course results, prefer runs where only a small fraction of trajectories are dropped, because dropped trajectories weaken the W&B learning curves and can bias which successes the optimizer sees.

The local `outcome_success` metric is a training proxy. Newer course outputs also log it as `proxy_outcome_success` to make that explicit. For a full benchmark-grade extension, replace `ReplayRetailEnv` with the official tau2 Gym / domain runtime and use DB + COMMUNICATE pass rate as the primary eval metric. Keep reference-action metrics as diagnostics only.

The optional bridge in `course/02_weave_evals/official_tau2_eval_bridge.md` keeps this separation clean: ART labs can run in the Python 3.11 training environment, while official tau2 scoring runs in a separate Python 3.12 environment and is imported back into W&B comparison tables with `import_tau2_results.py`.
