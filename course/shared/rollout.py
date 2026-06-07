from __future__ import annotations

from typing import Any

from .config import config_from_env
from .data import augment_system_message
from .retail_env import ReplayRetailEnv, is_state_changing_tool
from .rewards import TAU_STYLE_REWARD_PROFILES, normalize_reward_profile, score_trajectory
from .schemas import RetailScenario
from .tracing import bind_weave_to_active_wandb_run, weave_op


async def rollout_retail(
    model: Any,
    scenario: RetailScenario,
    *,
    split: str = "train",
    temperature: float = 0.8,
    max_turns: int | None = None,
    max_completion_tokens: int | None = None,
    terminate_on_invalid: bool | None = None,
    request_logprobs: bool = True,
) -> Any:
    bind_weave_to_active_wandb_run()
    return await _rollout_retail_traced(
        model,
        scenario,
        split=split,
        temperature=temperature,
        max_turns=max_turns,
        max_completion_tokens=max_completion_tokens,
        terminate_on_invalid=terminate_on_invalid,
        request_logprobs=request_logprobs,
    )


@weave_op("rollout_retail")
async def _rollout_retail_traced(
    model: Any,
    scenario: RetailScenario,
    *,
    split: str = "train",
    temperature: float = 0.8,
    max_turns: int | None = None,
    max_completion_tokens: int | None = None,
    terminate_on_invalid: bool | None = None,
    request_logprobs: bool = True,
) -> Any:
    import art

    cfg = config_from_env()
    should_terminate_on_invalid = cfg.terminate_on_invalid if terminate_on_invalid is None else terminate_on_invalid
    reward_profile = normalize_reward_profile()
    strict_reference_actions = reward_profile not in TAU_STYLE_REWARD_PROFILES

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": augment_system_message(scenario.system_message)},
            {"role": "user", "content": scenario.user_message},
        ],
        tools=scenario.tools,
        metadata={
            "scenario_id": scenario.id,
            "split": split,
            "task": scenario.metadata.get("task", "retail"),
            "terminate_on_invalid": should_terminate_on_invalid,
            "reward_profile": reward_profile,
            "strict_reference_actions": strict_reference_actions,
        },
    )
    env = ReplayRetailEnv(
        scenario,
        terminate_on_invalid=should_terminate_on_invalid,
        strict_reference_actions=strict_reference_actions,
        allow_reference_state_action_jumps=cfg.allow_reference_state_action_jumps and not strict_reference_actions,
    )
    client = model.openai_client()
    turns = max_turns if max_turns is not None else scenario.max_turns
    completion_budget = max_completion_tokens or cfg.rollout_max_completion_tokens
    done = False
    turns_used = 0
    first_failure_turn = 0
    first_failure_type = ""
    first_state_action_turn = 0
    first_expected_state_action_turn = 0
    read_only_deviations_before_state_action = 0
    accepted_state_action_jump_count = 0
    first_accepted_state_action_jump_turn = 0
    skipped_reference_turns_before_state_action = 0
    last_accepted_reference_state_output_index = -1
    last_accepted_reference_state_turn_index = -1

    for _ in range(turns):
        turns_used += 1
        request: dict[str, Any] = {
            "model": model.get_inference_name(),
            "messages": trajectory.messages(),
            "tools": trajectory.tools,
            "temperature": temperature,
            "max_completion_tokens": completion_budget,
        }
        if request_logprobs:
            request["logprobs"] = True
            request["top_logprobs"] = 0
        completion = await client.chat.completions.create(**request)
        choice = completion.choices[0]
        trajectory.messages_and_choices.append(choice)

        step = env.step(choice.message)
        if first_expected_state_action_turn == 0 and any(is_state_changing_tool(name) for name in step.expected_tool_names):
            first_expected_state_action_turn = turns_used
        if first_state_action_turn == 0 and any(is_state_changing_tool(name) for name in step.actual_tool_names):
            first_state_action_turn = turns_used
        if first_state_action_turn == 0:
            read_only_deviations_before_state_action += step.read_only_reference_mismatches
        if step.accepted_state_action_jump:
            accepted_state_action_jump_count += 1
            if first_accepted_state_action_jump_turn == 0:
                first_accepted_state_action_jump_turn = turns_used
            skipped_reference_turns_before_state_action += step.skipped_reference_turns_before_state_action
            if step.accepted_reference_state_output_index is not None:
                last_accepted_reference_state_output_index = step.accepted_reference_state_output_index
            if step.accepted_reference_state_turn_index is not None:
                last_accepted_reference_state_turn_index = step.accepted_reference_state_turn_index
        if first_failure_turn == 0:
            if step.unknown_tool_calls:
                first_failure_turn = turns_used
                first_failure_type = "unknown_tool"
            elif step.bad_state_actions or step.invalid_state_mutations:
                first_failure_turn = turns_used
                first_failure_type = "bad_state_action"
            elif step.missing_state_actions:
                first_failure_turn = turns_used
                first_failure_type = "missing_state_action"
            elif step.bad_read_only_calls or step.read_only_reference_mismatches:
                first_failure_turn = turns_used
                first_failure_type = "read_only_reference_mismatch"
        trajectory.messages_and_choices.extend(step.tool_messages)
        trajectory.messages_and_choices.extend(step.user_messages)
        if step.done:
            done = True
            break
    truncated_by_max_turn = not done

    result = score_trajectory(
        trajectory,
        scenario,
        invalid_tool_calls=env.invalid_tool_calls,
        invalid_state_mutations=env.invalid_state_mutations,
        read_only_reference_mismatches=env.read_only_reference_mismatches,
        unknown_tool_calls=env.unknown_tool_calls,
        bad_read_only_calls=env.bad_read_only_calls,
        bad_state_actions=env.bad_state_actions,
        missing_state_actions=env.missing_state_actions,
        truncated_by_max_turn=truncated_by_max_turn,
        turn_count=turns_used,
    )
    trajectory.reward = result.reward
    trajectory.metrics.update(result.metrics)
    trajectory.metrics["terminated_on_invalid"] = 1.0 if env.terminated_on_invalid else 0.0
    trajectory.metrics["first_failure_observed"] = 1.0 if first_failure_turn else 0.0
    trajectory.metrics["first_failure_turn"] = float(first_failure_turn)
    trajectory.metrics["first_state_action_turn"] = float(first_state_action_turn)
    trajectory.metrics["first_expected_state_action_turn"] = float(first_expected_state_action_turn)
    trajectory.metrics["read_only_reference_mismatches_before_state_action"] = float(
        read_only_deviations_before_state_action
    )
    trajectory.metrics["accepted_state_action_jump"] = 1.0 if accepted_state_action_jump_count else 0.0
    trajectory.metrics["accepted_state_action_jump_count"] = float(accepted_state_action_jump_count)
    trajectory.metrics["first_accepted_state_action_jump_turn"] = float(first_accepted_state_action_jump_turn)
    trajectory.metrics["skipped_reference_turns_before_state_action"] = float(
        skipped_reference_turns_before_state_action
    )
    trajectory.metrics["last_accepted_reference_state_output_index"] = float(
        last_accepted_reference_state_output_index
    )
    trajectory.metrics["last_accepted_reference_state_turn_index"] = float(last_accepted_reference_state_turn_index)
    trajectory.metadata["first_failure_type"] = first_failure_type or "none"
    trajectory.log(result.explanation)
    return trajectory.finish()
