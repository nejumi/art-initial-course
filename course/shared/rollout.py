from __future__ import annotations

from typing import Any

from .config import config_from_env
from .data import augment_system_message
from .retail_env import ReplayRetailEnv
from .rewards import normalize_reward_profile, score_trajectory
from .schemas import RetailScenario
from .tracing import weave_op


@weave_op("rollout_retail")
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
    import art

    cfg = config_from_env()
    should_terminate_on_invalid = cfg.terminate_on_invalid if terminate_on_invalid is None else terminate_on_invalid
    reward_profile = normalize_reward_profile()
    strict_reference_actions = reward_profile not in {"tau_sparse", "tau_irc"}

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
    )
    client = model.openai_client()
    turns = max_turns if max_turns is not None else scenario.max_turns
    completion_budget = max_completion_tokens or cfg.rollout_max_completion_tokens
    done = False
    turns_used = 0

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
    trajectory.log(result.explanation)
    return trajectory.finish()
