from __future__ import annotations

from typing import Any

from .config import config_from_env
from .retail_env import ReplayRetailEnv
from .rewards import score_trajectory
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
    request_logprobs: bool = True,
) -> Any:
    import art

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": scenario.system_message},
            {"role": "user", "content": scenario.user_message},
        ],
        tools=scenario.tools,
        metadata={
            "scenario_id": scenario.id,
            "split": split,
            "task": scenario.metadata.get("task", "retail"),
        },
    )
    env = ReplayRetailEnv(scenario)
    client = model.openai_client()
    turns = max_turns if max_turns is not None else scenario.max_turns
    completion_budget = max_completion_tokens or config_from_env().rollout_max_completion_tokens

    for _ in range(turns):
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
            break

    result = score_trajectory(trajectory, scenario, invalid_tool_calls=env.invalid_tool_calls)
    trajectory.reward = result.reward
    trajectory.metrics.update(result.metrics)
    trajectory.log(result.explanation)
    return trajectory.finish()
